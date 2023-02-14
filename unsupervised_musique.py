from dataclasses import dataclass
from typing import Optional, Union
import math
from tqdm import tqdm
import wandb
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from dataset import prepare_musique, MuSiQueDataset
from transformers import AutoModel, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import get_scheduler, set_seed
from transformers.file_utils import PaddingStrategy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
from utils import get_args, mean_pooling, padding, normalize_answer
from utils import prepare_linear, prepare_optim_and_scheduler, padding
import numpy as np
import time
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(555)

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        batch_size = len(features)
        sents = [feature.pop("sents") for feature in features]
        lengths = [len(sents[i]) for i in range(len(sents))]
        sents = [s for ss in sents for s in ss]
        sents = [{"input_ids": x} for x in sents]

        batch = self.tokenizer.pad(
            sents,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        answer_name = "answs"
        raw_answers = [feature.pop(answer_name) for feature in features]
        context_name = "supps"
        contexts = [feature.pop(context_name) for feature in features]
        ds_name = "ds"
        ds = [feature.pop(ds_name) for feature in features]
        slabel_name = "sent_labels"
        slabels = [feature.pop(slabel_name) for feature in features]
        label_name = "labels"
        labels = [feature.pop(label_name) for feature in features]
        sent_name = "num_s"
        num_s = [feature.pop(sent_name) for feature in features]

        # Add back labels
        batch['contexts'] = contexts
        batch['answers'] = raw_answers
        batch['s_maps'] = ds
        batch['sent_labels'] = slabels
        batch['labels'] = labels
        batch['lengths'] = lengths
        batch['num_s'] = num_s
        return batch

def prepare_dataloader(tok, answ_tok, args):
    data_collator = DataCollatorForMultipleChoice(tok, padding='longest', max_length=512)
    dataloaders = []
    for split in ['train', 'dev']:
        everything = prepare_musique(tok, answ_tok, split, fixed=args.truncate_paragraph, max_e=args.max_e_len)
        dataset = MuSiQueDataset(everything)
        shuffle = True if split == 'train' else False
        bs = args.batch_size if split == 'train' else args.eval_batch_size
        dataloader = DataLoader(dataset, shuffle=shuffle, collate_fn=data_collator, batch_size=bs)
        dataloaders.append(dataloader)
    return dataloaders

def prepare_model(args):
    model = AutoModel.from_pretrained(args.model_dir)
    model = model.to(device)
    linear = prepare_linear(model.config.hidden_size)
    return model, linear

def run_lm(model, batch, train):
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    return outputs

def run_sent_model(linear, tok, input_ids, lm_outputs, num_s, s_maps):
    m = nn.LogSoftmax(dim=-1)
    indices = (input_ids == tok.unk_token_id).nonzero(as_tuple=False)
    lm_outs = lm_outputs[0][indices[:, 0], indices[:, 1]]
    assert sum(num_s) == lm_outs.size(0)
    groups = []
    start, end = 0, 0
    for n, s_map in zip(num_s, s_maps):
        end += n
        curr_indices = indices[start:end]
        curr_lm_outs = lm_outs[start:end]
        out = []
        for i in range(len(s_map)):
            this_idxes = s_map[i]
            this_idxes = torch.tensor(this_idxes)
            this_combs = curr_lm_outs[this_idxes]
            this_combs = this_combs.mean(dim=0).view(1, -1)
            out.append(this_combs)
        out = torch.cat(out, dim=0)
        out = linear(out).view(-1)
        out = m(out)
        groups.append(out)
        start = end
    return groups

def get_selected(sents, ks, mode):
    all_vals, all_outs = [], []
    for s in sents:
        if mode == "topk":
            top_vals, top_outs = torch.topk(s, k=ks) if len(s) > ks else torch.topk(s, k=len(s))
        elif mode == "sample":
            top_outs = torch.from_numpy(np.random.choice(range(len(s)), ks, replace=False)) if len(s) > ks else torch.arange(len(s))
            top_vals = s[top_outs]
        all_vals.append(top_vals)
        all_outs.append(top_outs)
    return all_vals, all_outs

def pad_answers(tokenizer, contexts, raw_answers, topks, labels, train):
    lens = []
    out_cs = []
    for cont, cur_topks in zip(contexts, topks):
        curr = [cont[j] for j in cur_topks]
        out_cs += curr
        l = len(curr)
        lens.append(l)
        raw_answers = [[a] * l for a, l in zip(raw_answers, lens)]
    raw_answers = [a for ans in raw_answers for a in ans]

    out_cs = [{"input_ids": c} for c in out_cs]
    out = tokenizer.pad(
        out_cs,
        padding='longest',
        return_tensors="pt",
    )
    raw_answers = [{"input_ids": a} for a in raw_answers]
    answers = tokenizer.pad(
        raw_answers,
        padding='longest',
        return_tensors="pt",
        return_attention_mask=False,
    )['input_ids']
    return out['input_ids'].to(device), out['attention_mask'].to(device), answers.to(device)

def run_answer_model(model, input_ids, attn_mask, answs, tokenizer, beam, train):
    answs[answs==model.config.pad_token_id] = -100
    if train:
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=answs)
    else:
        outputs = model.generate(input_ids, num_beams=beam, min_length=1, max_length=20)
    return outputs

def process_answ(answ, ps_out, in_len):
    answ = answ.loss.view(in_len, -1)
    start, end = 0, 0
    outs = []
    for curr in ps_out:
        end += len(curr)
        out = answ[start:end]
        out = (-out).sum(dim=-1)
        outs.append(out)
        start = end
    return outs

def run_model(batch, model, linear, answer_model, tokenizer, answer_tokenizer, ks, beam=1, mode="topk", train=True):
    for key in batch:
        if key == "input_ids" or key == "attention_mask":
            batch[key] = batch[key].to(device)
    bs = len(batch["answers"])
    lm_outs = run_lm(model, batch, train=train)
    souts = run_sent_model(linear, tokenizer, batch["input_ids"], lm_outs, batch['num_s'], batch['s_maps'])
    vals, outs = get_selected(souts, ks, mode)
    answer_in, answer_attn, labels = pad_answers(answer_tokenizer, batch["contexts"], batch['answers'], outs, batch['labels'], train)
    in_len = len(answer_in)
    answ_out = run_answer_model(answer_model, answer_in, answer_attn, labels, answer_tokenizer, beam, train)
    loss = 0.
    if train:
        answ_out = process_answ(answ_out, outs, in_len)
        for l, ps in zip(answ_out, answ_out):
            l = l + ps
            l = torch.logsumexp(l, dim=-1)
            loss -= l.mean()
        loss /= bs
    return answ_out, outs, loss

def update_sp(preds, golds):
    sp_em, sp_f1, sp_prec, sp_recall = 0, 0, 0, 0
    for cur_sp_pred, gold_sp_pred in zip(preds, golds):
        tp, fp, fn = 0, 0, 0
        for e in cur_sp_pred:
            if e in gold_sp_pred:
                tp += 1
            else:
                fp += 1
        for e in gold_sp_pred:
            if e not in cur_sp_pred:
                fn += 1
        prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
        em = 1.0 if fp + fn == 0 else 0.0
        sp_em += em
        sp_f1 += f1
        sp_prec += prec
        sp_recall += recall
    sp_em /= len(preds)
    sp_f1 /= len(preds)
    sp_prec /= len(preds)
    sp_recall /= len(preds)
    return sp_em, sp_f1, sp_prec, sp_recall

def f1_score(predictions, ground_truths):
    f1s, precs, recalls = 0, 0, 0
    for prediction, ground_truth in zip(predictions, ground_truths):
        prediction = normalize_answer(prediction)
        ground_truth = normalize_answer(ground_truth)
        ZERO_METRIC = (0, 0, 0)

        if prediction in ['yes', 'no', 'noanswer'] and prediction != ground_truth:
            f1, precision, recall = ZERO_METRIC
            continue
        if ground_truth in ['yes', 'no', 'noanswer'] and prediction != ground_truth:
            f1, precision, recall = ZERO_METRIC
            continue

        prediction_tokens = prediction.split()
        ground_truth_tokens = ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1, precision, recall = ZERO_METRIC
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        f1s += f1
        precs += precision
        recalls += recall
    f1s /= len(predictions)
    precs /= len(predictions)
    recalls /= len(predictions)
    return f1s, precs, recalls

def evaluate(steps, args, model, linear, answ_model, tok, answ_tok, dataloader, split):
    sent_results = []
    gold_sents = []
    answ_results = []
    gold_answ = []
    for step, eval_batch in enumerate(dataloader):
        eval_outs, sent_outs, _ = run_model(eval_batch, model, linear, answ_model, tok, answ_tok, ks=1, beam=args.beam, train=False)
        sent_preds = []
        for sent_out, s_map in zip(sent_outs, eval_batch['s_maps']):
            sent_pred = sent_out.item()
            sent_pred = s_map[sent_pred]
            sent_preds.append(sent_pred)
        sent_results += sent_preds
        gold_sents += eval_batch['sent_labels']
        predictions = []
        for eval_out in eval_outs:
            pred = tok.decode(eval_out, skip_special_tokens=True)
            predictions.append(pred)
        answ_results += predictions
        gold_answ += eval_batch["labels"]
    supp_em, supp_f1, prec, recall = update_sp(sent_results, gold_sents)
    f1, _, _ = f1_score(answ_results, gold_answ)
    if not args.nolog:
        wandb.log({
            "step": steps,
            f"{split} Supp F1": supp_f1,
            f"{split} Supp EM": supp_em,
            f"{split} Supp Prec": prec,
            f"{split} Supp Rec": recall,
            f"{split} Answ F1a": f1,
        })
    if args.save_results and split == "Valid":
        torch.save((sent_results, gold_sents, answ_results, gold_answ), f"logging/unsupervised|{args.run_name}|step-{steps}.pt")
    return f1

def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    answer_tokenizer = AutoTokenizer.from_pretrained(args.answer_model_dir)
    train_dataloader, eval_dataloader = prepare_dataloader(tokenizer, answer_tokenizer, args)

    model_name = args.model_dir.split('/')[-1]
    run_name=f'model-{model_name} lr-{args.learning_rate} bs-{args.batch_size*args.gradient_accumulation_steps} mode-{args.mode} topk-{args.topks}'
    args.run_name = run_name
    model, linear = prepare_model(args)
    answer_model = AutoModelForSeq2SeqLM.from_pretrained(args.answer_model_dir)
    answer_model = answer_model.to(device)
    if args.gradient_checkpoint:
        model.gradient_checkpointing_enable()
        answer_model.gradient_checkpointing_enable()

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.epoch * num_update_steps_per_epoch
    total_batch_size = args.batch_size * args.gradient_accumulation_steps
    optim, lr_scheduler = prepare_optim_and_scheduler([model, linear, answer_model], args)

    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    if not args.nolog:
        wandb.init(name=run_name,
               project='musique_unp',
               tags=['musique'])
        wandb.config.lr = args.learning_rate
        wandb.watch(model)
        wandb.watch(answer_model)

    best_valid = float('-inf')
    model.train()
    answer_model.train()
    train_time = 0
    test_time = []
    for epoch in range(args.epoch):
        for step, batch in enumerate(train_dataloader):
            if completed_steps % args.eval_steps == 0 and completed_steps > 0 and step % args.gradient_accumulation_steps == 0:
                model.eval()
                answer_model.eval()
                with torch.no_grad():
                    st_time = time.time()
                    valid_acc = evaluate(completed_steps, args, model, linear, answer_model,
                                             tokenizer, answer_tokenizer, eval_dataloader, "Valid")
                    ed_time = time.time()
                    test_time.append(ed_time-st_time)
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    if args.save_model:
                        model.save_pretrained(f"{args.output_model_dir}/{run_name}")
                model.train()
                answer_model.train()
            st_time = time.time()
            _, _, loss = run_model(batch, model, linear, answer_model, tokenizer, answer_tokenizer, args.topks, args.beam, mode=args.mode)
            loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optim.step()
                lr_scheduler.step()
                optim.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                if not args.nolog:
                    wandb.log({
                        "step": completed_steps,
                        "Train Loss": loss.item()})
            ed_time = time.time()
            train_time += ed_time-st_time
    print(train_time)
    print(test_time, sum(test_time)/len(test_time))

if __name__ == '__main__':
    main()
