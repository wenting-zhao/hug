from dataclasses import dataclass
from typing import Optional, Union
import math
from tqdm import tqdm
import wandb
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from dataset import prepare_multirc, MultiRCDataset
from dataset import prepare_fever, FeverDataset
from dataset import prepare_hotpotqa, HotpotQADataset
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
from utils import prepare_linear, prepare_optim_and_scheduler, padding, collect_multirc_docs, collect_fever_docs
import numpy as np
from sklearn.metrics import f1_score
import time

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
    if args.dataset == "multirc":
        docs = collect_multirc_docs()
    elif args.dataset == "fever":
        docs = collect_fever_docs()
    else:
        docs = None
    data_collator = DataCollatorForMultipleChoice(tok, padding='longest', max_length=512)
    dataloaders = []
    for split in ['train', 'val', 'test']:
        if args.dataset == "hotpotqa" and split == "test": continue
        if args.dataset == "hotpotqa":
            prepare = prepare_hotpotqa
            Dataset = HotpotQADataset
        elif args.dataset == "multirc":
            prepare = prepare_multirc
            Dataset = MultiRCDataset
        elif args.dataset == "fever":
            prepare = prepare_fever
            Dataset = FeverDataset
        else:
            raise NotImplementedError
        everything = prepare(tok, answ_tok, split, docs, fixed=args.truncate_paragraph, max_e=args.max_e_len)
        dataset = Dataset(everything)
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

def pad_answers(tokenizer, contexts, raw_answers, topks, labels, train, dataset):
    if dataset == "multirc":
        lens = []
        out_cs = []
        for cont, cur_topks in zip(contexts, topks):
            if not train:
                curr = [cont[j] for j in cur_topks[:1]]
            else:
                curr = [cont[j] for j in cur_topks]
            out_cs += curr
            l = len(curr)
            lens.append(l)
        if not train:
            assert len(labels) == len(out_cs)
            out_cs = [[cont] * len(l) * 2 for l, cont in zip(labels, out_cs)]
            out_cs = [cc for c in out_cs for cc in c]
        else:
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
    elif dataset == "fever":
        lens = [len(k) for k in topks]
        out_cs = []
        for cont, cur_topks in zip(contexts, topks):
            curr = [cont[j] for j in cur_topks]
            out_cs += curr
        contexts = [{"input_ids": c} for c in out_cs]
        out = tokenizer.pad(
            contexts,
            padding='longest',
            return_tensors="pt",
        )
        raw_answers = [[a] * l for a, l in zip(raw_answers, lens)]
        raw_answers = [a for ans in raw_answers for a in ans]
        raw_answers = [{"input_ids": a} for a in raw_answers]
        answers = tokenizer.pad(
            raw_answers,
            padding='longest',
            return_tensors="pt",
            return_attention_mask=False,
        )['input_ids']
    return out['input_ids'].to(device), out['attention_mask'].to(device), answers.to(device)

def run_answer_model(model, input_ids, attn_mask, answs, tokenizer, train, dataset):
    if dataset == "multirc":
        answs[answs==model.config.pad_token_id] = -100
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=answs).loss
        outputs = outputs.view(input_ids.size(0), -1).sum(dim=-1)
    elif dataset == "fever":
        answs[answs==model.config.pad_token_id] = -100
        if train:
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=answs).loss
            outputs = outputs.view(input_ids.size(0), -1).sum(dim=-1)
        else:
            answs = tokenizer("supports", return_tensors="pt", return_attention_mask=False)['input_ids'].to(device)
            answs = answs.repeat(input_ids.size(0), 1)
            output1 = model(input_ids=input_ids, attention_mask=attn_mask, labels=answs).loss
            output1 = output1.view(input_ids.size(0), -1).sum(dim=-1).view(-1, 1)
            answs = tokenizer("refutes", return_tensors="pt", return_attention_mask=False)['input_ids'].to(device)
            answs = answs.repeat(input_ids.size(0), 1)
            output2 = model(input_ids=input_ids, attention_mask=attn_mask, labels=answs).loss
            output2 = output2.view(input_ids.size(0), -1).sum(dim=-1).view(-1, 1)
            outputs = torch.cat([output1, output2], dim=1)
    return -outputs

def process_answ(answ, ps_out, train, dataset):
    start, end = 0, 0
    outs = []
    for curr in ps_out:
        if train or dataset != "multirc":
            end += len(curr)
        else:
            end += len(curr) * 2
        out = answ[start:end]
        outs.append(out)
        start = end
    return outs

def run_model(batch, model, linear, answer_model, tokenizer, answer_tokenizer, ks, dataset, mode="topk", train=True):
    for key in batch:
        if key == "input_ids" or key == "attention_mask":
            batch[key] = batch[key].to(device)
    bs = len(batch["answers"])
    lm_outs = run_lm(model, batch, train=train)
    souts = run_sent_model(linear, tokenizer, batch["input_ids"], lm_outs, batch['num_s'], batch['s_maps'])
    vals, outs = get_selected(souts, ks, mode)
    answer_in, answer_attn, labels = pad_answers(answer_tokenizer, batch["contexts"], batch['answers'], outs, batch['labels'], train, dataset)
    in_len = len(answer_in)
    answ_out = run_answer_model(answer_model, answer_in, answer_attn, labels, answer_tokenizer, train, dataset)
    passed = outs if (train or dataset != "multirc") else batch['labels']
    answ_out = process_answ(answ_out, passed, train, dataset)
    loss = 0.
    if train:
        for l, ps in zip(answ_out, vals):
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

def update_answer(preds, golds):
    f1s, ems = [], []
    for question_preds, question_labels in zip(preds, golds):
        f1 = f1_score(y_true=question_labels, y_pred=question_preds, average="macro")
        f1s.append(f1)
        em = int(sum(p == l for p, l in zip(question_preds, question_labels)) == len(question_preds))
        ems.append(em)
    f1_m = float(sum(f1s) / len(f1s))
    em = sum(ems) / len(ems)
    labels = [ll for l in golds for ll in l]
    predictions = [pp for p in preds for pp in p]
    f1_a = float(f1_score(y_true=labels, y_pred=predictions))
    acc = sum(p == l for p, l in zip(predictions, labels)) / len(predictions)
    return em, f1_m, f1_a, acc

def evaluate(steps, args, model, linear, answ_model, tok, answ_tok, dataloader, split):
    sent_results = []
    gold_sents = []
    answ_results = []
    gold_answ = []
    for step, eval_batch in enumerate(dataloader):
        eval_outs, sent_outs, _ = run_model(eval_batch, model, linear, answ_model, tok, answ_tok, dataset=args.dataset, ks=1, train=False)
        sent_preds = []
        for sent_out, s_map in zip(sent_outs, eval_batch['s_maps']):
            sent_pred = sent_out.cpu().tolist()
            sent_preds.append(sent_pred)
        sent_results += sent_preds
        gold_sents += eval_batch['sent_labels']
        predictions = []
        for eval_out in eval_outs:
            eval_out = eval_out.view(-1, 2)
            pred = eval_out.argmax(dim=-1)
            predictions.append(pred.cpu().tolist())
        answ_results += predictions
        if args.dataset == "fever":
            gold_answ += [[elm] for elm in eval_batch["labels"]]
        elif args.dataset == "multirc":
            gold_answ += eval_batch["labels"]
    supp_em, supp_f1, prec, recall = update_sp(sent_results, gold_sents)
    em, f1_m, f1_a, acc = update_answer(answ_results, gold_answ)
    if not args.nolog:
        wandb.log({
            "step": steps,
            f"{split} Supp F1": supp_f1,
            f"{split} Supp EM": supp_em,
            f"{split} Supp Prec": prec,
            f"{split} Supp Rec": recall,
            f"{split} Answ EM": em,
            f"{split} Answ F1a": f1_a,
            f"{split} Answ F1m": f1_m,
            f"{split} Answ Acc": acc,
        })
    if args.save_results and split == "Valid":
        torch.save((sent_results, gold_sents, answ_results, gold_answ), f"logging/unsupervised|{args.run_name}|step-{steps}.pt")
    return f1_a

def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    answer_tokenizer = AutoTokenizer.from_pretrained(args.answer_model_dir)
    train_dataloader, eval_dataloader, test_dataloader = prepare_dataloader(tokenizer, answer_tokenizer, args)

    model_name = args.model_dir.split('/')[-1]
    run_name=f'rag model-{model_name} lr-{args.learning_rate} bs-{args.batch_size*args.gradient_accumulation_steps} mode-{args.mode} topk-{args.topks}'
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

    if args.eval_steps == 5000:
        args.eval_steps = num_update_steps_per_epoch
    print(f"eval steps: {args.eval_steps}")

    if not args.nolog:
        wandb.init(name=run_name,
               project='rag',
               tags=[args.dataset])
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
                    valid_acc = evaluate(completed_steps, args, model, linear, answer_model,
                                             tokenizer, answer_tokenizer, eval_dataloader, "Valid")
                    st_time = time.time()
                    test_acc = evaluate(completed_steps, args, model, linear, answer_model,
                                             tokenizer, answer_tokenizer, test_dataloader, "Test")
                    ed_time = time.time()
                    test_time.append(ed_time-st_time)
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    if args.save_model:
                        model.save_pretrained(f"{args.output_model_dir}/{run_name}")
                model.train()
                answer_model.train()
            st_time = time.time()
            _, _, loss = run_model(batch, model, linear, answer_model, tokenizer, answer_tokenizer, args.topks, mode=args.mode, dataset=args.dataset)
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
