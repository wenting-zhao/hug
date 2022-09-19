from dataclasses import dataclass
from typing import Optional, Union
import math
from datasets import load_metric
from tqdm import tqdm
import wandb
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from dataset import prepare_fever, FeverDataset
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import get_scheduler, set_seed
from transformers.file_utils import PaddingStrategy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
from utils import get_args, load_hotpotqa, mean_pooling, padding, normalize_answer
from utils import prepare_linear, prepare_optim_and_scheduler, padding, collect_fever_docs

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

        context_name = "supps"
        contexts = [feature.pop(context_name) for feature in features]
        ds_name = "ds"
        ds = [feature.pop(ds_name) for feature in features]
        label_name = "labels"
        labels = [feature.pop(label_name) for feature in features]
        slabel_name = "sent_labels"
        slabels = [feature.pop(slabel_name) for feature in features]
        sent_name = "num_s"
        num_s = [feature.pop(sent_name) for feature in features]

        # Add back labels
        batch['contexts'] = contexts
        batch['s_maps'] = ds
        batch['sent_labels'] = slabels
        batch['labels'] = labels
        batch['lengths'] = lengths
        batch['num_s'] = num_s
        return batch

def prepare_dataloader(tok, answ_tok, args):
    docs = collect_fever_docs()
    data_collator = DataCollatorForMultipleChoice(tok, padding='longest', max_length=512)
    dataloaders = []
    for split in ['train', 'val', 'test']:
        everything = prepare_fever(tok, answ_tok, split, docs, fixed=args.truncate_paragraph, max_e=args.max_e_len)
        dataset = FeverDataset(everything)
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
        out = [curr_lm_outs]
        for i in range(1, len(s_map)):
            this_idxes = s_map[i]
            this_idxes = torch.tensor(this_idxes)
            this_combs = curr_lm_outs[this_idxes]
            this_combs = this_combs.mean(dim=1)
            out.append(this_combs)
        out = torch.cat(out, dim=0)
        out = linear(out).view(-1)
        out = m(out)
        groups.append(out)
        start = end
    return groups

def pad_answers(tokenizer, contexts, raw_answers):
    lens = [len(c) for c in contexts]
    contexts = [c for cs in contexts for c in cs]
    contexts = [{"input_ids": c} for c in contexts]
    out = tokenizer.pad(
        contexts,
        padding='longest',
        return_tensors="pt",
    )
    raw_answers = [[a] * l for a, l in zip(raw_answers, lens)]
    raw_answers = [a for ans in raw_answers for a in ans]
    answers = torch.tensor(raw_answers, device=device)
    return out['input_ids'].to(device), out['attention_mask'].to(device), answers

def run_answer_model(model, input_ids, attn_mask, answs, train):
    m = nn.LogSoftmax(dim=-1)
    logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
    normalized = m(logits)
    if train:
        normalized = normalized[torch.arange(len(normalized), device=device), answs]
    return normalized

def process_answ(answ, ps_out):
    start, end = 0, 0
    outs = []
    for curr in ps_out:
        end += len(curr)
        out = answ[start:end]
        outs.append(out)
        start = end
    return outs

def run_model(batch, model, linear, answer_model, tokenizer, answer_tokenizer, train=True):
    for key in batch:
        if key == "input_ids" or key == "attention_mask":
            batch[key] = batch[key].to(device)
    bs = len(batch["labels"])
    lm_outs = run_lm(model, batch, train=train)
    souts = run_sent_model(linear, tokenizer, batch["input_ids"], lm_outs, batch['num_s'], batch['s_maps'])
    answer_in, answer_attn, labels = pad_answers(answer_tokenizer, batch["contexts"], batch['labels'])
    in_len = len(answer_in)
    answ_out = run_answer_model(answer_model, answer_in, answer_attn, labels, train=train)
    answ_out = process_answ(answ_out, souts)
    loss = 0.
    if train:
        for l, ps in zip(answ_out, souts):
            l = l + ps
            l = torch.logsumexp(l, dim=-1)
            loss -= l.mean()
        loss /= bs
    return answ_out, souts, loss

def update_sp(preds, golds):
    sp_em, sp_f1 = 0, 0
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
    sp_em /= len(preds)
    sp_f1 /= len(preds)
    return sp_em, sp_f1

def evaluate(steps, args, model, linear, answ_model, tok, answ_tok, dataloader, split):
    metric = load_metric("accuracy")
    sent_results = []
    gold_sents = []
    if args.save_results and split == "Valid":
        answ_results = []
        gold_answ = []
    for step, eval_batch in enumerate(dataloader):
        eval_outs, sent_outs, _ = run_model(eval_batch, model, linear, answ_model, tok, answ_tok, train=False)
        sent_preds = []
        for sent_out, s_map in zip(sent_outs, eval_batch['s_maps']):
            sent_pred = sent_out.argmax(dim=-1).item()
            flattened = [sm for sms in s_map for sm in sms]
            sent_pred = flattened[sent_pred]
            sent_preds.append(sent_pred)
        sent_results += sent_preds
        gold_sents += eval_batch['sent_labels']
        predictions = []
        for eval_out, sent_pred in zip(eval_outs, sent_preds):
            pred = eval_out[sent_pred].argmax(dim=-1)
            predictions.append(pred.item())
        metric.add_batch(
            predictions=predictions,
            references=eval_batch["labels"],
        )
        if args.save_results and split == "Valid":
            answ_results += predictions
            gold_answ += eval_batch["labels"]
    eval_metric = metric.compute()
    supp_em, supp_f1 = update_sp(sent_results, gold_sents)
    if not args.nolog:
        wandb.log({
            "step": steps,
            f"{split} Supp F1": supp_f1,
            f"{split} Supp EM": supp_em,
            f"{split} Answ EM": eval_metric,
        })
    if args.save_results and split == "Valid":
        torch.save((sent_results, gold_sents, answ_results, gold_answ), f"logging/unsupervised|{args.run_name}|step-{steps}.pt")
    print(supp_em, supp_f1, eval_metric['accuracy'])
    return eval_metric['accuracy']

def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    answer_tokenizer = AutoTokenizer.from_pretrained(args.answer_model_dir)
    train_dataloader, eval_dataloader, test_dataloader = prepare_dataloader(tokenizer, answer_tokenizer, args)

    model_name = args.model_dir.split('/')[-1]
    run_name=f'nobart model-{model_name} lr-{args.learning_rate} bs-{args.batch_size*args.gradient_accumulation_steps} max_e_len-{args.max_e_len}'
    args.run_name = run_name
    model, linear = prepare_model(args)
    answer_model = AutoModelForSequenceClassification.from_pretrained(args.answer_model_dir, num_labels=2)
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
               project='fever_unp',
               tags=['fever'])
        wandb.config.lr = args.learning_rate
        wandb.watch(model)
        wandb.watch(answer_model)

    best_valid = float('-inf')
    model.train()
    answer_model.train()
    for epoch in range(args.epoch):
        for step, batch in enumerate(train_dataloader):
            if completed_steps % args.eval_steps == 0 and completed_steps > 0 and step % args.gradient_accumulation_steps == 0:
                model.eval()
                answer_model.eval()
                with torch.no_grad():
                    valid_acc = evaluate(completed_steps, args, model, linear, answer_model,
                                             tokenizer, answer_tokenizer, eval_dataloader, "Valid")
                    test_acc = evaluate(completed_steps, args, model, linear, answer_model,
                                             tokenizer, answer_tokenizer, test_dataloader, "Test")
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    if args.save_model:
                        model.save_pretrained(f"{args.output_model_dir}/{run_name}")
                model.train()
                answer_model.train()
            _, _, loss = run_model(batch, model, linear, answer_model, tokenizer, answer_tokenizer)
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

if __name__ == '__main__':
    main()
