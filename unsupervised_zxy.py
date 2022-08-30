from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union
import math
from tqdm import tqdm
from dataset import prepare_zxy, UnsupHotpotQAZXYDataset
from datasets import load_metric
import numpy as np
from transformers import AutoModel, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import get_scheduler, set_seed
from transformers.file_utils import PaddingStrategy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
import wandb
from utils import get_args, load_hotpotqa, mean_pooling, padding, normalize_answer
from utils import prepare_linear, prepare_optim_and_scheduler, padding

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
        def prepare(name):
            out = [feature.pop(name) for feature in features]
            out = [x for xx in out for x in xx]
            out = [{"input_ids": x} for x in out]
            out = self.tokenizer.pad(
                out,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            return out
        z = prepare('z')
        x = prepare('x')
        zx = prepare('zx')
        y = prepare('y')
        ds_name = "ds"
        ds = [feature.pop(ds_name) for feature in features]
        sent_name = "num_s"
        num_s = [feature.pop(sent_name) for feature in features]
        slabel_name = "sent_labels"
        slabels = [feature.pop(slabel_name) for feature in features]
        answer_name = "answers"
        answers = [feature.pop(answer_name) for feature in features]

        # Add back labels
        batch = dict()
        # limit to 400 so we don't get OOM
        batch['z'] = z['input_ids'][:, :400]
        batch['z_attn'] = z['attention_mask'][:, :400]
        batch['x'] = x['input_ids']
        batch['zx'] = zx['input_ids'][:, :400]
        batch['zx_attn'] = zx['attention_mask'][:, :400]
        batch['y'] = y['input_ids']
        batch['ds'] = ds
        batch['num_s'] = num_s
        batch['answers'] = answers
        batch['s_labels'] = slabels
        return batch

def prepare_dataloader(data, tok, args):
    data_collator = DataCollatorForMultipleChoice(tok, padding='longest', max_length=512)
    everything = prepare_zxy(tok, "train", data, max_sent=args.max_paragraph_length, max_e=args.max_e_len)
    teverything = prepare_zxy(tok, "validation", data, max_sent=args.max_paragraph_length, max_e=args.max_e_len)
    train_dataset = UnsupHotpotQAZXYDataset(everything)
    eval_dataset = UnsupHotpotQAZXYDataset(teverything)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.eval_batch_size)
    return train_dataloader, eval_dataloader

def forward(model, input_ids, attn_mask, labels, train, beam=2):
    if train:
        bs = len(input_ids)
        labels[labels==model.config.pad_token_id] = -100
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels).loss
        outputs = -outputs.view(bs, -1).sum(dim=-1)
    else:
        outputs = model.generate(input_ids, num_beams=beam, min_length=1, max_length=20)
    return outputs

def postprocessing(answ, num_s):
    start, end = 0, 0
    outs = []
    for curr in num_s:
        end += curr
        out = answ[start:end]
        outs.append(out)
        start = end
    return outs

def run_model(batch, zx_model, zxy_model, beam=2, train=True):
    for key in batch:
        if 'x' in key or 'z' in key or 'y' in key:
            batch[key] = batch[key].to(device)
    bs = len(batch['num_s'])
    p_zx = forward(zx_model, batch['z'], batch['z_attn'], batch['x'], train=True, beam=beam)
    p_zxy = forward(zxy_model, batch['zx'], batch['zx_attn'], batch['y'], train=train, beam=beam)
    p_zx = postprocessing(p_zx, batch["num_s"])
    p_zxy = postprocessing(p_zxy, batch["num_s"])
    loss = 0.
    if train:
        for first, second in zip(p_zx, p_zxy):
            l = first + second
            l = torch.logsumexp(l, dim=-1)
            loss = loss - l
        loss = loss / bs
    return p_zx, p_zxy, loss

def update_sp(preds, golds):
    sp_em, sp_f1 = 0, 0
    for cur_sp_pred, gold_sp_pred in zip(preds, golds):
        tp, fp, fn = 0, 0, 0
        for e in cur_sp_pred:
            if e in gold_sp_pred:
                for v in cur_sp_pred[e]:
                    if v in gold_sp_pred[e]:
                        tp += 1
                    else:
                        fp += 1
        for e in gold_sp_pred:
            if e not in cur_sp_pred:
                fn += len(gold_sp_pred[e])
            else:
                for v in gold_sp_pred[e]:
                    if v not in cur_sp_pred[e]:
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

def evaluate(steps, args, zx_model, zxy_model, tok, dataloader, split):
    exact_match = load_metric("exact_match")
    para_results = []
    gold_paras = []
    answ_results = []
    for step, eval_batch in enumerate(dataloader):
        bs = len(eval_batch["num_s"])
        p_zx, p_zxy, _ = run_model(eval_batch, zx_model, zxy_model, beam=args.beam, train=False)
        sent_preds = []
        for one, m in zip(p_zx, eval_batch["ds"]):
            _, indices = torch.sort(one)
            indices = indices.cpu().tolist()
            top2 = indices[:2]
            idx = 1
            while m[top2[0]][0] == m[top2[1]][0]:
                idx += 1
                top2[1] = idx
            i, j = top2
            sent_pred = {m[top2[0]][0]: m[top2[0]][1], m[top2[1]][0]: m[top2[1]][1]}
            sent_preds.append(sent_pred)
        para_results += sent_preds
        gold_paras += eval_batch['s_labels']
        preds = []
        for p1, p2 in zip(p_zx, p_zxy):
            chosen = p1.argmax(dim=-1)
            chosen = p2[chosen]
            chosen = tok.decode(chosen, skip_special_tokens=True)
            preds.append(normalize_answer(chosen))
        gold = [normalize_answer(s) for s in eval_batch["answers"]]
        exact_match.add_batch(
            predictions=preds,
            references=gold,
        )
    eval_metric = exact_match.compute()
    supp_em, supp_f1 = update_sp(para_results, gold_paras)
    if not args.nolog:
        wandb.log({
            "step": steps,
            f"{split} Likelihood": sum(likelihoods)/len(likelihoods),
        })
    if args.save_results and split == "Valid":
        torch.save((para_results, gold_paras, answ_results), f"logging/unsupervised|{args.run_name}|step-{steps}.pt")
    return eval_metric['exact_match']


def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    data = load_hotpotqa()
    train_dataloader, eval_dataloader = prepare_dataloader(data, tokenizer, args)

    model_name = args.model_dir.split('/')[-1]
    run_name=f'model-{model_name} lr-{args.learning_rate} bs-{args.batch_size*args.gradient_accumulation_steps} beam-{args.beam}'
    args.run_name = run_name
    zx_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    zx_model = zx_model.to(device)
    zxy_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    zxy_model = zxy_model.to(device)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.epoch * num_update_steps_per_epoch
    total_batch_size = args.batch_size * args.gradient_accumulation_steps
    optim, lr_scheduler = prepare_optim_and_scheduler([zx_model, zxy_model], args)

    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    if not args.nolog:
        wandb.init(name=run_name,
               project='hotpotqa_unsup_zxy',
               tags=['hotpotqa'])
        wandb.config.lr = args.learning_rate
        wandb.watch(zx_model)
        wandb.watch(zxy_model)

    best_valid = float('-inf')
    zx_model.train()
    zxy_model.train()
    for epoch in range(args.epoch):
        for step, batch in enumerate(train_dataloader):
            if completed_steps % args.eval_steps == 0 and completed_steps > 0 and step % args.gradient_accumulation_steps == 0:
                zx_model.eval()
                zxy_model.eval()
                with torch.no_grad():
                    valid_acc = evaluate(completed_steps, args, zx_model, zxy_model, tokenizer, eval_dataloader, "Valid")
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    if args.save_model:
                        all_layers[0].save_pretrained(f"{args.output_model_dir}/{run_name}")
                zx_model.train()
                zxy_model.train()
            _, _, loss = run_model(batch, zx_model, zxy_model)
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
