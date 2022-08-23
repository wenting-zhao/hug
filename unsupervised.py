from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import math
from tqdm import tqdm
import wandb
from dataset import prepare_pipeline, UnsupHotpotQADataset
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
        paras = [feature.pop("paras") for feature in features]
        lengths = [len(paras[i]) for i in range(len(paras))]
        paras = [p for ps in paras for p in ps]
        paras = [{"input_ids": x} for x in paras]

        batch = self.tokenizer.pad(
            paras,
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
        ds_name2 = "ds2"
        ds2 = [feature.pop(ds_name2) for feature in features]
        sent_name = "num_s"
        num_s = [feature.pop(sent_name) for feature in features]
        slabel_name = "s_labels"
        slabels = [feature.pop(slabel_name) for feature in features]
        smap_name = "s_maps"
        smaps = [feature.pop(smap_name) for feature in features]

        # Add back labels
        batch['contexts'] = contexts
        batch['answers'] = raw_answers
        batch['lengths'] = lengths
        batch['ds'] = ds
        batch['ds2'] = ds2
        batch['num_s'] = num_s
        batch['s_maps'] = smaps
        batch['s_labels'] = slabels
        return batch

def prepare_model(args):
    model = AutoModel.from_pretrained(args.model_dir)
    model = model.to(device)
    linear1 = prepare_linear(model.config.hidden_size)
    linear2 = prepare_linear(model.config.hidden_size)
    return [model, linear1, linear2]

def prepare_dataloader(data, tok, answer_tok, args):
    paras, supps, answs, ds = prepare_pipeline(tok, answer_tok, "train", data, max_sent=args.max_paragraph_length, k=args.k_distractor, fixed=args.truncate_paragraph, sentence=args.sentence)
    tparas, tsupps, tansws, tds = prepare_pipeline(tok, answer_tok, "validation", data, max_sent=args.max_paragraph_length, k=args.k_distractor, fixed=args.truncate_paragraph, sentence=args.sentence)
    train_dataset = UnsupHotpotQADataset(paras, supps, answs, ds)
    eval_dataset = UnsupHotpotQADataset(tparas, tsupps, tansws, tds)
    data_collator = DataCollatorForMultipleChoice(tok, padding='longest', max_length=512)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.eval_batch_size)
    return train_dataloader, eval_dataloader

def run_lm(model, batch, train):
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    return outputs

def run_para_model(linear, outputs, dropout_p, ds, ds2, train):
    pooled_output = outputs[1]
    m = nn.LogSoftmax(dim=-1)
    if train:
        dropout = nn.Dropout(dropout_p)
        pooled_output = dropout(pooled_output)
    logits = linear(pooled_output).view(-1)
    start, end = 0, 0
    all_normalized = []
    for i in range(len(ds)):
        d = ds[i]
        d2 = ds2[i]
        end += len(d)
        normalized = m(logits[start:end])
        t = []
        for i in d2:
            l = d2[i]
            new_t = torch.logsumexp(normalized[l[0]:l[-1]+1], dim=-1)
            t.append(new_t)
        normalized = torch.stack(t)
        all_normalized.append(normalized)
        start = end
    return all_normalized

def run_sent_model(linear, tok, input_ids, lm_outputs, ds, num_s, train):
    m = nn.LogSoftmax(dim=-1)
    indices = (input_ids == tok.unk_token_id).nonzero(as_tuple=False)
    lm_outs = lm_outputs[0][indices[:, 0], indices[:, 1]]
    groups = []
    start, end = 0, 0
    for d, n in zip(ds, num_s):
        end += n
        p_len = max(list(d.values())) + 1
        curr_indices = indices[start:end]
        curr_lm_outs = lm_outs[start:end]
        group_by_p = [[] for _ in range(p_len)]
        l = curr_indices[:, 0] - curr_indices[:, 0][0]
        l = l.cpu().tolist()
        for i, elm in enumerate(l):
            group_by_p[d[elm]].append(curr_lm_outs[i])
        for i in range(len(group_by_p)):
            group_by_p[i] = torch.stack(group_by_p[i])
            combs = torch.combinations(torch.arange(group_by_p[i].shape[0]))
            C = len(combs)
            paired = group_by_p[i][combs,:]
            added = torch.sum(paired, dim=1)
            group_by_p[i] = torch.cat([group_by_p[i], added], dim=0)
            group_by_p[i] = linear(group_by_p[i]).view(-1)
            group_by_p[i] = m(group_by_p[i])
        groups.append(group_by_p)
        start = end
    return groups

def get_selected(paras, sents, kp, ks, mode):
    all_ps_vals, all_top_pouts, all_top_souts = [], [], []
    for p, s in zip(paras, sents):
        if mode == "topk":
            top_pvals, top_pouts = torch.topk(p, k=kp) if len(p) > kp else [p, torch.arange(len(p))]
            top_souts = [torch.topk(sent, k=ks) if len(sent) > ks else [sent, torch.arange(len(sent))] for sent in s]
            top_svals = [i[0] for i in top_souts]
            top_souts = [i[1] for i in top_souts]
        elif mode == "sample":
            top_pouts = torch.from_numpy(np.random.choice(range(len(p)), kp, replace=False)) if len(p) > kp else torch.arange(len(p))
            top_pvals = p[top_pouts]
            top_souts = [torch.from_numpy(np.random.choice(range(len(sent)), ks, replace=False)) if len(sent) > ks else torch.arange(len(sent)) for sent in s]
            top_svals = [s[i][top_souts[i]] for i in range(len(s))]
        elif mode == "topk_sample":
            top_p = torch.topk(p, k=1).indices.item()
            top_s = [torch.topk(sent, k=1).indices.item() for sent in s]
            plist = list(range(len(p)))
            plist.remove(top_p)
            slist = [list(range(len(sent))) for sent in s]
            [sl.remove(curr_top_s) for sl, curr_top_s in zip(slist, top_s)]
            rand_pouts = np.random.choice(plist, kp, replace=False) if len(plist) > kp else np.array(plist, dtype=np.compat.long)
            rand_souts = [np.random.choice(sl, ks, replace=False) if len(sl) > ks else np.array(sl, dtype=np.compat.long) for sl in slist]
            rand_pouts = np.append(rand_pouts, top_p)
            rand_souts = [np.append(sl, ts) for sl, ts in zip(rand_souts, top_s)]
            top_pouts = torch.from_numpy(rand_pouts)
            top_souts = [torch.from_numpy(sl) for sl in rand_souts]
            top_pvals = p[top_pouts]
            top_svals = [s[i][top_souts[i]] for i in range(len(s))]
        else:
            raise NotImplementedError
        ps_vals = [val + top_svals[idx] for idx, val in zip(top_pouts, top_pvals)]
        ps_vals = torch.cat(ps_vals, dim=0)
        all_ps_vals.append(ps_vals)
        all_top_pouts.append(top_pouts)
        all_top_souts.append(top_souts)
    return all_ps_vals, all_top_pouts, all_top_souts

def pad_answers(tokenizer, contexts, raw_answers, topkp, topks):
    lens = []
    out_cs = []
    for cont, cur_tokp, cur_topks in zip(contexts, topkp, topks):
        l = 0
        for elm in cur_tokp:
            curr = [cont[elm][j] for j in cur_topks[elm]]
            out_cs += curr
            l += len(curr)
        lens.append(l)
    out_cs = [{"input_ids": c} for c in out_cs]
    out = tokenizer.pad(
        out_cs,
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

def run_answer_model(model, input_ids, attn_mask, answs, tokenizer, beam, train):
    answs[answs==model.config.pad_token_id] = -100
    if train:
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=answs)
    else:
        outputs = model.generate(input_ids, num_beams=beam, min_length=1, max_length=20)
        scores = model(input_ids=input_ids, attention_mask=attn_mask, labels=outputs)
        outputs = (outputs, scores)
    return outputs

def process_answ(answ, ps_out, in_len):
    return_two = False
    if len(answ) == 2:
        return_two = True
        text, answ = answ
    answ = answ.loss.view(in_len, -1)
    start, end = 0, 0
    outs = []
    text_outs = []
    for curr in ps_out:
        end += len(curr)
        out = answ[start:end]
        out = (out).sum(dim=-1)
        outs.append(out)
        if return_two:
            text_out = text[start:end]
            text_outs.append(text_out)
        start = end
    if return_two:
        return (text_outs, outs)
    else:
        return outs

def run_model(batch, layers, answer_model, tokenizer, answer_tokenizer, max_p, reg_coeff, t, mode, topkp, topks, beam=2, train=True):
    for key in batch:
        if key == "input_ids" or key == "attention_mask":
            batch[key] = batch[key].to(device)
    bs = len(batch["answers"])
    lm_outs = run_lm(layers[0], batch, train=train)
    pouts = run_para_model(layers[1], lm_outs, layers[0].config.hidden_dropout_prob, batch["ds"], batch["ds2"], train=train)
    souts = run_sent_model(layers[2], tokenizer, batch["input_ids"], lm_outs, batch["ds"], batch["num_s"], train=train)
    para_sent, top_pouts, top_souts = get_selected(pouts, souts, topkp, topks, mode=mode)
    answer_in, answer_attn, labels = pad_answers(
            answer_tokenizer, batch["contexts"], batch['answers'], top_pouts, top_souts)
    in_len = len(answer_in)
    answ_out = run_answer_model(answer_model, answer_in, answer_attn, labels, answer_tokenizer, beam=beam, train=train)
    answ_out = process_answ(answ_out, para_sent, in_len)
    loss = 0.
    if train:
        for l, ps in zip(answ_out, para_sent):
            l *= torch.exp(ps)
            l = torch.sum(l, dim=-1)
            loss -= l.mean()
    loss /= bs
    return answ_out, (para_sent, top_pouts, top_souts), loss

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
                        fn += len(gold_sp_pred[e])
        prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
        em = 1.0 if fp + fn == 0 else 0.0
        sp_em += em
        sp_f1 += f1
    sp_em /= len(preds)
    sp_f1 /= len(preds)
    return sp_em, sp_f1

def evaluate(steps, args, layers, answ_model, tok, answ_tok, dataloader, split):
    m = nn.LogSoftmax(dim=-1)
    exact_match = load_metric("exact_match")
    para_results = []
    gold_paras = []
    answ_results = []
    for step, eval_batch in enumerate(dataloader):
        bs = len(eval_batch["answers"])
        gold = answ_tok.batch_decode(eval_batch['answers'], skip_special_tokens=True)
        eval_outs, sent_preds, _ = run_model(
                eval_batch, layers, answ_model, tok, answ_tok, max_p=True,
                reg_coeff=args.reg_coeff, t=args.sentence_thrshold, train=False,
                mode="topk", topkp=args.topkp, topks=args.topks, beam=args.beam)
        eval_outs, scores = eval_outs
        para_sent, top_pouts, top_souts = sent_preds
        ans_prior_preds = []
        for i in range(len(scores)):
            scores[i] = scores[i] + para_sent[i]
            j = para_sent[i].argmax(dim=-1).item()
            curr_out = eval_outs[i][j]
            pred = tok.decode(curr_out, skip_special_tokens=True)
            ans_prior_preds.append(pred)
        gold = [normalize_answer(s) for s in gold]
        ans_prior_preds = [normalize_answer(s) for s in ans_prior_preds]
        if args.save_results and split == "Valid":
            answ_results.append((ans_prior_preds, gold))
        exact_match.add_batch(
            predictions=ans_prior_preds,
            references=gold,
        )
        prior_sent_preds = [dict() for _ in scores]
        for i in range(len(scores)):
            x, y = top_pouts[i][:2].cpu().tolist()
            if x not in eval_batch['s_maps'][i]:
                prior_sent_preds[i][x] = None
            else:
                s0 = eval_batch['s_maps'][i][x][top_souts[i][x][0].item()]
                prior_sent_preds[i][x] = s0
            if y not in eval_batch['s_maps'][i]:
                prior_sent_preds[i][y] = None
            else:
                s1 = eval_batch['s_maps'][i][y][top_souts[i][y][0].item()]
                prior_sent_preds[i][y] = s1
        para_results += prior_sent_preds
        gold_paras += eval_batch['s_labels']
    eval_metric = exact_match.compute()
    supp_em, supp_f1 = update_sp(para_results, gold_paras)
    if not args.nolog:
        wandb.log({
            "step": steps,
            f"{split} Supp F1": supp_f1,
            f"{split} Supp EM": supp_em,
            f"{split} Answ EM": eval_metric,
        })
    if args.save_results and split == "Valid":
        torch.save((para_results, gold_paras, answ_results), f"logging/unsupervised|{args.run_name}|step-{steps}.pt")
    return eval_metric['exact_match']


def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    answer_tokenizer = AutoTokenizer.from_pretrained(args.answer_model_dir)
    data = load_hotpotqa()
    train_dataloader, eval_dataloader = prepare_dataloader(data, tokenizer, answer_tokenizer, args)

    model_name = args.model_dir.split('/')[-1]
    run_name=f'model-{model_name} lr-{args.learning_rate} bs-{args.batch_size*args.gradient_accumulation_steps} k-{args.k_distractor} tp-{args.truncate_paragraph} beam-{args.beam} topkp-{args.topkp} topks-{args.topks} mode-{args.mode}'
    args.run_name = run_name
    all_layers = prepare_model(args)
    answer_model = AutoModelForSeq2SeqLM.from_pretrained(args.answer_model_dir)
    answer_model = answer_model.to(device)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.epoch * num_update_steps_per_epoch
    total_batch_size = args.batch_size * args.gradient_accumulation_steps
    optim, lr_scheduler = prepare_optim_and_scheduler(all_layers+[answer_model], args)

    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    if not args.nolog:
        wandb.init(name=run_name,
               project='hotpotqa_unsup_independent',
               tags=['hotpotqa'])
        wandb.config.lr = args.learning_rate
        wandb.watch(all_layers[0])
        wandb.watch(answer_model)

    best_valid = float('-inf')
    all_layers[0].train()
    answer_model.train()
    for epoch in range(args.epoch):
        for step, batch in enumerate(train_dataloader):
            if completed_steps % args.eval_steps == 0 and completed_steps > 0 and step % args.gradient_accumulation_steps == 0:
                all_layers[0].eval()
                answer_model.eval()
                with torch.no_grad():
                    valid_acc = evaluate(completed_steps, args, all_layers, answer_model,
                                             tokenizer, answer_tokenizer, eval_dataloader, "Valid")
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    if args.save_model:
                        all_layers[0].save_pretrained(f"{args.output_model_dir}/{run_name}")
                all_layers[0].train()
                answer_model.train()
            _, _, loss = run_model(batch, all_layers, answer_model, tokenizer,
                    answer_tokenizer, reg_coeff=args.reg_coeff, t=args.sentence_thrshold, max_p=args.max_p,
                    mode=args.mode, topkp=args.topkp, topks=args.topks)
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
