from collections import Counter
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import math
from tqdm import tqdm
import wandb
from dataset import prepare_simplified, SimplifiedHotpotQADataset
from datasets import load_metric
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
        num_choices = len(features[0]["paras"])
        paras = [feature.pop("paras") for feature in features]
        lengths = [len(i) for i in paras]
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

        # Add back labels
        batch['contexts'] = contexts
        batch['answers'] = raw_answers
        batch['lengths'] = lengths
        batch['ds'] = ds
        return batch

def prepare_model(args):
    model = AutoModel.from_pretrained(args.model_dir)
    model = model.to(device)
    linear = prepare_linear(model.config.hidden_size)
    return [model, linear]

def prepare_dataloader(data, tok, answer_tok, args):
    paras, supps, answs, ds = prepare_simplified(tok, answer_tok, "train", data, max_sent=args.max_paragraph_length, k=args.k_distractor, fixed=args.truncate_paragraph, sentence=args.sentence)
    tparas, tsupps, tansws, tds = prepare_simplified(tok, answer_tok, "validation", data, max_sent=args.max_paragraph_length, k=args.k_distractor, fixed=args.truncate_paragraph, sentence=args.sentence)
    train_dataset = SimplifiedHotpotQADataset(paras, supps, answs, ds)
    eval_dataset = SimplifiedHotpotQADataset(tparas, tsupps, tansws, tds)
    data_collator = DataCollatorForMultipleChoice(tok, padding='longest', max_length=512)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.eval_batch_size)
    return train_dataloader, eval_dataloader

def run_lm(model, batch, bs, train=True):
    model, linear = model
    m = nn.LogSoftmax(dim=-1)
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    pooled_output = outputs[1]
    if train:
        dropout = nn.Dropout(model.config.hidden_dropout_prob)
        pooled_output = dropout(pooled_output)
    logits = linear(pooled_output).view(-1)
    if train:
        return m(logits)
    else:
        return logits

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
        scores = model(input_ids=input_ids, attention_mask=attn_mask, labels=outputs).loss
        outputs = (outputs, scores)
    return outputs

def run_model(batch, layers, answer_model, tokenizer, answer_tokenizer, max_p, reg_coeff, t, beam=2, train=True):
    for key in batch:
        if key == "input_ids" or key == "attention_mask":
            batch[key] = batch[key].to(device)
    bs = len(batch["answers"])
    pouts = run_lm(layers, batch, bs, train=train)
    answer_in, answer_attn, labels = pad_answers(
            answer_tokenizer, batch["contexts"], batch['answers'])
    in_len = len(answer_in)
    answ_out = run_answer_model(answer_model, answer_in, answer_attn, labels, answer_tokenizer, beam=beam, train=train)
    if train:
        loss = answ_out.loss.view(in_len, -1)
        loss = (-loss).sum(dim=-1)
        loss += pouts
        loss = torch.logsumexp(loss, dim=-1)
        loss = -loss.mean()
    else:
        loss = 0.
    return answ_out, pouts, loss

def evaluate(steps, args, layers, answ_model, tok, answ_tok, dataloader, split):
    m = nn.LogSoftmax(dim=-1)
    exact_match = load_metric("exact_match")
    metric = load_metric("accuracy", "multilabel")
    prior_exact_match = load_metric("exact_match")
    prior_metric = load_metric("accuracy", "multilabel")
    prior_ents = []
    pos_ents = []
    if args.save_results and split == "Valid":
        para_results = []
        answ_results = []
    for step, eval_batch in enumerate(dataloader):
        bs = len(eval_batch["answers"])
        gold = answ_tok.batch_decode(eval_batch['answers'], skip_special_tokens=True)
        eval_outs, para_preds, _ = run_model(
                eval_batch, layers, answ_model, tok, answ_tok, max_p=True,
                reg_coeff=args.reg_coeff, t=args.sentence_thrshold, train=False, beam=args.beam)
        lens = eval_batch["lengths"]
        lens.insert(0, 0)
        for i in range(1, len(lens)):
            lens[i] += lens[i-1]
        eval_outs, scores = eval_outs
        scores = scores.view(eval_outs.shape[0], -1)
        para_preds = [para_preds[lens[i]:lens[i+1]] for i in range(len(lens)-1)]
        for i in range(len(para_preds)):
            para_preds[i] = m(para_preds[i])
        eval_outs = [eval_outs[lens[i]:lens[i+1], :] for i in range(len(lens)-1)]
        scores = [-scores[lens[i]:lens[i+1], :] for i in range(len(lens)-1)]
        preds = []
        for i in range(len(scores)):
            mask = scores[i] !=0
            scores[i] = (scores[i]*mask).sum(dim=-1)/mask.sum(dim=-1)
            scores[i] = scores[i] + para_preds[i]
            j = scores[i].argmax(dim=-1).item()
            curr_out = eval_outs[i][j]
            pred = tok.decode(curr_out, skip_special_tokens=True)
            preds.append(pred)
        gold = [normalize_answer(s) for s in gold]
        preds = [normalize_answer(s) for s in preds]
        if args.save_results and split == "Valid":
            answ_results.append((preds, gold))
        exact_match.add_batch(
            predictions=preds,
            references=gold,
        )
        para_tmp = []
        for s, d in zip(scores, eval_batch["ds"]):
            pred = torch.topk(s, 2, dim=-1).indices.cpu().tolist()
            curr = [0] * 10
            curr_idx = 1
            x, y = d[pred[0]], d[pred[curr_idx]]
            while x == y:
                pred = torch.topk(s, curr_idx+2, dim=-1).indices.cpu().tolist()
                curr_idx += 1
                x, y = d[pred[0]], d[pred[curr_idx]]
            curr[x] = 1
            curr[y] = 1
            para_tmp.append(curr)
        labels = [1] * 2 + [0] * 8
        labels = [labels for _ in range(len(para_tmp))]
        metric.add_batch(
            predictions=para_tmp,
            references=labels,
        )
        idxes = [pred.argmax(dim=-1).item() for pred in para_preds]
        para_tmp = []
        for pred, d in zip(para_preds, eval_batch["ds"]):
            curr = [0] * 10
            topk = torch.topk(pred, 2, dim=-1).indices.cpu().tolist()
            curr_idx = 1
            x, y = d[topk[0]], d[topk[curr_idx]]
            while x == y:
                topk = torch.topk(pred, curr_idx+2, dim=-1).indices.cpu().tolist()
                curr_idx += 1
                x, y = d[topk[0]], d[topk[curr_idx]]
            curr[x] = 1
            curr[y] = 1
            para_tmp.append(curr)
        prior_metric.add_batch(
            predictions=para_tmp,
            references=labels,
        )
        if args.save_results and split == "Valid":
            para_results += idxes
        preds = []
        for i in range(len(idxes)):
            curr_out = eval_outs[i][idxes[i]]
            pred = tok.decode(curr_out, skip_special_tokens=True)
            preds.append(pred)
        preds = [normalize_answer(s) for s in preds]
        prior_exact_match.add_batch(
            predictions=preds,
            references=gold,
        )
    pos_eval_metric = exact_match.compute()
    pos_para_acc = metric.compute()
    prior_eval_metric = prior_exact_match.compute()
    prior_para_acc = prior_metric.compute()
    if not args.nolog:
        wandb.log({
            "step": steps,
            f"{split} Prior Para": prior_para_acc,
            f"{split} Prior Acc": prior_eval_metric,
            f"{split} Posterior Para": pos_para_acc,
            f"{split} Posterior Acc": pos_eval_metric})
    if args.save_results and split == "Valid":
        torch.save((para_results, answ_results), f"logging/{args.run_name}|step-{steps}.pt")
    return pos_eval_metric['exact_match']


def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    answer_tokenizer = AutoTokenizer.from_pretrained(args.answer_model_dir)
    data = load_hotpotqa()
    train_dataloader, eval_dataloader = prepare_dataloader(data, tokenizer, answer_tokenizer, args)

    model_name = args.model_dir.split('/')[-1]
    run_name=f'model-{model_name} lr-{args.learning_rate} bs-{args.batch_size*args.gradient_accumulation_steps} k-{args.k_distractor} tp-{args.truncate_paragraph} beam-{args.beam} reg-{args.reg_coeff}'
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
               project='hotpotqa_unsup_simplified_independent_partition',
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
                    answer_tokenizer, reg_coeff=args.reg_coeff, t=args.sentence_thrshold, max_p=args.max_p)
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
