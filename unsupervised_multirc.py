from dataclasses import dataclass
from typing import Optional, Union
import math
from tqdm import tqdm
import wandb
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from dataset import prepare_multirc, MultiRCDataset
from transformers import AutoModel
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import get_scheduler, set_seed
from transformers.file_utils import PaddingStrategy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
from utils import get_args, mean_pooling, padding, normalize_answer
from utils import prepare_linear, prepare_optim_and_scheduler, padding, collect_multirc_docs
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
        count_name = "counts"
        counts = [feature.pop(count_name) for feature in features]

        # Add back labels
        batch['contexts'] = contexts
        batch['s_maps'] = ds
        batch['sent_labels'] = slabels
        batch['labels'] = [torch.tensor(l, dtype=torch.long, device=device) for l in labels]
        batch['lengths'] = lengths
        batch['num_s'] = num_s
        batch['counts'] = counts
        return batch

def prepare_dataloader(tok, answ_tok, args):
    docs = collect_multirc_docs()
    data_collator = DataCollatorForMultipleChoice(tok, padding='longest', max_length=512)
    dataloaders = []
    for split in ['train', 'val', 'test']:
        everything = prepare_multirc(tok, answ_tok, split, docs, fixed=args.truncate_paragraph, max_e=args.max_e_len)
        dataset = MultiRCDataset(everything)
        shuffle = True if split == 'train' else False
        bs = args.batch_size if split == 'train' else args.eval_batch_size
        dataloader = DataLoader(dataset, shuffle=shuffle, collate_fn=data_collator, batch_size=bs)
        dataloaders.append(dataloader)
    return dataloaders

def prepare_model(args):
    model = AutoModel.from_pretrained(args.model_dir)
    model = model.to(device)
    linear = prepare_linear(model.config.hidden_size)
    linear1 = prepare_linear(model.config.hidden_size)
    linear2 = prepare_linear(model.config.hidden_size)
    return model, [linear, linear1, linear2]

def pad_answers(tokenizer, contexts):
    out_cs = [{"input_ids": c[0]} for c in contexts]
    out = tokenizer.pad(
        out_cs,
        padding='longest',
        return_tensors="pt",
    )
    return out['input_ids'].to(device), out['attention_mask'].to(device)

def run_answer_model(model, linear0, linear1, input_ids, attn_mask, answ_tok, train):
    m = nn.LogSoftmax(dim=-1)
    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
    indices = (input_ids == answ_tok.unk_token_id).nonzero(as_tuple=False)
    outputs = outputs[0][indices[:, 0], indices[:, 1]]
    if train:
        dropout = nn.Dropout(model.config.hidden_dropout_prob)
        outputs = dropout(outputs)
    outs0 = linear0(outputs).view(-1, 1)
    outs1 = linear1(outputs).view(-1, 1)
    outs = torch.cat([outs0, outs1], dim=1)
    outs = m(outs)
    return outs

def process_answ(answ, labels, train):
    start, end = 0, 0
    outs = []
    for l in labels:
        end += len(l)
        out = answ[start:end].view(-1, len(l), 2)
        if train:
            out = out[:, torch.arange(len(l), device=device), l]
            out = torch.sum(out, dim=-1)
        outs.append(out)
        start = end
    return outs

def run_model(batch, model, linear, answer_model, tokenizer, answer_tokenizer, ks, mode="topk", train=True):
    sent_linear, ans_linear0, ans_linear1 = linear
    for key in batch:
        if key == "input_ids" or key == "attention_mask":
            batch[key] = batch[key].to(device)
    answer_in, answer_attn = pad_answers(answer_tokenizer, batch["contexts"])
    answ_out = run_answer_model(answer_model, ans_linear0, ans_linear1, answer_in, answer_attn, answer_tokenizer, train=train)
    answ_out = process_answ(answ_out, batch["labels"], train)
    loss = 0.
    if train:
        for l in answ_out:
            l = torch.logsumexp(l, dim=-1)
            loss -= l.mean()
        bs = len(answ_out)
        loss /= bs
    return answ_out, loss

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
    return em, f1_m, f1_a

def evaluate(steps, args, model, linear, answ_model, tok, answ_tok, dataloader, split):
    sent_results = []
    gold_sents = []
    counts = []
    answ_results = []
    gold_answ = []
    for step, eval_batch in enumerate(dataloader):
        eval_outs, _ = run_model(eval_batch, model, linear, answ_model, tok, answ_tok, ks=1, train=False)
        predictions = []
        for eval_out in eval_outs:
            pred = eval_out[0].argmax(dim=-1)
            predictions.append(pred.cpu().tolist())
        answ_results += predictions
        gold_answ += [labels.cpu().tolist() for labels in eval_batch["labels"]]
    em, f1_m, f1_a = update_answer(answ_results, gold_answ)
    if not args.nolog:
        wandb.log({
            "step": steps,
            f"{split} Answ EM": em,
            f"{split} Answ F1a": f1_a,
            f"{split} Answ F1m": f1_m,
        })
    if args.save_results and split == "Valid":
        torch.save((sent_results, gold_sents, answ_results, gold_answ), f"logging/unsupervised|{args.run_name}|step-{steps}.pt")
    return em

def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    answer_tokenizer = AutoTokenizer.from_pretrained(args.answer_model_dir)
    train_dataloader, eval_dataloader, test_dataloader = prepare_dataloader(tokenizer, answer_tokenizer, args)

    model_name = args.model_dir.split('/')[-1]
    run_name=f'nobart model-{model_name} lr-{args.learning_rate} bs-{args.batch_size*args.gradient_accumulation_steps} mode-{args.mode} topk-{args.topks}'
    args.run_name = run_name
    model, linear = prepare_model(args)
    answer_model = AutoModel.from_pretrained(args.answer_model_dir)
    answer_model = answer_model.to(device)
    if args.gradient_checkpoint:
        model.gradient_checkpointing_enable()
        answer_model.gradient_checkpointing_enable()

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.epoch * num_update_steps_per_epoch
    total_batch_size = args.batch_size * args.gradient_accumulation_steps
    optim, lr_scheduler = prepare_optim_and_scheduler(linear+[model, answer_model], args)

    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    if args.eval_steps == 5000:
        args.eval_steps = num_update_steps_per_epoch
    print(f"eval steps: {args.eval_steps}")

    if not args.nolog:
        wandb.init(name=run_name,
               project='multirc_unp',
               tags=['multirc'])
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
            _, loss = run_model(batch, model, linear, answer_model, tokenizer, answer_tokenizer, args.topks, mode=args.mode)
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
