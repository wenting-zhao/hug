from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import math
from tqdm import tqdm
import wandb
from dataset import prepare_pipeline, UnsupHotpotQADataset
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
from utils import get_args, load_hotpotqa, mean_pooling, padding
from utils import prepare_linear, prepare_mlp, prepare_optim_and_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(555)

label2ij = []
cnt = 0
for i in range(10):
    for j in range(i+1, 10):
        label2ij.append([i, j])
        cnt += 1
label2ij = torch.tensor(label2ij).to(device)

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        batch_size = len(features)
        num_choices = len(features[0]["paras"])
        plabel_name = "para_labels"
        plabels = [feature.pop(plabel_name) for feature in features]
        slabel_name = "sent_labels"
        slabels = [feature.pop(slabel_name) for feature in features]
        max_slabel = max([len(l) for l in slabels])
        slabels = [slabels[i] + [0] * (max_slabel - len(slabels[i])) for i in range(len(slabels))]
        paras = [feature.pop("paras") for feature in features]
        paras = list(chain(*paras))
        paras = [{"input_ids": x} for x in paras]

        batch = self.tokenizer.pad(
            paras,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}

        answer_name = "answs"
        raw_answers = [feature.pop(answer_name) for feature in features]
        context_name = "supps"
        contexts = [feature.pop(context_name) for feature in features]

        # Add back labels
        batch['contexts'] = contexts
        batch['answers'] = raw_answers
        batch["plabels"] = torch.tensor(plabels, dtype=torch.int64)
        batch["slabels"] = torch.tensor(slabels, dtype=torch.float)
        return batch

def prepare_model(args):
    model = AutoModel.from_pretrained(args.model_dir)
    model = model.to(device)
    linear = prepare_linear(model.config.hidden_size)
    mlp = prepare_mlp(model.config.hidden_size*3)
    sent_linear = prepare_linear(model.config.hidden_size)
    return [model, linear, mlp, sent_linear]

def prepare_dataloader(data, tok, answer_tok, args):
    data = prepare_pipeline(tok, answer_tok, data)
    train_dataset = UnsupHotpotQADataset(data["train"])
    eval_dataset = UnsupHotpotQADataset(data["valid"])
    test_dataset = UnsupHotpotQADataset(data["test"])
    data_collator = DataCollatorForMultipleChoice(tok, padding='longest', max_length=512)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.eval_batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.eval_batch_size)
    return train_dataloader, eval_dataloader, test_dataloader

def run_lm(model, batch, bs, num_choices, train):
    input_ids = batch["input_ids"].view(bs*num_choices, -1)
    attention_mask = batch["attention_mask"].view(bs*num_choices, -1)
    if train:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    else:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs, attention_mask

def run_para_model(layers, outputs, attention_mask, bs, num_choices, train):
    linear, mlp = layers
    m = nn.Softmax(dim=-1)
    sentence_embeddings = mean_pooling(outputs, attention_mask)
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    sentence_embeddings = sentence_embeddings.view(bs, num_choices, -1)
    if train:
        single_outs = linear(sentence_embeddings).view(bs, -1)
    else:
        with torch.no_grad():
            single_outs = linear(sentence_embeddings).view(bs, -1)
    single_outs = m(single_outs)
    combs = torch.combinations(torch.arange(num_choices))
    C = len(combs)
    paired = sentence_embeddings[:,combs,:]
    diff = torch.abs(paired[:,:,0] - paired[:,:,1])
    pairs = torch.cat([paired.view(bs,C,-1), diff], dim=-1).view(-1, 3*sentence_embeddings.shape[-1])
    if train:
        outs = mlp(pairs).view(bs, -1)
    else:
        with torch.no_grad():
            outs = mlp(pairs).view(bs, -1)
    outs = m(outs)
    res = []
    st, ed = 0, 9
    for i in range(9):
        res.append(single_outs[:, i].view(bs, -1) * outs[:, st:ed])
        st = ed
        ed += (9-i-1)
    res = torch.cat(res, dim=1)
    return outs

def process_para_outs(pouts, lm_outputs, raw_input_ids, bs, num_choices, max_p):
    para_preds = pouts.argmax(dim=-1)
    ijs = torch.stack([label2ij[x] for x in para_preds])
    _, seq_len, emb_len = lm_outputs[0].shape
    ids = torch.arange(bs)[:, None]
    sent_in = lm_outputs[0].view(bs, num_choices, seq_len, emb_len)
    if max_p:
        sent_in = sent_in[ids, ijs]
        input_ids = raw_input_ids[ids, ijs]
        input_ids = input_ids.view(bs*2, seq_len)
        sent_in = sent_in.view(bs*2, seq_len, -1)
    else:
        input_ids = raw_input_ids.view(bs*10, seq_len)
        sent_in = sent_in.view(bs*10, seq_len, -1)
    return input_ids, sent_in, ijs

def run_sent_model(linear, tok, input_ids, embs, train):
    indices = (input_ids == tok.unk_token_id).nonzero(as_tuple=False)
    outs = embs[indices[:, 0], indices[:, 1]]
    if train:
        outs = linear(outs)
    else:
        with torch.no_grad():
            outs = linear(outs)
    outs = torch.sigmoid(outs).view(-1)
    x = indices[:, 0]
    L = len(x)
    rows = torch.nn.functional.one_hot(x)
    cols = rows.cumsum(0)[torch.arange(L), x] - 1
    cols = torch.nn.functional.one_hot(cols)
    outs = (outs[:, None, None] *
             cols[:, None, :] *
             rows[:, :, None]).sum(0)
    return outs

def process_sent_outs(souts, max_p):
    sent_preds = (souts > 0.1).nonzero(as_tuple=True)
    values = souts[sent_preds[0], sent_preds[1]]
    values = padding(sent_preds[0], values)
    values[values==0] = 1
    values = values.prod(dim=-1)
    sent_preds = [sent_preds[0].tolist(), sent_preds[1].tolist()]
    ls = [[] for _ in range(sent_preds[0][-1]+1)]
    for x, y in zip(sent_preds[0], sent_preds[1]):
        ls[x].append(y)
    if max_p:
        ls = [ls[i:i+2] for i in range(0, len(ls), 2)]
    else:
        ls = [ls[i:i+10] for i in range(0, len(ls), 10)]
    return values, ls

def get_relevant(tokenizer, contexts, raw_answers, para_indices, sent_indices, max_p):
    out = []
    if max_p:
        for c, pidx, sidx in zip(contexts, para_indices, sent_indices):
            q, supp = c
            c1, c2 = [], []
            for j in sidx[0]:
                c1 += supp[pidx[0]][j]
            for j in sidx[1]:
                c2 += supp[pidx[1]][j]
            out.append(q+c1+c2)
    else:
        for i, c in enumerate(contexts):
            q, supp = c
            for pair in label2ij:
                c1, c2 = [], []
                p1, p2 = pair
                for j in sent_indices[i][p1]:
                    c1 += supp[p1][j]
                for j in sent_indices[i][p2]:
                    c1 += supp[p2][j]
                out.append(q+c1+c2)
    out = [{"input_ids": x} for x in out]
    out = tokenizer.pad(
        out,
        padding='longest',
        max_length=512,
        return_tensors="pt",
    )

    if not max_p:
        raw_answers = [[x] * len(label2ij) for x in raw_answers]
        raw_answers = [x for sub in raw_answers for x in sub]
    raw_answers = [{"input_ids": x} for x in raw_answers]
    answers = tokenizer.pad(
        raw_answers,
        padding='longest',
        max_length=512,
        return_tensors="pt",
    )["input_ids"]

    return out['input_ids'].to(device), out['attention_mask'].to(device), answers.to(device)

def run_answer_model(model, input_ids, attn_mask, answs, tokenizer, train):
    answs[answs==model.config.pad_token_id] = -100
    if train:
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=answs)
    else:
        with torch.no_grad():
            outputs = model.generate(input_ids, num_beams=2, min_length=1, max_length=20)
    return outputs

def run_model(batch, layers, answer_model, tokenizer, answer_tokenizer, max_p, train=True):
    for key in batch:
        if key != "contexts" and key != "answers":
            batch[key] = batch[key].to(device)
    bs = len(batch['plabels'])
    num_choices = len(batch['input_ids'][0])
    lm_outputs, attention_mask = run_lm(layers[0], batch, bs, num_choices, train=train)
    pouts = run_para_model(layers[1:3], lm_outputs, attention_mask, bs, num_choices, train=train)
    input_ids, sent_in, para_ids = process_para_outs(
        pouts, lm_outputs, batch['input_ids'], bs, num_choices, max_p=max_p)
    sent_out = run_sent_model(layers[-1], tokenizer, input_ids, sent_in, train=train)
    sent_values, sent_idx = process_sent_outs(sent_out, max_p=max_p)
    answer_in, answer_attn, labels = get_relevant(
            answer_tokenizer, batch["contexts"], batch['answers'],
            para_ids.tolist(), sent_idx, max_p=max_p)
    answ_out = run_answer_model(answer_model, answer_in, answer_attn, labels, answer_tokenizer, train=train)
    if max_p:
        if train:
            loss = answ_out.loss.view(bs, -1).mean(-1)
            pout_max = torch.max(pouts, dim=-1)[0]
            loss -= torch.log(pout_max)
            sent_max = sent_values.view(bs, -1).mean(-1)
            loss -= torch.log(sent_max)
            loss = loss.mean()
        else:
            loss = 0.
    else:
        combs = torch.combinations(torch.arange(num_choices))
        C = len(combs)
        sent_values = sent_values.view(bs, -1)
        paired = sent_values[:,combs]
        sent_product = paired[:,:,0] * paired[:,:,1]
        sent_product = sent_product.view(bs, C)
        loss = answ_out.loss.reshape(bs, len(label2ij), -1).mean(dim=-1)
        loss -= torch.log(pouts)
        loss -= torch.log(sent_product)
        loss = loss.mean()
    return answ_out, para_ids, sent_idx, loss

def evaluate(steps, args, layers, answ_model, tok, answ_tok, dataloader, split):
    exact_match = load_metric("exact_match")
    layers[0].eval()
    answ_model.eval()
    results = []
    para_acc = []
    for step, eval_batch in enumerate(dataloader):
        gold = answ_tok.batch_decode(eval_batch['answers'], skip_special_tokens=True)
        eval_outs, para_ids, sent_ids, _ = run_model(
                eval_batch, layers, answ_model, tok, answ_tok, max_p=True, train=False)
        preds = tok.batch_decode(eval_outs, skip_special_tokens=True)
        exact_match.add_batch(
            predictions=preds,
            references=gold,
        )
        results.append(eval_outs)
        gold_paras = torch.stack([label2ij[x] for x in eval_batch['plabels']])
        para_acc.append(torch.all(gold_paras == para_ids, dim=1))
    eval_metric = exact_match.compute()
    para_acc = torch.cat(para_acc)
    para_acc = para_acc.sum() / torch.numel(para_acc)
    if not args.nolog:
        wandb.log({
            "step": steps,
            f"{split} Para": para_acc.item(),
            f"{split} Acc": eval_metric})
    if args.save_results:
        torch.save(results, f"logging/{args.run_name}|step-{steps}.pt")
    return eval_metric['exact_match']


def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    answer_tokenizer = AutoTokenizer.from_pretrained(args.answer_model_dir)
    data = load_hotpotqa()
    train_dataloader, eval_dataloader, test_dataloader = prepare_dataloader(data, tokenizer, answer_tokenizer, args)

    model_name = args.model_dir.split('/')[-1]
    if args.max_p:
        run_name=f'max_p model-{model_name} lr-{args.learning_rate} bs-{args.batch_size*args.gradient_accumulation_steps} warmup-{args.warmup_ratio}'
    else:
        run_name=f'model-{model_name} lr-{args.learning_rate} bs-{args.batch_size*args.gradient_accumulation_steps} warmup-{args.warmup_ratio}'
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
               project='hotpotqa_unsup',
               tags=['hotpotqa'])
        wandb.config.lr = args.learning_rate
        wandb.watch(all_layers[0])
        wandb.watch(answer_model)

    best_valid = float('-inf')
    for epoch in range(args.epoch):
        for step, batch in enumerate(train_dataloader):
            if completed_steps % args.eval_steps == 0 and completed_steps > 0:
                valid_acc = evaluate(completed_steps, args, all_layers, answer_model,
                                         tokenizer, answer_tokenizer, eval_dataloader, "Valid")
                evaluate(completed_steps, args, all_layers, answer_model,
                             tokenizer, answer_tokenizer, test_dataloader, "Test")
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    if args.save_model:
                        all_layers[0].save_pretrained(f"{args.output_model_dir}/{run_name}")
                all_layers[0].train()
                answer_model.train()
            answ, _, _, loss = run_model(batch, all_layers, answer_model, tokenizer, answer_tokenizer, max_p=args.max_p)
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
