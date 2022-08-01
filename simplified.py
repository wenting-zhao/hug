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
from utils import prepare_linear, prepare_mlp, prepare_optim_and_scheduler

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
        return batch

def prepare_model(args):
    model = AutoModel.from_pretrained(args.model_dir)
    model = model.to(device)
    mlp = prepare_mlp(model.config.hidden_size*3)
    return [model, mlp]

def prepare_dataloader(data, tok, answer_tok, args):
    paras, supps, answs = prepare_simplified(tok, answer_tok, "train", data, max_sent=args.max_paragraph_length, k=args.k_distractor, fixed=args.truncate_paragraph)
    tparas, tsupps, tansws = prepare_simplified(tok, answer_tok, "validation", data, max_sent=args.max_paragraph_length, k=args.k_distractor, fixed=args.truncate_paragraph)
    train_dataset = SimplifiedHotpotQADataset(paras[0], supps[0], answs[0])
    eval_dataset = SimplifiedHotpotQADataset(paras[1], supps[1], answs[1])
    test_dataset = SimplifiedHotpotQADataset(tparas, tsupps, tansws)
    data_collator = DataCollatorForMultipleChoice(tok, padding='longest', max_length=512)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.eval_batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.eval_batch_size)
    return train_dataloader, eval_dataloader, test_dataloader

def run_lm(model, batch, bs, tot):
    input_ids = batch["input_ids"].view(bs*tot, -1)
    attention_mask = batch["attention_mask"].view(bs*tot, -1)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs, attention_mask

def run_para_model(mlp, outputs, attention_mask, bs, tot):
    m = nn.LogSoftmax(dim=-1)
    sentence_embeddings = mean_pooling(outputs, attention_mask)
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    sentence_embeddings = sentence_embeddings.view(bs, tot, -1)
    combs = torch.combinations(torch.arange(tot))
    C = len(combs)
    paired = sentence_embeddings[:,combs,:]
    diff = torch.abs(paired[:,:,0] - paired[:,:,1])
    pairs = torch.cat([paired.view(bs,C,-1), diff], dim=-1).view(-1, 3*sentence_embeddings.shape[-1])
    outs = mlp(pairs).view(bs, -1)
    outs = m(outs)
    return outs

def pad_answers(tokenizer, contexts, raw_answers):
    contexts = [c for cs in contexts for c in cs]
    contexts = [{"input_ids": x} for x in contexts]
    out = tokenizer.pad(
        contexts,
        padding='longest',
        return_tensors="pt",
    )
    raw_answers = [{"input_ids": x} for x in raw_answers]
    answers = tokenizer.pad(
        raw_answers,
        padding='longest',
        return_tensors="pt",
        return_attention_mask=False,
    )['input_ids']
    n = len(contexts) // len(raw_answers)
    answers = answers.repeat(1, n)
    answers = answers.view(answers.shape[0]*n, -1) 

    return out['input_ids'].to(device), out['attention_mask'].to(device), answers.to(device)

def run_answer_model(model, input_ids, attn_mask, answs, tokenizer, beam, train):
    answs[answs==model.config.pad_token_id] = -100
    if train:
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=answs)
    else:
        outputs = model.generate(input_ids, num_beams=beam, min_length=1, max_length=20)
        scores = model(input_ids=input_ids, attention_mask=attn_mask, labels=answs).loss
        outputs = (outputs, scores)
    return outputs

def run_model(batch, layers, answer_model, tokenizer, answer_tokenizer, max_p, reg_coeff, t, beam=2, train=True):
    for key in batch:
        if key != "contexts" and key != "answers":
            batch[key] = batch[key].to(device)
    bs = len(batch["answers"])
    tot = len(batch['input_ids'][0])
    num_choices = len(batch['contexts'][0])
    lm_outputs, attention_mask = run_lm(layers[0], batch, bs, tot)
    pouts = run_para_model(layers[1], lm_outputs, attention_mask, bs, tot)
    answer_in, answer_attn, labels = pad_answers(
            answer_tokenizer, batch["contexts"], batch['answers'])
    answ_out = run_answer_model(answer_model, answer_in, answer_attn, labels, answer_tokenizer, beam=beam, train=train)
    if train:
        loss = answ_out.loss.view(bs, num_choices, -1)
        loss = (-loss).sum(dim=-1)
        loss += pouts
        if reg_coeff > 0:
            normalized = torch.exp(loss)
        loss = torch.logsumexp(loss, dim=-1)
        loss = -loss.mean()
        if reg_coeff > 0:
            entropy = torch.mean(-torch.sum(normalized * torch.log(normalized + 1e-9), dim = 1), dim = 0)
            loss += reg_coeff * entropy
    else:
        loss = 0.
    return answ_out, pouts, loss

def evaluate(steps, args, layers, answ_model, tok, answ_tok, dataloader, split):
    exact_match = load_metric("exact_match")
    metric = load_metric("accuracy")
    prior_exact_match = load_metric("exact_match")
    prior_metric = load_metric("accuracy")
    prior_ents = []
    pos_ents = []
    if args.save_results and split == "Valid":
        para_results = []
        answ_results = []
    for step, eval_batch in enumerate(dataloader):
        bs = len(eval_batch["answers"])
        num_choices = len(eval_batch['contexts'][0])
        gold = answ_tok.batch_decode(eval_batch['answers'], skip_special_tokens=True)
        eval_outs, para_preds, _ = run_model(
                eval_batch, layers, answ_model, tok, answ_tok, max_p=True,
                reg_coeff=args.reg_coeff, t=args.sentence_thrshold, train=False, beam=args.beam)
        eval_outs, scores = eval_outs
        eval_outs = eval_outs.view(bs, num_choices, -1)
        scores = scores.view(bs, num_choices, -1)
        scores = -scores
        mask = scores!=0
        scores = (scores*mask).sum(dim=-1)/mask.sum(dim=-1)
        scores = scores + para_preds
        idxes = scores.argmax(dim=-1).view(-1, 1)
        pos_eval_outs = eval_outs[torch.arange(len(eval_outs))[:,None], idxes]
        pos_eval_outs = pos_eval_outs.view(bs, -1)
        preds = tok.batch_decode(pos_eval_outs, skip_special_tokens=True)
        gold = [normalize_answer(s) for s in gold]
        preds = [normalize_answer(s) for s in preds]
        if args.save_results and split == "Valid":
            answ_results.append((preds, gold))
        exact_match.add_batch(
            predictions=preds,
            references=gold,
        )
        predictions = idxes.view(-1)
        labels = [0] * len(predictions)
        metric.add_batch(
            predictions=predictions,
            references=labels,
        )
        normal_scores = torch.exp(scores)
        pos_entropy = -torch.sum(normal_scores * torch.log(normal_scores + 1e-9), dim = 1)
        pos_ents += pos_entropy.cpu().tolist()
        predictions = para_preds.argmax(dim=-1)
        prior_metric.add_batch(
            predictions=predictions,
            references=labels,
        )
        if args.save_results and split == "Valid":
            para_results += predictions.cpu().tolist()
        idxes = predictions.view(-1, 1)
        prior_eval_outs = eval_outs[torch.arange(len(eval_outs))[:,None], idxes]
        prior_eval_outs = prior_eval_outs.view(bs, -1)
        preds = tok.batch_decode(prior_eval_outs, skip_special_tokens=True)
        preds = [normalize_answer(s) for s in preds]
        prior_exact_match.add_batch(
            predictions=preds,
            references=gold,
        )
        normal_scores = torch.exp(para_preds)
        prior_entropy = -torch.sum(normal_scores * torch.log(normal_scores + 1e-9), dim = 1)
        prior_ents += prior_entropy.cpu().tolist()
    pos_eval_metric = exact_match.compute()
    pos_para_acc = metric.compute()
    prior_eval_metric = prior_exact_match.compute()
    prior_para_acc = prior_metric.compute()
    if not args.nolog:
        wandb.log({
            "step": steps,
            f"{split} Prior Entropy": sum(prior_ents) / len(prior_ents),
            f"{split} Posterior Entropy": sum(pos_ents) / len(pos_ents),
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
    train_dataloader, eval_dataloader, test_dataloader = prepare_dataloader(data, tokenizer, answer_tokenizer, args)

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
               project='hotpotqa_unsup_simplified_entropy_test',
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
                    evaluate(completed_steps, args, all_layers, answer_model,
                                 tokenizer, answer_tokenizer, test_dataloader, "Test")
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
