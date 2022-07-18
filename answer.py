from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import math
from tqdm import tqdm
import wandb
from dataset import prepare_answers, HotpotQADataset
from datasets import load_metric
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import get_scheduler, set_seed
from transformers.file_utils import PaddingStrategy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
import wandb
from utils import load_hotpotqa, get_args
from utils import prepare_optim_and_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(555)

@dataclass
class DataCollatorForQA:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        answers = [feature.pop(label_name) for feature in features]
        answers = [{"input_ids": x} for x in answers]
        contexts = [feature.pop("paras") for feature in features]

        batch = self.tokenizer.pad(
            answers,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Add back labels
        batch['contexts'] = contexts
        return batch

def get_relevant(tokenizer, contexts, para_indices, sent_indices):
    out = []
    for c, pidx, sidx in zip(contexts, para_indices, sent_indices):
        c1 = c[pidx[0]][sidx[0]]
        c2 = c[pidx[1]][sidx[1]]
        out.append(c1+c2)
    out = [{"input_ids": x} for x in out]
    out = tokenizer.pad(
        out,
        padding='longest',
        max_length=512,
        return_tensors="pt",
    )
    return out['input_ids'].to(device), out['attention_mask'].to(device)

def run_model(model, batch, tokenizer, train=True):
    bs = len(batch['input_ids'])
    para_indices = [(0, 1)] * bs
    sent_indices = [((0,), (0,))] * bs
    contexts, attention_mask = get_relevant(tokenizer, batch['contexts'], para_indices, sent_indices)
    batch['input_ids'] = batch['input_ids'].to(device)
    batch['input_ids'][batch['input_ids']==model.config.pad_token_id] = -100
    if train:
        outputs = model(input_ids=contexts, attention_mask=attention_mask, labels=batch['input_ids'])
    else:
        with torch.no_grad():
            outputs = model.generate(contexts, num_beams=2, min_length=0, max_length=20)
    return outputs

def evaluate(steps, args, model, tok, dataloader, split):
    exact_match = load_metric("exact_match")
    model.eval()
    results = []
    for step, eval_batch in enumerate(dataloader):
        gold = tok.batch_decode(eval_batch['input_ids'], skip_special_tokens=True)
        eval_outs = run_model(model, eval_batch, tok, train=False)
        preds = tok.batch_decode(eval_outs, skip_special_tokens=True)
        exact_match.add_batch(
            predictions=preds,
            references=gold,
        )
        results.append(eval_outs)
    eval_metric = exact_match.compute()
    if not args.nolog:
        wandb.log({
            "step": steps,
            f"{split} Acc": eval_metric})
    if args.save_results:
        torch.save(results, f"logging/{args.run_name}|step-{steps}.pt")
    return eval_metric['exact_match']

def prepare_dataloader(data, tok, args):
    (train_paras, valid_paras), (train_labels, valid_labels) = prepare_answers(tok, "train", data, threshold=args.max_paragraph_length)
    test_paras, test_labels = prepare_answers(tok, "validation", data, threshold=args.max_paragraph_length)
    train_dataset = HotpotQADataset(train_paras, train_labels)
    eval_dataset = HotpotQADataset(valid_paras, valid_labels)
    test_dataset = HotpotQADataset(test_paras, test_labels)
    data_collator = DataCollatorForQA(tok, padding='longest', max_length=512)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.eval_batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.eval_batch_size)
    return train_dataloader, eval_dataloader, test_dataloader

def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    model = model.to(device)

    data = load_hotpotqa()
    train_dataloader, eval_dataloader, test_dataloader = prepare_dataloader(data, tokenizer, args)

    model_name = args.model_dir.split('/')[-1]
    run_name=f'model-{model_name} lr-{args.learning_rate} bs-{args.batch_size*args.gradient_accumulation_steps} warmup-{args.warmup_ratio}'
    args.run_name = run_name

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.epoch * num_update_steps_per_epoch
    total_batch_size = args.batch_size * args.gradient_accumulation_steps
    optim, lr_scheduler = prepare_optim_and_scheduler([model], args)

    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    if not args.nolog:
        wandb.init(name=run_name,
               project='hotpotqa_answer',
               tags=['hotpotqa'])
        wandb.config.lr = args.learning_rate
        wandb.watch(model)

    best_valid = float('-inf')
    for epoch in range(args.epoch):
        for step, batch in enumerate(train_dataloader):
            if completed_steps % args.eval_steps == 0 and completed_steps > 0:
                valid_acc = evaluate(completed_steps, args, model, tokenizer, eval_dataloader, "Valid")
                evaluate(completed_steps, args, model, tokenizer, test_dataloader, "Test")
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    if args.save_model:
                        model.save_pretrained(f"{args.output_model_dir}/{run_name}")
            model.train()
            outputs = run_model(model, batch, tokenizer)
            loss = outputs.loss
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
