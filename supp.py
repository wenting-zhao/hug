import argparse
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import math
from tqdm import tqdm
import wandb
from z_dataset import prepare, HotpotQADataset
from datasets import load_metric
from transformers import AutoModel
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import get_scheduler, set_seed
from transformers.file_utils import PaddingStrategy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(555)
THRESHOLDS=[0.01, 0.02, 0.03, 0.04, 0.05,
            0.06, 0.07, 0.08, 0.09, 0.10,
            0.15, 0.20, 0.25, 0.30, 0.35,
            0.40, 0.45, 0.50, 0.55, 0.60,
            0.65, 0.70, 0.75, 0.80, 0.85,
            0.90, 0.95]

@dataclass
class DataCollatorForMultipleChoice:
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
        batch_size = len(features)
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        paras = [feature.pop("paras") for feature in features]
        paras = [{"input_ids": x} for x in paras]

        batch = self.tokenizer.pad(
            paras,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_label = max([len(l) for l in labels])
        labels = [labels[i] + [0] * (max_label - len(labels[i])) for i in range(len(labels))]
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.float)
        return batch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nolog', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument("--batch_size", '-b', default=1, type=int,
                        help="batch size per gpu.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="eval batch size per gpu.")
    parser.add_argument("--eval_steps", default=5000, type=int,
                        help="number of steps between each evaluation.")
    parser.add_argument("--epoch", '-epoch', default=10, type=int,
                        help="The number of epochs for fine-tuning.")
    parser.add_argument("--model_dir", default="roberta-large", type=str,
                        help="The directory where the pretrained model will be loaded.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--output_model_dir", default="./saved_supp_models", type=str,
                        help="The directory where the pretrained model will be saved.")
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Warmup ratio in the lr scheduler."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    args = parser.parse_args()
    return args

def run_model(model, linear, tok, batch, train=True):
    bs = len(batch['labels'])
    for key in batch:
        batch[key] = batch[key].to(device)
    if train:
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    else:
        with torch.no_grad():
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    indices = (batch['input_ids'] == tok.unk_token_id).nonzero(as_tuple=False)
    outs = outputs.last_hidden_state[indices[:, 0], indices[:, 1]]
    outs = linear(outs)
    outs = torch.sigmoid(outs).view(-1)
    #final = []
    #length = batch['labels'].shape[1]
    #st = 0
    #for i in range(bs):
    #    num = len((indices[:, 0] == i).nonzero())
    #    curr = outs[st:st+num]
    #    curr_zeros = torch.zeros(length-num).to(device)
    #    curr = torch.cat([curr, curr_zeros], dim=0)
    #    final.append(curr)
    #    st += num
    #outs = torch.stack(final)
    # the following code is adapted from sasha's suggestion
    x = indices[:, 0]
    L = len(x)
    rows = torch.nn.functional.one_hot(x)
    cols = rows.cumsum(0)[torch.arange(L), x] - 1
    cols = torch.nn.functional.one_hot(cols)
    outs = (outs[:, None, None] *
             cols[:, None, :] *
             rows[:, :, None]).sum(0)
    if outs.shape[1] < batch['labels'].shape[1]:
        outs = torch.cat([outs, torch.zeros(bs, batch['labels'].shape[1]-outs.shape[1]).to(device)], dim=1)
    return outs

def evaluate(steps, args, model, linear, tok, dataloader, split, threshold=THRESHOLDS):
    model.eval()
    results = []
    labels = []
    for step, eval_batch in enumerate(dataloader):
        eval_outs = run_model(model, linear, tok, eval_batch, train=False)
        results.append(eval_outs)
        labels.append(eval_batch["labels"])
    best_acc, best_t, best_preds = 0, 0, 0
    for t in THRESHOLDS:
        acc = []
        for pred, ref in zip(results, labels):
            curr = pred.detach().clone()
            curr[curr>t] = 1
            curr[curr<=t] = 0
            acc.append(torch.all(curr == ref, dim=1))
        acc = torch.cat(acc)
        acc = acc.sum() / torch.numel(acc)
        if acc > best_acc:
            best_acc, best_t = acc.item(), t
    if not args.nolog:
        wandb.log({
            "step": steps,
            "threshold": best_t,
            f"{split} Acc": best_acc})
    if args.save_results:
        torch.save(results, f"logging/{args.run_name}|step-{steps}.pt")
    return best_acc, best_t

def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir)
    model = model.to(device)
    linear = nn.Linear(model.config.hidden_size, 1)
    linear = linear.to(device)

    (train_paras, valid_paras), (train_labels, valid_labels) = prepare(args.model_dir, "train")
    test_paras, test_labels = prepare(args.model_dir, "validation")
    train_dataset = HotpotQADataset(train_paras, train_labels)
    eval_dataset = HotpotQADataset(valid_paras, valid_labels)
    test_dataset = HotpotQADataset(test_paras, test_labels)
    data_collator = DataCollatorForMultipleChoice(tokenizer, padding='longest', max_length=512)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.eval_batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.eval_batch_size)

    model_name = args.model_dir.split('/')[-1]
    run_name=f'supp model-{model_name} lr-{args.learning_rate} bs-{args.batch_size*args.gradient_accumulation_steps} warmup-{args.warmup_ratio}'
    args.run_name = run_name

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in linear.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in linear.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optim = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    loss_fct = nn.BCELoss()

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.epoch * num_update_steps_per_epoch
    total_batch_size = args.batch_size * args.gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optim,
        num_warmup_steps=int(args.warmup_ratio*args.max_train_steps),
        num_training_steps=args.max_train_steps,
    )

    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    if not args.nolog:
        wandb.init(name=run_name,
               project='hotpotqa_supp',
               tags=['hotpotqa'])
        wandb.config.lr = args.learning_rate
        wandb.watch(model)

    best_valid = float('-inf')
    for epoch in range(args.epoch):
        for step, batch in enumerate(train_dataloader):
            if completed_steps % args.eval_steps == 0 and completed_steps > 0:
                valid_acc, best_t = evaluate(completed_steps, args, model, linear, tokenizer, eval_dataloader, "Valid")
                evaluate(completed_steps, args, model, linear, tokenizer, test_dataloader, "Test", [best_t])
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    if args.save_model:
                        model.save_pretrained(f"{args.output_model_dir}/{run_name}")
            model.train()
            outs = run_model(model, linear, tokenizer, batch)
            loss = loss_fct(outs, batch["labels"])
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
