import argparse
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import math
from tqdm import tqdm
import wandb
from dataset import prepare_paragraphs, HotpotQADataset
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
from utils import load_hotpotqa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(555)

cnt = 0
label2ij = dict()
for i in range(10):
    for j in range(i+1, 10):
        label2ij[cnt] = (i, j)
        cnt += 1

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
        num_choices = len(features[0]["paras"])
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
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
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.float)
        return batch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nolog', action='store_true')
    parser.add_argument('--save_model', action='store_true')
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

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def run_model(model, linear, batch, train=True):
    m = nn.Softmax(dim=-1)
    bs = len(batch['labels'])
    num_choices = len(batch['input_ids'][0])
    for key in batch:
        if key != "labels":
            batch[key] = batch[key].view(bs*num_choices, -1)
        batch[key] = batch[key].to(device)
    if train:
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    else:
        with torch.no_grad():
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    sentence_embeddings = mean_pooling(outputs, batch["attention_mask"])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    outs = linear(sentence_embeddings)
    outs = outs.view(bs, -1)
    outs = m(outs)
    return outs

def evaluate(steps, args, model, linear, dataloader, split):
    metric = load_metric("accuracy", "multilabel")
    model.eval()
    for step, eval_batch in enumerate(dataloader):
        eval_outs = run_model(model, linear, eval_batch, train=False)
        top2 = torch.topk(eval_outs, 2)[0][:, -1].view(-1, 1)
        eval_outs[eval_outs>=top2] = 1
        eval_outs[eval_outs!=1] = 0
        eval_outs = eval_outs.int().cpu().tolist()
        labels = eval_batch["labels"].int().cpu().tolist()
        metric.add_batch(
            predictions=eval_outs,
            references=labels,
        )
    eval_metric = metric.compute()
    if not args.nolog:
        wandb.log({
            "step": steps,
            f"{split} Acc": eval_metric})
    return eval_metric['accuracy']

def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir)
    model = model.to(device)
    linear = nn.Linear(model.config.hidden_size, 1)
    linear = linear.to(device)

    data = load_hotpotqa()
    (train_paras, valid_paras), (train_labels, valid_labels) = prepare_paragraphs(tokenizer, "train", data, baseline=True)
    test_paras, test_labels = prepare_paragraphs(tokenizer, "validation", data, baseline=True)
    train_dataset = HotpotQADataset(train_paras, train_labels)
    eval_dataset = HotpotQADataset(valid_paras, valid_labels)
    test_dataset = HotpotQADataset(test_paras, test_labels)
    data_collator = DataCollatorForMultipleChoice(tokenizer, padding='longest', max_length=512)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.eval_batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.eval_batch_size)

    model_name = args.model_dir.split('/')[-1]
    run_name=f'baseline model-{model_name} lr-{args.learning_rate} bs-{args.batch_size*args.gradient_accumulation_steps} warmup-{args.warmup_ratio}'

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
               project='hotpotqa_embeddings',
               tags=['hotpotqa'])
        wandb.config.lr = args.learning_rate
        wandb.watch(model)

    best_valid = float('-inf')
    for epoch in range(args.epoch):
        for step, batch in enumerate(train_dataloader):
            if completed_steps % args.eval_steps == 0 and completed_steps > 0:
                valid_acc = evaluate(completed_steps, args, model, linear, eval_dataloader, "Valid")
                evaluate(completed_steps, args, model, linear, test_dataloader, "Test")
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    if args.save_model:
                        model.save_pretrained(f"{args.output_model_dir}/{run_name}")
            model.train()
            outs = run_model(model, linear, batch)
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
