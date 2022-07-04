import argparse
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import math
from tqdm import tqdm
import wandb
from z_dataset import sentence_level_prepare, HotpotQADataset
from utils import padding
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
        paras = [p for ps in paras for p in ps]
        lengths = [len(p) for p in paras]
        paras = [p for ps in paras for p in ps]
        paras = [{"input_ids": x} for x in paras]

        batch = self.tokenizer.pad(
            paras,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if isinstance(labels[0], list):
            labels = [l for ls in labels for l in ls]
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        batch["lengths"] = torch.tensor(lengths, dtype=torch.int64)
        return batch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nolog', action='store_true')
    parser.add_argument('--baseline', action='store_true')
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
    parser.add_argument("--max_paragraph_length", default=10, type=int,
                        help="The maximum number of sentences allowed in a paragraph.")
    parser.add_argument("--model_dir", default="roberta-large", type=str,
                        help="The directory where the pretrained model will be loaded.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--output_model_dir", default="./saved_models", type=str,
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
    if args.baseline:
        args.max_paragraph_length = 1000
    return args

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def run_model(model, mlp, linear, batch, train=True, baseline=False):
    m = nn.Softmax(dim=-1)
    for key in batch:
        batch[key] = batch[key].to(device)
    if train:
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    else:
        with torch.no_grad():
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    sentence_embeddings = mean_pooling(outputs, batch["attention_mask"])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    embs_groupby_example = []
    embs_groupby_example0 = []
    embs_groupby_example1 = []
    indices = []
    indices0 = []
    indices1 = []
    st = 0
    for i in range(len(batch['lengths'])):
        curr = []
        ed = st + batch['lengths'][i]
        embs = sentence_embeddings[st:ed]
        curr.append(embs)
        pair_indices = torch.combinations(torch.arange(len(embs)), r=2)
        triple_indices = torch.combinations(torch.arange(len(embs)), r=3)
        curr.append(curr[-1][pair_indices, :].sum(dim=1))
        curr.append(curr[-1][triple_indices, :].sum(dim=1))
        curr = torch.cat(curr, dim=0)
        if baseline:
            embs_groupby_example.append(curr)
            indices += [i] * len(curr)
        else:
            if i % 2 == 0:
                embs_groupby_example0.append(curr)
                indices0.append(len(curr))
            else:
                embs_groupby_example1.append(curr)
                indices1.append(len(curr))
        st = ed
    if baseline:
        indices = torch.tensor(indices).to(device)
        embs_groupby_example = torch.cat(embs_groupby_example, dim=0)
        outs = linear(embs_groupby_example).view(-1)
        outs = padding(indices, outs)
        outs = torch.cat([outs, torch.zeros(len(outs), 1).to(device)], dim=1)
    else:
        assert len(embs_groupby_example0) == len(embs_groupby_example1)
        first_embs = torch.cat(embs_groupby_example0, dim=0)
        outs0 = linear(first_embs).view(-1)
        sec_embs = []
        prod_lens = []
        for i in range(len(embs_groupby_example0)):
            combs = torch.cartesian_prod(torch.arange(len(embs_groupby_example0[i])), torch.arange(len(embs_groupby_example1[i])))
            diff = torch.abs(embs_groupby_example0[i][combs[:, 0]] - embs_groupby_example1[i][combs[:, 1]])
            pairs = torch.cat([embs_groupby_example0[i][combs[:, 0]], embs_groupby_example1[i][combs[:, 1]], diff], dim=1)
            sec_embs.append(pairs)
        sec_embs = torch.cat(sec_embs, dim=0)
        outs1 = mlp(sec_embs).view(-1)
        st = 0
        idx = 0
        outs = []
        indices = []
        for i in range(len(indices0)):
            tmp = []
            for _ in range(indices0[i]):
                ed = st + indices1[i]
                tmp.append(outs0[idx] * outs1[st:ed])
                st = ed
                idx += 1
            tmp = torch.cat(tmp, dim=0)
            indices += [i] * len(tmp)
            outs.append(tmp)
        outs = torch.cat(outs, dim=0)
        indices = torch.tensor(indices).to(device)
        outs = padding(indices, outs)
        outs = torch.cat([outs, torch.zeros(len(outs), 1).to(device)], dim=1)
    return outs

def evaluate(steps, args, model, mlp, linear, dataloader, split):
    model.eval()
    if args.save_results:
        results = []
    acc = []
    for step, eval_batch in enumerate(dataloader):
        eval_outs = run_model(model, mlp, linear, eval_batch, train=False, baseline=args.baseline)
        predictions = eval_outs.argmax(dim=-1).view(-1, 1)
        labels = eval_batch["labels"].view(-1, 1)
        if args.baseline:
            predictions = predictions.view(-1, 2)
            labels = labels.view(-1, 2)
        acc.append(torch.all(predictions == labels, dim=1))
        if args.save_results:
            results.append(eval_outs.cpu())
    acc = torch.cat(acc)
    acc = acc.sum() / torch.numel(acc)
    if not args.nolog:
        wandb.log({
            "step": steps,
            f"{split} Acc": acc})
    if args.save_results:
        torch.save(results, f"logging/{args.run_name}|step-{steps}.pt")
    return acc

def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir)
    model = model.to(device)
    linear = nn.Linear(model.config.hidden_size, 1)
    linear = linear.to(device)
    mlp = nn.Sequential(
            nn.Linear(model.config.hidden_size*3, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
            )
    mlp = mlp.to(device)

    (train_paras, valid_paras), (train_labels, valid_labels) = \
            sentence_level_prepare(args.model_dir, "train", baseline=args.baseline, threshold=args.max_paragraph_length)
    test_paras, test_labels = sentence_level_prepare(args.model_dir, "validation", baseline=args.baseline, threshold=args.max_paragraph_length)
    train_dataset = HotpotQADataset(train_paras, train_labels)
    eval_dataset = HotpotQADataset(valid_paras, valid_labels)
    test_dataset = HotpotQADataset(test_paras, test_labels)
    data_collator = DataCollatorForMultipleChoice(tokenizer, padding='longest', max_length=512)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.eval_batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.eval_batch_size)

    model_name = args.model_dir.split('/')[-1]
    run_name=f'sent supp model-{model_name} lr-{args.learning_rate} bs-{args.batch_size*args.gradient_accumulation_steps} warmup-{args.warmup_ratio}'
    if args.baseline: run_name = 'baseline ' + run_name
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
            "params": [p for n, p in mlp.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in mlp.named_parameters() if any(nd in n for nd in no_decay)],
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
    loss_fct = nn.CrossEntropyLoss()

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
                valid_acc = evaluate(completed_steps, args, model, mlp, linear, eval_dataloader, "Valid")
                test_acc = evaluate(completed_steps, args, model, mlp, linear, test_dataloader, "Test")
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    if args.save_model:
                        model.save_pretrained(f"{args.output_model_dir}/{run_name}")
            model.train()
            outs = run_model(model, mlp, linear, batch, baseline=args.baseline)
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
