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
from transformers import set_seed
from transformers.file_utils import PaddingStrategy
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import wandb
from utils import load_hotpotqa, get_args, mean_pooling
from utils import prepare_linear, prepare_mlp, prepare_optim_and_scheduler

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
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

def run_model(layers, batch, train=True):
    model, linear, mlp = layers
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
    sentence_embeddings = sentence_embeddings.view(bs, num_choices, -1)
    single_outs = linear(sentence_embeddings).view(bs, -1)
    single_outs = m(single_outs)
    combs = torch.combinations(torch.arange(num_choices))
    C = len(combs)
    paired = sentence_embeddings[:,combs,:]
    diff = torch.abs(paired[:,:,0] - paired[:,:,1])
    pairs = torch.cat([paired.view(bs,C,-1), diff], dim=-1).view(-1, 3*sentence_embeddings.shape[-1])
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

def evaluate(steps, args, layers, dataloader, split):
    metric = load_metric("accuracy")
    layers[0].eval()
    if args.save_results:
        results = []
    for step, eval_batch in enumerate(dataloader):
        eval_outs = run_model(layers, eval_batch, train=False)
        predictions = eval_outs.argmax(dim=-1)
        if args.save_results:
            results.append(eval_outs.cpu())
        metric.add_batch(
            predictions=predictions,
            references=eval_batch["labels"],
        )
    eval_metric = metric.compute()
    if not args.nolog:
        wandb.log({
            "step": steps,
            f"{split} Acc": eval_metric})
    if args.save_results:
        torch.save(torch.cat(results, dim=0), f"logging/{args.run_name}|step-{steps}.pt")
    return eval_metric['accuracy']

def prepare_dataloader(data, tok, args):
    (train_paras, valid_paras), (train_labels, valid_labels) = prepare_paragraphs(tok, "train", data)
    test_paras, test_labels = prepare_paragraphs(tok, "validation", data)
    train_dataset = HotpotQADataset(train_paras, train_labels)
    eval_dataset = HotpotQADataset(valid_paras, valid_labels)
    test_dataset = HotpotQADataset(test_paras, test_labels)
    data_collator = DataCollatorForMultipleChoice(tok, padding='longest', max_length=512)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.eval_batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.eval_batch_size)
    return train_dataloader, eval_dataloader, test_dataloader

def prepare_model(args):
    model = AutoModel.from_pretrained(args.model_dir)
    model = model.to(device)
    linear = prepare_linear(model.config.hidden_size)
    mlp = prepare_mlp(model.config.hidden_size*3)
    return (model, linear, mlp)


def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    data = load_hotpotqa()
    train_dataloader, eval_dataloader, test_dataloader = prepare_dataloader(data, tokenizer, args)

    model_name = args.model_dir.split('/')[-1]
    run_name=f'model-{model_name} lr-{args.learning_rate} bs-{args.batch_size*args.gradient_accumulation_steps} warmup-{args.warmup_ratio}'
    args.run_name = run_name
    all_layers = prepare_model(args)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.epoch * num_update_steps_per_epoch
    total_batch_size = args.batch_size * args.gradient_accumulation_steps
    optim, lr_scheduler = prepare_optim_and_scheduler(all_layers, args)
    loss_fct = nn.CrossEntropyLoss()

    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    if not args.nolog:
        wandb.init(name=run_name,
               project='hotpotqa_embeddings',
               tags=['hotpotqa'])
        wandb.config.lr = args.learning_rate
        wandb.watch(all_layers[0])

    best_valid = float('-inf')
    for epoch in range(args.epoch):
        for step, batch in enumerate(train_dataloader):
            if completed_steps % args.eval_steps == 0 and completed_steps > 0:
                valid_acc = evaluate(completed_steps, args, all_layers, eval_dataloader, "Valid")
                evaluate(completed_steps, args, all_layers, test_dataloader, "Test")
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    if args.save_model:
                        all_layers[0].save_pretrained(f"{args.output_model_dir}/{run_name}")
            all_layers[0].train()
            outs = run_model(all_layers, batch)
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
