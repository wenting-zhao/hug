import argparse
from datasets import load_dataset
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_hotpotqa():
    return load_dataset('hotpot_qa', 'distractor')

def padding(indices, values):
    L = len(indices)
    rows = torch.nn.functional.one_hot(indices)
    cols = rows.cumsum(0)[torch.arange(L), indices] - 1
    cols = torch.nn.functional.one_hot(cols)
    outs = (values[:, None, None] *
            cols[:, None, :] *
            rows[:, :, None]).sum(0)
    return outs

def padding_long(indices, values):
    final = []
    length = max(indices)
    st = 0
    for i in indices:
        curr = values[st:st+i]
        curr_zeros = torch.zeros(length-i).to(device)
        curr = torch.cat([curr, curr_zeros], dim=0)
        final.append(curr)
        st += i
    outs = torch.stack(final)
    return outs

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

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
    parser.add_argument("--max_matrix", default=5000, type=int,
                        help="The largest n when doing matrix operation.")
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
