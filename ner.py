import sys
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from tqdm import tqdm
import torch
from utils import load_hotpotqa

split = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=0)

data = load_hotpotqa()[split]
ners = []
for context in tqdm(data["context"]):
    sents = context["sentences"]
    curr_ners = []
    for sent in sents:
        outs = nlp(sent)
        outs = [[o["word"] for o in out] for out in outs] 
        curr_ners.append(outs)
    ners.append(curr_ners)
torch.save(ners, f"cache/hotpotqa_{split}.pt")
