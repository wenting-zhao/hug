from collections import defaultdict
from itertools import combinations
import random
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import normalize_answer
import json
from rank_bm25 import BM25Okapi

def update_sp(preds, golds):
    sp_em, sp_f1, sp_prec, sp_recall = 0, 0, 0, 0
    for cur_sp_pred, gold_sp_pred in zip(preds, golds):
        tp, fp, fn = 0, 0, 0
        for e in cur_sp_pred:
            if e in gold_sp_pred:
                tp += 1
            else:
                fp += 1
        for e in gold_sp_pred:
            if e not in cur_sp_pred:
                fn += 1
        prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
        em = 1.0 if fp + fn == 0 else 0.0
        sp_em += em
        sp_f1 += f1
        sp_prec += prec
        sp_recall += recall
    sp_em /= len(preds)
    sp_f1 /= len(preds)
    sp_prec /= len(preds)
    sp_recall /= len(preds)
    return sp_em, sp_f1, sp_prec, sp_recall

num_sent = int(sys.argv[1])
option = sys.argv[2]
all_refs, all_preds = [], []
data = []
with open(f"data/musique/musique_ans_v1.0_dev.jsonl", 'r') as fin:
    for line in fin:
        data.append(json.loads(line))
for d in data:
    q = d["question"]
    ps = []
    supp = []
    for p in d["paragraphs"]:
        ps.append(p["paragraph_text"])
        if p["is_supporting"]:
            supp.append(p["idx"])
    ratios = []
    mapping = []
    if option == "bm25":
        for i, p in enumerate(ps):
            for j, sp in enumerate(p):
                mapping.append((i, j))
        corpus = [pp for p in ps for pp in p]
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = q.split(" ")
        flattened = bm25.get_scores(tokenized_query)
    else:
        for i, p in enumerate(ps):
            subratios = []
            for j, sp in enumerate(p):
                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform([sp, q])
                dense = vectors.todense()
                subratio = dense[0] * dense[1].T
                subratio = subratio[0, 0]
                subratios.append(subratio)
                mapping.append((i, j))
            ratios.append(subratios)
        flattened = [rr for r in ratios for rr in r]
    preds = np.argsort(flattened)[-num_sent:]
    pred_dict = defaultdict(list)
    for pred in preds:
        pred = mapping[pred]
        pred_dict[pred[0]].append(pred[1])
    all_refs.append(supp)
    all_preds.append(pred_dict)
print(update_sp(all_preds, all_refs))
