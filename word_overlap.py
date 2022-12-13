from collections import defaultdict
from itertools import combinations
import random
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import load_hotpotqa, normalize_answer
from rank_bm25 import BM25Okapi

def update_sp(preds, golds):
    sp_em, sp_f1, sp_prec, sp_recall = 0, 0, 0, 0
    for cur_sp_pred, gold_sp_pred in zip(preds, golds):
        tp, fp, fn = 0, 0, 0
        for e in cur_sp_pred:
            if e in gold_sp_pred:
                for v in cur_sp_pred[e]:
                    if v in gold_sp_pred[e]:
                        tp += 1
                    else:
                        fp += 1
            else:
                for v in cur_sp_pred[e]:
                    fp += 1
        for e in gold_sp_pred:
            if e not in cur_sp_pred:
                fn += len(gold_sp_pred[e])
            else:
                for v in gold_sp_pred[e]:
                    if v not in cur_sp_pred[e]:
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
    return sp_prec, sp_recall, sp_em, sp_f1

data = load_hotpotqa()["validation"]
num_sent = int(sys.argv[1])
option = sys.argv[2]
all_refs, all_preds = [], []
for context, q, supp in zip(data["context"], data["question"], data['supporting_facts']):
    ts = context["title"]
    sents = context["sentences"]
    l = []
    for i in range(len(ts)):
        if ts[i] in supp["title"]:
            l.append(i)
    assert len(l) == 2
    assert l[0] < l[1]
    x, y = l
    tmp = defaultdict(list)
    for t, sid in zip(supp["title"], supp["sent_id"]):
        if t == ts[x]:
            if sid >= len(sents[x]):
                print("INDEX OUT OF RANGE", sid, len(sents[x]))
                continue
            tmp[x].append(sid)
        else:
            if sid >= len(sents[y]):
                print("INDEX OUT OF RANGE", sid, len(sents[y]))
                continue
            tmp[y].append(sid)
        ps = sents
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
    all_refs.append(tmp)
    all_preds.append(pred_dict)
print(update_sp(all_preds, all_refs))
