import pickle
import sys
import torch
from utils import load_hotpotqa
from collections import Counter, defaultdict
from utils import normalize_answer

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
        for e in gold_sp_pred:
            if e not in cur_sp_pred:
                fn += len(gold_sp_pred[e])
            else:
                for v in gold_sp_pred[e]:
                    if v not in cur_sp_pred[e]:
                        fn += len(gold_sp_pred[e])
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

def check_ans_in_span(ans, span):
    if isinstance(span, list):
        span = ' '.join(span)
    norm_p = normalize_answer(span)
    return normalize_answer(ans) in norm_p

def check_para_correct(pred, ref):
    wrong = 0
    for k in pred:
        if k not in ref:
            wrong += 1
    return wrong

def pretty_print(q, p, a, gold, pred):
    x, y = list(pred.keys())
    goldx, goldy = list(gold.keys())
    print("q:", q)
    print("gold p:", f"{goldx}: {gold[goldx]} | {goldy}: {gold[goldy]}")
    print("pred p:", f"{x}: {pred[x]} | {y}: {pred[y]}")
    paras = set([x, y, goldx, goldy])
    for i in sorted(paras):
        print(f"p{i}:", p[i])
    print("pred a:", a[0], "   |   in span:", check_ans_in_span(a[0], p[x]+p[y]))
    print("gold a:", a[1], "   |   in span:", check_ans_in_span(a[1], p[0]+p[1]))
    print("="*100)

def get_answers(ans):
    pred_ans = [x[0] for x in ans] 
    gold_ans = [x[1] for x in ans] 
    pred_ans = [a for aa in pred_ans for a in aa]
    gold_ans = [a for aa in gold_ans for a in aa]
    return pred_ans, gold_ans

def intersection_helper(target, a, b, c, d):
    aa = len(target.intersection(a)) / len(target)
    bb = len(target.intersection(b)) / len(target)
    cc = len(target.intersection(c)) / len(target)
    dd = len(target.intersection(d)) / len(target)
    return aa, bb, cc, dd

def get_confusion_matrix(paras, pred_ans, gold_ans, gold_supps, para_texts):
    print(Counter([p for ps in paras for p in ps]))
    pcac = set()
    pcai = set()
    piac = set()
    piai = set()
    i = 0
    pcorrect_supps = set()
    for para, pred_ans, gold_ans, ss, text in zip(paras, pred_ans, gold_ans, gold_supps, para_texts):
        x, y = para
        wrong_para = check_para_correct(para, ss)
        if wrong_para == 1:
            pcorrect_supps.add(i)
        if wrong_para == 0 and pred_ans == gold_ans:
            pcac.add(i)
        elif wrong_para == 0 and pred_ans != gold_ans:
            pcai.add(i)
        elif wrong_para > 0 and pred_ans == gold_ans:
            piac.add(i)
        else:
            piai.add(i)
        i += 1
    print("partial correct:", len(pcorrect_supps)/len(paras), intersection_helper(pcorrect_supps, pcac, pcai, piac, piai))
    return pcac, pcai, piac, piai

def get_changes(d1, d2):
    ks = d1.keys()
    for k2 in ks:
        for k1 in ks:
            overlap = d1[k1].intersection(d2[k2])
            #if k2 == "piac" or k2 == "piai" or k2 == "pcai":
            if k1 == "piac" and k2 == "pcac":
                for i in overlap:
                    #pretty_print(questions[i], paragraphs[i], (pred_ans1[i], gold_ans[i]), gold_paras[i], pred_paras1[i])
                    pretty_print(questions[i], paragraphs[i], (pred_ans2[i], gold_ans[i]), gold_paras[i], pred_paras2[i])
            overlap = len(overlap)
            print(f"{k1} -> {k2}: {overlap}")

data = load_hotpotqa()["validation"]
paragraphs = data["context"]
paragraphs = [x["sentences"] for x in paragraphs]
questions = data["question"]
fname1 = sys.argv[1]
fname2 = sys.argv[2]
pred_paras1, gold_paras, ans1 = torch.load(fname1)
pred_paras2, _, ans2 = torch.load(fname2)
pred_ans1, gold_ans = get_answers(ans1)
pred_ans2, _ = get_answers(ans2)
pcac1, pcai1, piac1, piai1 = get_confusion_matrix(pred_paras1, pred_ans1, gold_ans, gold_paras, paragraphs)
pcac2, pcai2, piac2, piai2 = get_confusion_matrix(pred_paras2, pred_ans2, gold_ans, gold_paras, paragraphs)
print("para acc:", (len(pcac1)+len(pcai1))/len(pred_paras1), (len(pcac2)+len(pcai2))/len(pred_paras2))
print("ans em:", (len(pcac1)+len(piac1))/len(pred_paras1), (len(pcac2)+len(piac2))/len(pred_paras2))
print("pcac:", len(pcac1)/len(pred_paras1), len(pcac2)/len(pred_paras1))
print("pcai:", len(pcai1)/len(pred_paras1), len(pcai2)/len(pred_paras1))
print("piac:", len(piac1)/len(pred_paras1), len(piac2)/len(pred_paras1))
print("piai:", len(piai1)/len(pred_paras1), len(piai2)/len(pred_paras1))

get_changes({"pcac": pcac1, "pcai": pcai1, "piac": piac1, "piai": piai1},
        {"pcac": pcac2, "pcai": pcai2, "piac": piac2, "piai": piai2})

#new1 = [i for i in range(len(data["type"])) if data["type"][i] == "bridge"]
#new2 = [i for i in range(len(data["type"])) if data["type"][i] == "comparison"]
#new_para11 = [pred_paras1[i] for i in new1]
#new_para12 = [pred_paras1[i] for i in new2]
#new_para21 = [pred_paras2[i] for i in new1]
#new_para22 = [pred_paras2[i] for i in new2]
#gold_paras1 = [gold_paras[i] for i in new1]
#gold_paras2 = [gold_paras[i] for i in new2]
#print(update_sp(new_para11, gold_paras1))
#print(update_sp(new_para12, gold_paras2))
#print(update_sp(new_para21, gold_paras1))
#print(update_sp(new_para22, gold_paras2))
print(update_sp(pred_paras1, gold_paras))
print(update_sp(pred_paras2, gold_paras))
