from collections import defaultdict
from itertools import combinations
import random
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import load_hotpotqa, normalize_answer

data = load_hotpotqa()["validation"]
num_distractor = int(sys.argv[1])
acc = 0
for context, q, supp in zip(data["context"], data["question"], data['supporting_facts']):
    ts = context["title"]
    if len(ts) == 2: acc += 1; continue
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
    if num_distractor == 1:
        total = list(range(len(ts)))
        total.remove(l[0])
        total.remove(l[1])
        distractor = random.choice(total)
        ps = [sents[x], sents[y], sents[distractor]]
        l = [0, 1]
    else:
        ps = sents
    ratios = []
    ps = [' '.join(p) for p in ps]
    #ps = [set(normalize_answer(p).split()) for p in ps]
    #q = set(normalize_answer(q).split())
    #for p in ps:
    #    ratio = len(p.intersection(q)) / len(q)
    for p in ps:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([p, q])
        dense = vectors.todense()
        ratio = dense[0] * dense[1].T
        ratio = ratio[0, 0]

        ratios.append(ratio)
    top2 = np.argsort(ratios)[-2:]
    if top2[0] in l and top2[1] in l:
        acc += 1
print(acc/len(data["question"]))
