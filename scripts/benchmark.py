import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import gzip
import json
import numpy
import random
from scipy.stats import ttest_ind
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain

with open('data/valerius_flaccus/golds.pkl', 'rb') as ifd:
    golds = pickle.load(ifd)
device = "cuda:0"

df = pd.read_pickle('data/valerius_flaccus/data.pkl')


def load_embs(input):
    embs = {}
    locs = []
    with gzip.open(input, 'rt') as ifd:
        for line in ifd:
            j = json.loads(line)
            loc = j["location"]
            locs.append(loc)
            bk, ch, v = loc.split('.')
            ch = int(ch)
            v = int(v)
            embs[bk] = embs.get(bk, {})
            embs[bk][ch] = embs[bk].get(ch, {})
            embs[bk][ch][v] = numpy.array(j["embedding"])
    return embs, locs

def calc_vf(input, output, golds, embs=None):
    if embs == None:
        embs = load_embs(input)

    ref_dists = []
    nonref_dists = []
    for src_loc, tgt in golds.items():
        src_bk, src_ch, src_v = src_loc.split('.')
        src_ch = int(src_ch)
        src_v = int(src_v)

        tgt_bk, tgt_ch, tgt_v = tgt.split('.')
        if tgt_bk == "aen_virg":
            tgt_bk = "aen"
        tgt_ch = int(tgt_ch)
        tgt_v = int(tgt_v)

        try:
            src_emb = embs[src_bk][src_ch][src_v]
        except:
            print(f"Src emb not found {src_bk,src_ch, src_v}")
        try:
            tgt_emb = embs[tgt_bk][tgt_ch][tgt_v]
        except:
            print(f"Tgt emb not found {tgt_bk, tgt_ch, tgt_v}")
            continue
        ref_dists.append(numpy.dot(src_emb, tgt_emb))
        poss = [v for k, v in embs[tgt_bk][tgt_ch].items() if k!= tgt_v]
        random.shuffle(poss)
        nonref_dists.append(numpy.dot(src_emb, poss[0]))


    result = ttest_ind(ref_dists, nonref_dists, equal_var=False)
    t_stat = result.statistic
    p_value = result.pvalue 
    conf_int = result.confidence_interval(confidence_level = 0.95)

    alpha = 0.05
    if p_value < alpha:
        print(f"Reject null hypothesis, the difference is significant")
    else:
        print(f"Cannot reject null hypothesis, the difference is not significant")

    with open(output, 'wt') as ofd:
        s_ref_dists = sum(ref_dists)
        s_nonref_dists = sum(nonref_dists)
        ofd.write(json.dumps({"reference_similarity": s_ref_dists,
                               "nonreference_similarity": s_nonref_dists,
                               "ratio": s_ref_dists / s_nonref_dists,
                               "diff": (s_ref_dists - s_nonref_dists)/len(ref_dists),
                               "num_refs": len(ref_dists),
                               "t_stat": t_stat,
                               "p_value": p_value,
                               "CI": conf_int
                            }) + "\n")

calc_vf('data/valerius_flaccus/embs.json.gz', 'data/valerius_flaccus/output.txt', golds)
