import random
import logging
import gzip
import json
import re
import argparse
import numpy

logger = logging.getLogger("score_embeddings")

book2id = {k : i for i, k in enumerate("""GEN
EXO
LEV
NUM
DEU
JOS
JDG
RUT
1SA
2SA
1KI
2KI
1CH
2CH
EZR
NEH
EST
JOB
PSA
PRO
ECC
SOS
ISA
JER
LAM
EZE
DAN
HOS
JOE
AMO
OBA
JON
MIC
NAH
HAB
ZEP
HAG
ZEC
MAL
MAT
MAR
LUK
JOH
ACT
ROM
1CO
2CO
GAL
EPH
PHP
COL
1TH
2TH
1TI
2TI
TIT
PHM
HEB
JAM
1PE
2PE
1JO
2JO
3JO
JDE
REV""".split())}
id2book = {i : b for b, i in book2id.items()}

mapping = {
    "1KGS" : "1KI",
    "2KGS" : "2KI",    
    "JAS" : "JAM",
    "JUDG" : "JDG",
    "PS" : "PSA",
    "JUDE" : "JDE",
    "SONG" : "SOS",
    "PHIL" : "PHP",
    "PHLM" : "PHM",    
    "SON" : "SOS",
    "PHI" : "PHP",
    "JUD" : "JDE"
}

def to_label(s):
    book, chapter, verse = re.sub(r"^b\.", "", s).upper().split(".")
    book3 = mapping.get(book, book[:3])
    if book3 not in book2id:
        raise Exception(s)
    return (book2id[book3], chapter, verse)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", dest="gold", help="Input file")
    parser.add_argument("--embeddings", dest="embeddings", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--vote_threshold", dest="vote_threshold", default=50, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    books = set()
    labels = set()
    golds = {}
    with open(args.gold, "rt") as ifd:
        ifd.readline()
        for line in ifd:
            if "-" not in line:
                a, b, votes = line.strip().split()
                a = to_label(a)
                b = to_label(b)
                votes = int(votes)
                labels.add(a)
                labels.add(b)
                books.add(a[0])
                books.add(b[0])
                if a < b:
                    src, tgt = a, b
                else:
                    src, tgt = b, a
                key = (src, tgt)
                golds[src] = golds.get(src, {})
                golds[src][tgt] = golds[src].get(tgt, 0) + votes

    golds = {s : {k : v for k, v in tgts.items() if v >= args.vote_threshold} for s, tgts in golds.items()}
    golds = {s : tgts for s, tgts in golds.items() if len(tgts) > 0}

    embs = {}
    with gzip.open(args.embeddings, "rt") as ifd:
        for line in ifd:
            j = json.loads(line)
            bk, ch, v = to_label(j["id"])
            embs[bk] = embs.get(bk, {})
            embs[bk][ch] = embs[bk].get(ch, {})
            embs[bk][ch][v] = numpy.array(j["embedding"])

    ref_dists = []
    nonref_dists = []
    for (src_bk, src_ch, src_v), tgts in golds.items():
        for (tgt_bk, tgt_ch, tgt_v), votes in tgts.items():
            try:
                src_emb = embs[src_bk][src_ch][src_v]
                tgt_emb = embs[tgt_bk][tgt_ch][tgt_v]
            except:
                continue
            ref_dists.append(numpy.dot(src_emb, tgt_emb))
            poss = [v for k, v in embs[tgt_bk][tgt_ch].items() if k != tgt_v]
            random.shuffle(poss)
            nonref_dists.append(numpy.dot(src_emb, poss[0]))


    with open(args.output, "wt") as ofd:
        ofd.write(json.dumps({"reference_distance" : sum(ref_dists), "nonreference_distance" : sum(nonref_dists)}) + "\n")
