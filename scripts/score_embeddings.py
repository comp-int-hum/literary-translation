import random
import logging
import gzip
import json
import re
import argparse
import numpy
from utils import Location, NT


logger = logging.getLogger("score_embeddings")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", dest="gold", help="Input file")
    parser.add_argument("--embeddings", dest="embeddings", nargs="+", help="Embedding files")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--vote_threshold", dest="vote_threshold", default=0, type=int)
    parser.add_argument("--testament", dest="testament", default="none")
    parser.add_argument("--manuscript", dest="manuscript", default="none")
    parser.add_argument("--language", dest="language", required=True)
    parser.add_argument("--condition", dest="condition", required=True)
    parser.add_argument("--random_seed", dest="random_seed", default=None, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.random_seed:
        random.seed(args.random_seed)
    
    books = set()
    labels = set()
    golds = {}
    num_golds = 0
    with open(args.gold, "rt") as ifd:
        ifd.readline()
        for line in ifd:
            if "-" not in line:
                a, b, votes = line.strip().split()
                a = Location(a)
                b = Location(b)
                if (not ((a.testament() == args.testament and b.testament() == args.testament) or args.testament == "both")) or (a["book"] == b["book"]):
                    continue
                votes = int(votes)
                labels.add(a)
                labels.add(b)
                books.add(a["book"])
                books.add(b["book"])
                if a < b:
                    src, tgt = a, b
                else:
                    src, tgt = b, a
                key = (src, tgt)
                golds[src] = golds.get(src, {})
                if votes > args.vote_threshold:
                    golds[src][tgt] = votes
                    num_golds += 1

    embs = {}
    for fname in args.embeddings:
        with gzip.open(fname, "rt") as ifd:
            for line in ifd:
                j = json.loads(line)
                loc = Location(j["location"])
                bk = loc["book"]
                ch = loc["chapter"]
                v = loc["verse"]
                embs[bk] = embs.get(bk, {})
                embs[bk][ch] = embs[bk].get(ch, {})
                embs[bk][ch][v] = numpy.array(j["embedding"])

    ref_dists = []
    nonref_dists = []
    for src_loc, tgts in golds.items():
        src_bk = src_loc["book"]
        src_ch = src_loc["chapter"]
        src_v = src_loc["verse"]
        
        for tgt_loc, votes in tgts.items():
            tgt_bk = tgt_loc["book"]
            tgt_ch = tgt_loc["chapter"]
            tgt_v = tgt_loc["verse"]
            fail = False
            try:
                src_emb = embs[src_bk][src_ch][src_v]
            except:
                logger.info("Could not find location %s:%s:%s", src_bk, src_ch, src_v)
                fail = True
            try:
                tgt_emb = embs[tgt_bk][tgt_ch][tgt_v]
            except:
                fail = True
                logger.info("Could not find location %s:%s:%s", tgt_bk, tgt_ch, tgt_v)
            if fail:
                continue
            ref_dists.append(numpy.dot(src_emb, tgt_emb))
            poss = [v for k, v in embs[tgt_bk][tgt_ch].items() if k != tgt_v]
            random.shuffle(poss)
            nonref_dists.append(numpy.dot(src_emb, poss[0]))
        
    s_ref_dists = sum(ref_dists)
    s_nonref_dists = sum(nonref_dists)
    ratio = s_ref_dists / s_nonref_dists
    
    with gzip.open(args.output, "wt") as ofd:
        ofd.write(json.dumps({"reference_similarity" : s_ref_dists, "nonreference_similarity" : s_nonref_dists, "ratio" : ratio, "num_refs" : len(ref_dists), "file_name" : args.embeddings, "testament" : args.testament, "language" : args.language, "condition" : args.condition, "manuscript" : args.manuscript}) + "\n")
