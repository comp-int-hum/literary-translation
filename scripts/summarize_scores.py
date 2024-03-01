import argparse
import json
import gzip
import re
import os.path
import pandas

captions = {
    "none" : "no vocabulary masking",
    "target" : "masked vocabulary from human translation",
    "reference" : "masked vocabulary from translations of referent sentences",
    "both" : "masked vocabulary from both human and referent translations"
}
# human source, target
# no vocab masking delta w.r.t. human source, w.r.t. human target
# masking variants w.r.t. no vocab masking

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", dest="inputs", nargs="+", help="Input files")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    vals = {}
    rvals = {}
    svals = {}
    cols = set()
    rows = set()
    excls = set()
    for fname in args.inputs:
        with open(fname, "rt") as ifd:
            j = json.loads(ifd.read())
            src = j["source"]
            rows.add(src)
            tgt = j["target"]
            cols.add(tgt)
            exc = j["exclude"]
            excls.add(exc)
            ratio = j["ratio"]
            vals[src] = vals.get(src, {})
            vals[src][tgt] = vals[src].get(tgt, {})
            vals[src][tgt][exc] = ratio

            key = (src, tgt)
            rvals[key] = rvals.get(key, {})
            rvals[key][exc] = ratio

    with open(args.output, "wt") as ofd:
        human_scores = dict(sum([[(src, {"Language" : src, "Score" : exs["none"]}) for tgt, exs in tgts.items() if tgt == src] for src, tgts in vals.items()], []))
        ofd.write(pandas.DataFrame(human_scores.values()).to_latex(na_rep="", caption="Inter-textuality scores of human translations", float_format="%.2f", index=False) + "\n")
        
        for exc in excls:
            data = []
            for src in sorted(rows):
                row = {"Source" : src}
                for tgt in sorted(cols):
                    if tgt != src:
                        row[tgt] = (vals[src][tgt][exc] - human_scores[src]["Score"]) / (1.0 - human_scores[src]["Score"])
                data.append(row)
            df = pandas.DataFrame(data)
            ofd.write(df.to_latex(na_rep="", caption="Percent-delta from source language human translation with {}".format(captions[exc]), float_format="%.2f", index=False) + "\n")
            data = []
            for src in sorted(rows):
                row = {"Source" : src}
                for tgt in sorted(cols):
                    if tgt != src:
                        row[tgt] = (vals[src][tgt][exc] - human_scores[tgt]["Score"]) / (1.0 - human_scores[tgt]["Score"])
                data.append(row)
            df = pandas.DataFrame(data)
            ofd.write(df.to_latex(na_rep="", caption="Percent-delta from target language human translation with {}".format(captions[exc]), float_format="%.2f", index=False) + "\n")

        
            
        #rdata = [dict([("pair", "{}-to-{}".format(t, s))] + [(k, v) for k, v in vs.items()]) for (t, s), vs in rvals.items() if t != s]
        #df = pandas.DataFrame(rdata, columns=["pair", "none", "reference", "target", "both"])
        #ofd.write(df.to_latex(na_rep="", caption="By masking", float_format="%.2f", index=False, columns=["pair", "none", "reference", "target", "both"]) + "\n")

