import argparse
import json
import gzip
import re
import os.path
import pandas

captions = {
    "original" : "untranslated",
    "unconstrained" : "no vocabulary masking",
    "exclude_human" : "masked vocabulary from human translation",
    "exclude_references" : "masked vocabulary from translations of referent sentences",
    "exclude_both" : "masked vocabulary from both human and referent translations"
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", dest="inputs", nargs="+", help="Input files")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    originals = {}
    vals = {}
    for fname in args.inputs:
        with gzip.open(fname, "rt") as ifd:
            j = json.loads(ifd.read())
            testament = j["testament"]
            language = j["language"]
            condition = j["condition"]
            ratio = j["ratio"]
            if condition == "original":
                originals[testament] = ratio
            else:
                key = (testament, condition)
                vals[testament] = vals.get(testament, {})
                vals[testament][condition] = vals[testament].get(condition, {})
                vals[testament][condition][language] = ratio
                
    rvals = []
    for testament, conditions in vals.items():
        for condition, languages in conditions.items():
            d = {k : v for k, v in languages.items()}
            d["Testament"] = testament
            d["Condition"] = condition
            rvals.append(d)
    with open(args.output, "wt") as ofd:
        df = pandas.DataFrame(rvals)
        df["Condition"] = df["Condition"].astype(pandas.CategoricalDtype(["human_translation", "unconstrained", "exclude_human", "exclude_references", "exclude_both"], ordered=True))
        df["Testament"] = df["Testament"].astype(pandas.CategoricalDtype(["old", "new"], ordered=True))
        df = df[["Testament", "Condition"] + [x for x in df.columns if x not in ["Condition", "Testament"]]].sort_values(["Testament", "Condition"])
        ofd.write(df.to_latex(na_rep="", caption="Inter-textuality scores of translations (original old and new testaments have scores of {:.2f} and {:.2f}, repectively)".format(originals["old"], originals["new"]), float_format="%.2f", index=False) + "\n")
