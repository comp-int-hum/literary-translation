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
    human_translations = {}
    vals = {}
    for fname in args.inputs:
        #print(fname)
        with gzip.open(fname, "rt") as ifd:
            j = json.loads(ifd.read())
            testament = j["testament"]
            language = j["language"]
            condition = j["condition"]
            manuscript = j["manuscript"]
            ratio = j["ratio"]
            #print(j["num_refs"])
            #print(condition)
            if condition == "original":
                originals[testament] = originals.get(testament, {})
                originals[testament][manuscript] = ratio
            else:
                key = (testament, condition)
                vals[testament] = vals.get(testament, {})
                vals[testament][manuscript] = vals[testament].get(manuscript, {})
                vals[testament][manuscript][condition] = vals[testament][manuscript].get(condition, {})
                vals[testament][manuscript][condition][language] = ratio
    #print(originals)
    #sys.exit()
    rvals = []
    for testament, manuscripts in vals.items():
        for manuscript, conditions in manuscripts.items():
            for condition, languages in conditions.items():
                #print(testament, manuscript, condition)
                d = {k : -(v - vals[testament]["BC"]["human_translation"][k]) / (1.0 - vals[testament]["BC"]["human_translation"][k]) for k, v in languages.items()}
                d["Testament"] = testament
                d["Condition"] = condition
                #print(manuscript)
                d["Manuscript"] = manuscript
                rvals.append(d)

    svals = []
    for testament, manuscripts in vals.items():
        for manuscript, conditions in manuscripts.items():
            for condition, languages in conditions.items():
                orig_score = originals[testament].get(manuscript, sum(originals[testament].values()) / len(originals[testament]))
                d = {k : -(v - orig_score) / (1.0 - orig_score) for k, v in languages.items()}
                d["Testament"] = testament
                d["Condition"] = condition
                d["Manuscript"] = manuscript                
                svals.append(d)

    ovals = []
    for testament, manuscripts in vals.items():
        for manuscript, conditions in manuscripts.items():
            for condition, languages in conditions.items():
                d = {k : v for k, v in languages.items()}
                d["Testament"] = testament
                d["Condition"] = condition
                d["Manuscript"] = manuscript
                ovals.append(d)
    def float_format(val):
        if val == 0:
            retval = 0.0
        else:
            retval = val * 100
        return "{:.0f}".format(retval) #" " * (4 - len(retval)) + retval
            
    with open(args.output, "wt") as ofd:

        pre = ["Testament", "Condition", "Manuscript"]
        post = []
        
        df = pandas.DataFrame(svals)
        df["Condition"] = df["Condition"].astype(pandas.CategoricalDtype(["human_translation", "unconstrained", "exclude_human", "exclude_references", "exclude_both"], ordered=True))
        df["Testament"] = df["Testament"].astype(pandas.CategoricalDtype(["old", "new"], ordered=True))
        df = df[pre + [x for x in df.columns if x not in pre + post] + post].sort_values(["Testament", "Manuscript", "Condition"])
        ofd.write(df.to_latex(na_rep="", caption="Percent-change in inter-textuality score of translation compared to original text", float_format=float_format, index=False).replace("_", "\\_") + "\n")


        df = pandas.DataFrame(rvals)
        df["Condition"] = df["Condition"].astype(pandas.CategoricalDtype(["human_translation", "unconstrained", "exclude_human", "exclude_references", "exclude_both"], ordered=True))
        df["Testament"] = df["Testament"].astype(pandas.CategoricalDtype(["old", "new"], ordered=True))
        df = df[pre + [x for x in df.columns if x not in pre + post] + post].sort_values(["Testament", "Manuscript", "Condition"])
        df = df[df["Condition"] != "human_translation"]
        ofd.write(df.to_latex(na_rep="", caption="Percent-change in inter-textuality score of machine translation compared to human reference", float_format=float_format, index=False).replace("_", "\\_") + "\n")


        df = pandas.DataFrame(ovals)
        df["Condition"] = df["Condition"].astype(pandas.CategoricalDtype(["human_translation", "unconstrained", "exclude_human", "exclude_references", "exclude_both"], ordered=True))
        df["Testament"] = df["Testament"].astype(pandas.CategoricalDtype(["old", "new"], ordered=True))
        df = df[pre + [x for x in df.columns if x not in pre + post] + post].sort_values(["Testament", "Manuscript", "Condition"])
        ofd.write(df.to_latex(na_rep="", caption="Inter-textuality scores of translations", float_format="%.3f", index=False).replace("_", "\\_") + "\n")
        

        pvals = []
        for testament, manuscripts in originals.items():
            for manuscript, score in manuscripts.items():
                pvals.append({"Testament" : testament, "Manuscript" : manuscript, "Score" : score})
        df = pandas.DataFrame(pvals)
        ofd.write(df.to_latex(na_rep="", caption="Inter-textuality scores of originals", float_format="%.3f", index=False).replace("_", "\\_") + "\n")
