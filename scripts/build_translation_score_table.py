# we take in a list of files which are jsonl files. 
# sort by manuscript
# we want to sort these files by language
# then by condition, then we want to order 
#For each one we want to grab the "bleu_score" value

# MANUSCRIPTS = set([f.split("/")[-3] for f in all_files])
# CONDITIONS = set([f.split("/")[-2] for f in all_files])
# LANGUAGES = set([f.split("/")[-1].split("-")[0] for f in all_files])
import pytextable as pytex
import pandas as pd
import numpy as np

def build_translation_score_table(files, output):
    """
    This function takes in a list of bleu scores and outputs a .tex table of all the translation scores.
    
    Parameters:
    assume that these files are all from the same manuscript.
    we want to make a table that looks like

    Condition | Language 1 | Language 2 | Language 3
    unconstrained | BLEU | BLEU | BLEU
    exclude | BLEU | BLEU | BLEU
    ...
    """
    table_data = []
    # header will be Condition, English, Finnish, Swedish, Japanese, Marathi, Turkish, 
    langs = ["English", "Finnish", "Turkish", "Swedish", "Marathi"]
    for condition in ["unconstrained", "exclude_references", "exclude_human", "exclude_both"]:
        row = [condition.replace("_", " ")]
        for lang in langs:
            # we load the unconstrained, then exclude reference, then exclude human, then exclude both
            for file in files:
                if lang in file and condition in file:
                    value = pd.read_json(file)["bleu"].values[0]
                    value = round(value, 2)
                    row.append(value)

        table_data.append(row)
    arr = np.asarray(table_data)


    pytex.write(arr, output, header=["Condition"]+list(langs), caption="BLEU scores for translations", label="tab:translation_scores")


if __name__ == "__main__":
    import argparse
    import glob
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", dest="inputs", nargs="+", help="Input files")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--manuscript", dest="manuscript", help="Manuscript name")
    parser.add_argument("--testament", dest="testament")
    args = parser.parse_args()

    # first filter the inputs by if they have the manuscript name in them
    files = [f for f in args.inputs if args.manuscript in f and args.testament in f]
    build_translation_score_table(files, args.output)

    



