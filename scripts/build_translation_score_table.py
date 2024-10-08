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

def build_translation_score_table(files, output, manuscript):
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
    langs = set([f.split("_")[-1].split(".")[0] for f in files])
    # remove "hebrew" from langs
    try:
        langs.remove("Hebrew")
    except:
        pass
    try:
        langs.remove("Greek")
    except:
        pass
    langs.remove("Japanese")
  
    # header will be Condition, English, Finnish, Swedish, Japanese, Marathi, Turkish, 
    for condition in ["unconstrained", "exclude_references", "exclude_human", "exclude_both"]:
        row = [condition]
        for lang in langs:
            # we load the unconstrained, then exclude reference, then exclude human, then exclude both
            for file in files:
                if lang in file and condition in file:
                    value = pd.read_json(file)["system_score"].values[0]
                    # round it 
                    value = round(value, 2)
                    row.append(value)
    # we want to report values as a delta from the unconstrained value for each language

        table_data.append(row)

    # table_data[i, j] would be the BLEU score for the ith condition and jth language
    # the unconstrained condition is i=0, 
    arr = np.asarray(table_data)
    print(["condition"] + list(langs))
    print(arr)
    # for columns 1 to the end, we want to subtract the value of the first row in the same column
    # for c in range(1, arr.shape[1]):
    #     arr[1:, c] = arr[1:, c] - arr[0, c]:w

    # where dtype is float in the array, round it
    

 
    pytex.write(arr, output, header=["Condition"]+list(langs), caption=f"COMET scores for translations from {manuscript} ", label="tab:translation_scores")


if __name__ == "__main__":
    import argparse
    import glob
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", dest="inputs", nargs="+", help="Input files")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--manuscript", dest="manuscript", help="Manuscript name")
    args = parser.parse_args()

    # first filter the inputs by if they have the manuscript name in them
    files = [f for f in args.inputs if args.manuscript in f]
    build_translation_score_table(files, args.output, args.manuscript)

    


