# this file takes in a bunch of chiasm score files, separates them into manuscripts, then condenses each one to a dataframe, then makes an array
# that is 

# import pytextable as pytex
import pandas as pd
import numpy as np
import os
import json
import argparse
import glob
import pickle
import gzip
from tqdm import tqdm
from utils import Bible, Location

def load(file):
    with gzip.open(file, "rb") as ifd:
            data = [json.loads(line) for line in ifd]
    return data


def create_array(score_files, threshold):
    data = {}
    for score_file in tqdm(score_files[:2]):
        scores = load(score_file)
        df = pd.DataFrame.from_records(scores)
        df["chiasm_length"] = df["p"].apply(lambda x: 1 if x < threshold else 0)
        df["chiasm_length"] *= df["n"]
        # group by i and keep the n with the smallest p-value
        df = df.groupby("i").apply(lambda x: x.loc[x["p"].idxmin()]).reset_index(drop=True)
        locations = df['start'].apply(lambda x: Location(x))
        data[score_file] = {'locations': locations, 'lengths': df['chiasm_length'].values.reshape(1,-1)[0]}
    
    # filepath: (location, chiasm_length_at_location_i)
    return data


def find_disputed(array, n):
    # find the columns that have the largest variation
    # we do this by finding the standard deviation of each column
    # then we sort the columns by the standard deviation
    # then we take the top n columns
    stds = np.std(array, axis=0)
    high = np.argsort(stds)[::-1][:n]
    return high


def get_passages_for_inspection(high, array, files, trs, locations):
    table_data = []
    for i, starting_pos in enumerate(high):
        # grabbing the column of the array
        column = array[:,starting_pos]
        # j is the index of the file, k is the chiasm size
        for translation_idx, chiasm_length in enumerate(column):
            
            # j is the index of the file, k is the chiasm size
            file_name = files[translation_idx]
            file_name = "_".join(file_name.split("/"))
            bible = trs[file_name.split('-chiasm')[0]]
            verses = locations[starting_pos: starting_pos+chiasm_length+1]
            print(verses)
            text = ""
            for verse in verses:
                text += verse + " "
                text += bible[verse] + "\n"
            # print(files.iloc[translation_idx])
            # # print(trs[file_name.split('-chiasm')[0]].data[:5])
            # print(trs[file_name.split('-chiasm')[0]].get_passage(start=i, end=i+chiasm_length, sep=''))
            # text = trs[file_name.split('-chiasm')[0]].get_passage(start=i, end=i+chiasm_length, sep='')
        
            table_data.append([i, file_name, starting_pos, chiasm_length, text])
        
    # make a dataframe
    return pd.DataFrame(table_data, columns=["index", "file", "starting_pos", "chiasm_length", "text"])


# we end up with something like this, once we turn data into a dataframe
# 0	work_old_WLC-Hebrew_exclude_human_Japanese-chiasm	[18, 6, 8, 6, 14, 20, 18, 16, 19, 18, 11, 9, 7..

# then we just want to find the columns that have the largest variation
# we do this by finding the standard deviation of each column
# then we sort the columns by the standard deviation
# then we take the top n columns
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", dest="scores", nargs="+", help="Scores files")
    parser.add_argument("--translations", dest="translations", nargs="+", help="Translations files")
    parser.add_argument("--array", dest="array", help="Output array file")
    parser.add_argument("--passages", dest="passages", help="Output passages file")
    parser.add_argument("--p", dest="p", type=float, help="Threshold for significance")
    parser.add_argument("--n", dest="n", type=int, help="Number of passages to output")


    args = parser.parse_args()

    TRANSLATIONS = {}
    for file in args.translations:
        translation = Bible(file)
        TRANSLATIONS["_".join(file.split("/")).split(".json.gz")[0]] = translation
    print(TRANSLATIONS.keys())
    data_dict = create_array(args.scores, args.p)
    # print(data_dict)
    # save dict to a the array output
    # with open(args.array, "w") as ofd:
        # json.dump(data_dict, ofd)
    # make an array of the second value in the tuple
    # array = np.array([x[1] for x in data_dict.values()])
    array = np.asarray([v['lengths'] for _,v in data_dict.items()])
    locations = np.asarray([v['locations'] for _,v in data_dict.items()])
    # locations = np.array([x[0] for x in data_dict.values()])
    files = [k for k,_ in data_dict.items()]
    print(files)
    high = find_disputed(array, args.n)

    passages = get_passages_for_inspection(high, array, files, TRANSLATIONS, locations)
    passages.to_csv(args.passages, index=False)

