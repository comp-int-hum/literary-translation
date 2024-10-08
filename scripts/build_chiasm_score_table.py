# this file take in a bunch of jsonl files which are chiasm score files and, for each manuscript
# load them into an array

import pickle
import numpy as np
import pandas as pd
import argparse as ap
import os
import pytextable as pytex
from utils import Location
# pearsons correlation
from scipy.stats import pearsonr
# spearman's correlation




if __name__=="__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--bc", dest="bc", help="List of pickle files")
    parser.add_argument("--other", dest="other", help="List of pickle files")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    with open(args.bc, "rb") as ifd:
        bc = pickle.load(ifd)
    with open(args.other, "rb") as ifd:
        other = pickle.load(ifd)


    # df = pd.DataFrame({"file": bc.keys(), "data": bc.values()})
    # df['file'] = df['file'].apply(os.path.basename)
    # df['condition'] = df['file'].apply(lambda x: "-".join(x.split("_")[3:-1]))
    # df['language'] = df['file'].apply(lambda x: x.split("_")[-1].split('-chiasm')[0])
    # df['start_locs'] = df['data'].apply(lambda x: [Location(i) for i in x['locations']])
    # df['lengths'] = df['data'].apply(lambda x: x['lengths'])
    # df['num_verses'] = df['lengths'].apply(len)

    # # want to zip the lengths and start_locs together
    # # I want a column that is [(start_loc, length), ...]
    # df['start_loc_length'] = df.apply(lambda x: list(zip(x['start_locs'], x['lengths'])), axis=1)
    # print(df['start_loc_length'].head())
    # make the condition and language the index
    # df = df.set_index(['condition', 'language'])
    # then lookup, like human-translation Finnish
    # now we want to add in the other scores
    # we want to join on the condition and language
    # we want to add in the other scores
    df_2 = pd.DataFrame({"file": other.keys(), "data": other.values()})
    df_2['file'] = df_2['file'].apply(os.path.basename)
    df_2['condition'] = df_2['file'].apply(lambda x: "-".join(x.split("_")[3:-1]))
    df_2['language'] = df_2['file'].apply(lambda x: x.split("_")[-1].split('-chiasm')[0])
    df_2['start_locs'] = df_2['data'].apply(lambda x: [Location(i) for i in x['locations']])
    df_2['lengths'] = df_2['data'].apply(lambda x: x['lengths'])
    df_2['num_verses'] = df_2['lengths'].apply(len)
    df_2['start_loc_length'] = df_2.apply(lambda x: list(zip(x['start_locs'], x['lengths'])), axis=1)
    # now append the two dataframes
    # df = df.append(df_2)
    #HACK
    df = df_2
    # drop
    # want to find the originals, where condition = original
    original = df[df['condition'] == 'original']['lengths'].tolist()[0]
   
    # make a new lengths column "standardized" which is the start_loc_length column standardized by the original. only keep entries where location is in the original and the other
    # def standarize(row, original):
    #     # row is a list of tuples [(start_loc, length), ...]
    #     # original is a list of tuples [(start_loc, length), ...]
    #     # we want to find the intersection of the start_locs
    #     # return a new row that only keeps entries for which the start_loc is in the original
    #     verses = set([x[0] for x in original[0]])
    #     # for i, v in enumerate(row):
    #     #     print(v)
    #     #     if i ==5:
    #     #         break
    #     # exit()
    #     # print(len(verses))
    #     # print(iter(verses))
    #     return [i for i in row if i[0] in verses]
    
    # df['duplicates'] = df['start_locs'].apply(lambda x: len(set(x)))
    # df['standardized'] = df.apply(lambda x: standarize(x['start_loc_length'], original.values), axis=1)
    # df['new_num_verses'] = df['standardized'].apply(len)

    # sometimes the lengths are different, so we want to find the intersection of the lengths
    # print(df[['num_verses', 'condition', 'language']])
    # I want to make a new row 'corr' which is the correlation between the original the the 'lengths' value
    df['corr'] = df['lengths'].apply(lambda x: pearsonr(original, x)[0])
    # sort  by corr
    df = df.sort_values(by='corr', ascending=False)

    # print(df[['num_verses', 'corr', 'condition', 'language']])
    # round the corr column to 2 decimal places
    df['corr'] = df['corr'].apply(lambda x: round(x, 2))
    # the table we want to save is df[['condition', 'language', 'corr']]
    pytex.write(df[['condition', 'language', 'corr']].values,args.output,  header=['Condition', 'Language', 'Chiasm Correlation'], caption="Pearson's correlation coefficient between original and translation chiasms", label="tab:correlation")
