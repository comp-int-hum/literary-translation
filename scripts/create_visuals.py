import sys
import re
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
import matplotlib as mpl
from utils import Bible, Location
import json
import argparse
import gzip
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def get_color(value, cmap):
    # return the color as a hue, saturation, value tuple
    return cmap(value)

# get_color(0.5, cm)

ENG = Bible('../narrative-analysis/work/english.jsonl')
print(ENG[Location({"book": "GEN", "chapter": 1, "verse": 1})])

# HACK:
# given a start Location and a length n, we want to retrieve the passage
# try to increment verses, if we reach the end of the chapter, increment chapter
# if we reach the end of the book, increment book
def retrieve_passage(start, n):
    # start is a Location object
    # n is an integer
    # returns a string
    current = start
    passage = [ENG[current]]
    for i in range(n-1):
        current = Location({"book": current['book'], "chapter": current['chapter'], "verse": current['verse'] + 1})
        if current not in ENG:
            current = Location({"book": current['book'], "chapter": current['chapter'] + 1, "verse": 1})
        passage.append(ENG[current])
    return passage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", dest="score", help="Work directory")
    parser.add_argument("--embedding", dest="embeddings", help="Embeddings file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--thres", dest="thres", type=float, help="Threshold for significance")

    args = parser.parse_args()

    with gzip.open(args.score, "rb") as ifd:
        data = [json.loads(line) for line in ifd]
    
    scores = pd.DataFrame.from_records(data)
    # convert start and end columns to Location objects
    scores["start"] = scores["start"].apply(lambda x: Location(x))
    scores["end"] = scores["end"].apply(lambda x: Location(x))
    print(scores.head())

    locations = []
    embeds = []
    with gzip.open(args.embedding, "rb") as ifd:
        for line in ifd:
            obj = json.loads(line)
            locations.append(Location(obj['location']))
            embeds.append(obj['embedding'])
        locations = np.asarray(locations)
        embeds = np.asarray(embeds)
    
    cos_sim = cosine_similarity(embeds,embeds)
    print(cos_sim.shape)

    # now we want to find the top chiasms (meaning the lowest p-values)
    # group by i and keep the n with the smallest p-value
    scores = scores.groupby("i").apply(lambda x: x.loc[x["p"].idxmin()]).reset_index(drop=True)
    # filter by threshold
    scores = scores[scores["p"] < args.thres]
    # sort by p-value
    scores = scores.sort_values(by="p")
    # let's take the top 5 and use ENG to retrieve the passage
    scores = scores.head(5)
    scores['text'] = scores.apply(lambda x: retrieve_passage(x['start'], x['n']), axis=1)
    print(scores.iloc[0]['text'])
    # now we have them as sentences, scores['text'] is the equivalent of sentences below
    cm = plt.cm.get_cmap('Set2')
    for _, row in scores.iterrows():
        sentences = row['text']
        # the similarity matrix is the subset of cos_sim at the indices of the sentences (i to i+n-1)
        similarity_matrix = cos_sim[row['i']:row['i']+row['n'], row['i']:row['i']+row['n']]
        pairs = []
        for i in range(len(sentences)//2):
            pairs.append((i, len(sentences) - i - 1))
        if len(sentences) % 2:
            pairs.append((len(sentences)//2, len(sentences)//2))


        plt.figure()

        fs = 12
        for i, pair in enumerate(pairs):
            # print(i, pair)
            a = plt.text(i/10, -pair[0]/fs+1, f'{sentences[pair[0]]}',fontsize=fs)
            b = plt.text(i/10, -pair[1]/fs+1, f'{sentences[pair[1]]}',fontsize=fs)
            # make axes invisible
            plt.axis('off')
            # font family times new roman
            a.set_bbox(dict(facecolor=get_color(similarity_matrix[pair[0],pair[1]], cm), alpha=0.5, edgecolor=get_color(similarity_matrix[pair[0],pair[1]], cm)))
            b.set_bbox(dict(facecolor=get_color(similarity_matrix[pair[1],pair[0]], cm), alpha=0.5, edgecolor=get_color(similarity_matrix[pair[1],pair[0]], cm)))
            # save the figure

        # when we save a Location to a string, it becomes a dict. We want the keys of the dict joined with '.'
        loc = [str(v) for _,v in row["start"].items()]
        name = '.'.join(loc)
        plt.savefig(f'{args.output}/{name}_{row["n"]}.png', bbox_inches='tight')

        # add colorbar of viridis

    



# len(sentences)//2, len(sentences) % 2
# # so okay we have an odd chiasm of 2-levels.

# # we do this for the top chiasms, 

