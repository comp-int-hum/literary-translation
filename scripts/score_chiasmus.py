import argparse
import numpy as np
import pandas as pd
import gzip
import json
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from utils import Location
import multiprocessing as mp
import os
import glob

def get_chiasm_score(cos_sim, i, n):
    # the basic chiasm score is the sum of the reversed diagonal elements of the cosine similarity matrix
    # then we add a penalty for high similarity scores between different levels. 
    chiasm = cos_sim[i:i+n, i:i+n]
    # now reverse the diagonal
    chiasm = np.fliplr(chiasm)
    score = chiasm.trace()
    # if it's odd, subtract the middle value -- it's a self-similarity score, always 1
    if n % 2 == 1:
        score -= chiasm[n//2, n//2]
    # should normalize the score to the number of lines in the chiasm. 
    # if the chiasm is even, we divide by n, if it is odd, we divide by n-1, this is to avoid penalizing odd chiasmi
    # this is the average of lines that should be similar
    if n%2 == 0:
        div = n
    else:
        div = n-1
    
    
    neg_score = np.sum(chiasm[0, 1:-1]) + np.sum(chiasm[-1, 1:-1])
    # need to normalize this to n
    score = score/div - neg_score/div

  
    return score

  # adding a penalty for high similarity scores between different levels
    # for j in range(1,n):
    #     if cos_sim[i+j, i+n-1-j] > sim_threshold:
    #         score -= cos_sim[i+j, i+n-1-j]

def score_file(emb_file):
        n_list = N_LIST
        locations = []
        embeds = []
        with gzip.open(emb_file, "rb") as ifd:
            for line in ifd:
                obj = json.loads(line)
                locations.append(Location(obj['location']))
                embeds.append(obj['embedding'])
            locations = np.asarray(locations)
            embeds = np.asarray(embeds)

        cos_sim = cosine_similarity(embeds,embeds)
        np.random.seed(42)
        random_starts = np.random.randint(0, cos_sim.shape[0]-max(n_list)+1, 1000)

        p_values = {}
        candidate_scores = {}
        for n in n_list:
            p_values[n] = []
            candidate_scores[n] = []
            scores = []
            for i in random_starts:
                scores.append(get_chiasm_score(cos_sim, i, n, 0.95))
            avg = 0
            avg_p = 0
            for i in tqdm(range(0, cos_sim.shape[0]-n+1)):
                # get the score of that chiasmus of size n
                candidate= get_chiasm_score(cos_sim, i, n, 0.95)  
                candidate_scores[n].append(candidate)
                avg += candidate
                # calculate p-value
                # adding 1 to the numerator and denominator to avoid division by zero (laplace smoothing)
                p = (sum([1 for s in scores if s >= candidate]) + 1) / (len(scores)+1)
                avg_p += p
                p_values[n].append(p)
                # p_values is a dictionary with keys as n-values and values as lists of p-values for each starting position
                # e.g. p_values[3][0] is the p-value for the chiasmus of size 3 starting at the first position (p_value[n][i])
        # I think we want to save all of this information, write p_values to a jsonl file
        # fields for starting position i, n-value, p-value
        # the name of the output file will the same as the embeddings, just rather than "**/*-embedded.json.gz"
        # it will be "**/*-chiasm.jsonl"
        # candidate_scores is a dictionary with keys as n-values and values as lists of candidate scores for each starting position
        with gzip.open(f"{emb_file.replace('-embedded.json.gz', '-chiasm.json.gz')}-{SUFFIX}", "wt") as ofd:
            for n, p_list in p_values.items():
                for i, p in enumerate(p_list):
                    ofd.write(json.dumps({"n": n, "i": i, "p": p, "candidate_score": candidate_scores[n][i],
                                          "start": locations[i], 
                                          "end": locations[i+n-1]
                                          }) + "\n")
                
    
        print(f"Finished scoring {emb_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", dest="work", help="Work directory")
    parser.add_argument("--embed", dest="embeddings", help="Embeddings file")
    parser.add_argument("--n", dest="n_list", nargs="+", help="n value for chiasmus")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    # I want to use glob to look recursively under the work directory for all embeddings files.
    # it's int he form {work_dir}/{some}/{path}/{language}-embedded.json.gz
    # I want call my score_chiasmus.py script on each of these files in a subprocess

    # I want to use the multiprocessing module to do this
    # I want to use the Pool class to create a pool of workers
    # I want to use the map method to apply the function to each of the files
    emb_files = glob.glob(os.path.join(args.work, "**/*-embedded.json.gz"), recursive=True)
    print(emb_files)
    N_LIST = [int(n) for n in args.n_list]
    SUFFIX = 'new_alg'
    # now we create a pool and map the function to the list of files
    # want to map these so they run in parallel, order doesn't matter
    with mp.Pool() as pool:
        pool.map(score_file, emb_files)


    




    # locations = []
    # embeds = []
    # with gzip.open(args.embeddings, "rb") as ifd:
    #     for line in ifd:
    #         obj = json.loads(line)
    #         locations.append(Location(obj['location']))
    #         embeds.append(obj['embedding'])
    #     locations = np.asarray(locations)
    #     embeds = np.asarray(embeds)

    # cos_sim = cosine_similarity(embeds,embeds)
    

    # # want to take 1000 random starting positions and calculate the chiasmus score for each
    # # then we can use this to calculate the p-value for each chiasmus of size n
    # # want it to be the same set everytime, a random sample but with a fixed seed
    # np.random.seed(42)
    # random_starts = np.random.randint(0, cos_sim.shape[0]-max(n_list)+1, 1000)

    # p_values = {}
    # for n in n_list:
    #     p_values[n] = []
    #     scores = []
    #     for i in random_starts:
    #         scores.append(get_chiasm_score(cos_sim, i, n, 0.95))
    #     avg = 0
    #     avg_p = 0
    #     for i in tqdm(range(0, cos_sim.shape[0]-n+1)):
    #         # get the score of that chiasmus of size n
    #         candidate= get_chiasm_score(cos_sim, i, n, 0.95)
    #         avg += candidate
    #         # calculate p-value
    #         # adding 1 to the numerator and denominator to avoid division by zero (laplace smoothing)
    #         p = (sum([1 for s in scores if s >= candidate]) + 1) / (len(scores)+1)
    #         avg_p += p
    #         p_values[n].append(p)
    #         # p_values is a dictionary with keys as n-values and values as lists of p-values for each starting position
    #         # e.g. p_values[3][0] is the p-value for the chiasmus of size 3 starting at the first position (p_value[n][i])
   

    # # I think we want to save all of this information, write p_values to a jsonl file
    # # fields for starting position i, n-value, p-value
    # with open("p_values.jsonl", "wt") as ofd:
    #     for n, p_list in p_values.items():
    #         for i, p in enumerate(p_list):
    #             ofd.write(json.dumps({"n": n, "i": i, "p": p}) + "\n")
    
