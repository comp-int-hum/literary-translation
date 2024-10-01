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

def get_chiasm_score(cos_sim, i, n, sim_threshold=0.95, penalty=False):
    # the basic chiasm score is the sum of the reversed diagonal elements of the cosine similarity matrix
    # then we add a penalty for high similarity scores between different levels. 
    # chiasm = cos_sim[i:i+n, i:i+n][::-1]
    chiasm = cos_sim[i:i+n, i:i+n]
    # now reverse the diagonal
    chiasm = np.fliplr(chiasm)
    score = chiasm.trace()
    
    # if the size of the chiasm is odd, then we want to subtract the value for the middle diagonal element
    if n % 2 == 1:
        score -= chiasm[n//2, n//2]
    
    # should normalize the score to the number of lines in the chiasm
    score /= n

    # adding a penalty for high similarity scores between different levels
    if penalty:
        for j in range(1,n):
            if cos_sim[i+j, i+n-1-j] > sim_threshold:
                score -= cos_sim[i+j, i+n-1-j]
    return score

def score_file(emb_file, output, penalty):
        # I want to call the score_chiasmus.py script on the embeddings file
        # I want to pass the n_list and the output file
        # I want to use the subprocess module to call the script
        # I want to use the check_call function to call the script
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
        for n in n_list:
            p_values[n] = []
            scores = []
            for i in random_starts:
                scores.append(get_chiasm_score(cos_sim, i, n, 0.95, penalty))
            avg = 0
            avg_p = 0
            for i in tqdm(range(0, cos_sim.shape[0]-n+1)):
                # get the score of that chiasmus of size n
                candidate= get_chiasm_score(cos_sim, i, n, 0.95)
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
        with open(output, "wt") as ofd:
            for n, p_list in p_values.items():
                for i, p in enumerate(p_list):
                    ofd.write(json.dumps({"n": n, "i": i, "p": p, 
                                          "start": locations[i], 
                                          "end": locations[i+n-1]
                                          }) + "\n")
                
    
        print(f"Finished scoring {emb_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", dest="embeddings", help="Embeddings file")
    parser.add_argument("--n", dest="n_list", nargs="+", help="n value for chiasmus")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--penalty", dest="penalty", type=bool, help="Whether to include the repetition penalty")
    args = parser.parse_args()



    N_LIST = [int(n) for n in args.n_list]
    # now we create a pool and map the function to the list of files
    # want to map these so they run in parallel, order doesn't matter
    # with mp.Pool() as pool:
    #     pool.map(score_file, emb_files)

    score_file(args.embeddings, args.output, args.penalty)
    

