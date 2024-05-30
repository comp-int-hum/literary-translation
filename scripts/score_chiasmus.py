import argparse
import numpy as np
import pandas as pd
import gzip
import json
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import Location

def get_chiasm_score(cos_sim, i, n, sim_threshold=0.95):
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
    # print(score)
    # adding a penalty for high similarity scores between different levels
    # for j in range(1,n):
    #     if cos_sim[i+j, i+n-1-j] > sim_threshold:
    #         score -= cos_sim[i+j, i+n-1-j]
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", dest="embeddings", help="Embeddings file")
    parser.add_argument("--n-list", dest="n_list", nargs="+", help="n value for chiasmus")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()


    n_list = [int(n) for n in args.n_list]
    locations = []
    embeds = []
    with gzip.open(args.embeddings, "rb") as ifd:
        for line in ifd:
            obj = json.loads(line)
            locations.append(Location(obj['location']))
            embeds.append(obj['embedding'])
        locations = np.asarray(locations)
        embeds = np.asarray(embeds)

    cos_sim = cosine_similarity(embeds,embeds)

    # want to take 1000 random starting positions and calculate the chiasmus score for each
    # then we can use this to calculate the p-value for each chiasmus of size n
    # want it to be the same set everytime, a random sample but with a fixed seed
    np.random.seed(42)
    random_starts = np.random.randint(0, cos_sim.shape[0]-max(n_list)+1, 1000)

    p_values = {}
    for n in n_list:
        p_values[n] = []
        scores = []
        for i in random_starts:
            scores.append(get_chiasm_score(cos_sim, i, n, 0.95))
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
   
    # # make a plot of p values for each n
    # for n in n_list:
    #     plt.hist(p_values[n], label="n={}".format(n), bins=20)
    # plt.legend()
    # plt.title("P-values for different n-values")
    # plt.xlabel("P-value")
    # plt.ylabel("Frequency")
    # plt.savefig("p_values.png")
    
    # we want to find what proportion of starting positions have a p-value less than 0.005 for each n
    # for n in n_list:
        # print("n={}: {}%".format(n, sum([1 for p in p_values[n] if p < 0.005])/len(p_values[n])*100))

    # I think we want to save all of this information, write p_values to a jsonl file
    # fields for starting position i, n-value, p-value
    with open("p_values.jsonl", "wt") as ofd:
        for n, p_list in p_values.items():
            for i, p in enumerate(p_list):
                ofd.write(json.dumps({"n": n, "i": i, "p": p}) + "\n")
    
    # min_len = min([len(v) for _,v in p_values.items()])
    # p_values_array = np.array([v[:min_len] for _,v in p_values.items()]).T

    # # we have an array of p-values for different n-values for a list of starting positions
    # # for each starting position, want to find the n-value with the lowest p-value
    # # then save a file of the starting position and the n-value
    # best_n = np.argmin(p_values_array, axis=1)
    # best_p = np.min(p_values_array, axis=1)
    # with open(args.output, "wt") as ofd:
    #     for i, (n, p) in enumerate(zip(best_n, best_p)):
    #         ofd.write("{}\t{}\t{}\n".format(i, args.n_list[n], p))
