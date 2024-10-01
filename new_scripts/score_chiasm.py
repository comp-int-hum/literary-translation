# this needs to take in a book, a grouping level, a set of features, and perform it over every value of n (no need to waste feature extraction)

# what's the output format? 
import json
import random
import pickle

import numpy as np
import argparse as ap
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import normalize

from sentence_transformers import SentenceTransformer

from tqdm import tqdm

from utils import *


class Location:
    def __init__(self, location_str):
        parts = location_str.split('.')
        self.book = parts[0]
        self.chapter = int(parts[1])
        self.verse = int(parts[2])

############# Feature Extraction ################
class NeuralEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self
        

    def transform(self, X):
        self.embeddings = self.model.encode(X, show_progress_bar=True)
        return self.embeddings

def get_feature_matrix(texts: list, model, features: list):
    ngrams = TfidfVectorizer(
        ngram_range=(1,3),     # N-gram range (unigram, bigram, etc.)
        min_df=1,               # Minimum document frequency
        max_df=1.0,               # Maximum document frequency
        strip_accents="unicode",
        tokenizer=hebrew_word_tokenizer,
        analyzer='word',             # Analyze at word-level
        stop_words=stops,              # No stop words by default (can add Hebrew stop words if necessary)
        max_features= 2000
    )

     #Step 3: Dimensionality reduction with TruncatedSVD for TF-IDF (Sparse)
    svd = TruncatedSVD(n_components=100)
 
    # Create a FeatureUnion to combine the n-gram features and neural embeddings
    feats = []
    if 'ngram' in features:
        feats.append(('ngram', Pipeline([
            ('tfidf', ngrams),
            #('svd', svd)
        ])))
    if 'neural' in features:
        feats.append(('neural_embeddings', Pipeline([
        ('embed', NeuralEmbeddingTransformer(model=model)),  # Custom embedding transformer
        ('scaler', StandardScaler())  # Step 6: Scale neural embeddings
        ])))

    
    assert len(feats) > 0, "There must be at least one feature set"
    
    combined_features = FeatureUnion(feats)

    # Transform to get the combined feature matrix
    combined_matrix = combined_features.fit_transform(texts)

    # Normalize the combined features
    combined_matrix = normalize(combined_matrix)
    return combined_matrix

############# Chiasm Score ################
# def get_chiasm_score(cos_sim, i, n):

#     # the basic chiasm score is the sum of the reversed diagonal elements of the cosine similarity matrix
#     chiasm = cos_sim[i:i+n, i:i+n]
#     # now reverse the diagonal
#     chiasm = np.fliplr(chiasm)
#     els = np.diagonal(chiasm)[:n//2]

#     return np.mean(els)

def get_chiasm_score(cos_sim, i, n, ):

    # the basic chiasm score is the sum of the reversed diagonal elements of the cosine similarity matrix
    chiasm = cos_sim[i:i+n, i:i+n]
    # now reverse the diagonal
    chiasm = np.fliplr(chiasm)
    els = np.diagonal(chiasm)[:n//2]

    thres = 0.3
    score = sum([el > thres for el in els])/len(els)
    return score


def main(args):
    # STEPS:
    # 0. load a book
    with open(args.book, 'rt') as f:
        records = [json.loads(line) for line in f]
    
    df = pd.DataFrame.from_records(records)
    # if I instead pass in a list of tuples of location and text, then we're fine
    texts = df['line'].tolist()
    
    # 1. group texts given a level (includes cleaning)
    groups, indices = group_verses(texts, group_type=args.group)
    if len(groups) == 0:
        print(f"No relevant groups found for [{args.group}] in [{args.book}], skipping...")
        # with open(args.output, 'wb') as f:
        #     
        #     pickle.dump({'dummy': None}, f)
        exit()
    
    if 'neural' in args.feats:
        model = SentenceTransformer('intfloat/multilingual-e5-small')
    else:
        model=None
    # 2. extract features
    feats = get_feature_matrix(groups, model, args.feats)
    
    # 3. compute observed and randomized chiasm scores
    cos_sim = cosine_similarity(feats, feats)

    N = list(range(4, min(21, cos_sim.shape[0])))

    scores = {}
    n_scores = {}
    for n in tqdm(N):
        os = []
        for i in range(len(groups)-n):
            os.append(get_chiasm_score(cos_sim, i, n=n))
        scores[n] = os
        
        # ns = []
        # num_trials = 1_000
        # for _ in range(num_trials):
        #     ns.append(get_chiasm_score(cos_sim, random.choice(range(len(groups)-n)), n=n))
        # n_scores[n] = ns

    # save to file for inspection/visualization. 
    with open(args.output, 'wb') as f:
        # 'null_scores': n_scores,
        pickle.dump({'scores': scores, 'indices': indices}, f)

if __name__=="__main__":
    # argparser here
    parser = ap.ArgumentParser()
    parser.add_argument("--book", type=str)
    parser.add_argument("--group", choices=['half', 'verse', 'pesucha', 'setuma'])
    parser.add_argument("--feats", nargs="+", choices=['ngram', 'neural'])
    parser.add_argument("--output", type=str)

    args = parser.parse_args()

    
    main(args)
    
    
    

