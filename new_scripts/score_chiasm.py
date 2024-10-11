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

# from utils import *

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re

############## Helpers ################
with open('heb_stopwords.txt', 'rt') as f:
    stops = f.readlines()
    stops = [s.strip() for s in stops]

def remove_nikkud(text, keep_end=False):
    # Define the regex pattern
    if keep_end:
        return re.sub(r'[\u0591-\u05AF\u05B0-\u05BD\u05BF\u05C1-\u05C2\u05C4-\u05C7]', '', text)
    return re.sub(r'[\u0591-\u05C7]', '', text)

def hebrew_word_tokenizer(text):
    """
    Tokenizer function to split Hebrew text into words.
    This will capture words and also consider special characters like apostrophes.
    
    :param text: A string of Hebrew text.
    :return: A list of tokenized words.
    """
    # Regular expression to match Hebrew words, including those with apostrophes (׳)
    hebrew_word_pattern = r"\b[\u0590-\u05FF׳]+(?:-[\u0590-\u05FF׳]+)*\b"
    
    # Use re.findall to get all the words matching the pattern
    return re.findall(hebrew_word_pattern, text)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Location(dotdict):
    def __init__(self, location_str):
        parts = location_str.split('.')
        self['book'] = parts[0]
        self['chapter'] = int(parts[1])
        self['verse'] = int(parts[2])

    def __str__(self):
        return f"{self.book}.{self.chapter}.{self.verse}"
    def __hash__(self):
        return hash(repr(self))

# Define special symbols
ATNACH = '\u0591'  # Unicode for Atnach (half-verse separator)
PESUCHA = '׃ פ'     # Pesucha separator
SETUMA = '׃ ס'      # Setuma separator

def group_verses(data, group_type):
    """
    Group verses into half-verses, pesucha, or setuma sections.
    
    Args:
    - data: list of tuples of locations and verse strings
    - group_type: how to group the verses, can be 'half', 'pesucha', or 'setuma'
    
    Returns:
    - A list of grouped verse strings.
    - A list of lists of verse indices in their groups
    """
    
    locs, verses = data
    
    # the half-verse processing requires different processing as each verse must be processed individually
    if group_type == "verse":
        indices = [[i] for i in range(len(verses))] # [[0], [1], [2]...] bc each group is just one verse
        return [remove_nikkud(v) for v in verses], indices
        # return verses, indices
    
    # Define the splitting logic based on group type
    elif group_type == 'half':
        groups = []
        indices = []
        for idx, verse in enumerate(verses):
            words = verse.split()
            split = None
            for i, w in enumerate(words):
                if ATNACH in w:
                    split = i
            if split != None: # if there are half-verses, add them individually
                groups.append(' '.join(words[:split+1]))
                groups.append(' '.join(words[split+1:]))
                indices.append([f"{idx}a"])
                indices.append([f"{idx}b"])
            else: # otherwise, add the whole verse
                groups.append(verse)
                indices.append([f"{idx}"])

        # some post-processing: remove nikkud and PESUCHA and SETUMA symbols
        groups = [remove_nikkud(g, keep_end=False).replace(PESUCHA, '׃').replace(SETUMA, '׃') for g in groups]
        # groups = [g.replace(PESUCHA, '׃').replace(SETUMA, '׃') for g in groups]
        return groups, indices
    
    ##################################################
    # Concatenate all verses into a single text block
    # I think to make the indexing possible, I need to change how I do this. Need to iterate through so I can
    # keep track of verse delineationgs
    elif group_type in ['pesucha', 'setuma']:
        split_char = PESUCHA if group_type == 'pesucha' else SETUMA
        groups = []
        indices = []
        curr_text = ''
        curr_book = locs[0].book
        # print(f"curr_book is {curr_book}")
        curr_vs = []
        for idx, verse in enumerate(verses):
            book = locs[idx].book
            # if we're still in the same book:
            if book == curr_book:
                # print(f"{book} equals {curr_book}")
                if split_char in verse[-4:]:
                    # print(f"{split_char} in verse: {verse[-4:]}")
                    # add the final verse text in
                    curr_text += verse
                    curr_vs.append(idx)
                    # the group is complete, add it to groups
                    groups.append(remove_nikkud(curr_text, keep_end=True).replace(PESUCHA, '׃').replace(SETUMA, '׃'))
                    # groups.append(curr_text.replace(PESUCHA, '׃').replace(SETUMA, '׃'))
                    indices.append(curr_vs)
                    # print(f"adding {[locs[x] for x in curr_vs]} to groups")
                    # reset counters
                    curr_text = ''
                    curr_vs = []
                else:
                    curr_text += verse
                    curr_vs.append(idx)
            # otherwise, we just add whatever is leftover as a group and reset everything
            else:
                # print(f"moved onto a new book: {book}")
                # print(curr_text[-4:])
                groups.append(remove_nikkud(curr_text, keep_end=True).replace(PESUCHA, '׃').replace(SETUMA, '׃'))
                # groups.append(curr_text.replace(PESUCHA, '׃').replace(SETUMA, '׃'))
                # print(f"adding {[locs[x] for x in curr_vs]} to groups")
                indices.append(curr_vs)
                curr_text = verse
                curr_vs = [idx]
                curr_book = book
                
        return groups, indices
     
    else:
        raise ValueError(f"Unknown group type: {group_type}. Choose from 'verse', 'half', 'pesucha', or 'setuma'.")
    
################ Feature Extraction ################
# Custom transformer for neural embeddings
class NeuralEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self
        

    def transform(self, X):
        self.embeddings = self.model.encode(X, show_progress_bar=True)
        return self.embeddings
    
def get_feature_matrix(texts: list, features: list):
    ngrams = TfidfVectorizer(
        ngram_range=(1,3),     # N-gram range (unigram, bigram, etc.)
        min_df=1,               # Minimum document frequency
        max_df=1.0,               # Maximum document frequency
        strip_accents="unicode",
        tokenizer=hebrew_word_tokenizer,
        analyzer='word',             # Analyze at word-level
        stop_words=stops,              # No stop words by default (can add Hebrew stop words if necessary)
        max_features= 5000
    )

    # Create a FeatureUnion to combine the n-gram features and neural embeddings
    feats = []
    if 'ngram' in features:
        feats.append(('ngram', Pipeline([
            ('tfidf', ngrams),
        ])))
    if 'neural' in features:
        feats.append(('neural_embeddings', Pipeline([
        ('embed', NeuralEmbeddingTransformer(model=SentenceTransformer('intfloat/multilingual-e5-small'))),  # Custom embedding transformer
        ('scaler', StandardScaler())  # Step 6: Scale neural embeddings
        ])))


    assert len(feats) > 0, "There must be at least one feature set"

    combined_features = FeatureUnion(feats)

    # Transform to get the combined feature matrix
    combined_matrix = combined_features.fit_transform(texts)

    # Normalize the combined features
    if len(features) > 1:
        combined_matrix = normalize(combined_matrix)
    return combined_matrix, combined_features


############# Chiasm Score ################

def get_chiasm_score(cos_sim, i, n, sim_threshold=0.9):
    # the basic chiasm score is the sum of the reversed diagonal elements of the cosine similarity matrix
    chiasm = cos_sim[i:i+n, i:i+n]

    # now reverse the diagonal
    chiasm = np.fliplr(chiasm)
    els = np.diagonal(chiasm)[:n//2]
  
    score = np.mean(els)
    for j in range(1,n):
        if cos_sim[i+j, i+n-1-j] > sim_threshold:
            score -= cos_sim[i+j, i+n-1-j]

    return score

def main(args):
    # STEPS:
    # 0. load a book
    with open(args.input, 'rt', encoding='utf-8') as ifd:
        OT_data = []
        for line in ifd:
            OT_data.append(json.loads(line))
    
    df = pd.DataFrame.from_records(OT_data)
    texts = df['line'].tolist()
    translations = df['translation'].tolist()
    locs = [Location(x) for x in df['heb_ref'].tolist()]
    # data = list(zip(locs, texts))
    
    # 1. group texts given a level (includes cleaning)
    groups, indices = group_verses((locs, texts), group_type=args.group)
    if len(groups) == 0:
        print(f"No relevant groups found for [{args.group}] in [{args.output}], skipping...")
        exit()
    
    # 2. extract features
    feats, vectorizer = get_feature_matrix(groups, args.feats)
    
    # 3. compute observed and randomized chiasm scores
    cos_sim = cosine_similarity(feats, feats)

    N = [4,5]

    scores = {}
    for n in tqdm(N):
        os = []
        for i in range(len(groups)-n):
            opt = True
            try:
                curr_book = locs[indices[i][0]].book
                for gr in indices[i:i+n]:
                    for v in gr:
                        if locs[v].book != curr_book:
                            #print(f"{locs[v].book} does not match {curr_book}, so skipping")
                            opt=False
                            break
                        if not opt:
                            break
                if opt:
                    os.append(get_chiasm_score(cos_sim, i, n=n))
                else:
                    os.append(0)
            except:
                os.append(0)
        scores[n] = os
        

    # these are all the scores


    # save to file for inspection/visualization. 
    with open(args.output, 'wb') as f:
        # 'null_scores': n_scores,
        pickle.dump({'scores': scores, 'indices': indices}, f)

if __name__=="__main__":
    # argparser here
    parser = ap.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--group", choices=['half', 'verse', 'pesucha', 'setuma'])
    parser.add_argument("--feats", nargs="+", choices=['ngram', 'neural'])
    parser.add_argument("--output", type=str)

    args = parser.parse_args()

    
    main(args)
    
    
    

