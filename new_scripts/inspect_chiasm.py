
# here we want to load the pickle, if it has a 'dummy' category, just create a dummy output and move on

# but otherwise we take the scores, n_scores, and indices, and just continue where we left off, also the book
import argparse as ap
import numpy as np
import pandas as pd
import json
import pickle
from score_chiasm import Location


def modified_z_scores(scores):
    # Calculate the median of the scores
    median = np.median(scores)
    
    # Calculate the Median Absolute Deviation (MAD)
    mad = np.median([np.abs(score - median) for score in scores])
    
    # If MAD is 0 (happens if all values are the same), return a list of zeros
    if mad == 0:
        return np.zeros(len(scores))
    
    # Calculate the modified Z-scores
    modified_z_scores = [0.6745 * (score - median) / mad for score in scores]
    
    return modified_z_scores

def main(args):
    with open(args.input, 'rt', encoding='utf-8') as ifd:
        OT_data = []
        for line in ifd:
            OT_data.append(json.loads(line))
    
    df = pd.DataFrame.from_records(OT_data)
    # print(df.head())
    # exit()
    # texts = df['line'].tolist()
    # translations = df['translation'].tolist()
    locs = [Location(x) for x in df['heb_ref'].tolist()]

    with open(args.scores, 'rb') as f:
        saved_data = pickle.load(f)

    scores = saved_data['scores']
    indices = saved_data['indices']
    print(indices[:5])
    
    top_chiasms = []
    vis = {k: 0 for k in [x.book for x in locs]}
    
    for n, dd in scores.items():
        os = dd['os']
        els = dd['els']
        os = modified_z_scores(os)
        thres = 3.5
        # thres = np.mean(os) + 2*np.std(os)
        print(f"using threshold: {thres}")
        
        for i, (o, e) in enumerate(zip(os, els)):
            if o > thres:
                try:
                    book = locs[indices[i][0]].book
                    vis[book] +=1
                except:
                    pass
                    # print(indices[i][0])
                    # book = locs[int(indices[i][0][:-1])].book
                 
                refs=[]
                text = []
                heb_text = []
                for line in indices[i:i+n]:
                    if 'half' in args.scores:
                        # then we know it's a half-verse processing
                        for idx in line:
                            if 'a' in idx:
                                refs.append(df['heb_ref'].tolist()[int(idx[:-1])])
                                text.append(df['trans_a'].tolist()[int(idx[:-1])])
                                heb_text.append(df['half_a'].tolist()[int(idx[:-1])])
                            elif 'b' in idx:
                                # print(df.iloc[int(idx[:-1])])
                                refs.append(df['heb_ref'].tolist()[int(idx[:-1])])
                                text.append(df['trans_b'].tolist()[int(idx[:-1])])
                                heb_text.append(df['half_b'].tolist()[int(idx[:-1])])
                            else:
                                refs.append(df['heb_ref'].tolist()[int(idx)])
                                text.append(df['translation'].tolist()[int(idx)])
                                heb_text.append(df['line'].tolist()[int(idx)])
                    else:
                        refs.append([df['heb_ref'].tolist()[idx] for idx in line])
                        text.append([df['translation'].tolist()[idx] for idx in line])
                        heb_text.append([df['line'].tolist()[idx] for idx in line])
                
                top_chiasms.append({"n": n,
                                    "thres": thres,
                                    "score": o,
                                    "raw_score": e,
                                    "refs": "\n".join([str(x) for x in refs]),
                                    "text": "\n".join([str(x) for x in heb_text]),
                                    "translation": "\n".join([str(x) for x in text])})
    
    df = pd.DataFrame.from_records(top_chiasms)
    print(df.head())
    #df = df.sample(args.subset)

    if len(df) > 0:
        # df.sort_values(by='score', ascending=False, inplace=True)
        df.sort_values(by='score', ascending=False, inplace=True)
        df = df[:1000]
        #df.sample(args.subset).to_excel(args.output)
        df.head(500).to_json(args.output+".json", orient='records', lines=True, force_ascii=False)
        # df.head(500).to_excel(args.output)

    
        print(df['n'].value_counts())
        #print(top_chiasms[0])
        with open(f"{args.output}.vis", 'wb') as f:
         pickle.dump(vis, f)

    #pd.DataFrame.from_records(top_chiasms).to_excel(args.output)

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--scores", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--subset", type=int)


    args = parser.parse_args()

    main(args)
