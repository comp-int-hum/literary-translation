
# here we want to load the pickle, if it has a 'dummy' category, just create a dummy output and move on

# but otherwise we take the scores, n_scores, and indices, and just continue where we left off, also the book
import argparse as ap
import numpy as np
import pandas as pd
import json
import pickle
from score_chiasm import Location

def main(args):
    with open(args.input, 'rt', encoding='utf-8') as ifd:
        OT_data = []
        for line in ifd:
            OT_data.append(json.loads(line))
    
    df = pd.DataFrame.from_records(OT_data)
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
    for n, os in scores.items():
        thres = np.mean(os) + 2*np.std(os)
        
        for i, o in enumerate(os):
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
                                    "refs": "\n".join([str(x) for x in refs]),
                                    "text": "\n".join([str(x) for x in heb_text]),
                                    "translation": "\n".join([str(x) for x in text])})
    print(top_chiasms[0])
    with open(f"{args.output}.vis", 'wb') as f:
        pickle.dump(vis, f)

    pd.DataFrame.from_records(top_chiasms).to_excel(args.output)

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--scores", type=str)
    parser.add_argument("--output", type=str)


    args = parser.parse_args()

    main(args)
