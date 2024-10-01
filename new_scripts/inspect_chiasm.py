
# here we want to load the pickle, if it has a 'dummy' category, just create a dummy output and move on

# but otherwise we take the scores, n_scores, and indices, and just continue where we left off, also the book
import argparse as ap
import numpy as np
import pandas as pd
import json
import pickle

def main(args):
    with open(args.book, 'rt') as f:
        records = [json.loads(line) for line in f]
    
    df = pd.DataFrame.from_records(records)

    with open(args.scores, 'rb') as f:
        saved_data = pickle.load(f)
    if 'dummy' in saved_data.keys():
        # make a dummy output
        print("Detected dummy input, skipping...")
        with open(args.output, 'wt') as f:
            pass
        exit()

    scores = saved_data['scores']
    #null_scores = saved_data['null_scores']
    indices = saved_data['indices']

    ss_chiasms = []
    for n_val, os in scores.items():
        for i, o in enumerate(os):
            # if the percentage of >50% of the lines are similar enough
            if o > 0.5:
                refs=[]
                text = []
                heb_text = []
                #print(indices[i:i+n_val])
                for line in indices[i:i+n_val]:
                    if type(line[0]) == str:
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
                
                ss_chiasms.append({"refs": "\n".join([str(x) for x in refs]),
                                "text": "\n".join([str(x) for x in heb_text]),
                                "translation": "\n".join([str(x) for x in text])})
    # ss_chiasms = []
    # for n_val, scores in scores.items(): # for each value of n
    #     n_scores = null_scores[n_val] # need to test each observed score against null distribution
    #     p_values = np.array([np.mean(n_scores >= obs_score) for obs_score in scores])
    #     for i, p_val in enumerate(p_values):
    #         if p_val < args.alpha:
    #             refs=[]
    #             text = []
    #             heb_text = []
    #             for line in indices[i:i+n_val]:
    #                 if type(line[0]) == str:
    #                     # then we know it's a half-verse processing
    #                     for idx in line:
    #                         if 'a' in idx:
    #                             refs.append(df['heb_ref'].iloc[int(idx[:-1])])
    #                             text.append(df['trans_a'].iloc[int(idx[:-1])])
    #                             heb_text.append(df['half_a'].iloc[int(idx[:-1])])
    #                         elif 'b' in idx:
    #                             refs.append(df['heb_ref'].iloc[int(idx[:-1])])
    #                             text.append(df['trans_b'].iloc[int(idx[:-1])])
    #                             heb_text.append(df['half_b'].iloc[int(idx[:-1])])
    #                         else:
    #                             refs.append(df['heb_ref'].iloc[int(idx)])
    #                             text.append(df['translation'].iloc[int(idx)])
    #                             heb_text.append(df['line'].iloc[int(idx)])
    #                 else:
    #                     refs.extend([df['heb_ref'].iloc[idx] for idx in line])
    #                     text.extend([df['translation'].iloc[idx] for idx in line])
    #                     heb_text.extend([df['line'].iloc[idx] for idx in line])
 
    #             ss_chiasms.append({"refs": "\n".join(refs),
    #                             "text": "\n".join(heb_text),
    #                             "translation": "\n".join(text)})

    pd.DataFrame.from_records(ss_chiasms).to_excel(args.output)

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--book", type=str)
    parser.add_argument("--scores", type=str)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--output", type=str)


    args = parser.parse_args()

    main(args)