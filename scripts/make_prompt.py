import argparse as ap
import json
import random
import pandas as pd

from utils import Location


def main(args):
    with open(args.original, 'rt', encoding='utf-8') as ifd:
        orig_data = []
        for line in ifd:
            orig_data.append(json.loads(line))

    orig_df = pd.DataFrame.from_records(orig_data)

    # just choose two random locations and look them up in 
    selected_examples = random.sample(range(len(orig_df)), 2)
    print(selected_examples)

    refs = [Location(orig_df['eng_ref'][i]) for i in selected_examples]
    orig_texts = [orig_df['line'][i] for i in selected_examples] # or "text", but whatever
    print(refs)
    


    with open(args.human_translation, 'rt', encoding='utf-8') as ifd:
        trans_data = []
        for line in ifd:
            trans_data.append(json.loads(line))

    

    trans_df = pd.DataFrame.from_records(trans_data)
    # trans_df['location'] = trans_df['location'].apply(lambda x: Location(x))

    aligned = orig_df.merge(trans_df, on='location')
    print(aligned.head())
    exit()
    
    aligned_data = {Location(item['location']): (item['text'], trans_data[item['location']]['text'])
                    for item in orig_data if Location(item['location']) in trans_data}
    
    print(aligned_data)

    #Randomly select `n_examples` from the aligned data
    selected_examples = random.sample(aligned_data.items(), 2)

    # Construct the few-shot translation prompt
    prompt = f"Translate the following {args.src} phrases into {args.tgt}:\n\n"
    for idx, (_, (ancient_text, human_translation)) in enumerate(selected_examples, start=1):
        prompt += f"{idx}. {args.src}: \"{ancient_text}\"\n   {args.tgt}: \"{human_translation}\"\n\n"

    with open(args.output, 'wt') as f:
        f.write(prompt)




if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--original")
    parser.add_argument("--human_translation")
    parser.add_argument("--src")
    parser.add_argument("--tgt")
    parser.add_argument("--output")

    args = parser.parse_args()

    main(args)
