import argparse as ap
import json
import random
import pandas as pd

from utils import Location


def load_json_lines(file_path):
    """Load JSON lines from a file into a dictionary keyed by location."""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            try:
                location = Location(item['location'])  # Convert to string for consistent keys
            except:
                print(item['location'])
            data[location] = item['text']
    return data


def main(args):
    args.src = args.src.replace("_", ' ')
    original = load_json_lines(args.original)
    translation = load_json_lines(args.human_translation)
    # just choose two random locations and look them up in 
    n_examples = 2
    selected_examples = random.sample(list(original.keys()), n_examples)
    original_texts = [original[ex] for ex in selected_examples]
    translated_texts = [translation[ex] if ex in translation else '' for ex in selected_examples]


    # Construct the few-shot translation prompt
    prompt = f"Translate the following {args.src} phrases into {args.tgt}:\n\n"
    for idx, (o_text, t_text) in enumerate((list(zip(original_texts, translated_texts))), start=1):
        prompt += f"{idx}. {args.src}: \"{o_text}\"\n   {args.tgt}: \"{t_text}\"\n\n"

    prompt += f"Now, translate this {args.src} phrase:\n"
    prompt += f"{n_examples+1}. {args.src}: \"INPUT_TEXT\"\n   {args.tgt}:"

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
