# used in scons as action="python3 scripts/score_translations.py --pred ${SOURCES[0]} --ref ${SOURCES[1]} --output ${TARGET}",

# Path: scripts/score_translations.py
from utils import Bible, Location
import json
import argparse
import gzip
import pandas as pd
from tqdm import tqdm
import evaluate


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", dest="pred", help="Predicted file")
    parser.add_argument("--ref", dest="ref", help="Reference file")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    bleu = evaluate.load("bleu")
    # two fiels, location and text

    with gzip.open(args.pred, "rb") as ifd:
        pred = [json.loads(line) for line in ifd]
        pred_locations = [Location(obj["location"]) for obj in pred]
        pred_texts = [obj["text"] for obj in pred]

    with gzip.open(args.ref, "rb") as ifd:
        ref = [json.loads(line) for line in ifd]
        ref_locations = [Location(obj["location"]) for obj in ref]
        ref_texts = [obj["text"] for obj in ref]

    # align the ref and pred locations
    #make one into a series and then join on that column?
    data = pd.DataFrame({"location": pred_locations, "text": pred_texts})
    data = data.set_index("location")
    ref = pd.DataFrame({"location": ref_locations, "text": ref_texts})
    ref = ref.set_index("location")

    data = data.join(ref, rsuffix="_ref")
    len_before = len(data)
    data = data.dropna()
    print(f"Dropped {len_before-len(data)} rows for which there was no reference.")
    # show the dropped rows
    pred = data["text"]
    ref = data["text_ref"]
    
    # save this result to a jsonl file
    with open(args.output, "wt") as ofd:
        score = bleu.compute(predictions=pred, references=ref)
        ofd.write(json.dumps(score) + "\n")
