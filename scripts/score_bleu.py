import json
import sacrebleu
import argparse as ap
from evaluate import load
from utils import Location

bleu_metric = load("sacrebleu")

def load_json_lines(file_path):
    """Load JSON lines from a file into a dictionary keyed by location."""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            location = Location(item['location'])  # Convert to string for consistent keys
            data[location] = item['text']
    return data

def main(predictions_file, references_file):
    # Load predictions and references
    predictions = load_json_lines(predictions_file)
    references = load_json_lines(references_file)

    # Align predictions and references
    aligned_refs = []
    aligned_preds = []

    for location in predictions:
        if location in references:
            aligned_preds.append(predictions[location])
            aligned_refs.append(references[location])
    print(len(aligned_refs), len(aligned_preds))
    #print(aligned_preds[0])
    #print(aligned_refs[0])
    # Calculate the BLEU score using sacrebleu
    #bleu = sacrebleu.corpus_bleu(aligned_preds, [[ref] for ref in aligned_refs])

    # Print the BLEU score
    #print("BLEU score:", bleu.score)

    results = bleu_metric.compute(predictions=aligned_preds, references=[[ref] for ref in aligned_refs])
    print("SacreBLEU score:", round(results['score'], 2))

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--preds")
    parser.add_argument("--refs")

    args = parser.parse_args()
    # Replace these with your actual file paths
    predictions_file = args.preds
    references_file = args.refs
    
    main(predictions_file, references_file)

