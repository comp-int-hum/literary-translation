import evaluate
import json
import argparse as ap
from utils import Location


def load_json_lines(file_path):
    data = {}
    with open(file_path, 'r') as ifd:
        for line in ifd:
            item = json.loads(line)
            location = Location(item['location'])
            data[location] = item['text']

    return data




if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('--preds')
    parser.add_argument('--refs')
    parser.add_argument('--sources')
    parser.add_argument('--lang')
    parser.add_argument('--output')

    args = parser.parse_args()


    predictions = load_json_lines(args.preds)
    references = load_json_lines(args.refs)
    sources = load_json_lines(args.sources)

    aligned_refs = []
    aligned_preds = []
    aligned_sources = []

    for loc, _ in sources.items():
        if loc in references and loc in predictions:
            if predictions[loc] is not None:
                aligned_refs.append(references[loc])
                aligned_preds.append(predictions[loc])
                aligned_sources.append(sources[loc])

    print(f"Found {len(aligned_sources)} aligned sentences across all three")

    if len(aligned_sources) < 1000:
        print(f"Not enough lines to score")
        exit()
    # Load evaluation metrics from Hugging Face's evaluate library
    comet_metric = evaluate.load("comet")
    sacrebleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    
    wrapped_refs = [[r] for r in aligned_refs]

    if args.lang == "Turkish":
        print("Morphologically rich language found. Using 'char' tokenization for sacreBLEU")
        tokenize = "char"
    else:
        tokenize = 'none'
    sacrebleu_results = sacrebleu_metric.compute(predictions=aligned_preds, references=wrapped_refs, tokenize=tokenize)
    print("SacreBLEU Score:", sacrebleu_results['score'])

    # 2. Compute ChrF score
    chrf_results = chrf_metric.compute(predictions=aligned_preds, references=wrapped_refs)
    print("ChrF Score:", chrf_results['score'])

    # 3. Compute COMET score
    # COMET expects source sentences as well, but if not available, you can pass `None`.
    comet_results = comet_metric.compute(predictions=aligned_preds, references=aligned_refs, sources=aligned_sources)
    print("COMET Score:", comet_results['mean_score']*100)

    with open(args.output, 'wt') as ofd:
        ofd.write(f"SacreBLEU Score: {sacrebleu_results['score']}\n")
        ofd.write(f"ChrF Score: {chrf_results['score']}\n")
        ofd.write(f"COMET Score: {comet_results['mean_score']*100}\n")
