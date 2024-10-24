import logging
import gzip
import json
import argparse
import nltk
import iso639
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel, M2M100ForConditionalGeneration, M2M100Tokenizer
from utils import Location
import stopwordsiso as sw
import sys
import os
from tqdm import tqdm

logger = logging.getLogger("translate_document")

# marathi japanese

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--disallow_target", dest="disallow_target", help="Document in target language to avoid")
    parser.add_argument("--disallow_referenced", dest="disallow_referenced", help="Reference annotation")
    parser.add_argument("--vote_threshold", dest="vote_threshold", default=50, type=int)
    parser.add_argument("--source_lang", dest="source_lang", default="en_XX")
    parser.add_argument("--target_lang", dest="target_lang", default="fr_XX")
    parser.add_argument("--device", dest="device", default="cpu")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=10)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # let's make some model specific processing
    MAP = {'facebook/m2m100_1.2B': {'model': M2M100ForConditionalGeneration,
                                    'tokenizer': M2M100Tokenizer},
           'facebook/nllb-200-3.3B': {'model': AutoModelForSeq2SeqLM,
                                                'tokenizer': NllbTokenizer}
           }
    opt = MAP[args.model]
    
    if args.model == 'facebook/m2m100_1.2B':
        tokenizer = opt['tokenizer'].from_pretrained(args.model)
        tokenizer.src_lang = args.source_lang

        model = opt['model'].from_pretrained(args.model, load_in_4bit=True)
        bos_id = tokenizer.get_lang_id(args.target_lang)

    elif args.model == 'facebook/nllb-200-3.3B':
        model = opt['model'].from_pretrained(args.model, load_in_8bit=True)
        tokenizer = opt['tokenizer'].from_pretrained(args.model, src_lang=args.source_lang, tgt_lang=args.target_lang
                                                     )
    else:
        raise Error(f"model f{model} not recognized. You might need to add model-specific loading logic.")
    #model = M2M100ForConditionalGeneration.from_pretrained(args.model, load_in_8bit=True)
    try:
        model.to(args.device)
    except:
        pass

    try:
        target_stopwords = set(nltk.corpus.stopwords.words(iso639.find(args.target_lang.split("_")[0])["name"].lower()))
    except:
        target_stopwords = set(sw.stopwords(iso639.find(args.target_lang.split("_")[0])["iso639_1"]))

    disallow = {}
    if args.disallow_target:
        with open(args.disallow_target, "rt") as ifd:
            for i, line in enumerate(ifd):
                item = json.loads(line)
                disallow[Location(item["location"])] = sum([[w.lower(), w.title()] for w in item["text"].split() if w.lower() not in target_stopwords], [])

    back_references = {}
    previous_translations = {}
    if args.disallow_referenced:
        with open(args.disallow_referenced, "rt") as ifd:
            ifd.readline()
            for line in ifd:
                if "-" not in line:
                    a, b, votes = line.strip().split()
                    a = Location(a)
                    b = Location(b)
                    if a < b:
                        src, tgt = a, b
                    else:
                        src, tgt = b, a
                    back_references[tgt] = back_references.get(tgt, set())
                    if int(votes) > args.vote_threshold:
                        back_references[tgt].add(src)

    with open(args.input, "rt") as ifd:#, #open(args.output, "wt") as ofd:
        all_lines = []
        for i, line in enumerate(ifd):
            item = json.loads(line)
            all_lines.append((Location(item["location"]), item["text"]))


    existing_data = {}
    if os.path.exists(f"{args.output}"):
        with open(f"{args.output}", 'rt') as ifd:
            for line in ifd:
                item = json.loads(line)
                location = Location(item['location'])
                existing_data[location] = item['text']
    
        logger.info(f"\n\nbefore pruning, {len(all_lines)} lines to translate")
        all_prompts = [p for p in all_lines if p[0] not in existing_data]
        logger.info(f"after pruning, {len(all_prompts)} lines to translate\n\n")
  
    with open(args.output, "a") as ofd:
        batch = []
        for i, line in enumerate(tqdm(all_lines)):
            batch.append(line)
            if len(batch) == args.batch_size:
                encoded = tokenizer([t for _, t in batch], return_tensors="pt", padding=True)
                encoded.to(args.device)
                bad_token_ids = sum(
                    [
                        
                        [
                            tokenizer([w], add_special_tokens=False).input_ids[0] for w in disallow.get(iid, [])
                        ] + sum(
                            [previous_translations.get(pid, []) for pid in back_references.get(iid, [])],
                            []
                        ) for iid, _ in batch
                    ],
                    []
                )
                generated_tokens = model.generate(
                    **encoded,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(args.target_lang), #bos_id
                    max_length=100,
                    bad_words_ids=bad_token_ids if bad_token_ids else None
                )
                translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                for (location, text), trans in zip(batch, translations):

                    toks = trans.split()
                    if args.disallow_referenced:
                        previous_translations[location] = sum(
                            [
                                [
                                    tokenizer([w.lower()], add_special_tokens=False).input_ids[0],
                                    tokenizer([w.title()], add_special_tokens=False).input_ids[0]
                                ] for w in toks if w.lower() not in target_stopwords
                            ],
                            []
                        )
                        
                                    
                    ofd.write(json.dumps({"location" : location, "text" : trans}) + "\n")
                batch = []
                logger.info("Processed %d sentences", i + 1)

        if len(batch) > 0:
            encoded = tokenizer([t for _, t in batch], return_tensors="pt", padding=True)
            encoded.to(args.device)
            bad_token_ids = sum(
                [
                    [
                        tokenizer([w], add_special_tokens=False).input_ids[0] for w in disallow.get(iid, [])
                    ] + sum(
                            [previous_translations.get(pid, []) for pid in back_references.get(iid, [])],
                            []
                        ) for iid, _ in batch
                ],
                []
            )
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(args.target_lang),
                max_length=100,
                bad_words_ids=bad_token_ids if bad_token_ids else None
            )
            toks = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            for (location, text), trans in zip(batch, toks):
                ofd.write(json.dumps({"location" : location, "text" : trans}) + "\n")
            logger.info("Processed %d sentences", i + 1 + len(batch))
