import logging
import gzip
import json
import argparse
import nltk
import iso639
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils import Location
import stopwordsiso as sw
import sys
import torch
import os
from tqdm import tqdm

logger = logging.getLogger("translate_document")

def make_input_example(prompt, text):
    # Add the new input text to translate
    return prompt.replace("INPUT_TEXT", text)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--prompt", dest="prompt", help = "Prompt file")
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

    # load the prompt
    with open(args.prompt, 'rt') as f:
        prompt = f.readlines()
        prompt = "".join(prompt)

    # load model in 4-bit
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quantization_config,)
                                                 #attn_implementation="flash_attention_2")
    # model = model.to_bettertransformer()

    # model = AutoModelForCausalLM.from_pretrained(args.model,
    #                                              torch_dtype=torch.bfloat16,
    #                                              attn_implementation="flash_attention_2")
    MAP= {
        "English": "eng",
        "Finnish": "fin",
        "Turkish": "tur",
        "Swedish": "swe",
        "Marathi": "mar"
    }
    try:
        target_stopwords = set(nltk.corpus.stopwords.words(iso639.find(MAP[args.target_lang].split("_")[0])["name"].lower()))
    except:
        target_stopwords = set(sw.stopwords(iso639.find(MAP[args.target_lang].split("_")[0])["iso639_1"]))

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
            all_lines.append((Location(item["location"]), make_input_example(prompt, item["text"])))

    existing_data = {}
    if os.path.exists(f"{args.output}_copy"):
        with open(f"{args.output}_copy", 'rt') as ifd:
            for line in ifd:
                item = json.loads(line)
                location = Location(item['location'])
                existing_data[location] = item['text']

        logger.info(f"\n\nbefore pruning, {len(all_lines)} lines to translate. Found {len(existing_data)} lines already")
        all_lines = [p for p in all_lines if p[0] not in existing_data]
        logger.info(f"after pruning, {len(all_lines)-len(existing_data)} remaining lines to translate\n\n")
    else:
        logger.info("no previous translations found")
    


    with open(f"{args.output}_copy", "a") as ofd:
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
                        max_new_tokens=100,
                        bad_words_ids=bad_token_ids if bad_token_ids else None,
                        num_return_sequences = 1,
                        repetition_penalty = 1.1
                    )
                translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                print(translations[0], [l for l, _ in batch][0])
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
                            
                    #print(trans)        
                    ofd.write(json.dumps({"location" : location, "text" : trans}, ensure_ascii=False) + "\n")
                batch = []
                logger.info("Processed %d sentences", i + 1)

                # sanity checking:
                if i >= 2_000:
                    break

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
                        max_new_tokens=100,
                        bad_words_ids=bad_token_ids if bad_token_ids else None,
                        num_return_sequences = 1,
                        repetition_penalty = 1.1
                    )
                toks = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                for (location, text), trans in zip(batch, toks):
                    ofd.write(json.dumps({"location" : location, "text" : trans}, ensure_ascii=False) + "\n")
                logger.info("Processed %d sentences", i + 1 + len(batch))
