import logging
import gzip
import json
import argparse

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

logger = logging.getLogger("translate_document")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--source_lang", dest="source_lang", default="en_XX")
    parser.add_argument("--target_lang", dest="target_lang", default="fr_XX")
    parser.add_argument("--device", dest="device", default="cpu")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if "_" not in args.source_lang:
        args.source_lang = args.source_lang + "_XX"

    if "_" not in args.target_lang:
        args.target_lang = args.target_lang + "_XX"        
    
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model.to(args.device)
    tokenizer.src_lang = args.source_lang
    
    with gzip.open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:

        batch = []
        #for i, item in enumerate(xml.findall(".//*[@type='verse']")):
        for i, line in enumerate(ifd):
            item = json.loads(line)
            batch.append((item["id"], item["text"]))
            if len(batch) == args.batch_size:
                encoded = tokenizer([t for _, t in batch], return_tensors="pt", padding=True)
                encoded.to(args.device)
                generated_tokens = model.generate(
                    **encoded,
                    forced_bos_token_id=tokenizer.lang_code_to_id[args.target_lang]
                )
                toks = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                for (label, text), trans in zip(batch, toks):
                    ofd.write(json.dumps({"id" : label, "text" : translation}) + "\n")
                batch = []
                logger.info("Processed %d sentences", i)
                
        if len(batch) > 0:
            encoded = tokenizer([t for _, t in batch], return_tensors="pt", padding=True)
            encoded.to(args.device)
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.lang_code_to_id[args.target_lang]
            )
            toks = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            for (label, text), trans in zip(batch, toks):
                ofd.write(json.dumps({"id" : label, "text" : translation}) + "\n")
