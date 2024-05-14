import logging
import gzip
import json
import argparse
#from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from utils import Location
from sentence_transformers import SentenceTransformer





logger = logging.getLogger("embed_document")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    #parser.add_argument("--language", dest="language", default="en", help="Language code")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--device", dest="device", default="cpu")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=500)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model = SentenceTransformer('sentence-transformers/LaBSE')
    
    #model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    #tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    #tokenizer.src_lang = args.language
    model.to(args.device)
    
    with gzip.open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        batch = []
        for i, line in enumerate(ifd):
            item = json.loads(line)
            loc = Location(item["location"])
            batch.append((loc, item["text"]))
            if len(batch) == args.batch_size:
                encs = model.encode([x for _, x in batch], show_progress_bar=False)

                # encoded = tokenizer([t for _, t in batch], return_tensors="pt", padding=True)
                # encoded.to(args.device)
                # out = model.generate(
                #     **encoded,
                #     forced_bos_token_id=tokenizer.lang_code_to_id[args.language],
                #     return_dict_in_generate=True,
                #     output_hidden_states=True
                # )
                #for (loc, text), emb, mask in zip(batch, out["encoder_hidden_states"][-1], encoded["attention_mask"]):
                #     emb = emb[mask==1].sum(0) / emb.shape[0]
                assert len(batch) == encs.shape[0]
                for (loc, text), emb in zip(batch, encs):
                    ofd.write(json.dumps({"location" : loc, "embedding" : emb.tolist()}) + "\n")
                batch = []
                
                logger.info("Processed %d sentences", i + 1)
                #print(encs)
                #sys.exit()
                
        if len(batch) > 0:
            encs = model.encode([x for _, x in batch], show_progress_bar=False)
            #encoded = tokenizer([t for _, t in batch], return_tensors="pt", padding=True)
            #encoded.to(args.device)
            #out = model.generate(
            #    **encoded,
            #    forced_bos_token_id=tokenizer.lang_code_to_id[args.language],
            #    return_dict_in_generate=True,
            #    output_hidden_states=True
            #)
            #for (loc, text), emb, mask in zip(batch, out["encoder_hidden_states"][-1], encoded["attention_mask"]):
            #    emb = emb[mask==1].sum(0) / emb.shape[0]
            for (loc, text), emb in zip(batch, encs):
                ofd.write(json.dumps({"location" : loc, "embedding" : emb.tolist()}) + "\n")
            logger.info("Processed %d sentences", i + 1 + len(batch))
