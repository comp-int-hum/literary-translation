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

    model = SentenceTransformer('sentence-transformers/LaBSE', device=args.device)
    

    model.to(args.device)
    
    with open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        batch = []
        for i, line in enumerate(ifd):
            item = json.loads(line)
            loc = Location(item["location"])
            if item["text"]== None:
                continue
            batch.append((loc, item["text"]))
            
            if len(batch) == args.batch_size:
                encs = model.encode([x for _, x in batch], show_progress_bar=True)

                assert len(batch) == encs.shape[0]
                for (loc, text), emb in zip(batch, encs):
                    ofd.write(json.dumps({"location" : loc, "embedding" : emb.tolist()}) + "\n")
                batch = []
                
                logger.info("Processed %d sentences", i + 1)
        
                
        if len(batch) > 0:
            encs = model.encode([x for _, x in batch], show_progress_bar=True)
            
            for (loc, text), emb in zip(batch, encs):
                ofd.write(json.dumps({"location" : loc, "embedding" : emb.tolist()}) + "\n")
            logger.info("Processed %d sentences", i + 1 + len(batch))
