import logging
import gzip
import json
import argparse
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("embed_document")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--language", dest="language", default="en", help="Language code")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--device", dest="device", default="cpu")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=500)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    model = SentenceTransformer('sentence-transformers/LaBSE', device=args.device)
    model.to(args.device)
    with gzip.open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        batch = []
        for i, line in enumerate(ifd):
            item = json.loads(line)
            batch.append((item["id"], item["text"]))
            if len(batch) == args.batch_size:
                embeddings = model.encode([s for _, s in batch])
                for (label, text), emb in zip(batch, embeddings):
                    ofd.write(json.dumps({"id" : label, "embedding" : emb.tolist()}) + "\n")
                batch = []
                logger.info("Processed %d sentences", i)
        if len(batch) > 0:
            embeddings = model.encode([s for _, s in batch])
            for (label, text), emb in zip(batch, embeddings):
                ofd.write(json.dumps({"id" : label, "embedding" : emb.tolist()}) + "\n")
