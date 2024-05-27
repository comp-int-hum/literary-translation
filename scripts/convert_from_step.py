import re
import gzip
import json
import xml.etree.ElementTree as et
import tarfile
import argparse
import os.path
import unicodedata as ud
from utils import Location, step_lookup, book2id


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", dest="inputs", nargs="+", help="Input path")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--testament", dest="testament", choices=["new", "old", "both"], default="both")
    args = parser.parse_args()

    data = {}
    for fname in args.inputs:
        if os.path.isfile(fname) and "Hebrew" in fname:
            with open(fname, "rt") as ifd:
                for line in ifd:
                    toks = line.split("\t")
                    ref_match = re.match(r"^(\S{3})\.(\d+)\.(\d+)\#(\d+).*", toks[0] if len(toks) > 0 else "")
                    if ref_match:
                        word = toks[1]
                        new_word = []
                        for c in word:
                            if ud.name(c).startswith("HEBREW LETTER"):
                                new_word.append(c)
                        new_word = "".join(new_word)
                        book, ch, v, w = ref_match.groups()
                        book = book.upper()
                        book = step_lookup.get(book, book)
                        ch = int(ch)
                        v = int(v)
                        w = int(w)
                        data[book] = data.get(book, {})
                        data[book][ch] = data[book].get(ch, {})
                        data[book][ch][v] = data[book][ch].get(v, {})
                        data[book][ch][v][w] = new_word
                        
    with gzip.open(args.output, "wt") as ofd:
        for book, chs in data.items():
            for ch, vs in chs.items():
                for v, ws in vs.items():
                    loc = Location({"book" : book, "chapter" : ch, "verse" : v})
                    txt = " ".join([w for i, w in sorted(list(ws.items()))])
                    if args.testament == "both" or loc.testament() == args.testament:
                        print(txt.strip())
                        ofd.write(json.dumps({"location" : loc, "text" : txt.strip()}) + "\n")
