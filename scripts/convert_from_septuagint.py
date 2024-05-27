import gzip
import json
import xml.etree.ElementTree as et
import argparse
from utils import Location


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--testament", dest="testament", choices=["new", "old", "both"], default="both")
    args = parser.parse_args()

    loc = None
    with gzip.open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        #xml = et.parse(ifd)
        #for i, item in enumerate(xml.findall(".//*[@type='verse']")):
        for line in ifd:
            try:
                book, abbrev, bn, ch, v, txt = line.strip().split("\t")
            except:
                book, abbrev, temp, v, txt = line.strip().split("\t")
                bn = temp[:2]
                ch = temp[2:]
            if "(" not in book and "Esdr" not in book and "Judith" not in book and "Tobit" not in book and "Maccabees" not in book and "Odes" not in book and "Wisdom" not in book and "Sirach" not in book and "Psalms of Solomon" not in book and "Baruch" not in book and "Letter of Jeremiah" not in book and "Sussana" not in book and "Dragon" not in book:
                try:
                    loc = Location({"book" : abbrev, "chapter" : int(ch), "verse" : int(v)}) #item.attrib["id"])
                except ValueError:
                    vA, vB = v.split("-")
                    vA = int(vA)
                    vB = int(vB)
                    loc = Location({"book" : abbrev, "chapter" : int(ch), "verse" : int(vA)}) #item.attrib["id"])
                if args.testament == "both" or loc.testament() == args.testament:
                    ofd.write(json.dumps({"location" : loc, "text" : txt.strip()}) + "\n")

