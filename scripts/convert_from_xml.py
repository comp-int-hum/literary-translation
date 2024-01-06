import gzip
import json
import xml.etree.ElementTree as et
import argparse
from utils import Location


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    loc = None
    with open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        xml = et.parse(ifd)
        for i, item in enumerate(xml.findall(".//*[@type='verse']")):
            ofd.write(json.dumps({"location" : Location(item.attrib["id"]), "text" : item.text.strip()}) + "\n")

