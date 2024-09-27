import gzip
import json
import xml.etree.ElementTree as et
import argparse
from utils import Location


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--testament", dest="testament", choices=["NT", "OT", "both"], default="both")
    args = parser.parse_args()

    loc = None
    with open(args.input, "rt") as ifd, open(args.output, "wt") as ofd:
        xml = et.parse(ifd)
        for i, item in enumerate(xml.findall(".//*[@type='verse']")):
            loc = Location(item.attrib["id"])
            if args.testament == "both" or loc.testament() == args.testament:
                ofd.write(json.dumps({"location" : loc, "text" : item.text.strip()}, ensure_ascii=False) + "\n")

