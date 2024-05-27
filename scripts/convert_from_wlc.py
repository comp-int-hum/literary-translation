import re
import gzip
import json
import xml.etree.ElementTree as et
import tarfile
import argparse
from utils import Location


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--testament", dest="testament", choices=["new", "old", "both"], default="both")
    args = parser.parse_args()

    loc = None
    with tarfile.open(args.input, "r") as tifd, gzip.open(args.output, "wt") as ofd:
        for member in tifd.getmembers():
            if member.isfile():
                with tifd.extractfile(member) as ifd:
                    xml = et.parse(ifd)
                    abbr = xml.find(".//*abbrev").text
                    for chap in xml.findall(".//*c"):
                        ch = int(chap.get("n"))
                        v = 1
                        for verse in chap.findall("v"):
                            v = int(verse.get("n"))
                            text = re.sub(r"\s+", " ", " ".join(verse.itertext()).strip())
                            loc = Location(
                                {
                                    "book" : abbr,
                                    "chapter" : ch,
                                    "verse" : v
                                }
                            )
                            if args.testament == "both" or loc.testament() == args.testament:
                                ofd.write(json.dumps({"location" : loc, "text" : text.strip()}) + "\n")
