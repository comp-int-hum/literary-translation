import re

book2id = {k : i for i, k in enumerate("""GEN
EXO
LEV
NUM
DEU
JOS
JDG
RUT
1SA
2SA
1KI
2KI
1CH
2CH
EZR
NEH
EST
JOB
PSA
PRO
ECC
SOS
ISA
JER
LAM
EZE
DAN
HOS
JOE
AMO
OBA
JON
MIC
NAH
HAB
ZEP
HAG
ZEC
MAL
MAT
MAR
LUK
JOH
ACT
ROM
1CO
2CO
GAL
EPH
PHP
COL
1TH
2TH
1TI
2TI
TIT
PHM
HEB
JAM
1PE
2PE
1JO
2JO
3JO
JDE
REV""".split())}
id2book = {i : b for b, i in book2id.items()}

mapping = {
    "1KGS" : "1KI",
    "2KGS" : "2KI",    
    "JAS" : "JAM",
    "JUDG" : "JDG",
    "PS" : "PSA",
    "JUDE" : "JDE",
    "SONG" : "SOS",
    "PHIL" : "PHP",
    "PHLM" : "PHM",    
    "SON" : "SOS",
    "PHI" : "PHP",
    "JUD" : "JDE"
}

class Location(dict):
    def __init__(self, value):
        if isinstance(value, str):
            book, chapter, verse = re.sub(r"^b\.", "", value).upper().split(".")
            book3 = mapping.get(book, book[:3])
            assert book3 in book2id
            self["book"] = book3
            self["chapter"] = int(chapter)
            self["verse"] = int(verse)
        elif isinstance(value, (dict,)):
            for k in ["book", "chapter", "verse"]:
                self[k] = value[k]
        else:
            raise Exception("Not sure how to turn '{}' into a location".format(value))

    def __cmp__(self, other):
        a = (book2id[self["book"]], self["chapter"], self["verse"])
        b = (book2id[other["book"]], other["chapter"], other["verse"])
        return -1 if a < b else 0 if a == b else 1

    def __gt__(self, other):
        return (book2id[self["book"]], self["chapter"], self["verse"]) > (book2id[other["book"]], other["chapter"], other["verse"])

    def __le__(self, other):
        return (book2id[self["book"]], self["chapter"], self["verse"]) <= (book2id[other["book"]], other["chapter"], other["verse"])

    def __lt__(self, other):
        return (book2id[self["book"]], self["chapter"], self["verse"]) < (book2id[other["book"]], other["chapter"], other["verse"])    

    def __hash__(self):
        return hash(repr(self))

    #def to_label(s):
#    book, chapter, verse = re.sub(r"^b\.", "", s).upper().split(".")
#    book3 = mapping.get(book, book[:3])
#    if book3 not in book2id:
#        raise Exception(s)
#    return (book2id[book3], chapter, verse)
