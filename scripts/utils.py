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

NT = set(
    """MAT
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
REV""".split()
    )


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


lu = {
    "EX" : "EXO",
    "DEUT" : "DEU",
    "GE" : "GEN",
    "JOSH" : "JOS",
    "JDGS" : "JDG",
    "RUTH" : "RUT",
    "1SM" : "1SA",
    "2SM" : "2SA",
    "1 KINGS" : "1KI",
    "2 KINGS" : "2KI",
    "1 SAM" : "1SA",
    "2 SAM" : "2SA",
    "JUDG" : "JDG",
    "ESTH" : "EST",
    "1CHR" : "1CH",
    "2CHR" : "2CH",
    "1 CHR" : "1CH",
    "2 CHR" : "2CH",    
    "EZRA" : "EZR",
    "PS" : "PSA",
    "PRV" : "PRO",
    "ZECH" : "ZEC",
    "PROV" : "PRO",
    "EZEK" : "EZE",
    "ECCL" : "ECC",
    "SSOL" : "SOS",
    "SONG" : "SOS",
    "AMOS" : "AMO",
    "AM" : "AMO",
    "JOEL" : "JOE",
    "OBAD" : "OBA",
    "OB" : "OBA",
    "JONAH" : "JON",
    "NAHUM" : "NAH",
    "ZEPH" : "ZEP",
    "MARK" : "MAR",
    "LUKE" : "LUK",
    "JOHN" : "JOH",
    "ACTS" : "ACT",
    "1COR" : "1CO",
    "2COR" : "2CO",    
    "PHI" : "PHP",
    "1TIM" : "1TI",
    "2TIM" : "2TI",
    "TITUS" : "TIT",
    "PHMN" : "PHM",
    "JAS" : "JAM",
    "1PET" : "1PE",
    "2PET" : "2PE",
    "1JN" : "1JO",
    "2JN" : "2JO",
    "3JN" : "3JO",
    "JUDE" : "JDE",
}

step_lookup = {
    "EZK" : "EZE",
    "JOL" : "JOE",
    "NAM" : "NAH",
    "SNG" : "SOS"
}

class Location(dict):
    def __init__(self, value):
        if isinstance(value, str):
            book, chapter, verse = re.sub(r"^b\.", "", value).upper().split(".")
            book3 = mapping.get(book, book[:3])
            assert book3 in book2id
            self["book"] = book3
            self["chapter"] = int(chapter)
            self["verse"] = int(re.sub(r"\D", "", verse))
        elif isinstance(value, (dict,)):
            for k in ["book", "chapter", "verse"]:
                if k == "book":
                    nv = lu.get(value[k].upper(), value[k].upper())
                    assert nv in book2id, nv
                    self[k] = nv
                else:
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

    def testament(self):
        return "new" if self["book"] in NT else "old"
