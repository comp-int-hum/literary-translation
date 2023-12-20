import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
from glob import glob
import time
import imp
import sys
import json
from steamroller import Environment

# workaround needed to fix bug with SCons and the pickle module
del sys.modules['pickle']
sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
import pickle

# actual variable and environment objects
vars = Variables("custom.py")

vars.AddVariables(
    ("BIBLE_CORPUS", "", "data"),
    ("CROSS_REFERENCE_FILE", "", "data/biblical-cross-references.txt"),
    ("DEVICE", "", "cuda"),
    ("BATCH_SIZE", "", 4096)
)

env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[],
    BUILDERS={
        "ConvertFromXML" : Builder(
            action="python scripts/convert_from_xml.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        "EmbedDocument" : Builder(
            action="python scripts/embed_document.py --input ${SOURCES[0]} --language ${LANG} --device ${DEVICE} --output ${TARGETS[0]} --batch_size ${BATCH_SIZE}"
        ),
        "TranslateDocument" : Builder(
            action="python scripts/translate_document.py --input ${SOURCES[0]} --output ${TARGETS[0]} --source_lang ${SRC_LANG} --target_lang ${TGT_LANG} --device ${DEVICE} --batch_size ${BATCH_SIZE}"
        ),
        "ScoreEmbeddings" : Builder(
            action="python scripts/score_embeddings.py --gold ${SOURCES[0]} --embeddings ${SOURCES[1]} --output ${TARGETS[0]}"
        )
    }
)

# function for width-aware printing of commands
#def print_cmd_line(s, target, source, env):
#    if len(s) > int(env["OUTPUT_WIDTH"]):
#        print(s[:int(float(env["OUTPUT_WIDTH"]) / 2) - 2] + "..." + s[-int(float(env["OUTPUT_WIDTH"]) / 2) + 1:])
#    else:
#        print(s)

# and the command-printing function
#env['PRINT_CMD_LINE_FUNC'] = print_cmd_line

# and how we decide if a dependency is out of date
env.Decider("timestamp-newer")

lang_map = {
    "English" : "en",
    #"Hebrew" : "he",
    #"Latin" : "la",
    #"Greek" : "el",
    "French" : "fr",
}

for fname in glob(os.path.join(env["BIBLE_CORPUS"], "bibles", "*xml")):
    base, _ = os.path.splitext(os.path.basename(fname))
    if base in lang_map:
        orig = env.ConvertFromXML(
            "work/{}.json.gz".format(base),
            fname
        )
        emb = env.EmbedDocument(
            "work/{}_embedded.json.gz".format(base),
            orig,
            LANG=lang_map[base]
        )
        score = env.ScoreEmbeddings(
            "work/{}_score.json".format(base),
            [env["CROSS_REFERENCE_FILE"], emb]
        )
        if base != "English":
            lang = "en"
            trans = env.TranslateDocument(
                "work/{}_{}.xml".format(base, lang),                
                orig,
                SRC_LANG=lang_map[base],
                TGT_LANG=lang,
                BATCH_SIZE=64
            )
            emb = env.EmbedDocument(
                "work/{}_{}_embedded.json.gz".format(base, lang),
                trans,
                LANG=lang
            )
