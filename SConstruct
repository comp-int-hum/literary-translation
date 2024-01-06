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
    ("BATCH_SIZE", "", 16),
    ("VOTE_THRESHOLD", "", 50),
    (
        "LANGUAGE_MAP",
        "",
        {
            "English" : "en_XX",
            "French" : "fr_XX",            
            "Turkish" : "tr_TR",
            "Swedish" : "sv_SE",
        }
    ),
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
            action="python scripts/translate_document.py --input ${SOURCES[0]} --output ${TARGETS[0]} --source_lang ${SRC_LANG} --target_lang ${TGT_LANG} --device ${DEVICE} --batch_size ${BATCH_SIZE} ${'--disallow_target ' + DISALLOW_TARGET if DISALLOW_TARGET else ''} ${'--disallow_referenced ' + DISALLOW_REFERENCED if DISALLOW_REFERENCED else ''} --vote_threshold ${VOTE_THRESHOLD}"
        ),
        "ScoreEmbeddings" : Builder(
            action="python scripts/score_embeddings.py --gold ${SOURCES[0]} --embeddings ${SOURCES[1]} --output ${TARGETS[0]} --vote_threshold ${VOTE_THRESHOLD}"
        ),
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

lang_map = env["LANGUAGE_MAP"]
r_lang_map = {v : k for k, v in lang_map.items()}

for fname in glob(os.path.join(env["BIBLE_CORPUS"], "bibles", "*xml")):
    base, _ = os.path.splitext(os.path.basename(fname))
    if base in lang_map:
        lang = lang_map[base]
        orig = env.ConvertFromXML(
            "work/{}.json.gz".format(base),
            fname
        )
        emb = env.EmbedDocument(
            "work/{}-embedded.json.gz".format(base),
            orig,
            LANG=lang
        )
        score = env.ScoreEmbeddings(
           "work/{}-score.json".format(base),
           [env["CROSS_REFERENCE_FILE"], emb]
        )
        for other_lang in set([v for k, v in lang_map.items() if v != lang]):
            trans = env.TranslateDocument(
                "work/{}-{}.json.gz".format(base, other_lang),
                [
                    orig
                ],
                SRC_LANG=lang,
                TGT_LANG=other_lang,
                BATCH_SIZE=16
            )
            excl_targ_trans = env.TranslateDocument(
                "work/{}-excl-targ-{}.json.gz".format(base, other_lang),
                [
                    orig                    
                ],
                SRC_LANG=lang,
                TGT_LANG=other_lang,
                BATCH_SIZE=16,
                DISALLOW_TARGET="work/{}.json.gz".format(r_lang_map[other_lang])
            )
            env.Depends(excl_targ_trans, "work/{}.json.gz".format(r_lang_map[other_lang]))
            excl_ref_trans = env.TranslateDocument(
                "work/{}-excl-ref-{}.json.gz".format(base, other_lang),
                [
                    orig          
                ],
                SRC_LANG=lang,
                TGT_LANG=other_lang,
                BATCH_SIZE=16,
                DISALLOW_REFERENCED=env["CROSS_REFERENCE_FILE"]
            )
            env.Depends(excl_ref_trans, env["CROSS_REFERENCE_FILE"])
            excl_both_trans = env.TranslateDocument(
                "work/{}-excl-both-{}.json.gz".format(base, other_lang),
                [
                    orig          
                ],
                SRC_LANG=lang,
                TGT_LANG=other_lang,
                BATCH_SIZE=16,
                DISALLOW_TARGET="work/{}.json.gz".format(r_lang_map[other_lang]),                
                DISALLOW_REFERENCED=env["CROSS_REFERENCE_FILE"]
            )
            env.Depends(excl_both_trans, [env["CROSS_REFERENCE_FILE"], "work/{}.json.gz".format(r_lang_map[other_lang])])
            emb = env.EmbedDocument(
                "work/{}-{}-embedded.json.gz".format(base, other_lang),
                trans,
                LANG=other_lang
            )
            excl_targ_emb = env.EmbedDocument(
                "work/{}-excl-targ-{}-embedded.json.gz".format(base, other_lang),
                excl_targ_trans,
                LANG=other_lang
            )
            excl_ref_emb = env.EmbedDocument(
                "work/{}-excl-ref-{}-embedded.json.gz".format(base, other_lang),
                excl_ref_trans,
                LANG=other_lang
            )
            excl_both_emb = env.EmbedDocument(
                "work/{}-excl-both-{}-embedded.json.gz".format(base, other_lang),
                excl_both_trans,
                LANG=other_lang
            )   
            score = env.ScoreEmbeddings(
                "work/{}-{}-score.json".format(base, other_lang),
                [env["CROSS_REFERENCE_FILE"], emb]
            )
            excl_targ_score = env.ScoreEmbeddings(
                "work/{}-excl-targ-{}-score.json".format(base, other_lang),
                [env["CROSS_REFERENCE_FILE"], excl_targ_emb]
            )
            excl_ref_score = env.ScoreEmbeddings(
                "work/{}-excl-ref-{}-score.json".format(base, other_lang),
                [env["CROSS_REFERENCE_FILE"], excl_ref_emb]
            )
            excl_both_score = env.ScoreEmbeddings(
                "work/{}-excl-both-{}-score.json".format(base, other_lang),
                [env["CROSS_REFERENCE_FILE"], excl_both_emb]
            )
