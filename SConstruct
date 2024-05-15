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
    ("GPU_ACCOUNT", "", None),
    ("GPU_QUEUE", "", None),
    ("BIBLE_CORPUS", "", os.path.expanduser("~/corpora/bible-corpus")),
    ("CROSS_REFERENCE_FILE", "", os.path.expanduser("~/corpora/biblical-cross-references.txt")),
    ("DEVICE", "", "cpu"),
    ("BATCH_SIZE", "", 50),
    ("VOTE_THRESHOLD", "", 50),
    (
        "LANGUAGE_MAP",
        "",
        {
            "Hebrew" : ("he_IL", "heb_Hebr"),
            "Greek" : ("el_XX", "ell_Grek"),
            "English" : ("en_XX", "eng_Latn"),
            "Japanese" : ("ja_XX", "jpn_Jpan"),
            "Finnish" : ("fi_FI", "fin_Latn"),
            "Turkish" : ("tr_TR", "tur_Latn"),
            "Swedish" : ("sv_SE", "swe_Latn"),
            "Marathi" : ("mr_IN", "mar_Deva")
        }
    ),
    ("ORIGINAL_LANGUAGES", "", {"old" : "Hebrew", "new" : "Greek"}),
    ("TRANSLATION_LANGUAGES", "", ["English", "Japanese", "Finnish", "Turkish", "Swedish", "Marathi"]),
    ("USE_PRECOMPUTED_EMBEDDINGS", "", False),
)

env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[],
    BUILDERS={
        "ConvertFromXML" : Builder(
            action="python scripts/convert_from_xml.py --testament ${TESTAMENT} --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        "EmbedDocument" : Builder(
            action="python scripts/embed_document.py --input ${SOURCES[0]} --device ${DEVICE} --output ${TARGETS[0]} --batch_size ${BATCH_SIZE}"
        ),
        "TranslateDocument" : Builder(
            action="python scripts/translate_document.py --input ${SOURCES[0]} --output ${TARGETS[0]} --source_lang ${SRC_LANG} --target_lang ${TGT_LANG} --device ${DEVICE} --batch_size ${BATCH_SIZE} ${'--disallow_referenced ' + DISALLOW_REFERENCED if DISALLOW_REFERENCED else ''} ${'--vote_threshold ' + str(VOTE_THRESHOLD) if VOTE_THRESHOLD else ''} ${'--disallow_target ' + SOURCES[1].rstr() if len(SOURCES) == 2 else ''}"
        ),
        "ScoreEmbeddings" : Builder(
            action="python scripts/score_embeddings.py --gold ${SOURCES[0]} --embeddings ${SOURCES[1]} --output ${TARGETS[0]} --vote_threshold ${VOTE_THRESHOLD} --testament ${TESTAMENT} --language ${LANGUAGE} --condition ${CONDITION_NAME} --random_seed ${RANDOM_SEED}"
        ),
        "SummarizeScores" : Builder(
            action="python scripts/summarize_scores.py --output ${TARGETS[0]} --inputs ${SOURCES}"
        )
    }
)

# how we decide if a dependency is out of date
env.Decider("timestamp-newer")

lang_map = env["LANGUAGE_MAP"]
r_lang_map = {v : k for k, v in lang_map.items()}

embeddings = {}
for testament in ["old", "new"]:
    original_language = env["ORIGINAL_LANGUAGES"][testament]
    embeddings[testament] = embeddings.get(testament, {original_language : {}})
    if env["USE_PRECOMPUTED_EMBEDDINGS"]:
        renv = env.Override({"TESTAMENT" : testament, "LANGUAGE" : original_language, "CONDITION_NAME" : "original"})
        emb = env.File(renv.subst("work/${TESTAMENT}/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz"))
        orig = None
        human_trans = None
    else:
        orig = env.ConvertFromXML(
            "work/${TESTAMENT}/${CONDITION_NAME}/${LANGUAGE}.json.gz",
            "${BIBLE_CORPUS}/bibles/${LANGUAGE}.xml",
            LANGUAGE=original_language,
            TESTAMENT=testament,
            CONDITION_NAME="original"
        )        
        emb = env.EmbedDocument(
            "work/${TESTAMENT}/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz",
            orig,
            LANGUAGE=original_language,
            TESTAMENT=testament,
            CONDITION_NAME="original",
            STEAMROLLER_ACCOUNT=env.get("GPU_ACCOUNT", None),
            STEAMROLLER_QUEUE=env.get("GPU_QUEUE", None),
            BATCH_SIZE=1000,
            STEAMROLLER_GPU_COUNT=1,
            STEAMROLLER_MEMORY="32G",
            STEAMROLLER_TIME="00:30:00",
            DEVICE="cuda"            
        )[0]
    embeddings[testament][original_language]["original"] = emb
    for other_language in env["TRANSLATION_LANGUAGES"]:
        embeddings[testament][other_language] = embeddings[testament].get(other_language, {})
        if env["USE_PRECOMPUTED_EMBEDDINGS"]:
            renv = env.Override({"TESTAMENT" : testament, "LANGUAGE" : other_language, "CONDITION_NAME" : "human_translation"})
            emb = env.File(renv.subst("work/${TESTAMENT}/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz"))
        else:
            human_trans = env.ConvertFromXML(
                "work/${TESTAMENT}/${CONDITION_NAME}/${LANGUAGE}.json.gz",
                "${BIBLE_CORPUS}/bibles/${LANGUAGE}.xml",
                LANGUAGE=other_language,
                TESTAMENT=testament,
                CONDITION_NAME="human_translation"
            )
            emb = env.EmbedDocument(
                "work/${TESTAMENT}/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz",
                human_trans,
                LANGUAGE=other_language,
                TESTAMENT=testament,
                CONDITION_NAME="human_translation",
                STEAMROLLER_ACCOUNT=env.get("GPU_ACCOUNT", None),
                STEAMROLLER_QUEUE=env.get("GPU_QUEUE", None),
                BATCH_SIZE=1000,
                STEAMROLLER_GPU_COUNT=1,
                STEAMROLLER_MEMORY="32G",
                STEAMROLLER_TIME="00:30:00",
                DEVICE="cuda"                
            )[0]
        embeddings[testament][other_language]["human_translation"] = emb
        src_lang = env["LANGUAGE_MAP"][original_language][1]
        tgt_lang = env["LANGUAGE_MAP"][other_language][1]        

        for condition_name, inputs, args in [
                ("unconstrained", orig, {}),
                ("exclude_human", [orig, human_trans], {}),
                ("exclude_references", orig, {"DISALLOW_REFERENCED" : env["CROSS_REFERENCE_FILE"]}),
                ("exclude_both", [orig, human_trans], {"DISALLOW_REFERENCED" : env["CROSS_REFERENCE_FILE"]}),
        ]:
            if env["USE_PRECOMPUTED_EMBEDDINGS"]:
                renv = env.Override({"TESTAMENT" : testament, "LANGUAGE" : other_language, "CONDITION_NAME" : condition_name})
                emb = env.File(renv.subst("work/${TESTAMENT}/nllb_translations/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz"))
            else:
                translation = env.TranslateDocument(
                    "work/${TESTAMENT}/nllb_translations/${CONDITION_NAME}/${LANGUAGE}.json.gz",
                    inputs,
                    TESTAMENT=testament,
                    LANGUAGE=other_language,
                    SRC_LANG=src_lang,
                    TGT_LANG=tgt_lang,
                    CONDITION_NAME=condition_name,
                    STEAMROLLER_ACCOUNT=env.get("GPU_ACCOUNT", None),
                    STEAMROLLER_QUEUE=env.get("GPU_QUEUE", None),
                    BATCH_SIZE=512,
                    STEAMROLLER_GPU_COUNT=1,
                    STEAMROLLER_MEMORY="32G",
                    STEAMROLLER_TIME="00:30:00",
                    DEVICE="cuda",
                    **args
                )

                emb = env.EmbedDocument(
                    "work/${TESTAMENT}/nllb_translations/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz",
                    translation,
                    TESTAMENT=testament,
                    LANGUAGE=other_language,
                    CONDITION_NAME=condition_name,
                    STEAMROLLER_ACCOUNT=env.get("GPU_ACCOUNT", None),
                    STEAMROLLER_QUEUE=env.get("GPU_QUEUE", None),
                    BATCH_SIZE=1000,
                    STEAMROLLER_GPU_COUNT=1,
                    STEAMROLLER_MEMORY="32G",
                    STEAMROLLER_TIME="00:30:00",
                    DEVICE="cuda"                    
                )[0]
            embeddings[testament][other_language][condition_name] = emb

scores = []
for testament, languages in embeddings.items():
    for language, conditions in languages.items():
        for condition_name, emb in conditions.items():
            scores.append(
                env.ScoreEmbeddings(
                    "work/${TESTAMENT}/${CONDITION_NAME}/${LANGUAGE}-score.json.gz",
                    [env["CROSS_REFERENCE_FILE"], emb],
                    TESTAMENT=testament,
                    LANGUAGE=language,
                    CONDITION_NAME=condition_name,
                    RANDOM_SEED=1
                )
            )

summary = env.SummarizeScores(
    "work/summary.tex",
    sorted(scores)
)
