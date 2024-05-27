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
    ("DATA_PATH", "", os.path.expanduser("~/corpora")),
    ("BIBLE_CORPUS", "", os.path.expanduser("${DATA_PATH}/bible-corpus")),
    ("CROSS_REFERENCE_FILE", "", os.path.expanduser("~/corpora/biblical-cross-references.txt")),
    ("STEP_BIBLE_PATH", "", os.path.expanduser("~/corpora/STEPBible-Data/Translators Amalgamated OT+NT")),
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
    (
        "ORIGINALS",
        "",
        [
            {"testament" : "old", "language" : "Greek", "form" : "Septuagint", "file" : "${DATA_PATH}/grb.tsv.gz", "manuscript" : "Sinaiticus"},
            {"testament" : "new", "language" : "Greek", "form" : "Septuagint", "file" : "${DATA_PATH}/grb.tsv.gz", "manuscript" : "Sinaiticus"},
            {"testament" : "old", "language" : "Hebrew", "form" : "STEP", "file" : "${STEP_BIBLE_PATH}", "manuscript" : "WLC"},
        ]
    ),
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
        "ConvertFromWLC" : Builder(
            action="python scripts/convert_from_wlc.py --testament ${TESTAMENT} --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        "ConvertFromSTEP" : Builder(
            action="python scripts/convert_from_step.py --testament ${TESTAMENT} --inputs ${SOURCES} --output ${TARGETS[0]}"
        ),        
        "ConvertFromSeptuagint" : Builder(
            action="python scripts/convert_from_septuagint.py --testament ${TESTAMENT} --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),        
        "EmbedDocument" : Builder(
            action="python scripts/embed_document.py --input ${SOURCES[0]} --device ${DEVICE} --output ${TARGETS[0]} --batch_size ${BATCH_SIZE}"
        ),
        "TranslateDocument" : Builder(
            action="python scripts/translate_document.py --input ${SOURCES[0]} --output ${TARGETS[0]} --source_lang ${SRC_LANG} --target_lang ${TGT_LANG} --device ${DEVICE} --batch_size ${BATCH_SIZE} ${'--disallow_referenced ' + DISALLOW_REFERENCED if DISALLOW_REFERENCED else ''} ${'--vote_threshold ' + str(VOTE_THRESHOLD) if VOTE_THRESHOLD else ''} ${'--disallow_target ' + SOURCES[1].rstr() if len(SOURCES) == 2 else ''}"
        ),
        "ScoreEmbeddings" : Builder(
            action="python scripts/score_embeddings.py --gold ${SOURCES[0]} --embeddings ${SOURCES[1]} --output ${TARGETS[0]} --vote_threshold ${VOTE_THRESHOLD} --testament ${TESTAMENT} --language ${LANGUAGE} --condition ${CONDITION_NAME} --random_seed ${RANDOM_SEED} --manuscript ${MANUSCRIPT}"
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
human_translations = {}
for testament in ["old", "new"]:
    embeddings[testament] = embeddings.get(testament, {})
    human_translations[testament] = human_translations.get(testament, {})
    for language in env["TRANSLATION_LANGUAGES"]:
        manuscript = "BC"
        condition_name = "human_translation"
        embeddings[testament][manuscript] = embeddings[testament].get(manuscript, {})
        embeddings[testament][manuscript][language] = embeddings[testament][manuscript].get(language, {})
        renv = env.Override(
            {
                "TESTAMENT" : testament,
                "LANGUAGE" : language,
                "CONDITION_NAME" : condition_name,
                "MANUSCRIPT" : manuscript
            }
        )
        if env["USE_PRECOMPUTED_EMBEDDINGS"]:
            emb = renv.File(renv.subst("work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz"))
            human_trans = None
        else:
            human_trans = renv.ConvertFromXML(
                "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}.json.gz",
                "${BIBLE_CORPUS}/bibles/${LANGUAGE}.xml",
            )
            emb = renv.EmbedDocument(
                "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz",
                human_trans,
                STEAMROLLER_ACCOUNT=env.get("GPU_ACCOUNT", None),
                STEAMROLLER_QUEUE=env.get("GPU_QUEUE", None),
                BATCH_SIZE=1000,
                STEAMROLLER_GPU_COUNT=1,
                STEAMROLLER_MEMORY="32G",
                STEAMROLLER_TIME="02:00:00",
                DEVICE="cuda"                
            )[0]
        human_translations[testament][language] = human_trans
        embeddings[testament][manuscript][language][condition_name] = emb
        



for original in env["ORIGINALS"]:
    condition_name = "original"
    original_language = original["language"]
    testament = original["testament"]
    manuscript = "{}-{}".format(original["manuscript"], original_language)
    embeddings[testament] = embeddings.get(testament, {})
    embeddings[testament][manuscript] = embeddings[testament].get(manuscript, {})
    embeddings[testament][manuscript][original_language] = embeddings[testament][manuscript].get(original_language, {})
    renv = env.Override(
        {
            "TESTAMENT" : testament,
            "LANGUAGE" : original_language,
            "CONDITION_NAME" : condition_name,
            "MANUSCRIPT" : manuscript
        }
    )
    if env["USE_PRECOMPUTED_EMBEDDINGS"]:
        orig_emb = renv.File(renv.subst("work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz"))
        orig = None
        human_trans = None
    else:
        try:
            f = env.File(original["file"])
        except:
            f = env.Dir(original["file"]).glob("*")
            #print(f)
            #print(env.Glob(d))
        orig = getattr(renv, "ConvertFrom" + original["form"])(
            "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}.json.gz",
            f
        )
        orig_emb = renv.EmbedDocument(
            "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz",
            orig,
            STEAMROLLER_ACCOUNT=env.get("GPU_ACCOUNT", None),
            STEAMROLLER_QUEUE=env.get("GPU_QUEUE", None),
            BATCH_SIZE=1000,
            STEAMROLLER_GPU_COUNT=1,
            STEAMROLLER_MEMORY="32G",
            STEAMROLLER_TIME="02:00:00",
            DEVICE="cuda"            
        )[0]
    embeddings[testament][manuscript][original_language][condition_name] = orig_emb

    for other_language in env["TRANSLATION_LANGUAGES"]:
        embeddings[testament][manuscript][other_language] = embeddings[testament][manuscript].get(other_language, {})        
        # if "human_translation" not in embeddings[testament][manuscript][other_language]:
        #     oenv = renv.Override(
        #         {
        #             "LANGUAGE" : other_language,
        #             "CONDITION_NAME" : "human_translation",
        #             "MANUSCRIPT" : "BC"#.format(original_language)
        #         }
        #     )
        #     if env["USE_PRECOMPUTED_EMBEDDINGS"]:
        #         emb = oenv.File(renv.subst("work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz"))
        #     else:
        #         human_trans = oenv.ConvertFromXML(
        #             "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}.json.gz",
        #             "${BIBLE_CORPUS}/bibles/${LANGUAGE}.xml",
        #         )
        #         emb = oenv.EmbedDocument(
        #             "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz",
        #             human_trans,
        #             STEAMROLLER_ACCOUNT=env.get("GPU_ACCOUNT", None),
        #             STEAMROLLER_QUEUE=env.get("GPU_QUEUE", None),
        #             BATCH_SIZE=1000,
        #             STEAMROLLER_GPU_COUNT=1,
        #             STEAMROLLER_MEMORY="32G",
        #             STEAMROLLER_TIME="00:30:00",
        #             DEVICE="cuda"                
        #         )[0]
        #     embeddings[testament][manuscript][other_language]["human_translation"] = emb
        src_lang = env["LANGUAGE_MAP"][original_language][1]
        tgt_lang = env["LANGUAGE_MAP"][other_language][1]        
        human_trans = human_translations[testament][other_language]
        
        for condition_name, inputs, args in [
                ("unconstrained", orig, {}),
                ("exclude_human", [orig, human_trans], {}),
                ("exclude_references", orig, {"DISALLOW_REFERENCED" : env["CROSS_REFERENCE_FILE"]}),
                ("exclude_both", [orig, human_trans], {"DISALLOW_REFERENCED" : env["CROSS_REFERENCE_FILE"]}),
        ]:
            tenv = renv.Override(
                {
                    "TESTAMENT" : testament,
                    "LANGUAGE" : other_language,
                    "CONDITION_NAME" : condition_name,
                    "MANUSCRIPT" : manuscript
                }
            )
            if env["USE_PRECOMPUTED_EMBEDDINGS"]:
                emb = tenv.File(
                    renv.subst(
                        "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz"
                    )
                )
            else:
                translation = tenv.TranslateDocument(
                    "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}.json.gz",
                    inputs,
                    SRC_LANG=src_lang,
                    TGT_LANG=tgt_lang,
                    STEAMROLLER_ACCOUNT=env.get("GPU_ACCOUNT", None),
                    STEAMROLLER_QUEUE=env.get("GPU_QUEUE", None),
                    BATCH_SIZE=256,
                    STEAMROLLER_GPU_COUNT=1,
                    STEAMROLLER_MEMORY="32G",
                    STEAMROLLER_TIME="03:00:00",
                    DEVICE="cuda",
                    **args
                )

                emb = tenv.EmbedDocument(
                    "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz",
                    translation,
                    STEAMROLLER_ACCOUNT=env.get("GPU_ACCOUNT", None),
                    STEAMROLLER_QUEUE=env.get("GPU_QUEUE", None),
                    BATCH_SIZE=1000,
                    STEAMROLLER_GPU_COUNT=1,
                    STEAMROLLER_MEMORY="32G",
                    STEAMROLLER_TIME="02:00:00",
                    DEVICE="cuda"                    
                )[0]
            embeddings[testament][manuscript][other_language][condition_name] = emb

scores = []
for testament, manuscripts in embeddings.items():
    for manuscript, languages in manuscripts.items():
        for language, conditions in languages.items():
            for condition_name, emb in conditions.items():
                tenv = renv.Override(
                    {
                        "TESTAMENT" : testament,
                        "LANGUAGE" : language,
                        "CONDITION_NAME" : condition_name,
                        "MANUSCRIPT" : manuscript
                    }
                )
                scores.append(
                    tenv.ScoreEmbeddings(
                        "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}-score.json.gz",
                        [env["CROSS_REFERENCE_FILE"], emb],
                        RANDOM_SEED=1
                    )
                )

summary = env.SummarizeScores(
   "work/summary.tex",
   sorted(scores)
)
