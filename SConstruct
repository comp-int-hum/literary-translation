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
import sys
import json
from steamroller import Environment

# I'm not running this on the grid, so remove everything to do with clusters,
# need to download the data, it's pretty easy to find where it is
# grb.tsv is on your repo, can download it directly
# STEP for OT
# we will have to face the misalignment issues in the JHUBC somehow with a note that future work will include it
# hopefully Hale has done some looking into that 

# workaround needed to fix bug with SCons and the pickle module

# actual variable and environment objects
vars = Variables()

vars.AddVariables(
    ("ENGINE", "", "slurm"),
    ("GPU_ACCOUNT", "", "CAINES-SL3-GPU"),
    ("GPU_QUEUE", "", "ampere"),
    # ("DATA_PATH", "", os.path.expanduser("~/corpora")),
    ("DATA_PATH", "", "data"),
    # ("BIBLE_CORPUS", "", os.path.expanduser("${DATA_PATH}/bible-corpus")),
    ("BIBLE_CORPUS", "", "${DATA_PATH}/bibles2"),
    # ("CROSS_REFERENCE_FILE", "", os.path.expanduser("~/corpora/biblical-cross-references.txt")),
    ("CROSS_REFERENCE_FILE", "", "${DATA_PATH}/biblical-cross-references.txt"),
    # ("STEP_BIBLE_PATH", "", os.path.expanduser("~/corpora/STEPBible-Data/Translators Amalgamated OT+NT")),
    ("STEP_BIBLE_PATH", "", "${DATA_PATH}/STEP"),
    ("DEVICE", "", "cuda"),
    ("BATCH_SIZE", "", 50),
    ("VOTE_THRESHOLD", "", 50),
    (
        "LANGUAGE_MAP",
        "",
        {
            "Hebrew" : ("he_IL", "heb_Hebr", "he"),
            "Greek" : ("el_XX", "ell_Grek", "el"),
            "English" : ("en_XX", "eng_Latn", "en"),
            #"Japanese" : ("ja_XX", "jpn_Jpan"),
            "Finnish" : ("fi_FI", "fin_Latn", "fi"),
            "Turkish" : ("tr_TR", "tur_Latn", "tr"),
            "Swedish" : ("sv_SE", "swe_Latn", "sv"),
            "Marathi" : ("mr_IN", "mar_Deva", "mr"),
        }
    ),
    (
        "ORIGINALS",
        "",
        [
            {"testament" : "OT", "language" : "Greek", "form" : "Septuagint", "file" : "${DATA_PATH}/LXX/LXX_aligned.json", "manuscript" : "LXX"},
            {"testament" : "NT", "language" : "Greek", "form" : "Septuagint", "file" : "${STEP_BIBLE_PATH}/NT_aligned.json", "manuscript" : "TAGNT"},
            {"testament" : "OT", "language" : "Hebrew", "form" : "STEP", "file" : "${STEP_BIBLE_PATH}/OT_aligned.json", "manuscript" : "TAHOT"},
        ]
    ),
    ("TRANSLATION_LANGUAGES", "", ["English", ]),#"Finnish", "Turkish", "Swedish", "Marathi"]), # can remove Japenese "Japanese",
    ("USE_PRECOMPUTED_EMBEDDINGS", "", False),
    ("MODELS", "", ["facebook/m2m100_1.2B"]),#"facebook/m2m100_1.2B",]), #"CohereForAI/aya-23-8B"]), #"facebook/nllb-200-distilled-600M", 
)

env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[],
    BUILDERS={
        "ConvertFromXML" : Builder(
            action="python scripts/convert_from_xml.py --testament ${TESTAMENT} --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        # "ConvertFromWLC" : Builder(
        #     action="python scripts/convert_from_wlc.py --testament ${TESTAMENT} --input ${SOURCES[0]} --output ${TARGETS[0]}"
        # ),
        "ConvertFromSTEP" : Builder(
            action="python scripts/convert_from_aligned_step.py --input ${SOURCE} --output ${TARGET} --lang ${LANGUAGE}"
        ),        
        # "ConvertFromSeptuagint" : Builder(
        #     action="python scripts/convert_from_septuagint.py --testament ${TESTAMENT} --input ${SOURCES[0]} --output ${TARGETS[0]}"
        # ),        
        "EmbedDocument" : Builder(
            action="python scripts/embed_document.py --input ${SOURCES[0]} --device ${DEVICE} --output ${TARGETS[0]} --batch_size ${BATCH_SIZE}"
        ),
        "TranslateDocument" : Builder(
            action="python scripts/translate_document.py --input ${SOURCES[0]} --output ${TARGETS[0]} --model ${MODEL} --source_lang ${SRC_LANG} --target_lang ${TGT_LANG} --device ${DEVICE} --batch_size ${BATCH_SIZE} ${'--disallow_referenced ' + DISALLOW_REFERENCED if DISALLOW_REFERENCED else ''} ${'--vote_threshold ' + str(VOTE_THRESHOLD) if VOTE_THRESHOLD else ''} ${'--disallow_target ' + SOURCES[1].rstr() if len(SOURCES) == 2 else ''}"
        ),
        # "ScoreEmbeddings" : Builder(
        #     action="python scripts/score_embeddings.py --gold ${SOURCES[0]} --embeddings ${SOURCES[1]} --output ${TARGETS[0]} --vote_threshold ${VOTE_THRESHOLD} --testament ${TESTAMENT} --language ${LANGUAGE} --condition ${CONDITION_NAME} --random_seed ${RANDOM_SEED} --manuscript ${MANUSCRIPT}"
        # ),
        # "SummarizeScores" : Builder(
        #     action="python scripts/summarize_scores.py --output ${TARGETS[0]} --inputs ${SOURCES}"
        # ),
    }
)

# how we decide if a dependency is out of date
env.Decider("timestamp-newer")

lang_map = env["LANGUAGE_MAP"]
r_lang_map = {v : k for k, v in lang_map.items()}

# HERE WE ARE EMBEDDING HUMAN TRANSLATIONS
embeddings = {}
human_translations = {}
for testament in ["OT", "NT"]:
    embeddings[testament] = embeddings.get(testament, {})
    human_translations[testament] = human_translations.get(testament, {})
    
    for language in env["TRANSLATION_LANGUAGES"]:
        manuscript = "JHUBC"
        condition_name = "human_translation"
        embeddings[testament][manuscript] = embeddings[testament].get(manuscript, {})
        embeddings[testament][manuscript][language] = embeddings[testament][manuscript].get(language, {})
 

        renv = env.Override(
            {
                "TESTAMENT" : testament,
                "LANGUAGE" : language,
                "CONDITION_NAME" : condition_name,
                "MANUSCRIPT" : manuscript,
            }
        )

        if env["USE_PRECOMPUTED_EMBEDDINGS"]:
            emb = renv.File(renv.subst("work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}/${MODEL}-embedded.json.gz"))
            human_trans = None
        else:
            human_trans = renv.ConvertFromXML(
                "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}.json.gz",
                "${BIBLE_CORPUS}/${LANGUAGE}.xml",
            )
#             emb = renv.EmbedDocument(
#                 "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz",
#                 human_trans,
#                 DEVICE="cuda"                
#             )[0]
        human_translations[testament][language] = human_trans
#         embeddings[testament][manuscript][language][condition_name] = emb

# HERE WE EMBED THE ORIGINAL MANUSCRIPTS, THEN PRODUCE TRANSLATIONS AND EMBED THOSE
for original in env["ORIGINALS"]:
    condition_name = "original"
    original_language = original["language"]
    original_file = original["file"]
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

        orig = renv.ConvertFromSTEP(
                "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}.json.gz",
                original_file,
                LANGUAGE=original_language
            )
        
        #orig_emb = renv.EmbedDocument(
        #    "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz",
        #    orig,
        #    BATCH_SIZE=1000,
        #    DEVICE="cuda"            
        #)[0]
    #embeddings[testament][manuscript][original_language][condition_name] = orig_emb
    #exit()
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
        src_lang = env["LANGUAGE_MAP"][original_language][2]
        tgt_lang = env["LANGUAGE_MAP"][other_language][2]        
        human_trans = human_translations[testament][other_language]
        
        for model in env["MODELS"]:
            #print(model)
            model_name = model.replace('/', '_')
            embeddings[testament][manuscript][other_language][model_name] = embeddings[testament][manuscript][other_language].get(model_name, {})
            for condition_name, inputs, args in [
                    ("unconstrained", orig, {}),
                    ("exclude_human", [orig, human_trans], {}),
                    ("exclude_references", orig, {"DISALLOW_REFERENCED" : env["CROSS_REFERENCE_FILE"]}),
                    #("exclude_both", [orig, human_trans], {"DISALLOW_REFERENCED" : env["CROSS_REFERENCE_FILE"]}),
            ][:1]:
                tenv = renv.Override(
                    {
                        "TESTAMENT" : testament,
                        "LANGUAGE" : other_language,
                        "CONDITION_NAME" : condition_name,
                        "MANUSCRIPT" : manuscript,
                        "MODEL": model_name,
                    }
                )
                if env["USE_PRECOMPUTED_EMBEDDINGS"]:
                    emb = tenv.File(
                        renv.subst(
                            "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${MODEL}/${LANGUAGE}-embedded.json.gz"
                        )
                    )
                else: 
                    
                    translation = tenv.TranslateDocument(
                        "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${MODEL}/${LANGUAGE}.json.gz",
                        inputs,
                        SRC_LANG=src_lang,
                        TGT_LANG=tgt_lang,
                        BATCH_SIZE=1000,
                        DEVICE="cuda",
                        MODEL=model,
                        STEAMROLLER_ENGINE=env.get("ENGINE", "local"),
                        STEAMROLLER_ACCOUNT=env.get("GPU_ACCOUNT", None),
                        STEAMROLLER_QUEUE=env.get("GPU_QUEUE", None),
                        STEAMROLLER_GPU_COUNT=1,
                        STEAMROLLER_MEMORY="32G",
                        STEAMROLLER_TIME="00:10:00",
                        **args
                    )

                    # emb = tenv.EmbedDocument(
                    #     "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${MODEL}/${LANGUAGE}-embedded.json.gz",
                    #     translation,
                    #     BATCH_SIZE=1000,
                    #     DEVICE="cuda"                    
                    # )[0]
                #embeddings[testament][manuscript][other_language][condition_name][model_name] = emb

