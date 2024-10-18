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
import imp
from steamroller import Environment

# workaround needed to fix bug with SCons and the pickle module
del sys.modules['pickle']
sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
import pickle


# actual variable and environment objects
vars = Variables()

vars.AddVariables(
    ("DATA_PATH", "", "data"),
    ("WORK_DIR", "", "work"),
    ("BIBLE_CORPUS", "", "${DATA_PATH}/bibles"),
    ("CROSS_REFERENCE_FILE", "", "${DATA_PATH}/biblical-cross-references.txt"),
    ("STEP_BIBLE_PATH", "", "${DATA_PATH}/STEP"),
    ("DEVICE", "", "cuda"),
    ("BATCH_SIZE", "", 50),
    ("VOTE_THRESHOLD", "", 50),
    (
        "LANGUAGE_MAP",
        "",
        {
            "Hebrew" : ("he_IL", "heb_Hebr", "he", "Ancient_Hebrew"),
            "Greek" : ("el_XX", "ell_Grek", "el", "Ancient_Greek"),
            "English" : ("en_XX", "eng_Latn", "en", "English"),
            "Finnish" : ("fi_FI", "fin_Latn", "fi", "Finnish"),
            "Turkish" : ("tr_TR", "tur_Latn", "tr", "Turkish"),
            "Swedish" : ("sv_SE", "swe_Latn", "sv", "Swedish"),
            "Marathi" : ("mr_IN", "mar_Deva", "mr", "Marathi"),
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
    ("TRANSLATION_LANGUAGES", "", ["English", "Finnish", "Turkish", "Swedish", "Marathi"]),
    ("MODELS", "",["CohereForAI/aya-23-8B"]),
)

env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[],
    BUILDERS={
        "ConvertFromXML" : Builder(
            action="python scripts/convert_from_xml.py --testament ${TESTAMENT} --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        "ConvertFromSTEP" : Builder(
            action="python scripts/convert_from_aligned_step.py --input ${SOURCE} --output ${TARGET} --lang ${LANGUAGE}"
        ),          
        "EmbedDocument" : Builder(
            action="python scripts/embed_document.py --input ${SOURCES[0]} --device ${DEVICE} --output ${TARGETS[0]} --batch_size ${BATCH_SIZE}"
        ),
        "MakePrompt": Builder(
            action="python scripts/make_prompt.py --original ${SOURCES[0]} --human_translation ${SOURCES[1]} --src ${SRC_LANG} --tgt ${TGT_LANG} --output ${TARGET}"
        ),
        "TranslateDocument" : Builder(
            action="python scripts/translate_document_aya.py --prompt ${PROMPT} --input ${SOURCES[0]} --output ${TARGETS[0]} --model ${MODEL} --source_lang ${SRC_LANG} --target_lang ${TGT_LANG} --device ${DEVICE} --batch_size ${BATCH_SIZE} ${'--disallow_referenced ' + DISALLOW_REFERENCED if DISALLOW_REFERENCED else ''} ${'--vote_threshold ' + str(VOTE_THRESHOLD) if VOTE_THRESHOLD else ''} ${'--disallow_target ' + SOURCES[1].rstr() if len(SOURCES) == 2 else ''}"
        ),
        "PostProcess": Builder(
            action="python scripts/postprocess.py --inputs ${SOURCES} --output ${TARGET} --src ${SRC_LANG} --tgt ${TGT_LANG}"
        ),
        "ScoreTranslation": Builder(
            action="python scripts/score_translation.py --preds ${SOURCES[0]} --refs ${SOURCES[1]} --sources ${SOURCES[2]} --lang ${TGT_LANG} --output ${TARGET}"
        ),
        "InspectTranslation": Builder(
            action="python scripts/inspect_translations.py --gold ${SOURCES[0]} --embeddings ${SOURCES[1:]} --output ${TARGET}"
        ),
    }
)

# how we decide if a dependency is out of date
env.Decider("timestamp-newer")

lang_map = env["LANGUAGE_MAP"]
r_lang_map = {v : k for k, v in lang_map.items()}

# keeping track of output files for later analysis
embeddings = {}
human_translations = {}
originals ={}
mt_translations = {}

# Pre-Processing and Embedding Human Translations 
for testament in ["OT", "NT"]:
    embeddings[testament] = embeddings.get(testament, {})
    human_translations[testament] = human_translations.get(testament, {})
    
    for language in env["TRANSLATION_LANGUAGES"]:
        manuscript = "JHUBC"
        condition_name = "human_translation"
        embeddings[testament][manuscript] = embeddings[testament].get(manuscript, {})
        embeddings[testament][manuscript][language] = embeddings[testament][manuscript].get(language, {})

        # this bit is confusing, I think we need to just use
        renv = env.Override(
            {
                "TESTAMENT" : testament,
                "LANGUAGE" : language,
                "CONDITION_NAME" : condition_name,
                "MANUSCRIPT" : manuscript,
            }
        )

        human_trans = env.ConvertFromXML(
            renv.subst("${WORK_DIR}/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}.json.gz")
            "${BIBLE_CORPUS}/${LANGUAGE}.xml",
        )
        emb = env.EmbedDocument(
            renv.subst("${WORK_DIR}/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz")
            human_trans,
            BATCH_SIZE=1000,
            DEVICE="cuda"                
        )[0]

        # save the processed translation and embedding
        human_translations[testament][language] = human_trans
        embeddings[testament][manuscript][language][condition_name] = emb

# Pre-Processing and Embedding Original Documents 
for original in env["ORIGINALS"]:
    condition_name = "original"
    original_language = original["language"]
    original_file = original["file"]
    testament = original["testament"]
    manuscript = "{}-{}".format(original["manuscript"], original_language)
    
    embeddings[testament] = embeddings.get(testament, {})
    embeddings[testament][manuscript] = embeddings[testament].get(manuscript, {})
    embeddings[testament][manuscript][original_language] = embeddings[testament][manuscript].get(original_language, {})

    originals[testament] = originals.get(testament, {})
    originals[testament][manuscript] = originals[testament].get(manuscript, {})
    originals[testament][manuscript][original_language] = embeddings[testament][manuscript].get(original_language, {})

    renv = env.Override(
        {
            "TESTAMENT" : testament,
            "LANGUAGE" : original_language,
            "CONDITION_NAME" : condition_name,
            "MANUSCRIPT" : manuscript
        }
    )

    orig = renv.ConvertFromSTEP(
            renv.subst("${WORK_DIR}/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}.json.gz")
            original_file,
            LANGUAGE=original_language
        )
    originals[testament][manuscript][original_language] = orig
    
    orig_emb = renv.EmbedDocument(
        renv.subst("${WORK_DIR}/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz")
        orig,
        BATCH_SIZE=1000,
        DEVICE="cuda"            
    )[0]
    
    embeddings[testament][manuscript][original_language][condition_name] = orig_emb


    for other_language in env["TRANSLATION_LANGUAGES"]:
        mt_translations[testament] = mt_translations.get(testament, {})
        mt_translations[testament][manuscript] = mt_translations[testament].get(manuscript, {})
        mt_translations[testament][manuscript][other_language] = mt_translations[testament][manuscript].get(other_language, {})
        
        embeddings[testament][manuscript][other_language] = embeddings[testament][manuscript].get(other_language, {})        
        
        src_lang = env["LANGUAGE_MAP"][original_language][3]
        tgt_lang = env["LANGUAGE_MAP"][other_language][3]        
        
        # grab the appropriate human translation for the current testament and translation language
        human_trans = human_translations[testament][other_language]
        
        prompt = env.MakePrompt(os.path.join(env["WORK_DIR"], renv["TESTAMENT"], renv["MANUSCRIPT"], tgt_lang + ".txt"),
                                [orig, human_trans],
                                SRC_LANG=src_lang, 
                                TGT_LANG=tgt_lang)


        for model in env["MODELS"]:
            model_name = model.replace('/', '_')
            embeddings[testament][manuscript][other_language][model_name] = embeddings[testament][manuscript][other_language].get(model_name, {})
            mt_translations[testament][manuscript][other_language][model_name] = mt_translations[testament][manuscript][other_language].get(model_name, {})
            
            for condition_name, inputs, args in [
                    ("unconstrained", orig, {}),
            ]:
                embeddings[testament][manuscript][other_language][condition_name] = embeddings[testament][manuscript][other_language].get(condition_name, {})  
                
                tenv = renv.Override(
                    {
                        "TESTAMENT" : testament,
                        "LANGUAGE" : other_language,
                        "CONDITION_NAME" : condition_name,
                        "MANUSCRIPT" : manuscript,
                        "MODEL": model,
                    }
                )
                
                translation = tenv.TranslateDocument(
                    tenv.subst("${WORK_DIR}/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${MODEL}/${LANGUAGE}.json.gz")
                    inputs,
                    PROMPT=prompt,
                    SRC_LANG=src_lang,
                    TGT_LANG=tgt_lang,
                    BATCH_SIZE=15,
                    DEVICE="cuda",
                    MODEL=model,
                    **args
                )
                env.Depends(translation, prompt)
                
                mt_translations[testament][manuscript][other_language][model_name][condition_name] = translation
                    
                preds = env.PostProcess(
                    tenv.subst("${WORK_DIR}/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${MODEL}/${LANGUAGE}.json")
                    translation,
                    SRC_LANG=src_lang,
                    TGT_LANG=tgt_lang
                )
                
                # collect the bits you need for translation scoring, comet needs both human refs and sources
                human_translation = human_translations[testament][other_language]
                original = originals[testament][manuscript][original_language]

                env.ScoreTranslation(os.path.join(env["WORK_DIR"], testament, manuscript, "_".join([other_language, 'score.txt'])),
                                      [preds, human_translation, original],
                                      TGT_LANG=other_language)
                            
                    
                emb = tenv.EmbedDocument(
                    tenv.subst("${WORK_DIR}/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${MODEL}/${LANGUAGE}-embedded.json.gz")
                    preds,
                    BATCH_SIZE=1000,
                    DEVICE="cuda"                    
                    )[0]
    
                embeddings[testament][manuscript][other_language][condition_name]= emb

                ## Now we can get the INTT score  
                intt_score = env.InspectTranslation(
                    tenv.subst("${WORK}/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${MODEL}/${LANGAUGE}-score-CI.json")
                    [env["CROSS_REFERENCE_FILE"], [emb]],
                )

                


