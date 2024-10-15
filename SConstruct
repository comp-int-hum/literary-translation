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
#from steamroller import Environment

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
    ("GPU_ACCOUNT", "", None),
    ("GPU_QUEUE", "", None),
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
            "Hebrew" : ("he_IL", "heb_Hebr", "he", "Ancient_Hebrew"),
            "Greek" : ("el_XX", "ell_Grek", "el", "Ancient_Greek"),
            "English" : ("en_XX", "eng_Latn", "en", "English"),
            #"Japanese" : ("ja_XX", "jpn_Jpan"),
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
    ("TRANSLATION_LANGUAGES", "", ["English", "Finnish","Turkish", "Swedish", "Marathi"]), # can remove Japenese "Japanese",
    ("USE_PRECOMPUTED_EMBEDDINGS", "", False),
    ("MODELS", "",["CohereForAI/aya-23-8B"]),#"facebook/m2m100_1.2B",]), #"CohereForAI/aya-23-8B"]), #"facebook/nllb-200-distilled-600M", ['facebook/nllb-200-3.3B']),#["CohereForAI/aya-23-8B"]),#"facebook/m2m100_1.2B",]), #"CohereForAI/aya-23-8B"]), #"facebook/nllb-200-distilled-600M", 
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
        # "TranslateDocument" : Builder(
        #     action="python scripts/translate_document.py --input ${SOURCES[0]} --output ${TARGETS[0]} --model ${MODEL} --source_lang ${SRC_LANG} --target_lang ${TGT_LANG} --device ${DEVICE} --batch_size ${BATCH_SIZE} ${'--disallow_referenced ' + DISALLOW_REFERENCED if DISALLOW_REFERENCED else ''} ${'--vote_threshold ' + str(VOTE_THRESHOLD) if VOTE_THRESHOLD else ''} ${'--disallow_target ' + SOURCES[1].rstr() if len(SOURCES) == 2 else ''}"
        # ),
        "MakePrompt": Builder(
            action="python scripts/make_prompt.py --original ${SOURCES[0]} --human_translation ${SOURCES[1]} --src ${SRC_LANG} --tgt ${TGT_LANG} --output ${TARGET}"
        ),
        "TranslateDocument" : Builder(
            action="python scripts/translate_document_aya.py --prompt ${PROMPT} --input ${SOURCES[0]} --output ${TARGETS[0]} --model ${MODEL} --source_lang ${SRC_LANG} --target_lang ${TGT_LANG} --device ${DEVICE} --batch_size ${BATCH_SIZE} ${'--disallow_referenced ' + DISALLOW_REFERENCED if DISALLOW_REFERENCED else ''} ${'--vote_threshold ' + str(VOTE_THRESHOLD) if VOTE_THRESHOLD else ''} ${'--disallow_target ' + SOURCES[1].rstr() if len(SOURCES) == 2 else ''}"
        ),
        "PostProcess": Builder(
            action="python scripts/postprocess.py --inputs ${SOURCES} --output ${TARGET} --src ${SRC_LANG} --tgt ${TGT_LANG}"
        ),
        "Score": Builder(
            action="python scripts/score_translation.py --preds ${SOURCES[0]} --refs ${SOURCES[1]} --sources ${SOURCES[2]} --lang ${TGT_LANG} --ouput ${TARGET}"
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
originals ={}
translations = {}
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
            human_trans = os.path.join("work", renv['TESTAMENT'], renv['MANUSCRIPT'], renv['CONDITION_NAME'], renv['LANGUAGE'] + ".json.gz")
            # human_trans = renv.ConvertFromXML(
            #     "aya_translations/work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}.json.gz",
            #     "${BIBLE_CORPUS}/${LANGUAGE}.xml",
            # )
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
    if env["USE_PRECOMPUTED_EMBEDDINGS"]:
        orig_emb = renv.File(renv.subst("aya_translations/work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz"))
        orig = None
        human_trans = None
    else:
        orig = env.File(os.path.join("work",renv['TESTAMENT'], renv['MANUSCRIPT'],renv['CONDITION_NAME'], renv['LANGUAGE']+".json.gz"))
        # orig = renv.ConvertFromSTEP(
        #         "aya_translations/work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}.json.gz",
        #         original_file,
        #         LANGUAGE=original_language
        #     )
        originals[testament][manuscript][original_language] = orig
        #orig_emb = renv.EmbedDocument(
        #    "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${LANGUAGE}-embedded.json.gz",
        #    orig,
        #    BATCH_SIZE=1000,
        #    DEVICE="cuda"            
        #)[0]
    #embeddings[testament][manuscript][original_language][condition_name] = orig_emb
    #exit()
    for other_language in env["TRANSLATION_LANGUAGES"]:
        translations[testament] = translations.get(testament, {})
        translations[testament][manuscript] = translations[testament].get(manuscript, {})
        translations[testament][manuscript][other_language] = translations[testament][manuscript].get(other_language, {})
        embeddings[testament][manuscript][other_language] = embeddings[testament][manuscript].get(other_language, {})        
        src_lang = env["LANGUAGE_MAP"][original_language][3]
        tgt_lang = env["LANGUAGE_MAP"][other_language][3]        
        human_trans = human_translations[testament][other_language]
        # here we will make a prompt 
        prompt = env.MakePrompt(os.path.join("work", renv["TESTAMENT"], renv["MANUSCRIPT"], tgt_lang + ".txt"),
                                [orig, human_trans],
                                SRC_LANG=f"{src_lang}", 
                                TGT_LANG=tgt_lang)
        
        for model in env["MODELS"]:
            #print(model)
            model_name = model.replace('/', '_')
            embeddings[testament][manuscript][other_language][model_name] = embeddings[testament][manuscript][other_language].get(model_name, {})
            translations[testament][manuscript][other_language][model_name] = translations[testament][manuscript][other_language].get(model_name, {})
            for condition_name, inputs, args in [
                    ("unconstrained", orig, {}),
                    # ("exclude_human", [orig, human_trans], {}),
                    # ("exclude_references", orig, {"DISALLOW_REFERENCED" : env["CROSS_REFERENCE_FILE"]}),
                    #("exclude_both", [orig, human_trans], {"DISALLOW_REFERENCED" : env["CROSS_REFERENCE_FILE"]}),
            ]:
                tenv = renv.Override(
                    {
                        "TESTAMENT" : testament,
                        "LANGUAGE" : other_language,
                        "CONDITION_NAME" : condition_name,
                        "MANUSCRIPT" : manuscript,
                        "MODEL": model,
                    }
                )
                if env["USE_PRECOMPUTED_EMBEDDINGS"]:
                    emb = tenv.File(
                        renv.subst(
                            "aya_translations/work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${MODEL}/${LANGUAGE}-embedded.json.gz"
                        )
                    )
                else:
                    if manuscript != "TAGNT-Greek":
                        translation = tenv.TranslateDocument(
                        "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${MODEL}/${LANGUAGE}.json.gz",
                        inputs,
                        PROMPT=prompt,
                        SRC_LANG=src_lang,
                        TGT_LANG=tgt_lang,
                        BATCH_SIZE=20,
                        DEVICE="cuda",
                        MODEL=model,
                        **args
                    )
                    env.Depends(translation, prompt)
                    #translation = env.File(f"aya_translations/work/{tenv['TESTAMENT']}/{tenv['MANUSCRIPT']}/{tenv['CONDITION_NAME']}/{tenv['MODEL']}/{tenv['LANGUAGE']}.json.gz",)
                    translations[testament][manuscript][other_language][model_name][condition_name] = translation
                    # emb = tenv.EmbedDocument(
                    #     "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${MODEL}/${LANGUAGE}-embedded.json.gz",
                    #     translation,
                    #     BATCH_SIZE=1000,
                    #     DEVICE="cuda"                    
                    # )[0]
                #embeddings[testament][manuscript][other_language][condition_name][model_name] = emb

#print(originals)
# for testament in ["NT", "OT"]:
#     for manuscript in ["LXX-Greek", "TAHOT-Hebrew", "TAGNT-Greek"]:
#         try:
#             original_language = manuscript.split('-')[1]
#             original = originals[testament][manuscript][original_language]
#             #print(f"original is {original}")
#             for other_language in env["TRANSLATION_LANGUAGES"]:
#                 human_translation = human_translations[testament][other_language] # e.g. [NT][Swedish]
#                 #print(f"human translation is {human_translation}")
#                 src_lang = env["LANGUAGE_MAP"][original_language][3] #e.g. Ancient Greek
#                 tgt_lang = env["LANGUAGE_MAP"][other_language][3] #e.g. Finnish
#                 for model in ['CohereForAI/aya-23-8B']:#(except this will always be aya)
#                     model_name = model.replace('/', '_')
#                     for condition_name in ["unconstrained", "exclude_human", "exclude_references"]:
#                         dummy_out = translations[testament][manuscript][other_language][model_name][condition_name]
#                         #print(f"machine translation is {dummy_out.get_dir()}")
#                         #print(dummy_out.get_path())
#                         file_patterns = os.path.join(os.path.dirname(dummy_out.get_path()), f'{other_language}*_[4567]')
#                         #os.listdir(os.path.dirname(dummy_out.get_path()))
#                         # Get all matching files
#                         matching_files = glob(file_patterns)
#                         if len(matching_files) > 0:
#                         #print(matching_files)
#                         # print(matching_files)
#                             env.PostProcess(os.path.join("aya_translations", "work", testament, manuscript, condition_name, model, other_language+'.json'), 
#                                         matching_files, 
#                                         SRC_LANG=src_lang,
#                                         TGT_LANG=tgt_lang)
#         except KeyError:
#             continue

                                # those are inputs
                        # target is "work/${TESTAMENT}/${MANUSCRIPT}/${CONDITION_NAME}/${MODEL}/${LANGUAGE}.json"
                        #combined_trans = postprocess target, inputs, src, tgt
                        # then we score it
                        # score translation (dw about making a fancy table script, just make it yourself)
                        # preds is combined_trans, refs is human_translation, source is original. sweet

