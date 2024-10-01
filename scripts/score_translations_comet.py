from comet import download_model, load_from_checkpoint
import os
import glob
from tqdm import tqdm
import json
import pickle
import torch
import unicodedata

from utils import Bible

torch.set_float32_matmul_precision("high")

model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")

# def remove_accents(s):
#     return ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))
#
# # want to load in all the translations
# TRANSLATIONS = {}
# translation_files = glob.glob(os.path.join("translations", "*.json.gz"))
# for file in translation_files:
#     translation = Bible(file)
#     TRANSLATIONS["_".join(os.path.basename(file).split("/")).split('.json.gz')[0]] = translation
#
# originals = [k for k in TRANSLATIONS.keys() if "original" in k]
# print(TRANSLATIONS.keys())

# os.makedirs("work/comet_translations", exist_ok=True)
#
# for original in originals:
#     lang = original.split("_")[-1]
#    #  # print(original)
#     ref_bible = TRANSLATIONS[original]
#     for translation in tqdm(TRANSLATIONS.keys()):
#         if lang in translation and "Japanese" not in translation:
#             data = []
#             hyp_bible = TRANSLATIONS[translation]
#             print(f"pair: {ref_bible}    {hyp_bible}")
#             print()
#             for i in range(len(ref_bible.data)):
#                 # print(ref_bible.indices[i])
#                 # print(hyp_bible.indices[i])
#                 try:
#                     data.append({
#                         "src": remove_accents(ref_bible.indices[i]),
#                         "mt": hyp_bible.indices[i]
#                      })
#                 except:
#                     print(f"No entry found for {ref_bible.data[i]['location']} in {hyp_bible}")
#             # save it to a json lines file under the name of the translation
#             with open(f"work/comet_translations/{translation}.jsonl", "w") as ofd:
#                 for line in data:
#                     ofd.write(json.dumps(line, ensure_ascii=False) + "\n")
#
#
# exit()
model = load_from_checkpoint(model_path)
# now load the json
# for each translation, compare it to the original
files = glob.glob("work/new_comet_translations/new_comet_translations/*.jsonl")

outfiles = [f.replace('.jsonl', '.out') for f in files]

for i, file in enumerate(files):
    print(file)
    if not os.path.isfile(outfiles[i]):
        with open(file, "r") as ifd:
            data = [json.loads(line) for line in ifd]
            model_output = model.predict(data, batch_size=64, gpus=1)
            with open(f"{file.replace('.jsonl', '')}.out", "w") as ofd:
                ofd.write(json.dumps({"system_score":model_output.system_score, "num_lines": len(data), "scores": model_output.scores}))
    else:
        print(f"{outfiles[i]} exists already, skipping \n")
