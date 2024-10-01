# from comet import download_model, load_from_checkpoint
import os
import glob
from tqdm import tqdm
import json
import torch
from utils import Bible
import unicodedata

torch.set_float32_matmul_precision("high")

# model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")

# want to load in all the translations
TRANSLATIONS = {}
translation_files = glob.glob(os.path.join("translations", "*.json.gz"))
translation_files = [f for f in translation_files if 'human_translation' not in f]
human_translations = glob.glob(os.path.join("human_translations", "*.jsonl"))
for h in human_translations:
    translation_files.append(h)

for file in translation_files:
    translation = Bible(file)
    TRANSLATIONS["_".join(os.path.basename(file).split("/")).split('.json.gz')[0]] = translation

originals = [k for k in TRANSLATIONS.keys() if "original" in k]
# print(TRANSLATIONS.keys())

# os.makedirs("work/comet_translations", exist_ok=True)

def remove_accents(s):
    return ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))


for original in originals:
    ref_bible = TRANSLATIONS[original]
    # print(len(ref_bible.data))
    # print(ref_bible.data[0])
    # exit()
    testament = original.split("_")[1] # old or new
    manuscript = original.split("_")[2] # Sinaiticus-Greek or WLC-Hebrew
    for lang in ["English", "Finnish", "Swedish", "Turkish", "Marathi"]:
        mt = [f for f in TRANSLATIONS.keys() if lang in f and manuscript in f and testament in f]
        human = [f for f in TRANSLATIONS.keys() if lang in f and manuscript not in f and testament in f][0]
        # HACK: these seem to cause the issue, try to see if you can run it without these. Maybe there's an issue with these files
        mt = [f for f in mt if 'exclude_human' not in f]
        # print(mt)
        # exit()
        mt.append(human)
        # print(mt, human)
        for translation in tqdm(mt):
            print(translation)
            data = []
            hyp_bible = TRANSLATIONS[translation]
            for item in ref_bible.data:
                # find the location in the reference bible and then look that up in the hypothesis bible
                loc = item["location"]
                try:
                    data.append({
                            "src": remove_accents(ref_bible[loc]),
                            "mt": remove_accents(hyp_bible[loc])
                        })
                except:
                    print(f"didn't find {loc} in {translation}")
                    print()
                    # exit()

            with open(f"work/new_comet_translations/{translation}.jsonl", "w") as ofd:
                for line in data:
                    ofd.write(json.dumps(line, ensure_ascii=False) + "\n")
                
                
                
    # for translation in tqdm(TRANSLATIONS.keys()):
    #     lang = translation.split("_")[-1]
    #     if lang in translation and testament in translation and (orig_manu in translation or "BC" in translation) and lang != "Japanese":
    #         data = []
    #         hyp_bible = TRANSLATIONS[translation]
    #         print(original, translation, lang, testament)
    #         for i in range(len(ref_bible)):
    #             # print(ref_bible.indices[i])
    #             # print(hyp_bible.indices[i])
    #             try:
    #                 data.append({
    #                     "src": remove_accents(ref_bible.indices[i]),
    #                     "mt": remove_accents(hyp_bible.indices[i])
    #                 })
    #             except KeyError:
    #                 print(f"didn't find {ref_bible.data[i]['location']['book']}")
    #                 print(ref_bible.data[i])
    #                 print(remove_accents(ref_bible.data[i]['text']))
    #                 exit()
    #                 # exit()
    #         # save it to a json lines file under the name of the translation
    #         



# model = load_from_checkpoint(model_path)
# # now load the json
# # for each translation, compare it to the original
# files = glob.glob("work/comet_translations/*.jsonl")
# for file in files:
#     with open(file, "r") as ifd:
#         data = [json.loads(line) for line in ifd]
#         model_output = model.predict(data, batch_size=64, gpus=1)
#         # save the output to a file
#         with open(f"{file}.out", "w") as ofd:
#             for line in model_output:
#                 ofd.write(json.dumps(line) + "\n")
# print(data)
        # want to compare the translations
        # for each verse in the original, compare the translation to the original
# print(data)
            # want to compare the translations
            # want to compare the translations
            # for each verse in the original, compare the translation to the original

# data = [
#     {
#         "src": "The output signal provides constant sync so the display never glitches.",
#         "mt": "Das Ausgangssignal bietet eine konstante Synchronisation, so dass die Anzeige nie stört."
#     },
#     {
#         "src": "Kroužek ilustrace je určen všem milovníkům umění ve věku od 10 do 15 let.",
#         "mt": "Кільце ілюстрації призначене для всіх любителів мистецтва у віці від 10 до 15 років."
#     },
#     {
#         "src": "Mandela then became South Africa's first black president after his African National Congress party won the 1994 election.",
#         "mt": "その後、1994年の選挙でアフリカ国民会議派が勝利し、南アフリカ初の黒人大統領となった。"
#     }
# ]

# print(model_output)
