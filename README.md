
# About this repo
This repo contains code for reproducing experiments found in "Characterizing the Effect of Translation on Intertextuality using Multilingual Embedding Spaces" and "Computational Discovery of Chiasmus in Ancient Religious Text", submitted to ARR for the October 2024 reviewing cycle.

We use [Scons](https://scons.org/), a build-system written in Python, to manage our experiments. The supreme advantage of using a build system to manage experiments is complete reproducibility and transparency of process. The key to this is that Scons allows you to **define a dependency structure for your experiments**: we define each of our steps (e.g. fetching data, pre-processing, training, post-processing, analysis, etc.) as a build rule which takes in (a) source file(s) and produces (a) target file(s). If an underlying source file changes (for example, you change how you clean your data in  pre-processing, leading to a new ```cleanded_data.json```) all target files which rely on it (e.g. model checkpoints, predictions, score tables, etc.) will be remade the next time you run ```scons```. This guarantees correctness of your experiments, ensuring that you are always working with the most up-to-date version of all of your scripts and data at the same time.

For experiments which should be submitted to a GPU cluster, we use [steamroller](https://libraries.io/pypi/SteamRoller), a wrapper for Scons, to seamlessly manage grid submission scripts. Submission variables like account name and number of GPUs are defined in a ```custom.py``` script, and running ```steamroller``` in the terminal will submit each script that would have been run locally by scons to the grid as a job, respecting the dependency strcuture of your SConstruct file.

We provide all raw data under ```data``` and all of of our machine-generated translations (and their embedded versions used to characterize intertextuality) under the ```intt_work``` dir. Individual translations may be accessed as ```work/{testament}/{manuscript}/{condition}/{model_name}/{language}.json.gz``` . N.B. that they are compressed with the gzip module and may be opened with ```gzip.open(file, 'rt')```

For the chiasm experiments, we provide all chiasm score files and outputs under the ```chiasm_work``` dir.


# How to use

After cloning this repository, recreating the environment from the provided ```pyproject.toml``` file, and changing into the directory, you can avoid needing to generate the translations and embeddings by creating a file `custom.py` with the contents:

```
USE_PRECOMPUTED_EMBEDDINGS = True
```

Downloading the file `work.tgz` and unpacking it with:

```
$ tar xpfz work.tgz
```

Setting up a virtual environment:

```
$ python3 -m venv local
$ source local/bin/activate
$ pip install -r requirements.txt
$ python -c "import nltk; nltk.download('stopwords')"
```

And you can run the non-translation/embedding experiments with:

```
$ steamroller -Q
```

## Other Notes
* In ```notebooks/clean_data.ipynb```, you can see how we take raw .xml files for the OT Hebrew/Greek and NT Greek manuscripts and turn them into the cleaned, aligned .json files under the ```data``` dir.
* ```scripts/benchmark.py``` houses the code for our benchmark experiments on the [Valerius Flaccus Intertextuality dataset](https://openhumanitiesdata.metajnl.com/articles/153/files/65b7ab32731ea.pdf) 