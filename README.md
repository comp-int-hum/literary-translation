
# About this repo
We use Scons, a build-system written in Python, to manage our experiments. The supreme advantage of using a build system to manage experiments is complete reproducibility and transparency of process. The key to this is that Scons allows you to **define a dependency structure for your experiments**: we define each of our steps (e.g. fetching data, pre-processing, training, post-processing, analysis, etc.) as a build rule which takes in (a) source file(s) and produces (a) target file(s). If an underlying source file changes (for example, you change how you clean your data in  pre-processing, leading to a new ```cleanded_data.json```) all target files which rely on it (e.g. model checkpoints, predictions, score tables, etc.) will be remade the next time you run ```scons```. This guarantees correctness of your experiments, ensuring that you are always working with the most up-to-date version of all of your scripts and data at the same time.

As Scons is written in Python, it also makes defining arrays/ablations of experiments easier to write and read than bash scripting. 

For experiments which should be submitted to a GPU cluster grid, we use ```streamroller```, a wrapper for Scons, to seamlessly manage grid submission scripts. Submission variables like account name and number of GPUs are defined in a ```custom.py``` script, and running ```steamroller``` in the terminal will submit each script that would have been run locally by ```scons``` to the grid as a job, respecting the dependency strcuture of your SConstruct file.

For the intertextuality experiments, we provide all raw data under ```data``` and all of of our machine-generated translations (and their embedded versions used to characterize intertextuality) under the ```intt_work``` dir. Individual translations may be accessed as ```work/{testament}/{manuscript}/{condition}/{model_name}/{language}.json.gz``` . N.B. that they are compressed with the gzip module and may be opened with ```gzip.open(file, 'rt')```

For the chiasm experiments, we provide all chiasm score files and outputs under the ```chiasm_work``` dir.


# How to use
To recreate our full experiments:
* clone this repo
* create the appropriate environment from the provided pyproject.toml
* remove the extension from one of the SConstruct.X files (e.g. to run the chiasm experiments, ```mv SConstruct.chiasm SConstruct```)
* run ```scons -n``` to print to terminal which scripts will be run in which order
* to fully rerun, simply run the command ```scons``, which will run each script in an order following the dependency structure of the SConstruct file.


# Chiasm Detection Experiments

## Data Processing


## Experiments


## Postprocessing/Analysis 


# Intertextuality Experiments




# Literary translation experiments





After cloning this repository and changing into the directory, you can avoid needing to generate the translations and embeddings by creating a file `custom.py` with the contents:

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

And you can run the non-translation/embedding aspects with:

```
$ steamroller -Q
```