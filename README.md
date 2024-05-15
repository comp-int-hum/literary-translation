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