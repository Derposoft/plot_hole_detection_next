## Low-Shot Claim Verification for Fictional Claims
Data can be downloaded by running `download_ficclaim.sh`. Alternatively, to use your own dataset, original text files can be placed in the `data/raw/` folder before the "steps to run" are followed (then that data will automatically be used). A long-form fictional text dataset can be downloaded and afterwards placed in the `data/raw/` folder by running `python data/ao3scraper.py`.

### Steps to run:
1. install conda env using conda_env.yml: `conda env create --file=conda_env_gpu.yml`
2. install spacy lexicon: `python -m spacy download en_core_web_sm`
3. see model training options and train a model: `python train.py --help`
4. `python clean_data.py` to delete all cached data ONLY for debugging purposes (warning: this will destroy your cached data!)


### Reproducibility:
Despite our efforts to the contrary, reproducibility is not assured. Results that are slightly different than our own can occur with even small changes such as different CUDA versions or different GPUs (https://pytorch.org/docs/stable/notes/randomness.html for more info).
