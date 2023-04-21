## Low-Shot Claim Verification for Fictional Claims

This repo is a WIP for obtaining new results for anpudated version of the paper [Low-Shot Learning for Fictional Claim Verification](https://arxiv.org/abs/2304.02769). The TODOs for this repo are:

1. collect all previous models:
    - Pattern-based methods
        - ~~BERT (to-be implemented on my own)~~
        - ~~LSTM (to-be implemented on my own)~~
        - ~~TextCNN (https://github.com/Cheneng/TextCNN)~~
    - Evidence-based methods
        - ~~DeClarE (https://github.com/atulkumarin/DeClare)~~
        - ~~MAC (https://github.com/nguyenvo09/EACL2021)~~
        - ~~GET (https://github.com/CRIPAC-DIG/GET)~~
        - HAN/EHIAN/CICD? (they have no source code on github...? not going to do these since they're not trivial to implement)

2. standardize interface among all models and connect them all to a unified trainer/evaluator that outputs the following metrics: F1, Precision, Recall for both T/F and Macro/Micro (is this level of detail necessary?)
    - ~~standardization scheme (?): forward(self, claim/query, document/article)~~
    - ~~standardization scheme v2: forward(self, document: torch.Tensor)~~
    - ~~how should the claim/query and document/article be encoded? (we should probably use the same encoder as the relative models?)~~
    - standardization scheme v3: forward(self, X, document, **kwargs). let the model have access to our encodings, and also the raw text. shift other repos' transformations into the model forward() and hope the performance hit isn't as bad as it sounds.

3. Generate our data


### Steps to run:
1. install conda env using conda_env.yml: `conda env create --file=conda_env_gpu.yml`
2. install spacy lexicon: `python -m spacy download en_core_web_sm`
3. see model training options and train a model: `python train.py --help`
4. `python clean_data.py` to delete all cached data ONLY for debugging purposes (warning: this will destroy your cached data!)


### Reproducibility:
Despite our efforts to the contrary, reproducibility is not assured. Results that are slightly different than our own can occur with even small changes such as different CUDA versions or different GPUs (https://pytorch.org/docs/stable/notes/randomness.html for more info).
