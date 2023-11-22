from copy import deepcopy
from functools import lru_cache
import nltk
import numpy as np
import os
from pathlib import Path
from sys import platform
from typing import List, Tuple
import json
import requests
import openai
import time


nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
ROOT = Path(__file__)
osl = os.listdir
ospj = os.path.join


def get_datafiles() -> list:
    """
    returns list of stories(.txt file) in raw_story_file folder
    """
    return [x for x in Path(ROOT.parent / "raw").iterdir() if str(x).endswith(".txt")]


@lru_cache(maxsize=None)
def get_openapi_key():
    with open("keys.json", "r") as f:
        api_key = json.load(f)["openai"]
    return api_key


def negater(sentence: str, max_retry: int = 6) -> str:
    """
    Basic logic
    1. Send the sentence to chat gpt and call it good
    2. Keep retrying chatgpt thing in case we get rate limited, but give up after a while
       so that a human can debug if we failed for over a minute
    """
    openai.api_key = get_openapi_key()
    task = f'Negate the following: "{sentence}"'
    n_retries = 0
    while max_retry == -1 or n_retries <= max_retry:
        try:
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt=task,
                max_tokens=20,
                temperature=0.7,
            )
            negated_sentence = response.choices[0].text.strip()
            print("Original:", sentence, " --- Negated:", negated_sentence)
            return negated_sentence
        except openai.error.APIError:
            max_retry += 1
            sleeptime = 10
            time.sleep(sleeptime)
            print(f"[ERROR] could not negate {sentence} due to rate limiting... sleeping for {sleeptime} seconds")
    print(f"[ERROR] could not negate {sentence} within {max_retry} retries. exiting!")
    exit(0)


def negater_old(sentence: str) -> str:
    """
    Basic logic
    1. Check if the word is a verb using nltk tagger
    2. If the word is verb, look for antonyms
    3. If antonym exist get a random antonym and put it in the word's place
    4. If antonym does not exist, put "not" in front of the word if that's not a corpula, if the word is a corpula put "not" after the word.
    """
    wordnet = nltk.corpus.wordnet
    to_be_verbs = {"was", "is", "are", "am"}
    tgt = sentence.split(" ")
    tags = nltk.pos_tag(tgt)
    res = list()
    negated = False
    for word, tag in zip(tgt, tags):
        if tag[1][0] == "V" and not negated:
            negated = True
            # is the verb a to-be verb?
            if word in to_be_verbs:
                res.append(f"{word} not")
                continue

            # if not a to-be verb, can we find an antonym?
            antonyms = []
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    cands = l.antonyms()
                    if cands:
                        antonyms.append(cands[0].name())
            antonyms = list(set(antonyms))
            if len(antonyms) > 0:
                res.append(np.random.choice(antonyms, 1, replace=False)[0])

            # no antonym exists; just prepend "not" and hope things work out
            else:
                res.append(f"not {word}")
        else:
            res.append(word)
    return " ".join(res)


def generate_continuity_errors(
    document: str, n: int, n_continuity_errors: int = 1
) -> Tuple[List[str], List[int]]:
    """
    negate random lines in a story to create continuity errors storyliens
    :param document: string document.
    :param n: number of samples to generate.
    :returns: (X, y) tuple for X=list of synthetic documents, y=list of labels
    """
    sentences = nltk.sent_tokenize(document)
    n_samples = n
    X = []
    y = []
    for _ in range(n_samples):
        n_errors = min(n_continuity_errors, len(sentences))
        samples = np.random.choice(range(len(sentences)), n_errors, replace=False)
        X.append(deepcopy(sentences))
        y.append(samples.tolist())
        for sample in samples:
            X[-1][sample] = negater(X[-1][sample])
    X = ["\n".join(x) for x in X]
    return X, y


def generate_continuity_errors_all(
    document: str, n_samples: int
) -> Tuple[List[str], List[int]]:
    """
    negate random lines in a story to create continuity errors storyliens
    :param document: string document.
    :param n: number of samples to generate.
    :returns: (X, y) tuple for X=list of synthetic documents, y=list of labels
    """
    n_errs_to_generate = [1, 2, 5]
    sentences = nltk.sent_tokenize(document)
    Xs = []
    ys = []
    for _ in range(n_samples):
        n_errs_injected = 0
        n_errors = min(max(n_errs_to_generate), len(sentences))
        samples = np.random.choice(range(len(sentences)), n_errors, replace=False)
        X = deepcopy(sentences)
        y = []
        for sample in samples:
            X[sample] = negater(X[sample])
            y.append(sample)
            n_errs_injected += 1
            if n_errs_injected in n_errs_to_generate:
                X_text = "\n".join(X)
                Xs.append(deepcopy(X_text))
                ys.append(deepcopy(y))
    return Xs, ys


def write_synthetic_datapoint_to_file(X, y, path, plot_hole_type):
    """
    write a synthetic datapoint to a file.
    :param X: synthetic document
    :param y: synthetic label
    :param path: path to write the file to
    :param plot_hole_type: type of plot hole, will be written at top of document
    :returns: None. file will be written at path. first line will be "plot_hole_type y", and
    rest of the lines will be X.
    """
    with open(path, "w", encoding="utf-8") as synthetic_document_f:
        synthetic_document_f.write(f"{plot_hole_type} {y}\n")
        synthetic_document_f.write(X[1:])


def generate_synthetic_data(
    n_stories=10, n_synth=1, train_ratio=0.5, n_continuity_errors=1
):
    dataset = get_datafiles()[: 2 * n_stories]
    n_docs = len(dataset)
    for doc_idx in range(len(dataset)):
        train_test_prefix = "train/" if doc_idx < n_docs * train_ratio else "test/"
        doc_path = dataset[doc_idx]
        with open(doc_path, "r", encoding="utf8") as document_f:
            document = " ".join([x.strip() for x in document_f.readlines()])
            X_continuity, y_continuity = generate_continuity_errors(
                document, n_synth, n_continuity_errors=n_continuity_errors
            )
            for i in range(n_synth):
                if i >= len(X_continuity):
                    break
                doc_name = (
                    str(doc_path)
                    .split("\\" if platform == "win32" else "/")[-1]
                    .split(".")[0]
                )
                continuity_path = (
                    ROOT.parent
                    / f"synthetic/{train_test_prefix}synthetic_{doc_name}_{n_continuity_errors}-err_continuity{i}.txt"
                )
                X, y = X_continuity[i], y_continuity[i]
                write_synthetic_datapoint_to_file(
                    X=X, y=y, path=continuity_path, plot_hole_type="continuity"
                )


def perform_datagen_recovery(dataset):
    ### hacc section :fumo:
    train_dir = ROOT.parent / f"synthetic/train/"
    files_in_train_dir = [x for x in os.listdir(train_dir) if x.endswith(".txt")]
    def docname_from_synth(synthdatapt: str):
        res = synthdatapt.split("thetic_")[1].split("-")[0][:-2] + ".txt"
        return res
    docnames_in_train_dir = [docname_from_synth(str(x)) for x in files_in_train_dir]
    def docname_from_path(path: str):
        res = path.split("/")[-1]
        return res
    dataset = [x for x in dataset if docname_from_path(str(x)) not in docnames_in_train_dir]
    return dataset
    ###


def generate_synthetic_data_all(
    n_stories=10, n_synth=1, train_ratio=0.5
):
    dataset = get_datafiles()[: 2 * n_stories]
    pre_recovery_dataset_count = len(dataset)
    dataset = perform_datagen_recovery(dataset)
    post_recovery_dataset_count = len(dataset)
    if pre_recovery_dataset_count != post_recovery_dataset_count:
        print("[DEBUG] Performed recovery from a failed  datagen; "
              f"{pre_recovery_dataset_count} raw docs sheared to {post_recovery_dataset_count} raw docs")
    n_docs = len(dataset)
    for doc_idx in range(len(dataset)):
        train_test_prefix = "train/" if doc_idx < n_docs * train_ratio else "test/"
        doc_path = dataset[doc_idx]
        with open(doc_path, "r", encoding="utf8") as document_f:
            document = " ".join([x.strip() for x in document_f.readlines()])
            Xs, ys = generate_continuity_errors_all(
                document, n_synth
            )
            for i, (x, y) in enumerate(zip(Xs, ys)):
                doc_name = (
                    str(doc_path)
                    .split("\\" if platform == "win32" else "/")[-1]
                    .split(".")[0]
                )
                n_continuity_errors = len(y)
                continuity_path = (
                    ROOT.parent
                    / f"synthetic/{train_test_prefix}synthetic_{doc_name}_{n_continuity_errors}-err_continuity{i}.txt"
                )
                write_synthetic_datapoint_to_file(
                    X=x, y=y, path=continuity_path, plot_hole_type="continuity"
                )



if __name__ == "__main__":
    generate_synthetic_data_all(1000, 10)

"""
# JUNK SECTION

    sentence = "This is a regular sentence."
    negated_sentence = negater(sentence)
    print(negated_sentence)


def generate_unresolvedstory_errors(
    document: str, n: int, p: float = 0.1
) -> Tuple[List[str], List[int]]:
    ""
    removes random n lines from the end of a story to create unresolved storyliens
    :param document: string document.
    :param n: number of samples to generate.
    :param p: percentage of sentences to cut off of the end at most
    ""
    X = []
    # Preprocessing - remove new line character and empty lines
    sentences = nltk.sent_tokenize(document)
    n_sentences = len(sentences)
    most_sentences_to_remove = max(n + 1, int(p * n_sentences))

    # Given number of lines will be random #See below 0 to 20% of Number of Sentences
    samples = np.random.choice(
        range(1, most_sentences_to_remove),
        min(n, most_sentences_to_remove - 1),
        replace=False,
    )

    # Create n text with n lines from the last removed
    for sample in samples:
        X.append(".\n".join(sentences[:-sample]))
    y = samples / n_sentences
    return X, y

"""
