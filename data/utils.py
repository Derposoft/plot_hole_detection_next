from copy import deepcopy
from gensim.corpora import Dictionary
import gensim.downloader as api
import nltk
import numpy as np
import os
import pickle as pkl
from sentence_transformers import SentenceTransformer
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, default_collate, random_split
from typing import List
from tqdm import tqdm

from models.model_utils import SENTENCE_ENCODER_DIM
import data.generate_synthetic_data as datagen


ospj = os.path.join
osl = os.listdir
device = "cuda" if torch.cuda.is_available() else "cpu"


def encode_stories(encoder, stories: List[List[str]]):
    """
    :param encoder: SentenceTransformer encoder model to use for encoding stories
    :param stories: list of stories to encode. each "story" is a list of sentences.
    :returns: list of encoded stories. story_i is of the shape (n_sentences_i, encoder_dim)
    where n_sentences_i is the number of sentences in story_i and encoder_dim is the
    output dim of the provided encoder.
    """
    output = []
    for story in tqdm(stories):
        output.append(
            torch.stack([torch.Tensor(encoder.encode(sentence)) for sentence in story])
        )
    return output


class SentenceEncoder:
    def __init__(self, encoder_name="all-MiniLM-L6-v2"):
        """
        :param encoder_name: the name of the encoder model to use.
            supported encoders can be found in SENTENCE_ENCODER_DIM
        """
        # ensure that encoder is supported
        assert (
            encoder_name in SENTENCE_ENCODER_DIM
        ), f"encoder name must be one of {list(SENTENCE_ENCODER_DIM.keys())}"
        self.encoder_name = encoder_name
        self.encoder_dim = SENTENCE_ENCODER_DIM[self.encoder_name]

        # create encoder
        if encoder_name == "word2vec":
            self.encoder_w2v = api.load("word2vec-google-news-300")
        elif encoder_name == "tfidf":
            print("not implemented!")
            sys.exit()
        else:
            self.encoder_sentencetransformer = SentenceTransformer(
                f"sentence-transformers/{encoder_name}"
            )
            self.encoder_sentencetransformer.eval().to(device)

    def encode(self, sentence: str, **kwargs):
        """
        :param sentences: n sentence string(s) to encode
        :returns: encoded sentence in the form of a
        """
        if self.encoder_name == "word2vec":
            words = sentence.split()
            words_in_w2v_model = [word for word in words if word in self.encoder_w2v]
            if not words_in_w2v_model:
                return torch.Tensor([0] * SENTENCE_ENCODER_DIM[self.encoder_name])
            return sum(
                [
                    torch.Tensor(np.copy(self.encoder_w2v[word]))
                    for word in words_in_w2v_model
                ]
            )
        elif self.encoder_name == "tfidf":
            print("not implemented!")
            sys.exit()
        else:
            return self.encoder_sentencetransformer.encode(sentence)


class StoryDataset(Dataset):
    def __init__(self, X, y, kgs=None, raw_stories=None, file_names=None):
        Dataset.__init__(self)
        self.raw_stories = raw_stories
        self.X = X
        self.y = y
        self.kgs = kgs

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        import create_knowledge_graph as kg_utils

        kg_node_dim, kg_edge_dim = kg_utils.KG_NODE_DIM, kg_utils.KG_EDGE_DIM
        if not self.kgs:
            kg_node_dim, kg_edge_dim = 1, 1
        n_nodes, n_edges = 1, 1
        kg = {
            "node_feats": torch.zeros([n_nodes, kg_node_dim]),
            "edge_indices": torch.zeros((2, 1)).long(),
            "edge_feats": torch.zeros([n_edges, kg_edge_dim]),
        }
        if self.kgs and len(self.kgs[idx]["node_feats"] > 0):
            kg = self.kgs[idx]
        return self.X[idx], self.y[idx], kg, self.raw_stories[idx]

    def get_num_sentences_per_story(self):
        return len(self.X[0])


def custom_dataloader_collate(data):
    x_padded = torch.nn.utils.rnn.pad_sequence([x[0] for x in data], batch_first=True)
    y_padded = torch.nn.utils.rnn.pad_sequence([x[1] for x in data], batch_first=True)
    docs_padded = torch.nn.utils.rnn.pad_sequence([x[3] for x in data], batch_first=True)

    X, y = default_collate([(x[0], x[1]) for x in zip(x_padded, y_padded)])
    kgs = [x[2] for x in data]
    documents = default_collate([x for x in docs_padded])
    return X, y, kgs, documents


def create_story_dataloader(dataset, batch_size=8):
    """
    :param dataset: story dataset to feed to the dataloader
    :param batch_size: batch_size to initialize dataloader with
    :returns: dataloader with custom collation for kgs for a story dataset
    """
    return DataLoader(
        dataset,
        collate_fn=custom_dataloader_collate,
        batch_size=batch_size,
    )


def try_get_final_ficclaim_data(
    batch_size=8, final_data_path = "FicClaim/", n_cont_errors = 1, train_ratio=0.8
) -> tuple[DataLoader, DataLoader]:
    if not os.path.exists(final_data_path):
        return None, None
    cache_file = f"FicClaim-{n_cont_errors}-error.pkl"
    cache_files = osl(final_data_path)
    if cache_file not in cache_files:
        return None, None

    print(f"Final FicClaim preprocessed data found. Loading...")
    with open(ospj(final_data_path, cache_file), "rb") as f:
        dataset = pkl.load(f)
    train_dataset, test_dataset = random_split(dataset, [train_ratio, 1 - train_ratio])
    train_dataloader = create_story_dataloader(train_dataset, batch_size)
    test_dataloader = create_story_dataloader(test_dataset, batch_size)
    return train_dataloader, test_dataloader


def generate_data(
    batch_size=8,
    data_path="data/synthetic/train",
    cache_path="data/encoded/train",
    encoder="all-MiniLM-L6-v2",
    n_stories=5,
    n_synth=1,
    get_kgs=False,
    optimize_space=False,
    n_continuity_errors=1,
):
    import create_knowledge_graph as kg_utils

    """
    :param batch_size: batch_size for output dataloaders
    :param data_path: location of data
    :param cache_path: location of cached data
    :param encoder: name of encoder to use to encode story sentences
    :param n_stories: number of stories to use
    :param n_synth: number of synthetic datapoints to create per story
    :returns: continuity_dataloader dataloader

    first check to see if cached story encodings exist for this n_stories choice at
    cache_path. otherwise:
    1. parses data files at data_path; if num files in data_path < n_stories*2,
       generate new synthetic data
    2. encoding each story by sentence
    2.5. generate kgs for each story
    3. preprocess via padding smaller stories with 0s for same-length stories
    4. labels for continuity errors are 1-hot encoded
    5. returns dataloaders of these stories
    6. cache these tensors
    """
    # check if cached stories exist for this n_stories
    kg_suffix = "_kg" if get_kgs else ""
    cache_file = f"{n_stories}-{n_synth}-stories_{encoder}-encoded_{n_continuity_errors}-cont-errors{kg_suffix}.pkl"
    final_cache_file_test = f"test_{n_continuity_errors}_error.pkl"
    final_cache_file_train = f"train_{n_continuity_errors}_error.pkl"
    optimized_space_cache_file = f"{n_stories}-{n_synth}-stories_{encoder}-encoded_{n_continuity_errors}-cont-errors_kg.pkl"
    cache_files = osl(cache_path)
    if optimize_space and optimized_space_cache_file in cache_files:
        cache_file = optimized_space_cache_file
    if final_cache_file_test in cache_files:
        cache_file = final_cache_file_test
    if final_cache_file_train in cache_files:
        cache_file = final_cache_file_train
    if cache_file in cache_files:
        print(f"Using cached encoded data file: {cache_file}")
        with open(ospj(cache_path, cache_file), "rb") as f:
            continuity_dataset = pkl.load(f)
        continuity_dataloader = create_story_dataloader(continuity_dataset, batch_size)
        return continuity_dataloader

    # ensure enough synthetic data is available, otherwise generate more
    def get_data_files(n_cont_errors):
        return [x for x in osl(data_path) if x.endswith(".txt") and f"{n_cont_errors}-err" in x]
    data_files = get_data_files(n_continuity_errors)
    if len(data_files) < n_stories * n_synth:
        print(
            f"{n_stories*n_synth} datapoints necessary but only {len(data_files)//2} exist. regenerating synthetic data."
        )
        datagen.generate_synthetic_data_all(n_stories, n_synth)
        data_files = get_data_files(n_continuity_errors)

    # parse all data files in data_path and separate them by error type
    continuity_files = []
    continuity_data = []
    continuity_labels = []
    for data_file in tqdm(data_files):
        with open(ospj(data_path, data_file), "r") as f:
            lines = f.readlines()
            problem_metadata = lines[0].split()
            problem = problem_metadata[0]
            if problem == "continuity":
                labels = problem_metadata[1:]
                labels = [
                    int(x.removeprefix("[").removesuffix("]").removesuffix(","))
                    for x in labels
                ]
                continuity_files.append(data_file)
                continuity_data.append(lines[1:])
                continuity_labels.append(labels)
            else:
                print("ERROR: unsupported problem type passed to generate_data!")
                sys.exit(1)

    # cut returned data down to requested dataset size
    def first_n(data, n):
        return data[: min(len(data), n)]

    n = n_stories * n_synth
    continuity_files = first_n(continuity_files, n)
    continuity_data = first_n(continuity_data, n)
    continuity_labels = first_n(continuity_labels, n)

    # generate kgs for chosen stories
    continuity_kgs = []
    if get_kgs:
        print("get_kgs set to True, generating KGs for stories.")
        continuity_docs = [" ".join(lines) for lines in continuity_data]
        continuity_kgs = kg_utils.generate_kgs(continuity_docs)

    # create tokenized documents for some downstream models
    longest_story_length = max([len(story) for story in continuity_data])
    max_claim_length = 15  # TODO choose this properly
    continuity_raw_stories = deepcopy(continuity_data)
    doc2idx_dict = Dictionary()
    for story in continuity_raw_stories:
        tokenized_story = [nltk.tokenize.word_tokenize(sentence) for sentence in story]
        doc2idx_dict.add_documents(tokenized_story)

    def preprocess_raw_stories(stories: List[List[str]]):
        """
        preprocess each of the stories by converting them into a tensor of word indices.
        these can be passed into an embedding layer later. we pad each sentence so that they're
        all the same length, then pad each story so that they have the same number of
        sentences.
        """
        preprocessed_stories = []
        for story in stories:
            preprocessed_story = []
            for sentence in story:
                sentence = nltk.tokenize.word_tokenize(sentence)
                preprocessed_sentence = doc2idx_dict.doc2idx(sentence)
                preprocessed_sentence = first_n(preprocessed_sentence, max_claim_length)
                if len(preprocessed_sentence) < max_claim_length:
                    preprocessed_sentence += [-1] * (
                        max_claim_length - len(preprocessed_sentence)
                    )
                preprocessed_story.append(preprocessed_sentence)
            preprocessed_story = torch.Tensor(preprocessed_story)
            if len(preprocessed_story) < longest_story_length:
                preprocessed_story_pad = torch.zeros(
                    [longest_story_length - len(preprocessed_story), max_claim_length]
                )
                preprocessed_story = torch.cat(
                    [preprocessed_story, preprocessed_story_pad], dim=0
                )
            preprocessed_stories.append(preprocessed_story)
        return torch.stack(preprocessed_stories) + 1

    continuity_raw_stories = preprocess_raw_stories(continuity_raw_stories)

    # encode all data file sentences using encoder
    print("encoding stories...")
    encoder = SentenceEncoder(encoder_name=encoder)
    continuity_data = encode_stories(encoder, continuity_data)

    # pad all stories to meet the length of the longest story
    continuity_data = [
        F.pad(story, (0, 0, 0, longest_story_length - len(story)))
        for story in continuity_data
    ]
    continuity_data = torch.stack(continuity_data)

    # 1-hot encode continuity error labels, turn labels into tensors
    continuity_labels = torch.stack(
        [
            torch.sum(torch.eye(longest_story_length)[continuity_label], dim=0)
            for continuity_label in continuity_labels
        ]
    )

    # save encoded stories into cache
    continuity_dataset = StoryDataset(
        X=continuity_data,
        y=continuity_labels,
        kgs=continuity_kgs,
        raw_stories=continuity_raw_stories,
        file_names=data_files,
    )
    with open(ospj(cache_path, cache_file), "wb") as f:
        pkl.dump(continuity_dataset, f)

    # create dataloaders for each error type
    continuity_dataloader = create_story_dataloader(continuity_dataset, batch_size)
    return continuity_dataloader


"""
# JUNK SECTION (all unresolved story stuff)

    skip_unresolved=True,  # speed up when only doing continuity
        unresolved_dataloader = create_story_dataloader(unresolved_dataset, batch_size)

    unresolved_files = []
    unresolved_data = []
    unresolved_labels = []
if problem == "unresolved":
                label = problem_metadata[1]
                unresolved_files.append(data_file)
                unresolved_data.append(lines[1:])
                unresolved_labels.append(float(label))

    unresolved_files = first_n(unresolved_files, n)
    unresolved_data = first_n(unresolved_data, n)
    unresolved_labels = first_n(unresolved_labels, n)

    


    unresolved_kgs = []
        unresolved_docs = [" ".join(lines) for lines in unresolved_data]
        if skip_unresolved:
            unresolved_kgs = continuity_kgs
        else:
            unresolved_kgs = kg_utils.generate_kgs(unresolved_docs)
    unresolved_raw_stories = deepcopy(unresolved_data)
    for story in unresolved_raw_stories:
        tokenized_story = [nltk.tokenize.word_tokenize(sentence) for sentence in story]
        doc2idx_dict.add_documents(tokenized_story)
    if skip_unresolved:
        unresolved_raw_stories = continuity_raw_stories
    else:
        unresolved_raw_stories = preprocess_raw_stories(unresolved_raw_stories)
    if skip_unresolved:
        unresolved_data = continuity_data
    else:
        unresolved_data = encode_stories(encoder, unresolved_data)

    unresolved_data = [
        F.pad(story, (0, 0, 0, longest_story_length - len(story)))
        for story in unresolved_data
    ]
    if not skip_unresolved:
        unresolved_data = torch.stack(unresolved_data)

    unresolved_labels = torch.FloatTensor(unresolved_labels)
    unresolved_dataset = (
        StoryDataset(
            unresolved_data, unresolved_labels, unresolved_kgs, unresolved_raw_stories
        )
        if not skip_unresolved
        else None
    )
    unresolved_dataloader = create_story_dataloader(unresolved_dataset, batch_size)
"""
