"""
Version of create_knowledge_graph.py with simplified dependencies. Is likely to replace
the original create_knowledge_graph.py in the future. If you see this commend and this happens
to be the only create_knowledge_graph.py in the repository, then the replacement has 
already been made.
"""

import argparse
import en_core_web_sm
import glob
import json
from multiprocessing import Pool
import nltk
import numpy as np
import os
import pickle
import sys
import time
import traceback
import torch
import gensim.downloader as api
from sentence_transformers import SentenceTransformer

from corenlp import StanfordCoreNLP

SENTENCE_ENCODER_DIM = {
    "all-MiniLM-L6-v2": 384,
    "paraphrase-albert-small-v2": 768,
    "word2vec": 300,
}
SENTENCE_ENCODER = "word2vec"
CAP_TOT_EDGES = 50
KG_NODE_DIM = 100
KG_EDGE_DIM = SENTENCE_ENCODER_DIM[SENTENCE_ENCODER]
KG_NODE_EMBEDDINGS = torch.eye(KG_NODE_DIM)


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


nltk.download("maxent_ne_chunker", quiet=True)
nltk.download("words", quiet=True)
nltk.download("punkt", quiet=True)
spacy_nlp = en_core_web_sm.load()
stanford_core_nlp_path = "./stanford-corenlp"
nlp = None


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)
CAP_TOT_EDGES = 100  # We cap the number of edges to prevent graphs that will slow down our entire training process
SENTENCE_TRANFORMER_MODEL = "all-MiniLM-L6-v2"
SENTENCE_ENCODER = "word2vec"
KG_NODE_DIM = 100
KG_EDGE_DIM = SENTENCE_ENCODER_DIM[SENTENCE_ENCODER]
KG_NODE_EMBEDDINGS = torch.eye(KG_NODE_DIM)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None


def perform_triple_extraction_pipeline(doc):
    t = time.time()
    annotated = nlp.annotate(
        doc,
        properties={
            "annotators": "openie",  # "tokenize,ssplit,pos,lemma,ner,depparse,openie",
            "pipelineLanguage": "en",
        },
    )
    print(f"triple extraction time: {time.time() - t}")
    try:
        result = json.loads(annotated)
        # triple = [x["openie"] for x in result["sentences"]][0]
        # print(f"triples: {triple}")
    except:
        # if the corenlp server fails to run the pipeline on the document, return a dummy KG.
        # this shouldn't happen too often (if ever??), but the data generation takes too long
        # so i'm not taking any chances. print a log here so i can go back through the generation
        # and see if it actually failed. all failures found locally were memory issues.
        print("ERROR: corenlp server failed to run on given doc! returning dummy... ")
        result = {"sentences": [{"openie": [("dummy a", "dummy relation", "dummy c")]}]}
    # result = make_kg(result)
    # print(f"graphgen time: {time.time() - t}")
    return result


def make_kg(doc_pipeline_output, debug=True):
    t = time.time()
    # Graph object representing {u: {v1: rel1, v2: rel2, ...}}
    node2idx = {}
    edge_list = []
    edge_text = []
    for sentence in doc_pipeline_output["sentences"]:
        for triple in sentence["openie"]:
            # Extract subject, relation, and object from knowledge triple and add to g
            s, r, o = triple["subject"], triple["relation"], triple["object"]
            if o not in node2idx:
                node2idx[o] = len(node2idx)
            if s not in node2idx:
                node2idx[s] = len(node2idx)
            edge_list.append([node2idx[s], node2idx[o]])
            edge_text.append(r)
            if len(edge_list) > CAP_TOT_EDGES:
                break

    # Encode node_feats, edge_list, edge_feats in required format for PyG
    node2idx_adjusted_to_max_node_dim = (
        np.array(list(range(len(node2idx)))) % KG_NODE_DIM
    )
    node_feat = torch.clone(KG_NODE_EMBEDDINGS)[node2idx_adjusted_to_max_node_dim]
    edge_list = torch.Tensor(edge_list).t().contiguous().long()
    edge_feat = torch.stack(
        [model.encode(x, convert_to_tensor=True) for x in edge_text]
    )
    print(f"kg construction time: {time.time() - t}")
    g = {
        "node_feats": node_feat,
        "edge_indices": edge_list,
        "edge_feats": edge_feat,
    }
    if debug:
        # We usually want debug=False since this slows down our graph generation process (and also hugely increases
        # the amount of space that we end up using)
        g["node_labels"] = list(node2idx.keys())
        g["edge_labels"] = edge_text
    return g


def start_pipeline():
    global nlp
    if not nlp:
        nlp = StanfordCoreNLP(
            stanford_core_nlp_path,
            quiet=True,
            threads=3,  # os.cpu_count() // 3,
            timeout=60000,
            memory="8g",
        )


def create_model():
    global model
    if not model:
        model = SentenceEncoder(SENTENCE_ENCODER)


def stop_pipeline():
    global nlp
    if nlp:
        nlp.close()
        nlp = None


def generate_kgs(docs):
    try:
        start_pipeline()
        t = time.time()
        # Create KGs in parallel
        with Pool(os.cpu_count() // 2) as pool:
            print("starting triple extraction.")
            t = time.time()
            all_triples_info = pool.map(perform_triple_extraction_pipeline, docs)
            print(
                f"triple extraction done in {time.time()-t} s. starting KG construction."
            )
        stop_pipeline()

        create_model()
        with Pool(os.cpu_count() // 2) as pool:
            kgs = pool.map(make_kg, all_triples_info)
        print(f"tot time: {time.time()-t}")
        return kgs
    except:
        stop_pipeline()
        traceback.print_exc()
        sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Usage: python3 knowledge_graph.py path/to/input/data"
    )
    parser.add_argument(
        "input_dir",
        type=str,
    )
    parser.add_argument("--name", default="", type=str, required=False)
    args = parser.parse_args()
    pkl_file_name = f"knowledge_graphs_{args.name}.pkl"
    print(f"beginning generation. saving in: {pkl_file_name}")

    # Get documents and run kg generator
    files = glob.glob(os.path.join(args.input_dir, "*.txt"))
    docs = []
    for file in files:
        with open(file, "r") as f:
            lines = f.read().splitlines()[1:]
        doc = " ".join(lines)
        docs.append(doc)
    kgs = generate_kgs(docs)

    with open(pkl_file_name, "wb") as f:
        pickle.dump((kgs, docs), f)
