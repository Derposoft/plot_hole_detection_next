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
import torch
import traceback

from knowledge_graph.corenlp import StanfordCoreNLP
from models.model_utils import SENTENCE_ENCODER_DIM
from sentence_transformers import SentenceTransformer
from data.utils import SentenceEncoder


nltk.download("maxent_ne_chunker", quiet=True)
nltk.download("words", quiet=True)
nltk.download("punkt", quiet=True)
spacy_nlp = en_core_web_sm.load()
stanford_core_nlp_path = "./stanford-corenlp"
nlp = None


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)
CAP_TOT_EDGES = 50
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
            "annotators": "tokenize,ssplit,pos,lemma,ner,parse,depparse,coref,openie",
            "pipelineLanguage": "en",
        },
    )
    print(f"triple extraction time: {time.time() - t}")
    try:
        return json.loads(annotated)
    except:
        # if the corenlp server fails to run the pipeline on the document, return a dummy KG.
        # this shouldn't happen too often (if ever??), but the data generation takes too long
        # so i'm not taking any chances. print a log here so i can go back through the generation
        # and see if it actually failed. all failures found locally were memory issues.
        print("ERROR: corenlp server failed to run on given doc! returning dummy... ")
        return {"sentences": [{"openie": [("dummy a", "dummy relation", "dummy c")]}]}


def make_kg(doc_pipeline_output):
    t = time.time()
    # Graph object representing {u: {v1: rel1, v2: rel2, ...}}
    node2idx = {}
    edge_list = []
    edge_feat = []
    for sentence in doc_pipeline_output["sentences"]:
        for triple in sentence["openie"]:
            # Extract subject, relation, and object from knowledge triple and add to g
            s, r, o = triple["subject"], triple["relation"], triple["object"]
            if o not in node2idx:
                node2idx[o] = len(node2idx)
            if s not in node2idx:
                node2idx[s] = len(node2idx)
            edge_list.append([node2idx[s], node2idx[o]])
            edge_feat.append(r)
            if len(edge_list) > CAP_TOT_EDGES:
                break

    # Encode node_feats, edge_list, edge_feats in required format for PyG
    node2idx_adjusted_to_max_node_dim = (
        np.array(list(range(len(node2idx)))) % KG_NODE_DIM
    )
    node_feat = torch.clone(KG_NODE_EMBEDDINGS)[node2idx_adjusted_to_max_node_dim]
    edge_list = torch.Tensor(edge_list).t().contiguous().long()
    edge_feat = torch.stack(
        [model.encode(x, convert_to_tensor=True) for x in edge_feat]
    )
    print(f"kg construction time: {time.time() - t}")
    return {
        "node_feats": node_feat,
        "edge_indices": edge_list,
        "edge_feats": edge_feat,
    }


def start_pipeline():
    global nlp
    if not nlp:
        nlp = StanfordCoreNLP(
            stanford_core_nlp_path,
            quiet=True,
            threads=2,  # os.cpu_count() // 3,
            timeout=60000,
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
    args = parser.parse_args()

    # Get documents and run kg generator
    files = glob.glob(os.path.join(args.input_dir, "*.txt"))
    docs = []
    for file in files:
        with open(file, "r") as f:
            lines = f.read().splitlines()
        doc = " ".join(lines)
        docs.append(doc)
    kgs = generate_kgs(docs)

    with open("knowledge_graphs.pkl", "wb") as f:
        pickle.dump(kgs, f)
