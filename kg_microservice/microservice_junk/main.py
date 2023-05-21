from flask import Flask, request
import os
import time
import json
import gensim.downloader as api
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
import torch

from .corenlp import StanfordCoreNLP


SENTENCE_ENCODER_DIM = {
    "all-MiniLM-L6-v2": 384,
    "paraphrase-albert-small-v2": 768,
    "word2vec": 300,
}
CORENLP_SERVER_URL = os.environ["CORENLP"]
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


app = Flask(__name__)
nlp = StanfordCoreNLP(
    # stanford_core_nlp_path,
    CORENLP_SERVER_URL,
    quiet=False,
)
model = SentenceEncoder(SENTENCE_ENCODER)
device = "cpu"


@app.route("/", methods=["POST"])
def index():
    doc = request.form["text"]

    # CORENLP PIECE
    t = time.time()
    annotated = nlp.annotate(
        doc,
        properties={
            "annotators": "openie",  # "tokenize,ssplit,pos,lemma,ner,parse,depparse,coref,openie",
            "pipelineLanguage": "en",
        },
    )
    print(f"triple extraction time: {time.time() - t}")
    try:
        corenlp_output = json.loads(annotated)
    except:
        # if the corenlp server fails to run the pipeline on the document, return a dummy KG.
        # this shouldn't happen too often (if ever??), but the data generation takes too long
        # so i'm not taking any chances. print a log here so i can go back through the generation
        # and see if it actually failed. all failures found locally were memory issues.
        print("ERROR: corenlp server failed to run on given doc! returning dummy... ")
        corenlp_output = {
            "sentences": [{"openie": [("dummy a", "dummy relation", "dummy c")]}]
        }

    # GRAPH MAKING PIECE
    t = time.time()
    # Graph object representing {u: {v1: rel1, v2: rel2, ...}}
    node2idx = {}
    edge_list = []
    edge_feat = []
    for sentence in corenlp_output["sentences"]:
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


if __name__ == "__main__":
    app.run()
