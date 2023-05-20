import os

# for some reproducibility, may be removable later
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import argparse
import json
import numpy as np
import random
from scipy.stats import ttest_1samp
from sklearn.metrics import (
    mean_squared_error,
    precision_recall_fscore_support,
    f1_score,
)
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from time import time
import torchtext.vocab as vocab

from baselines.models.lstm import BaselineLSTM
from baselines.models.get import GraphBasedSemanticStructure
from baselines.models.mac import HierachicalMultiHeadAttentionModel
from baselines.models.text_cnn import TextCNN
from baselines.models.DeClarE import DeClareModel

from data import utils
from models.bert import ContinuityBERT, UnresolvedBERT
import knowledge_graph.create_knowledge_graph as kg_utils

device = "cuda" if torch.cuda.is_available() else "cpu"
PR_THRESHOLD = None


def set_seed(seed):
    """
    :param seed: seed to use for reproducibility purposes
    :returns: None. sets seed as per https://pytorch.org/docs/stable/notes/randomness.html
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def test(*, model, test_data, metrics=["f1", "prec", "rec"], verbosity=10, debug=False):
    """
    :param model: the model to test
    :param test_data: test dataloader
    :param metrics: list of metrics to calculate, print, and return.
        options are "f1", "prec", "rec", "mse".
    :param verbosity: whether or not to print extra output, lower=more verbose
    :returns: prints and returns metrics
    """
    # collect metrics
    y_preds = []
    y_true = []
    for _, (X, y, kgs, documents) in enumerate(test_data):
        X, y = X.to(device), y.to(device)
        for kg in kgs:
            for k in kg:
                kg[k] = kg[k].to(device)
        with torch.no_grad():
            y_preds.append(model(X, kgs=kgs, documents=documents))
        y_true.append(y)
        if debug:
            break
    y_preds, y_true = y_preds, y_true

    # calculate metrics
    y_true = torch.cat(y_true).cpu().flatten()
    y_preds = torch.cat(y_preds).cpu().flatten()
    y_preds = y_preds >= PR_THRESHOLD
    results = {}

    def get_f1s(y_true, y_preds):
        f1 = f1_score(y_true, y_preds)
        f1_macro, prec_macro, rec_macro, supp_macro = precision_recall_fscore_support(
            y_true, y_preds, average="macro"
        )
        f1_micro, prec_micro, rec_micro, supp_micro = precision_recall_fscore_support(
            y_true, y_preds, average="micro"
        )
        results["f1"] = f1
        results["f1_macro"] = f1_macro
        results["prec_macro"] = prec_macro
        results["rec_macro"] = rec_macro
        results["f1_micro"] = f1_micro
        results["prec_micro"] = prec_micro
        results["rec_micro"] = rec_micro

    def get_mse(y_true, y_preds):
        results["mse"] = mean_squared_error(y_true, y_preds)

    if "f1" in metrics:
        get_f1s(y_true, y_preds)
    if "mse" in metrics:
        get_mse(y_true, y_preds)
    return results


def train(
    *,
    model,
    train_data,
    test_data,
    opt,
    criterion,
    epochs=10,
    metrics=["f1"],
    verbosity=5,
    debug=False,
):
    """
    :param model: the model to test
    :param train_data: train dataloader
    :param test_data: test dataloader
    :param verbosity: whether or not to print extra output, lower=more verbose
    :returns: nothing. trains given model using train_data and tests it every epoch with test_data
    """
    best_metrics = {}
    for epoch in range(epochs):
        start_time = time()
        tot_loss = 0
        for i, (X, y, kgs, documents) in enumerate(train_data):
            X, y = X.to(device), y.to(device)
            for kg in kgs:
                for k in kg:
                    kg[k] = kg[k].to(device)
            y_hat = model(X, kgs=kgs, documents=documents)
            loss = criterion(y_hat, y)
            tot_loss += loss.item()
            loss.backward()
            opt.step()
            if debug:
                break
        tot_loss /= len(train_data)
        results = None
        if (epoch + 1) % verbosity == 0:
            results = test(
                model=model, test_data=test_data, metrics=metrics, verbosity=0
            )
            if "f1" in metrics:
                if results["f1"] > best_metrics.get("f1", 0):
                    best_metrics = results
            else:
                if results["mse"] < best_metrics.get("mse", float("inf")):
                    best_metrics = results
        if verbosity <= 0 or (epoch + 1) % verbosity == 0:
            results_str = f", metrics: {results}" if results != None else ""
            print(
                f"epoch {epoch+1} time: {time()-start_time:0.3}s, train loss: {tot_loss:0.4}{results_str}"
            )
    print(f"post-training summary -- best {best_metrics}")
    return best_metrics


def get_training_artifacts(config: dict):
    """
    :param config: cli input arguments
    :returns: model_constructor, train_data, test_data, criterion, metrics
    """
    problem_type = config["problem_type"]
    model_type = config["model_type"]
    use_kg = "_kg" in model_type

    # get appropriate data, metrics, and criterion for our problem
    batch_size = config["batch_size"]
    continuity_train_data, unresolved_train_data = utils.generate_data(
        batch_size=batch_size,
        n_stories=n_stories,
        n_synth=n_synth,
        data_path="data/synthetic/train",
        cache_path="data/encoded/train",
        get_kgs=use_kg,
        encoder=encoder_type,
        optimize_space=optimize_space,
        n_continuity_errors=config["n_continuity_errors"],
    )
    continuity_test_data, unresolved_test_data = utils.generate_data(
        batch_size=batch_size,
        n_stories=n_stories,
        n_synth=n_synth,
        data_path="data/synthetic/test",
        cache_path="data/encoded/test",
        get_kgs=use_kg,
        encoder=encoder_type,
        optimize_space=optimize_space,
        n_continuity_errors=config["n_continuity_errors"],
    )
    if problem_type == "continuity":
        train_data, test_data = continuity_train_data, continuity_test_data
        criterion = nn.CrossEntropyLoss()
        metrics = ["f1"]
    elif problem_type == "unresolved":
        train_data, test_data = unresolved_train_data, unresolved_test_data
        criterion = nn.MSELoss()
        metrics = ["mse"]
    else:
        raise ValueError(
            f"'{problem_type}' is not a valid problem type. Please check valid problem types via --help"
        )
    if model_type == "get" or model_type == "mac":
        # unfortunately this model is a huge n^2 memory suck TODO fix this if we can?
        # we're technically already running it in batch sizes of ~100 due to the way the model
        # was built, but whether or not we can push this up is something to look into!
        train_data = utils.create_story_dataloader(train_data.dataset, batch_size=1)
        test_data = utils.create_story_dataloader(test_data.dataset, batch_size=1)

    # create a model constructor for our loop
    def model_constructor() -> nn.Module:
        if model_type == "bert" or model_type == "bert_kg":
            if problem_type == "continuity":
                return ContinuityBERT(
                    n_heads=config["n_heads"],
                    n_layers=config["n_layers"],
                    n_gnn_layers=config["n_gnn_layers"],
                    hidden_dim=config["hidden_dim"],
                    input_dim=utils.SENTENCE_ENCODER_DIM[encoder_type],
                    use_kg=use_kg,
                    kg_node_dim=kg_utils.KG_NODE_DIM,
                    kg_edge_dim=kg_utils.KG_EDGE_DIM,
                    dropout=config["dropout"],
                )
            elif problem_type == "unresolved":
                return UnresolvedBERT(
                    n_heads=config["n_heads"],
                    n_layers=config["n_layers"],
                    n_gnn_layers=config["n_gnn_layers"],
                    hidden_dim=config["hidden_dim"],
                    input_dim=utils.SENTENCE_ENCODER_DIM[encoder_type],
                    use_kg=use_kg,
                    kg_node_dim=kg_utils.KG_NODE_DIM,
                    kg_edge_dim=kg_utils.KG_EDGE_DIM,
                    dropout=config["dropout"],
                )
        elif model_type == "lstm":
            if problem_type == "continuity":
                # TODO tune this model
                config["n_layers"] = 3
                config["hidden_dim"] = 300
                return BaselineLSTM(
                    n_layers=config["n_layers"],
                    input_dim=utils.SENTENCE_ENCODER_DIM[encoder_type],
                    hidden_dim=config["hidden_dim"],
                )
        elif model_type == "get":
            if problem_type == "continuity":
                model_config = {}
                model_config["cuda"] = device == "cuda"
                model_config["embedding"] = None
                model_config["embedding_input_dim"] = 0
                model_config["embedding_output_dim"] = 100
                # This is never used so not sure why GET devs included it
                model_config["num_classes"] = 2
                model_config["output_size"] = 1
                model_config["fixed_length_left"] = 30
                model_config["fixed_length_right"] = 100
                model_config["use_claim_source"] = 0
                model_config["use_article_source"] = 0
                model_config["num_att_heads_for_words"] = 1  # first level
                model_config["num_att_heads_for_evds"] = 1  # second level
                model_config["dropout_gnn"] = 0.5
                model_config["dropout_left"] = 0.2
                model_config["dropout_right"] = 0.2
                model_config["hidden_size"] = 300
                model_config["gsl_rate"] = 0.8
                return GraphBasedSemanticStructure(model_config)
        elif model_type == "mac":
            if problem_type == "continuity":
                model_config = {}
                model_config["cuda"] = device == "cuda"
                model_config["embedding"] = None
                model_config["embedding_input_dim"] = 0
                model_config["embedding_output_dim"] = 100
                model_config["num_classes"] = 2
                model_config["output_size"] = 1
                model_config["fixed_length_left"] = 30
                model_config["fixed_length_right"] = 100
                model_config["use_claim_source"] = 0
                model_config["use_article_source"] = 0
                model_config["num_att_heads_for_words"] = 1  # first level
                model_config["num_att_heads_for_evds"] = 1  # second level
                model_config["dropout_gnn"] = 0.5
                model_config["dropout_left"] = 0.2
                model_config["dropout_right"] = 0.2
                model_config["hidden_size"] = 300
                model_config["gsl_rate"] = 0.8
                return HierachicalMultiHeadAttentionModel(model_config)
        elif model_type == "textcnn":
            if problem_type == "continuity":
                model_config = {}
                model_config["sentence_max_size"] = 10  # max tokens/sent
                model_config["label_num"] = 1
                return TextCNN(model_config)
        elif model_type == "declare":
            if problem_type == "continuity":
                nb_lstm_units = 64
                glove = vocab.GloVe(name="6B", dim=100)
                glove_embeddings = nn.Embedding.from_pretrained(glove.vectors)
                claim_source_vocab_size = article_source_vocab_size = len(glove.vectors)
                return DeClareModel(
                    glove.vectors.numpy(),
                    claim_source_vocab_size,
                    article_source_vocab_size,
                    nb_lstm_units,
                )  # TODO implement this

        # default case -- model is unimplemented
        raise ValueError(f"{model_type} not implemented for {problem_type}")

    return model_constructor, train_data, test_data, criterion, metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gen_data_only", action="store_true", help="only generate data, no training"
    )
    parser.add_argument(
        "--n_stories",
        default=1000,
        type=int,
        help="number of stories to use (for both test and train)",
    )
    parser.add_argument(
        "--n_synth",
        default=1,
        type=int,
        help="number of synthetic datapoints to use per story",
    )
    parser.add_argument("--train_ratio", default=0.5, type=float, help="train ratio")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument(
        "--n_continuity_errors", default=1, type=int, choices=[1, 2, 5, 10]
    )
    parser.add_argument("--n_heads", default=8, type=int)
    parser.add_argument("--n_layers", default=3, type=int)
    parser.add_argument("--n_gnn_layers", default=2, type=int)
    parser.add_argument("--hidden_dim", default=20, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--n_runs", default=5, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--pr_threshold", default=0.3, type=float)
    parser.add_argument(
        "--encoder_type",
        default="all-MiniLM-L6-v2",
        type=str,
        choices=list(utils.SENTENCE_ENCODER_DIM.keys()),
    )
    parser.add_argument(
        "--model_type",
        default="continuity_bert",
        type=str,
        choices=["bert", "bert_kg", "lstm", "get", "mac", "declare", "textcnn"],
    )
    parser.add_argument(
        "--problem_type",
        default="continuity",
        type=str,
        choices=["continuity", "unresolved"],
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--optimize_space",
        default=False,
        type=bool,
        help="if a _kg dataset is already generated, reuse it for non_kg data",
    )
    parser.add_argument(
        "--verbosity",
        default=1,
        type=int,
        help="verbosity of output if != 0; lower is more verbose",
    )
    parser.add_argument(
        "--debug",
        default=False,
        type=bool,
        help="run in debug mode (no real training)",
    )
    parser.add_argument(
        "--settings_json",
        default="",
        type=str,
        help="JSON with optimal settings for the given model",
    )
    config = parser.parse_args()
    config = vars(config)
    settings_json = config.get("settings_json", "")
    if settings_json != "":
        with open(settings_json, "r") as f:
            user_provided_settings = json.load(f).get(config["model_type"], {})
        config.update(user_provided_settings)
    return config


if __name__ == "__main__":
    """
    create and train baseline continuity and unresolved error models
    """
    ### hyperparameters ###
    config = parse_args()
    set_seed(config["seed"])
    model_type = config["model_type"]
    train_ratio = config["train_ratio"]
    PR_THRESHOLD = config["pr_threshold"]

    # read data
    batch_size = config["batch_size"]
    n_stories = config["n_stories"]
    n_synth = config["n_synth"]
    use_kg = "kg" in model_type
    encoder_type = config["encoder_type"]
    optimize_space = config["optimize_space"]
    gen_data_only = config["gen_data_only"]
    print("reading data...")
    print("done.")

    # create training artifacts
    print("creating training artifacts...")
    (
        model_constructor,
        train_data,
        test_data,
        criterion,
        metrics,
    ) = get_training_artifacts(config)
    print("done.")
    if gen_data_only:
        print("gen_data_only is True. skipping training and exiting.")
        sys.exit(0)

    # start runs
    print(f"training {model_type} model...")
    all_runs_metrics = []
    for i in range(config["n_runs"]):
        print(f"run {i+1} start -- seed={config['seed']}")
        # create model
        model = model_constructor()
        model = model.to(device)
        opt = Adam(model.parameters(), lr=config["lr"])
        # train model
        best_test_metrics = train(
            model=model,
            train_data=train_data,
            test_data=test_data,
            opt=opt,
            criterion=criterion,
            epochs=config["n_epochs"],
            metrics=metrics,
            verbosity=config["verbosity"],
        )
        all_runs_metrics.append(best_test_metrics)
        config["seed"] += 1
    for i in range(len(all_runs_metrics)):
        print(f"run {i+1}: {all_runs_metrics[i]}")
    print(f"done.")

    # calculate final metrics
    UNRESOLVED_ERROR_HUMAN_BENCHMARK = 2.51e-3
    UNRESOLVED_ERROR_RANDOM_MODEL = 1.37e-2
    CONTINUITY_ERROR_HUMAN_BENCHMARK = 0.5
    CONTINUITY_ERROR_RANDOM_MODEL = 0.026
    confidence_interval_95_zval = 1.96
    is_continuity_problem = config["problem_type"] == "continuity"
    main_metric = "f1" if is_continuity_problem else "mse"
    all_runs_main_metric = [
        single_run_metric[main_metric] for single_run_metric in all_runs_metrics
    ]
    if not is_continuity_problem:
        t_human, p_human = ttest_1samp(
            all_runs_main_metric, UNRESOLVED_ERROR_HUMAN_BENCHMARK, alternative="less"
        )
        t_random, p_random = ttest_1samp(
            all_runs_main_metric, UNRESOLVED_ERROR_RANDOM_MODEL, alternative="less"
        )
    else:
        t_human, p_human = ttest_1samp(
            all_runs_main_metric, CONTINUITY_ERROR_HUMAN_BENCHMARK, alternative="less"
        )
        t_random, p_random = ttest_1samp(
            all_runs_main_metric, CONTINUITY_ERROR_RANDOM_MODEL, alternative="less"
        )
    print(f"t,p-val for human<model: {t_human},{p_human}, significant: {p_human<0.05}")
    print(
        f"t,p-val for random<model: {t_random},{p_random}, significant: {p_random<0.05}"
    )
    std_dev = np.std(all_runs_main_metric)
    mean = np.mean(all_runs_main_metric)
    print(f"95% CI: {mean}+/-{std_dev*confidence_interval_95_zval}")
