
import argparse
import math
from multiprocessing import Pool
import os
import pickle as pkl
import shutil
import sys
from torch.utils.data import ConcatDataset

import data.utils as utils

def process_data_batch(docs_path, batch_id) -> str:
    utils.generate_data(
        batch_size = 8,
        data_path = docs_path,
        cache_path = ".",
        n_stories=5,
        n_synth=1,
        get_kgs = True,
    )
    return os.path.join(docs_path, f"data_{batch_id}.pkl")


def stitch_processed_data_batches(results):
    # Stitch together all of the processed data batches that were outputted from process_data_batch calls
    print(f"stitching together: {results}")
    datasets = []
    for result in results:
        with open(result, "rb") as f:
            continuity_dataset = pkl.load(f)
            datasets.append(continuity_dataset)
    dataset = ConcatDataset(datasets)
    return dataset


def process_all_data(data_path, batch_size=100, tempdirs="temp", n_cont_errors=1):
    files = [x for x in os.listdir(data_path) if x.endswith(".txt") and f"{n_cont_errors}-err" in x]
    n_files = len(files)
    n_batches = math.ceil(n_files / batch_size)

    # Copy files in each batch to a temporary directory
    if not os.path.exists(tempdirs):
        os.mkdir(tempdirs)
    temp_dirs = []
    for batch_idx in range(n_batches):
        docs = files[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        tempdir = os.path.join(tempdirs, f"temp_{batch_idx}")
        os.mkdir(tempdir)
        for doc in docs:
            shutil.copy(os.path.join(data_path, doc), tempdir)
        temp_dirs.append(tempdir)
    print(f"found {n_batches} batches. temp dirs: {temp_dirs}")
    exit(0)

    # Fire off subprocesses to process all batches
    with Pool(len(temp_dirs)) as pool:
        all_results = pool.starmap(
            process_data_batch, zip(temp_dirs, list(range(len(temp_dirs))))
        )

    # Stitch together data batches
    return stitch_processed_data_batches(all_results)


def parse_args():
    parser = argparse.ArgumentParser(description="process data in azure in batches")
    parser.add_argument("input_dir",type=str)
    parser.add_argument("--batch_size", type=int, default=100, required=False)
    parser.add_argument("--n_cont_errors", type=int, default=1, required=False)
    parser.add_argument(
        "--clear", action="store_true", help="clear tempdirs from prev runs"
    )
    parser.add_argument("--fix", action="store_true", help="fix a patched up directory")
    args = parser.parse_args()
    return args


def get_tempdir(args):
    return args.input_dir.replace("/", "_") + f"{args.n_cont_errors}-errs"

def attempt_fix_operation(args):
    tempdir = get_tempdir(args)
    input(f"tempdir={tempdir}. [Y]/Ctrl+C")
    dirs = os.listdir(tempdir)

    # Try to regenerate pickles that weren't generated the first time around
    pkls, failed_pkls = [], []
    for d in dirs:
        pkl_dir = os.path.join(tempdir, d)
        pkl_files = [x for x in os.listdir(pkl_dir) if x.endswith(".pkl")]
        assert len(pkl_files) <= 1
        if len(pkl_files) == 1:
            pkl_file = pkl_files[0]
            pkl_path = os.path.join(tempdir, d, pkl_file)
            pkls.append(pkl_path)
        else:
            failed_pkls.append(pkl_path)
    
    if failed_pkls:
        print("list of failed directories:")
        for failed in failed_pkls:
            print(failed)
        exit(0)
    
    # Stitch together pickles
    continuity_dataset = stitch_processed_data_batches(pkls)
    with open(os.path.join(tempdir, f"knowledge_graphs-{tempdir}.pkl"), "wb") as f:
        pkl.dump(continuity_dataset, f)
    sys.exit()


if __name__ == "__main__":
    args = parse_args()
    if args.fix:
        attempt_fix_operation(args)

    # Clear old data
    tempdir = get_tempdir(args)
    input(f"tempdir={tempdir}. [Y]/Ctrl+C")
    if tempdir[-1] == "_":
        tempdir = tempdir[:-1]
    if args.clear:
        if os.path.exists(tempdir):
            shutil.rmtree(tempdir)

    # Process all data in the given input directory at the given batch size
    continuity_dataset = process_all_data(
        args.input_dir, batch_size=args.batch_size, tempdirs=tempdir, n_cont_errors=args.n_cont_errors
    )

    # Save results locally
    with open(os.path.join(tempdir, f"knowledge_graphs-{tempdir}.pkl"), "wb") as f:
        pkl.dump(continuity_dataset, f)
