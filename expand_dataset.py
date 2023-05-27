# %%
import os
import pickle as pkl
import sys

# get input pkl file
input_data = sys.argv[1]
"""
"""
"""
input_data = "data/dataset/encoded/train/train_1_error.pkl"
input_data = "data/dataset/encoded/test/test_1_error.pkl"
input_data = "data/dataset/encoded/train/train_2_error.pkl"
input_data = "data/dataset/encoded/test/test_2_error.pkl"
input_data = "data/dataset/encoded/train/train_5_error.pkl"
input_data = "data/dataset/encoded/test/test_5_error.pkl"
"""
with open(input_data, "rb") as f:
    dataset, _ = pkl.load(f)

# %%
xs, ys = [], []
kgs = []
docs = []
for i, (x, y, kg, doc) in enumerate(dataset):
    xs.append(x)
    ys.append(y)
    kgs.append(kg)
    docs.append(doc)


# %%
def print_stats(xs, ys, docs):
    longest_x_length = max([len(x) for x in xs])
    smallest_x_length = min([len(x) for x in xs])

    longest_y_length = max([len(x) for x in ys])
    smallest_y_length = min([len(x) for x in ys])

    longest_story_length = max([len(x) for x in docs])
    smallest_story_length = min([len(x) for x in docs])

    print("x", longest_x_length, smallest_x_length)
    print("y", longest_y_length, smallest_y_length)
    print("stories", longest_story_length, smallest_story_length)


# %%
import torch
import torch.nn.functional as F

longest_story_length = max([len(x) for x in xs])
max_claim_length = 10


def pad_x(x):
    x = F.pad(x, (0, 0, 0, longest_story_length - len(x)))
    return x


def pad_y(y):
    y = F.pad(y, (0, longest_story_length - len(y)))
    return y


def pad_docs(doc):
    if len(doc) < longest_story_length:
        preprocessed_story_pad = torch.zeros(
            [longest_story_length - len(doc), max_claim_length]
        )
        doc = torch.cat([doc, preprocessed_story_pad], dim=0)
    return doc


# %%
print_stats(xs, ys, docs)
for i in range(len(xs)):
    xs[i] = pad_x(xs[i])
    ys[i] = pad_y(ys[i])
    docs[i] = pad_docs(docs[i])
print_stats(xs, ys, docs)

# %%
from data.utils import StoryDataset

fixed_dataset = StoryDataset(xs, ys, kgs, docs)

#new_data_file = input_data.replace("test/", "test_fixed/").replace(
#    "train/", "train_fixed/"
#)
new_data_file = input_data
with open(new_data_file, "wb") as f:
    pkl.dump((fixed_dataset, fixed_dataset), f)

# %%
