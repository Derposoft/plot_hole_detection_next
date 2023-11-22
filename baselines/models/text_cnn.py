# -*- coding: utf-8 -*-

"""
Mostly copied from https://github.com/Cheneng/TextCNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.vocab as vocab


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def forward(self):
        pass


class TextCNN(BasicModule):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        # need to embed words since this is a word-by-word model
        glove = vocab.GloVe(name="6B", dim=100)
        self.embedding = nn.Embedding.from_pretrained(glove.vectors)
        config["word_embedding_dimension"] = 100

        # original parameters
        self.config = config
        # self.out_channel = config["out_channel"]
        self.conv3 = nn.Conv2d(1, 1, (3, config["word_embedding_dimension"]))
        self.conv4 = nn.Conv2d(1, 1, (4, config["word_embedding_dimension"]))
        self.conv5 = nn.Conv2d(1, 1, (5, config["word_embedding_dimension"]))
        self.Max3_pool = nn.MaxPool2d((config["sentence_max_size"] - 3 + 1, 1))
        self.Max4_pool = nn.MaxPool2d((config["sentence_max_size"] - 4 + 1, 1))
        self.Max5_pool = nn.MaxPool2d((config["sentence_max_size"] - 5 + 1, 1))
        self.linear1 = nn.Linear(3, config["label_num"])

    def forward(self, _, documents: torch.Tensor, **kargs):
        x = self.embedding(documents.long()) # (batch_size, n_sentences, num_words, embedding_size)
        batch_size, n_sent_per_story, n_words, embed_size = x.shape
        x = x.view(-1, n_words, embed_size).unsqueeze(1)

        # Convolution
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))

        # Pooling
        x1 = self.Max3_pool(x1)
        x2 = self.Max4_pool(x2)
        x3 = self.Max5_pool(x3)

        # capture and concatenate the features
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch_size, n_sent_per_story, -1)

        # project the features to the labels
        x = self.linear1(x)
        x = x.view(batch_size, -1)
        return x


if __name__ == "__main__":
    print("running the TextCNN and BasicModule...")
    model = BasicModule()
