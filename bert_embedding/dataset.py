# coding=utf-8
"""BERT embedding dataset."""

from typing import List

from mxnet.gluon.data import Dataset

__author__ = 'Gary Lai'


class BertEmbeddingDataset(Dataset):
    """Dataset for BERT Embedding

    Parameters
    ----------
    sentences : List[str].
        Sentences for embeddings.
    transform : BERTDatasetTransform, default None.
        transformer for BERT input format
    """

    def __init__(self, sentences: List[str], transform=None):
        self.sentences = sentences
        self.transform = transform

    def __getitem__(self, idx):
        sentence = (self.sentences[idx], 0)
        if self.transform:
            return self.transform(sentence)
        else:
            return sentence

    def __len__(self):
        return len(self.sentences)
