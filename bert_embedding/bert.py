# ========================================================================
# Copyright 2019 ELIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
from typing import List

import mxnet as mx
from mxnet.gluon.data import Dataset, DataLoader
from gluonnlp.data import BERTSentenceTransform, BERTBasicTokenizer

__author__ = "Gary Lai"

class BertEmbeddingDataset(Dataset):

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


class BertEmbedding:

    def __init__(self, ctx=0, max_seq_length=25, batch_size=256):
        self.ctx = mx.gpu(ctx)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def embedding(self, sentences: List[List[str]]):
        tokenizer = BERTBasicTokenizer()
        tramsform = BERTSentenceTransform(tokenizer=tokenizer, max_seq_length=self.max_seq_length)
