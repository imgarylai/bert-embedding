from typing import List

import gluonnlp
import mxnet as mx
from gluonnlp.data import BERTTokenizer, BERTSentenceTransform
from mxnet.gluon.data import DataLoader

from bert_embedding.dataset import BertEmbeddingDataset

__author__ = "Gary Lai"


class BertEmbedding:

    def __init__(self, ctx=0, model='bert_12_768_12', dataset_name='book_corpus_wiki_en_uncased', max_seq_length=25, batch_size=256):
        self.ctx = mx.gpu(ctx)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.bert, self.vocab = gluonnlp.model.get_model(model, dataset_name=dataset_name,
                                                         pretrained=True, ctx=self.ctx, use_pooler=True,
                                                         use_decoder=False, use_classifier=False)

    def embedding(self, sentences: List[str], oov='sum'):
        iter = self.data_loader(sentences=sentences, batch_size=self.batch_size)
        batches = []
        for batch_id, (token_ids, valid_length, token_types) in enumerate(iter):
            token_ids = token_ids.as_in_context(self.ctx)
            valid_length = valid_length.as_in_context(self.ctx)
            token_types = token_types.as_in_context(self.ctx)
            sequence_outputs, pooled_outputs = self.bert(token_ids, token_types, valid_length.astype('float32'))
            [batches.append((token_id, sequence_output, pooled_output)) for token_id, sequence_output, pooled_output in zip(token_ids.asnumpy(), sequence_outputs.asnumpy(), pooled_outputs.asnumpy())]
            # batches.append((token_ids, sequence_outputs))
        # return batches
        if oov == 'sum':
            return self.oov_sum(batches)
        else:
            return self.oov_last(batches)

    def data_loader(self, sentences, batch_size, shuffle=False):
        tokenizer = BERTTokenizer(self.vocab)
        transform = BERTSentenceTransform(tokenizer=tokenizer, max_seq_length=self.max_seq_length, pair=False)
        dataset = BertEmbeddingDataset(sentences, transform)
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    def oov_sum(self, batches):
        # batch:
        #   token_ids (max_seq_length, ),
        #   sequence_outputs (max_seq_length, dim, )
        #   pooled_output (dim, )
        sentences = []
        for token_ids, sequence_outputs, pooled_output in batches:
            tokens = []
            tensors = []
            for token_id, sequence_output in zip(token_ids, sequence_outputs):
                if token_id == 1:
                    break
                if token_id == 2 or token_id == 3:
                    continue
                token = self.vocab.idx_to_token[token_id]
                if token.startswith('##'):
                    token = token[2:]
                    tokens[-1] += token
                    tensors[-1] += sequence_output
                else:
                    tokens.append(token)
                    tensors.append(sequence_output)
            sentences.append((pooled_output, tokens, tensors))
        return sentences

    def oov_last(self, batches):
        sentences = []
        for token_ids, sequence_outputs, pooled_output in batches:
            tokens = []
            tensors = []
            for token_id, sequence_output in zip(token_ids, sequence_outputs):
                if token_id == 1:
                    break
                if token_id == 2 or token_id == 3:
                    continue
                token = self.vocab.idx_to_token[token_id]
                if token.startswith('##'):
                    token = token[2:]
                    tokens[-1] += token
                    tensors[-1] = sequence_output
                else:
                    tokens.append(token)
                    tensors.append(sequence_output)
            sentences.append((pooled_output, tokens, tensors))
        return sentences
