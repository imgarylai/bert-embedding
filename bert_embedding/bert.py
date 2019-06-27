# coding=utf-8
"""BERT embedding."""
import logging
from typing import List

import mxnet as mx
from mxnet.gluon.data import DataLoader
import gluonnlp
from gluonnlp.data import BERTTokenizer, BERTSentenceTransform

from bert_embedding.dataset import BertEmbeddingDataset

__author__ = 'Gary Lai'

logger = logging.getLogger(__name__)


class BertEmbedding(object):
    """
    Encoding from BERT model.

    Parameters
    ----------
    ctx : Context.
        running BertEmbedding on which gpu device id.
    dtype: str
        data type to use for the model.
    model : str, default bert_12_768_12.
        pre-trained BERT model
    dataset_name : str, default book_corpus_wiki_en_uncased.
        pre-trained model dataset
    params_path: str, default None
        path to a parameters file to load instead of the pretrained model.
    max_seq_length : int, default 25
        max length of each sequence
    batch_size : int, default 256
        batch size
    """

    def __init__(self, ctx=mx.cpu(), dtype='float32', model='bert_12_768_12',
                 dataset_name='book_corpus_wiki_en_uncased', params_path=None,
                 max_seq_length=25, batch_size=256):
        """
        Encoding from BERT model.

        Parameters
        ----------
        ctx : Context.
            running BertEmbedding on which gpu device id.
        dtype: str
        data type to use for the model.
        model : str, default bert_12_768_12.
            pre-trained BERT model
        dataset_name : str, default book_corpus_wiki_en_uncased.
            pre-trained model dataset
        params_path: str, default None
            path to a parameters file to load instead of the pretrained model.
        max_seq_length : int, default 25
            max length of each sequence
        batch_size : int, default 256
            batch size
        """
        self.ctx = ctx
        self.dtype = dtype
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        if params_path is not None:
            # Don't download the pretrained models if we have a parameter path
            pretrained = False
        else:
            pretrained = True
        self.bert, self.vocab = gluonnlp.model.get_model(model,
                                                         dataset_name=self.dataset_name,
                                                         pretrained=pretrained,
                                                         ctx=self.ctx,
                                                         use_pooler=False,
                                                         use_decoder=False,
                                                         use_classifier=False)
        self.bert.cast(self.dtype)

        if params_path:
            logger.info('Loading params from %s', params_path)
            self.bert.load_parameters(params_path, ctx=ctx, ignore_extra=True)

        lower = 'uncased' in self.dataset_name

        self.tokenizer = BERTTokenizer(self.vocab, lower=lower)
        self.transform = BERTSentenceTransform(tokenizer=self.tokenizer,
                                               max_seq_length=self.max_seq_length,
                                               pair=False)

    def __call__(self, sentences, oov_way='avg', filter_spec_tokens=True):
        """
        Get tokens, tokens embedding

        Parameters
        ----------
        sentences : List[str]
            sentences for encoding.
        oov_way : str, default avg.
            use **avg**, **sum** or **last** to get token embedding for those out of
            vocabulary words
        filter_spec_tokens : bool
            filter [CLS], [SEP] tokens.

        Returns
        -------
        List[(List[str], List[ndarray])]
            List of tokens, and tokens embedding
        """
        return self.embedding(sentences, oov_way, filter_spec_tokens)

    def embedding(self, sentences, oov_way='avg', filter_spec_tokens=True):
        """
        Get tokens, tokens embedding

        Parameters
        ----------
        sentences : List[str]
            sentences for encoding.
        oov_way : str, default avg.
            use **avg**, **sum** or **last** to get token embedding for those out of
            vocabulary words
        filter_spec_tokens : bool
            filter [CLS], [SEP] tokens.

        Returns
        -------
        List[(List[str], List[ndarray])]
            List of tokens, and tokens embedding
        """
        data_iter = self.data_loader(sentences=sentences)
        batches = []
        for token_ids, valid_length, token_types in data_iter:
            token_ids = token_ids.as_in_context(self.ctx)
            valid_length = valid_length.as_in_context(self.ctx)
            token_types = token_types.as_in_context(self.ctx)
            sequence_outputs = self.bert(token_ids, token_types,
                                         valid_length.astype(self.dtype))
            for token_id, sequence_output in zip(token_ids.asnumpy(),
                                                 sequence_outputs.asnumpy()):
                batches.append((token_id, sequence_output))
        return self.oov(batches, oov_way, filter_spec_tokens)

    def data_loader(self, sentences, shuffle=False):
        dataset = BertEmbeddingDataset(sentences, self.transform)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle)

    def oov(self, batches, oov_way='avg', filter_spec_tokens=True):
        """
        How to handle oov. Also filter out [CLS], [SEP] tokens.

        Parameters
        ----------
        batches : List[(tokens_id,
                        sequence_outputs,
                        pooled_output].
            batch   token_ids (max_seq_length, ),
                    sequence_outputs (max_seq_length, dim, ),
                    pooled_output (dim, )
        oov_way : str
            use **avg**, **sum** or **last** to get token embedding for those out of
            vocabulary words
        filter_spec_tokens : bool
            filter [CLS], [SEP] tokens.

        Returns
        -------
        List[(List[str], List[ndarray])]
            List of tokens, and tokens embedding
        """
        sentences = []
        for token_ids, sequence_outputs in batches:
            tokens = []
            tensors = []
            oov_len = 1
            for token_id, sequence_output in zip(token_ids, sequence_outputs):
                if token_id == 1:
                    # [PAD] token, sequence is finished.
                    break
                if (token_id in (2, 3)) and filter_spec_tokens:
                    # [CLS], [SEP]
                    continue
                token = self.vocab.idx_to_token[token_id]
                if token.startswith('##'):
                    token = token[2:]
                    tokens[-1] += token
                    if oov_way == 'last':
                        tensors[-1] = sequence_output
                    else:
                        tensors[-1] += sequence_output
                    if oov_way == 'avg':
                        oov_len += 1
                else:  # iv, avg last oov
                    if oov_len > 1:
                        tensors[-1] /= oov_len
                        oov_len = 1
                    tokens.append(token)
                    tensors.append(sequence_output)
            if oov_len > 1:  # if the whole sentence is one oov, handle this special case
                tensors[-1] /= oov_len
            sentences.append((tokens, tensors))
        return sentences
