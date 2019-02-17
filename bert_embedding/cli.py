import argparse
import io

import mxnet as mx
import numpy as np

from bert_embedding import BertEmbedding

__author__ = "Gary Lai"


def main():
    np.set_printoptions(threshold=5)
    parser = argparse.ArgumentParser(description='Get embeddings from BERT',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--gpu', type=int, default=None,
                        help='id of the gpu to use. Set it to empty means to use cpu.')
    parser.add_argument('--model', type=str, default='bert_12_768_12',
                        help='pre-trained model')
    parser.add_argument('--dataset_name', type=str, default='book_corpus_wiki_en_uncased',
                        help='dataset')
    parser.add_argument('--max_seq_length', type=int, default=25,
                        help='max length of each sequence')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--oov_way', type=str, default='avg',
                        help='how to handle oov\n'
                             'avg: average all oov embeddings to represent the original token\n'
                             'sum: sum all oov embeddings to represent the original token\n'
                             'last: use last oov embeddings to represent the original token\n')
    parser.add_argument('--sentences', type=str, nargs='+', default=None,
                        help='sentence for encoding')
    parser.add_argument('--file', type=str, default=None,
                        help='file for encoding')

    args = parser.parse_args()
    context = mx.gpu(args.gpu) if args.gpu else mx.cpu()
    bert = BertEmbedding(ctx=context, model=args.model, dataset_name=args.dataset_name,
                         max_seq_length=args.max_seq_length, batch_size=args.batch_size)
    result = []
    sents = []
    if args.sentences:
        sents = args.sentences
        result = bert.embedding(sents, oov_way=args.oov_way)
    elif args.file:
        with io.open(args.file, 'r', encoding='utf8') as in_file:
            for line in in_file:
                sents.append(line.strip())
        result = bert.embedding(sents, oov_way=args.oov_way)
    else:
        print('Please specify --sentence or --file')

    if result:
        for sent, embeddings in zip(sents, result):
            print('Text: {}'.format(sent))
            sentence_embedding, _, tokens_embedding = embeddings
            print('Sentence embedding: {}'.format(sentence_embedding))
            print('Tokens embedding: {}'.format(tokens_embedding))


if __name__ == '__main__':
    main()
