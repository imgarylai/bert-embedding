import subprocess
import time


def test_bert_embedding():
    process = subprocess.check_call(['python', './bert_embedding/cli.py',
                                     '--model', 'bert_12_768_12', '--dataset_name', 'book_corpus_wiki_en_uncased',
                                     '--max_seq_length', '25', '--batch_size', '256',
                                     '--oov_way', 'avg', '--sentences', '"is this jacksonville ?"'])
    time.sleep(5)