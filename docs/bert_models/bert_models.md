# Available pre-trained BERT models

| |book_corpus_wiki_en_uncased|book_corpus_wiki_en_cased|wiki_multilingual
|---|---|---|---|
|bert_12_768_12|✓|✓|✓|
|bert_24_1024_16|x|✓|x|

## Usage

```python
# Use the large pre-trained BERT model from Google
from bert_embedding.bert import BertEmbedding

bert = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')
```

Source: [gluonnlp](http://gluon-nlp.mxnet.io/model_zoo/bert/index.html) 