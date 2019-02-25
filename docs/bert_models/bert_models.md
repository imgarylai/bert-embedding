# Available pre-trained BERT models

| |book_corpus_wiki_en_uncased|book_corpus_wiki_en_cased|wiki_multilingual|wiki_multilingual_cased|wiki_cn|
|---|---|---|---|---|---|
|bert_12_768_12|✓|✓|✓|✓|✓|
|bert_24_1024_16|x|✓|x|x|x|

## Usage

Example of using the large pre-trained BERT model from Google 

```python
from bert_embedding.bert import BertEmbedding

bert = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')
```

Source: [gluonnlp](http://gluon-nlp.mxnet.io/model_zoo/bert/index.html) 