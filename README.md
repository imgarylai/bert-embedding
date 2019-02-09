# Bert Embeddings

## Install

```
pip install bert_embedding
```
If you want to run on GPU machine, please install `mxnet-cu92`.

```
pip install mxnet-cu92
```

## Usage

```python
from bert_embedding import BertEmbedding

sentences = ["Hello World", "Token level embeddings from BERT model on mxnet and gluonnlp"]
bert = BertEmbedding()
result = bert.embedding(sentences)
```