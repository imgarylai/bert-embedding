.. bert-embedding documentation master file, created by
   sphinx-quickstart on Sat Feb  9 21:19:15 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to bert-embedding's documentation!
==========================================

`BERT <https://arxiv.org/abs/1810.04805/>`_, published by `Google <https://github.com/google-research/bert>`_ , is new way to obtain pre-trained language model word representation. Many NLP tasks are benefit from BERT to get the SOTA.

The goal of this project is to obtain the sentence and token embedding from BERT's pre-trained model. In this way, instead of building and do fine-tuning for an end-to-end NLP model, you can build your model by just utilizing the sentence or token embedding.

This project is implemented with `@MXNet <https://github.com/apache/incubator-mxnet>`_. Special thanks to `@gluon-nlp <https://github.com/dmlc/gluon-nlp>`_ team.

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference/bert_embedding

.. toctree::
   :maxdepth: 2
   :caption: BERT models

   bert_models/bert_models

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
