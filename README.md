# CALM
Context Aware Language Models

Code for building language models that adapt to different contexts. This code was originally written to support the experiments in the paper 
[Improving Context Aware Language Models](https://arxiv.org/abs/1704.06380). It has since been modified to support experiments for the paper "Low-Rank RNN Adaptation for Context-Aware Language Modeling" (link coming soon). Read the paper for a complete description of the model.

The main idea is that metadata or other context information can be used to adapt or control a language model. The adaptation takes place by allowing a context embedding to transform the weights of the recurrent layer of the model. We call this model the FactorCell model.

Train a model using 
`
./rnnlm.py /s0/ajaech/exptest --data ../data/dbpedia_small_train.tsv --valdata ../data/dbpedia_small_val.tsv
`
I will work on documenting the code more. Send me a message if you want some help getting started.
