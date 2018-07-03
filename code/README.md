
Descrpition of files:

* beam.py - functions used in beam search decoding
* char2vec.py - used for creating word embedding layers
* dataset.py - load datasets and create minibatches
* default_params.json - hyperparameter file
* factorcell.py - implementation of FactorCell
* helper.py - various helper functions
* metrics.py - helper functions for evaluation
* model.py - defines the tensorflow graph
* nn_impl.py - tensorflow sampled softmax loss (modified to handle context)
* rnnlm.py - main script to train and evaluate
* vocab.py - create, save, and load vocabularies


Train a model using 
`
./rnnlm.py path/to/expdir --data ../data/dbpedia_small_train.tsv --valdata ../data/dbpedia_small_val.tsv
`

Evaluate the model using 

`
./rnnlm.py path/to/expdir --data ../data/dbpedia_small_val.tsv --mode=eval
`