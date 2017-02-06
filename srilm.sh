#!/usr/bin/bash

echo "preparing data"
python batcher.py --mode=eval --out /s0/ajaech/reddit.eval
python batcher.py --mode=train --out /s0/ajaech/reddit.train

echo "building model"
ngram-count -text /s0/ajaech/reddit.train -vocab /s0/ajaech/vocab.txt -order 4 -kndiscount -lm /s0/ajaech/reddit.lm -unk

echo "calculating perplexity"
ngram -ppl /s0/ajaech/reddit.eval -lm /s0/ajaech/reddit.lm -unk
