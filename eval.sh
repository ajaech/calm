#!/usr/bin/bash

# data="/g/ssli/data/LowResourceLM/tweets/val.tsv.gz"
for path in exps/bloom*; do
    [ -d "${path}" ] || continue # if not a directory, skip
    dirname="$(basename "${path}")"
    
    echo $dirname
    ./rnnlm.py --mode=eval exps/$dirname --threads 4 > exps/$dirname/ppl.txt 2> exps/$dirname/error.ppl.log
done
