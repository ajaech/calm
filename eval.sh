#!/usr/bin/bash

# data="/g/ssli/data/LowResourceLM/tweets/val.tsv.gz"
for path in exps/reddit*; do
    [ -d "${path}" ] || continue # if not a directory, skip
    dirname="$(basename "${path}")"

    echo $dirname
    ./rnnlm.py --mode=eval exps/$dirname --threads 6 --partition_override True > exps/$dirname/ppl.txt
done
