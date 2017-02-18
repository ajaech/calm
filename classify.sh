#!/usr/bin/bash

data="/g/ssli/data/LowResourceLM/tweets/val.tsv.gz"

for path in exps/tweet*; do
    [ -d "${path}" ] || continue # if not a directory, skip
    dirname="$(basename "${path}")"

    echo $dirname
    ./rnnlm.py --mode=classify --data $data exps/$dirname --threads 6 \
        --partition_override True > exps/$dirname/f1.txt
done
