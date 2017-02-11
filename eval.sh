#!/usr/bin/bash

for path in exps/lang*; do
    [ -d "${path}" ] || continue # if not a directory, skip
    dirname="$(basename "${path}")"

    echo $dirname
    ./rnnlm.py --mode=eval exps/$dirname --threads 5 --data dataset3.txt.gz > exps/$dirname/ppl.txt
done
