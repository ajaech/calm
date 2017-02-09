#!/usr/bin/bash

for path in exps/lang*; do
    [ -d "${path}" ] || continue # if not a directory, skip
    dirname="$(basename "${path}")"

    echo $dirname
    ./rnnlm.py --mode=classify --data dataset3.txt.gz exps/$dirname --threads 6 > exps/$dirname/f1.txt
done
