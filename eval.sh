#!/usr/bin/bash

for path in exps/*; do
    [ -d "${path}" ] || continue # if not a directory, skip
    dirname="$(basename "${path}")"

    echo $dirname
    ./rnnlm.py --mode=eval exps/$dirname --threads 6 > exps/$dirname/ppl.txt
done
