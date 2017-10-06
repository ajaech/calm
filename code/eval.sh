#!/usr/bin/bash

data="/path/to/data"

suffix="ppl"

for path in exps/*; do
    [ -d "${path}" ] || continue # if not a directory, skip
    dirname="$(basename "${path}")"

    cmd="echo $path; ./rnnlm.py --mode=eval $path --data=$data --threads 6 > $path/$suffix.txt 2> $path/error.$suffix.log"
    if [ -f $path/$suffix.txt ]; then
        if test $path/model.bin.index -nt $path/$suffix.txt; then
            echo $cmd
        fi
    else
        echo $cmd
    fi
done
