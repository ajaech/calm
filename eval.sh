#!/usr/bin/bash

# data="/g/ssli/data/LowResourceLM/tweets/val.tsv.gz"
for path in exps/bloom*; do
    [ -d "${path}" ] || continue # if not a directory, skip
    dirname="$(basename "${path}")"

    cmd="./rnnlm.py --mode=eval $path --threads 6 > $path/ppl.txt 2> $path/error.ppl.log"
    if [ -f $path/ppl.txt ]; then        
        if test $path/model.bin -nt $path/ppl.txt; then
            echo $cmd
        fi
    else
        echo $cmd
    fi
    
done
