#!/usr/bin/bash

#data="/g/ssli/data/LowResourceLM/tweets/val.tsv.gz"
data="/homes/ajaech/Downloads/scotus.tsv.gz"
for path in exps/scotus*; do
    [ -d "${path}" ] || continue # if not a directory, skip
    dirname="$(basename "${path}")"

    echo "echo $dirname"
    cmd="./rnnlm.py --mode=eval $path --data=$data --threads 6 > $path/ppl.txt 2> $path/error.ppl.log"
    if [ -f $path/ppl.txt ]; then        
        if test $path/model.bin -nt $path/ppl.txt; then
            echo $cmd
        fi
    else
        echo $cmd
    fi
    
done
