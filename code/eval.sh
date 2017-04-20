#!/usr/bin/bash

#data="/g/ssli/data/LowResourceLM/tweets/val.tsv.gz"
data="/homes/ajaech/Downloads/scotus.tsv.gz"
#data="/n/falcon/s0/ajaech/reddit.tsv.bz2"
for path in exps/scotus*; do
    [ -d "${path}" ] || continue # if not a directory, skip
    dirname="$(basename "${path}")"

    cmd="./rnnlm.py --mode=eval $path --data=$data --threads 6 > $path/ppl.txt 2> $path/error.ppl.log"
    if [ -f $path/ppl.txt ]; then        
        if test $path/model.bin -nt $path/ppl.txt; then
            echo "echo $dirname"
            echo $cmd
        fi
    else
        echo "echo $dirname"
        echo $cmd
    fi
    
done
