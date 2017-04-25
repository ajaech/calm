#!/usr/bin/bash
# Script to help with evaluating trained models

data="../data/smallreddit.tsv"
for path in exps/*; do
    [ -d "${path}" ] || continue # if not a directory, skip
    dirname="$(basename "${path}")"

    cmd="./rnnlm.py --mode=eval $path --data=$data --threads 6 > $path/ppl.txt 2> $path/error.ppl.log"
    # Skip files that have been evaluated more recently then the last checkpoint was saved
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
