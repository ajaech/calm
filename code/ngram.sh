#!/usr/bin/bash -e

expdir=/s0/ajaech/ngram
traindir=$expdir/train
testdir=$expdir/test
mkdir -p $expdir
mkdir -p $traindir
mkdir -p $testdir


python batcher.py --filename tweets.tsv.gz --out $traindir/train.txt --mode=train
python batcher.py --filename tweets.tsv.gz --out $testdir/eval.txt --mode=eval

cut -f1 $testdir/eval.txt > $testdir/labels.txt
# cut -f2 $testdir/eval.txt | sed -e 's/\(.\)/\1 /g' > $testdir/text.txt
cut -f2 $testdir/eval.txt > $testdir/text.txt

python vocab.py exps/lang0/word_vocab.pickle  > $expdir/vocab.txt
echo "SPACE" >> $expdir/vocab.txt

order=5

for lang in "en" "fr" "es" "eu" "ca" "gl" "pt" "und"
do
    echo $lang
    grep ^$lang $traindir/train.txt  | cut -f2 > $traindir/$lang.txt
    ngram-count -order $order -wbdiscount -text $traindir/$lang.txt -vocab $expdir/vocab.txt \
        -lm $traindir/$lang.lm
    ngram -ppl $testdir/text.txt -order $order -lm $traindir/$lang.lm -debug 1  \
        > $testdir/$lang.ppl
done
