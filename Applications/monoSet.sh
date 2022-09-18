#!/bin/bash
#./polyCell.py -h

F='Quals/monoCell/'
K=1
n=0
s=0

./polyCell.py --strat fixedMax -k $K -v -f $F -n $n -s $s
./polyCell.py --strat depth -k $K -v -f $F -n $n -s $s
./polyCell.py --strat depthExt -k $K -v -f $F -n $n -s $s
./polyCell.py --strat fixedMin -k $K -vp -f $F -n $n -s $s
