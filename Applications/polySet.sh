#!/bin/bash
#./polyCell.py -h

F='Quals/polyCell/'
K=6


./polyCell.py --strat fixedMax -k $K -v -f $F
./polyCell.py --strat depth -k $K -v -f $F
./polyCell.py --strat depthExt -k $K -vp -f $F
./polyCell.py --strat fixedMin -k $K -v -f $F

