#!/bin/bash
#./polyCell.py -h

F='polyCell/vid2'

./polyCell.py --strat fixedMax -k 6 -v -f $F
./polyCell.py --strat depth -k 6 -v -f $F
./polyCell.py --strat depthExt -k 6 -vp -f $F
