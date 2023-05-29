#!/bin/bash

depth=16


./previewDBS.py -D $depth -r
./previewDBS.py -D $depth -ar

./postproc.py -D $depth -r
