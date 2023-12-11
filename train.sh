#!/bin/bash

python train.py --model double
WINDOW_SIZE=15 python train.py --model double
WINDOW_SIZE=25 python train.py --model double
COLLECTION_NAME=fraxtil python train.py --model double
COLLECTION_NAME=fraxtil WINDOW_SIZE=15 python train.py --model double
COLLECTION_NAME=fraxtil WINDOW_SIZE=25 python train.py --model double
