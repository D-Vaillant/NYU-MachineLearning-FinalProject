#!/bin/bash

python evaluate.py
echo
WINDOW_SIZE=15 python evaluate.py
echo
WINDOW_SIZE=25 python evaluate.py
echo
COLLECTION_NAME=fraxtil python evaluate.py
echo
WINDOW_SIZE=15 COLLECTION_NAME=fraxtil python evaluate.py
echo
WINDOW_SIZE=25 COLLECTION_NAME=fraxtil python evaluate.py
