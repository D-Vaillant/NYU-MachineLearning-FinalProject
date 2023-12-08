""" datafactory.py
It's dirty work, but someone's gotta do it.
"""

import simfile
from typing import List, Iterable
import numpy as np
from config import WINDOW_SIZE
from itertools import islice
import torch

def numericize_input(seq: Iterable[int]) -> int:
    return sum((2**i) * v for i, v in enumerate(seq))

def yield_tokens(charts: Iterable[np.array]) -> List[str]:
    for chart in charts:
        symbols = chart[['c0', 'c1', 'c2', 'c3']]
        for row in symbols:
            yield numericize_input(row)

PAD_IDX = 0

special_symbols = ['<PAD>']

def numericize_input(seq: Iterable[int]) -> int:
    return sum((2**i) * v for i, v in enumerate(seq))

def get_symbols(song_data):
    symbols = song_data[['c0', 'c1', 'c2', 'c3']]
    return [numericize_input(row) for row in symbols]

def produce_windows(raw_data: Iterable, window_size: int):
    """
    ex. if raw_data = [1,2,3,4,5] and window_size=3, this produces:
    [1,2,3] 4
    [2,3,4] 5"""
    data_len = len(raw_data)
    if not (window_size+1 < data_len):
        return [], []
    X_data = []
    y_data = []
    # A bit of a fancy way to go about doing this.
    for i, y in enumerate(islice(raw_data, window_size, None)):
      X_data.append(raw_data[i: i+window_size])
      y_data.append(y)
    return X_data, y_data

def make_data_from_dataset(datasets, normalize=False, window_size: int=WINDOW_SIZE):
    X_data = []
    y_data = []
    for dataset in datasets:
        raw_data = get_symbols(dataset)
        raw_vocab = [i for i in range(1,16)]
        new_X_data, new_y_data = produce_windows(raw_data, window_size)
        X_data += new_X_data
        y_data += new_y_data
    # reshape X to be [samples, time steps, features]
    X = torch.tensor(X_data, dtype=torch.float32).reshape(len(X_data), window_size, 1)
    if normalize:
        X = X / float(len(raw_vocab))  # Normalize.
    else:
        X = X.int()
    y = torch.tensor(y_data)
    print(X.shape, y.shape)
    return X, y