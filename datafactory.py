""" datafactory.py
It's dirty work, but someone's gotta do it.
"""
import logging
from itertools import islice
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from config import WINDOW_SIZE, ONLY_PAD, DEVICE


# Some hfile functions.
def iterate_through_hfile(hfile, collection_name: str,
                          skip_simfile=lambda t: False,
                          skip_chart=lambda t: False):
    """ hfile: h5py.File, collection: collection name.
    """
    collection = hfile[collection_name]
    #print(h5_song_data.attrs['name'])
    #print(h5_song_data.attrs['difficulty'])
    for pack in collection.values():
        for simfile in pack.values():
            if skip_simfile(simfile):
                continue
            for chart in simfile.values():
                if skip_chart(chart):
                    continue
                else:
                    yield chart

def numericize_input(seq: Iterable[int],
                     only_pad: bool=ONLY_PAD) -> int:
    """ If ONLY_PAD, screams at you if you include a hand or a quad. """
    # QUAD: All 4, 15.
    # HAND: 15-8, 15-4, 15-2, 15-1; 7, 11, 13, 14
    output = sum((2**i) * v for i, v in enumerate(seq))
    handquads = [7, 11, 13, 14, 15]
    padmapper = {x: y for x, y in
                 zip((set(range(1,16)) - set(handquads)), range(1, 11))}
    if not only_pad:
        pass
    elif output in handquads:
        raise Exception("HAND/QUAD found while processing")
    else:
        output = padmapper[output]
    return output

def yield_tokens(charts: Iterable[np.array]):
    """ Generator that yields integers. """
    for chart in charts:
        symbols = chart[['c0', 'c1', 'c2', 'c3']]
        for row in symbols:
            yield numericize_input(row)

def get_symbols(song_data: np.array) -> list[int]:
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

def make_windowed_data(datasets: Iterable[np.array],
                       normalize: bool=False,
                       window_size: int=WINDOW_SIZE) -> tuple[Tensor, Tensor]:
    X_data = []
    y_data = []
    for dataset in datasets:
        raw_data = get_symbols(dataset)
        maxsym = max(raw_data)
        raw_vocab = [i for i in range(1, maxsym)]
        new_X_data, new_y_data = produce_windows(raw_data, window_size)
        X_data += new_X_data
        y_data += new_y_data
    # reshape X to be [samples, time steps, features]
    X = torch.tensor(X_data, dtype=torch.float32, device=DEVICE).reshape(len(X_data), window_size, 1)
    X.to(DEVICE)
    if normalize:
        X = X / float(maxsym)  # Normalize.
    else:
        X = X.int()
    y = torch.tensor(y_data, device=DEVICE)
    print(X.shape, y.shape)
    return X, y
