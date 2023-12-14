# evaluate.py
"""
Used to... evaluate...
"""
import os
import numpy as np
from argparse import ArgumentParser
from collections import OrderedDict

import h5py
import torch
import torch.nn as nn

from models import SimpleModel, TwoLayerLSTM
from datafactory import make_windowed_data
from config import WINDOW_SIZE, DEVICE, VOCAB_SIZE, COLLECTION_NAME, holdouts



def evaluate(model, X, y):
    total_values = len(y)
    total_correct = 0
    with torch.no_grad():
        prediction = model(X)
        for x, y_ in zip(prediction, y):
            if y_ == x.argmax():
                total_correct += 1
    return (total_correct, total_values)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', choices=['single', 'double'], default='single')
    parser.add_argument('--window', choices=[5,15,25], type=int)
    args = parser.parse_args()
    if args.model == 'single':
        Model = SimpleModel
    elif args.model == 'double':
        Model = TwoLayerLSTM
    WINDOW_SIZE = args.window
    model = Model(vocab_size=VOCAB_SIZE)
    state_dict = torch.load(f"saved_models/{args.model}_dancedance_{COLLECTION_NAME}_{WINDOW_SIZE}.pth", map_location=DEVICE)

    # Loading a state dict that was made using parallelized data, onto a single device.
    # Sigh.
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
    best_model = model.load_state_dict(state_dict=new_state_dict)


    raw_test_data = []
    raw_train_data = []
    collection_name = COLLECTION_NAME
    skip_train_eval = True

    print(f"Evaluating {COLLECTION_NAME} with window size {WINDOW_SIZE}.")
    with h5py.File("data.hdf5") as hfile:
        collection = hfile[collection_name]
        for pack in collection.values():
            for simfile in pack.values():
                # Oof, not my favorite way of doing this.
                if simfile.attrs['title'] not in holdouts[collection_name]:
                    if skip_train_eval:
                        continue
                    # My computer is not strong enough to handle this.
                    for chart in simfile.values():
                        raw_train_data.append(chart[...])
                else:
                    for chart in simfile.values():
                        raw_test_data.append(chart[...])

    model.eval()

    print(f"If we guessed randomly our accuracy would have been {1/VOCAB_SIZE:.2%}.")

    X_test, y_test = make_windowed_data(raw_test_data, normalize=True, window_size=WINDOW_SIZE)
    total_test_correct, total_test_values = evaluate(model, X_test, y_test)
    print(f"Accuracy, in the end, for test data: {total_test_correct}/{total_test_values}, {total_test_correct/total_test_values:.2%}")

    if not skip_train_eval:
        X_train, y_train = make_windowed_data(raw_train_data, normalize=True, window_size=WINDOW_SIZE)
        total_train_correct, total_train_values = evaluate(model, X_train, y_train)
        print(f"Accuracy, in the end, for train data: {total_train_correct}/{total_train_values}, {total_train_correct/total_train_values:.2%}")
