# train.py
from typing import Iterable, Tuple
from itertools import islice
import numpy as np
import torch
from torch import nn, Tensor
import torch.optim as optim
import torch.utils.data as data
import h5py
import time
import math
from config import DEVICE, using_silicon, holdouts, WINDOW_SIZE, VOCAB_SIZE
from datafactory import make_windowed_data

from models import SimpleModel, TransformerModel


# Lots of different ways to incorporate pause tokens, but all of them involve finding the beat gaps between notes.
# I'll do several approaches: 
# 1. No pause tokens. Ignore all beat values, just nonstop inputs.
# 2. Pause token for gaps longer than some threshold. Single token.
# 3. Multiple pause tokens - half beat, beat, long. Max of one between.
# 4. Two pause tokens - long and short. Multiple tokens between inputs potentially.

def simple_trainer(model, X, y, n_epochs, batch_size, loss_fn,
                   save_loc: str=f'dancedance_{WINDOW_SIZE}.pth'):
    # Shuffle set to false due to M2 bug.
    loader = data.DataLoader(data.TensorDataset(X, y),
                             generator=torch.Generator(device=DEVICE),
                             shuffle=not using_silicon, batch_size=batch_size)
    best_model = None
    best_loss = np.inf
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        loss = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                # print(X_batch)
                y_pred = model(X_batch)
                loss += loss_fn(y_pred, y_batch)
            if loss < best_loss:
                best_loss = loss
                best_model = model.state_dict(
                )
            print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

    torch.save(best_model, save_loc)


def transformer_trainer(model: nn.Module, train_data,
                        criterion, optimizer, scheduler,
                        epoch: int = 0):
    model.train()
    log_interval = 200

    total_loss = 0.
    start_time = time.time()

    num_batches = len(train_data) // WINDOW_SIZE

    for batch, i in enumerate(range(0, train_data.size(0) - 1, WINDOW_SIZE)):
        data, targets = get_batch(train_data, i)
        output = model(data)
        output_flat = output.view(-1, VOCAB_SIZE)
        loss = criterion(output_flat, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data


def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(WINDOW_SIZE, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


if __name__ == "__main__":
    raw_data = []
    collection_name = 'ddr'
    with h5py.File("data.hdf5") as hfile:
        # for chart in iterate_through_hfile(hfile, collection_name,
        #                       skip_simfile=lambda s: s.attrs['title'] in holdouts[collection_name]):
        #     raw_data.append(chart[...])
        collection = hfile[collection_name]
        for pack in collection.values():
            for simfile in pack.values():
                if simfile.attrs['title'] in holdouts[collection_name]:
                    continue
                for chart in simfile.values():
                    raw_data.append(chart[...])

    #X, y = make_data_from_dataset(raw_data)

    n_epochs = 40
    batch_size = 128
    ntokens = VOCAB_SIZE  # size of vocabulary
    emsize = 10  # embedding dimension
    d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 2  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.2  # dropout probability


    # train_data = Tensor(get_symbols(raw_data[2])).int().to(DEVICE)

    if False:
        model = TransformerModel(VOCAB_SIZE, emsize, nhead, d_hid, nlayers, dropout).to(DEVICE)
        lr = 5.0  # learning rate
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

        for epoch in range(n_epochs):
            transformer_trainer(model, train_data,
                                loss_fn, optimizer, scheduler,
                                epoch)
    else:
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        model = SimpleModel(vocab_size=VOCAB_SIZE)
        optimizer = optim.Adam(model.parameters())
        X, y = make_windowed_data(raw_data, normalize=True, window_size=WINDOW_SIZE)
        simple_trainer(model, X, y, n_epochs, batch_size,
                       loss_fn=loss_fn)