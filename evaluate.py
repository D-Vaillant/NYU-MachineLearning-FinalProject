# evaluate.py
"""
Used to... evaluate...
"""

import numpy as np
import torch
import torch.nn as nn
from models import SimpleModel
from datafactory import make_data_from_dataset
import h5py
from config import WINDOW_SIZE

holdouts = {
    'fraxtil': ['Let It Go', 'Mosh Pit', 'Crazy', 'Blue']
}

model = SimpleModel(vocab_size=16)
state_dict = torch.load("dancedance.pth")
best_model = model.load_state_dict(state_dict=state_dict)

raw_data = []

if __name__ == '__main__':
  with h5py.File("data.hdf5") as hfile:
      collection = hfile['fraxtil']
      #print(h5_song_data.attrs['name'])
      #print(h5_song_data.attrs['difficulty'])
      for pack in collection.values():
          for simfile in pack.values():
              # Oof, not my favorite way of doing this.
              if simfile.attrs['title'] not in holdouts['fraxtil']:
                  continue
              for chart in simfile.values():
                  raw_data.append(chart[...])

  best_model.eval()

  X, y = make_data_from_dataset(raw_data, window_size=WINDOW_SIZE)

  total_values = len(y)
  total_correct = 0
  with torch.no_grad():
      for x in X:
          prediction = best_model(x)
          if y == prediction:
              total_correct += 1

print(f"Accuracy, in the end: {total_correct}/{total_values}, {total_correct/total_values:.2%}")