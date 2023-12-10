# evaluate.py
"""
Used to... evaluate...
"""
import numpy as np
import torch
import torch.nn as nn
from models import SimpleModel
from datafactory import make_windowed_data
import h5py
from config import WINDOW_SIZE, DEVICE, VOCAB_SIZE

holdouts = {
    'fraxtil': ['Let It Go', 'Mosh Pit', 'Crazy', 'Blue']
}

model = SimpleModel(vocab_size=16)
state_dict = torch.load("dancedance.pth", map_location=DEVICE)
best_model = model.load_state_dict(state_dict=state_dict)

raw_data = []

if __name__ == '__main__':
  with h5py.File("data.hdf5") as hfile:
      collection = hfile['fraxtil']
      for pack in collection.values():
          for simfile in pack.values():
              # Oof, not my favorite way of doing this.
              if simfile.attrs['title'] not in holdouts['fraxtil']:
                  continue
              for chart in simfile.values():
                  raw_data.append(chart[...])

  model.eval()

  X, y = make_windowed_data(raw_data, normalize=True, window_size=WINDOW_SIZE)

  total_values = len(y)
  total_correct = 0
  with torch.no_grad():
        prediction = model(X)
        for x, y_ in zip(X, y):
          if y_ == x.argmax():
              total_correct += 1

print(f"Accuracy, in the end: {total_correct}/{total_values}, {total_correct/total_values:.2%}")
print(f"If we guessed randomly our accuracy would have been {1/VOCAB_SIZE:.2%}.")