""" config.py
Contains things we import everywhere else.
"""
import torch

WINDOW_SIZE = 20
PAD_IDX = 0

# Apple Silicon support.
if torch.backends.mps.is_available():
    using_silicon = True
    DEVICE = torch.device('mps:0')
else:
    using_silicon = False
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(DEVICE)

# Filter by title.
holdouts = {
    'fraxtil': ['Let It Go', 'Mosh Pit', 'Crazy', 'Blue']
}