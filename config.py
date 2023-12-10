""" config.py
Contains things we import everywhere else.
"""
import torch

WINDOW_SIZE = 20
PAD_IDX = 0
ONLY_PAD = True

if ONLY_PAD:
    VOCAB_SIZE = 10
else:
    VOCAB_SIZE = 16
  
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
    'fraxtil': ('Let It Go', 'Mosh Pit', 'Crazy', 'Blue'),
    'ddr': ('Ah La La La', 'Hold Tight', 'NU FLOW', 'THE ANCIENT KING IS BACK')
}