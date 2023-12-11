import math

import torch
from torch import nn, Tensor


class SimpleModel(nn.Module):
    def __init__(self, vocab_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(256, vocab_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x

class TwoLayerLSTM(nn.Module):
    def __init__(self, vocab_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(128, vocab_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src))
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size))
        pos_embedding = torch.zeros(max_len, 1, emb_size)
        pos_embedding[:, 0, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 0, 1::2] = torch.cos(position * div_term)

        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pe', pos_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        """
        Arguments:
            token_embedding: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        positioned_embedding = token_embedding + self.pe[token_embedding.size(0)]
        return self.dropout(positioned_embedding)
