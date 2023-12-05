import math
import torch
import torch.nn as nn

from preprocess import PREFIX_TO_TRAFFIC_ID, PREFIX_TO_APP_ID, AUX_ID
from train import train_op


class CustomEmbedding(nn.Module):
    """Embedding layer"""

    def __init__(self, n_channels: int, n_dims: int):
        """
        Args:
            n_channels: Channels of embedding.
            n_dims: The dimensions of embedding.
        """
        super(CustomEmbedding, self).__init__()
        self.n_channels = n_channels
        self.n_dims = n_dims
        self.embedding = nn.Linear(self.n_dims, self.n_dims)

    def forward(self, input_data):
        """

        Args:
            input_data: Intput data with shape(B, C, L)

        Returns:
            Data with shape(B, Channel, Dimension)
        """
        input_data = input_data.reshape(-1, self.n_channels, self.n_dims)
        embedded = self.embedding(input_data)

        return embedded


class PositionalEncoding(nn.Module):
    """Position encoding layer in transformer model"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        # ps shape ``[seq_len, batch_size, embedding_dim]``

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        pe = self.pe[:x.size(1), :].permute(1, 0, 2)
        # x = x + pe
        x = torch.add(x, pe)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Normal transformer block"""

    def __init__(
            self,
            embed_dim: int,
            n_heads: int = 6,
            d1: int = 1024,
            d2: int = 50,
            dropout: float = 0.1
    ):
        """Initialize the transformer block.

        Args:
            embed_dim: Embedding dimensions from the output of embedding layer.
            n_heads: Number of heads for multi-head attention.
            d1: The output dimensions of the first fully connected layer.
            d2: The output dimensions of the second fully connected layer.
            dropout: Dropout ratio.
        """
        super(TransformerBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                                    num_heads=n_heads,
                                                    batch_first=True)
        # (N,L,E) when batch_first=True
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, d1),
            nn.ReLU(),
            nn.Linear(d1, d2),
            nn.Dropout(dropout)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.embed_size = embed_dim

    def forward(self, x, mask=None):
        # Multihead attention
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=mask)
        attn_output = self.dropout1(attn_output)
        # Residual connection and layer normalization
        x = self.layer_norm1(x + attn_output)

        # Feedforward network
        ff_output = self.feedforward(x)
        ff_output = self.dropout2(ff_output)
        # Residual connection and layer normalization
        x = self.layer_norm2(x + ff_output)

        return x


class MTT(nn.Module):
    def __init__(
            self,
            seq_len=1500,
            embed_n=30,
            embed_d=50,
            trans_h=5,
            trans_d1=1024,
    ):
        super(MTT, self).__init__()
        self.seq_len = seq_len
        self.embed_n = embed_n
        self.embed_d = embed_d

        self.embedding = CustomEmbedding(embed_n, embed_d)
        self.pos_encoder = PositionalEncoding(embed_d, dropout=0.1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_d, nhead=trans_h, dim_feedforward=trans_d1,
                                                   batch_first=True)

        self.transformer_blk1 = nn.TransformerEncoder(encoder_layer, num_layers=7)

        # Task-specific layers
        self.task1_output = nn.Linear(seq_len, len(PREFIX_TO_TRAFFIC_ID))
        self.task2_output = nn.Linear(seq_len, len(PREFIX_TO_APP_ID))
        self.task3_output = nn.Linear(seq_len, len(AUX_ID))

    def forward(self, x):
        x = self.embedding(x)  # Shape(B, embed_n*embed_d)
        x = self.pos_encoder(x)

        x = self.transformer_blk1(x)

        x = torch.flatten(x, start_dim=1)

        output1 = self.task1_output(x)
        output2 = self.task2_output(x)
        output3 = self.task3_output(x)
        return output1, output2, output3


def train():
    model = MTT()
    task_weights = (4, 2, 1)
    train_op(model, task_weights=task_weights)


if __name__ == '__main__':
    train()
