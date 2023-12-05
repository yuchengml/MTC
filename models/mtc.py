import torch
import torch.nn as nn
import torch.nn.functional as F

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


class Bottleneck(nn.Module):
    """Bottleneck block in 1D-CNN blocks"""

    def __init__(self, in_channels, mid_channels, out_channels, residual_channels=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.conv2 = nn.Conv1d(mid_channels + residual_channels, mid_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(mid_channels)
        self.conv3 = nn.Conv1d(mid_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.07)

    def forward(self, x, residual_1=None):
        residual_input = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        if residual_1 is not None:
            x = torch.concat((x, residual_1), dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        residual_2 = self.dropout(x)

        x = self.conv3(residual_2)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Residual connection
        x += residual_input
        return x, residual_2


class FusionDownBlock(nn.Module):
    """Fusion block for down-sampling which connect 1D-CNN to transformer block."""

    def __init__(self,
                 down_cnn_in_channels, down_cnn_out_channels):
        super(FusionDownBlock, self).__init__()

        # 1x1 convolution to match dimensions
        self.conv_down = nn.Conv1d(down_cnn_in_channels, down_cnn_out_channels, kernel_size=1)
        self.layer_norm = nn.LayerNorm([down_cnn_out_channels, 50])

    def forward(self, cnn_features):
        # Down-sample
        # Match dimensions using 1x1 convolution for CNN features
        down_sampled_features = self.conv_down(cnn_features)
        # Down-sample CNN features using average pooling
        down_sampled_features = F.avg_pool1d(down_sampled_features, kernel_size=down_sampled_features.size(1))
        down_sample_out = self.layer_norm(down_sampled_features)

        return down_sample_out


class FusionUpBlock(nn.Module):
    """Fusion block for up-sampling which connect transformer block to 1D-CNN."""

    def __init__(self, up_cnn_in_channels, up_cnn_out_channels, interpolate_size):
        super(FusionUpBlock, self).__init__()
        self.interpolate_size = interpolate_size
        self.conv_up = nn.Conv1d(up_cnn_in_channels, up_cnn_out_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm1d(up_cnn_out_channels)

    def forward(self, transformer_features):
        # Up-sample
        up_sampled_features = self.conv_up(transformer_features)
        up_sampled_features = self.batch_norm(up_sampled_features)
        # Transformer features using interpolation
        up_sampled_out = F.interpolate(
            up_sampled_features, size=self.interpolate_size, mode='linear', align_corners=True
        )

        return up_sampled_out


class MTC(nn.Module):
    """Main network to construct transformer, 1D-CNN and fusion blocks"""

    def __init__(
            self,
            seq_len: int = 1500,
            embed_n: int = 30,
            embed_d: int = 50,
            trans_h: int = 5,
            trans_d1: int = 1024
    ):
        """ Initialize MTC model with expected inputs with shape(B, C, L). `B` is batch size, `C` is channel and
         `L` is sequence length.

        Args:
            seq_len: Sequence length.
            embed_n: Embedding channels.
            embed_d: Embedding dimensions.
            trans_h: Number of heads in transformer block.
            trans_d1: Dimensions of the first feedforward net in transformer block.
        """
        super(MTC, self).__init__()
        self.seq_len = seq_len
        self.embed_n = embed_n
        self.embed_d = embed_d

        # Create embedding layer
        self.embedding = CustomEmbedding(embed_n, embed_d)

        # Define the transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_d, nhead=trans_h, dim_feedforward=trans_d1,
                                                   batch_first=True)

        # Create transformer blocks
        self.transformer_blk1 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.transformer_blk2 = nn.TransformerEncoder(encoder_layer, num_layers=5)
        self.transformer_blk3 = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Create each bottleneck as 1D-CNN blocks
        self.cnn_blk1_b1 = Bottleneck(1, 50, 100)
        self.cnn_blk2_b1 = Bottleneck(100, 50, 100)
        self.cnn_blk2_b2 = Bottleneck(100, 50, 100, residual_channels=50)
        self.cnn_blk3_b1 = Bottleneck(100, 100, 100)  # paper said third dim should be 200
        self.cnn_blk3_b2 = Bottleneck(100, 100, 100, residual_channels=100)  # paper said third dim should be 200

        # Create fusion blocks
        self.fusion_blk1_up = FusionUpBlock(30, 50, interpolate_size=seq_len)
        self.fusion_blk1_down = FusionDownBlock(50, 30)

        self.fusion_blk2_up = FusionUpBlock(30, 100, interpolate_size=seq_len)
        self.fusion_blk2_down = FusionDownBlock(100, 30)

        # Create layer normalization layers
        self.layer_norm_1 = nn.LayerNorm([30, 50])
        self.layer_norm_2 = nn.LayerNorm([30, 50])

        # Down sample for cnn
        self.fc = nn.Linear(seq_len, 50)
        self.norm = nn.BatchNorm1d(100)

        # Task-specific layers
        self.task1_output = nn.Linear(50 * 100, len(PREFIX_TO_TRAFFIC_ID))
        self.task2_output = nn.Linear(seq_len, len(PREFIX_TO_APP_ID))
        self.task3_output = nn.Linear(50 * 100, len(AUX_ID))

    def forward(self, x):
        t_x = self.embedding(x)  # Shape(batch, embed_n, embed_d)  # (128, 30, 50)
        t_x = self.transformer_blk1(t_x)  # (128, 30, 50)
        c_x, _ = self.cnn_blk1_b1(x)  # (128, 100, 1500)

        c_x, residual_c_x = self.cnn_blk2_b1(c_x)  # (128, 100, 1500), (128, 50, 1500)
        residual_c_x = self.fusion_blk1_down(residual_c_x)  # (128, 30, 50)
        t_x = self.layer_norm_1(t_x + residual_c_x)  # (128, 30, 50)
        t_x = self.transformer_blk2(t_x)  # (128, 30, 50)
        residual_t_x = self.fusion_blk1_up(t_x)  # (128, 50, 1500)
        c_x, _ = self.cnn_blk2_b2(c_x, residual_t_x)  # (128, 100, 1500), (128, 100, 1500)

        c_x, residual_c_x = self.cnn_blk3_b1(c_x)  # (128, 100, 1500), (128, 100, 1500)
        residual_c_x = self.fusion_blk2_down(residual_c_x)  # (128, 30, 50)
        t_x = self.layer_norm_2(t_x + residual_c_x)  # (128, 30, 50)
        t_x = self.transformer_blk3(t_x)  # (128, 30, 50)

        residual_t_x = self.fusion_blk2_up(t_x)  # (128, 100, 1500)
        c_x, _ = self.cnn_blk3_b2(c_x, residual_t_x)  # (128, 100, 1500)

        c_x = F.relu(self.norm(self.fc(c_x)))

        t_x = torch.flatten(t_x, start_dim=1)
        c_x = torch.flatten(c_x, start_dim=1)

        output1 = self.task1_output(c_x)
        output2 = self.task2_output(t_x)
        output3 = self.task3_output(c_x)

        return output1, output2, output3


def train():
    model = MTC()
    task_weights = (6, 2, 1)
    train_op(model, task_weights=task_weights)


if __name__ == '__main__':
    train()
