import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from preprocess import PREFIX_TO_TRAFFIC_ID, PREFIX_TO_APP_ID, AUX_ID
from train import train_op


class Chomp1d(nn.Module):
    """Define a 1D chomping layer to trim the input tensor's temporal dimension. This module is used to
     trim the size of the data after convolution, making it the same size as the input data."""

    def __init__(self, chomp_size: int):
        """
        Args:
            chomp_size: Padding size.
        """
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """This is the module of TCN, consisting of 8 parts. The 'downsample' mentioned in
    the two (convolution + chomp + ReLU + dropout) components refers to downsampling, which actually
    implements the residual connection part."""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs=1, num_channels=None, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        if num_channels is None:
            num_channels = [30] * 7

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

        # Task-specific layers
        self.task1_output = nn.Linear(30 * 1500, len(PREFIX_TO_TRAFFIC_ID))
        self.task2_output = nn.Linear(30 * 1500, len(PREFIX_TO_APP_ID))
        self.task3_output = nn.Linear(30 * 1500, len(AUX_ID))

    def forward(self, x):
        x = self.network(x)

        x = torch.flatten(x, start_dim=1)

        output1 = self.task1_output(x)
        output2 = self.task2_output(x)
        output3 = self.task3_output(x)

        return output1, output2, output3


def train():
    model = TCN()
    task_weights = (6, 2, 1)
    train_op(model, task_weights=task_weights)


if __name__ == '__main__':
    train()
