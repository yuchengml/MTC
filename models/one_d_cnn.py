import torch.nn as nn

from preprocess import PREFIX_TO_TRAFFIC_ID, PREFIX_TO_APP_ID, AUX_ID
from train import train_op


class OneDCNN(nn.Module):
    """Construct 1D-CNN model"""

    def __init__(self):
        super(OneDCNN, self).__init__()
        # Create first convolution layer
        self.conv_1 = nn.Sequential(
            nn.Conv1d(1, 30, 4, 3),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            # nn.MaxPool2d((1, 3), 3, padding=(0, 1)),
        )

        # Create second convolution layer
        self.conv_2 = nn.Sequential(
            nn.Conv1d(30, 30, 5, 1),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout(p=0.07),
        )

        # Create feed forward layer
        self.fcs = nn.Sequential(
            # nn.Flatten(start_dim=1),
            nn.Linear(247, 200),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.Flatten(),
        )

        self.task1_output = nn.Linear(30 * 50, len(PREFIX_TO_TRAFFIC_ID))
        self.task2_output = nn.Linear(30 * 50, len(PREFIX_TO_APP_ID))
        self.task3_output = nn.Linear(30 * 50, len(AUX_ID))

    def forward(self, x):
        # x = torch.unsqueeze(x, 1)
        # print("x.shape:",x.shape)
        x = self.conv_1(x)
        # print("x.shape:",x.shape)
        x = self.conv_2(x)
        # print("x.shape:",x.shape)
        x = self.fcs(x)
        # print("x.shape:",x.shape)

        output1 = self.task1_output(x)
        output2 = self.task2_output(x)
        output3 = self.task3_output(x)
        return output1, output2, output3


def train():
    model = OneDCNN()
    task_weights = (2, 2, 1)
    train_op(model, task_weights=task_weights)


if __name__ == '__main__':
    train()
