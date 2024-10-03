import torch
from torch import nn


class BasicNet(nn.Module):
    def __init__(self, input_size: int):
        super(BasicNet, self).__init__()

        layer = 512
        self.layers = nn.Sequential(
            nn.Linear(input_size, layer),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(layer),
            nn.Linear(layer, layer),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(layer),
            nn.Linear(layer, layer),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(layer),
            nn.Linear(layer, 3),
            nn.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layers(x)
        return x