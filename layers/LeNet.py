import torch
from torch import nn

class LeNet(nn.Module):
    def __init__(self, input_len, num_features,d_ff, d_model):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, padding=1), nn.Sigmoid(),
            nn.Conv2d(6, 16, kernel_size=3, padding=1), nn.Sigmoid(),
        )

        self.output = nn.Sequential(
            nn.Linear(num_features*16, d_ff), nn.Sigmoid(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        b, l, c = x.shape
        x = x.reshape(b, 1, l, c)
        feature = self.net(x)

        b,c,l,_ = feature.shape
        output = self.output(feature.permute(0,2,1,3).reshape(b,l,-1))

        return output

