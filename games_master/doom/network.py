import torch
import torchvision.transforms as tv
import numpy as np
import matplotlib.pyplot as plt

from config import CFG


class ImageDQN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=32, kernel_size=8, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=64, out_channels=128,  kernel_size=1, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=128, out_channels=64,  kernel_size=1, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            # torch.nn.Conv2d(in_channels=64, out_channels=128,  kernel_size=1, stride=1),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.Conv2d(in_channels=128, out_channels=64,  kernel_size=1, stride=1),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.Conv2d(in_channels=64, out_channels=64,  kernel_size=1, stride=1),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=64, out_channels=128,  kernel_size=1, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=128, out_channels=64,  kernel_size=1, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=64, out_channels=32,  kernel_size=1, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(384, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, CFG.ACT_RANGE),
        )

    def forward(self, X):
        y = self.net(X)
        return y
