import torch
import torchvision.transforms as tv
import numpy as np
import matplotlib.pyplot as plt

from games_master.mario_bros.config import CFG


class ImageDQN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=64, out_channels=128,  kernel_size=1, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=128, out_channels=64,  kernel_size=1, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=64, out_channels=128,  kernel_size=1, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=128, out_channels=64,  kernel_size=1, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64,  kernel_size=1, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=64, out_channels=40,  kernel_size=1, stride=1),
            torch.nn.ReLU(inplace=True),
            # torch.nn.Conv2d(in_channels=128, out_channels=64,  kernel_size=1, stride=1),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.Conv2d(in_channels=64, out_channels=40,  kernel_size=1, stride=1),
            # torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(1120, 560),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(560, 14),
        )
        self.img = tv.transforms.Compose([tv.transforms.ToTensor()])

    def forward(self, X):
        X = torch.stack([self.img(x) for x in X])
        y = self.net(X)
        return y
