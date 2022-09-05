import torch
import torchvision.transforms as tv
import numpy as np
import matplotlib.pyplot as plt

from games_master.mario_bros.config import CFG


class ImageDQN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 20, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(20, 20, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(20, 20, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(20, 20, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(40, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 11),
        )
        self.img = tv.transforms.Compose([tv.transforms.ToTensor()])

    def forward(self, X):
        X = torch.stack([self.img(x) for x in X])
        y = self.net(X)
        return y
