import torch
import torchvision.transforms as tv
import numpy as np
import matplotlib.pyplot as plt

from config import CFG


class ImageDQN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1 if CFG.GRAYSCALE else 3, 20, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(20, 20, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(20*483, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 5 if not CFG.CONTINUOUS else 2000),
        )

        if CFG.GRAYSCALE:
            self.img = tv.transforms.Compose([tv.transforms.ToTensor(),
                                          tv.transforms.Grayscale(),
                                          tv.Lambda(lambda x: tv.functional.crop(x, 0, 0, 88, 96))
                                          ])
        else:
            self.img = tv.transforms.Compose([tv.transforms.ToTensor(),
                                          tv.Lambda(lambda x: tv.functional.crop(x, 0, 0, 88, 96))
                                          ])

    def forward(self, X):
        X = torch.stack([self.img(x) for x in X])
        y = self.net(X)
        return y
