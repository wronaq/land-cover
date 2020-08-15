import torch
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    @staticmethod
    def contraction(in_channels, num_features):
        seq = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_features,
                kernel_size=3,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=num_features),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=3,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=num_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        return seq

    @staticmethod
    def expansion(in_channels, num_features):
        seq = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=num_features, kernel_size=3
            ),
            nn.BatchNorm2d(num_features=num_features),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_features, out_channels=num_features, kernel_size=3
            ),
            nn.BatchNorm2d(num_features=num_features),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=num_features, out_channels=num_features, kernel_size=2
            ),
        )

        return seq
