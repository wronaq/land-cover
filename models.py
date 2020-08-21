import torch
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self, in_channels, features, num_classes):
        super(Unet, self).__init__()

        self.encoder = nn.Sequential(
            self.contraction(in_channels, features),
            self.contraction(features, features * 2),
            self.contraction(features * 2, features * 4),
            self.contraction(features * 4, features * 8),
        )
        self.decoder = nn.Sequential(
            self.expansion(features * 8, features * 16, features * 8),
            self.expansion(features * 16, features * 8, features * 4),
            self.expansion(features * 8, features * 4, features * 2),
            self.expansion(features * 4, features * 2, features),
        )
        self.final = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=3, padding=0,),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, padding=0,),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(),
            nn.Conv2d(features, num_classes, kernel_size=1, padding=0,),
            nn.BatchNorm2d(num_features=num_classes),
            nn.LogSoftmax(dim=2),
        )

    def forward(self, x):
        enc0 = self.encoder[0](x)
        enc1 = self.encoder[1](enc0)
        enc2 = self.encoder[2](enc1)
        enc3 = self.encoder[3](enc2)

        dec0 = self.decoder[0](enc3)
        dec1 = self.decoder[1](torch.cat((enc3, dec0), dim=1))
        dec2 = self.decoder[2](torch.cat((enc2, dec1), dim=1))
        dec3 = self.decoder[3](torch.cat((enc1, dec2), dim=1))

        final = self.final(torch.cat((enc0, dec3), dim=1))

        return final

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
    def expansion(in_channels, num_features, out_channels):
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
            nn.ConvTranspose2d(
                in_channels=num_features,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
            ),
        )

        return seq
