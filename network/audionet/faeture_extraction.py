import torch
import torch.nn as nn


class SpecEncoderGlobal(nn.Module):
    def __init__(
        self,
        in_channels=2,
        channels=(16, 32, 64),
        dropout=0.1,
        out_dim=256,
        use_compress=True,
        pool_size=(4, 4),
    ):
        super().__init__()
        self.use_compress = use_compress

        blocks = []
        prev_channels = in_channels
        for out_channels in channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Dropout2d(dropout),
                )
            )
            prev_channels = out_channels

        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_channels * pool_size[0] * pool_size[1], out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return self.fc(x)


class DepthResNet18Encoder(nn.Module):
    def __init__(self, out_dim=64, pretrained=True):
        super().__init__()

        self.uses_torchvision = False
        try:
            from torchvision import models

            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                backbone = models.resnet18(weights=weights)
            except Exception:
                backbone = models.resnet18(pretrained=pretrained)

            old_conv1 = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv1.out_channels,
                kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride,
                padding=old_conv1.padding,
                bias=old_conv1.bias is not None,
            )
            if pretrained:
                with torch.no_grad():
                    backbone.conv1.weight[:] = old_conv1.weight.mean(dim=1, keepdim=True)

            self.features = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, out_dim),
                nn.ReLU(inplace=True),
            )
            self.uses_torchvision = True
        except Exception:
            self.features = nn.Sequential(
                conv_block(1, 16),
                conv_block(16, 32),
                conv_block(32, 64),
                conv_block(64, 128),
            )
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, out_dim),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.fc(x)


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )
