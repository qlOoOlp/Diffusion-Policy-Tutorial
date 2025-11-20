import torch
from torch import nn
from typing import override
from einops import rearrange


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    @override
    def forward(self, x):
        return self.convs(x)



        class UNet(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()

        self.down1 = ConvBlock(in_ch, 64)
        self.down2 = ConvBlock(64, 128)
        self.bot1 = ConvBlock(128, 256)
        self.up2 = ConvBlock(128 + 256, 128)
        self.up1 = ConvBlock(128 + 64, 64)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    
    @override
    def forward(self, x):
        x1 = self.down1(x)              # (b, 64, H, W)
        x = self.maxpool(x1)            # (b, 64, H/2, W/2)
        x2 = self.down2(x)              # (b, 128, H/2, W/2)
        x = self.maxpool(x2)            # (b, 128, H/4, W/4)
        x = self.bot1(x)                # (b, 256, H/4, W/4)
        x = self.upsample(x)            # (b, 256, H/2, W/2)
        x = torch.cat([x, x2], dim=1)   # (b, 128 + 256, H/2, W/2)
        x = self.up2(x)                 # (b, 128, H/2, W/2)
        x = self.upsample(x)            # (b, 128, H, W)
        x = torch.cat([x, x1], dim=1)   # (b, 128 + 64, H, W)
        x = self.up1(x)                 # (b, 64, H, W)
        x = self.out(x)                 # (b, 1, H, W)
        return x