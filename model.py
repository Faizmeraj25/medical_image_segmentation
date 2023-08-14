import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # Contracting path
        self.contracting_path = nn.Sequential(
            DoubleConv(in_channels, 64),
            Downsample(64, 128),
            Downsample(128, 256),
            Downsample(256, 512),
            Downsample(512, 1024),
        )
        
        # Expanding path
        self.expanding_path = nn.Sequential(
            Upsample(1024, 512),
            Upsample(512, 256),
            Upsample(256, 128),
            Upsample(128, 64),
        )
        
        # Final convolution to produce output
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        contracting_outputs = []
        for module in self.contracting_path:
            x = module(x)
            contracting_outputs.append(x)
        
        x = self.expanding_path[0](x)
        for i, module in enumerate(self.expanding_path[1:], 1):
            x = torch.cat([x, contracting_outputs[-i]], dim=1)
            x = module(x)
        
        output = self.output_conv(x)
        return output


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.upsample(x)
