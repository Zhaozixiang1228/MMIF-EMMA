import torch
import torch.nn as nn
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        out = self.conv(x)
        return out


class UNet5(nn.Module):

    def __init__(self, input_channel=1):
        super(UNet5, self).__init__()

        self.down_sample=nn.MaxPool2d(2)
        self.down1 = ConvBlock(input_channel, 64)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        self.down4 = ConvBlock(256, 512)
        self.down5 = ConvBlock(512, 1024)

        self.up_sample5=nn.Sequential(
            nn.ConvTranspose2d(1024,512, 4, 2, 1, bias=False),
            nn.ReLU()
        )
        self.up4 = ConvBlock(1024,512)
        self.up_sample4 = nn.Sequential(
            nn.ConvTranspose2d(512,256, 4, 2, 1, bias=False),
            nn.ReLU()
        )
        self.up3 = ConvBlock(512,256)
        self.up_sample3 = nn.Sequential(
            nn.ConvTranspose2d(256,128, 4, 2, 1, bias=False),
            nn.ReLU()
        )
        self.up2 = ConvBlock(256,128)
        self.up_sample2 = nn.Sequential(
            nn.ConvTranspose2d(128,64, 4, 2, 1, bias=False),
            nn.ReLU()
        )
        self.up1 = ConvBlock(128,64)
        self.last = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.down_sample(d1))
        d3 = self.down3(self.down_sample(d2))
        d4 = self.down4(self.down_sample(d3))
        out = self.down5(self.down_sample(d4))

        out = self.up4(torch.cat((self.up_sample5(out),d4),1))
        out = self.up3(torch.cat((self.up_sample4(out),d3),1))
        out = self.up2(torch.cat((self.up_sample3(out),d2),1))
        out = self.up1(torch.cat((self.up_sample2(out),d1),1))
        out = self.last(out)

        return out
