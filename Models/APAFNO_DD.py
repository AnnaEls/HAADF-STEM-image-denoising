import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ParamConvRegVariableDilation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=1,
        padding=(1, 1),
        groups=1,
        bias=False
    ):
        super().__init__()

        self.kernel_size_i = kernel_size[0]
        self.kernel_size_j = kernel_size[1]

        mask = torch.ones(3, 3)
        mask[1, 1] = 0

        self.register_buffer("mask", mask)
        self.register_buffer("indices", torch.nonzero(mask).t())

        num_params = 8

        self.params = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, num_params)
        )
        nn.init.kaiming_uniform_(self.params, a=math.sqrt(5))

        self.stride = stride
        self.padding = padding
        self.dilation = [
            self.kernel_size_i // 2,
            self.kernel_size_j // 2
        ]
        self.groups = groups

        self.register_buffer(
            "zero_kernel",
            torch.zeros(out_channels, in_channels // groups, 3, 3)
        )

    def forward(self, x, shuffle_true=0):
        kernel = self.zero_kernel.clone()

        dilation_real = [
            self.dilation[0] - shuffle_true,
            self.dilation[1] - shuffle_true
        ]

        padding = [
            self.padding[0] - shuffle_true,
            self.padding[1] - shuffle_true
        ]

        kernel[:, :, self.indices[0], self.indices[1]] = self.params

        if dilation_real[0] <= 0 or dilation_real[1] <= 0:
            dilation_real = [1, 1]
            padding = [1, 1]

            kernel = self.zero_kernel.clone()
            kernel[:, :, 1, 1] = torch.sum(self.params, dim=2)

        x = F.pad(x, [padding[1], padding[1], padding[0], padding[0]])

        return F.conv2d(
            x,
            kernel,
            bias=None,
            stride=self.stride,
            padding=0,
            dilation=dilation_real,
            groups=self.groups
        )

#===============================
#Encoder and decoder
#===============================
#Convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)

#Encoder block
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)
        return x, skip

class DonutEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super().__init__()

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.spatial_blind = ParamConvRegVariableDilation(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.channel_mixer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.MaxPool2d(2)

    def forward(self, x, shuffle_true=0):
        x = self.spatial_blind(x, shuffle_true=shuffle_true)
        x = self.channel_mixer(x)

        skip = x
        x = self.pool(skip)

        return x, skip

#Decoder block
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(2*out_ch, out_ch)
    def forward(self, x1, x2): # x1: upsampled, x2: skip
        x1 = self.up(x1)

        # Pad x1 if needed (for odd input sizes)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
#===============================
#Model
#===============================
class APAFNO_DD(nn.Module):
    def __init__(self,in_channels=1, base_ch=32, depth = 5, add_dilation = (3,3)):
        super().__init__()
        # Encoder
        self.depth = depth
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i in range(depth):
            if i == 0:
                self.downs.append(DonutEncoderBlock(in_channels, base_ch, kernel_size = add_dilation))
            elif i == 1:
                self.downs.append(DonutEncoderBlock(base_ch, base_ch*2, kernel_size = add_dilation))
            else:
                self.downs.append(EncoderBlock(base_ch*2**(i-1), base_ch*2**i))

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch*2**(depth-1), base_ch*2**depth)

        for i in range(depth):
            if i == 0:
                self.ups.append(DecoderBlock(base_ch*2**depth, base_ch*2**(depth-1))) 
            else:
                self.ups.append(DecoderBlock(base_ch*2**(depth-i), base_ch*2**(depth-i-1)))

        self.out = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, x):
        skips = []
        for i, down in enumerate(self.downs):
            x, skip = down(x)
            skips.append(skip)

        x = self.bottleneck(x)


        for i, up in enumerate(self.ups):
            x = up(x, skips[self.depth-i-1])

        return self.out(x)
