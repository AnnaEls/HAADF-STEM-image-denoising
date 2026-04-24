import torch
import torch.nn as nn
import torch.nn.functional as F

#===============================
#Weight initialization
#===============================
def init_unet_kaiming(model: nn.Module, zero_out_head: bool = True):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.ConvTranspose2d):
            kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
            factor_h = (kh + 1) // 2
            factor_w = (kw + 1) // 2
            center_h = factor_h - 1 if kh % 2 == 1 else factor_h - 0.5
            center_w = factor_w - 1 if kw % 2 == 1 else factor_w - 0.5
            og_h = torch.arange(kh).reshape(-1, 1).float()
            og_w = torch.arange(kw).reshape(1, -1).float()
            filt = (1 - torch.abs(og_h - center_h) / factor_h) * (1 - torch.abs(og_w - center_w) / factor_w)

            with torch.no_grad():
                w = torch.zeros_like(m.weight)
                for c in range(min(m.out_channels, m.in_channels)):
                    w[c, c, :, :] = filt
                m.weight.copy_(w)
                if m.bias is not None:
                    m.bias.zero_()

        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)) and hasattr(m, 'weight'):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    if zero_out_head and hasattr(model, "out") and isinstance(model.out, nn.Conv2d):
        nn.init.zeros_(model.out.weight)
        if model.out.bias is not None:
            nn.init.zeros_(model.out.bias)
            
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
                    
