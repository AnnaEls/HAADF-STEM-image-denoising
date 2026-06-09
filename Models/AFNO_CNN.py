import torch
import torch.nn as nn
import torch.nn.functional as F

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

#=========================================
#       APAFNO bottleneck
#=========================================
class AFNOAmpPhaseBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels_amp=None,
        hidden_channels_phase=None,
        phase_scale=0.1,
        norm='layer'
    ):
        super().__init__()

        hidden_channels_amp = hidden_channels_amp or channels
        hidden_channels_phase = hidden_channels_phase or channels
        self.phase_scale = phase_scale

        # --- Amplitude operator ---
        self.amp_mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels_amp),
            nn.GELU(),
            nn.Linear(hidden_channels_amp, channels)
        )

        # --- Phase operator (learns Δφ) ---
        self.phase_mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels_phase),
            nn.GELU(),
            nn.Linear(hidden_channels_phase, channels)
        )


        # --- Normalization ---

        if norm == 'layer':
            self.norm = nn.LayerNorm(channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # ---- FFT ----
        x_fft = torch.fft.fft2(x, norm='ortho')  # (B,C,H,W), complex

        amp = torch.abs(x_fft)                   # (B,C,H,W)
        phase = torch.angle(x_fft)               # (B,C,H,W)

        # ---- reshape for channel mixing ----
        amp = amp.permute(0, 2, 3, 1)             # (B,H,W,C)
        phase = phase.permute(0, 2, 3, 1)

        # ---- Amplitude learning ----
        amp_out = self.amp_mlp(amp)

        # ---- Phase learning (predict Δφ) ----
        delta_phase = self.phase_mlp(phase)
        delta_phase = self.phase_scale * delta_phase

        phase_out = phase + delta_phase

        # ---- wrap phase to [-π, π] ----
        phase_out = torch.atan2(
            torch.sin(phase_out),
            torch.cos(phase_out)
        )

        # ---- reconstruct complex spectrum ----
        amp_out = amp_out.permute(0, 3, 1, 2)
        phase_out = phase_out.permute(0, 3, 1, 2)

        x_fft_out = amp_out*torch.exp(1j * phase_out)


        # ---- inverse FFT ----
        x_out = torch.fft.ifft2(x_fft_out, norm='ortho').real
       
        return x_out 


#===============================
#Model
#===============================
class APAFNO_CNN_Att(nn.Module):
    def __init__(self,in_channels=1, base_ch=32, depth = 1, hidden_channels_amp = None, hidden_channels_phase = None):
        super().__init__()
        # Encoder
        self.depth = depth
        self.downs = nn.ModuleList()
        self.ups_afno = nn.ModuleList()
        self.ups_cnn = nn.ModuleList()
        
        for i in range(depth):
            if i == 0:
                self.downs.append(EncoderBlock(in_channels, base_ch))
            else:
                self.downs.append(EncoderBlock(base_ch*2**(i-1), base_ch*2**i))

        # Bottleneck
        self.bottleneck_afno = AFNOAmpPhaseBlock(channels=base_ch*2**(depth-1), hidden_channels_amp=hidden_channels_amp, hidden_channels_phase=hidden_channels_phase)
        self.bottleneck_unet = ConvBlock(base_ch*2**(depth-1), base_ch*2**depth)
        self.proj_unet = nn.Conv2d(base_ch*2**depth, base_ch*2**(depth-1), 1)

        for i in range(depth):
            if i == 0:
                self.ups_afno.append(DecoderBlock(base_ch*2**(depth-1), base_ch*2**(depth-1)))
                self.ups_cnn.append(DecoderBlock(base_ch*2**(depth-1), base_ch*2**(depth-1)))
            else:
                self.ups_afno.append(DecoderBlock(base_ch*2**(depth-i), base_ch*2**(depth-i-1)))
                self.ups_cnn.append(DecoderBlock(base_ch*2**(depth-i), base_ch*2**(depth-i-1)))
        
        self.out_afno = nn.Conv2d(base_ch, in_channels, 1)
        self.out_cnn = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, x):
        skips = []
        for i, down in enumerate(self.downs):
            x, skip = down(x)
            skips.append(skip)
            

        x_afno = self.bottleneck_afno(x)

        x_unet = self.bottleneck_unet(x)
        x_unet = self.proj_unet(x_unet)

        x_att = self.CAFFM(x_afno, x_unet)
        x = x_att

        for i, up in enumerate(self.ups):
            x = up(x, skips[self.depth-i-1])

        x = self.out(x)       
        
        return x                      
