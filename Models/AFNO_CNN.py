import torch, torch.nn as nn, torch.nn.functional as F
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


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class AFNOTransformerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, hidden_dim_afno):
        super().__init__()
        self.afno = AFNOAmpPhaseBlock(dim, hidden_channels_amp=None, hidden_channels_phase=None)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)  # before AFNO
        self.norm2 = nn.LayerNorm(dim)  # before MLP

    def forward(self, x, skip):
        B, C, H, W = x.shape

        # --- Skip 1 around AFNO ---
        skip_1 = x

        #----Norm1
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = self.afno(x) + skip_1

        # --- Skip 2 around MLP ---
        x_perm = x.permute(0, 2, 3, 1)

        x_perm = self.norm2(x_perm)

        x = self.mlp(x_perm.contiguous().view(B, H * W, C))
        x = x.view(B, H, W, C)

        x = x.permute(0, 3, 1, 2).contiguous()

        return x

#===============================
#Encoder and decoder
#===============================
#Convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU()
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
    def forward(self, x1, x2, skip = True): # x1: upsampled, x2: skip
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
class DFNO(nn.Module):
    def __init__(self,in_channels=1,base_ch=32, depth=3, mlp_ratio=6, hidden_dim_afno=64):
        super().__init__()
        # Encoder
        self.encoder = EncoderBlock(in_channels, base_ch)

        #Bottleneck
        self.bottleneck_afno =nn.ModuleList([
            AFNOTransformerBlock(base_ch, mlp_ratio, hidden_dim_afno)
            for _ in range(depth)
        ])
        self.bottleneck_cnn = ConvBlock(base_ch, base_ch*2)
        self.proj = nn.Conv2d(base_ch*2, base_ch, 1)

        # Decoder
        self.decoder_cnn = DecoderBlock(base_ch, base_ch)
        self.decoder_afno = DecoderBlock(base_ch, base_ch)

        self.out_conv_afno = nn.Conv2d(base_ch, in_channels,  1)
        self.out_conv_cnn = nn.Conv2d(base_ch, in_channels,  1)


    def forward(self, x):
        # Encoder
        x_cnn, skip_1 = self.encoder(x) #common encoder
        x_afno = x_cnn

        #Bottleneck
        x_cnn = self.bottleneck_cnn(x_cnn)
        x_cnn = self.proj(x_cnn)

        for blk in self.bottleneck_afno:
           x_afno = blk(x_afno,x_cnn)

        y_afno = self.decoder_afno(x_afno, skip_1)
        y_cnn = self.decoder_cnn(x_cnn, skip_1)

        y_afno = self.out_conv_afno(y_afno)
        y_cnn = self.out_conv_cnn(y_cnn)

        return y_afno, y_cnn
