import torch
import torch.nn as nn
import torch.nn.functional as F

#===============================
#Weight initialization
#===============================
def init_kaiming(model: nn.Module, zero_out_head: bool = True):
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
            
class AFNOAmpPhaseBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels=None,
        phase_scale=1.0,
        norm='layer'
    ):
        super().__init__()

        hidden_channels = hidden_channels or channels
        self.phase_scale = phase_scale

        # --- Amplitude operator ---
        self.amp_mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels)
        )

        # --- Phase operator (learns Δφ) ---
        self.phase_mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels)
        )

        if norm == 'layer':
            self.norm = nn.LayerNorm(channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        residual = x

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

        x_fft_out = amp_out * torch.exp(1j * phase_out)

        # ---- inverse FFT ----
        x_out = torch.fft.ifft2(x_fft_out, norm='ortho').real

        # ---- residual + norm ----
        x_out = self.norm(x_out.permute(0, 2, 3, 1))
        x_out = x_out.permute(0, 3, 1, 2)

        return x_out + residual

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
        self.afno = AFNOAmpPhaseBlock(dim, hidden_dim_afno)

        self.mlp = MLP(dim, int(dim * mlp_ratio))
        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)  # before AFNO


    def forward(self, x):
        B, C, H, W = x.shape

        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = self.afno(x) 

        x_perm = x.permute(0, 2, 3, 1)
        x = self.mlp(x_perm.contiguous().view(B, H * W, C))
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x

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
        self.afno = AFNOAmpPhaseBlock(dim, hidden_dim_afno)

        self.mlp = MLP(dim, int(dim * mlp_ratio))
        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)  # before AFNO


    def forward(self, x):
        B, C, H, W = x.shape

        # --- Skip 1 around AFNO ---
        skip_1 = x

        #----Norm1
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = self.afno(x) #+ skip_1

        # --- Skip 2 around MLP ---
        x_perm = x.permute(0, 2, 3, 1)
        x = self.mlp(x_perm.contiguous().view(B, H * W, C))
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x #+ skip_1

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
class APAFNO(nn.Module):
    def __init__(self,in_channels=1,base_ch=32, depth=3, mlp_ratio=6, hidden_dim_afno=64):
        super().__init__()
        # Encoder
        self.encoder = EncoderBlock(in_channels, base_ch)
        self.encoder_afno = EncoderBlock(in_channels, base_ch)

        #Bottleneck
        self.bottleneck_afno =nn.ModuleList([
            AFNOTransformerBlock(base_ch, mlp_ratio, hidden_dim_afno)
            for _ in range(depth)
        ])

        self.bottleneck_cnn = ConvBlock(base_ch, base_ch*2)

        # Decoder
        self.decoder_afno = DecoderBlock(base_ch, base_ch)
        self.out_conv_afno = nn.Conv2d(base_ch, in_channels,  1)

        self.decoder_cnn = DecoderBlock(base_ch*2, base_ch)
        self.out_conv_cnn = nn.Conv2d(base_ch, in_channels,  1)

    def forward(self, x):
        # Encoder
        x_afno, skip_1 = self.encoder_afno(x) #AFNO encoder

        # Bottleneck

        for blk in self.bottleneck_afno:
           x_afno = blk(x_afno)

        y_afno = self.decoder_afno(x_afno, skip_1)
        y_afno = self.out_conv_afno(y_afno)


        return y_afno
