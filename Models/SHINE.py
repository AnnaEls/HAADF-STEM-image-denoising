import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ParamConvRegVariableDilation(nn.Module):
    """
    Learnable 3x3 donut convolution with variable effective dilation.

    The center weight is zero unless dilation becomes invalid.
    In that fallback case, the operation becomes a 1x1-like center convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=(1, 1),
        groups=1,
        bias=False
    ):
        super().__init__()

        self.kernel_size_i = kernel_size[0]
        self.kernel_size_j = kernel_size[1]

        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.dilation = [
            self.kernel_size_i // 2,
            self.kernel_size_j // 2
        ]

        # Donut mask: all 3x3 positions except center
        mask = torch.zeros(3, 3, dtype=torch.float32)
        mask[0, 0] = 1
        mask[0, 1] = 1
        mask[0, 2] = 1
        mask[1, 0] = 1
        mask[1, 2] = 1
        mask[2, 0] = 1
        mask[2, 1] = 1
        mask[2, 2] = 1

        indices = torch.nonzero(mask).t()

        self.register_buffer("mask", mask)
        self.register_buffer("indices", indices)

        self.register_buffer(
            "zero_kernel",
            torch.zeros(
                out_channels,
                in_channels // groups,
                3,
                3,
                dtype=torch.float32
            )
        )

        self.params = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                8,
                dtype=torch.float32
            )
        )

        nn.init.kaiming_uniform_(self.params, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))
        else:
            self.bias = None

    def forward(self, x, shuffle_true=0):
        x = x.float()

        kernel = self.zero_kernel.clone()
        kernel[:, :, self.indices[0], self.indices[1]] = self.params

        dilation_real = [
            self.dilation[0] - shuffle_true,
            self.dilation[1] - shuffle_true
        ]

        padding = [
            self.padding[0] - shuffle_true,
            self.padding[1] - shuffle_true
        ]

        # Fallback if dilation becomes invalid
        if dilation_real[0] <= 0 or dilation_real[1] <= 0:
            kernel = self.zero_kernel.clone()
            kernel[:, :, 1, 1] = torch.sum(self.params, dim=2)

            dilation_real = [1, 1]
            padding = [1, 1]

        x = F.pad(
            x,
            [padding[1], padding[1], padding[0], padding[0]],
            mode="constant",
            value=0.0
        )

        out = F.conv2d(
            x,
            kernel,
            bias=self.bias,
            stride=self.stride,
            padding=0,
            dilation=dilation_real,
            groups=self.groups
        )

        return out

class SHINE_FP32(nn.Module):
    """
    Full-precision SHINE model.

    Recommended for single-image denoising:
        frame_num = 1
        in_channels = 1
        out_channels = 1
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        add_dilation=(0, 0),
        frame_num=1,
        filters=32,
        blocks=14,
        bias=False
    ):
        super().__init__()

        self.n_frames = frame_num
        self.n_filters = filters
        self.n_block = blocks
        self.in_channels = in_channels

        self.add_dilation_i = add_dilation[0]
        self.add_dilation_j = add_dilation[1]

        self.activation = nn.GELU()

        ratio_1 = 1.0 if frame_num == 1 else 3.0 / 4.0

        if frame_num > 1:
            side_channels = in_channels * (self.n_frames - 1)
            side_filters = int(self.n_filters * (1.0 - ratio_1))

            layers = [
                nn.Sequential(
                    nn.Conv2d(
                        side_channels,
                        side_filters,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias
                    ),
                    self.activation
                )
            ]

            for _ in range(max(self.add_dilation_i, self.add_dilation_j)):
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            side_filters,
                            side_filters,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=bias
                        ),
                        self.activation
                    )
                )

            self.convolution_layer0_a = nn.Sequential(*layers)

        self.convolution_layer0_b = nn.Conv2d(
            in_channels,
            int(self.n_filters * ratio_1),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

        self.first_output = ParamConvRegVariableDilation(
            in_channels=self.n_filters,
            out_channels=self.n_filters,
            kernel_size=(
                (self.add_dilation_i + 1) * 2 + 1,
                (self.add_dilation_j + 1) * 2 + 1
            ),
            stride=1,
            padding=(1 + self.add_dilation_i, 1 + self.add_dilation_j),
            bias=bias
        )

        self.convolution_layer10 = nn.ModuleList()
        self.convolution_layer2 = nn.ModuleList()

        for _ in range(4):
            self.convolution_layer10.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.n_filters,
                        self.n_filters,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias
                    ),
                    self.activation,
                    nn.Conv2d(
                        self.n_filters,
                        self.n_filters,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias
                    )
                )
            )

            self.convolution_layer2.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.n_filters,
                        self.n_filters,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias
                    ),
                    self.activation,
                    nn.Conv2d(
                        self.n_filters,
                        self.n_filters,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias
                    )
                )
            )

        self.dilated_convs = nn.ModuleList()
        self.dilated_convs_res = nn.ModuleList()

        kernel_size_i = (self.add_dilation_i + 1 + 4) * 2 + 1
        kernel_size_j = (self.add_dilation_j + 1 + 4) * 2 + 1

        padding_i = kernel_size_i // 2
        padding_j = kernel_size_j // 2

        for _ in range(4):
            self.dilated_convs.append(
                ParamConvRegVariableDilation(
                    in_channels=self.n_filters,
                    out_channels=self.n_filters,
                    kernel_size=(kernel_size_i, kernel_size_j),
                    stride=1,
                    padding=(padding_i, padding_j),
                    bias=bias
                )
            )

            self.dilated_convs_res.append(
                ParamConvRegVariableDilation(
                    in_channels=self.n_filters,
                    out_channels=self.n_filters,
                    kernel_size=(kernel_size_i - 4, kernel_size_j - 4),
                    stride=1,
                    padding=(padding_i - 2, padding_j - 2),
                    bias=bias
                )
            )

            kernel_size_i = (kernel_size_i // 4 + 2 + 4) * 2 + 1
            kernel_size_j = (kernel_size_j // 4 + 2 + 4) * 2 + 1

            padding_i = kernel_size_i // 2
            padding_j = kernel_size_j // 2

        # output channels are:
        # first_output: 64
        # dilated_convs_res[0]: 64
        # four dilated_convs: 4 * 64
        # three more dilated_convs_res: 3 * 64
        # total = 9 * filters
        feature_size = self.n_filters * 9

        self.outconvs = nn.Sequential(
            nn.Conv2d(
                feature_size,
                feature_size // 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias
            ),
            self.activation,
            nn.Conv2d(
                feature_size // 2,
                self.n_filters,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias
            ),
            self.activation
        )

        self.last_out = nn.Conv2d(
            self.n_filters,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

        self.pool2d = nn.AvgPool2d(kernel_size=2)

        self.upsample_list = nn.ModuleList()

        for i in range(4):
            self.upsample_list.append(
                nn.Sequential(
                    nn.Upsample(
                        scale_factor=2 ** i,
                        mode="bilinear",
                        align_corners=False
                    ),
                    nn.Conv2d(
                        self.n_filters,
                        self.n_filters,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=bias
                    ),
                    self.activation
                )
            )

        self.float()

    def forward(self, input_image, shuffle=0):
        input_image = input_image.float()

        N, C, nH, nW = input_image.shape

        pad_h = 0
        pad_w = 0

        if nH % 64 != 0:
            pad_h = 64 - nH % 64

        if nW % 64 != 0:
            pad_w = 64 - nW % 64

        if pad_h > 0 or pad_w > 0:
            input_image = F.pad(
                input_image,
                [0, pad_w, 0, pad_h],
                mode="constant",
                value=0.0
            )

        if self.n_frames > 1:
            center = self.n_frames // 2

            target = input_image[:, center:center + 1, :, :]

            side_features = torch.cat(
                (
                    input_image[:, :center, :, :],
                    input_image[:, center + 1:, :, :]
                ),
                dim=1
            )

            target = self.activation(self.convolution_layer0_b(target))
            side_features = self.convolution_layer0_a(side_features)

            base = torch.cat((target, side_features), dim=1)

        else:
            base = self.activation(self.convolution_layer0_b(input_image))

        output = self.activation(self.first_output(base, shuffle_true=0))

        base = self.activation(self.convolution_layer10[0](base) + base)

        merged = self.activation(self.dilated_convs_res[0](base, shuffle_true=0))
        output = torch.cat((output, merged), dim=1)

        base = self.activation(self.convolution_layer2[0](base) + base)

        for i in range(4):
            merged = self.activation(self.dilated_convs[i](base, shuffle_true=0))
            merged = self.upsample_list[i](merged)

            output = torch.cat((output, merged), dim=1)

            if i < 3:
                base = self.pool2d(base)

                base = self.activation(self.convolution_layer10[i + 1](base) + base)

                merged = self.activation(
                    self.dilated_convs_res[i + 1](base, shuffle_true=0)
                )
                merged = self.upsample_list[i + 1](merged)

                output = torch.cat((output, merged), dim=1)

                base = self.activation(self.convolution_layer2[i + 1](base) + base)

        output = self.outconvs(output)
        output = self.last_out(output)

        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :nH, :nW]

        return output
