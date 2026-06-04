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

  class DonutEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super().__init__()

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.block = nn.Sequential(
            ParamConvRegVariableDilation(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            ),
            nn.ReLU(inplace=True),

            ParamConvRegVariableDilation(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, shuffle_true=0):
        for layer in self.block:
            if isinstance(layer, ParamConvRegVariableDilation):
                x = layer(x, shuffle_true=shuffle_true)
            else:
                x = layer(x)
        return x
