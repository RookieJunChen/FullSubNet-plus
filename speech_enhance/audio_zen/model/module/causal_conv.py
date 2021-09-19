import torch
import torch.nn as nn


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, encoder_activate_function, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 1),
            **kwargs  # 这里不是左右 pad，而是上下 pad 为 0，左右分别 pad 1...
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = getattr(nn, encoder_activate_function)()

    def forward(self, x):
        """
        2D Causal convolution.

        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, output_padding=(0, 0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            output_padding=output_padding
        )
        self.norm = nn.BatchNorm2d(out_channels)
        if is_last:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.

        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class TCNBlock(nn.Module):
    def __init__(self, in_channels=257, hidden_channel=512, out_channels=257, kernel_size=3, dilation=1,
                 use_skip_connection=True, causal=False):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, hidden_channel, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.GroupNorm(1, hidden_channel, eps=1e-8)
        padding = (dilation * (kernel_size - 1)) // 2 if not causal else (
                dilation * (kernel_size - 1))
        self.depthwise_conv = nn.Conv1d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=1,
                                        groups=hidden_channel, padding=padding, dilation=dilation)
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(1, hidden_channel, eps=1e-8)
        self.sconv = nn.Conv1d(hidden_channel, out_channels, 1)
        # self.tcn_block = nn.Sequential(
        #     nn.Conv1d(in_channels, hidden_channel, 1),
        #     nn.PReLU(),
        #     nn.GroupNorm(1, hidden_channel, eps=1e-8),
        #     nn.Conv1d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=1,
        #               groups=hidden_channel, padding=padding, dilation=dilation, bias=True),
        #     nn.PReLU(),
        #     nn.GroupNorm(1, hidden_channel, eps=1e-8),
        #     nn.Conv1d(hidden_channel, out_channels, 1)
        # )

        self.causal = causal
        self.padding = padding
        self.use_skip_connection = use_skip_connection

    def forward(self, x):
        """
            x: [channels, T]
        """
        if self.use_skip_connection:
            y = self.conv1x1(x)
            y = self.norm1(self.prelu1(y))
            y = self.depthwise_conv(y)
            if self.causal:
                y = y[:, :, :-self.padding]
            y = self.norm2(self.prelu2(y))
            output = self.sconv(y)
            return x + output
        else:
            y = self.conv1x1(x)
            y = self.norm1(self.prelu1(y))
            y = self.depthwise_conv(y)
            if self.causal:
                y = y[:, :, :-self.padding]
            y = self.norm2(self.prelu2(y))
            output = self.sconv(y)
            return output


class STCNBlock(nn.Module):
    def __init__(self, in_channels=257, hidden_channel=512, out_channels=257, kernel_size=3, dilation=1,
                 use_skip_connection=True, causal=False):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, hidden_channel, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.GroupNorm(1, hidden_channel, eps=1e-8)
        padding = (dilation * (kernel_size - 1)) // 2 if not causal else (
                dilation * (kernel_size - 1))
        self.depthwise_conv = nn.Conv1d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=1,
                                        groups=hidden_channel, padding=padding, dilation=dilation)
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(1, hidden_channel, eps=1e-8)
        self.sconv = nn.Conv1d(hidden_channel, out_channels, 1)
        # self.tcn_block = nn.Sequential(
        #     nn.Conv1d(in_channels, hidden_channel, 1),
        #     nn.PReLU(),
        #     nn.GroupNorm(1, hidden_channel, eps=1e-8),
        #     nn.Conv1d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=1,
        #               groups=hidden_channel, padding=padding, dilation=dilation, bias=True),
        #     nn.PReLU(),
        #     nn.GroupNorm(1, hidden_channel, eps=1e-8),
        #     nn.Conv1d(hidden_channel, out_channels, 1)
        # )

        self.causal = causal
        self.padding = padding
        self.use_skip_connection = use_skip_connection

    def forward(self, x):
        """
            x: [channels, T]
        """
        if self.use_skip_connection:
            y = self.conv1x1(x)
            y = self.norm1(self.prelu1(y))
            y = self.depthwise_conv(y)
            if self.causal:
                y = y[:, :, :-self.padding]
            y = self.norm2(self.prelu2(y))
            output = self.sconv(y)
            return x + output
        else:
            y = self.conv1x1(x)
            y = self.norm1(self.prelu1(y))
            y = self.depthwise_conv(y)
            if self.causal:
                y = y[:, :, :-self.padding]
            y = self.norm2(self.prelu2(y))
            output = self.sconv(y)
            return output


if __name__ == '__main__':
    a = torch.rand(2, 1, 19, 200)
    l1 = CausalConvBlock(1, 20, kernel_size=(3, 2), stride=(2, 1), padding=(0, 1), )
    l2 = CausalConvBlock(20, 40, kernel_size=(3, 2), stride=(1, 1), padding=1, )
    l3 = CausalConvBlock(40, 40, kernel_size=(3, 2), stride=(2, 1), padding=(0, 1), )
    l4 = CausalConvBlock(40, 40, kernel_size=(3, 2), stride=(1, 1), padding=1, )
    print(l1(a).shape)
    print(l4(l3(l2(l1(a)))).shape)
