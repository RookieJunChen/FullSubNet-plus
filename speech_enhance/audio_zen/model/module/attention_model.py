import torch
import torch.nn as nn
import math


class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        # batch_size, num_channels, T = input_tensor.size()
        # Average pooling along each channel
        squeeze_tensor = input_tensor.mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor


class ChannelCBAMLayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelCBAMLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        # batch_size, num_channels, T = input_tensor.size()
        # Average pooling along each channel
        mean_squeeze_tensor = input_tensor.mean(dim=2)
        max_squeeze_tensor, _ = torch.max(input_tensor, dim=2) #input_tensor.max(dim=2)
        # channel excitation
        mean_fc_out_1 = self.relu(self.fc1(mean_squeeze_tensor))
        max_fc_out_1 = self.relu(self.fc1(max_squeeze_tensor))
        fc_out_1 = mean_fc_out_1 + max_fc_out_1
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = mean_squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor


class ChannelECAlayer(nn.Module):
    """
     a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ChannelECAlayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SelfAttentionlayer(nn.Module):
    """
    Easy self attention.
    """

    def __init__(self, amp_dim=257, att_dim=257):
        super(SelfAttentionlayer, self).__init__()
        self.d_k = amp_dim
        self.q_linear = nn.Linear(amp_dim, att_dim)
        self.k_linear = nn.Linear(amp_dim, att_dim)
        self.v_linear = nn.Linear(amp_dim, att_dim)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(att_dim, amp_dim)

    def forward(self, q, k, v):
        q = self.q_linear(q)    # [B, T, F]
        k = self.k_linear(k)
        v = self.v_linear(v)
        output = self.attention(q, k, v)
        output = self.out(output)
        return output

    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = self.sigmoid(scores)
        output = torch.matmul(scores, v)
        return output
