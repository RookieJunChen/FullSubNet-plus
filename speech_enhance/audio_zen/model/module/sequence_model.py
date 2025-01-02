import torch
import torch.nn as nn
from audio_zen.model.module.causal_conv import TCNBlock

class SequenceModel(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_size,
            num_layers,
            bidirectional,
            sequence_model="GRU",
            output_activate_function="Tanh"
    ):
        """
        序列模型，可选 LSTM 或 CRN，支持子带输入

        Args:
            input_size: 每帧输入特征大小
            output_size: 每帧输出特征大小
            hidden_size: 序列模型隐层单元数量
            num_layers:  层数
            bidirectional: 是否为双向
            sequence_model: LSTM | GRU
            output_activate_function: Tanh | ReLU
        """
        super().__init__()
        # Sequence layer
        self.sequence_model_type = sequence_model
        if self.sequence_model_type == "LSTM":
            self.sequence_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif self.sequence_model_type == "GRU":
            self.sequence_model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif self.sequence_model_type == "TCN":
            self.sequence_model = nn.Sequential(
                TCNBlock(in_channels=input_size, out_channels=input_size, dilation=1),
                TCNBlock(in_channels=input_size, out_channels=input_size, dilation=2),
                TCNBlock(in_channels=input_size, out_channels=input_size, dilation=5),
                TCNBlock(in_channels=input_size, out_channels=input_size, dilation=9),
                TCNBlock(in_channels=input_size, out_channels=input_size, dilation=1),
                TCNBlock(in_channels=input_size, out_channels=input_size, dilation=2),
                TCNBlock(in_channels=input_size, out_channels=input_size, dilation=5),
                TCNBlock(in_channels=input_size, out_channels=input_size, dilation=9),
                nn.ReLU()
            )
        elif self.sequence_model_type == "TCN-subband":
            self.sequence_model = nn.Sequential(
                TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=1),
                TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=2),
                TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=5),
                TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=9),
                TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=1),
                TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=2),
                TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=5),
                TCNBlock(in_channels=input_size, hidden_channel=384, out_channels=input_size, dilation=9),
                nn.ReLU()
            )
        else:
            raise NotImplementedError(f"Not implemented {sequence_model}")

        if self.sequence_model_type == "LSTM" or self.sequence_model_type == "GRU":
            # Fully connected layer
            if bidirectional:
                self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
            else:
                self.fc_output_layer = nn.Linear(hidden_size, output_size)
        elif self.sequence_model_type == "TCN":
            self.fc_output_layer = nn.Linear(input_size, output_size)
        else:
            self.fc_output_layer = nn.Linear(input_size, output_size)

        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            else:
                raise NotImplementedError(f"Not implemented activation function {self.activate_function}")

        self.output_activate_function = output_activate_function

    def forward(self, x):
        """
        Args:
            x: [B, F, T]
        Returns:
            [B, F, T]
        """
        assert x.dim() == 3
        if self.sequence_model_type == "TCN" or self.sequence_model_type == "TCN-subband":
            x = self.sequence_model(x)  # [B, F, T]
            o = self.fc_output_layer(x.permute(0, 2, 1))    # [B, F, T] => [B, T, F]
            if self.output_activate_function:
                o = self.activate_function(o)
            o = o.permute(0, 2, 1)  # [B, T, F] => [B, F, T]
            return o
        else:
            self.sequence_model.flatten_parameters()
            # contiguous 使元素在内存中连续，有利于模型优化，但分配了新的空间
            # 建议在网络开始大量计算前使用一下
            x = x.permute(0, 2, 1).contiguous()  # [B, F, T] => [B, T, F]
            o, _ = self.sequence_model(x)
            o = self.fc_output_layer(o)
            if self.output_activate_function:
                o = self.activate_function(o)
            o = o.permute(0, 2, 1).contiguous()  # [B, T, F] => [B, F, T]
        return o


class Complex_SequenceModel(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_size,
            num_layers,
            bidirectional,
            sequence_model="GRU",
            output_activate_function="Tanh"
    ):
        """
        序列模型，可选 LSTM 或 CRN，支持子带输入

        Args:
            input_size: 每帧输入特征大小
            output_size: 每帧输出特征大小
            hidden_size: 序列模型隐层单元数量
            num_layers:  层数
            bidirectional: 是否为双向
            sequence_model: LSTM | GRU
            output_activate_function: Tanh | ReLU
        """
        super().__init__()
        # Sequence layer
        if sequence_model == "LSTM":
            self.real_sequence_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self.imag_sequence_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif sequence_model == "GRU":
            self.real_sequence_model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self.imag_sequence_model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            raise NotImplementedError(f"Not implemented {sequence_model}")

        # Fully connected layer
        if bidirectional:
            self.real_fc_output_layer = nn.Linear(hidden_size * 2, output_size)
            self.imag_fc_output_layer = nn.Linear(hidden_size * 2, output_size)
        else:
            self.real_fc_output_layer = nn.Linear(hidden_size, output_size)
            self.imag_fc_output_layer = nn.Linear(hidden_size, output_size)

        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            elif output_activate_function == "PReLU":
                self.activate_function = nn.PReLU()
            else:
                raise NotImplementedError(f"Not implemented activation function {self.activate_function}")

        self.output_activate_function = output_activate_function

    def forward(self, x):
        """
        Args:
            x: [B, F, T]
        Returns:
            [B, F, T]
        """
        assert x.dim() == 3
        self.real_sequence_model.flatten_parameters()
        self.imag_sequence_model.flatten_parameters()

        real, imag = torch.chunk(x, 2, 1)
        # contiguous 使元素在内存中连续，有利于模型优化，但分配了新的空间
        # 建议在网络开始大量计算前使用一下
        real = real.permute(0, 2, 1).contiguous()  # [B, F, T] => [B, T, F]
        imag = imag.permute(0, 2, 1).contiguous()

        r2r = self.real_sequence_model(real)[0]
        r2i = self.imag_sequence_model(real)[0]
        i2r = self.real_sequence_model(imag)[0]
        i2i = self.imag_sequence_model(imag)[0]

        real_out = r2r - i2i
        imag_out = i2r + r2i

        real_out = self.real_fc_output_layer(real_out)
        imag_out = self.imag_fc_output_layer(imag_out)

        # o = self.fc_output_layer(o)
        if self.output_activate_function:
            real_out = self.activate_function(real_out)
            imag_out = self.activate_function(imag_out)
        real_out = real_out.permute(0, 2, 1).contiguous()  # [B, T, F] => [B, F, T]
        imag_out = imag_out.permute(0, 2, 1).contiguous()

        o = torch.cat([real_out, imag_out], 1)
        return o


def _print_networks(nets: list):
    print(f"This project contains {len(nets)} networks, the number of the parameters: ")
    params_of_all_networks = 0
    for i, net in enumerate(nets, start=1):
        params_of_network = 0
        for param in net.parameters():
            params_of_network += param.numel()

        print(f"\tNetwork {i}: {params_of_network / 1e6} million.")
        params_of_all_networks += params_of_network

    print(f"The amount of parameters in the project is {params_of_all_networks / 1e6} million.")


if __name__ == '__main__':
    import datetime

    with torch.no_grad():
        ipt = torch.rand(1, 514, 1000)
        model = Complex_SequenceModel(
            input_size=257,
            output_size=257,
            hidden_size=256,
            bidirectional=False,
            num_layers=2,
            sequence_model="LSTM"
        )

        start = datetime.datetime.now()
        opt = model(ipt)
        end = datetime.datetime.now()
        print(f"{end - start}")
        _print_networks([model, ])
