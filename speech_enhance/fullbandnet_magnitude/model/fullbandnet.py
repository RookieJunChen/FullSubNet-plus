import torch
from torch.nn import functional

from audio_zen.model.base_model import BaseModel
from audio_zen.model.module.sequence_model import SequenceModel
from audio_zen.model.module.attention_model import ChannelSELayer, ChannelECAlayer, ChannelCBAMLayer

# for log
from utils.logger import log
print=log

class Model(BaseModel):
    def __init__(
            self,
            num_freqs,
            hidden_size,
            sequence_model,
            output_activate_function,
            look_ahead,
            channel_attention_model="SE",
            norm_type="offline_laplace_norm",
            weight_init=True,
    ):
        """
        Fullband Model (cIRM mask)

        Args:
            num_freqs:
            hidden_size:
            sequence_model:
            output_activate_function:
            look_ahead:
        """
        super().__init__()
        if channel_attention_model:
            if channel_attention_model == "SE":
                self.channel_attention = ChannelSELayer(num_channels=self.num_channels)
            elif channel_attention_model == "ECA":
                self.channel_attention = ChannelECAlayer(channel=self.num_channels)
            elif channel_attention_model == "CBAM":
                self.channel_attention = ChannelCBAMLayer(num_channels=self.num_channels)
            else:
                raise NotImplementedError(f"Not implemented channel attention model {self.channel_attention}")
        self.fullband_model = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=hidden_size,
            num_layers=3,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=output_activate_function
        )

        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        if weight_init:
            print("Initializing model...")
            self.apply(self.weight_init)

    def forward(self, noisy_mag):
        """
        Args:
            noisy_mag: [B, 1, F, T], noisy magnitude spectrogram

        Returns:
            [B, 1, F, T], the magnitude of the enhanced spectrogram
        """
        assert noisy_mag.dim() == 4

        noisy_mag = functional.pad(noisy_mag, [0, self.look_ahead])  # Pad look ahead
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes the mag feature as inputs."

        fb_input = self.norm(noisy_mag).reshape(batch_size, num_channels * num_freqs, num_frames)  # [B, F, T]
        fb_input = self.channel_attention(fb_input)
        output = self.fullband_model(fb_input).reshape(batch_size, 1, num_freqs, num_frames)

        return output[:, :, :, self.look_ahead:]


if __name__ == "__main__":
    import datetime

    with torch.no_grad():
        ipt = torch.rand(1, 1, 161, 100)
        model = Model(
            num_freqs=161,
            look_ahead=1,
            sequence_model="LSTM",
            output_activate_function=None,
            hidden_size=512,
        )

        a = datetime.datetime.now()
        print(model(ipt).min())
        print(model(ipt).shape)
        b = datetime.datetime.now()
        print(f"{b - a}")
