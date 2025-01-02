import torch
from torch.nn import functional
import torch.cuda.nvtx as nvtx

from audio_zen.acoustics.feature import drop_band
from audio_zen.model.base_model import BaseModel
from audio_zen.model.module.sequence_model import SequenceModel
from audio_zen.model.module.attention_model import ChannelSELayer, ChannelECAlayer, ChannelCBAMLayer, \
    ChannelTimeSenseSELayer, ChannelTimeSenseAttentionSELayer, ChannelTimeSenseSEWeightLayer

# for log
from utils.logger import log

print = log


class FullSubNet_Plus(BaseModel):
    def __init__(self,
                 num_freqs,
                 look_ahead,
                 sequence_model="TCN",
                 fb_num_neighbors=15,
                 sb_num_neighbors=15,
                 fb_output_activate_function="ReLU",
                 sb_output_activate_function="ReLU",
                 fb_model_hidden_size=64,
                 sb_model_hidden_size=64,
                 channel_attention_model="SE",
                 norm_type="offline_laplace_norm",
                 num_groups_in_drop_band=2,
                 output_size=2,
                 subband_num=1,
                 weight_init=True,
                 ):
        """
        FullSubNet model (cIRM mask)

        Args:
            num_freqs: Frequency dim of the input
            sb_num_neighbors: Number of the neighbor frequencies in each side
            look_ahead: Number of use of the future frames
            sequence_model: Chose one sequence model as the basic model (GRU, LSTM)
        """
        super().__init__()
        
        # Simplify to only use TCN
        assert sequence_model == "TCN", "Only TCN is supported for ONNX export"
        
        self.num_channels = num_freqs if subband_num == 1 else (num_freqs // subband_num + 1)

        # Simplified SE attention
        self.channel_attention = ChannelSELayer(num_channels=self.num_channels)
        self.channel_attention_real = ChannelSELayer(num_channels=self.num_channels)
        self.channel_attention_imag = ChannelSELayer(num_channels=self.num_channels)

        # Use TCN for all sequence models
        self.fb_model = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model="TCN",
            output_activate_function=fb_output_activate_function
        )

        self.fb_model_real = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model="TCN",
            output_activate_function=fb_output_activate_function
        )

        self.fb_model_imag = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model="TCN",
            output_activate_function=fb_output_activate_function
        )

        self.sb_model = SequenceModel(
            input_size=(sb_num_neighbors * 2 + 1) + 3 * (fb_num_neighbors * 2 + 1),
            output_size=output_size,
            hidden_size=sb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model="TCN",
            output_activate_function=sb_output_activate_function
        )

        # Store configuration
        self.subband_num = subband_num
        self.sb_num_neighbors = sb_num_neighbors
        self.fb_num_neighbors = fb_num_neighbors
        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        self.num_groups_in_drop_band = num_groups_in_drop_band
        self.output_size = output_size

        if weight_init:
            self.apply(self.weight_init)

    def static_unfold(self, input, num_neighbor):
        """Static version of unfold operation"""
        batch_size, num_channels, num_freqs, num_frames = input.shape
        
        if num_neighbor < 1:
            return input.permute(0, 2, 1, 3).reshape(batch_size, num_freqs, num_channels, 1, num_frames)
        
        # Use fixed padding
        pad_size = num_neighbor
        padded = torch.nn.functional.pad(
            input.reshape(batch_size * num_channels, 1, num_freqs, num_frames),
            [0, 0, pad_size, pad_size],
            mode="reflect"
        )
        
        # Manual unfold using fixed indices
        sub_band_unit_size = num_neighbor * 2 + 1
        output = []
        
        for i in range(num_freqs):
            start_idx = i
            end_idx = i + sub_band_unit_size
            band = padded[:, :, start_idx:end_idx, :]
            output.append(band)
        
        output = torch.stack(output, dim=1)
        output = output.reshape(batch_size, num_channels, sub_band_unit_size, num_frames, num_freqs)
        output = output.permute(0, 4, 1, 2, 3).contiguous()
        
        return output

    def forward(self, noisy_mag, noisy_real, noisy_imag):
        """
        Args:
            noisy_mag: noisy magnitude spectrogram

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy_mag: [B, 1, F, T]
            noisy_real: [B, 1, F, T]
            noisy_imag: [B, 1, F, T]
            return: [B, 2, F, T]
        """
        nvtx.range_push("FullSubNet_Forward")
        
        # Input processing
        nvtx.range_push("Input_Processing")
        assert noisy_mag.dim() == 4
        noisy_mag = functional.pad(noisy_mag, [0, self.look_ahead])
        noisy_real = functional.pad(noisy_real, [0, self.look_ahead])
        noisy_imag = functional.pad(noisy_imag, [0, self.look_ahead])
        nvtx.range_pop()
        
        # Fullband processing
        nvtx.range_push("Fullband_Processing")
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes the mag feature as inputs."

        if self.subband_num == 1:
            fb_input = self.norm(noisy_mag).reshape(batch_size, num_channels * num_freqs, num_frames)  # [B, F, T]
            fb_input = self.channel_attention(fb_input)
        else:
            pad_num = self.subband_num - num_freqs % self.subband_num
            # Fullband model
            fb_input = functional.pad(self.norm(noisy_mag), [0, 0, 0, pad_num], mode="reflect")
            fb_input = fb_input.reshape(batch_size, (num_freqs + pad_num) // self.subband_num,
                                        num_frames * self.subband_num)  # [B, subband_num, T]
            fb_input = self.channel_attention(fb_input)
            fb_input = fb_input.reshape(batch_size, num_channels * (num_freqs + pad_num), num_frames)[:, :num_freqs, :]
        fb_output = self.fb_model(fb_input).reshape(batch_size, 1, num_freqs, num_frames)
        nvtx.range_pop()
        
        # Real/Imag processing
        nvtx.range_push("Complex_Processing")
        fbr_input = self.norm(noisy_real).reshape(batch_size, num_channels * num_freqs, num_frames)  # [B, F, T]
        fbr_input = self.channel_attention_real(fbr_input)
        fbr_output = self.fb_model_real(fbr_input).reshape(batch_size, 1, num_freqs, num_frames)
        fbi_input = self.norm(noisy_imag).reshape(batch_size, num_channels * num_freqs, num_frames)  # [B, F, T]
        fbi_input = self.channel_attention_imag(fbi_input)
        fbi_output = self.fb_model_imag(fbi_input).reshape(batch_size, 1, num_freqs, num_frames)
        nvtx.range_pop()
        
        # Unfolding operations
        nvtx.range_push("Unfolding")
        fb_output_unfolded = self.static_unfold(fb_output, num_neighbor=self.fb_num_neighbors)
        fb_output_unfolded = fb_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1,
                                                        num_frames)
        fbr_output_unfolded = self.static_unfold(fbr_output, num_neighbor=self.fb_num_neighbors)
        fbr_output_unfolded = fbr_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1,
                                                          num_frames)
        fbi_output_unfolded = self.static_unfold(fbi_output, num_neighbor=self.fb_num_neighbors)
        fbi_output_unfolded = fbi_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1,
                                                          num_frames)
        nvtx.range_pop()
        
        # Final processing
        nvtx.range_push("Final_Processing")
        noisy_mag_unfolded = self.static_unfold(fb_input.reshape(batch_size, 1, num_freqs, num_frames),
                                         num_neighbor=self.sb_num_neighbors)
        noisy_mag_unfolded = noisy_mag_unfolded.reshape(batch_size, num_freqs, self.sb_num_neighbors * 2 + 1,
                                                        num_frames)
        sb_input = torch.cat([noisy_mag_unfolded, fb_output_unfolded, fbr_output_unfolded, fbi_output_unfolded], dim=2)
        sb_input = self.norm(sb_input)

        # Speeding up training without significant performance degradation. These will be updated to the paper later.
        if batch_size > 1:
            sb_input = drop_band(sb_input.permute(0, 2, 1, 3),
                                 num_groups=self.num_groups_in_drop_band)  # [B, (F_s + F_f), F//num_groups, T]
            num_freqs = sb_input.shape[2]
            sb_input = sb_input.permute(0, 2, 1, 3)  # [B, F//num_groups, (F_s + F_f), T]

        sb_input = sb_input.reshape(
            batch_size * num_freqs,
            (self.sb_num_neighbors * 2 + 1) + 3 * (self.fb_num_neighbors * 2 + 1),
            num_frames
        )

        # [B * F, (F_s + F_f), T] => [B * F, 2, T] => [B, F, 2, T]
        sb_mask = self.sb_model(sb_input)
        sb_mask = sb_mask.reshape(batch_size, num_freqs, self.output_size, num_frames).permute(0, 2, 1, 3).contiguous()

        output = sb_mask[:, :, :, self.look_ahead:]
        nvtx.range_pop()
        nvtx.range_pop()
        return output
