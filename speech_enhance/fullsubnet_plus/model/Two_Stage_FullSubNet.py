import torch
from torch.nn import functional

from audio_zen.acoustics.feature import drop_band
from audio_zen.model.base_model import BaseModel
from audio_zen.acoustics.feature import mag_phase
import torch.nn as nn
from audio_zen.model.module.sequence_model import SequenceModel, Complex_SequenceModel
from fullsubnet_plus.model.Complex_fullsubnet import Complex_FullSubNet
from fullsubnet_plus.model.Amp_Attention_fullsubnet import FullSub_Att_FullSubNet, Full_Att_FullSubNet
from fullsubnet_plus.model.fullbandnet import FullBandNet

# for log
from utils.logger import log

print = log


class Two_Stage_FullSubNet_Large(BaseModel):
    def __init__(self,
                 num_freqs,
                 look_ahead,
                 sequence_model,
                 fb_num_neighbors,
                 sb_num_neighbors,
                 fb_output_activate_function,
                 sb_output_activate_function,
                 fb_model_hidden_size,
                 sb_model_hidden_size,
                 channel_attention_model="SE",
                 norm_type="offline_laplace_norm",
                 num_groups_in_drop_band=2,
                 subband_num=1,
                 kersize=[3, 5, 10],
                 weight_init=True,
                 ):
        """
        Two Stage model (cIRM mask)
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM"), f"{self.__class__.__name__} only support GRU and LSTM."

        self.amp_model = FullSub_Att_FullSubNet(
            num_freqs=num_freqs,
            look_ahead=look_ahead,
            sequence_model=sequence_model,
            fb_num_neighbors=fb_num_neighbors,
            sb_num_neighbors=sb_num_neighbors,
            fb_output_activate_function=fb_output_activate_function,
            sb_output_activate_function=sb_output_activate_function,
            fb_model_hidden_size=fb_model_hidden_size,
            sb_model_hidden_size=sb_model_hidden_size,
            channel_attention_model=channel_attention_model,
            norm_type=norm_type,
            num_groups_in_drop_band=1,
            output_size=1,
            subband_num=subband_num,
            kersize=kersize,
            weight_init=weight_init
        )

        self.middle_fc = nn.Linear(num_freqs * 2, num_freqs, bias=True)

        self.complex_model = Complex_FullSubNet(
            num_freqs=num_freqs,
            look_ahead=look_ahead,
            sequence_model=sequence_model,
            fb_num_neighbors=fb_num_neighbors,
            sb_num_neighbors=sb_num_neighbors,
            fb_output_activate_function=fb_output_activate_function,
            sb_output_activate_function=sb_output_activate_function,
            fb_model_hidden_size=fb_model_hidden_size,
            sb_model_hidden_size=sb_model_hidden_size,
            output_size=1,
            norm_type=norm_type,
            num_groups_in_drop_band=num_groups_in_drop_band,
            weight_init=weight_init
        )

        self.num_groups_in_drop_band = num_groups_in_drop_band

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_complex):
        """
        Args:
            noisy: noisy real part and imag part spectrogram

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy_complex: [B, F, T]
            return: [B, 2, F, T]
        """
        assert noisy_complex.dim() == 3
        noisy_mag, noisy_angle = mag_phase(noisy_complex)  # [B, F, T]
        noisy_mag = noisy_mag.unsqueeze(1)  # [B, 1, F, T]
        noisy_angle = noisy_angle.unsqueeze(1)  # [B, 1, F, T]
        noisy_real = (noisy_complex.real).unsqueeze(1)  # [B, 1, F, T]
        noisy_imag = (noisy_complex.imag).unsqueeze(1)  # [B, 1, F, T]

        IRM = self.amp_model(noisy_mag)  # IRM: [B, 1, F, T]
        enhanced_mag = IRM * noisy_mag  # [B, 1, F, T]

        enhanced_real = enhanced_mag * torch.cos(noisy_angle)
        enhanced_imag = enhanced_mag * torch.sin(noisy_angle)
        real_con = torch.cat([enhanced_real, noisy_real], dim=2)  # [B, 1, 2 * F, T]
        imag_con = torch.cat([enhanced_imag, noisy_imag], dim=2)  # [B, 1, 2 * F, T]

        # [B, 1, 2 * F, T] => [B, 1, T, 2 * F] => model => [B, 1, 2 * F, T]
        real_input = (self.middle_fc(real_con.permute(0, 1, 3, 2))).permute(0, 1, 3, 2)
        imag_input = (self.middle_fc(imag_con.permute(0, 1, 3, 2))).permute(0, 1, 3, 2)

        complex_input = torch.cat([real_input, imag_input], dim=1)  # [B, 2, F, T]
        cIRM = self.complex_model(complex_input)

        return IRM, cIRM


class Two_Stage_FullSubNet_Small(BaseModel):
    def __init__(self,
                 num_freqs,
                 look_ahead,
                 sequence_model,
                 fb_num_neighbors,
                 sb_num_neighbors,
                 fb_output_activate_function,
                 sb_output_activate_function,
                 fb_model_hidden_size,
                 sb_model_hidden_size,
                 channel_attention_model="SE",
                 norm_type="offline_laplace_norm",
                 num_groups_in_drop_band=2,
                 subband_num=1,
                 kersize=[3, 5, 10],
                 weight_init=True,
                 ):
        """
        Two Stage model (cIRM mask)
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM"), f"{self.__class__.__name__} only support GRU and LSTM."

        self.amp_model = FullBandNet(
            num_freqs=num_freqs,
            hidden_size=fb_model_hidden_size,
            look_ahead=look_ahead,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function,
            channel_attention_model=channel_attention_model,
            norm_type=norm_type,
            subband_num=subband_num,
            kersize=kersize,
            weight_init=weight_init
        )

        self.middle_fc = nn.Linear(num_freqs * 2, num_freqs, bias=True)

        self.complex_model = Complex_FullSubNet(
            num_freqs=num_freqs,
            look_ahead=look_ahead,
            sequence_model=sequence_model,
            fb_num_neighbors=fb_num_neighbors,
            sb_num_neighbors=sb_num_neighbors,
            fb_output_activate_function=fb_output_activate_function,
            sb_output_activate_function=sb_output_activate_function,
            fb_model_hidden_size=fb_model_hidden_size,
            sb_model_hidden_size=sb_model_hidden_size,
            output_size=2,
            norm_type=norm_type,
            num_groups_in_drop_band=num_groups_in_drop_band,
            weight_init=weight_init
        )

        self.num_groups_in_drop_band = num_groups_in_drop_band

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_complex):
        """
        Args:
            noisy: noisy real part and imag part spectrogram

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy_complex: [B, F, T]
            return: [B, 2, F, T]
        """
        assert noisy_complex.dim() == 3

        noisy_mag, noisy_angle = mag_phase(noisy_complex)  # [B, F, T]
        noisy_mag = noisy_mag.unsqueeze(1)  # [B, 1, F, T]
        noisy_angle = noisy_angle.unsqueeze(1)  # [B, 1, F, T]
        noisy_real = (noisy_complex.real).unsqueeze(1)  # [B, 1, F, T]
        noisy_imag = (noisy_complex.imag).unsqueeze(1)  # [B, 1, F, T]

        IRM = self.amp_model(noisy_mag)  # IRM: [B, 1, F, T]
        enhanced_mag = IRM * noisy_mag  # [B, 1, F, T]

        enhanced_real = enhanced_mag * torch.cos(noisy_angle)
        enhanced_imag = enhanced_mag * torch.sin(noisy_angle)
        real_con = torch.cat([enhanced_real, noisy_real], dim=2)  # [B, 1, 2 * F, T]
        imag_con = torch.cat([enhanced_imag, noisy_imag], dim=2)  # [B, 1, 2 * F, T]

        # [B, 1, 2 * F, T] => [B, 1, T, 2 * F] => model => [B, 1, 2 * F, T]
        real_input = (self.middle_fc(real_con.permute(0, 1, 3, 2))).permute(0, 1, 3, 2)
        imag_input = (self.middle_fc(imag_con.permute(0, 1, 3, 2))).permute(0, 1, 3, 2)

        complex_input = torch.cat([real_input, imag_input], dim=1)  # [B, 2, F, T]
        cIRM = self.complex_model(complex_input)

        return IRM, cIRM


class Two_Stage_AmpAttenion_FullSubNet(BaseModel):
    def __init__(self,
                 num_freqs,
                 look_ahead,
                 sequence_model,
                 fb_num_neighbors,
                 sb_num_neighbors,
                 fb_output_activate_function,
                 sb_output_activate_function,
                 fb_model_hidden_size,
                 sb_model_hidden_size,
                 channel_attention_model="SE",
                 norm_type="offline_laplace_norm",
                 num_groups_in_drop_band=2,
                 subband_num=1,
                 kersize=[3, 5, 10],
                 weight_init=True,
                 ):
        """
        Two Stage model (cIRM mask)
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM"), f"{self.__class__.__name__} only support GRU and LSTM."

        self.amp_model = FullBandNet(
            num_freqs=num_freqs,
            hidden_size=fb_model_hidden_size,
            look_ahead=look_ahead,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function,
            channel_attention_model=channel_attention_model,
            norm_type=norm_type,
            subband_num=subband_num,
            kersize=kersize,
            weight_init=weight_init
        )

        self.middle_fc = nn.Linear(num_freqs * 2, num_freqs, bias=True)

        self.real_model = FullSub_Att_FullSubNet(
            num_freqs=num_freqs,
            look_ahead=look_ahead,
            sequence_model=sequence_model,
            fb_num_neighbors=fb_num_neighbors,
            sb_num_neighbors=sb_num_neighbors,
            fb_output_activate_function=fb_output_activate_function,
            sb_output_activate_function=sb_output_activate_function,
            fb_model_hidden_size=fb_model_hidden_size,
            sb_model_hidden_size=sb_model_hidden_size,
            channel_attention_model=channel_attention_model,
            norm_type=norm_type,
            num_groups_in_drop_band=1,
            output_size=1,
            subband_num=subband_num,
            kersize=kersize,
            weight_init=weight_init
        )

        self.imag_model = FullSub_Att_FullSubNet(
            num_freqs=num_freqs,
            look_ahead=look_ahead,
            sequence_model=sequence_model,
            fb_num_neighbors=fb_num_neighbors,
            sb_num_neighbors=sb_num_neighbors,
            fb_output_activate_function=fb_output_activate_function,
            sb_output_activate_function=sb_output_activate_function,
            fb_model_hidden_size=fb_model_hidden_size,
            sb_model_hidden_size=sb_model_hidden_size,
            channel_attention_model=channel_attention_model,
            norm_type=norm_type,
            num_groups_in_drop_band=1,
            output_size=1,
            subband_num=subband_num,
            kersize=kersize,
            weight_init=weight_init
        )

        self.num_groups_in_drop_band = num_groups_in_drop_band

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_complex):
        """
        Args:
            noisy: noisy real part and imag part spectrogram

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy_complex: [B, F, T]
            return: [B, 2, F, T]
        """
        assert noisy_complex.dim() == 3
        noisy_mag, noisy_angle = mag_phase(noisy_complex)  # [B, F, T]
        noisy_mag = noisy_mag.unsqueeze(1)  # [B, 1, F, T]
        noisy_angle = noisy_angle.unsqueeze(1)  # [B, 1, F, T]
        noisy_real = (noisy_complex.real).unsqueeze(1)  # [B, 1, F, T]
        noisy_imag = (noisy_complex.imag).unsqueeze(1)  # [B, 1, F, T]

        IRM = self.amp_model(noisy_mag)  # IRM: [B, 1, F, T]
        enhanced_mag = IRM * noisy_mag  # [B, 1, F, T]

        enhanced_real = enhanced_mag * torch.cos(noisy_angle)
        enhanced_imag = enhanced_mag * torch.sin(noisy_angle)
        real_con = torch.cat([enhanced_real, noisy_real], dim=2)  # [B, 1, 2 * F, T]
        imag_con = torch.cat([enhanced_imag, noisy_imag], dim=2)  # [B, 1, 2 * F, T]

        # [B, 1, 2 * F, T] => [B, 1, T, 2 * F] => model => [B, 1, 2 * F, T]
        real_input = (self.middle_fc(real_con.permute(0, 1, 3, 2))).permute(0, 1, 3, 2)
        imag_input = (self.middle_fc(imag_con.permute(0, 1, 3, 2))).permute(0, 1, 3, 2)

        real_output = self.real_model(real_input)
        imag_output = self.imag_model(imag_input)
        cIRM = torch.cat([real_output, imag_output], dim=1)  # [B, 2, F, T]

        return IRM, cIRM
