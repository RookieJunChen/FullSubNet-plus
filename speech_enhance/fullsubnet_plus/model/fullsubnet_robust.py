import torch
from speech_enhance.fullsubnet_plus.model.fullsubnet_plus import FullSubNet_Plus
from audio_zen.acoustics.feature import drop_band

class FullSubNet_Robust(FullSubNet_Plus):
    """
    ONNX/TensorRT friendly version of FullSubNet_Plus with:
    - Static unfolding operation
    - TCN-only sequence models
    - SE-only channel attention
    - Fixed batch size of 1 for inference
    """
    def __init__(self, 
                 num_freqs,
                 look_ahead,
                 fb_num_neighbors=15,
                 sb_num_neighbors=15,
                 fb_model_hidden_size=32,
                 sb_model_hidden_size=32,
                 norm_type="offline_laplace_norm",
                 output_size=2):
        
        # Call parent with memory-optimized parameters
        super().__init__(
            num_freqs=num_freqs,
            look_ahead=look_ahead,
            sequence_model="TCN",
            fb_num_neighbors=fb_num_neighbors,
            sb_num_neighbors=sb_num_neighbors,
            fb_output_activate_function="ReLU",
            sb_output_activate_function="ReLU",
            fb_model_hidden_size=fb_model_hidden_size,
            sb_model_hidden_size=sb_model_hidden_size,
            channel_attention_model="SE",
            norm_type=norm_type,
            num_groups_in_drop_band=1,
            output_size=output_size,
            subband_num=1,
            weight_init=True
        )
        

        # Override unfold with static version
        self.unfold = self.static_unfold

    def forward(self, noisy_mag, noisy_real, noisy_imag):
        """ONNX-friendly forward pass with proper complex handling"""
        # Basic setup and ensure padding is applied to all inputs
        noisy_mag = noisy_mag.contiguous()
        noisy_real = noisy_real.contiguous()
        noisy_imag = noisy_imag.contiguous()
        
        # Add padding to all inputs
        noisy_mag = torch.nn.functional.pad(noisy_mag, [0, self.look_ahead])
        noisy_real = torch.nn.functional.pad(noisy_real, [0, self.look_ahead])
        noisy_imag = torch.nn.functional.pad(noisy_imag, [0, self.look_ahead])
        
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        
        # Normalize inputs to prevent NaN
        eps = 1e-8
        noisy_mag = noisy_mag.clamp(min=eps)
        
        # Process fullband features with gradient scaling to prevent NaN
        with torch.cuda.amp.autocast(enabled=False):
            fb_in = self.norm(noisy_mag).reshape(batch_size, num_channels * num_freqs, num_frames)
            fb_out = self.channel_attention(fb_in)
            fb_out = self.fb_model(fb_out).reshape(batch_size, 1, num_freqs, num_frames)
            
            fbr_in = self.norm(noisy_real).reshape(batch_size, num_channels * num_freqs, num_frames)
            fbr_out = self.channel_attention_real(fbr_in)
            fbr_out = self.fb_model_real(fbr_out).reshape(batch_size, 1, num_freqs, num_frames)
            
            fbi_in = self.norm(noisy_imag).reshape(batch_size, num_channels * num_freqs, num_frames)
            fbi_out = self.channel_attention_imag(fbi_in)
            fbi_out = self.fb_model_imag(fbi_out).reshape(batch_size, 1, num_freqs, num_frames)
        
        # Get unfolded features
        fb_unfolded = self.unfold(fb_out, self.fb_num_neighbors)
        fbr_unfolded = self.unfold(fbr_out, self.fb_num_neighbors)
        fbi_unfolded = self.unfold(fbi_out, self.fb_num_neighbors)
        noisy_unfolded = self.unfold(noisy_mag, self.sb_num_neighbors)
        
        # Prepare subband input
        sb_input = torch.cat([
            noisy_unfolded.reshape(batch_size, num_freqs, self.sb_num_neighbors * 2 + 1, num_frames),
            fb_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1, num_frames),
            fbr_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1, num_frames),
            fbi_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1, num_frames)
        ], dim=2)
        
        sb_input = self.norm(sb_input)
        
        # Process through subband model
        sb_input = sb_input.reshape(
            batch_size * num_freqs,
            (self.sb_num_neighbors * 2 + 1) + 3 * (self.fb_num_neighbors * 2 + 1),
            num_frames
        )
        
        # Process in chunks with gradient scaling
        chunk_size = 8
        num_chunks = (num_freqs + chunk_size - 1) // chunk_size
        
        output = torch.zeros(batch_size, self.output_size, num_freqs, num_frames, 
                            device=sb_input.device, dtype=sb_input.dtype)
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, num_freqs)
            
            chunk_input = sb_input[start_idx * batch_size:end_idx * batch_size]
            with torch.cuda.amp.autocast(enabled=False):
                chunk_output = self.sb_model(chunk_input)
            
            chunk_output = chunk_output.reshape(batch_size, end_idx - start_idx, 
                                              self.output_size, num_frames)
            chunk_output = chunk_output.permute(0, 2, 1, 3)
            
            output[:, :, start_idx:end_idx] = chunk_output
        
        # Ensure output is properly scaled and bounded
        output = output.clamp(-10, 10)  # Prevent extreme values
        
        return output[:, :, :, self.look_ahead:]

    def export_onnx(self, path, input_shape=(1, 1, 65, 100)):
        """Export model to ONNX format"""
        dummy_mag = torch.randn(input_shape, dtype=torch.float32)
        dummy_real = torch.randn(input_shape, dtype=torch.float32)
        dummy_imag = torch.randn(input_shape, dtype=torch.float32)
        
        torch.onnx.export(
            self,
            (dummy_mag, dummy_real, dummy_imag),
            path,
            input_names=['noisy_mag', 'noisy_real', 'noisy_imag'],
            output_names=['enhanced'],
            dynamic_axes={
                'noisy_mag': {3: 'time'},
                'noisy_real': {3: 'time'},
                'noisy_imag': {3: 'time'},
                'enhanced': {3: 'time'}
            },
            opset_version=11
        )

    @classmethod
    def from_pretrained(cls, checkpoint_path, map_location="cpu"):
        """
        Create a FullSubNet_Robust instance from a pretrained FullSubNet_Plus checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        config = checkpoint.get("config", {})
        
        # Extract relevant parameters from checkpoint
        model = cls(
            num_freqs=config.get("num_freqs", 257),  # Default for 512 FFT
            look_ahead=config.get("look_ahead", 2),
            fb_num_neighbors=config.get("fb_num_neighbors", 15),
            sb_num_neighbors=config.get("sb_num_neighbors", 15),
            fb_model_hidden_size=config.get("fb_model_hidden_size", 32),
            sb_model_hidden_size=config.get("sb_model_hidden_size", 32),
            norm_type=config.get("norm_type", "offline_laplace_norm"),
            output_size=config.get("output_size", 2)
        )
        
        # Load state dict
        model.load_state_dict(checkpoint["model"])
        return model 