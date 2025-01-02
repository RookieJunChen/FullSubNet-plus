import torch
import argparse
import toml
from pathlib import Path
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from speech_enhance.fullsubnet_plus.model.fullsubnet_robust import FullSubNet_Robust

def export_to_onnx(model, save_path, input_shape=(1, 161, 400)):
    """Export PyTorch model to ONNX format"""
    batch_size, num_freqs, seq_len = input_shape
    
    # Create dummy inputs
    dummy_mag = torch.randn(batch_size, 1, num_freqs, seq_len, device='cuda')
    dummy_real = torch.randn(batch_size, 1, num_freqs, seq_len, device='cuda')
    dummy_imag = torch.randn(batch_size, 1, num_freqs, seq_len, device='cuda')
    
    # Define dynamic axes
    dynamic_axes = {
        'magnitude': {0: 'batch_size', 3: 'sequence_length'},
        'real': {0: 'batch_size', 3: 'sequence_length'},
        'imaginary': {0: 'batch_size', 3: 'sequence_length'},
        'output': {0: 'batch_size', 3: 'sequence_length'}
    }
    
    # Export with dynamic batch size
    torch.onnx.export(
        model,
        (dummy_mag, dummy_real, dummy_imag),
        save_path,
        input_names=['magnitude', 'real', 'imaginary'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,  # Specify dynamic axes
        opset_version=11,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        training=torch.onnx.TrainingMode.EVAL,
        verbose=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    )
    print(f"ONNX model saved to: {save_path}")

def verify_onnx(onnx_path, input_shape=(1, 161, 400)):
    """Verify ONNX model"""
    import onnx
    import onnxruntime
    import numpy as np

    # Check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Create ONNX Runtime session with CUDA provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    
    # Prepare inputs
    batch_size, num_freqs, seq_len = input_shape
    dummy_mag = np.random.randn(batch_size, 1, num_freqs, seq_len).astype(np.float32)
    dummy_real = np.random.randn(batch_size, 1, num_freqs, seq_len).astype(np.float32)
    dummy_imag = np.random.randn(batch_size, 1, num_freqs, seq_len).astype(np.float32)
    
    # Get input names from model
    input_names = [input.name for input in ort_session.get_inputs()]
    print(f"ONNX model input names: {input_names}")
    
    # Run ONNX inference
    ort_inputs = {
        input_names[0]: dummy_mag,
        input_names[1]: dummy_real,
        input_names[2]: dummy_imag
    }
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print("ONNX model verified successfully!")
    return ort_outputs[0]

def convert_onnx_to_trt(onnx_path, engine_path, precision='fp16'):
    """Convert ONNX model to TensorRT engine"""
    import tensorrt as trt
    import pycuda.autoinit
    
    logger = trt.Logger(trt.Logger.INFO)
    
    # Create builder and network
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    # Set precision flags
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Create network with explicit batch
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Create ONNX parser
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX model")
    
    # Create optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    
    # Set shape ranges for all three inputs
    for input_idx in range(3):  # magnitude, real, imaginary
        input_tensor = network.get_input(input_idx)
        input_shape = input_tensor.shape
        
        # Set shape ranges for dynamic batch size and sequence length
        min_shape = (1, 1, input_shape[2], 32)  # minimum sequence length
        opt_shape = (1, 1, input_shape[2], 400)  # optimal sequence length
        max_shape = (1, 1, input_shape[2], 1000)  # maximum sequence length
        
        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    
    config.add_optimization_profile(profile)
    
    # Build and save engine
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"TensorRT engine saved to: {engine_path}")

def get_input_shape(config):
    """Get input shape from acoustics config"""
    n_fft = config["acoustics"]["n_fft"]  # 512 from your config
    num_freqs = n_fft // 2 + 1  # 257 frequencies
    
    # Calculate sequence length based on hop_length
    hop_length = config["acoustics"]["hop_length"]  # 256 from your config
    # Assuming 1 second of audio at 16kHz
    audio_length = config["acoustics"]["sr"]  # 16000
    sequence_length = (audio_length - n_fft) // hop_length + 1
    
    batch_size = 1  # Default for inference
    
    shape = (batch_size, num_freqs, sequence_length)
    print(f"Input shape calculated from acoustics config: {shape}")
    return shape

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config", required=True, type=str)
    parser.add_argument("-M", "--model_checkpoint", required=True, type=str)
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    args = parser.parse_args()

    # Load config
    config = toml.load(args.config)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
    
    # Create robust model instance
    model = FullSubNet_Robust.from_pretrained(args.model_checkpoint)
    model.cuda().eval()

    # Get input shape from actual feature transformation
    input_shape = get_input_shape(config)
    print(f"Using input shape from feature transform: {input_shape}")

    # Export paths
    base_path = Path(args.model_checkpoint).parent
    model_name = Path(args.model_checkpoint).stem
    onnx_path = base_path / f"{model_name}.onnx"
    engine_path = base_path / f"{model_name}.engine"

    # Export to ONNX
    print("Exporting to ONNX...")
    export_to_onnx(model, onnx_path, input_shape)
    
    # Rest of the export process...

if __name__ == "__main__":
    main() 