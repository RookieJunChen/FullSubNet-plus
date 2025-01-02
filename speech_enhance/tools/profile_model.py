import torch
import time
import argparse
from pathlib import Path
import toml
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import onnxruntime
import sys
import os
from audio_zen.utils import initialize_module

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

def load_tensorrt_engine(engine_path):
    """Load TensorRT engine"""
    logger = trt.Logger(trt.Logger.INFO)
    with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def profile_inference_trt(engine, input_size=(1, 161, 400), warmup=50, iterations=100):
    """Profile TensorRT engine inference performance"""
    import pycuda.autoinit
    
    # Create execution context
    context = engine.create_execution_context()
    
    # Set optimal input shape
    context.set_optimization_profile_async(0, stream.handle)
    context.set_binding_shape(0, input_size)
    
    # Allocate host and device buffers
    h_input = cuda.pagelocked_empty(input_size, dtype=np.float32)
    h_output = cuda.pagelocked_empty(input_size, dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    stream = cuda.Stream()

    # Warmup
    for _ in range(warmup):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    # Measure performance
    start_time = time.time()
    for _ in range(iterations):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    return avg_time

def profile_inference(model, input_shape=(1, 161, 400), warmup=50, iterations=100):
    """Profile model inference performance"""
    device = next(model.parameters()).device
    batch_size, num_freqs, seq_len = input_shape
    
    # Create dummy inputs matching inferencer format
    dummy_mag = torch.randn(batch_size, 1, num_freqs, seq_len, device=device)
    dummy_real = torch.randn(batch_size, 1, num_freqs, seq_len, device=device)
    dummy_imag = torch.randn(batch_size, 1, num_freqs, seq_len, device=device)
    
    # Warmup
    for _ in range(warmup):
        _ = model(dummy_mag, dummy_real, dummy_imag)
    
    torch.cuda.synchronize()
    
    # Measure performance
    start_time = time.time()
    for _ in range(iterations):
        _ = model(dummy_mag, dummy_real, dummy_imag)
        torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    return avg_time

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

def profile_inference_onnx(onnx_path, input_size=(1, 161, 400), warmup=50, iterations=100):
    """Profile ONNX model inference performance"""
    # Initialize ONNX Runtime
    session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    
    # Create dummy input
    dummy_input = np.random.randn(*input_size).astype(np.float32)
    ort_inputs = {session.get_inputs()[0].name: dummy_input}
    
    # Warmup
    for _ in range(warmup):
        _ = session.run(None, ort_inputs)
    
    # Measure performance
    start_time = time.time()
    for _ in range(iterations):
        _ = session.run(None, ort_inputs)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    return avg_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config", required=True, type=str)
    parser.add_argument("-M", "--model_checkpoint", required=True, type=str)
    args = parser.parse_args()

    # Load config
    config = toml.load(args.config)
    
    # Initialize model using the same method as base_inferencer
    model = initialize_module(config["model"]["path"], args=config["model"]["args"], initialize=True)
    
    # Load checkpoint the same way as base_inferencer
    model_checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
    model_static_dict = model_checkpoint["model"]
    
    model.load_state_dict(model_static_dict)
    model.cuda()
    model.eval()

    # Get input shape from config
    input_shape = get_input_shape(config)
    print(f"Using input shape: {input_shape}")
    
    # Profile original PyTorch model
    print("\nProfiling PyTorch model...")
    torch_time = profile_inference(model, input_shape)
    print(f"PyTorch Average Inference Time: {torch_time*1000:.2f}ms")
    
    # Profile ONNX model if it exists
    onnx_path = Path(args.model_checkpoint).with_suffix('.onnx')
    if onnx_path.exists():
        print("\nProfiling ONNX model...")
        onnx_time = profile_inference_onnx(onnx_path, input_shape)
        print(f"ONNX Average Inference Time: {onnx_time*1000:.2f}ms")
        print(f"ONNX Speedup vs PyTorch: {torch_time/onnx_time:.2f}x")
    
    # Profile TensorRT model if it exists
    engine_path = Path(args.model_checkpoint).with_suffix('.engine')
    if engine_path.exists():
        trt_model = load_tensorrt_engine(engine_path)
        print("\nProfiling TensorRT model...")
        trt_time = profile_inference_trt(trt_model, input_shape)
        print(f"TensorRT Average Inference Time: {trt_time*1000:.2f}ms")
        print(f"TensorRT Speedup vs PyTorch: {torch_time/trt_time:.2f}x")
        if onnx_path.exists():
            print(f"TensorRT Speedup vs ONNX: {onnx_time/trt_time:.2f}x")

if __name__ == "__main__":
    main() 