import torch
from models.drowsiness_cnn import DrowsinessNet
import onnx
import onnxruntime as ort
import numpy as np

def export_to_onnx(model_path='best_model.pth', output_path='drowsiness_model.onnx'):
    """Export PyTorch model to ONNX format"""
    
    # Load model
    device = torch.device('cpu')
    model = DrowsinessNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 1, 64, 64)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully!")
    
    # Test inference
    test_onnx_inference(output_path)

def test_onnx_inference(onnx_path):
    """Test ONNX Runtime inference"""
    
    # Create session
    session = ort.InferenceSession(onnx_path)
    
    # Prepare input
    input_data = np.random.randn(1, 1, 64, 64).astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {'input': input_data})
    
    print(f"ONNX inference output shape: {outputs[0].shape}")
    print(f"ONNX inference successful!")

def benchmark_models(pytorch_model_path, onnx_model_path, num_iterations=100):
    """Compare PyTorch vs ONNX inference speed"""
    import time
    
    # PyTorch
    device = torch.device('cpu')
    model = DrowsinessNet().to(device)
    model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
    model.eval()
    
    dummy_input = torch.randn(1, 1, 64, 64)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark PyTorch
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    pytorch_time = (time.time() - start) / num_iterations
    
    # ONNX
    session = ort.InferenceSession(onnx_model_path)
    input_data = dummy_input.numpy()
    
    # Warmup
    for _ in range(10):
        _ = session.run(None, {'input': input_data})
    
    # Benchmark ONNX
    start = time.time()
    for _ in range(num_iterations):
        _ = session.run(None, {'input': input_data})
    onnx_time = (time.time() - start) / num_iterations
    
    # Results
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"PyTorch inference time: {pytorch_time*1000:.2f} ms")
    print(f"ONNX inference time: {onnx_time*1000:.2f} ms")
    print(f"Speedup: {pytorch_time/onnx_time:.2f}x")
    print(f"Latency reduction: {((pytorch_time-onnx_time)/pytorch_time)*100:.1f}%")

if __name__ == "__main__":
    export_to_onnx()
    benchmark_models('best_model.pth', 'drowsiness_model.onnx')
