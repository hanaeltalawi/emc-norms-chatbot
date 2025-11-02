import torch
import subprocess
import sys

def check_gpu_availability():
    """Comprehensive GPU availability check"""
    print("=" * 50)
    print("GPU/CUDA AVAILABILITY CHECK")
    print("=" * 50)
    
    # 1. Basic PyTorch CUDA check
    print("1. PyTorch CUDA Support:")
    print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"   torch.cuda.device_count(): {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("   ❌ No CUDA-capable GPU detected by PyTorch")
    
    # 2. Check nvidia-smi
    print("\n2. NVIDIA System Management Interface:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("   ✅ nvidia-smi command successful")
            # Extract GPU info from nvidia-smi
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA' in line or 'GPU' in line or 'Memory' in line:
                    if line.strip():
                        print(f"   {line.strip()}")
        else:
            print("   ❌ nvidia-smi failed")
    except FileNotFoundError:
        print("   ❌ nvidia-smi not found (NVIDIA drivers likely not installed)")
    except subprocess.TimeoutExpired:
        print("   ❌ nvidia-smi timed out")
    except Exception as e:
        print(f"   ❌ nvidia-smi error: {e}")
    
    # 3. Check CUDA version compatibility
    print("\n3. CUDA Version Compatibility:")
    if torch.cuda.is_available():
        print(f"   PyTorch CUDA version: {torch.version.cuda}")
        print(f"   Current CUDA device: {torch.cuda.current_device()}")
    else:
        print("   ❌ CUDA not available for PyTorch")
    
    # 4. Test actual tensor operations
    print("\n4. GPU Tensor Operations Test:")
    if torch.cuda.is_available():
        try:
            # Create a tensor and move to GPU
            x = torch.randn(1000, 1000)
            x_gpu = x.cuda()
            
            # Perform operation on GPU
            y_gpu = torch.mm(x_gpu, x_gpu.t())
            
            # Move back to CPU
            y_cpu = y_gpu.cpu()
            
            print("   ✅ GPU tensor operations successful")
            print(f"   Tensor device: {x_gpu.device}")
            print(f"   Operation result shape: {y_cpu.shape}")
            
        except Exception as e:
            print(f"   ❌ GPU operations failed: {e}")
    else:
        print("   ❌ Skipping GPU operations - CUDA not available")
    
    # 5. Check HuggingFace embeddings device (like your app uses)
    print("\n5. HuggingFace Embeddings Device Test:")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Embeddings would use device: {device}")
        
        # Test creating embeddings
        if torch.cuda.is_available():
            embeddings = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                model_kwargs={'device': 'cuda'}
            )
            print("   ✅ HuggingFace embeddings configured for GPU")
        else:
            print("   ❌ HuggingFace embeddings falling back to CPU")
            
    except Exception as e:
        print(f"   ❌ HuggingFace test failed: {e}")
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    if torch.cuda.is_available():
        print("✅ GPU SHOULD BE AVAILABLE for your application")
        print("⚠️  If your app still uses CPU, check:")
        print("   - VectorStoreManager device configuration")
        print("   - Model loading device settings")
        print("   - Memory constraints")
    else:
        print("❌ GPU NOT AVAILABLE - app will use CPU")
        print("💡 Solutions:")
        print("   - Install NVIDIA drivers")
        print("   - Install CUDA toolkit")
        print("   - Check GPU compatibility")

if __name__ == "__main__":
    check_gpu_availability()