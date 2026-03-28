"""
Test script to verify setup and model loading.
Run this after setup to make sure everything works.
"""
import sys
from pathlib import Path

def test_cuda():
    """Test CUDA availability."""
    print("\n=== Testing CUDA ===")
    import torch
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Quick GPU test
        x = torch.randn(1000, 1000, device="cuda")
        y = x @ x.T
        print(f"GPU compute test: OK (matrix mult completed)")
        return True
    else:
        print("WARNING: CUDA not available! Check your PyTorch installation.")
        return False


def test_transformers():
    """Test transformers library."""
    print("\n=== Testing Transformers ===")
    from transformers import AutoTokenizer, AutoConfig
    
    # Test with a small model config
    print("Transformers import: OK")
    return True


def test_nnsight():
    """Test nnsight library."""
    print("\n=== Testing nnsight ===")
    try:
        import nnsight
        print(f"nnsight version: {nnsight.__version__}")
        print("nnsight import: OK")
        return True
    except ImportError as e:
        print(f"nnsight import failed: {e}")
        return False


def test_local_model():
    """Check if local LM Studio model exists."""
    print("\n=== Checking Local Model ===")
    
    model_path = Path(r"C:\Users\locke\.lmstudio\models\bartowski\qwen_qwen3-4b-instruct-2507")
    
    if model_path.exists():
        print(f"Model directory found: {model_path}")
        gguf_files = list(model_path.glob("*.gguf"))
        if gguf_files:
            print(f"GGUF files: {[f.name for f in gguf_files]}")
            return True
        else:
            print("No GGUF files found in directory")
            return False
    else:
        print(f"Model directory not found: {model_path}")
        print("This is OK - we can download from HuggingFace instead")
        return False


def test_huggingface_model():
    """Test if we can load Qwen3 from HuggingFace."""
    print("\n=== Testing HuggingFace Model Access ===")
    from transformers import AutoConfig
    
    model_id = "Qwen/Qwen2.5-3B-Instruct"  # Similar size, definitely available
    
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print(f"Model config loaded: {model_id}")
        print(f"Hidden size: {config.hidden_size}")
        print(f"Num layers: {config.num_hidden_layers}")
        return True
    except Exception as e:
        print(f"Failed to load config: {e}")
        return False


def main():
    print("=" * 60)
    print("COLLABORATION CIRCUITS - SETUP VERIFICATION")
    print("=" * 60)
    
    results = {
        "CUDA": test_cuda(),
        "Transformers": test_transformers(),
        "nnsight": test_nnsight(),
        "Local Model": test_local_model(),
        "HuggingFace": test_huggingface_model(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_critical_ok = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{name}: {status}")
        if name in ["CUDA", "Transformers"] and not passed:
            all_critical_ok = False
    
    if all_critical_ok:
        print("\n[OK] All critical tests passed! Ready to proceed.")
        print("\nNext step: Run 'python scripts/extract_test.py' to test activation extraction")
    else:
        print("\n[ERROR] Some critical tests failed. Please fix before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()

