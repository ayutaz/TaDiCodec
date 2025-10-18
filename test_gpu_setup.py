#!/usr/bin/env python3
"""
Simple test script to verify GPU setup for TaDiCodec
"""

import torch
import sys

print("="*60)
print("GPU Environment Test")
print("="*60)

# Test PyTorch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Test simple GPU operation
    print("\nTesting GPU operation...")
    device = torch.device("cuda")
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.matmul(x, y)
    print("GPU computation successful!")
else:
    print("WARNING: CUDA is not available!")
    sys.exit(1)

# Test imports
print("\nTesting package imports...")
try:
    from models.tts.tadicodec.inference_tadicodec import TaDiCodecPipline
    print("- TaDiCodecPipline imported successfully")
except Exception as e:
    print(f"- Failed to import TaDiCodecPipline: {e}")

try:
    from models.tts.llm_tts.inference_llm_tts import TTSInferencePipeline
    print("- TTSInferencePipeline imported successfully")
except Exception as e:
    print(f"- Failed to import TTSInferencePipeline: {e}")

try:
    from models.tts.llm_tts.inference_mgm_tts import MGMInferencePipeline
    print("- MGMInferencePipeline imported successfully")
except Exception as e:
    print(f"- Failed to import MGMInferencePipeline: {e}")

print("\n" + "="*60)
print("GPU environment setup complete!")
print("="*60)
