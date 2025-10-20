#!/usr/bin/env python3
"""
Safetensors形式のモデルをPyTorch bin形式に変換

TaDiCodecのファインチューニングには pytorch_model.bin が必要です。
Hugging FaceからダウンロードしたモデルはSafetensors形式なので変換します。
"""
import os
import torch
from safetensors.torch import load_file

def convert_safetensors_to_bin(safetensors_path, output_path):
    """
    Safetensors形式のモデルをPyTorch bin形式に変換

    Args:
        safetensors_path: 入力のsafetensorsファイルパス
        output_path: 出力のbinファイルパス
    """
    print(f"Loading model from: {safetensors_path}")

    # Safetensorsファイルをロード
    state_dict = load_file(safetensors_path)

    print(f"Model loaded successfully")
    print(f"  Keys: {len(state_dict)}")
    print(f"  First few keys: {list(state_dict.keys())[:5]}")

    # PyTorch bin形式で保存
    print(f"\nSaving model to: {output_path}")
    torch.save(state_dict, output_path)

    # ファイルサイズを確認
    input_size = os.path.getsize(safetensors_path) / 1024**2  # MB
    output_size = os.path.getsize(output_path) / 1024**2  # MB

    print(f"\nConversion complete!")
    print(f"  Input size:  {input_size:.2f} MB")
    print(f"  Output size: {output_size:.2f} MB")
    print(f"\nYou can now use this checkpoint for finetuning:")
    print(f"  --checkpoint_path {os.path.dirname(output_path)}")

if __name__ == "__main__":
    # TaDiCodecモデルのパス
    safetensors_path = "./ckpt/TaDiCodec/model.safetensors"
    output_path = "./ckpt/TaDiCodec/pytorch_model.bin"

    if not os.path.exists(safetensors_path):
        print(f"Error: {safetensors_path} not found!")
        print("Please download the TaDiCodec model first.")
        exit(1)

    convert_safetensors_to_bin(safetensors_path, output_path)
