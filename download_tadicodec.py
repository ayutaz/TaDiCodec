#!/usr/bin/env python3
"""
TaDiCodec モデルのダウンロード
"""
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from huggingface_hub import snapshot_download

print("Downloading TaDiCodec model from Hugging Face...")

local_dir = "./ckpt/TaDiCodec"

# モデルをダウンロード（シンボリックリンク無効）
snapshot_download(
    repo_id="amphion/TaDiCodec",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)

print(f"Download complete! Model saved to: {os.path.abspath(local_dir)}")
print(f"Tokenizer path: {os.path.abspath(local_dir)}/text_tokenizer")
