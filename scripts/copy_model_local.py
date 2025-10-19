#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TaDiCodecモデルをローカルにコピー
"""

import shutil
import os
from pathlib import Path

# Source (Hugging Face cache)
source = Path.home() / ".cache" / "huggingface" / "hub" / "models--amphion--TaDiCodec" / "snapshots" / "e93c6666168407297a6b0a85b33f93aa07ec08ae"

# Destination (local ckpt)
dest = Path("ckpt") / "TaDiCodec"

print(f"Source: {source}")
print(f"Destination: {dest}")

# Create destination directory
dest.mkdir(parents=True, exist_ok=True)

# Copy all files
print("\nCopying files...")
for item in source.iterdir():
    dest_item = dest / item.name
    if item.is_file():
        print(f"  Copying {item.name}...")
        shutil.copy2(item, dest_item)
    elif item.is_dir():
        print(f"  Copying directory {item.name}/...")
        if dest_item.exists():
            shutil.rmtree(dest_item)
        shutil.copytree(item, dest_item)

print(f"\n✅ All files copied to {dest}")
