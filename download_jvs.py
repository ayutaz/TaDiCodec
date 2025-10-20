#!/usr/bin/env python3
"""
JVS Emilia データセットをダウンロード
"""
from huggingface_hub import snapshot_download
import os

print("Downloading JVS Emilia dataset from ayousanz/jvs_emilia...")
print("This may take several minutes depending on your connection speed.")

local_dir = "./data/jvs_emilia"

# データセットをダウンロード
snapshot_download(
    repo_id="ayousanz/jvs_emilia",
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)

print(f"\nDownload complete! Dataset saved to: {os.path.abspath(local_dir)}")

# データセット構造を確認
print("\nDataset structure:")
for root, dirs, files in os.walk(local_dir):
    level = root.replace(local_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # 最初の5ファイルのみ表示
        print(f'{subindent}{file}')
    if len(files) > 5:
        print(f'{subindent}... and {len(files) - 5} more files')
    if level >= 2:  # 深さ2まで表示
        break
