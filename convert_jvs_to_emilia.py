#!/usr/bin/env python3
"""
JVS metadata.jsonl を Emilia 標準形式（各話者ディレクトリに audio.json）に変換

変換後の構造:
data/jvs_emilia/JA/
├── JA_jvs001/
│   ├── audio.json
│   └── wav/
│       ├── audio_0.wav
│       ├── audio_1.wav
│       └── ...
├── JA_jvs002/
│   └── ...
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def convert_jvs_to_emilia_format(metadata_path, base_dir):
    """
    metadata.jsonl を読み込んで、Emilia 標準形式に変換

    Args:
        metadata_path: metadata.jsonl のパス
        base_dir: ベースディレクトリ（data/jvs_emilia）
    """
    print(f"Reading metadata from: {metadata_path}")

    # metadata.jsonl を読み込む
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = [json.loads(line) for line in f]

    print(f"Total samples: {len(metadata)}")

    # 話者ごとにグループ化
    speaker_data = defaultdict(list)
    for item in metadata:
        speaker = item['speaker']
        speaker_data[speaker].append(item)

    print(f"Total speakers: {len(speaker_data)}")

    # 各話者ディレクトリに audio.json を作成
    for speaker, items in tqdm(speaker_data.items(), desc="Converting speakers"):
        # 話者ディレクトリのパス（JA_jvs001 など）
        speaker_dir_name = f"JA_{speaker}"
        speaker_dir = Path(base_dir) / "JA" / speaker_dir_name

        if not speaker_dir.exists():
            print(f"Warning: Speaker directory not found: {speaker_dir}")
            continue

        # audio.json を作成
        audio_json = {}
        wav_dir = speaker_dir / "wav"

        for idx, item in enumerate(items):
            # 元のWAVファイル名
            original_wav_name = Path(item['wav']).name
            original_wav_path = speaker_dir / "wav" / original_wav_name

            # 新しいWAVファイル名（audio_0.wav, audio_1.wav, ...）
            new_wav_name = f"audio_{idx}.wav"
            new_wav_path = speaker_dir / new_wav_name

            # WAVファイルをコピー（またはシンボリックリンク）
            if original_wav_path.exists():
                # ハードリンクを作成（ディスク容量を節約）
                try:
                    if not new_wav_path.exists():
                        os.link(str(original_wav_path), str(new_wav_path))
                except OSError:
                    # ハードリンクが失敗した場合はコピー
                    if not new_wav_path.exists():
                        shutil.copy2(str(original_wav_path), str(new_wav_path))

            # audio.json エントリを作成
            audio_json[str(idx)] = {
                "text": item['text'],
                "duration": item['duration'],
                "language": item.get('language', 'ja'),
            }

            # オプショナルフィールドを追加
            if 'dnsmos' in item:
                audio_json[str(idx)]['dnsmos'] = item['dnsmos']

        # audio.json を保存
        audio_json_path = speaker_dir / "audio.json"
        with open(audio_json_path, 'w', encoding='utf-8') as f:
            json.dump(audio_json, f, ensure_ascii=False, indent=2)

        print(f"  Created {audio_json_path} with {len(audio_json)} samples")

    print("\nConversion complete!")
    print("\nDataset structure:")
    print("data/jvs_emilia/JA/")
    print("├── JA_jvs001/")
    print("│   ├── audio.json")
    print("│   ├── audio_0.wav")
    print("│   ├── audio_1.wav")
    print("│   └── ...")
    print("└── ...")

if __name__ == "__main__":
    metadata_path = "./data/jvs_emilia/metadata.jsonl"
    base_dir = "./data/jvs_emilia"

    convert_jvs_to_emilia_format(metadata_path, base_dir)
