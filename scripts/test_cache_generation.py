#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
キャッシュ生成スクリプトのテスト

サンプルデータを生成してcreate_dataset_cache.pyの動作をテストします。

使用方法:
    python scripts/test_cache_generation.py
"""

import os
import json
import pickle
import shutil
import tempfile
import numpy as np
from pathlib import Path
import soundfile as sf


def generate_sample_audio(duration=2.0, sample_rate=24000):
    """
    サンプル音声を生成（正弦波）

    Args:
        duration: 音声の長さ（秒）
        sample_rate: サンプリングレート

    Returns:
        音声データ（numpy array）
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    # 440Hz（A4）の正弦波
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio


def create_sample_dataset(output_dir: str, num_speakers=3, audios_per_speaker=5):
    """
    Emilia形式のサンプルデータセットを生成

    Args:
        output_dir: 出力ディレクトリ
        num_speakers: 話者数
        audios_per_speaker: 話者あたりの音声ファイル数
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sample_texts = [
        "東京の天気は晴れです。",
        "こんにちは、元気ですか？",
        "今日は良い天気ですね。",
        "ありがとうございます。",
        "さようなら、また明日。",
        "コンピューターでプログラムを実行します。",
        "インターネットでデータを送信します。",
        "日本語の音声合成システムです。",
    ]

    for speaker_id in range(1, num_speakers + 1):
        # 話者ディレクトリを作成
        speaker_dir = output_path / f"speaker{speaker_id}"
        speaker_dir.mkdir(parents=True, exist_ok=True)

        # audio.jsonを作成
        audio_metadata = {}

        for audio_id in range(audios_per_speaker):
            # WAVファイルを生成
            duration = np.random.uniform(1.5, 5.0)  # 1.5〜5.0秒
            audio = generate_sample_audio(duration=duration, sample_rate=24000)

            wav_filename = f"audio_{audio_id}.wav"
            wav_path = speaker_dir / wav_filename

            # WAVファイルを保存
            sf.write(wav_path, audio, 24000)

            # メタデータを追加
            text = sample_texts[audio_id % len(sample_texts)]
            audio_metadata[str(audio_id)] = {
                "text": text,
                "duration": duration,
                "language": "ja"
            }

        # audio.jsonを保存
        audio_json_path = speaker_dir / "audio.json"
        with open(audio_json_path, 'w', encoding='utf-8') as f:
            json.dump(audio_metadata, f, ensure_ascii=False, indent=2)

        print(f"Created speaker{speaker_id} with {audios_per_speaker} audio files")

    print(f"\nSample dataset created at: {output_dir}")
    print(f"Total speakers: {num_speakers}")
    print(f"Total audio files: {num_speakers * audios_per_speaker}")


def verify_cache_files(cache_dir: str):
    """
    キャッシュファイルの内容を検証

    Args:
        cache_dir: キャッシュディレクトリ
    """
    cache_path = Path(cache_dir)

    print("\n" + "="*80)
    print("Verifying cache files...")
    print("="*80)

    # WAVパスキャッシュ
    wav_paths_cache = cache_path / "wav_paths_cache.pkl"
    if wav_paths_cache.exists():
        with open(wav_paths_cache, 'rb') as f:
            wav_paths = pickle.load(f)
        print(f"\n✓ wav_paths_cache.pkl:")
        print(f"  - Total files: {len(wav_paths)}")
        print(f"  - First 3 files:")
        for i, path in enumerate(wav_paths[:3]):
            print(f"    {i+1}. {path}")
    else:
        print("\n❌ wav_paths_cache.pkl not found")
        return False

    # durationキャッシュ
    duration_cache = cache_path / "duration_cache.pkl"
    if duration_cache.exists():
        with open(duration_cache, 'rb') as f:
            durations = pickle.load(f)
        print(f"\n✓ duration_cache.pkl:")
        print(f"  - Total durations: {len(durations)}")
        print(f"  - Average duration: {sum(durations) / len(durations):.2f} seconds")
        print(f"  - Min duration: {min(durations):.2f} seconds")
        print(f"  - Max duration: {max(durations):.2f} seconds")
    else:
        print("\n❌ duration_cache.pkl not found")
        return False

    # トークン数キャッシュ
    token_count_cache = cache_path / "bpe_token_count_cache.pkl"
    if token_count_cache.exists():
        with open(token_count_cache, 'rb') as f:
            token_counts = pickle.load(f)
        print(f"\n✓ bpe_token_count_cache.pkl:")
        print(f"  - Total token counts: {len(token_counts)}")
        print(f"  - Average token count: {sum(token_counts) / len(token_counts):.2f}")
        print(f"  - Min token count: {min(token_counts)}")
        print(f"  - Max token count: {max(token_counts)}")
    else:
        print("\n❌ bpe_token_count_cache.pkl not found")
        return False

    # JSONメタデータキャッシュ（オプション）
    json_paths_cache = cache_path / "json_paths_cache.pkl"
    if json_paths_cache.exists():
        with open(json_paths_cache, 'rb') as f:
            json_metadata = pickle.load(f)
        print(f"\n✓ json_paths_cache.pkl:")
        print(f"  - Total metadata entries: {len(json_metadata)}")
        print(f"  - First entry:")
        print(f"    {json_metadata[0]}")
    else:
        print("\n⚠️  json_paths_cache.pkl not found (optional)")

    # データの整合性チェック
    if len(wav_paths) == len(durations) == len(token_counts):
        print(f"\n✅ All cache files have consistent length: {len(wav_paths)}")
    else:
        print(f"\n❌ Cache files have inconsistent lengths:")
        print(f"  - wav_paths: {len(wav_paths)}")
        print(f"  - durations: {len(durations)}")
        print(f"  - token_counts: {len(token_counts)}")
        return False

    print("\n" + "="*80)
    print("✅ Cache verification completed successfully!")
    print("="*80)

    return True


def main():
    """メイン処理"""
    print("="*80)
    print("キャッシュ生成スクリプトのテスト")
    print("="*80)

    # 一時ディレクトリを作成
    temp_dir = tempfile.mkdtemp(prefix="tadicodec_test_")
    sample_data_dir = os.path.join(temp_dir, "sample_data")
    sample_cache_dir = os.path.join(temp_dir, "sample_cache")

    try:
        # ステップ1: サンプルデータセットを生成
        print("\n[Step 1] Generating sample dataset...")
        create_sample_dataset(
            output_dir=sample_data_dir,
            num_speakers=3,
            audios_per_speaker=5
        )

        # ステップ2: キャッシュ生成スクリプトを実行
        print("\n[Step 2] Running cache generation script...")
        print("Command:")
        print(f"  python scripts/create_dataset_cache.py \\")
        print(f"    --data_dir {sample_data_dir} \\")
        print(f"    --cache_dir {sample_cache_dir} \\")
        print(f"    --tokenizer_path ./ckpt/TaDiCodec_Japanese_Full/text_tokenizer")
        print("\nPlease run this command manually to test the cache generation script.")
        print("\nAlternatively, you can run:")
        print(f"  PYTHONIOENCODING=utf-8 python scripts/create_dataset_cache.py \\")
        print(f"    --data_dir {sample_data_dir} \\")
        print(f"    --cache_dir {sample_cache_dir} \\")
        print(f"    --tokenizer_path ./ckpt/TaDiCodec_Japanese_Full/text_tokenizer")

        # ユーザーに実行を促す
        print("\n" + "="*80)
        print("After running the cache generation script, you can verify the cache files:")
        print(f"  python -c \"from scripts.test_cache_generation import verify_cache_files; verify_cache_files('{sample_cache_dir}')\"")
        print("="*80)

        # 一時ディレクトリのパスを表示
        print(f"\nSample data directory: {sample_data_dir}")
        print(f"Sample cache directory: {sample_cache_dir}")
        print(f"\nTo clean up, run: rm -rf {temp_dir}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
