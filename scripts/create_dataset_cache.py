#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TaDiCodec Dataset Cache Generator

Emilia形式の音声データセットから学習用キャッシュファイルを生成します。

使用方法:
    python scripts/create_dataset_cache.py \
      --data_dir ./data/japanese \
      --cache_dir ./cache/japanese \
      --tokenizer_path ./ckpt/TaDiCodec_Japanese_Full/text_tokenizer

生成されるキャッシュファイル:
    - wav_paths_cache.pkl: WAVファイルパスのリスト
    - duration_cache.pkl: 音声の長さ（秒）のリスト
    - bpe_token_count_cache.pkl: BPEトークン数のリスト
    - json_paths_cache.pkl: JSONメタデータのリスト（オプション）
"""

import os
import json
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import logging

# HuggingFace Transformers
try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers not installed. Please install it with: pip install transformers")
    exit(1)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def scan_data_directory(data_dir: str) -> List[Path]:
    """
    データディレクトリをスキャンして話者ディレクトリを取得

    Args:
        data_dir: Emilia形式データのルートディレクトリ

    Returns:
        話者ディレクトリのリスト
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # 話者ディレクトリを検出（audio.json が存在するディレクトリ）
    speaker_dirs = []
    for item in data_path.iterdir():
        if item.is_dir():
            audio_json = item / "audio.json"
            if audio_json.exists():
                speaker_dirs.append(item)

    if len(speaker_dirs) == 0:
        logger.warning(f"No speaker directories found in {data_dir}")
        logger.warning("Expected structure: data_dir/speaker1/audio.json, data_dir/speaker2/audio.json, ...")

    return sorted(speaker_dirs)


def load_audio_metadata(speaker_dir: Path) -> Dict:
    """
    話者ディレクトリからaudio.jsonを読み込む

    Args:
        speaker_dir: 話者ディレクトリのパス

    Returns:
        audio.jsonの内容（辞書）
    """
    audio_json_path = speaker_dir / "audio.json"

    try:
        with open(audio_json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        logger.error(f"Failed to load {audio_json_path}: {e}")
        return {}


def process_speaker_directory(
    speaker_dir: Path,
    data_dir: Path,
    min_duration: float,
    max_duration: float,
    save_json_cache: bool
) -> Tuple[List[str], List[float], List[str], List[Dict]]:
    """
    話者ディレクトリを処理してメタデータを収集

    Args:
        speaker_dir: 話者ディレクトリのパス
        data_dir: データディレクトリのルート
        min_duration: 最小音声長（秒）
        max_duration: 最大音声長（秒）
        save_json_cache: JSONメタデータもキャッシュするか

    Returns:
        (wav_paths, durations, texts, json_metadata) のタプル
    """
    # audio.json を読み込み
    audio_metadata = load_audio_metadata(speaker_dir)

    wav_paths = []
    durations = []
    texts = []
    json_metadata = []

    # 各音声ファイルを処理
    for index_str, meta in audio_metadata.items():
        # インデックスを整数に変換（辞書のキーは文字列または整数の可能性あり）
        if isinstance(index_str, int):
            index = index_str
        else:
            try:
                index = int(index_str)
            except ValueError:
                # "mls_english_opus" 形式の場合、ファイル名がキーになる
                # この場合、index_str がファイル名
                wav_filename = index_str
                wav_path = speaker_dir / wav_filename
                if not wav_path.exists():
                    logger.warning(f"WAV file not found: {wav_path}")
                    continue
                index = None  # インデックスなし

        # WAVファイルパスを構築
        if index is not None:
            wav_filename = f"audio_{index}.wav"
            wav_path = speaker_dir / wav_filename

        # WAVファイルの存在確認
        if not wav_path.exists():
            logger.warning(f"WAV file not found: {wav_path}")
            continue

        # メタデータの検証
        if not isinstance(meta, dict):
            logger.warning(f"Invalid metadata format for {wav_path}")
            continue

        text = meta.get("text", "")
        duration = meta.get("duration", 0.0)
        language = meta.get("language", "ja")

        # テキストが空でないことを確認
        if not text or len(text.strip()) == 0:
            logger.warning(f"Empty text for {wav_path}, skipping")
            continue

        # durationでフィルタリング
        if duration < min_duration or duration > max_duration:
            continue

        # 相対パスを計算（data_dirからの相対パス）
        relative_path = wav_path.relative_to(data_dir)

        # データを収集
        wav_paths.append(str(relative_path))
        durations.append(duration)
        texts.append(text)

        if save_json_cache:
            json_metadata.append({
                "text": text,
                "duration": duration,
                "language": language
            })

    return wav_paths, durations, texts, json_metadata


def tokenize_texts(texts: List[str], tokenizer) -> List[int]:
    """
    テキストをトークナイズしてトークン数を計算

    Args:
        texts: テキストのリスト
        tokenizer: HuggingFace Transformersトークナイザー

    Returns:
        各テキストのトークン数のリスト
    """
    token_counts = []

    logger.info(f"Tokenizing {len(texts)} texts...")
    for text in tqdm(texts, desc="Tokenizing"):
        try:
            # トークナイズ（special tokensなし）
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            token_counts.append(len(token_ids))
        except Exception as e:
            logger.warning(f"Failed to tokenize text: {text[:50]}... Error: {e}")
            token_counts.append(0)

    return token_counts


def save_cache_files(
    cache_dir: str,
    wav_paths: List[str],
    durations: List[float],
    token_counts: List[int],
    json_metadata: List[Dict] = None
):
    """
    キャッシュファイルを保存

    Args:
        cache_dir: キャッシュ保存先ディレクトリ
        wav_paths: WAVファイルパスのリスト
        durations: 音声の長さのリスト
        token_counts: トークン数のリスト
        json_metadata: JSONメタデータのリスト（オプション）
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving cache files to {cache_dir}...")

    # WAVパスキャッシュ
    wav_paths_cache = cache_path / "wav_paths_cache.pkl"
    with open(wav_paths_cache, 'wb') as f:
        pickle.dump(wav_paths, f)
    logger.info(f"✓ wav_paths_cache.pkl saved ({len(wav_paths)} items)")

    # durationキャッシュ
    duration_cache = cache_path / "duration_cache.pkl"
    with open(duration_cache, 'wb') as f:
        pickle.dump(durations, f)
    logger.info(f"✓ duration_cache.pkl saved ({len(durations)} items)")

    # トークン数キャッシュ
    token_count_cache = cache_path / "bpe_token_count_cache.pkl"
    with open(token_count_cache, 'wb') as f:
        pickle.dump(token_counts, f)
    logger.info(f"✓ bpe_token_count_cache.pkl saved ({len(token_counts)} items)")

    # JSONメタデータキャッシュ（オプション）
    if json_metadata is not None and len(json_metadata) > 0:
        json_paths_cache = cache_path / "json_paths_cache.pkl"
        with open(json_paths_cache, 'wb') as f:
            pickle.dump(json_metadata, f)
        logger.info(f"✓ json_paths_cache.pkl saved ({len(json_metadata)} items)")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="Generate cache files for TaDiCodec dataset training"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Emilia format data root directory"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Cache files output directory"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./ckpt/TaDiCodec_Japanese_Full/text_tokenizer",
        help="HuggingFace tokenizer path (default: ./ckpt/TaDiCodec_Japanese_Full/text_tokenizer)"
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=1.0,
        help="Minimum audio duration in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=40.0,
        help="Maximum audio duration in seconds (default: 40.0)"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=24000,
        help="Audio sample rate (default: 24000)"
    )
    parser.add_argument(
        "--save_json_cache",
        action="store_true",
        default=True,
        help="Save JSON metadata cache (default: True)"
    )
    parser.add_argument(
        "--no_json_cache",
        action="store_true",
        help="Do not save JSON metadata cache"
    )

    args = parser.parse_args()

    # --no_json_cache が指定された場合
    if args.no_json_cache:
        args.save_json_cache = False

    # データディレクトリのスキャン
    logger.info(f"Scanning data directory: {args.data_dir}")
    data_path = Path(args.data_dir)
    speaker_dirs = scan_data_directory(args.data_dir)
    logger.info(f"Found {len(speaker_dirs)} speaker directories")

    if len(speaker_dirs) == 0:
        logger.error("No speaker directories found. Exiting.")
        return

    # データの収集
    all_wav_paths = []
    all_durations = []
    all_texts = []
    all_json_metadata = [] if args.save_json_cache else None

    filtered_count = 0
    total_count = 0

    logger.info("Processing speaker directories...")
    for speaker_dir in tqdm(speaker_dirs, desc="Processing speakers"):
        wav_paths, durations, texts, json_metadata = process_speaker_directory(
            speaker_dir=speaker_dir,
            data_dir=data_path,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            save_json_cache=args.save_json_cache
        )

        # 統計情報の更新
        audio_metadata = load_audio_metadata(speaker_dir)
        total_count += len(audio_metadata)
        filtered_count += len(audio_metadata) - len(wav_paths)

        # データを追加
        all_wav_paths.extend(wav_paths)
        all_durations.extend(durations)
        all_texts.extend(texts)
        if args.save_json_cache:
            all_json_metadata.extend(json_metadata)

    logger.info(f"Total audio files found: {total_count}")
    logger.info(f"Filtered by duration: {filtered_count}")
    logger.info(f"Valid audio files: {len(all_wav_paths)}")

    if len(all_wav_paths) == 0:
        logger.error("No valid audio files found. Exiting.")
        return

    # トークナイザーのロード
    logger.info(f"Loading tokenizer from {args.tokenizer_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        logger.info("✓ Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        logger.error("Please ensure the tokenizer path is correct and the tokenizer is properly initialized.")
        return

    # トークン数の計算
    token_counts = tokenize_texts(all_texts, tokenizer)

    # キャッシュファイルの保存
    save_cache_files(
        cache_dir=args.cache_dir,
        wav_paths=all_wav_paths,
        durations=all_durations,
        token_counts=token_counts,
        json_metadata=all_json_metadata if args.save_json_cache else None
    )

    # 統計情報の表示
    logger.info("\n" + "="*80)
    logger.info("Summary:")
    logger.info(f"  Total audio files processed: {len(all_wav_paths)}")
    logger.info(f"  Filtered by duration: {filtered_count}")
    logger.info(f"  Average duration: {sum(all_durations) / len(all_durations):.2f} seconds")
    logger.info(f"  Average token count: {sum(token_counts) / len(token_counts):.2f} tokens")
    logger.info(f"  Total duration: {sum(all_durations) / 3600:.2f} hours")
    logger.info(f"  Min duration: {min(all_durations):.2f} seconds")
    logger.info(f"  Max duration: {max(all_durations):.2f} seconds")
    logger.info(f"  Min token count: {min(token_counts)}")
    logger.info(f"  Max token count: {max(token_counts)}")
    logger.info("="*80)

    logger.info("\nCache generation completed successfully!")
    logger.info(f"Cache files saved to: {args.cache_dir}")
    logger.info("\nYou can now use these cache files for training with:")
    logger.info(f'  "mnt_path": "{args.data_dir}",')
    logger.info(f'  "cache_folder": "{args.cache_dir}",')
    logger.info('  "use_json_path_cache": true,')


if __name__ == "__main__":
    main()
