#!/usr/bin/env python3
"""
メルスペクトログラム事前計算スクリプト

全音声ファイルのメルスペクトログラムを事前に計算して保存します。
これにより学習時のI/Oボトルネックを解消し、学習速度を2-3倍高速化します。

機能:
- 音声ファイルからメルスペクトログラムを計算
- データ拡張バージョンも生成（オプション）
- 進捗表示とエラーハンドリング

使用方法:
    python scripts/precompute_mel_features.py \\
        --config egs/tts/TaDiCodec/tadicodec_japanese_finetune.json \\
        --output_dir ./cache/jvs_emilia/mel

予想時間: 30-60分（14,979サンプル）
必要ディスク容量: 30-50GB
"""

import argparse
import os
import sys
import json
import pickle
import numpy as np
import librosa
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

# プロジェクトルートをパスに追加
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.codec.melvqgan.melspec import MelSpectrogram
from models.tts.tadicodec.japanese_data_augmentation import JapaneseDataAugmentation
from utils.util import load_config


def compute_mel_spectrogram(
    audio: np.ndarray,
    mel_model: MelSpectrogram,
    mel_mean: float = 0.0,
    mel_var: float = 1.0,
) -> np.ndarray:
    """
    音声からメルスペクトログラムを計算

    Args:
        audio: 音声波形 (numpy array)
        mel_model: メルスペクトログラム変換モデル
        mel_mean: メルスペクトログラムの平均（正規化用）
        mel_var: メルスペクトログラムの分散（正規化用）

    Returns:
        mel: メルスペクトログラム [T, d]
    """
    # NumPy → Tensor
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()

    # バッチ次元を追加
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # [1, T]

    # メルスペクトログラムを計算
    with torch.no_grad():
        mel = mel_model(audio)  # [1, d, T]

    # 形状を変換: [1, d, T] → [T, d]
    mel = mel.squeeze(0).transpose(0, 1)

    # 正規化
    mel = (mel - mel_mean) / np.sqrt(mel_var)

    # Tensor → NumPy
    mel = mel.cpu().numpy()

    return mel


def precompute_mel_features(args):
    """
    メルスペクトログラムの事前計算メイン関数
    """
    print("="*80)
    print("メルスペクトログラム事前計算")
    print("="*80)

    # 設定ファイルのロード
    print(f"\n設定ファイル読み込み: {args.config}")
    cfg = load_config(args.config)

    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"出力ディレクトリ: {output_dir}")

    # キャッシュファイルのロード
    cache_folder = Path(cfg.preprocess.cache_folder)
    wav_paths_cache = cache_folder / "wav_paths_cache.pkl"
    json_paths_cache = cache_folder / "json_paths_cache.pkl"

    if not wav_paths_cache.exists():
        print(f"エラー: {wav_paths_cache} が見つかりません")
        print("先に scripts/create_dataset_cache.py を実行してください")
        sys.exit(1)

    print(f"\nキャッシュ読み込み: {wav_paths_cache}")
    with open(wav_paths_cache, "rb") as f:
        wav_paths = pickle.load(f)

    if cfg.preprocess.use_json_path_cache and json_paths_cache.exists():
        print(f"キャッシュ読み込み: {json_paths_cache}")
        with open(json_paths_cache, "rb") as f:
            json_paths = pickle.load(f)
    else:
        json_paths = None

    print(f"音声ファイル数: {len(wav_paths)}")

    # メルスペクトログラムモデルの初期化
    print("\nメルスペクトログラムモデル初期化...")
    mel_model = MelSpectrogram(
        sampling_rate=cfg.preprocess.sample_rate,
        n_fft=cfg.preprocess.n_fft,
        num_mels=cfg.preprocess.num_mels,
        hop_size=cfg.preprocess.hop_size,
        win_size=cfg.preprocess.win_size,
        fmin=cfg.preprocess.fmin,
        fmax=cfg.preprocess.fmax,
    )
    mel_model.eval()

    # データ拡張の初期化（オプション）
    if args.enable_augmentation:
        print("\nデータ拡張を有効化")
        augmentor = JapaneseDataAugmentation(
            use_pitch_shift=getattr(cfg.preprocess, "use_pitch_shift", True),
            use_speed_perturb=getattr(cfg.preprocess, "use_speed_perturb", True),
            use_accent_augment=False,  # メル計算前なのでアクセント拡張は無効
            use_code_switch=False,  # メル計算前なのでコードスイッチは無効
            pitch_shift_range=getattr(cfg.preprocess, "pitch_shift_range", [-4.0, 4.0]),
            speed_perturb_range=getattr(cfg.preprocess, "speed_perturb_range", [0.9, 1.1]),
            augment_prob=args.augment_prob,
        )
    else:
        augmentor = None

    # メルスペクトログラムを計算して保存
    print("\nメルスペクトログラム計算開始...")
    print(f"進捗: 0/{len(wav_paths)}")

    success_count = 0
    error_count = 0
    mnt_path = Path(cfg.preprocess.mnt_path)

    for idx, wav_path in enumerate(tqdm(wav_paths, desc="計算中")):
        try:
            # 音声ファイルパスの構築
            full_wav_path = mnt_path / wav_path.replace("_new", "")

            if not full_wav_path.exists():
                print(f"\n警告: ファイルが見つかりません: {full_wav_path}")
                error_count += 1
                continue

            # 音声の読み込み
            audio, sr = librosa.load(full_wav_path, sr=cfg.preprocess.sample_rate)

            # データ拡張（オプション）
            if augmentor is not None:
                # メタデータからテキストと言語を取得
                try:
                    # wav_path形式: "JA_jvs001/audio_0.wav"
                    # json_path形式: "JA_jvs001/audio.json"
                    speaker_dir = os.path.dirname(wav_path)
                    json_path = os.path.join(speaker_dir, "audio.json")
                    meta_path = mnt_path / json_path

                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta_dict = json.load(f)

                    audio_name = os.path.splitext(os.path.basename(wav_path))[0]
                    audio_id = audio_name.replace("audio_", "")

                    if audio_id in meta_dict:
                        text = meta_dict[audio_id].get("text", "")
                        language = meta_dict[audio_id].get("language", "ja")
                    else:
                        text = ""
                        language = "ja"
                except Exception as e:
                    # メタデータ読み込み失敗時はデフォルト値
                    text = ""
                    language = "ja"

                # 拡張を適用
                audio, _ = augmentor.augment(audio, text, sr=sr, language=language)

            # メルスペクトログラムを計算
            mel = compute_mel_spectrogram(
                audio,
                mel_model,
                mel_mean=cfg.preprocess.mel_mean,
                mel_var=cfg.preprocess.mel_var,
            )

            # 保存パスの構築
            # wav_path形式: "JA_jvs001/audio_0.wav"
            speaker_id = os.path.dirname(wav_path)
            audio_name = os.path.splitext(os.path.basename(wav_path))[0]  # "audio_0"

            speaker_output_dir = output_dir / speaker_id
            speaker_output_dir.mkdir(parents=True, exist_ok=True)

            output_path = speaker_output_dir / f"{audio_name}.npy"

            # メルスペクトログラムを保存
            np.save(output_path, mel.astype(np.float32))

            success_count += 1

        except Exception as e:
            print(f"\nエラー (idx={idx}, path={wav_path}): {e}")
            error_count += 1
            continue

    # 統計情報を保存
    stats = {
        "total_files": len(wav_paths),
        "success_count": success_count,
        "error_count": error_count,
        "config_path": args.config,
        "output_dir": str(output_dir),
        "augmentation_enabled": args.enable_augmentation,
        "augment_prob": args.augment_prob if args.enable_augmentation else 0.0,
    }

    stats_path = output_dir / "precompute_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("メルスペクトログラム事前計算完了")
    print("="*80)
    print(f"成功: {success_count}/{len(wav_paths)}")
    print(f"エラー: {error_count}/{len(wav_paths)}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"統計情報: {stats_path}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="メルスペクトログラム事前計算")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="設定ファイルパス (例: egs/tts/TaDiCodec/tadicodec_japanese_finetune.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./cache/jvs_emilia/mel",
        help="メルスペクトログラム出力ディレクトリ",
    )
    parser.add_argument(
        "--enable_augmentation",
        action="store_true",
        help="データ拡張を有効化（ピッチシフト、速度変化）",
    )
    parser.add_argument(
        "--augment_prob",
        type=float,
        default=0.5,
        help="データ拡張の適用確率（0.0-1.0）",
    )

    args = parser.parse_args()

    precompute_mel_features(args)


if __name__ == "__main__":
    main()
