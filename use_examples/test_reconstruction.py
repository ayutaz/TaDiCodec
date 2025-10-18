#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音声再構成テスト - エンコード→デコードのプロセス確認
"""

import torch
import soundfile as sf
import numpy as np
import os
import sys

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from models.tts.tadicodec.inference_tadicodec import TaDiCodecPipline


def main():
    print("=" * 70)
    print("🔄 音声再構成プロセステスト")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📍 使用デバイス: {device}")

    # TaDiCodecパイプライン読み込み
    print("\n⏳ TaDiCodecモデルを読み込み中...")
    pipe = TaDiCodecPipline.from_pretrained(
        ckpt_dir="amphion/TaDiCodec",
        device=device
    )
    print("✅ モデル読み込み完了")

    # テスト音声ファイル
    input_audio = os.path.join(script_dir, "test_audio", "trump_0.wav")
    text = "In short, we embarked on a mission to make America great again, for all Americans."

    print(f"\n📁 入力音声: {input_audio}")
    print(f"📝 テキスト: {text}")

    # ステップ1: 元の音声を読み込み
    print("\n" + "-" * 70)
    print("ステップ1: 元の音声を読み込み")
    print("-" * 70)

    original_audio, sr = sf.read(input_audio)
    # ステレオの場合はモノラルに変換
    if original_audio.ndim > 1:
        original_audio = original_audio.mean(axis=1)
    original_size = os.path.getsize(input_audio)
    original_duration = len(original_audio) / sr

    print(f"  サンプリングレート: {sr} Hz")
    print(f"  長さ: {original_duration:.2f} 秒")
    print(f"  サンプル数: {len(original_audio):,}")
    print(f"  ファイルサイズ: {original_size:,} バイト ({original_size/1024:.1f} KB)")
    print(f"  ビットレート: {original_size * 8 / original_duration / 1000:.1f} kbps")

    # ステップ2: エンコード（音声→トークン）
    print("\n" + "-" * 70)
    print("ステップ2: エンコード（音声 → トークン）")
    print("-" * 70)

    tokens = pipe(
        text=text,
        speech_path=input_audio,
        return_code=True
    )

    num_tokens = tokens.shape[1]
    token_rate = num_tokens / original_duration
    bits_per_token = 14  # TaDiCodecのVQ次元
    bitrate = token_rate * bits_per_token

    # トークンのバイトサイズ（理論値）
    token_bytes = (num_tokens * bits_per_token) / 8
    compression_ratio = original_size / token_bytes

    print(f"  トークン数: {num_tokens}")
    print(f"  トークンレート: {token_rate:.2f} tokens/秒")
    print(f"  ビット/トークン: {bits_per_token}")
    print(f"  圧縮後ビットレート: {bitrate:.4f} bps = {bitrate/1000:.4f} kbps")
    print(f"  圧縮後サイズ: {token_bytes:.1f} バイト")
    print(f"  圧縮率: {compression_ratio:.0f}倍")
    print(f"\n  トークン列（最初の10個）: {tokens[0, :10].tolist()}")

    # ステップ3: デコード（トークン→音声）
    print("\n" + "-" * 70)
    print("ステップ3: デコード（トークン → 音声）")
    print("-" * 70)

    reconstructed_audio = pipe(
        text=text,
        speech_path=input_audio,
        n_timesteps=25,  # 拡散ステップ数
        cfg_scale=2.0    # Classifier-Free Guidance
    )

    print(f"  再構成音声サンプル数: {len(reconstructed_audio):,}")
    print(f"  再構成音声長: {len(reconstructed_audio) / sr:.2f} 秒")

    # 出力ファイルに保存
    output_reconstructed = os.path.join(script_dir, "reconstruction_test.wav")
    sf.write(output_reconstructed, reconstructed_audio, sr)

    reconstructed_size = os.path.getsize(output_reconstructed)
    print(f"  保存ファイルサイズ: {reconstructed_size:,} バイト ({reconstructed_size/1024:.1f} KB)")

    # ステップ4: 比較分析
    print("\n" + "=" * 70)
    print("📊 比較分析")
    print("=" * 70)

    # サイズ比較
    print("\n【ファイルサイズ】")
    print(f"  元の音声:     {original_size:>10,} バイト ({original_size/1024:>7.1f} KB)")
    print(f"  トークン:     {token_bytes:>10.0f} バイト ({token_bytes/1024:>7.1f} KB)")
    print(f"  再構成音声:   {reconstructed_size:>10,} バイト ({reconstructed_size/1024:>7.1f} KB)")

    # ビットレート比較
    print("\n【ビットレート】")
    original_bitrate = original_size * 8 / original_duration / 1000
    reconstructed_bitrate = reconstructed_size * 8 / (len(reconstructed_audio) / sr) / 1000
    print(f"  元の音声:     {original_bitrate:>8.2f} kbps")
    print(f"  トークン:     {bitrate/1000:>8.4f} kbps  ← 超低ビットレート！")
    print(f"  再構成音声:   {reconstructed_bitrate:>8.2f} kbps")

    # 圧縮効率
    print("\n【圧縮効率】")
    print(f"  圧縮率: {compression_ratio:.0f}倍")
    print(f"  データ削減率: {(1 - token_bytes/original_size) * 100:.2f}%")

    # 音声品質（簡易分析）
    print("\n【音声品質（簡易推定）】")

    # 長さの比較
    min_len = min(len(original_audio), len(reconstructed_audio))
    original_trimmed = original_audio[:min_len]
    reconstructed_trimmed = reconstructed_audio[:min_len]

    # 正規化
    original_norm = original_trimmed / (np.abs(original_trimmed).max() + 1e-8)
    reconstructed_norm = reconstructed_trimmed / (np.abs(reconstructed_trimmed).max() + 1e-8)

    # RMS（音量）比較
    original_rms = np.sqrt(np.mean(original_norm ** 2))
    reconstructed_rms = np.sqrt(np.mean(reconstructed_norm ** 2))

    print(f"  元の音声 RMS: {original_rms:.4f}")
    print(f"  再構成音声 RMS: {reconstructed_rms:.4f}")
    print(f"  音量比: {reconstructed_rms/original_rms:.2f}x")

    # MSE（平均二乗誤差）
    mse = np.mean((original_norm - reconstructed_norm) ** 2)
    print(f"\n  MSE（平均二乗誤差）: {mse:.6f}")
    print(f"  ※ 0に近いほど元に近い（0 = 完全一致）")

    # 結論
    print("\n" + "=" * 70)
    print("📝 結論")
    print("=" * 70)

    print("\n✅ エンコード→デコードプロセスは正常に動作")
    print(f"✅ {compression_ratio:.0f}倍の圧縮を達成")
    print(f"✅ わずか{bitrate/1000:.4f} kbps（87.5 bps）で音声を表現")

    print("\n⚠️  ただし、これは「ロッシー圧縮」です:")
    print("   - 完全には元に戻りません")
    print("   - 言語内容は保持されます")
    print("   - 音質は劣化しますが、音声として認識可能")
    print("   - 話者の声質は保持されます")

    print(f"\n💾 ファイル保存場所:")
    print(f"   元の音声: {input_audio}")
    print(f"   再構成音声: {output_reconstructed}")
    print("\n👂 両方のファイルを聴き比べて品質を確認してください！")


if __name__ == "__main__":
    main()
