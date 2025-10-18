#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音声再構成の視覚化 - 元の音声と再構成音声の波形比較
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os

# 日本語フォントの設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def visualize_comparison():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ファイルパス
    original_path = os.path.join(script_dir, "test_audio", "trump_0.wav")
    reconstructed_path = os.path.join(script_dir, "reconstruction_test.wav")

    # 音声読み込み
    original_audio, sr_orig = sf.read(original_path)
    reconstructed_audio, sr_rec = sf.read(reconstructed_path)

    # ステレオの場合はモノラルに変換
    if original_audio.ndim > 1:
        original_audio = original_audio.mean(axis=1)

    # リサンプリングして同じサンプリングレートに
    if sr_orig != sr_rec:
        import librosa
        original_audio = librosa.resample(original_audio, orig_sr=sr_orig, target_sr=24000)
        sr_orig = 24000

    # 長さを揃える（短い方に合わせる）
    min_len = min(len(original_audio), len(reconstructed_audio))
    original_audio = original_audio[:min_len]
    reconstructed_audio = reconstructed_audio[:min_len]

    # 時間軸
    time = np.arange(min_len) / sr_rec

    # プロット作成
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 元の音声波形
    axes[0].plot(time, original_audio, linewidth=0.5, color='blue', alpha=0.7)
    axes[0].set_title('元の音声波形 (Original Audio)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('振幅', fontsize=12)
    axes[0].set_xlim([0, time[-1]])
    axes[0].grid(True, alpha=0.3)

    # 再構成音声波形
    axes[1].plot(time, reconstructed_audio, linewidth=0.5, color='red', alpha=0.7)
    axes[1].set_title('再構成音声波形 (Reconstructed Audio) - わずか84バイトから生成', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('振幅', fontsize=12)
    axes[1].set_xlim([0, time[-1]])
    axes[1].grid(True, alpha=0.3)

    # 差分
    diff = original_audio - reconstructed_audio
    axes[2].plot(time, diff, linewidth=0.5, color='purple', alpha=0.7)
    axes[2].set_title('差分 (Difference) - 失われた情報', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('時間 (秒)', fontsize=12)
    axes[2].set_ylabel('振幅', fontsize=12)
    axes[2].set_xlim([0, time[-1]])
    axes[2].grid(True, alpha=0.3)

    # 統計情報を追加
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    correlation = np.corrcoef(original_audio, reconstructed_audio)[0, 1]

    info_text = f"""
    統計情報:
    - MSE (平均二乗誤差): {mse:.6f}
    - RMSE (二乗平均平方根誤差): {rmse:.6f}
    - 相関係数: {correlation:.4f}

    ※ 相関係数1.0 = 完全一致、0.0 = 無関係
    """

    fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 0.98])

    # 保存
    output_path = os.path.join(script_dir, "waveform_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 波形比較画像を保存しました: {output_path}")

    # スペクトログラム比較も作成
    create_spectrogram_comparison(
        original_audio, reconstructed_audio, sr_rec, script_dir
    )


def create_spectrogram_comparison(original, reconstructed, sr, script_dir):
    """スペクトログラム比較を作成"""
    import librosa.display

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 元の音声のスペクトログラム
    D_orig = librosa.stft(original)
    S_orig_db = librosa.amplitude_to_db(np.abs(D_orig), ref=np.max)

    img1 = librosa.display.specshow(
        S_orig_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[0], cmap='viridis'
    )
    axes[0].set_title('元の音声のスペクトログラム', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('周波数 (Hz)', fontsize=10)
    axes[0].set_xlabel('時間 (秒)', fontsize=10)
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')

    # 再構成音声のスペクトログラム
    D_rec = librosa.stft(reconstructed)
    S_rec_db = librosa.amplitude_to_db(np.abs(D_rec), ref=np.max)

    img2 = librosa.display.specshow(
        S_rec_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[1], cmap='viridis'
    )
    axes[1].set_title('再構成音声のスペクトログラム (84バイトから生成)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('周波数 (Hz)', fontsize=10)
    axes[1].set_xlabel('時間 (秒)', fontsize=10)
    fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')

    plt.suptitle('スペクトログラム比較 - 周波数成分の違い', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 保存
    output_path = os.path.join(script_dir, "spectrogram_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ スペクトログラム比較画像を保存しました: {output_path}")

    print("\n" + "="*70)
    print("📊 視覚化完了")
    print("="*70)
    print(f"\n📁 生成されたファイル:")
    print(f"   - {os.path.join(script_dir, 'waveform_comparison.png')}")
    print(f"   - {os.path.join(script_dir, 'spectrogram_comparison.png')}")
    print("\n👀 画像を開いて、元の音声と再構成音声の違いを確認してください！")


if __name__ == "__main__":
    visualize_comparison()
