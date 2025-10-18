#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音声コーデック・TTSシステムのビットレート比較視覚化
"""

import matplotlib.pyplot as plt
import numpy as np

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def create_comparison_charts():
    # データ
    systems = [
        'TaDiCodec\n(本リポジトリ)',
        'FocalCodec',
        'SemantiCodec',
        'LMCodec',
        'DualCodec',
        'BigCodec',
        'VALL-E\n(EnCodec)',
        'MaskGCT\n(SoundStream)',
    ]

    bitrates = [
        0.0875,   # TaDiCodec
        0.16,     # FocalCodec
        0.31,     # SemantiCodec
        0.50,     # LMCodec
        0.75,     # DualCodec
        1.04,     # BigCodec
        1.5,      # VALL-E
        6.0,      # MaskGCT
    ]

    # 色分け（TaDiCodecを強調）
    colors = ['#FF4444'] + ['#4444FF'] * (len(systems) - 1)

    # Figure 1: ビットレート比較（対数スケール）
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    bars = ax1.barh(systems, bitrates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # TaDiCodecのバーを特別に装飾
    bars[0].set_hatch('///')

    ax1.set_xlabel('ビットレート (kbps) - 対数スケール', fontsize=13, fontweight='bold')
    ax1.set_title('音声コーデック・TTSシステムのビットレート比較\n（低いほど効率的）',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, axis='x', which='both')

    # 値ラベルを追加
    for i, (system, bitrate) in enumerate(zip(systems, bitrates)):
        if i == 0:  # TaDiCodec
            label = f'{bitrate:.4f} kbps\n(87.5 bps)\n🏆 世界最小'
            ax1.text(bitrate, i, label, va='center', ha='left', fontsize=10,
                    fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        else:
            ratio = bitrate / 0.0875
            label = f'{bitrate:.2f} kbps\n({ratio:.1f}倍)'
            ax1.text(bitrate, i, label, va='center', ha='left', fontsize=9)

    plt.tight_layout()
    plt.savefig('bitrate_comparison_log.png', dpi=150, bbox_inches='tight')
    print("✅ ビットレート比較図（対数スケール）を保存しました: bitrate_comparison_log.png")

    # Figure 2: 100時間分のデータサイズ比較
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    # 100時間のデータサイズ（MB）
    hours_100_mb = [br * 60 * 60 * 100 / 8 / 1024 for br in bitrates]

    bars2 = ax2.barh(systems, hours_100_mb, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2[0].set_hatch('///')

    ax2.set_xlabel('データサイズ (MB)', fontsize=13, fontweight='bold')
    ax2.set_title('100時間の音声データをトークン化した場合のサイズ比較\n（小さいほどメモリ効率的）',
                  fontsize=16, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='x')

    # 値ラベル
    for i, (system, size) in enumerate(zip(systems, hours_100_mb)):
        if i == 0:
            label = f'{size:.2f} MB\n🎯 USBメモリ1つで十分'
            ax2.text(size, i, label, va='center', ha='left', fontsize=10,
                    fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        else:
            label = f'{size:.1f} MB'
            ax2.text(size, i, label, va='center', ha='left', fontsize=9)

    plt.tight_layout()
    plt.savefig('data_size_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ データサイズ比較図を保存しました: data_size_comparison.png")

    # Figure 3: トークン数比較（10秒の音声）
    fig3, ax3 = plt.subplots(figsize=(12, 7))

    # 10秒の音声のトークン数（推定）
    # TaDiCodec: 6.25 Hz → 62.5 トークン
    # VALL-E: 50 Hz (EnCodecは通常50Hz) → 500トークン
    # MaskGCT: 推定200 Hz → 2000トークン

    tokens_10s = [
        62,      # TaDiCodec
        100,     # FocalCodec (推定)
        194,     # SemantiCodec (推定)
        313,     # LMCodec (推定)
        469,     # DualCodec (推定)
        650,     # BigCodec (推定)
        1000,    # VALL-E (50Hz x 20 codes)
        4000,    # MaskGCT (推定200Hz x 20 codes)
    ]

    bars3 = ax3.barh(systems, tokens_10s, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3[0].set_hatch('///')

    ax3.set_xlabel('トークン数', fontsize=13, fontweight='bold')
    ax3.set_title('10秒の音声をトークン化した場合のトークン数\n（少ないほどLLMのコンテキストに優しい）',
                  fontsize=16, fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3, axis='x')

    # 値ラベル
    for i, (system, tokens) in enumerate(zip(systems, tokens_10s)):
        if i == 0:
            label = f'{tokens} トークン\n💡 GPT-4で超効率的'
            ax3.text(tokens, i, label, va='center', ha='left', fontsize=10,
                    fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        else:
            ratio = tokens / 62
            label = f'{tokens} トークン\n({ratio:.1f}倍)'
            ax3.text(tokens, i, label, va='center', ha='left', fontsize=9)

    plt.tight_layout()
    plt.savefig('token_count_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ トークン数比較図を保存しました: token_count_comparison.png")

    # Figure 4: 総合比較レーダーチャート
    fig4, ax4 = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # 評価軸（すべて「高いほど良い」にスケール変換）
    categories = ['効率性\n(低ビットレート)', '速度', '品質', '多言語対応', '柔軟性']

    # TaDiCodecの評価（主観的）
    tadicodec_scores = [
        10,  # 効率性: 世界最高
        8,   # 速度: MGM-TTSは速い、AR-TTSは中程度
        8,   # 品質: 高品質
        9,   # 多言語: クロスリンガル対応
        9,   # 柔軟性: 2種類のTTS、3種類のモデル
    ]

    # VALL-Eの評価（比較）
    valle_scores = [
        6,   # 効率性: TaDiCodecの1/17
        7,   # 速度: 自己回帰のみ
        9,   # 品質: 非常に高品質
        6,   # 多言語: 限定的
        5,   # 柔軟性: 1つのモデル
    ]

    # MaskGCTの評価（比較）
    maskgct_scores = [
        2,   # 効率性: TaDiCodecの1/69
        9,   # 速度: MGM方式で高速
        8,   # 品質: 高品質
        7,   # 多言語: 対応
        6,   # 柔軟性: MGMのみ
    ]

    # 角度計算
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    tadicodec_scores += tadicodec_scores[:1]
    valle_scores += valle_scores[:1]
    maskgct_scores += maskgct_scores[:1]
    angles += angles[:1]

    # プロット
    ax4.plot(angles, tadicodec_scores, 'o-', linewidth=2, label='TaDiCodec (本リポジトリ)', color='red')
    ax4.fill(angles, tadicodec_scores, alpha=0.25, color='red')

    ax4.plot(angles, valle_scores, 'o-', linewidth=2, label='VALL-E (Microsoft)', color='blue')
    ax4.fill(angles, valle_scores, alpha=0.15, color='blue')

    ax4.plot(angles, maskgct_scores, 'o-', linewidth=2, label='MaskGCT (Amphion)', color='green')
    ax4.fill(angles, maskgct_scores, alpha=0.15, color='green')

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, fontsize=11)
    ax4.set_ylim(0, 10)
    ax4.set_yticks([2, 4, 6, 8, 10])
    ax4.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9)
    ax4.grid(True)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax4.set_title('音声TTSシステム総合比較\n（外側ほど優れている）',
                  fontsize=14, fontweight='bold', pad=30)

    plt.tight_layout()
    plt.savefig('comprehensive_comparison_radar.png', dpi=150, bbox_inches='tight')
    print("✅ 総合比較レーダーチャートを保存しました: comprehensive_comparison_radar.png")

    print("\n" + "="*70)
    print("📊 すべての比較グラフを作成しました！")
    print("="*70)
    print("\n📁 生成されたファイル:")
    print("   - bitrate_comparison_log.png       : ビットレート比較（対数スケール）")
    print("   - data_size_comparison.png         : 100時間分のデータサイズ")
    print("   - token_count_comparison.png       : 10秒音声のトークン数")
    print("   - comprehensive_comparison_radar.png : 総合評価レーダーチャート")


if __name__ == "__main__":
    create_comparison_charts()
