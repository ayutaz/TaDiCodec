#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyopenjtalk-plus の機能調査スクリプト
OpenJTalkから取得できるすべてのラベル情報を確認
"""

import sys
import os

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

try:
    import pyopenjtalk
    print("✅ pyopenjtalk imported successfully")
    print(f"Version: {pyopenjtalk.__version__ if hasattr(pyopenjtalk, '__version__') else 'unknown'}")
    print()
except ImportError as e:
    print(f"❌ Failed to import pyopenjtalk: {e}")
    print("Please install: pip install pyopenjtalk-plus")
    sys.exit(1)

# テスト文
test_sentences = [
    "東京の天気は晴れです。",
    "こんにちは、元気ですか？",
    "コンピューター",
    "新聞を読む",
]

print("="*70)
print("pyopenjtalk 機能テスト")
print("="*70)

for i, text in enumerate(test_sentences, 1):
    print(f"\n【テスト {i}】 テキスト: {text}")
    print("-"*70)

    # 1. 基本的な音素抽出
    print("\n1. g2p() - 音素列の取得:")
    try:
        phonemes = pyopenjtalk.g2p(text)
        print(f"   音素列: {phonemes}")
    except Exception as e:
        print(f"   エラー: {e}")

    # 2. 音素とカナの分離
    print("\n2. g2p(..., kana=True) - カナと音素の分離:")
    try:
        result = pyopenjtalk.g2p(text, kana=True)
        print(f"   結果: {result}")
    except Exception as e:
        print(f"   エラー: {e}")

    # 3. フルコンテキストラベルの取得（最重要）
    print("\n3. extract_fullcontext() - フルコンテキストラベル:")
    try:
        labels = pyopenjtalk.extract_fullcontext(text)
        print(f"   ラベル数: {len(labels)}")
        if len(labels) > 0:
            print(f"   最初のラベル例:")
            print(f"   {labels[0]}")
            if len(labels) > 1:
                print(f"   2番目のラベル例:")
                print(f"   {labels[1]}")
    except Exception as e:
        print(f"   エラー: {e}")

    # 4. 音素とアクセントの取得
    print("\n4. run_frontend() - 詳細な音素情報:")
    try:
        features = pyopenjtalk.run_frontend(text)
        print(f"   フィーチャー数: {len(features)}")
        if len(features) > 0:
            print(f"   最初のフィーチャー:")
            for key, value in features[0].items():
                print(f"     {key}: {value}")
    except Exception as e:
        print(f"   エラー: {e}")

print("\n" + "="*70)
print("利用可能な関数一覧:")
print("="*70)
functions = [f for f in dir(pyopenjtalk) if not f.startswith('_')]
for func in functions:
    print(f"  - {func}")

print("\n" + "="*70)
print("フルコンテキストラベルの詳細解析")
print("="*70)

# フルコンテキストラベルの詳細解析
text = "東京の天気は晴れです。"
labels = pyopenjtalk.extract_fullcontext(text)

print(f"\nテキスト: {text}")
print(f"ラベル数: {len(labels)}")
print("\n各ラベルの詳細:")

for i, label in enumerate(labels[:5]):  # 最初の5つだけ表示
    print(f"\n--- ラベル {i+1} ---")
    print(label)

    # ラベルのパース（OpenJTalkのフルコンテキストラベル形式）
    # 形式: p1^p2-p3+p4=p5/A:a1+a2+a3/B:...
    parts = label.split('/')
    if len(parts) > 0:
        # 音素部分
        phoneme_part = parts[0]
        print(f"  音素部分: {phoneme_part}")

        # その他の情報
        for j, part in enumerate(parts[1:], 1):
            if ':' in part:
                category, values = part.split(':', 1)
                print(f"  {category}: {values}")

print("\n" + "="*70)
print("まとめ")
print("="*70)
print("""
pyopenjtalk-plus で取得できる情報:

1. g2p(): 基本的な音素列
   例: "k o N n i ch i w a"

2. extract_fullcontext(): フルコンテキストラベル
   - 音素情報
   - アクセント型
   - モーラ位置
   - 品詞情報
   - イントネーション句境界
   - ピッチアクセント

3. run_frontend(): 詳細な音素フィーチャー
   - 各音素の詳細情報
   - アクセント位置
   - 韻律情報

推奨: extract_fullcontext() を使用して、
      すべての韻律・アクセント情報を取得
""")
