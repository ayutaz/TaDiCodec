#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
トークナイザー統合テスト

TaDiCodecパイプラインでトークナイザーがどのように使われているか確認
"""

import sys
import os

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from transformers import AutoTokenizer


def test_original_tokenizer():
    """オリジナルのTaDiCodecトークナイザーをテスト"""
    print("="*80)
    print("【オリジナルTaDiCodecトークナイザーのテスト】")
    print("="*80)

    tokenizer_path = os.path.join(parent_dir, "ckpt", "TaDiCodec", "text_tokenizer")

    if not os.path.exists(tokenizer_path):
        print(f"❌ トークナイザーが見つかりません: {tokenizer_path}")
        print("先にTaDiCodecモデルをダウンロードしてください。")
        return

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print(f"\n語彙サイズ: {len(tokenizer)}")
    print(f"トークナイザータイプ: {type(tokenizer).__name__}")

    # テストテキスト
    test_texts = [
        "Hello, world!",
        "東京の天気は晴れです。",
        "t o o ky o o",  # 音素列
        "k[POS_NOUN]",   # リッチトークン
    ]

    for text in test_texts:
        print(f"\n{'='*80}")
        print(f"テキスト: '{text}'")
        print("-"*80)

        # トークナイズ
        tokens = tokenizer.tokenize(text)
        print(f"トークン: {tokens[:20]}")

        # トークンID
        token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        print(f"トークンID: {token_ids[0][:20].tolist()}")

        # デコード
        decoded = tokenizer.decode(token_ids[0])
        print(f"デコード: '{decoded}'")


def test_extended_tokenizer():
    """拡張版トークナイザーをテスト"""
    print("\n\n" + "="*80)
    print("【完全版日本語トークナイザーのテスト】")
    print("="*80)

    tokenizer_path = os.path.join(parent_dir, "ckpt", "TaDiCodec_Japanese_Full", "text_tokenizer")

    if not os.path.exists(tokenizer_path):
        print(f"❌ トークナイザーが見つかりません: {tokenizer_path}")
        print("先に create_japanese_tokenizer_full.py を実行してください。")
        return

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print(f"\n語彙サイズ: {len(tokenizer)}")
    print(f"トークナイザータイプ: {type(tokenizer).__name__}")

    # テストテキスト
    test_texts = [
        "Hello, world!",
        "東京の天気は晴れです。",
        "t o o ky o o",  # 音素列
        "t[POS_OTHER][ACC_TYPE_5][TONE_0]",  # リッチトークン
    ]

    for text in test_texts:
        print(f"\n{'='*80}")
        print(f"テキスト: '{text}'")
        print("-"*80)

        # トークナイズ
        tokens = tokenizer.tokenize(text)
        print(f"トークン: {tokens[:20]}")

        # トークンID
        token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        print(f"トークンID: {token_ids[0][:20].tolist()}")

        # デコード
        decoded = tokenizer.decode(token_ids[0])
        print(f"デコード: '{decoded}'")


def test_phoneme_conversion():
    """音素変換の必要性を確認"""
    print("\n\n" + "="*80)
    print("【問題点: テキスト→音素変換が欠落】")
    print("="*80)

    print("""
TaDiCodecパイプラインの現在の動作:
1. ユーザーが日本語テキストを入力: "東京の天気は晴れです。"
2. tokenize_text() がそのままトークナイザーに渡す
3. トークナイザーが文字レベルでトークン化
   → ['▁', '東', '京', 'の', '天', '気', 'は', '晴', 'れ', 'です', '。']
4. これがtext_embeddingに入力される

問題点:
❌ 音素への変換が行われていない
❌ OpenJTalk韻律情報が活用されていない
❌ 拡張したトークン（[POS_NOUN]など）が使われていない

必要な処理:
1. 日本語テキスト → pyopenjtalk → 音素列+韻律情報
2. 音素列+韻律情報 → リッチトークン列
3. リッチトークン列 → トークナイザー → トークンID
4. トークンID → text_embedding

つまり、現在の実装では:
- ❌ トークナイザーの語彙を拡張しただけ
- ❌ テキスト→音素変換のロジックがパイプラインに統合されていない
- ❌ 完全版トークナイザーが実際には使われない

解決策:
1. TaDiCodecPiplineのtokenize_textメソッドをオーバーライド
2. テキスト→音素変換→リッチトークン→トークンIDの処理を追加
3. または、カスタムトークナイザークラスを作成
    """)


def main():
    print("\n" + "🔍 " * 40)
    print("トークナイザー統合テスト")
    print("🔍 " * 40)

    # テスト実行
    test_original_tokenizer()
    test_extended_tokenizer()
    test_phoneme_conversion()

    print("\n\n" + "="*80)
    print("【結論】")
    print("="*80)
    print("""
現在の実装状態:
✅ 完全版トークナイザーの語彙は作成済み（33,844トークン）
✅ OpenJTalk情報の100%抽出ロジックは実装済み
✅ リッチトークン生成ロジックは実装済み

❌ 未完成部分:
1. TaDiCodecパイプラインへの統合
2. テキスト→音素変換の自動化
3. カスタムトークナイザークラスの実装

次に必要な作業:
1. テキスト→音素変換機能を持つカスタムトークナイザーを作成
2. TaDiCodecPiplineに統合
3. 推論時の動作テスト
    """)


if __name__ == "__main__":
    main()
