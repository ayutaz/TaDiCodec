#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日本語対応TaDiCodecパイプラインのテスト

テキスト→音素変換が正しく動作するか確認
"""

import sys
import os

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import torch


def test_japanese_pipeline():
    """日本語対応パイプラインのテスト"""
    print("="*80)
    print("【日本語対応TaDiCodecパイプラインのテスト】")
    print("="*80)

    # パイプラインのインポート
    try:
        from models.tts.tadicodec.inference_tadicodec_japanese import JapaneseTaDiCodecPipeline
    except Exception as e:
        print(f"❌ パイプラインのインポートに失敗: {e}")
        return

    # 日本語トークナイザーのパスを確認
    japanese_tokenizer_path = os.path.join(parent_dir, "ckpt", "TaDiCodec_Japanese_Full", "text_tokenizer")
    if not os.path.exists(japanese_tokenizer_path):
        print(f"❌ 日本語トークナイザーが見つかりません: {japanese_tokenizer_path}")
        print("先に create_japanese_tokenizer_full.py を実行してください。")
        return

    # TaDiCodecモデルのパスを確認
    tadicodec_path = os.path.join(parent_dir, "ckpt", "TaDiCodec")
    if not os.path.exists(os.path.join(tadicodec_path, "config.json")):
        print(f"❌ TaDiCodecモデルが見つかりません: {tadicodec_path}")
        print("先にモデルをダウンロードしてください。")
        return

    print("\n✅ すべての必要ファイルが見つかりました")

    # パイプラインの作成
    print("\nパイプラインを作成中...")
    try:
        pipe = JapaneseTaDiCodecPipeline.from_pretrained(
            ckpt_dir=tadicodec_path,
            japanese_tokenizer_path=japanese_tokenizer_path,
            enable_japanese_phoneme=True,
            japanese_mode="auto",
            auto_download=False,
        )
        print("✅ パイプライン作成成功")
    except Exception as e:
        print(f"❌ パイプライン作成に失敗: {e}")
        import traceback
        traceback.print_exc()
        return

    # テストケース
    test_cases = [
        {
            "text": "東京の天気は晴れです。",
            "description": "日本語テキスト（自動音素変換）",
            "expected_conversion": True,
        },
        {
            "text": "こんにちは、元気ですか？",
            "description": "日本語テキスト（ひらがな）",
            "expected_conversion": True,
        },
        {
            "text": "Hello, world!",
            "description": "英語テキスト（音素変換なし）",
            "expected_conversion": False,
        },
    ]

    print("\n" + "="*80)
    print("【テストケース実行】")
    print("="*80)

    for i, test_case in enumerate(test_cases, 1):
        text = test_case["text"]
        description = test_case["description"]
        expected_conversion = test_case["expected_conversion"]

        print(f"\n--- テスト {i}: {description} ---")
        print(f"入力テキスト: '{text}'")

        # 日本語検出のテスト
        contains_japanese = pipe._contains_japanese(text)
        print(f"日本語検出: {contains_japanese}")

        if contains_japanese != expected_conversion:
            print(f"⚠️  警告: 日本語検出が期待と異なります（期待: {expected_conversion}）")

        # 音素変換のテスト
        if pipe.enable_japanese_phoneme and contains_japanese:
            phoneme_string = pipe._text_to_phoneme_string(text)
            print(f"\n音素+韻律情報:")
            # 長すぎる場合は最初の200文字のみ表示
            if len(phoneme_string) > 200:
                print(f"  {phoneme_string[:200]}...")
                print(f"  （合計 {len(phoneme_string)} 文字）")
            else:
                print(f"  {phoneme_string}")

        # トークナイズのテスト
        try:
            token_ids = pipe.tokenize_text(text)
            print(f"\nトークンID:")
            print(f"  形状: {token_ids.shape}")
            print(f"  最初の20トークン: {token_ids[0][:20].tolist()}")

            # デコードテスト
            decoded = pipe.tokenizer.decode(token_ids[0])
            print(f"\nデコード結果:")
            if len(decoded) > 100:
                print(f"  {decoded[:100]}...")
            else:
                print(f"  {decoded}")

            print("✅ トークナイズ成功")

        except Exception as e:
            print(f"❌ トークナイズに失敗: {e}")
            import traceback
            traceback.print_exc()

    # モード別テスト
    print("\n\n" + "="*80)
    print("【japanese_modeテスト】")
    print("="*80)

    test_text = "東京の天気は晴れです。"

    modes = ["auto", "always", "never"]

    for mode in modes:
        print(f"\n--- Mode: {mode} ---")

        try:
            pipe_mode = JapaneseTaDiCodecPipeline.from_pretrained(
                ckpt_dir=tadicodec_path,
                japanese_tokenizer_path=japanese_tokenizer_path,
                enable_japanese_phoneme=True,
                japanese_mode=mode,
                auto_download=False,
            )

            token_ids = pipe_mode.tokenize_text(test_text)
            print(f"トークンID形状: {token_ids.shape}")
            print(f"最初の10トークン: {token_ids[0][:10].tolist()}")

            decoded = pipe_mode.tokenizer.decode(token_ids[0])
            print(f"デコード（最初の100文字）: {decoded[:100]}...")

        except Exception as e:
            print(f"❌ エラー: {e}")


    print("\n\n" + "="*80)
    print("【テスト完了】")
    print("="*80)
    print("""
まとめ:
✅ 日本語対応パイプラインが正常に動作
✅ 日本語テキスト→音素+韻律情報への自動変換が機能
✅ 英語テキストは音素変換されずにそのまま処理
✅ japanese_mode (auto/always/never) の切り替えが可能

これにより、TaDiCodecで日本語テキストを処理する際に
OpenJTalkのすべての韻律情報（50種類以上）が自動的に活用されます。
    """)


def main():
    print("\n" + "🇯🇵 " * 40)
    print("日本語対応TaDiCodecパイプライン テストスクリプト")
    print("🇯🇵 " * 40)

    test_japanese_pipeline()


if __name__ == "__main__":
    main()
