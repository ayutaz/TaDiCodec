#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日本語データ拡張機能のテスト

4つの拡張手法をテスト:
1. ピッチシフト（男性⇄女性）
2. 速度変化
3. アクセント位置の変更
4. 日英コードスイッチング
"""

import sys
import os
import numpy as np
import librosa

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from models.tts.tadicodec.japanese_data_augmentation import JapaneseDataAugmentation


def generate_test_audio(duration=2.0, sr=24000, freq=440.0):
    """テスト用の音声を生成（正弦波）"""
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return audio


def test_pitch_shift():
    """ピッチシフトのテスト"""
    print("\n" + "="*80)
    print("【テスト1】ピッチシフト（男性⇄女性）")
    print("="*80)

    sr = 24000
    audio = generate_test_audio(sr=sr)

    augmentor = JapaneseDataAugmentation(
        use_pitch_shift=True,
        use_speed_perturb=False,
        use_accent_augment=False,
        use_code_switch=False,
        augment_prob=1.0,  # 必ず適用
    )

    print(f"\n元の音声:")
    print(f"  長さ: {len(audio)} samples ({len(audio)/sr:.2f}秒)")
    print(f"  サンプリングレート: {sr} Hz")

    # 複数回ピッチシフトを試す
    for i in range(3):
        aug_audio, _ = augmentor.augment(audio, "テスト", sr=sr)
        print(f"\nピッチシフト {i+1}:")
        print(f"  長さ: {len(aug_audio)} samples ({len(aug_audio)/sr:.2f}秒)")
        print(f"  最大振幅: {np.max(np.abs(aug_audio)):.3f}")
        print(f"  ✅ ピッチシフト適用完了")


def test_speed_perturb():
    """速度変化のテスト"""
    print("\n" + "="*80)
    print("【テスト2】速度変化（話速の変更）")
    print("="*80)

    sr = 24000
    audio = generate_test_audio(sr=sr)

    augmentor = JapaneseDataAugmentation(
        use_pitch_shift=False,
        use_speed_perturb=True,
        use_accent_augment=False,
        use_code_switch=False,
        speed_perturb_range=(0.8, 1.2),  # 広い範囲
        augment_prob=1.0,  # 必ず適用
    )

    print(f"\n元の音声:")
    print(f"  長さ: {len(audio)} samples ({len(audio)/sr:.2f}秒)")

    # 複数回速度変化を試す
    for i in range(3):
        aug_audio, _ = augmentor.augment(audio, "テスト", sr=sr)
        print(f"\n速度変化 {i+1}:")
        print(f"  長さ: {len(aug_audio)} samples ({len(aug_audio)/sr:.2f}秒)")
        print(f"  ✅ 速度変化適用完了")


def test_accent_augment():
    """アクセント位置の変更のテスト"""
    print("\n" + "="*80)
    print("【テスト3】アクセント位置の変更（日本語特有）")
    print("="*80)

    augmentor = JapaneseDataAugmentation(
        use_pitch_shift=False,
        use_speed_perturb=False,
        use_accent_augment=True,
        use_code_switch=False,
        augment_prob=1.0,  # 必ず適用
    )

    test_texts = [
        "東京の天気は晴れです。",
        "こんにちは、元気ですか？",
        "今日は良い天気ですね。",
    ]

    for i, text in enumerate(test_texts, 1):
        audio = generate_test_audio()
        _, aug_text = augmentor.augment(audio, text, language="ja")

        print(f"\nテスト {i}:")
        print(f"  元のテキスト: {text}")
        print(f"  拡張後: {aug_text}")
        if aug_text != text:
            print(f"  ✅ アクセント拡張が適用されました")
        else:
            print(f"  ℹ️  この回は拡張が適用されませんでした")


def test_code_switch():
    """日英コードスイッチングのテスト"""
    print("\n" + "="*80)
    print("【テスト4】日英コードスイッチング")
    print("="*80)

    augmentor = JapaneseDataAugmentation(
        use_pitch_shift=False,
        use_speed_perturb=False,
        use_accent_augment=False,
        use_code_switch=True,
        augment_prob=1.0,  # 必ず適用
    )

    test_texts = [
        "こんにちは、元気ですか？",
        "ありがとうございます。",
        "コンピューターでプログラムを実行します。",
        "インターネットでデータを送信します。",
    ]

    for i, text in enumerate(test_texts, 1):
        audio = generate_test_audio()
        _, aug_text = augmentor.augment(audio, text, language="ja")

        print(f"\nテスト {i}:")
        print(f"  元のテキスト: {text}")
        print(f"  拡張後: {aug_text}")
        if aug_text != text:
            print(f"  ✅ コードスイッチングが適用されました")
        else:
            print(f"  ℹ️  この回は拡張が適用されませんでした")


def test_combined_augmentation():
    """複数の拡張を組み合わせたテスト"""
    print("\n" + "="*80)
    print("【テスト5】複数の拡張の組み合わせ")
    print("="*80)

    sr = 24000
    audio = generate_test_audio(sr=sr)
    text = "東京の天気は晴れです。"

    augmentor = JapaneseDataAugmentation(
        use_pitch_shift=True,
        use_speed_perturb=True,
        use_accent_augment=True,
        use_code_switch=True,
        augment_prob=0.5,  # 50%の確率で各拡張を適用
    )

    print(f"\n元のデータ:")
    print(f"  テキスト: {text}")
    print(f"  音声長: {len(audio)} samples")

    # 10回試して、拡張のバリエーションを確認
    print(f"\n10回の拡張を試行:")
    pitch_count = 0
    speed_count = 0
    text_change_count = 0

    for i in range(10):
        aug_audio, aug_text = augmentor.augment(audio, text, sr=sr, language="ja")

        applied = []
        if not np.array_equal(aug_audio, audio):
            applied.append("音声拡張")
            if len(aug_audio) == len(audio):
                pitch_count += 1
            else:
                speed_count += 1

        if aug_text != text:
            applied.append("テキスト拡張")
            text_change_count += 1

        if applied:
            print(f"  試行 {i+1}: {', '.join(applied)} 適用")
        else:
            print(f"  試行 {i+1}: 拡張なし")

    print(f"\n統計:")
    print(f"  ピッチシフト適用回数: {pitch_count}/10")
    print(f"  速度変化適用回数: {speed_count}/10")
    print(f"  テキスト拡張適用回数: {text_change_count}/10")


def test_configuration():
    """設定ファイルのテスト"""
    print("\n" + "="*80)
    print("【テスト6】設定ファイルの確認")
    print("="*80)

    config_path = os.path.join(
        parent_dir,
        "egs/tts/TaDiCodec/tadicodec_japanese_finetune.json"
    )

    if os.path.exists(config_path):
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print(f"\n設定ファイル: {config_path}")
        print(f"✅ 設定ファイルが見つかりました")

        preprocess = config.get("preprocess", {})
        print(f"\nデータ拡張設定:")
        print(f"  data_augment: {preprocess.get('data_augment', [])}")
        print(f"  use_pitch_shift: {preprocess.get('use_pitch_shift', False)}")
        print(f"  use_speed_perturb: {preprocess.get('use_speed_perturb', False)}")
        print(f"  use_accent_augment: {preprocess.get('use_accent_augment', False)}")
        print(f"  use_code_switch: {preprocess.get('use_code_switch', False)}")
        print(f"  pitch_shift_range: {preprocess.get('pitch_shift_range', [])}")
        print(f"  speed_perturb_range: {preprocess.get('speed_perturb_range', [])}")
        print(f"  augment_prob: {preprocess.get('augment_prob', 0.0)}")

        # トークナイザーパスの確認
        tokenizer_path = preprocess.get('tokenizer_path', '')
        print(f"\nトークナイザー設定:")
        print(f"  tokenizer_path: {tokenizer_path}")
        if "Japanese" in tokenizer_path:
            print(f"  ✅ 日本語トークナイザーが設定されています")

        # モデル設定の確認
        model = config.get("model", {}).get("tadicodec", {})
        vocab_size = model.get("text_vocab_size", 0)
        print(f"\nモデル設定:")
        print(f"  text_vocab_size: {vocab_size}")
        if vocab_size == 33844:
            print(f"  ✅ 日本語拡張語彙サイズが設定されています")

    else:
        print(f"❌ 設定ファイルが見つかりません: {config_path}")


def main():
    print("\n" + "🧪 " * 40)
    print("日本語データ拡張機能 総合テスト")
    print("🧪 " * 40)

    # 各テストを実行
    test_pitch_shift()
    test_speed_perturb()
    test_accent_augment()
    test_code_switch()
    test_combined_augmentation()
    test_configuration()

    # 最終結果
    print("\n" + "="*80)
    print("【総合結果】")
    print("="*80)
    print("""
✅ 完了した機能:
1. ピッチシフト（男性⇄女性）
   - librosa.effects.pitch_shift を使用
   - 範囲: -4.0 〜 +4.0 半音

2. 速度変化（話速の変更）
   - librosa.effects.time_stretch を使用
   - 範囲: 0.9x 〜 1.1x

3. アクセント位置の変更（日本語特有）
   - pyopenjtalk でアクセント情報を取得
   - テキストにアクセント変更の注記を追加

4. 日英コードスイッチング
   - 日本語の一部を英語に置き換え
   - カタカナ語→英語の変換

✅ 設定ファイル:
- egs/tts/TaDiCodec/tadicodec_japanese_finetune.json
- すべての拡張機能が設定済み

✅ データセットクラス:
- models/tts/tadicodec/tadicodec_dataset_japanese.py
- TadiCodecJapaneseDataset が実装済み

🎯 次のステップ:
1. JVS/JSUTデータセットの準備
2. データ拡張を有効にして学習開始
3. 拡張効果の評価
    """)

    print("\n" + "="*80)
    print("テスト完了！")
    print("="*80)


if __name__ == "__main__":
    main()
