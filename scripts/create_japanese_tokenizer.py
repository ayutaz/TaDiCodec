#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日本語音素トークナイザーの作成
OpenJTalkのフルコンテキストラベル情報を最大限活用
"""

import sys
import os
import json
from typing import List, Dict, Tuple
from pathlib import Path

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import pyopenjtalk
from transformers import AutoTokenizer


class JapanesePhonemeTokenizer:
    """
    OpenJTalkのフルコンテキストラベルを活用した日本語音素トークナイザー

    取得する情報:
    - 音素（phoneme）
    - アクセント位置（accent position）
    - モーラ位置（mora position）
    - 品詞情報（POS tags）
    - イントネーション句境界（intonation phrase boundary）
    - ピッチアクセント型（pitch accent type）
    """

    def __init__(self):
        # 日本語音素セット
        self.japanese_phonemes = {
            # 母音
            'a', 'i', 'u', 'e', 'o',
            # 子音
            'k', 'ky', 'g', 'gy',
            's', 'sh', 'z', 'j',
            't', 'ty', 'ch', 'd', 'dy',
            'n', 'ny',
            'h', 'hy', 'b', 'by', 'p', 'py',
            'm', 'my',
            'y',
            'r', 'ry',
            'w',
            # 特殊音素
            'N',   # 撥音（ん）
            'Q',   # 促音（っ）
            'cl',  # 無声閉鎖音の閉鎖区間
            # ポーズ
            'pau', 'sil',
        }

        # アクセント・韻律マーカー
        self.prosody_markers = {
            '[ACCENT]',      # アクセント核
            '[PHRASE_END]',  # アクセント句末
            '[BREATH]',      # ブレス
            '[PAUSE]',       # ポーズ
        }

        # 品詞タグ（主要なもの）
        self.pos_tags = {
            '[NOUN]',        # 名詞
            '[VERB]',        # 動詞
            '[ADJ]',         # 形容詞
            '[PARTICLE]',    # 助詞
            '[AUX]',         # 助動詞
            '[INTERJECTION]',# 感動詞
        }

    def parse_fullcontext_label(self, label: str) -> Dict:
        """
        フルコンテキストラベルをパース

        ラベル形式:
        p1^p2-p3+p4=p5/A:a1+a2+a3/B:...

        各フィールドの意味:
        - p1-p5: 音素のコンテキスト（前々、前、現在、次、次次）
        - A: モーラ情報（位置、アクセント型、モーラ数）
        - F: アクセント句情報
        - G: ブレス群情報
        - I: イントネーション句情報
        - J: 発話全体情報
        """
        parts = label.split('/')

        # 音素部分のパース: p1^p2-p3+p4=p5
        phoneme_part = parts[0]
        import re
        phoneme_elements = re.split(r'[\^=\-+]', phoneme_part)

        result = {
            'phoneme': phoneme_elements[2] if len(phoneme_elements) > 2 else 'sil',
            'prev_phoneme': phoneme_elements[1] if len(phoneme_elements) > 1 else 'xx',
            'next_phoneme': phoneme_elements[3] if len(phoneme_elements) > 3 else 'xx',
        }

        # 各フィールドのパース
        for part in parts[1:]:
            if ':' in part:
                key, value = part.split(':', 1)
                result[key] = value

        return result

    def extract_accent_info(self, parsed_label: Dict) -> Dict:
        """
        アクセント情報を抽出

        A フィールド: a1+a2+a3
        - a1: モーラの位置（アクセント句内）
        - a2: モーラの位置（単語内）
        - a3: モーラ数（単語内）
        """
        accent_info = {
            'mora_position_in_phrase': 0,
            'mora_position_in_word': 0,
            'total_mora_in_word': 0,
            'is_accent_nucleus': False,
        }

        if 'A' in parsed_label:
            a_parts = parsed_label['A'].split('+')
            if len(a_parts) >= 3:
                try:
                    accent_info['mora_position_in_phrase'] = int(a_parts[0]) if a_parts[0] != 'xx' else 0
                    accent_info['mora_position_in_word'] = int(a_parts[1]) if a_parts[1] != 'xx' else 0
                    accent_info['total_mora_in_word'] = int(a_parts[2]) if a_parts[2] != 'xx' else 0
                except:
                    pass

        return accent_info

    def extract_phrase_info(self, parsed_label: Dict) -> Dict:
        """
        アクセント句情報を抽出

        F フィールド: モーラ数、アクセント型など
        """
        phrase_info = {
            'mora_count_in_phrase': 0,
            'accent_type': 0,
            'is_phrase_end': False,
        }

        if 'F' in parsed_label:
            import re
            f_parts = re.split(r'[#_@|]', parsed_label['F'])
            if len(f_parts) >= 2:
                try:
                    phrase_info['mora_count_in_phrase'] = int(f_parts[0]) if f_parts[0] != 'xx' else 0
                    phrase_info['accent_type'] = int(f_parts[1]) if f_parts[1] != 'xx' else 0
                except:
                    pass

        return phrase_info

    def text_to_phonemes_with_prosody(self, text: str) -> List[Tuple[str, Dict]]:
        """
        テキストを音素列 + 韻律情報に変換

        Returns:
            List[Tuple[str, Dict]]: [(音素, {韻律情報}), ...]
        """
        # フルコンテキストラベルの取得
        labels = pyopenjtalk.extract_fullcontext(text)

        # フロントエンド情報の取得（単語レベル）
        frontend_features = pyopenjtalk.run_frontend(text)

        phonemes_with_prosody = []

        for label in labels:
            # ラベルのパース
            parsed = self.parse_fullcontext_label(label)

            phoneme = parsed['phoneme']

            # sil（無音）はスキップ
            if phoneme == 'sil' or phoneme == 'xx':
                continue

            # 韻律情報の抽出
            accent_info = self.extract_accent_info(parsed)
            phrase_info = self.extract_phrase_info(parsed)

            # 韻律情報をまとめる
            prosody = {
                **accent_info,
                **phrase_info,
            }

            phonemes_with_prosody.append((phoneme, prosody))

        return phonemes_with_prosody

    def phonemes_to_tokens(self, phonemes_with_prosody: List[Tuple[str, Dict]]) -> List[str]:
        """
        音素+韻律情報をトークン列に変換

        例:
        - 通常の音素: "k", "o", "N"
        - アクセント核の音素: "k[ACCENT]", "y[ACCENT]"
        - 句末の音素: "o[PHRASE_END]"
        """
        tokens = []

        for phoneme, prosody in phonemes_with_prosody:
            token = phoneme

            # アクセント核のマーキング
            if prosody.get('is_accent_nucleus', False):
                token += '[ACCENT]'

            # 句末のマーキング
            if prosody.get('is_phrase_end', False):
                token += '[PHRASE_END]'

            tokens.append(token)

        return tokens

    def create_vocabulary(self, save_path: str = None) -> Dict:
        """
        日本語音素 + 韻律マーカーの語彙を作成
        """
        vocabulary = {
            'phonemes': sorted(list(self.japanese_phonemes)),
            'prosody_markers': sorted(list(self.prosody_markers)),
            'pos_tags': sorted(list(self.pos_tags)),
            'combined_tokens': []
        }

        # 音素 + 韻律マーカーの組み合わせも追加
        for phoneme in self.japanese_phonemes:
            for marker in ['', '[ACCENT]', '[PHRASE_END]']:
                if marker:
                    vocabulary['combined_tokens'].append(phoneme + marker)

        vocabulary['combined_tokens'] = sorted(vocabulary['combined_tokens'])

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(vocabulary, f, ensure_ascii=False, indent=2)
            print(f"✅ Vocabulary saved to {save_path}")

        return vocabulary

    def demonstrate(self):
        """デモンストレーション"""
        test_texts = [
            "東京の天気は晴れです。",
            "こんにちは、元気ですか？",
            "コンピューター",
        ]

        print("="*70)
        print("日本語音素トークナイザー デモンストレーション")
        print("="*70)

        for text in test_texts:
            print(f"\n【テキスト】 {text}")
            print("-"*70)

            # 音素+韻律情報の抽出
            phonemes_with_prosody = self.text_to_phonemes_with_prosody(text)

            print("\n音素と韻律情報:")
            for phoneme, prosody in phonemes_with_prosody[:10]:  # 最初の10個
                print(f"  {phoneme}: {prosody}")

            # トークン列に変換
            tokens = self.phonemes_to_tokens(phonemes_with_prosody)
            print(f"\nトークン列:")
            print(f"  {' '.join(tokens)}")


def extend_existing_tokenizer(
    base_tokenizer_path: str,
    output_path: str,
    japanese_vocabulary: Dict
):
    """
    既存のTaDiCodecトークナイザーに日本語音素を追加

    Args:
        base_tokenizer_path: 既存のトークナイザーのパス
        output_path: 新しいトークナイザーの保存先
        japanese_vocabulary: 日本語語彙辞書
    """
    print(f"\n{'='*70}")
    print("既存トークナイザーの拡張")
    print("="*70)

    # 既存トークナイザーのロード
    print(f"\n📥 Loading base tokenizer from: {base_tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)
        print(f"✅ Base tokenizer loaded")
        print(f"   Original vocabulary size: {len(tokenizer)}")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return None

    # 日本語トークンを追加
    new_tokens = []

    # 基本音素
    new_tokens.extend(japanese_vocabulary['phonemes'])

    # 韻律マーカー
    new_tokens.extend(japanese_vocabulary['prosody_markers'])

    # 組み合わせトークン
    new_tokens.extend(japanese_vocabulary['combined_tokens'])

    # 重複を除去
    new_tokens = list(set(new_tokens))

    print(f"\n➕ Adding {len(new_tokens)} Japanese tokens...")

    # トークンを追加
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"✅ Added {num_added} new tokens")
    print(f"   New vocabulary size: {len(tokenizer)}")

    # 保存
    print(f"\n💾 Saving extended tokenizer to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    print(f"✅ Tokenizer saved successfully")

    # テスト
    print(f"\n🧪 Testing extended tokenizer...")
    test_texts = [
        "This is a test.",  # 英語（既存）
        "これはテストです。",  # 日本語（新規）
        "k o N n i ch i w a [ACCENT]",  # 音素列
    ]

    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        print(f"   '{text}' → {tokens[:10]}...")

    return tokenizer


def main():
    """メイン処理"""
    print("🇯🇵 日本語音素トークナイザー作成スクリプト")
    print("="*70)

    # 日本語音素トークナイザーの初期化
    jp_tokenizer = JapanesePhonemeTokenizer()

    # デモンストレーション
    jp_tokenizer.demonstrate()

    # 語彙の作成
    print(f"\n{'='*70}")
    print("日本語語彙の作成")
    print("="*70)

    vocab_save_path = os.path.join(parent_dir, "ckpt", "japanese_phoneme_vocabulary.json")
    vocabulary = jp_tokenizer.create_vocabulary(vocab_save_path)

    print(f"\n📊 Vocabulary statistics:")
    print(f"   Basic phonemes: {len(vocabulary['phonemes'])}")
    print(f"   Prosody markers: {len(vocabulary['prosody_markers'])}")
    print(f"   Combined tokens: {len(vocabulary['combined_tokens'])}")
    print(f"   Total: {len(vocabulary['phonemes']) + len(vocabulary['prosody_markers']) + len(vocabulary['combined_tokens'])}")

    # 既存トークナイザーの拡張
    base_tokenizer_path = os.path.join(parent_dir, "ckpt", "TaDiCodec", "text_tokenizer")
    output_tokenizer_path = os.path.join(parent_dir, "ckpt", "TaDiCodec_Japanese", "text_tokenizer")

    if os.path.exists(base_tokenizer_path):
        extended_tokenizer = extend_existing_tokenizer(
            base_tokenizer_path,
            output_tokenizer_path,
            vocabulary
        )
    else:
        print(f"\n⚠️  Base tokenizer not found at: {base_tokenizer_path}")
        print(f"   Please download the TaDiCodec model first:")
        print(f"   python -c \"from models.tts.tadicodec.inference_tadicodec import TaDiCodecPipline; TaDiCodecPipline.from_pretrained('amphion/TaDiCodec')\"")

    print(f"\n{'='*70}")
    print("✅ 日本語音素トークナイザー作成完了")
    print("="*70)
    print(f"\n生成されたファイル:")
    print(f"   - {vocab_save_path}")
    print(f"   - {output_tokenizer_path}/")


if __name__ == "__main__":
    main()
