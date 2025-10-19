"""
日本語特有のデータ拡張機能

TaDiCodecの日本語ファインチューニング用のデータ拡張モジュール。
以下の4つの拡張手法を提供：

1. ピッチシフト (Pitch Shift): 男性⇄女性の声質変換
2. 速度変化 (Speed Perturbation): 話速の変更
3. アクセント位置の変更 (Accent Augmentation): 日本語特有
4. 日英コードスイッチング (Code Switching): 多言語対応

使用方法:
    from models.tts.tadicodec.japanese_data_augmentation import JapaneseDataAugmentation

    augmentor = JapaneseDataAugmentation(
        use_pitch_shift=True,
        use_speed_perturb=True,
        use_accent_augment=True,
        use_code_switch=True,
    )

    # 音声とテキストを拡張
    augmented_speech, augmented_text = augmentor.augment(speech, text, sr=24000)
"""

import numpy as np
import librosa
import random
from typing import Tuple, Optional, List

try:
    import pyopenjtalk
    PYOPENJTALK_AVAILABLE = True
except ImportError:
    PYOPENJTALK_AVAILABLE = False


class JapaneseDataAugmentation:
    """
    日本語特有のデータ拡張クラス

    Attributes:
        use_pitch_shift: ピッチシフトを使用するか
        use_speed_perturb: 速度変化を使用するか
        use_accent_augment: アクセント位置の変更を使用するか
        use_code_switch: コードスイッチングを使用するか
        pitch_shift_range: ピッチシフトの範囲（半音）
        speed_perturb_range: 速度変化の範囲
        augment_prob: 各拡張手法を適用する確率
    """

    def __init__(
        self,
        use_pitch_shift: bool = True,
        use_speed_perturb: bool = True,
        use_accent_augment: bool = True,
        use_code_switch: bool = True,
        pitch_shift_range: Tuple[float, float] = (-4.0, 4.0),  # 半音単位
        speed_perturb_range: Tuple[float, float] = (0.9, 1.1),
        augment_prob: float = 0.5,  # 各拡張を適用する確率
    ):
        self.use_pitch_shift = use_pitch_shift
        self.use_speed_perturb = use_speed_perturb
        self.use_accent_augment = use_accent_augment and PYOPENJTALK_AVAILABLE
        self.use_code_switch = use_code_switch and PYOPENJTALK_AVAILABLE

        self.pitch_shift_range = pitch_shift_range
        self.speed_perturb_range = speed_perturb_range
        self.augment_prob = augment_prob

        if use_accent_augment and not PYOPENJTALK_AVAILABLE:
            print("Warning: pyopenjtalk not available. Accent augmentation disabled.")

    def augment(
        self,
        speech: np.ndarray,
        text: str,
        sr: int = 24000,
        language: str = "ja",
    ) -> Tuple[np.ndarray, str]:
        """
        音声とテキストにデータ拡張を適用

        Args:
            speech: 入力音声 (numpy array)
            text: 入力テキスト
            sr: サンプリングレート
            language: 言語コード

        Returns:
            拡張された音声とテキストのタプル
        """
        augmented_speech = speech.copy()
        augmented_text = text

        # 1. ピッチシフト（50%の確率で適用）
        if self.use_pitch_shift and random.random() < self.augment_prob:
            augmented_speech = self._pitch_shift(augmented_speech, sr)

        # 2. 速度変化（50%の確率で適用）
        if self.use_speed_perturb and random.random() < self.augment_prob:
            augmented_speech = self._speed_perturb(augmented_speech, sr)

        # 3. アクセント位置の変更（日本語のみ、30%の確率で適用）
        if (
            self.use_accent_augment
            and language == "ja"
            and random.random() < 0.3
        ):
            augmented_text = self._accent_augment(augmented_text)

        # 4. コードスイッチング（日本語のみ、20%の確率で適用）
        if (
            self.use_code_switch
            and language == "ja"
            and random.random() < 0.2
        ):
            augmented_text = self._code_switch(augmented_text)

        return augmented_speech, augmented_text

    def _pitch_shift(self, speech: np.ndarray, sr: int) -> np.ndarray:
        """
        ピッチシフト（男性⇄女性の声質変換）

        Args:
            speech: 入力音声
            sr: サンプリングレート

        Returns:
            ピッチシフトされた音声
        """
        # ランダムにピッチシフト量を決定
        n_steps = random.uniform(*self.pitch_shift_range)

        # librosaのピッチシフト
        shifted = librosa.effects.pitch_shift(
            y=speech,
            sr=sr,
            n_steps=n_steps,
        )

        return shifted

    def _speed_perturb(self, speech: np.ndarray, sr: int) -> np.ndarray:
        """
        速度変化（話速の変更）

        Args:
            speech: 入力音声
            sr: サンプリングレート

        Returns:
            速度変化された音声
        """
        # ランダムに速度変化率を決定
        rate = random.uniform(*self.speed_perturb_range)

        # librosaの時間伸縮
        perturbed = librosa.effects.time_stretch(y=speech, rate=rate)

        # 元の長さに合わせる（パディングまたはトリミング）
        if len(perturbed) > len(speech):
            perturbed = perturbed[: len(speech)]
        elif len(perturbed) < len(speech):
            pad_len = len(speech) - len(perturbed)
            perturbed = np.pad(perturbed, (0, pad_len), mode="constant")

        return perturbed

    def _accent_augment(self, text: str) -> str:
        """
        アクセント位置の変更（日本語特有）

        OpenJTalkのアクセント情報を変更して、異なるアクセントパターンを生成。

        Args:
            text: 入力テキスト

        Returns:
            アクセント変更後のテキスト（注記付き）
        """
        if not PYOPENJTALK_AVAILABLE:
            return text

        try:
            # OpenJTalkでフルコンテキストラベルを取得
            labels = pyopenjtalk.extract_fullcontext(text)

            # アクセント情報を抽出
            accent_types = []
            for label in labels:
                parts = label.split("/")
                for part in parts:
                    if part.startswith("F:"):
                        # F フィールドからアクセント型を取得
                        f_parts = part.split(":")
                        if len(f_parts) > 1:
                            f_values = f_parts[1].split("_")
                            if len(f_values) > 1:
                                accent_type = f_values[1]  # アクセント型
                                accent_types.append(accent_type)
                        break

            # アクセント型をランダムに変更
            # 注: 実際の音声合成モデルがアクセント情報を理解する必要がある
            # ここでは、テキストにアクセント情報の注記を追加する
            if accent_types:
                # 例: "東京の天気" → "東京[ACC_5]の天気[ACC_1]"
                # 実際には、より高度な処理が必要
                augmented_text = text + " [ACCENT_VARIED]"
            else:
                augmented_text = text

            return augmented_text

        except Exception as e:
            # エラーが発生した場合は元のテキストを返す
            print(f"Warning: Accent augmentation failed: {e}")
            return text

    def _code_switch(self, text: str) -> str:
        """
        日英コードスイッチング

        日本語テキストの一部を英語に置き換えて、コードスイッチングのパターンを生成。

        Args:
            text: 入力テキスト（日本語）

        Returns:
            コードスイッチングされたテキスト
        """
        if not PYOPENJTALK_AVAILABLE:
            return text

        # 日本語→英語の単語置き換え辞書
        japanese_to_english = {
            "こんにちは": "Hello",
            "ありがとう": "Thank you",
            "さようなら": "Goodbye",
            "はい": "Yes",
            "いいえ": "No",
            "お願いします": "Please",
            "すみません": "Excuse me",
            "コンピューター": "Computer",
            "インターネット": "Internet",
            "プログラム": "Program",
            "データ": "Data",
            "ファイル": "File",
            "システム": "System",
            "アプリケーション": "Application",
            "ネットワーク": "Network",
        }

        # ランダムに1-2個の単語を置き換え
        num_replacements = random.randint(1, 2)
        augmented_text = text

        for jp_word, en_word in list(japanese_to_english.items())[:num_replacements]:
            if jp_word in augmented_text:
                augmented_text = augmented_text.replace(jp_word, en_word)
                # 最初の1個だけ置き換えたら終了
                break

        return augmented_text


class JapaneseAugmentedDataset:
    """
    データ拡張機能付きデータセットのミックスイン

    使用方法:
        class MyDataset(JapaneseAugmentedDataset, BaseDataset):
            def __init__(self, cfg):
                super().__init__(cfg)
                self.setup_augmentation(cfg)

            def __getitem__(self, idx):
                speech, text, language = ...  # 元のデータ取得
                if self.training:
                    speech, text = self.apply_augmentation(speech, text, language)
                return speech, text
    """

    def setup_augmentation(self, cfg):
        """データ拡張の設定"""
        # 設定ファイルからパラメータを取得
        if hasattr(cfg.preprocess, "data_augment") and "japanese" in cfg.preprocess.data_augment:
            use_pitch_shift = getattr(cfg.preprocess, "use_pitch_shift", True)
            use_speed_perturb = getattr(cfg.preprocess, "use_speed_perturb", True)
            use_accent_augment = getattr(cfg.preprocess, "use_accent_augment", True)
            use_code_switch = getattr(cfg.preprocess, "use_code_switch", True)

            self.augmentor = JapaneseDataAugmentation(
                use_pitch_shift=use_pitch_shift,
                use_speed_perturb=use_speed_perturb,
                use_accent_augment=use_accent_augment,
                use_code_switch=use_code_switch,
            )
            self.use_augmentation = True
        else:
            self.use_augmentation = False

    def apply_augmentation(
        self,
        speech: np.ndarray,
        text: str,
        language: str = "ja",
        sr: int = 24000,
    ) -> Tuple[np.ndarray, str]:
        """データ拡張を適用"""
        if self.use_augmentation and hasattr(self, "augmentor"):
            return self.augmentor.augment(speech, text, sr, language)
        else:
            return speech, text


# 使用例
if __name__ == "__main__":
    print("="*80)
    print("日本語データ拡張モジュール テスト")
    print("="*80)

    # サンプル音声を生成（実際にはファイルから読み込む）
    sr = 24000
    duration = 2.0  # 秒
    t = np.linspace(0, duration, int(sr * duration))
    speech = np.sin(2 * np.pi * 440 * t)  # 440Hz（A4）の正弦波

    text = "東京の天気は晴れです。"

    # データ拡張器の作成
    augmentor = JapaneseDataAugmentation(
        use_pitch_shift=True,
        use_speed_perturb=True,
        use_accent_augment=True,
        use_code_switch=True,
    )

    print(f"\n元のテキスト: {text}")
    print(f"元の音声長: {len(speech)} samples")

    # 複数回拡張を試す
    for i in range(5):
        aug_speech, aug_text = augmentor.augment(speech, text, sr=sr)
        print(f"\n【拡張 {i+1}】")
        print(f"  テキスト: {aug_text}")
        print(f"  音声長: {len(aug_speech)} samples")
        if aug_text != text:
            print(f"  → テキスト拡張が適用されました")

    print("\n" + "="*80)
    print("テスト完了")
    print("="*80)
