"""
日本語対応TaDiCodecパイプライン

オリジナルのTaDiCodecPiplineを拡張し、日本語テキストを自動的に
音素+韻律情報に変換してからトークナイズする機能を追加。

主な機能:
1. 日本語テキスト → pyopenjtalk → 音素列+韻律情報
2. 音素列+韻律情報 → リッチトークン列
3. リッチトークン列 → トークンID → text_embedding

使用方法:
    from models.tts.tadicodec.inference_tadicodec_japanese import JapaneseTaDiCodecPipeline

    pipe = JapaneseTaDiCodecPipeline.from_pretrained(
        "amphion/TaDiCodec",
        japanese_tokenizer_path="./ckpt/TaDiCodec_Japanese_Full/text_tokenizer"
    )

    # 日本語テキストが自動的に音素+韻律情報に変換される
    indices = pipe.encode(
        speech_path="sample.wav",
        text="東京の天気は晴れです。"
    )
"""

import os
import sys
from typing import Optional

import torch

# Add parent directory to path for importing
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    import pyopenjtalk
    PYOPENJTALK_AVAILABLE = True
except ImportError:
    PYOPENJTALK_AVAILABLE = False
    print("Warning: pyopenjtalk not available. Japanese phoneme conversion will be disabled.")

from models.tts.tadicodec.inference_tadicodec import TaDiCodecPipline


class JapaneseTaDiCodecPipeline(TaDiCodecPipline):
    """
    日本語対応TaDiCodecパイプライン

    オリジナルのTaDiCodecPiplineを継承し、tokenize_textメソッドを
    オーバーライドして日本語テキストの自動音素変換機能を追加。
    """

    def __init__(
        self,
        cfg,
        model_path: str,
        device: torch.device,
        tokenizer_path: Optional[str] = None,
        vocoder_ckpt_path: Optional[str] = None,
        enable_japanese_phoneme: bool = True,
        japanese_mode: str = "auto",  # "auto", "always", "never"
    ):
        """
        Args:
            cfg: Configuration object
            model_path: Path to TaDiCodec model
            device: Device to run on
            tokenizer_path: Path to tokenizer (use Japanese tokenizer for Japanese support)
            vocoder_ckpt_path: Path to vocoder checkpoint
            enable_japanese_phoneme: Enable Japanese phoneme conversion
            japanese_mode: When to apply Japanese phoneme conversion:
                - "auto": Detect if text contains Japanese characters
                - "always": Always apply phoneme conversion
                - "never": Never apply phoneme conversion (use original text)
        """
        super().__init__(cfg, model_path, device, tokenizer_path, vocoder_ckpt_path)

        self.enable_japanese_phoneme = enable_japanese_phoneme and PYOPENJTALK_AVAILABLE
        self.japanese_mode = japanese_mode

        if enable_japanese_phoneme and not PYOPENJTALK_AVAILABLE:
            print("Warning: Japanese phoneme conversion requested but pyopenjtalk is not available.")
            print("Install with: pip install pyopenjtalk-plus")

        # Import Japanese tokenizer if available
        if self.enable_japanese_phoneme:
            from scripts.create_japanese_tokenizer_full import CompleteJapanesePhonemeTokenizer
            self.jp_phoneme_tokenizer = CompleteJapanesePhonemeTokenizer()
        else:
            self.jp_phoneme_tokenizer = None

    def _contains_japanese(self, text: str) -> bool:
        """
        テキストに日本語文字が含まれているかチェック

        Args:
            text: Input text

        Returns:
            True if text contains Japanese characters
        """
        if not text:
            return False

        # Hiragana: U+3040-U+309F
        # Katakana: U+30A0-U+30FF
        # CJK Unified Ideographs: U+4E00-U+9FFF
        for char in text:
            code = ord(char)
            if (0x3040 <= code <= 0x309F or  # Hiragana
                0x30A0 <= code <= 0x30FF or  # Katakana
                0x4E00 <= code <= 0x9FFF):   # Kanji
                return True
        return False

    def _text_to_phoneme_string(self, text: str) -> str:
        """
        日本語テキストを音素+韻律情報の文字列に変換

        Args:
            text: Japanese text (e.g., "東京の天気は晴れです。")

        Returns:
            Phoneme string with prosody info (e.g., "t[POS_OTHER][ACC_TYPE_5] o[POS_OTHER]...")
        """
        if not self.enable_japanese_phoneme or not self.jp_phoneme_tokenizer:
            return text

        # テキスト → 音素+韻律情報
        phonemes_with_prosody = self.jp_phoneme_tokenizer.text_to_phonemes_with_full_prosody(text)

        # リッチトークン列に変換
        rich_tokens = self.jp_phoneme_tokenizer.phonemes_to_rich_tokens(phonemes_with_prosody)

        # スペース区切りの文字列に変換
        phoneme_string = " ".join(rich_tokens)

        return phoneme_string

    def tokenize_text(
        self, text: Optional[str] = None, prompt_text: Optional[str] = None
    ):
        """
        テキストをトークンIDに変換（日本語対応版）

        日本語テキストの場合、自動的に音素+韻律情報に変換してからトークナイズする。

        Args:
            text: Target text
            prompt_text: Prompt text (optional)

        Returns:
            Token IDs tensor
        """
        if self.tokenizer is None or text is None:
            return None

        # 日本語音素変換を適用するか判定
        should_convert = False
        if self.enable_japanese_phoneme:
            if self.japanese_mode == "always":
                should_convert = True
            elif self.japanese_mode == "auto":
                should_convert = self._contains_japanese(text)
                # プロンプトテキストもチェック
                if prompt_text and self._contains_japanese(prompt_text):
                    should_convert = True
            # japanese_mode == "never" の場合は should_convert = False のまま

        # 音素変換を適用
        if should_convert:
            # プロンプトテキストも変換
            if prompt_text is not None:
                phoneme_prompt_text = self._text_to_phoneme_string(prompt_text)
                phoneme_text = self._text_to_phoneme_string(text)
                combined_text = phoneme_prompt_text + " " + phoneme_text
            else:
                combined_text = self._text_to_phoneme_string(text)

            # トークナイズ
            text_token_ids = self.tokenizer(
                combined_text, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)
        else:
            # オリジナルの処理（音素変換なし）
            if prompt_text is not None:
                text_token_ids = self.tokenizer(
                    prompt_text + text, return_tensors="pt", add_special_tokens=False
                ).input_ids.to(self.device)
            else:
                text_token_ids = self.tokenizer(
                    text, return_tensors="pt", add_special_tokens=False
                ).input_ids.to(self.device)

        return text_token_ids

    @classmethod
    def from_pretrained(
        cls,
        ckpt_dir: str = "./ckpt/TaDiCodec",
        device: Optional[torch.device] = None,
        auto_download: bool = True,
        japanese_tokenizer_path: Optional[str] = None,
        enable_japanese_phoneme: bool = True,
        japanese_mode: str = "auto",
    ):
        """
        事前学習済みモデルから日本語対応パイプラインを作成

        Args:
            ckpt_dir: Path to checkpoint directory or HuggingFace model ID
            device: Device to run on
            auto_download: Auto-download from HuggingFace if not found locally
            japanese_tokenizer_path: Path to Japanese tokenizer
                (e.g., "./ckpt/TaDiCodec_Japanese_Full/text_tokenizer")
            enable_japanese_phoneme: Enable Japanese phoneme conversion
            japanese_mode: When to apply phoneme conversion ("auto", "always", "never")

        Returns:
            JapaneseTaDiCodecPipeline instance
        """
        import glob
        from utils.util import load_config

        resolved_device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Resolve checkpoint directory
        resolved_ckpt_dir = cls._resolve_model_path(
            ckpt_dir, auto_download=auto_download, model_type="tadicodec"
        )

        # Load config
        config_path = os.path.join(resolved_ckpt_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {resolved_ckpt_dir}")

        cfg = load_config(config_path)

        # Resolve vocoder path
        vocoder_dir = os.path.join(resolved_ckpt_dir, "vocoder")
        if not os.path.isdir(vocoder_dir):
            vocoder_dir = None

        # Use Japanese tokenizer if provided
        tokenizer_path = japanese_tokenizer_path
        if tokenizer_path is None:
            # Fallback to default tokenizer
            tokenizer_path = os.path.join(resolved_ckpt_dir, "text_tokenizer")
            if not os.path.exists(tokenizer_path):
                tokenizer_path = None

        # Create pipeline
        return cls(
            cfg=cfg,
            model_path=resolved_ckpt_dir,
            device=resolved_device,
            tokenizer_path=tokenizer_path,
            vocoder_ckpt_path=vocoder_dir,
            enable_japanese_phoneme=enable_japanese_phoneme,
            japanese_mode=japanese_mode,
        )


# Convenience function
def create_japanese_pipeline(
    model_path: str = "amphion/TaDiCodec",
    japanese_tokenizer_path: str = "./ckpt/TaDiCodec_Japanese_Full/text_tokenizer",
    device: Optional[torch.device] = None,
    auto_download: bool = True,
):
    """
    日本語対応パイプラインを簡単に作成するヘルパー関数

    Args:
        model_path: Path to TaDiCodec model or HuggingFace model ID
        japanese_tokenizer_path: Path to Japanese tokenizer
        device: Device to run on
        auto_download: Auto-download from HuggingFace if not found locally

    Returns:
        JapaneseTaDiCodecPipeline instance
    """
    return JapaneseTaDiCodecPipeline.from_pretrained(
        ckpt_dir=model_path,
        device=device,
        auto_download=auto_download,
        japanese_tokenizer_path=japanese_tokenizer_path,
        enable_japanese_phoneme=True,
        japanese_mode="auto",
    )
