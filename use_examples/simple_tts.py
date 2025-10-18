#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple TTS script for generating Japanese speech with Qwen2.5-3B

Usage:
    # From use_examples directory
    python simple_tts.py
    python simple_tts.py "あなたの好きなテキスト"

    # From project root directory
    python use_examples/simple_tts.py "あなたの好きなテキスト"
    python ./use_examples/simple_tts.py "あなたの好きなテキスト"
"""

import torch
import soundfile as sf
import sys
import os

# Add parent directory to path (works from anywhere)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from models.tts.llm_tts.inference_llm_tts import TTSInferencePipeline


def main():
    # Get text from command line argument or use default
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = "こんにちは、これはテストです。"

    print(f"📝 生成するテキスト: {text}")
    print("🔄 モデルを読み込んでいます...")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Qwen2.5-3B model (highest quality)
    pipeline = TTSInferencePipeline.from_pretrained(
        tadicodec_path="amphion/TaDiCodec",
        llm_path="amphion/TaDiCodec-TTS-AR-Qwen2.5-3B",
        device=device,
    )

    print("✅ モデル読み込み完了")
    print("🎙️ 音声を生成中...")

    # Set paths relative to script location
    prompt_audio_path = os.path.join(script_dir, "test_audio", "trump_0.wav")
    output_path = os.path.join(script_dir, "output.wav")

    # Generate speech
    audio = pipeline(
        text=text,
        prompt_text="In short, we embarked on a mission to make America great again, for all Americans.",
        prompt_speech_path=prompt_audio_path,
    )

    # Save audio
    sf.write(output_path, audio, 24000)

    print(f"✅ 完了！音声ファイルを保存しました: {output_path}")


if __name__ == "__main__":
    main()
