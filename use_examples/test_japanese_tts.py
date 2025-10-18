import torch
import soundfile as sf

# ref parent directory
import sys
sys.path.append('../')

from models.tts.llm_tts.inference_llm_tts import TTSInferencePipeline
from models.tts.llm_tts.inference_mgm_tts import MGMInferencePipeline


def test_ar_tts_japanese():
    """Test Autoregressive TTS with Japanese text"""
    print("\n" + "="*60)
    print("Testing AR-TTS with Japanese")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create AR-TTS pipeline
    pipeline = TTSInferencePipeline.from_pretrained(
        tadicodec_path="amphion/TaDiCodec",
        llm_path="amphion/TaDiCodec-TTS-AR-Qwen2.5-0.5B",
        device=device,
    )

    # Japanese text examples
    japanese_texts = [
        "こんにちは、私は音声合成システムです。",
        "今日は良い天気ですね。公園を散歩するのに最適な日です。",
        "昔々、あるところにおじいさんとおばあさんが住んでいました。"
    ]

    # Use English prompt (cross-lingual TTS)
    prompt_text = "In short, we embarked on a mission to make America great again, for all Americans."
    prompt_speech_path = "./test_audio/trump_0.wav"

    for i, text in enumerate(japanese_texts, 1):
        print(f"\n[{i}/3] Generating: {text}")

        # Generate audio
        audio = pipeline(
            text=text,
            prompt_text=prompt_text,
            prompt_speech_path=prompt_speech_path,
        )

        # Save audio
        output_path = f"./japanese_ar_tts_{i}.wav"
        sf.write(output_path, audio, 24000)
        print(f"✅ Saved to: {output_path}")


def test_mgm_tts_japanese():
    """Test MGM TTS with Japanese text"""
    print("\n" + "="*60)
    print("Testing MGM-TTS with Japanese")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create MGM pipeline
    pipeline = MGMInferencePipeline.from_pretrained(
        tadicodec_path="amphion/TaDiCodec",
        mgm_path="amphion/TaDiCodec-TTS-MGM",
        device=device,
    )

    # Japanese text examples
    japanese_texts = [
        "こんにちは、私は音声合成システムです。",
        "今日は良い天気ですね。公園を散歩するのに最適な日です。",
    ]

    # Use English prompt (cross-lingual TTS)
    prompt_text = "In short, we embarked on a mission to make America great again, for all Americans."
    prompt_speech_path = "./test_audio/trump_0.wav"

    for i, text in enumerate(japanese_texts, 1):
        print(f"\n[{i}/2] Generating: {text}")

        # Generate audio
        audio = pipeline(
            text=text,
            prompt_text=prompt_text,
            prompt_speech_path=prompt_speech_path,
        )

        # Save audio
        output_path = f"./japanese_mgm_tts_{i}.wav"
        sf.write(output_path, audio, 24000)
        print(f"✅ Saved to: {output_path}")


def test_mixed_language():
    """Test with mixed Japanese-English text"""
    print("\n" + "="*60)
    print("Testing Mixed Language (Japanese + English)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create MGM pipeline (faster than AR)
    pipeline = MGMInferencePipeline.from_pretrained(
        tadicodec_path="amphion/TaDiCodec",
        mgm_path="amphion/TaDiCodec-TTS-MGM",
        device=device,
    )

    # Mixed language text
    mixed_text = "これは TaDiCodec という speech synthesis system です。とても powerful で、multilingual に対応しています。"

    prompt_text = "In short, we embarked on a mission to make America great again, for all Americans."
    prompt_speech_path = "./test_audio/trump_0.wav"

    print(f"\nGenerating: {mixed_text}")

    # Generate audio
    audio = pipeline(
        text=mixed_text,
        prompt_text=prompt_text,
        prompt_speech_path=prompt_speech_path,
    )

    # Save audio
    output_path = "./japanese_english_mixed.wav"
    sf.write(output_path, audio, 24000)
    print(f"✅ Saved to: {output_path}")


if __name__ == "__main__":
    print("🇯🇵 Japanese TTS Testing Suite 🇯🇵")

    # Test 1: Autoregressive TTS
    test_ar_tts_japanese()

    # Test 2: MGM TTS
    test_mgm_tts_japanese()

    # Test 3: Mixed Language
    test_mixed_language()

    print("\n" + "="*60)
    print("🎉 All Japanese TTS tests completed!")
    print("="*60)
