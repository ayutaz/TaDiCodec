import torch
import soundfile as sf

# ref parent directory
import sys
sys.path.append('../')

from models.tts.llm_tts.inference_llm_tts import TTSInferencePipeline


def test_high_quality_japanese():
    """Test High-Quality AR-TTS with Qwen2.5-3B for Japanese"""
    print("\n" + "="*70)
    print("🎯 High-Quality Japanese TTS with Qwen2.5-3B (3 Billion Parameters)")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n📥 Loading Qwen2.5-3B model (this may take a few minutes)...")

    # Create high-quality AR-TTS pipeline with 3B model
    pipeline = TTSInferencePipeline.from_pretrained(
        tadicodec_path="amphion/TaDiCodec",
        llm_path="amphion/TaDiCodec-TTS-AR-Qwen2.5-3B",  # 3B model for higher quality
        device=device,
    )

    print("✅ Model loaded successfully!")

    # Japanese text examples with various styles
    japanese_examples = [
        {
            "text": "こんにちは、私は高精度な音声合成システムです。",
            "description": "基本的な挨拶"
        },
        {
            "text": "今日は良い天気ですね。公園を散歩するのに最適な日です。青い空と温かい日差しが気持ちいいですね。",
            "description": "日常会話（長文）"
        },
        {
            "text": "昔々、あるところにおじいさんとおばあさんが住んでいました。おじいさんは山へ柴刈りに、おばあさんは川へ洗濯に行きました。",
            "description": "物語調"
        },
        {
            "text": "人工知能技術の発展により、音声合成の品質は飛躍的に向上しました。深層学習を用いたモデルは、自然で表現豊かな音声を生成できます。",
            "description": "技術的な説明"
        },
        {
            "text": "ありがとうございます。あなたの助けに心から感謝しています。本当に嬉しいです。",
            "description": "感謝の表現"
        },
        {
            "text": "春は桜、夏は花火、秋は紅葉、冬は雪景色。日本の四季はそれぞれ美しい風景を見せてくれます。",
            "description": "詩的な表現"
        }
    ]

    # Use English prompt (cross-lingual TTS)
    prompt_text = "In short, we embarked on a mission to make America great again, for all Americans."
    prompt_speech_path = "./test_audio/trump_0.wav"

    print(f"\n📢 Generating {len(japanese_examples)} high-quality Japanese samples...")
    print("-" * 70)

    for i, example in enumerate(japanese_examples, 1):
        text = example["text"]
        description = example["description"]

        print(f"\n[{i}/{len(japanese_examples)}] {description}")
        print(f"テキスト: {text}")

        # Generate audio with high quality settings
        audio = pipeline(
            text=text,
            prompt_text=prompt_text,
            prompt_speech_path=prompt_speech_path,
            # High quality inference settings
            top_k=50,           # Default: 50
            top_p=0.98,         # Default: 0.98
            temperature=1.0,    # Default: 1.0
        )

        # Save audio
        output_path = f"./japanese_hq_{i}.wav"
        sf.write(output_path, audio, 24000)
        print(f"✅ Saved to: {output_path}")

    print("\n" + "="*70)
    print("🎉 All high-quality Japanese samples generated!")
    print("="*70)
    print("\n📁 Generated files:")
    for i, example in enumerate(japanese_examples, 1):
        print(f"  - japanese_hq_{i}.wav: {example['description']}")


if __name__ == "__main__":
    test_high_quality_japanese()
