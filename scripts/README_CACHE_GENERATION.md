# キャッシュ生成スクリプトの使用ガイド

## 概要

`create_dataset_cache.py` は、Emilia形式の音声データセットから学習用キャッシュファイルを生成するスクリプトです。

## 機能

### 生成されるキャッシュファイル

1. **`wav_paths_cache.pkl`**
   - WAVファイルの相対パスリスト
   - 例: `['speaker1/audio_0.wav', 'speaker1/audio_1.wav', ...]`

2. **`duration_cache.pkl`**
   - 各音声の長さ（秒）リスト
   - 例: `[2.5, 1.8, 3.2, ...]`

3. **`bpe_token_count_cache.pkl`**
   - 各テキストのBPEトークン数リスト
   - 例: `[15, 10, 20, ...]`

4. **`json_paths_cache.pkl`** (オプション)
   - 各音声のメタデータ（text, duration, language）
   - 例: `[{'text': '...', 'duration': 2.5, 'language': 'ja'}, ...]`

## 使用方法

### 基本的な使用方法

```bash
# 仮想環境をアクティベート
source .venv/Scripts/activate  # または .venv/Scripts/activate.ps1

# キャッシュ生成
python scripts/create_dataset_cache.py \
  --data_dir ./data/japanese \
  --cache_dir ./cache/japanese \
  --tokenizer_path ./ckpt/TaDiCodec_Japanese_Full/text_tokenizer
```

### カスタム設定

```bash
# 音声の長さでフィルタリング（2秒〜30秒）
python scripts/create_dataset_cache.py \
  --data_dir ./data/japanese \
  --cache_dir ./cache/japanese \
  --tokenizer_path ./ckpt/TaDiCodec_Japanese_Full/text_tokenizer \
  --min_duration 2.0 \
  --max_duration 30.0

# JSONメタデータキャッシュを無効化
python scripts/create_dataset_cache.py \
  --data_dir ./data/japanese \
  --cache_dir ./cache/japanese \
  --tokenizer_path ./ckpt/TaDiCodec_Japanese_Full/text_tokenizer \
  --no_json_cache
```

### コマンドライン引数

| 引数 | 説明 | デフォルト値 | 必須 |
|------|------|------------|------|
| `--data_dir` | Emilia形式データのルートディレクトリ | - | ✅ |
| `--cache_dir` | キャッシュ保存先ディレクトリ | - | ✅ |
| `--tokenizer_path` | HuggingFace Transformersトークナイザーのパス | `./ckpt/TaDiCodec_Japanese_Full/text_tokenizer` | ❌ |
| `--min_duration` | 最小音声長（秒） | `1.0` | ❌ |
| `--max_duration` | 最大音声長（秒） | `40.0` | ❌ |
| `--sample_rate` | サンプリングレート | `24000` | ❌ |
| `--save_json_cache` | JSONメタデータもキャッシュ | `True` | ❌ |
| `--no_json_cache` | JSONメタデータキャッシュを無効化 | - | ❌ |

## データ構造

### 入力（Emilia形式）

```
data/japanese/
├── speaker1/
│   ├── audio_0.wav
│   ├── audio_1.wav
│   ├── audio_2.wav
│   └── audio.json
├── speaker2/
│   ├── audio_0.wav
│   ├── audio_1.wav
│   └── audio.json
└── ...
```

### audio.json フォーマット

```json
{
  "0": {
    "text": "東京の天気は晴れです。",
    "duration": 2.5,
    "language": "ja"
  },
  "1": {
    "text": "こんにちは、元気ですか？",
    "duration": 1.8,
    "language": "ja"
  }
}
```

### 出力（キャッシュファイル）

```
cache/japanese/
├── wav_paths_cache.pkl
├── duration_cache.pkl
├── bpe_token_count_cache.pkl
└── json_paths_cache.pkl
```

## 実行例

### ステップ1: データの準備

```bash
# Emilia形式のデータを準備（例: JVS/JSUTから変換）
# データ構造は上記の「入力（Emilia形式）」を参照
```

### ステップ2: トークナイザーの準備

```bash
# 日本語拡張トークナイザーを生成（まだ生成していない場合）
python scripts/create_japanese_tokenizer_full.py
```

### ステップ3: キャッシュ生成

```bash
# キャッシュを生成
python scripts/create_dataset_cache.py \
  --data_dir ./data/japanese \
  --cache_dir ./cache/japanese \
  --tokenizer_path ./ckpt/TaDiCodec_Japanese_Full/text_tokenizer
```

### ステップ4: 学習設定ファイルの更新

生成したキャッシュを使用するように設定ファイルを更新：

```json
{
  "preprocess": {
    "mnt_path": "./data/japanese",
    "cache_folder": "./cache/japanese",
    "use_json_path_cache": true,
    "tokenizer_path": "./ckpt/TaDiCodec_Japanese_Full/text_tokenizer"
  }
}
```

### ステップ5: 学習開始

```bash
python bins/tts/train.py \
  --config egs/tts/TaDiCodec/tadicodec_japanese_finetune.json \
  --exp_name TaDiCodec_Japanese_Finetune
```

## 出力例

```
Scanning data directory: ./data/japanese
Found 100 speaker directories

Processing speaker directories...
100%|████████████████████| 100/100 [00:05<00:00, 20.00it/s]

Loading tokenizer from ./ckpt/TaDiCodec_Japanese_Full/text_tokenizer...
✓ Tokenizer loaded successfully

Tokenizing texts...
100%|████████████████████| 10000/10000 [00:30<00:00, 333.33it/s]

Saving cache files to ./cache/japanese...
✓ wav_paths_cache.pkl saved (10000 items)
✓ duration_cache.pkl saved (10000 items)
✓ bpe_token_count_cache.pkl saved (10000 items)
✓ json_paths_cache.pkl saved (10000 items)

================================================================================
Summary:
  Total audio files processed: 10000
  Filtered by duration: 150
  Average duration: 5.2 seconds
  Average token count: 25.3 tokens
  Total duration: 14.4 hours
  Min duration: 1.00 seconds
  Max duration: 39.98 seconds
  Min token count: 5
  Max token count: 150
================================================================================

Cache generation completed successfully!
Cache files saved to: ./cache/japanese

You can now use these cache files for training with:
  "mnt_path": "./data/japanese",
  "cache_folder": "./cache/japanese",
  "use_json_path_cache": true,
```

## トラブルシューティング

### エラー: "No speaker directories found"

**原因:** データディレクトリに `audio.json` を持つディレクトリが見つからない

**解決策:**
```bash
# データ構造を確認
ls -R ./data/japanese

# 正しい構造:
# data/japanese/speaker1/audio.json
# data/japanese/speaker1/audio_0.wav
# data/japanese/speaker1/audio_1.wav
```

### エラー: "Failed to load tokenizer"

**原因:** トークナイザーのパスが間違っているか、トークナイザーが初期化されていない

**解決策:**
```bash
# トークナイザーを生成
python scripts/create_japanese_tokenizer_full.py

# トークナイザーの存在確認
ls -la ./ckpt/TaDiCodec_Japanese_Full/text_tokenizer/
```

### エラー: "No valid audio files found"

**原因:** すべての音声ファイルがdurationフィルタで除外された

**解決策:**
```bash
# フィルタ範囲を広げる
python scripts/create_dataset_cache.py \
  --data_dir ./data/japanese \
  --cache_dir ./cache/japanese \
  --min_duration 0.5 \
  --max_duration 60.0
```

### 警告: "WAV file not found"

**原因:** `audio.json` に記載されているWAVファイルが存在しない

**解決策:**
- `audio.json` の内容を確認
- WAVファイル名が `audio_0.wav`, `audio_1.wav` 形式になっているか確認
- インデックスが連続しているか確認

## 高度な使用方法

### 複数のデータセットを統合

```bash
# JVSデータのキャッシュ生成
python scripts/create_dataset_cache.py \
  --data_dir ./data/jvs \
  --cache_dir ./cache/jvs \
  --tokenizer_path ./ckpt/TaDiCodec_Japanese_Full/text_tokenizer

# JSUTデータのキャッシュ生成
python scripts/create_dataset_cache.py \
  --data_dir ./data/jsut \
  --cache_dir ./cache/jsut \
  --tokenizer_path ./ckpt/TaDiCodec_Japanese_Full/text_tokenizer

# 学習時に両方を使用（設定ファイルで指定）
```

### キャッシュの検証

```python
import pickle

# キャッシュファイルの読み込み
with open('./cache/japanese/wav_paths_cache.pkl', 'rb') as f:
    wav_paths = pickle.load(f)

with open('./cache/japanese/duration_cache.pkl', 'rb') as f:
    durations = pickle.load(f)

with open('./cache/japanese/bpe_token_count_cache.pkl', 'rb') as f:
    token_counts = pickle.load(f)

print(f"Total files: {len(wav_paths)}")
print(f"First file: {wav_paths[0]}")
print(f"Duration: {durations[0]} seconds")
print(f"Token count: {token_counts[0]}")
```

## 関連ファイル

- **学習スクリプト**: `bins/tts/train.py`
- **データセットクラス**: `models/base/base_dataset.py`
- **日本語データセット**: `models/tts/tadicodec/tadicodec_dataset_japanese.py`
- **設定ファイル**: `egs/tts/TaDiCodec/tadicodec_japanese_finetune.json`

## 参考資料

- [JAPANESE_TRAINING_GUIDE.md](../JAPANESE_TRAINING_GUIDE.md) - 日本語学習の詳細ガイド
- [JAPANESE_FINETUNING_ROADMAP.md](../JAPANESE_FINETUNING_ROADMAP.md) - ファインチューニングのロードマップ
- [CLAUDE.md](../CLAUDE.md) - TaDiCodecの技術詳細
