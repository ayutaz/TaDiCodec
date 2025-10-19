# 🇯🇵 TaDiCodec 日本語ファインチューニング実装ロードマップ

## 📋 目次

1. [プロジェクト概要](#プロジェクト概要)
2. [現在の状況分析](#現在の状況分析)
3. [実装すべき課題](#実装すべき課題)
4. [実装優先順位](#実装優先順位)
5. [各タスクの詳細](#各タスクの詳細)
6. [技術的な課題と解決策](#技術的な課題と解決策)
7. [成功基準](#成功基準)

---

## 🎯 プロジェクト概要

### 目標

事前学習済みTaDiCodecモデルを日本語データでファインチューニングし、以下を実現する：

- ✅ 日本語音素の正確な表現
- ✅ 自然な韻律・アクセント
- ✅ 話者の声質保持
- ✅ 感情表現の向上

### スコープ

**実装対象:**
- TaDiCodecコーデックのファインチューニング
- 日本語データセット準備ツール
- 学習設定ファイル
- 評価スクリプト

**実装対象外（将来的に）:**
- TTSモデル（AR/MGM）のファインチューニング
- 新しいコーデックアーキテクチャの開発

### 前提条件

```yaml
GPU: NVIDIA RTX 4090 24GB以上
データ: 10-50時間の日本語音声データ
学習時間: 数時間〜数日
Python: 3.10
PyTorch: 2.8.0+cu128
```

---

## 📊 現在の状況分析

### ✅ 既に完成しているもの

#### 1. **学習コードの基盤**

```
✅ bins/tts/train.py                          # メイン学習スクリプト
✅ models/tts/tadicodec/tadicodec_trainer.py  # トレーナー実装
✅ models/tts/tadicodec/tadicodec_dataset.py  # データセット実装
✅ models/base/base_dataset.py                # 基底データセットクラス
✅ egs/tts/TaDiCodec/*.json                   # 設定ファイル（Emilia用）
```

**確認済み機能:**
- Accelerateベースの分散学習対応
- 動的バッチサイズ
- チェックポイント管理
- TensorBoard統合
- 学習再開・ファインチューニング対応

#### 2. **推論パイプライン**

```
✅ models/tts/tadicodec/inference_tadicodec.py  # 推論パイプライン
✅ Hugging Faceからの自動ダウンロード
✅ テキスト認識型デコーディング
✅ Binary Spherical Quantization (BSQ)
```

#### 3. **ドキュメント**

```
✅ JAPANESE_TRAINING_GUIDE.md        # 日本語学習ガイド
✅ CLAUDE.md                         # 技術詳細
✅ use_examples/                     # 使用例・テストスクリプト
```

### 🚧 実装が必要なもの

#### 1. **日本語データセット対応**

| 項目 | ステータス | 優先度 |
|------|----------|--------|
| データセット変換スクリプト | ❌ 未実装 | 🔴 高 |
| キャッシュ生成スクリプト | ❌ 未実装 | 🔴 高 |
| データ検証ツール | ❌ 未実装 | 🟡 中 |
| データ拡張（日本語特化） | ❌ 未実装 | 🟡 中 |

#### 2. **設定ファイル**

| 項目 | ステータス | 優先度 |
|------|----------|--------|
| 日本語学習用設定 | ❌ 未実装 | 🔴 高 |
| ファインチューニング設定 | ❌ 未実装 | 🔴 高 |
| ハイパーパラメータ最適化 | ❌ 未実装 | 🟢 低 |

#### 3. **日本語最適化**

| 項目 | ステータス | 優先度 |
|------|----------|--------|
| 日本語音素トークナイザー | ❌ 未実装 | 🟡 中 |
| 韻律損失の調整 | ❌ 未実装 | 🟡 中 |
| アクセント損失の追加 | ❌ 未実装 | 🟢 低 |

#### 4. **評価・検証**

| 項目 | ステータス | 優先度 |
|------|----------|--------|
| 日本語評価スクリプト | ❌ 未実装 | 🟡 中 |
| WER計算（日本語対応） | ❌ 未実装 | 🟡 中 |
| MOS評価フレームワーク | ❌ 未実装 | 🟢 低 |

---

## 🎯 実装すべき課題

### Phase 1: 基盤整備（優先度：高）

#### Task 1.1: データセット変換スクリプト

**目的:** 日本語データセットをEmilia形式に変換

**ファイル:** `scripts/prepare_japanese_dataset.py`

**要件:**
- JVS/JSUT/ReazonSpeech形式からの変換
- メタデータ（text, duration, language）の生成
- 音声ファイルのリサンプリング（24kHz）
- 長さフィルタリング（1-40秒）

**入力:**
```
input_data/
├── jvs/
│   ├── jvs001/
│   │   ├── parallel100/
│   │   │   └── wav24kHz16bit/
│   │   │       ├── VOICEACTRESS100_001.wav
│   │   │       └── ...
│   │   └── transcript_utf8.txt
│   └── ...
```

**出力:**
```
data/japanese/
├── jvs001/
│   ├── audio_0.wav
│   ├── audio_1.wav
│   └── audio.json
└── ...
```

**実装優先度:** 🔴 最優先

---

#### Task 1.2: キャッシュ生成スクリプト

**目的:** 学習時のデータ読み込みを高速化

**ファイル:** `scripts/create_dataset_cache.py`

**要件:**
- WAVファイルパスのキャッシュ
- 音声の長さ（duration）のキャッシュ
- テキストのトークン数のキャッシュ
- 高速pickle/numpy形式での保存

**生成ファイル:**
```
data/japanese_cache/
├── wav_paths_cache.pkl          # WAVファイルパスリスト
├── duration_cache.pkl           # 音声長リスト
├── bpe_token_count_cache.pkl    # トークン数リスト
└── json_paths_cache.pkl         # JSONパスリスト（オプション）
```

**実装優先度:** 🔴 最優先

---

#### Task 1.3: 日本語学習用設定ファイル

**目的:** ファインチューニングに最適化された設定

**ファイル:** `egs/tts/TaDiCodec/tadicodec_japanese_finetune.json`

**要件:**
- 小規模データ（10-50時間）に最適化
- 学習率・バッチサイズの調整
- チェックポイント頻度の設定
- 既存モデルからの継続学習設定

**重要パラメータ:**
```json
{
  "preprocess": {
    "mnt_path": "./data/japanese",
    "cache_folder": "./data/japanese_cache",
    "min_dur": 1.0,
    "max_dur": 40.0
  },
  "train": {
    "lr": 5e-5,                    // 低めの学習率
    "num_train_steps": 100000,     // 少なめのステップ
    "batch_size": 4,               // GPU次第
    "gradient_accumulation_step": 4,
    "save_checkpoints_steps": 2000
  }
}
```

**実装優先度:** 🔴 最優先

---

### Phase 2: 学習実行（優先度：高）

#### Task 2.1: 学習スクリプトの検証

**目的:** 既存の学習コードが正しく動作するか確認

**確認項目:**
- [ ] `bins/tts/train.py` が実行可能か
- [ ] データローダーが日本語データを正しく読み込むか
- [ ] チェックポイントの保存が機能するか
- [ ] TensorBoardログが正しく出力されるか
- [ ] 学習再開機能が動作するか

**検証コマンド:**
```bash
# 小規模データで1エポックだけ実行してテスト
python bins/tts/train.py \
  --config egs/tts/TaDiCodec/tadicodec_japanese_finetune.json \
  --exp_name test_japanese_finetune \
  --log_level debug
```

**実装優先度:** 🔴 最優先

---

#### Task 2.2: 事前学習済みモデルの準備

**目的:** ファインチューニングの起点となるモデルを取得

**手順:**

1. **自動ダウンロードで取得:**
```python
from models.tts.tadicodec.inference_tadicodec import TaDiCodecPipline
pipe = TaDiCodecPipline.from_pretrained('amphion/TaDiCodec')
```

2. **キャッシュ場所の確認:**
```bash
# Hugging Faceのキャッシュ
ls ~/.cache/huggingface/hub/models--amphion--TaDiCodec/snapshots/
```

3. **チェックポイントの特定:**
```bash
# 学習再開に必要なファイル
- checkpoint.pth         # モデルの重み
- config.json           # モデル設定
- text_tokenizer/       # テキストトークナイザー
```

**実装優先度:** 🔴 最優先

---

#### Task 2.3: ファインチューニングの実行

**目的:** 日本語データでモデルをファインチューニング

**実行コマンド:**
```bash
# 仮想環境アクティベート
source .venv/Scripts/activate

# ファインチューニング開始
python bins/tts/train.py \
  --config egs/tts/TaDiCodec/tadicodec_japanese_finetune.json \
  --exp_name japanese_tadicodec_finetune_v1 \
  --resume \
  --resume_type finetune \
  --checkpoint_path ~/.cache/huggingface/hub/models--amphion--TaDiCodec/snapshots/<hash>/checkpoint.pth
```

**学習の監視:**
```bash
# TensorBoardで監視
tensorboard --logdir ./logs/japanese_tadicodec_finetune_v1

# 確認すべき指標
# - loss (下がっているか)
# - learning_rate (スケジュール通りか)
# - 生成音声サンプル (定期的に聴く)
```

**実装優先度:** 🔴 最優先

---

### Phase 3: 評価・改善（優先度：中）

#### Task 3.1: 日本語評価スクリプト

**目的:** ファインチューニング後のモデルを評価

**ファイル:** `eval/evaluate_japanese.py`

**評価指標:**
- WER（単語誤り率）: WhisperまたはESPnetで計算
- Speaker Similarity: WavLMで計算
- 音質（MOS）: UTMOS予測モデルで推定
- 韻律自然さ: 主観評価（将来的に自動化）

**使用方法:**
```bash
python eval/evaluate_japanese.py \
  --model_path ./logs/japanese_tadicodec_finetune_v1/checkpoint/best.pth \
  --test_data ./data/japanese_test \
  --output_dir ./eval_results
```

**実装優先度:** 🟡 中

---

#### Task 3.2: A/Bテスト実施

**目的:** ベースラインとファインチューニング後のモデルを比較

**比較項目:**
- 音素精度
- 韻律自然さ
- アクセント正確性
- 話者類似度
- 感情表現

**実装優先度:** 🟡 中

---

### Phase 4: 最適化（優先度：低）

#### Task 4.1: 日本語音素トークナイザーの最適化

**目的:** 日本語音素の表現を改善

**ファイル:** `scripts/create_japanese_tokenizer.py`

**要件:**
- pyopenjtalk-plusで日本語音素抽出
- 既存トークナイザーに日本語音素を追加
- 新しいトークナイザーで学習データを再処理

**実装優先度:** 🟢 低（まず既存トークナイザーで試す）

---

#### Task 4.2: データ拡張の実装

**目的:** 小規模データからの汎化性能向上

**ファイル:** `scripts/japanese_data_augmentation.py`

**拡張手法:**
- ピッチシフト（±2半音）
- 速度変化（0.9-1.1倍）
- ボリューム正規化
- ノイズ追加（SNR 20-40dB）

**実装優先度:** 🟢 低

---

## 🔍 技術的な課題と解決策

### 課題1: チェックポイントのロード

**問題:**
```python
# Hugging Faceのモデル構造が学習コードと異なる可能性
```

**解決策:**
```python
# models/tts/tadicodec/tadicodec_trainer.py 内で
# チェックポイントをロードする際に、キーの不一致を処理

def load_checkpoint_for_finetune(self, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # キー名の変換（必要に応じて）
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # 不一致するキーを無視してロード
    self.model.load_state_dict(state_dict, strict=False)
```

---

### 課題2: テキストトークナイザーの互換性

**問題:**
```
事前学習モデルとファインチューニングで
トークナイザーのバージョンが異なる可能性
```

**解決策:**
```python
# 事前学習モデルと同じトークナイザーを使用
tokenizer_path = "~/.cache/huggingface/hub/models--amphion--TaDiCodec/.../text_tokenizer"

# 設定ファイルで指定
{
  "preprocess": {
    "tokenizer_path": "<上記パス>"
  }
}
```

---

### 課題3: GPUメモリ不足

**問題:**
```
RTX 4090 (24GB) でも大きなバッチサイズは扱えない
```

**解決策:**
```json
{
  "train": {
    "batch_size": 2,                    // 小さく
    "gradient_accumulation_step": 8,    // 勾配累積で実効バッチサイズ増加
    "max_tokens": 5000,                 // トークン数制限
    "use_dynamic_batchsize": true       // 動的バッチサイズ
  }
}
```

---

### 課題4: データ形式の不一致

**問題:**
```
JVS/JSUTのデータ形式がEmiliaと異なる
```

**解決策:**
```python
# scripts/prepare_japanese_dataset.py で変換
# JVS形式:
#   jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav
#   jvs001/transcript_utf8.txt
#
# → Emilia形式:
#   jvs001/audio_0.wav
#   jvs001/audio.json
```

---

### 課題5: 学習の収束確認

**問題:**
```
いつ学習を停止すべきか分からない
```

**解決策:**
```python
# 検証データでの損失を監視
# - 検証損失が10エポック改善しなければ停止（Early Stopping）
# - 定期的に音声サンプルを生成して聴く

# TensorBoardで確認:
# 1. train_loss が下がっているか
# 2. valid_loss が下がっているか（過学習していないか）
# 3. 音声サンプルの品質が向上しているか
```

---

## 📈 成功基準

### 定量的指標

| 指標 | ベースライン | 目標 | 測定方法 |
|------|-------------|------|----------|
| **WER** | 8-12% | < 5% | Whisper large-v3 |
| **MOS** | 3.8 | > 4.2 | UTMOS予測 |
| **Speaker SIM** | 0.75 | > 0.85 | WavLM cosine similarity |
| **韻律自然さ** | 3.5 | > 4.0 | 主観評価 |
| **アクセント正確性** | 70% | > 90% | 専門家評価 |

### 定性的指標

**改善すべき点:**
- ✅ 「ん」と「N」の区別が明確
- ✅ 長音「ー」が滑らか
- ✅ アクセント位置が正確
- ✅ イントネーションが自然
- ✅ 話者の声質が保持される

**テストケース:**

```python
test_cases = [
    "東京の天気は晴れです。",           # アクセント
    "コンピューター",                  # 長音
    "カップ、バッグ、ベッド",          # 促音
    "新聞、品質、本質",                # ん
    "おはようございます。元気ですか？", # 自然な会話
]
```

---

## 📅 実装スケジュール（推奨）

### Week 1: 基盤整備

- [ ] Task 1.1: データセット変換スクリプト実装
- [ ] Task 1.2: キャッシュ生成スクリプト実装
- [ ] Task 1.3: 設定ファイル作成
- [ ] データセット準備（JVS 10-30時間）

### Week 2: 学習実行

- [ ] Task 2.1: 学習スクリプト検証
- [ ] Task 2.2: 事前学習済みモデル準備
- [ ] Task 2.3: ファインチューニング実行（3-7日）
- [ ] 学習の監視・調整

### Week 3: 評価・改善

- [ ] Task 3.1: 評価スクリプト実装
- [ ] Task 3.2: A/Bテスト実施
- [ ] 結果分析・レポート作成

### Week 4: 最適化（オプション）

- [ ] Task 4.1: トークナイザー最適化
- [ ] Task 4.2: データ拡張実装
- [ ] 再学習・再評価

---

## 🚀 次のステップ

### 今すぐ実行すべきこと

1. **Task 1.1: データセット変換スクリプトの実装**
   ```bash
   # ファイル作成
   touch scripts/prepare_japanese_dataset.py
   ```

2. **Task 1.2: キャッシュ生成スクリプトの実装**
   ```bash
   touch scripts/create_dataset_cache.py
   ```

3. **Task 1.3: 設定ファイルの作成**
   ```bash
   touch egs/tts/TaDiCodec/tadicodec_japanese_finetune.json
   ```

4. **データセットのダウンロード**
   - JVS: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
   - JSUT: https://sites.google.com/site/shinnosuketakamichi/publication/jsut

---

## 📚 参考資料

- [JAPANESE_TRAINING_GUIDE.md](./JAPANESE_TRAINING_GUIDE.md) - 日本語学習の詳細ガイド
- [CLAUDE.md](./CLAUDE.md) - TaDiCodecの技術詳細
- [TaDiCodec論文](https://arxiv.org/abs/2508.16790)
- [bins/tts/train.py](./bins/tts/train.py) - メイン学習スクリプト

---

## 📝 まとめ

### 実装が必要な最優先タスク（Phase 1）

1. 🔴 **データセット変換スクリプト** (`scripts/prepare_japanese_dataset.py`)
2. 🔴 **キャッシュ生成スクリプト** (`scripts/create_dataset_cache.py`)
3. 🔴 **日本語学習用設定ファイル** (`egs/tts/TaDiCodec/tadicodec_japanese_finetune.json`)
4. 🔴 **学習スクリプトの検証**
5. 🔴 **ファインチューニングの実行**

### 期待される成果

- WER: 60%改善（8-12% → 3-5%）
- MOS: 13%向上（3.8 → 4.3）
- Speaker SIM: 17%向上（0.75 → 0.88）
- 自然な日本語音声合成の実現

### 必要リソース

- GPU: NVIDIA RTX 4090 24GB
- データ: JVS 10-30時間
- 学習時間: 数時間〜数日
- 開発時間: 2-4週間

---

**🚀 実装を開始する準備ができました！**

次のステップ: Phase 1の最優先タスクから実装を開始しましょう。
