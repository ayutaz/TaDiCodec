# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリで作業する際のガイダンスを提供します。

## プロジェクト概要

**TaDiCodec** (Text-aware Diffusion Speech Tokenizer) は、音声言語モデリングのための拡散ベースの音声トークナイザーです（NeurIPS 2025採択）。PyTorchによる研究実装を提供します。

### 主な特徴

1. **超低ビットレート圧縮**: 24kHz音声を0.0875kbps（87.5bps）で圧縮
2. **テキスト認識型デコーディング**: テキスト情報を活用して音声品質を向上
3. **ゼロショットTTS**: 事前学習済みモデルによる音声合成

### リポジトリの内容

- TaDiCodecトークナイザーの実装
- ゼロショットTTSモデル（自己回帰型とMGM型）
- 学習インフラ（開発中）
- Hugging Faceからの自動ダウンロード機能付き事前学習済みモデル

## 技術詳細

### 1. 超低ビットレート (0.0875 kbps) の実現方法

TaDiCodecは以下の技術により業界最低レベルのビットレートを実現しています：

#### フレームレート 6.25Hz

```python
# 設定ファイル: egs/tts/TaDiCodec/tadicodec_6_25hz_16384_bsq.json
サンプリングレート: 24000 Hz
hop_size: 480 サンプル
→ メルスペクトログラムのフレームレート = 24000 / 480 = 50 Hz

down_sample_factor: 8
→ VQコードのフレームレート = 50 / 8 = 6.25 Hz
```

**実装箇所** (`models/tts/tadicodec/modeling_tadicodec.py:314-320`):
```python
# エンコーダー出力をダウンサンプリング
vq_emb_pre = vq_emb_pre.transpose(1, 2)
vq_emb_pre = F.interpolate(
    vq_emb_pre, size=T // self.down_sample_factor, mode="linear"
)
vq_emb_pre = vq_emb_pre.transpose(1, 2)
```

#### ビットレート計算

```
VQ次元: 14ビット (vq_emb_dim=14)
コードブックサイズ: 2^14 = 16384 個
フレームレート: 6.25 Hz

ビットレート = 14 bits/frame × 6.25 frames/second = 87.5 bps = 0.0875 kbps
```

#### Binary Spherical Quantization (BSQ)

**実装** (`models/codec/amphion_codec/quantize/bsq.py:150-367`):

BSQは従来のベクトル量子化とは異なる革新的な手法：

1. **2値化**: 各次元を{-1, 1}に量子化
   ```python
   # models/codec/amphion_codec/quantize/bsq.py:210-215
   zhat = torch.where(
       z > 0,
       torch.tensor(1, dtype=z.dtype, device=z.device),
       torch.tensor(-1, dtype=z.dtype, device=z.device),
   )
   return z + (zhat - z).detach()  # Straight-through estimator
   ```

2. **超球面上での量子化**: L2正規化により単位超球面上で量子化
   ```python
   # models/tts/tadicodec/modeling_tadicodec.py:324
   vq_emb_pre = F.normalize(vq_emb_pre, dim=-1)
   ```

3. **エントロピー損失**: コードブックの均等利用を促進
   ```python
   # Soft entropy loss (bsq.py:265-297)
   per_sample_entropy = ...  # 各サンプルのエントロピーを最大化
   codebook_entropy = ...    # コードブック全体のエントロピーを最大化
   entropy_penalty = gamma0 * per_sample_entropy - gamma * codebook_entropy
   ```

4. **コミットメント損失**: 量子化前後の表現を近づける
   ```python
   commit_loss = torch.mean(((zq.detach() - z) ** 2).sum(dim=-1))
   ```

### 2. テキスト認識機能の実装

TaDiCodecの特徴的な機能として、デコーダーにテキスト情報を条件として与えます。

#### テキスト埋め込み

**実装** (`models/tts/tadicodec/modeling_tadicodec.py:178-180`):
```python
if self.use_text_cond:
    self.text_emb = nn.Embedding(text_vocab_size, hidden_size)
    # 語彙サイズ: 32100、埋め込み次元: 1024
```

#### デコーダーでのテキスト条件付け

**フロー** (`modeling_tadicodec.py:352-362`):
```python
# 学習時
if self.use_text_cond:
    text_emb = self.text_emb(text_ids)  # テキストを埋め込みに変換
else:
    text_emb = None

# VQ埋め込みを主条件として使用
cond_emb = vq_emb_post
# Classifier-Free Guidanceのための条件ドロップアウト
if torch.rand(1) < self.cond_drop_p:  # cond_drop_p = 0.2
    cond_emb = torch.zeros_like(cond_emb)
```

**デコーダーへの入力** (`modeling_tadicodec.py:377-384`):
```python
flow_pred = self.decoder(
    x=xt,                      # ノイズを加えたメルスペクトログラム
    x_mask=x_mask,             # パディングマスク
    text_embedding=text_emb,   # テキスト埋め込み
    text_mask=text_mask,       # テキストのマスク
    cond=cond_emb,            # VQコード埋め込み
    diffusion_step=new_t,     # 拡散ステップ
)
```

#### Classifier-Free Guidance (CFG)

**推論時の実装** (`modeling_tadicodec.py:564-593`):
```python
# 条件付き予測
flow_pred = self.decoder(
    x=xt_input,
    text_embedding=text_emb,
    cond=cond_emb,
    diffusion_step=t,
)

# 無条件予測（テキストとコードをドロップ）
uncond_flow_pred = self.decoder(
    x=xt_input,
    text_embedding=None,                    # テキストなし
    cond=torch.zeros_like(cond_emb),       # コードなし
    diffusion_step=t,
)

# CFGで組み合わせ
flow_pred_cfg = uncond_flow_pred + cfg * (flow_pred - uncond_flow_pred)
# cfg=2.0 の場合、条件付き予測の影響を2倍に強調

# 再スケーリング（Stable Diffusion方式）
if rescale_cfg > 0:
    flow_pred_std = flow_pred.std()
    cfg_std = flow_pred_cfg.std()
    if cfg_std > 1e-6:
        rescale_flow_pred = flow_pred_cfg * (flow_pred_std / cfg_std)
        flow_pred = rescale_cfg * rescale_flow_pred + (1 - rescale_cfg) * flow_pred_cfg
```

#### Flow Matching拡散プロセス

**順方向拡散** (`modeling_tadicodec.py:277-280`):
```python
# Flow matching公式: xt = (1 - (1 - σ) * t) * z + t * x
# z ~ N(0, 1) はノイズ、x は元のメルスペクトログラム
# σ = 1e-5 は数値安定性のための小さな定数
xt = ((1 - (1 - self.sigma) * t) * z + t * x) * mask + x * (1 - mask)
```

**逆方向拡散** (`modeling_tadicodec.py:541-597`):
```python
# Euler法による反復ノイズ除去
h = 1.0 / n_timesteps  # ステップサイズ
for i in range(n_timesteps):
    t = (0 + (i + 0.5) * h) * torch.ones(z.shape[0])
    flow_pred = self.decoder(...)  # フロー予測
    dxt = flow_pred * h
    xt = xt + dxt  # オイラー法による更新
```

### 3. ゼロショットTTS の実装

TaDiCodecを使った2種類のTTSアプローチを提供：

#### A. 自己回帰型TTS (AR-TTS)

**実装** (`models/tts/llm_tts/inference_llm_tts.py`):

**処理フロー**:
```
1. プロンプト音声をTaDiCodecでエンコード → 音声コード列
2. LLM（Qwen2.5またはPhi-3）で自己回帰生成
   - 入力: "Please speak: <テキスト>" + プロンプト音声コード
   - 出力: ターゲット音声コード列
3. 生成されたコードをTaDiCodecでデコード → メルスペクトログラム
4. Vocosで波形に変換
```

**音声コードの表現** (`inference_llm_tts.py:58-68`):
```python
def tensor_to_audio_string(self, tensor):
    """音声コードを文字列形式に変換
    例: [123, 456, 789] → "<|start_of_audio|><|audio_123|><|audio_456|><|audio_789|>"
    """
    result = "<|start_of_audio|>"
    for value in values:
        result += f"<|audio_{value}|>"
    return result
```

**LLMによる生成** (`inference_llm_tts.py:211-225`):
```python
# プロンプト作成
prompt = gen_chat_prompt_for_tts(
    (prompt_text or "") + text,
    "phi-3" if "Phi" in self.llm_path else "qwen2",
) + self.tensor_to_audio_string(prompt_speech_code)

# 自己回帰生成
generate_ids = self.llm.generate(
    input_ids=torch.tensor(input_ids).unsqueeze(0).to(self.device),
    min_new_tokens=12,
    max_new_tokens=400,
    do_sample=True,
    top_k=top_k,        # デフォルト: 50
    top_p=top_p,        # デフォルト: 0.98
    temperature=temperature,  # デフォルト: 1.0
)
```

**デコード** (`inference_llm_tts.py:236-244`):
```python
# 生成されたコードを抽出
combine_speech_code = self.extract_audio_ids(output)
indices = torch.tensor(combine_speech_code).unsqueeze(0).long()

# TaDiCodecでデコード
text_token_ids = self.tadicodec.tokenize_text(text, prompt_text)
rec_mel = self.tadicodec.decode(
    indices=indices,
    text_token_ids=text_token_ids,  # テキスト条件
    prompt_mel=prompt_mel,           # プロンプトメル
    n_timesteps=n_timesteps,         # 拡散ステップ数（デフォルト25）
)
```

#### B. MGM型TTS (Masked Generative Model)

**実装** (`models/tts/llm_tts/inference_mgm_tts.py`):

MGMはマスク拡散により並列生成を実現（自己回帰より高速）：

**処理フロー**:
```
1. プロンプト音声をTaDiCodecでエンコード
2. テキストをトークナイズ
3. ターゲット長を推定（プロンプトとテキストの比率から）
4. MGMマスク拡散で並列生成
5. TaDiCodecでデコード
```

**ターゲット長の推定** (`inference_mgm_tts.py:248-259`):
```python
if target_len is None:
    # テキストの長さ比からターゲット音声長を推定
    prompt_text_len = len(prompt_text.encode("utf-8"))
    target_text_len = len(text.encode("utf-8"))
    prompt_speech_len = librosa.get_duration(filename=prompt_speech_path)
    target_speech_len = prompt_speech_len * target_text_len / prompt_text_len
    target_len = int(target_speech_len * frame_rate)  # frame_rate = 6.25 Hz
```

**MGM逆拡散** (`inference_mgm_tts.py:268-275`):
```python
generated_codes = self.mgm.reverse_diffusion(
    prompt=prompt_codes,           # プロンプトコード
    target_len=target_len,         # ターゲット長
    phone_id=text_token_ids,       # テキストトークン
    n_timesteps=n_timesteps_mgm,   # MGM拡散ステップ（デフォルト25）
    cfg=1.5,                       # CFG強度
    rescale_cfg=0.75,              # CFG再スケーリング
)
```

#### コードスイッチング対応

**例** (`use_examples/test_mgm_tts.py:23`):
```python
text = "但是 to those who 知道 her well, it was a 标志 of her unwavering 决心 and spirit."
# 中国語と英語が自然に混在
```

両方のTTSモデルは多言語トークナイザーを使用し、言語を切り替えながらの音声合成が可能です。

## 環境構築（Windows）

このPCにはUVがインストール済みです。以下の手順で環境を構築できます：

### ステップ1: Python環境のセットアップ

```bash
# Python 3.10のインストール
uv python install 3.10

# 仮想環境の作成
uv venv --python 3.10

# 仮想環境の有効化（PowerShell）
.\.venv\Scripts\activate.ps1
# または（bash/Git Bash）
# source .venv/Scripts/activate
```

### ステップ2: 基本パッケージのインストール

```bash
# ビルドツールと基本パッケージ
uv pip install setuptools wheel psutil packaging ninja numpy hf_xet
```

### ステップ3: PyTorch（CUDA版）のインストール

```bash
# CUDA 12.8対応のPyTorch 2.8.0をインストール
uv pip install torch==2.8.0 torchaudio --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128

# CPUのみの場合（GPUなし）:
# uv pip install torch==2.8.0 torchaudio
```

### ステップ4: コアパッケージのインストール

```bash
# TaDiCodecの主要な依存関係
uv pip install transformers==4.42.4 librosa huggingface_hub accelerate scipy json5 resampy tqdm tensorboard einops safetensors omegaconf

# 追加の必須パッケージ
uv pip install pyopenjtalk-plus  # 日本語対応（Windows用）
uv pip install pyworld ruamel.yaml six tensorboardX
```

### ステップ5: Flash Attentionのインストール（推奨）

Flash Attentionは推論速度を大幅に向上させます（2-3倍高速化）。

**方法1: ビルド済みホイールを使用（推奨）**

```bash
# wheelをダウンロード
curl -L -o flash_attn-2.7.4.post1-cu128-torch2.8.0-cp310-cp310-win_amd64.whl "https://huggingface.co/kim512/flash_attn-2.7.4.post1/resolve/main/flash_attn-2.7.4.post1-cu128-torch2.8.0-cp310-cp310-win_amd64.whl"

# wheelファイル名を標準形式にリネーム（uvの制約のため）
cp flash_attn-2.7.4.post1-cu128-torch2.8.0-cp310-cp310-win_amd64.whl flash_attn-2.7.4.post1-cp310-cp310-win_amd64.whl

# インストール（UV_SKIP_WHEEL_FILENAME_CHECKを設定）
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install flash_attn-2.7.4.post1-cp310-cp310-win_amd64.whl

# wheelファイルをクリーンアップ
rm flash_attn*.whl
```

**方法2: Flash Attentionなしでも動作**

Flash Attentionのインストールに失敗しても、TaDiCodecは自動的に標準Attentionにフォールバックします。品質は同じですが、速度が遅くなります（2-3倍）。

### ステップ6: 環境のテスト

```bash
# GPU環境のテスト
.venv/Scripts/python.exe test_gpu_setup.py

# 期待される出力:
# GPU name: NVIDIA GeForce RTX 4070 Ti SUPER (または他のGPU)
# CUDA available: True
# CUDA version: 12.8
# Flash Attention version: 2.7.4.post1 (インストールした場合)
```

### トラブルシューティング

**問題1: `pyopenjtalk`のビルドエラー**
- 解決策: `pyopenjtalk-plus`を使用（上記手順に含まれています）

**問題2: Flash Attentionのインストール失敗**
- wheelファイル名を標準形式にリネームしたか確認
- `UV_SKIP_WHEEL_FILENAME_CHECK=1`を設定したか確認
- 失敗してもTaDiCodecは動作します（速度が遅くなるだけ）

**問題3: CUDA/GPU認識されない**
- NVIDIAドライバーが最新か確認
- CUDA 12.8対応のドライバー（バージョン525以上）が必要

### インストール完了後の確認

```bash
# パッケージのインポートテスト
.venv/Scripts/python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# TaDiCodecモジュールのインポートテスト
.venv/Scripts/python.exe -c "from models.tts.tadicodec.inference_tadicodec import TaDiCodecPipline; print('TaDiCodec import successful!')"

# Flash Attentionのテスト（インストールした場合）
.venv/Scripts/python.exe -c "import flash_attn; print('Flash Attention:', flash_attn.__version__)"
```

## よく使うコマンド

### モデルのダウンロードとテスト

```bash
# すべてのモデルをHugging Faceからダウンロード
python test_auto_download.py

# サンプルテスト（use_examples/から実行）
cd use_examples
python test_auto_download.py     # モデルダウンロード
python test_llm_tts.py           # 自己回帰型TTSのテスト
python test_mgm_tts.py           # MGM型TTSのテスト
python test_rec.py               # トークン化と再構成のテスト
```

### トレーニング（開発中）

```bash
# TaDiCodecモデルの学習
python bins/tts/train.py --config <config_path> --exp_name <experiment_name>

# 例:
python bins/tts/train.py --config egs/tts/TaDiCodec/tadicodec_6_25hz_16384_bsq.json --exp_name my_tadicodec

# 学習の再開
python bins/tts/train.py --config <config_path> --exp_name <experiment_name> --resume --checkpoint_path <checkpoint_path>
```

## アーキテクチャ詳細

### コアコンポーネント

#### 1. TaDiCodecモデル (`models/tts/tadicodec/modeling_tadicodec.py`)

**エンコーダー**:
- Llama型の非自己回帰transformerアーキテクチャ
- メルスペクトログラムを潜在表現に変換
- レイヤー数: 8、隠れ層次元: 1024、ヘッド数: 16
- テキストや拡散ステップの条件付けなし

**ベクトル量子化（VQ）**:
- Binary Spherical Quantization (BSQ)
- 14次元の2値ベクトル（2^14 = 16384コードブック）
- L2正規化により超球面上で量子化
- エントロピー損失により均等な符号利用

**デコーダー**:
- Llama型の非自己回帰transformerアーキテクチャ
- Flow matchingベースの拡散プロセス
- レイヤー数: 16、隠れ層次元: 1024、ヘッド数: 16
- テキスト埋め込み、VQコード、拡散ステップで条件付け
- Classifier-Free Guidanceによる品質向上

#### 2. TTSモデル

**自己回帰型TTS** (`models/tts/llm_tts/`):
- バックボーン: Qwen2.5（0.5B / 3B）またはPhi-3.5（4B）
- 音声コードを特殊トークンとして扱う
- 自己回帰的にコード列を生成
- コードスイッチング対応

**MGM TTS** (`models/tts/llm_tts/mgm.py`):
- Masked Generative Model（MaskGCTベース）
- マスク拡散による並列トークン生成
- 自己回帰より高速
- テキストで条件付け

#### 3. Vocoder

**Vocos** (`models/codec/amphion_codec/vocos.py`):
- メルスペクトログラムから波形への変換
- Transformerベースのアーキテクチャ
- レイヤー数: 30、次元: 1024

### データフロー

```
[入力音声] → メル抽出 → エンコーダー → ダウンサンプル(÷8) → VQ (BSQ) → コード
                                                                        ↓
[再構成音声] ← Vocoder ← デコーダー ← アップサンプル(×8) ← コード埋め込み
                           ↑
                     [テキスト埋め込み]
                     [プロンプトメル]
```

### 主要設計パターン

- **自動ダウンロードシステム**: `_resolve_model_path()`でHugging Faceから自動取得
- **パイプラインパターン**: `TaDiCodecPipline`、`TTSInferencePipeline`、`MGMInferencePipeline`でモデル読み込みと推論をカプセル化
- **Accelerateベースの学習**: Hugging Face Accelerateによる分散学習
- **設定駆動アーキテクチャ**: `config/`と`egs/`のJSON設定でハイパーパラメータを定義

### 重要ファイルの場所

- **推論パイプライン**:
  - `models/tts/tadicodec/inference_tadicodec.py`
  - `models/tts/llm_tts/inference_llm_tts.py`
  - `models/tts/llm_tts/inference_mgm_tts.py`
- **モデルアーキテクチャ**:
  - `models/tts/tadicodec/modeling_tadicodec.py`
  - `models/tts/tadicodec/llama_nar_prefix.py`
- **学習**:
  - `models/base/tts_trainer.py`
  - `models/tts/tadicodec/tadicodec_trainer.py`
- **量子化**:
  - `models/codec/amphion_codec/quantize/bsq.py`
- **設定ファイル**:
  - `config/base.json`
  - `config/tts.json`
  - `egs/tts/TaDiCodec/tadicodec_6_25hz_16384_bsq.json`

## モデル読み込み

すべてのモデルはローカルパスとHugging Faceモデル IDの両方に対応：

```python
from models.tts.tadicodec.inference_tadicodec import TaDiCodecPipline
from models.tts.llm_tts.inference_llm_tts import TTSInferencePipeline

# Hugging Faceから（自動ダウンロード）
pipe = TaDiCodecPipline.from_pretrained("amphion/TaDiCodec")
tts = TTSInferencePipeline.from_pretrained(
    tadicodec_path="amphion/TaDiCodec",
    llm_path="amphion/TaDiCodec-TTS-AR-Qwen2.5-0.5B"
)

# ローカルパスから
pipe = TaDiCodecPipline.from_pretrained("./ckpt/TaDiCodec")
```

## 事前学習済みモデル

### TaDiCodecトークナイザー
- `amphion/TaDiCodec`（推奨）
- `amphion/TaDiCodec-old`（レガシー、AR-Phi-3.5-4Bで使用）

### TTSモデル
- **自己回帰型**:
  - `amphion/TaDiCodec-TTS-AR-Qwen2.5-0.5B`
  - `amphion/TaDiCodec-TTS-AR-Qwen2.5-3B`
  - `amphion/TaDiCodec-TTS-AR-Phi-3.5-4B`（TaDiCodec-oldを使用）
- **MGMモデル**:
  - `amphion/TaDiCodec-TTS-MGM`

## 重要な注意事項

- **Pythonバージョン**: Python 3.10が必須
- **PyTorchバージョン**: torch==2.8.0（CUDA 12.8対応）
- **Flash Attention**: パフォーマンスに重要。Windowsではビルド済みホイール推奨
- **Transformersバージョン**: 互換性のため4.42.4に固定
- **音声フォーマット**: すべてのモデルは24kHz音声を想定
- **テキスト認識**: デコーダーはテキスト条件付けで再構成品質を向上
- **コードスイッチング**: TTSモデルは多言語コードスイッチングに対応（例: 英語と中国語の混在）

## 開発ステータス

### ✅ 完成
- TaDiCodec推論パイプライン
- 自己回帰型・MGM型TTS推論
- Hugging Faceからの自動ダウンロード
- 再構成パイプライン
- **日本語対応機能**（新規）
  - 日本語音素トークナイザー（OpenJTalk情報100%活用）
  - パイプライン統合（自動テキスト→音素変換）
  - データ拡張機能（4手法）
  - キャッシュ生成スクリプト

### 🚧 開発中
- TTSモデル学習スクリプト
- 評価スクリプト
- テキスト入力用の自動ASR

---

## 🇯🇵 日本語対応（Japanese Support）

**ステータス: ✅ 学習開始済み・動作確認済み（2025-10-20）**

TaDiCodecの日本語音声合成最適化が完了し、学習が開始されました。

### ✅ 実装完了機能

#### 1. 日本語音素トークナイザー（OpenJTalk情報100%活用）

**ファイル:** `scripts/create_japanese_tokenizer_full.py`

**特徴:**
- OpenJTalkの**全フィールド（A-K）**から50+種類の韻律情報を抽出
- 従来版（10-15%情報使用）から**100%情報活用**に改善
- 新規トークン: 1,833個追加（32,011 → 33,844トークン）
- 新規トークン使用率: **86%**

**OpenJTalk情報の活用:**
```
音素コンテキスト（5-gram）: pp, p, c, n, nn
A: モーラ情報（3項目）
B: 前品詞タグ（3項目）
C: 現品詞タグ（3項目）
D: 前アクセント句（3項目）
E: 次アクセント句（5項目）
F: 現アクセント句（8項目すべて）
G: 前ブレス群（5項目）
H: 次ブレス群（2項目）
I: 現ブレス群（8項目）
J: 発話レベル情報（2項目）
K: ブレス群数情報（3項目）

合計: 50+種類の韻律・アクセント情報
```

**生成:**
```bash
python scripts/create_japanese_tokenizer_full.py
```

生成されるファイル:
- `ckpt/japanese_phoneme_vocabulary_full.json` (1,869トークン)
- `ckpt/TaDiCodec_Japanese_Full/text_tokenizer/` (拡張トークナイザー)

#### 2. パイプライン統合（自動テキスト→音素変換）

**ファイル:** `models/tts/tadicodec/inference_tadicodec_japanese.py`

**特徴:**
- `JapaneseTaDiCodecPipeline` クラス
- 日本語テキストを自動的に音素+韻律情報に変換
- 日本語検出機能（auto/always/neverモード）

**使用方法:**
```python
from models.tts.tadicodec.inference_tadicodec_japanese import JapaneseTaDiCodecPipeline

# 日本語対応パイプラインの作成
pipe = JapaneseTaDiCodecPipeline.from_pretrained(
    ckpt_dir="amphion/TaDiCodec",
    japanese_tokenizer_path="./ckpt/TaDiCodec_Japanese_Full/text_tokenizer",
    enable_japanese_phoneme=True,
    japanese_mode="auto",  # 日本語を自動検出
)

# 日本語テキストが自動的に音素+韻律情報に変換される
indices = pipe.encode(
    speech_path="sample.wav",
    text="東京の天気は晴れです。"
)
```

**処理フロー:**
```
日本語テキスト「東京の天気は晴れです。」
    ↓ pyopenjtalk
音素列 + 韻律情報（50+種類）
    ↓ phonemes_to_rich_tokens
リッチトークン列（スペース区切り）
    t [POS_OTHER] [ACC_TYPE_5] [TONE_0] [MORA_FIRST] ...
    ↓ HuggingFace tokenizer
トークンID（173個、新規トークン86%）
    ↓ TaDiCodec text_embedding
音声特徴量
```

#### 3. データ拡張機能（4手法）

**ファイル:** `models/tts/tadicodec/japanese_data_augmentation.py`

**実装した拡張手法:**

1. **ピッチシフト（男性⇄女性）**
   - 範囲: -4.0 〜 +4.0 半音
   - 適用確率: 50%
   - 実装: `librosa.effects.pitch_shift`

2. **速度変化（話速の変更）**
   - 範囲: 0.9x 〜 1.1x
   - 適用確率: 50%
   - 実装: `librosa.effects.time_stretch`

3. **アクセント位置の変更（日本語特有）**
   - OpenJTalkのアクセント情報を活用
   - 適用確率: 30%（日本語のみ）

4. **日英コードスイッチング**
   - 日本語単語を英語に置き換え
   - 適用確率: 20%（日本語のみ）
   - 辞書: 15種類の日英単語ペア

**使用方法:**

データセットクラス経由（推奨）:
```python
from models.tts.tadicodec.tadicodec_dataset_japanese import TadiCodecJapaneseDataset

# 設定ファイルで拡張を有効化
# egs/tts/TaDiCodec/tadicodec_japanese_finetune.json:
# {
#   "preprocess": {
#     "data_augment": ["japanese"],
#     "use_pitch_shift": true,
#     "use_speed_perturb": true,
#     "use_accent_augment": true,
#     "use_code_switch": true,
#     "augment_prob": 0.5
#   }
# }

dataset = TadiCodecJapaneseDataset(cache_type="path", cfg=cfg)
```

直接使用:
```python
from models.tts.tadicodec.japanese_data_augmentation import JapaneseDataAugmentation

augmentor = JapaneseDataAugmentation(
    use_pitch_shift=True,
    use_speed_perturb=True,
    use_accent_augment=True,
    use_code_switch=True,
    augment_prob=0.5
)

# 音声とテキストを拡張
aug_speech, aug_text = augmentor.augment(speech, text, sr=24000, language="ja")
```

#### 4. キャッシュ生成スクリプト

**ファイル:** `scripts/create_dataset_cache.py`

**機能:**
- Emilia形式データセットから学習用キャッシュを生成
- 学習時のデータ読み込みを高速化

**生成されるキャッシュファイル:**
- `wav_paths_cache.pkl` - WAVファイルパスのリスト
- `duration_cache.pkl` - 音声の長さ（秒）のリスト
- `bpe_token_count_cache.pkl` - BPEトークン数のリスト
- `json_paths_cache.pkl` - JSONメタデータのリスト（オプション）

**使用方法:**
```bash
# キャッシュ生成
python scripts/create_dataset_cache.py \
  --data_dir ./data/japanese \
  --cache_dir ./cache/japanese \
  --tokenizer_path ./ckpt/TaDiCodec_Japanese_Full/text_tokenizer \
  --min_duration 1.0 \
  --max_duration 40.0
```

**テスト:**
```bash
# サンプルデータ生成とテスト
python scripts/test_cache_generation.py
```

### 期待される効果

#### トークナイザー改善
| 指標 | 従来版 | 完全版 | 改善 |
|------|-------|-------|------|
| OpenJTalk情報使用率 | 10-15% | 100% | +92.3pt |
| トークン数（例文） | 14 (文字) | 173 (音素+韻律) | 12.4x |
| 新規トークン使用率 | 0% | 86% | +86pt |
| 語彙サイズ | 32,011 | 33,844 | +1,833 |

#### ファインチューニング後の予測
| 指標 | ベースライン | 目標 | 改善率 |
|------|------------|------|--------|
| WER（単語誤り率） | 35% | 14% | -60% |
| MOS（自然性） | 4.0 | 4.5 | +13% |
| Speaker SIM（話者類似度） | 0.65 | 0.76 | +17% |
| Accent Accuracy（アクセント精度） | 70% | 95%+ | +36% |

#### データ拡張の効果
- 実効データ量: **2-3倍**（確率的適用により）
- 音声バリエーション: ピッチ×速度 = 理論上無限
- 多言語ロバスト性: コードスイッチングにより向上

#### 5. 前処理スクリプト（メル・テキストトークン事前計算）

**マルチバージョンメルスペクトログラム事前計算:**
```powershell
python scripts/precompute_mel_features_multi_version.py `
  --config egs/tts/TaDiCodec/tadicodec_japanese_finetune.json `
  --output_dir ./cache/jvs_emilia/mel_augmented
```

生成されるバージョン:
- `original`: オリジナル
- `pitch`: ピッチシフト（±4半音）
- `speed`: 速度変化（0.9x-1.1x）
- `both`: ピッチ+速度両方

出力: 14,979 × 4 = 59,916ファイル（約20GB）

**テキストトークン事前計算:**
```powershell
python scripts/precompute_text_tokens.py `
  --config egs/tts/TaDiCodec/tadicodec_japanese_finetune.json `
  --output_path ./cache/jvs_emilia/text_tokens_cache.pkl
```

出力: 14,979サンプルのトークンID（約50MB）

#### 6. 学習初期化の最適化（99.8%高速化）

**実装場所:** `models/base/tts_trainer.py:397-452`

**最適化内容:**
- SafetensorsをGPUに直接ロード（CPU経由を排除）
- テキスト埋め込みサイズミスマッチの自動処理（32,100 → 33,844）
- BFloat16混合精度（RTX 30/40シリーズ最適化）

**成果:**
- チェックポイント読み込み: 5分以上 → **0.75秒**（99.8%改善）
- 初期化時間: 5分以上 → **6秒**（95%改善）

### 🚀 学習の開始（実行確認済み）

#### クイックスタート

```powershell
# PowerShellで実行（仮想環境アクティブ時）
python bins/tts/train.py `
  --config egs/tts/TaDiCodec/tadicodec_japanese_finetune.json `
  --exp_name TaDiCodec_Japanese_JVS_Optimized `
  --resume_type finetune `
  --checkpoint_path ./ckpt/TaDiCodec
```

```cmd
REM または仮想環境を明示的に指定
.venv\Scripts\python.exe bins\tts\train.py --config egs\tts\TaDiCodec\tadicodec_japanese_finetune.json --exp_name TaDiCodec_Japanese_JVS_Optimized --resume_type finetune --checkpoint_path .\ckpt\TaDiCodec
```

**最適化された設定（tadicodec_japanese_finetune.json）:**
- 学習率: `1e-5`（安定性向上、振動抑制）
- ウォームアップ: `1,000` steps（高速収束）
- ログ保存頻度: `100` steps（リアルタイム監視）
- 混合精度: `bf16`（RTX 30/40シリーズ最適化）

**期待される初期化ログ:**
```
[DEBUG] Loading checkpoint for finetune
Loading checkpoint from ./ckpt/TaDiCodec for finetune...
Loading checkpoint done in 750.00ms  ← 0.75秒で完了！

INFO:models.tts.tadicodec.tadicodec_dataset_japanese:Japanese data augmentation initialized:
INFO:models.tts.tadicodec.tadicodec_dataset_japanese:  - Pitch shift: True
INFO:models.tts.tadicodec.tadicodec_dataset_japanese:  - Speed perturbation: True
INFO:models.tts.tadicodec.tadicodec_dataset_japanese:  - Accent augmentation: True
INFO:models.tts.tadicodec.tadicodec_dataset_japanese:  - Code switching: True

[DEBUG _train_epoch] _train_step() completed, loss=5.1443  ← Loss計算成功！
[DEBUG _train_epoch] _train_step() completed, loss=5.1834
...
```

#### 学習設定

**設定ファイル:** `egs/tts/TaDiCodec/tadicodec_japanese_finetune.json`

```json
{
  "train": {
    "batch_size": 8,
    "gradient_accumulation_step": 8,  // 実効バッチサイズ=64
    "max_steps": 100000,
    "lr": 5e-05,
    "warmup_steps": 5000,
    "mixed_precision": "bf16",  // BFloat16（RTX最適化）
    "save_checkpoint_stride": [5000],
    "save_summary_steps": 500
  },
  "preprocess": {
    "use_precomputed_mel": true,
    "use_multi_version_mel": true,
    "use_precomputed_text_tokens": true,
    "use_pitch_shift": true,
    "use_speed_perturb": true,
    "use_accent_augment": true,
    "use_code_switch": true
  }
}
```

#### TensorBoardで監視

```powershell
tensorboard --logdir logs/TaDiCodec_Japanese_Final2
```

ブラウザで `http://localhost:6006` を開く

**監視すべき指標:**
- `Epoch/Train diff Loss`: 拡散損失（5.0 → 1.5程度まで下降予定）
- `Epoch/Train vq Loss`: VQ量子化損失（安定）
- `learning_rate`: 学習率スケジュール

### 📚 詳細ドキュメント

完全な手順は以下のドキュメントを参照:
- **JAPANESE_QUICK_START.md**: 学習開始までのクイックガイド（実行コマンド・結果付き）
- **JAPANESE_PREPROCESSING_README.md**: 前処理の詳細ガイド
- **JAPANESE_TRAINING_GUIDE.md**: 学習の詳細ガイド
- **JAPANESE_TOKENIZER_COMPLETE.md**: トークナイザーの完全ドキュメント

### 日本語ファインチューニングの手順（詳細版）

#### ステップ1: 環境準備

```bash
# 仮想環境をアクティベート
source .venv/Scripts/activate  # Linux/Mac
# または
.venv/Scripts/activate.ps1  # Windows PowerShell

# 日本語トークナイザーを生成（初回のみ）
python scripts/create_japanese_tokenizer_full.py
```

#### ステップ2: データセットの準備

**推奨データセット:**
- **JVS Corpus**: 100話者、並列文30文（約30時間）
  - URL: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
  - ライセンス: CC BY-SA 4.0

- **JSUT Corpus**: 1話者、約10時間
  - URL: https://sites.google.com/site/shinnosuketakamichi/publication/jsut
  - ライセンス: CC BY-SA 4.0

**データ構造（Emilia形式）:**
```
data/japanese/
├── jvs001/
│   ├── audio_0.wav
│   ├── audio_1.wav
│   └── audio.json
├── jvs002/
│   └── ...
└── jsut_basic/
    └── ...
```

**audio.json フォーマット:**
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

#### ステップ3: キャッシュ生成

```bash
# データセットキャッシュを生成
python scripts/create_dataset_cache.py \
  --data_dir ./data/japanese \
  --cache_dir ./cache/japanese \
  --tokenizer_path ./ckpt/TaDiCodec_Japanese_Full/text_tokenizer \
  --min_duration 1.0 \
  --max_duration 40.0
```

**出力例:**
```
Total audio files processed: 10000
Average duration: 5.2 seconds
Average token count: 25.3 tokens
Total duration: 14.4 hours
```

#### ステップ4: 設定ファイルの確認

**ファイル:** `egs/tts/TaDiCodec/tadicodec_japanese_finetune.json`

**重要な設定:**
```json
{
  "preprocess": {
    "mnt_path": "./data/japanese",
    "cache_folder": "./cache/japanese",
    "use_json_path_cache": true,
    "tokenizer_path": "./ckpt/TaDiCodec_Japanese_Full/text_tokenizer",
    "data_augment": ["japanese"],
    "use_pitch_shift": true,
    "use_speed_perturb": true,
    "use_accent_augment": true,
    "use_code_switch": true,
    "augment_prob": 0.5
  },
  "model": {
    "tadicodec": {
      "text_vocab_size": 33844  // 日本語拡張語彙
    }
  },
  "train": {
    "batch_size": 8,
    "gradient_accumulation_step": 4,
    "max_steps": 100000,
    "adamw": {
      "lr": 5e-5
    }
  },
  "dataset": ["jvs", "jsut"],
  "dataset_path": {
    "jvs": "/path/to/jvs_ver1",
    "jsut": "/path/to/jsut_ver1.1"
  }
}
```

#### ステップ5: ファインチューニング実行

```bash
# ファインチューニング開始
python bins/tts/train.py \
  --config egs/tts/TaDiCodec/tadicodec_japanese_finetune.json \
  --exp_name TaDiCodec_Japanese_Finetune \
  --resume \
  --resume_type finetune \
  --checkpoint_path ~/.cache/huggingface/hub/models--amphion--TaDiCodec/snapshots/<hash>/checkpoint.pth
```

**学習の監視:**
```bash
# TensorBoardで監視
tensorboard --logdir ./logs/TaDiCodec_Japanese_Finetune

# 確認すべき指標
# - diff_loss (下がっているか)
# - vq_loss (安定しているか)
# - commit_loss (低下しているか)
# - learning_rate (スケジュール通りか)
```

#### ステップ6: 評価

```bash
# 評価スクリプト（今後実装予定）
python eval/evaluate_japanese.py \
  --model_path ./logs/TaDiCodec_Japanese_Finetune/checkpoint/best.pth \
  --test_data ./data/japanese_test \
  --output_dir ./eval_results
```

### トラブルシューティング

#### トークナイザーのエラー

**問題:** `FileNotFoundError: tokenizer_path not found`

**解決策:**
```bash
# トークナイザーを再生成
python scripts/create_japanese_tokenizer_full.py

# 存在確認
ls -la ./ckpt/TaDiCodec_Japanese_Full/text_tokenizer/
```

#### キャッシュ生成のエラー

**問題:** `No speaker directories found`

**解決策:**
データディレクトリの構造を確認:
```bash
# 正しい構造:
# data/japanese/speaker1/audio.json
# data/japanese/speaker1/audio_0.wav
ls -R ./data/japanese
```

#### GPU メモリ不足

**問題:** `CUDA out of memory`

**解決策:**
設定ファイルでバッチサイズを減らす:
```json
{
  "train": {
    "batch_size": 2,                    // 小さく
    "gradient_accumulation_step": 8     // 勾配累積で補う
  }
}
```

### 関連ドキュメント

- **[JAPANESE_TOKENIZER_COMPLETE.md](./JAPANESE_TOKENIZER_COMPLETE.md)** - トークナイザーの完全ドキュメント
- **[JAPANESE_INTEGRATION_COMPLETE.md](./JAPANESE_INTEGRATION_COMPLETE.md)** - パイプライン統合の完了レポート
- **[JAPANESE_TRAINING_GUIDE.md](./JAPANESE_TRAINING_GUIDE.md)** - 日本語学習の詳細ガイド
- **[JAPANESE_FINETUNING_ROADMAP.md](./JAPANESE_FINETUNING_ROADMAP.md)** - ファインチューニングのロードマップ
- **[scripts/README_CACHE_GENERATION.md](./scripts/README_CACHE_GENERATION.md)** - キャッシュ生成の詳細ガイド

### 技術的な詳細

#### トークン最適化の経緯

**初期実装の問題:**
- リッチトークンを連結して生成: `t[POS_OTHER][ACC_TYPE_5]...`
- 結果: 1,070トークン（76.4x増加）
- 新規トークン使用率: 2%（ほぼ未使用）

**修正後:**
- スペース区切りで生成: `t [POS_OTHER] [ACC_TYPE_5] ...`
- 結果: 173トークン（12.4x増加）
- 新規トークン使用率: 86%（大幅改善）

**変更箇所** (`scripts/create_japanese_tokenizer_full.py:529`):
```python
# BEFORE
combined_token = ''.join(token_parts)

# AFTER
combined_token = ' '.join(token_parts)
```

#### データ拡張パイプライン

```
入力: (音声, テキスト, 言語="ja")
  ↓
[1] ピッチシフト (50%確率)
  ↓
[2] 速度変化 (50%確率)
  ↓
[3] アクセント拡張 (30%確率, Japanese only)
  ↓
[4] コードスイッチング (20%確率, Japanese only)
  ↓
出力: (拡張音声, 拡張テキスト)
```

---

## 参考文献

論文: "TaDiCodec: Text-aware Diffusion Speech Tokenizer for Speech Language Modeling" (NeurIPS 2025)

以下のプロジェクトをベースに構築:
- MaskGCT
- Vocos
- Hugging Face Transformers
- vector-quantize-pytorch
- bsq-vit
- Amphion
- Accelerate
