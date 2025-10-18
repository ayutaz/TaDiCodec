# 環境情報（Environment Information）

このファイルは、TaDiCodecプロジェクトで使用した開発環境の詳細を記録したものです。

## 作成日時

2025年10月18日

## システム情報

### ハードウェア
- **GPU**: NVIDIA GeForce RTX 4070 Ti SUPER
- **VRAM**: 16GB

### ソフトウェア
- **OS**: Windows
- **Python**: 3.10.15
- **パッケージマネージャ**: uv 0.9.2 (141369ce7 2025-10-10)

### CUDA/PyTorch
- **CUDA**: 12.8
- **PyTorch**: 2.8.0+cu128
- **torchaudio**: 2.8.0+cu128

### 主要パッケージ
- **transformers**: 4.42.4
- **librosa**: 0.11.0
- **accelerate**: 1.10.1
- **flash-attn**: 2.7.4.post1
- **safetensors**: 0.6.2
- **omegaconf**: 2.3.0

## インストール済みパッケージ

完全なパッケージリストは `requirements-freeze.txt` を参照してください。

```bash
# 現在の環境を確認
uv pip freeze

# または
cat requirements-freeze.txt
```

## 環境の再現方法

### 方法1: requirements-freeze.txtを使用（推奨しない）

**注意**: `requirements-freeze.txt`にはFlash Attentionのローカルパスが含まれているため、そのままでは使用できません。

```bash
# このコマンドは失敗します
uv pip install -r requirements-freeze.txt
```

### 方法2: CLAUDE.mdの手順に従う（推奨）

`CLAUDE.md`の「環境構築（Windows）」セクションに記載された手順に従ってください。

主要な手順：
1. Python 3.10のインストール
2. 基本パッケージのインストール
3. PyTorch (CUDA版) のインストール
4. コアパッケージのインストール
5. Flash Attentionのインストール（オプション）

詳細は `CLAUDE.md` を参照してください。

### 方法3: Flash Attentionを除外してインストール

```bash
# Flash Attention以外のパッケージをインストール
grep -v "flash-attn" requirements-freeze.txt > requirements-no-flash.txt
uv pip install -r requirements-no-flash.txt

# 次に、Flash Attentionを手動でインストール（FLASH_ATTENTION_INSTALL.mdを参照）
```

## Flash Attentionについて

Flash Attentionは特殊なインストール手順が必要です。

- **インストール方法**: `FLASH_ATTENTION_INSTALL.md`を参照
- **wheelファイル**: ローカルでリネームが必要
- **環境変数**: `UV_SKIP_WHEEL_FILENAME_CHECK=1`の設定が必要
- **オプション**: インストールしなくてもTaDiCodecは動作します（速度が遅くなります）

詳細は `FLASH_ATTENTION_INSTALL.md` を参照してください。

## 特殊なパッケージの説明

### pyopenjtalk-plus

Windows用のpyopenjtalkの代替パッケージです。

- **理由**: 標準の`pyopenjtalk`はWindowsでのビルドに失敗
- **解決策**: `pyopenjtalk-plus`を使用
- **機能**: 日本語音素変換（TTS用）

### PyTorch (CUDA版)

CUDA 12.8対応のPyTorch 2.8.0を使用しています。

```bash
# インストールコマンド
uv pip install torch==2.8.0 torchaudio --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
```

### transformers

バージョン4.42.4に固定されています。

- **理由**: TaDiCodecとの互換性のため
- **注意**: より新しいバージョンでは動作しない可能性があります

## パッケージの総数

合計: **70パッケージ**

カテゴリー別：
- PyTorch関連: 3パッケージ
- Transformers/Hugging Face: 4パッケージ
- 音声処理: 6パッケージ
- 深層学習ユーティリティ: 10パッケージ
- その他の依存関係: 47パッケージ

## 検証コマンド

環境が正しくセットアップされているか確認するコマンド：

```bash
# GPU環境のテスト
.venv/Scripts/python.exe test_gpu_setup.py

# PyTorchの確認
.venv/Scripts/python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Flash Attentionの確認（インストールした場合）
.venv/Scripts/python.exe -c "import flash_attn; print('Flash Attention:', flash_attn.__version__)"

# TaDiCodecの確認
.venv/Scripts/python.exe -c "from models.tts.tadicodec.inference_tadicodec import TaDiCodecPipline; print('TaDiCodec import successful!')"
```

## トラブルシューティング

### 問題: パッケージのバージョンが異なる

**原因**: `requirements-freeze.txt`は特定の環境でのスナップショットです。

**解決策**:
1. `CLAUDE.md`の手順に従って、主要パッケージのみをインストール
2. バージョンの厳密な一致にこだわらず、互換性のある範囲で最新版を使用

### 問題: Flash Attentionのインストールに失敗

**解決策**:
1. `FLASH_ATTENTION_INSTALL.md`の詳細な手順に従う
2. それでも失敗する場合は、Flash Attentionなしで使用可能（速度が遅くなるだけ）

### 問題: CUDAバージョンの不一致

**原因**: GPUドライバーが古い、またはCUDAバージョンが異なる

**解決策**:
1. NVIDIAドライバーを最新版に更新（CUDA 12.8対応はバージョン525以上）
2. 自分の環境に合ったPyTorchをインストール（PyTorchの公式サイトで確認）

## 参考資料

- **環境構築手順**: `CLAUDE.md`
- **Flash Attentionインストール**: `FLASH_ATTENTION_INSTALL.md`
- **GPU環境テスト**: `test_gpu_setup.py`
- **公式README**: `README.md`

## メモ

このrequirements-freeze.txtは、2025年10月18日時点でのスナップショットです。

### 推奨事項

1. **新規セットアップ**: `CLAUDE.md`の手順に従ってください
2. **環境の複製**: このファイルを参考に、同じバージョンをインストール
3. **トラブル時**: 各ドキュメントのトラブルシューティングセクションを参照

### 更新履歴

- 2025-10-18: 初版作成
  - Python 3.10.15
  - PyTorch 2.8.0+cu128
  - Flash Attention 2.7.4.post1
  - Transformers 4.42.4

---

**注意**: このファイルは記録目的です。環境を再現する際は、必ず`CLAUDE.md`の手順に従ってください。
