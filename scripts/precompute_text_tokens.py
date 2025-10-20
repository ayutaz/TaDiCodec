#!/usr/bin/env python3
"""
テキストトークンID事前計算スクリプト

全音声ファイルのテキストをトークンIDに事前変換して保存します。
これにより学習時のデータローダー初期化を10-30倍高速化します。

機能:
- テキスト→トークンID変換（日本語トークナイザー使用）
- 進捗表示とエラーハンドリング
- キャッシュファイルとして保存

使用方法:
    python scripts/precompute_text_tokens.py \
        --config egs/tts/TaDiCodec/tadicodec_japanese_finetune.json \
        --output_path ./cache/jvs_emilia/text_tokens_cache.pkl

予想時間: 15-20分（14,979サンプル）
ディスク容量: 5-10MB
"""

import argparse
import os
import sys
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
from transformers import AutoTokenizer

# プロジェクトルートをパスに追加
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.util import load_config


def precompute_text_tokens(args):
    """
    テキストトークンIDの事前計算メイン関数
    """
    print("="*80)
    print("テキストトークンID事前計算")
    print("="*80)

    # 設定ファイルのロード
    print(f"\n設定ファイル読み込み: {args.config}")
    cfg = load_config(args.config)

    # 出力ディレクトリの作成
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"出力ファイル: {output_path}")

    # キャッシュフォルダの設定
    cache_folder = Path(cfg.preprocess.cache_folder)
    wav_paths_cache = cache_folder / "wav_paths_cache.pkl"
    json_paths_cache = cache_folder / "json_paths_cache.pkl"

    if not wav_paths_cache.exists():
        print(f"エラー: {wav_paths_cache} が見つかりません")
        print("先に scripts/create_dataset_cache.py を実行してください")
        sys.exit(1)

    print(f"\nキャッシュ読み込み: {wav_paths_cache}")
    with open(wav_paths_cache, "rb") as f:
        wav_paths = pickle.load(f)

    use_json_path_cache = getattr(cfg.preprocess, "use_json_path_cache", False)
    if use_json_path_cache and json_paths_cache.exists():
        print(f"キャッシュ読み込み: {json_paths_cache}")
        with open(json_paths_cache, "rb") as f:
            json_paths = pickle.load(f)
    else:
        json_paths = None

    print(f"音声ファイル数: {len(wav_paths)}")

    # トークナイザーの初期化
    print("\nトークナイザー初期化...")
    if hasattr(cfg.preprocess, "tokenizer_path"):
        tokenizer_path = cfg.preprocess.tokenizer_path
    else:
        tokenizer_path = "./ckpt/TaDiCodec/text_tokenizer"

    print(f"トークナイザーパス: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"語彙サイズ: {len(tokenizer)}")

    # テキストトークンIDを計算して保存
    print("\nテキストトークンID計算開始...")
    print(f"進捗: 0/{len(wav_paths)}")

    text_tokens_cache = {}
    success_count = 0
    error_count = 0
    mnt_path = Path(cfg.preprocess.mnt_path)

    for idx, wav_path in enumerate(tqdm(wav_paths, desc="計算中")):
        try:
            # メタデータからテキストを取得
            if json_paths is not None:
                # use_json_path_cache=true の場合
                try:
                    meta = json_paths[idx]
                except Exception as e:
                    print(f"\n警告 (idx={idx}): json_paths[idx] の取得失敗: {e}")
                    error_count += 1
                    continue
            else:
                # use_json_path_cache=false の場合
                # wav_path から json_path を構築
                index = int(wav_path.split("_")[-1].split(".")[0])
                audio_name = "_".join(wav_path.split("/")[-1].split("_")[:-1])
                dir_name = "/".join(wav_path.split("/")[:-1])
                json_name = audio_name + ".json"
                json_path = dir_name + "/" + json_name

                full_json_path = mnt_path / json_path

                if not full_json_path.exists():
                    print(f"\n警告: JSONファイルが見つかりません: {full_json_path}")
                    error_count += 1
                    continue

                try:
                    with open(full_json_path, "r", encoding="utf-8") as f:
                        meta_dict = json.load(f)
                    meta = meta_dict[index]
                except Exception as e:
                    print(f"\n警告 (idx={idx}, path={json_path}): {e}")
                    error_count += 1
                    continue

            # テキストを取得
            if isinstance(meta, dict) and "text" in meta:
                text = meta["text"]
            else:
                print(f"\n警告 (idx={idx}): テキストが見つかりません")
                error_count += 1
                continue

            # トークンIDに変換
            text_ids = tokenizer.encode(text, add_special_tokens=False)

            # キャッシュに保存
            text_tokens_cache[idx] = text_ids

            success_count += 1

        except Exception as e:
            print(f"\nエラー (idx={idx}, path={wav_path}): {e}")
            error_count += 1
            continue

    # キャッシュを保存
    print(f"\n\nキャッシュ保存中: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(text_tokens_cache, f)

    # 統計情報を保存
    stats = {
        "total_files": len(wav_paths),
        "success_count": success_count,
        "error_count": error_count,
        "config_path": args.config,
        "output_path": str(output_path),
        "tokenizer_path": tokenizer_path,
        "vocab_size": len(tokenizer),
    }

    stats_path = output_path.parent / "text_tokens_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("テキストトークンID事前計算完了")
    print("="*80)
    print(f"成功: {success_count}/{len(wav_paths)}")
    print(f"エラー: {error_count}/{len(wav_paths)}")
    print(f"出力ファイル: {output_path}")
    print(f"統計情報: {stats_path}")

    # ファイルサイズ表示
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"ファイルサイズ: {file_size_mb:.2f} MB")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="テキストトークンID事前計算")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="設定ファイルパス (例: egs/tts/TaDiCodec/tadicodec_japanese_finetune.json)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./cache/jvs_emilia/text_tokens_cache.pkl",
        help="テキストトークンキャッシュ出力パス",
    )

    args = parser.parse_args()

    precompute_text_tokens(args)


if __name__ == "__main__":
    main()
