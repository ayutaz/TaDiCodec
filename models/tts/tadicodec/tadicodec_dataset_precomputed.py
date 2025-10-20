# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
事前計算メルスペクトログラム対応TaDiCodecデータセット

メルスペクトログラムを事前計算して保存した後、学習時にディスクから高速に読み込むデータセット。
これにより学習速度を2-3倍高速化します。

使用方法:
    1. scripts/precompute_mel_features.py でメルスペクトログラムを事前計算
    2. 設定ファイルで use_precomputed_mel: true を指定
    3. このデータセットクラスが自動的に使用されます
"""

import logging
import os
import random
import numpy as np
from pathlib import Path
from models.tts.tadicodec.tadicodec_dataset_japanese import (
    TadiCodecJapaneseDataset,
    TadiCodecJapaneseCollator,
)


logger = logging.getLogger(__name__)


class TadiCodecPrecomputedDataset(TadiCodecJapaneseDataset):
    """
    事前計算メルスペクトログラム対応データセット

    TadiCodecJapaneseDatasetを継承し、メルスペクトログラムを
    ディスクから読み込む機能を追加。

    メリット:
    - メル計算のI/Oボトルネックを解消
    - 学習速度2-3倍高速化
    - CPUリソースをGPUにより集中させる

    前提条件:
    - scripts/precompute_mel_features.py を事前に実行済み
    - cfg.preprocess.mel_cache_dir にメルスペクトログラムが保存されている
    """

    def __init__(self, cache_type="path", cfg=None):
        super().__init__(cache_type, cfg)

        # メルキャッシュディレクトリの設定
        self.use_precomputed_mel = getattr(cfg.preprocess, "use_precomputed_mel", False)
        if self.use_precomputed_mel:
            mel_cache_dir = getattr(cfg.preprocess, "mel_cache_dir", "./cache/jvs_emilia/mel")
            self.mel_cache_dir = Path(mel_cache_dir)

            if not self.mel_cache_dir.exists():
                logger.error(f"メルキャッシュディレクトリが見つかりません: {self.mel_cache_dir}")
                logger.error("scripts/precompute_mel_features.py を先に実行してください")
                raise FileNotFoundError(f"Mel cache directory not found: {self.mel_cache_dir}")

            # 複数バージョンメル対応
            self.use_multi_version_mel = getattr(cfg.preprocess, "use_multi_version_mel", False)
            if self.use_multi_version_mel:
                self.mel_versions = ["original", "pitch", "speed", "both"]
                self.mel_version_weights = [0.25, 0.25, 0.25, 0.25]  # 均等確率
                logger.info(f"複数バージョンメルスペクトログラムを使用: {self.mel_cache_dir}")
                logger.info(f"  バージョン: {self.mel_versions}")
                logger.info(f"  選択確率: 各25%")
            else:
                logger.info(f"事前計算メルスペクトログラムを使用: {self.mel_cache_dir}")
        else:
            logger.info("事前計算メルスペクトログラムは使用しません（オンザフライ計算）")

    def _get_single_feature(self, speech, text, language, idx=None):
        """
        単一サンプルの特徴抽出

        事前計算メルスペクトログラムを使用する場合は、
        親クラスの処理に加えて precomputed_mel_path を追加

        Args:
            speech: 音声波形 (numpy array) - メル計算に使用されない
            text: テキスト (str)
            language: 言語コード (str)
            idx: サンプルインデックス（テキストトークンキャッシュ用）

        Returns:
            single_feature: 特徴量辞書（precomputed_mel_pathキーを含む）
        """
        # 親クラスの特徴抽出（データ拡張とトークン化）
        single_feature = super()._get_single_feature(speech, text, language, idx=idx)

        # 事前計算メルスペクトログラムを使用する場合のマーカー
        if self.use_precomputed_mel:
            single_feature["use_precomputed_mel"] = True
        else:
            single_feature["use_precomputed_mel"] = False

        return single_feature

    def __getitem__(self, idx):
        """
        データセットから1サンプルを取得

        事前計算メルスペクトログラムを使用する場合は、
        メルスペクトログラムをディスクから読み込み

        Args:
            idx: サンプルインデックス

        Returns:
            single_feature: 特徴量辞書
        """
        # 親クラスの__getitem__を呼び出す
        single_feature = super().__getitem__(idx)

        # 事前計算メルスペクトログラムを読み込む
        if self.use_precomputed_mel and single_feature is not None:
            try:
                wav_path = self.wav_paths[idx]
                # wav_path形式: "JA_jvs001/audio_0.wav"
                speaker_id = os.path.dirname(wav_path)
                audio_name = os.path.splitext(os.path.basename(wav_path))[0]  # "audio_0"

                # 複数バージョンメル対応
                if self.use_multi_version_mel:
                    # ランダムにバージョンを選択
                    selected_version = random.choices(
                        self.mel_versions,
                        weights=self.mel_version_weights
                    )[0]
                    mel_filename = f"{audio_name}_{selected_version}.npy"
                else:
                    mel_filename = f"{audio_name}.npy"

                mel_path = self.mel_cache_dir / speaker_id / mel_filename

                if not mel_path.exists():
                    logger.warning(f"事前計算メルが見つかりません: {mel_path}")
                    logger.warning("オンザフライ計算にフォールバック")
                    single_feature["precomputed_mel"] = None
                else:
                    # メルスペクトログラムを読み込み
                    mel = np.load(mel_path)  # [T, d]
                    single_feature["precomputed_mel"] = mel

            except Exception as e:
                logger.error(f"メル読み込みエラー (idx={idx}): {e}")
                single_feature["precomputed_mel"] = None

        return single_feature


# コレーターはJapaneseCollatorと同じ
TadiCodecPrecomputedCollator = TadiCodecJapaneseCollator
