@echo off
chcp 65001 >nul
REM TaDiCodec Japanese Preprocessing Script (Multi-Version)
REM 複数バージョンメルスペクトログラムを事前計算

echo ========================================
echo TaDiCodec 複数バージョンメル事前計算
echo ========================================
echo.
echo このスクリプトは各サンプルに対して4つのバージョンを生成します:
echo   1. original - 拡張なし（ベースライン）
echo   2. pitch    - ピッチシフトのみ（-4〜+4半音）
echo   3. speed    - 速度変化のみ（0.9x〜1.1x）
echo   4. both     - ピッチ+速度の両方
echo.
echo 予想時間: 3-4時間
echo 必要ディスク容量: 約200GB
echo 処理ファイル数: 14,979サンプル × 4バージョン = 59,916ファイル
echo.
echo 効果:
echo   - データ拡張効果: 100%% 維持
echo   - 学習初期化時間: 30秒以下 （2-5分 → 30秒）
echo   - 学習速度: 2-3倍高速化
echo.
echo 注意:
echo   - この処理は一度だけ実行すれば OK です
echo   - 途中で中断しても、次回は途中から再開できます
echo   - 既存のファイルはスキップされます
echo.
pause

REM Set environment variables
set PYTHONPATH=.
set WORK_DIR=.

echo.
echo ========================================
echo メルスペクトログラム複数バージョン計算
echo ========================================
echo.

REM Run preprocessing
.venv\Scripts\python.exe scripts\precompute_mel_features_multi_version.py ^
    --config egs\tts\TaDiCodec\tadicodec_japanese_finetune.json ^
    --output_dir .\cache\jvs_emilia\mel_augmented

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo エラー: メル計算が失敗しました
    pause
    exit /b 1
)

echo.
echo ========================================
echo 前処理完了
echo ========================================
echo.
echo 次のステップ:
echo   1. 設定ファイルを更新（自動的に実行されます）
echo   2. train_japanese_multi_version.bat を実行して学習を開始
echo.
echo 出力:
echo   - メルキャッシュ: .\cache\jvs_emilia\mel_augmented\
echo   - 統計情報: .\cache\jvs_emilia\mel_augmented\precompute_multi_version_stats.json
echo.
echo 各サンプルに対して4つのファイルが生成されています:
echo   - audio_0_original.npy （拡張なし）
echo   - audio_0_pitch.npy    （ピッチシフト）
echo   - audio_0_speed.npy    （速度変化）
echo   - audio_0_both.npy     （ピッチ+速度）
echo.
echo 学習時はこれらの中からランダムに1つが選択されます。
echo.
pause
