@echo off
chcp 65001 >nul
REM TaDiCodec Japanese Preprocessing Script
REM メルスペクトログラムを事前計算して学習を高速化

echo ========================================
echo TaDiCodec Japanese Preprocessing
echo ========================================
echo.
echo このスクリプトはメルスペクトログラムとテキストトークンを事前計算します
echo.
echo 予想時間: 35-40分
echo 必要ディスク容量: 30-50GB (メル) + 5-10MB (トークン)
echo 処理ファイル数: 14,979サンプル
echo.
echo 処理内容:
echo   [1/3] メルスペクトログラム計算 (18分)
echo   [2/3] テキストトークンID計算 (15-20分)
echo   [3/3] 設定ファイル更新
echo.
echo データ拡張:
echo   - ピッチシフト: 有効
echo   - 速度変化: 有効
echo   - 拡張確率: 50%%
echo.
pause

REM Set environment variables
set PYTHONPATH=.
set WORK_DIR=.

echo.
echo ========================================
echo ステップ 1/3: メルスペクトログラム計算
echo ========================================
echo.

REM Run preprocessing
.venv\Scripts\python.exe scripts\precompute_mel_features.py ^
    --config egs\tts\TaDiCodec\tadicodec_japanese_finetune.json ^
    --output_dir .\cache\jvs_emilia\mel ^
    --enable_augmentation ^
    --augment_prob 0.5

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo エラー: メル計算が失敗しました
    pause
    exit /b 1
)

echo.
echo ========================================
echo ステップ 2/3: テキストトークンID計算
echo ========================================
echo.

REM Run text token precomputation
.venv\Scripts\python.exe scripts\precompute_text_tokens.py ^
    --config egs\tts\TaDiCodec\tadicodec_japanese_finetune.json ^
    --output_path .\cache\jvs_emilia\text_tokens_cache.pkl

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo エラー: テキストトークン計算が失敗しました
    pause
    exit /b 1
)

echo.
echo ========================================
echo ステップ 3/3: 設定ファイル更新
echo ========================================
echo.
echo use_precomputed_mel と use_precomputed_text_tokens を true に変更します...

REM Update config file using Python
.venv\Scripts\python.exe -c "import json; cfg = json.load(open('egs/tts/TaDiCodec/tadicodec_japanese_finetune.json', 'r', encoding='utf-8')); cfg['preprocess']['use_precomputed_mel'] = True; cfg['preprocess']['use_precomputed_text_tokens'] = True; cfg['preprocess']['text_tokens_cache_path'] = './cache/jvs_emilia/text_tokens_cache.pkl'; json.dump(cfg, open('egs/tts/TaDiCodec/tadicodec_japanese_finetune.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 警告: 設定ファイルの自動更新に失敗しました
    echo 手動で use_precomputed_mel と use_precomputed_text_tokens を true に変更してください
) else (
    echo 設定ファイルを更新しました
)

echo.
echo ========================================
echo 前処理完了
echo ========================================
echo.
echo 次のステップ:
echo   1. train_japanese.bat を実行して学習を開始
echo   2. または tensorboard.bat で進捗を監視
echo.
echo 出力:
echo   - メルキャッシュ: .\cache\jvs_emilia\mel\
echo   - メル統計情報: .\cache\jvs_emilia\mel\precompute_stats.json
echo   - テキストトークンキャッシュ: .\cache\jvs_emilia\text_tokens_cache.pkl
echo   - トークン統計情報: .\cache\jvs_emilia\text_tokens_stats.json
echo.
echo 効果:
echo   - 学習開始時のデータローダー初期化: 5-15分 → 30秒以下 (10-30倍高速化)
echo   - OpenJTalk警告: 完全に解消
echo.
pause
