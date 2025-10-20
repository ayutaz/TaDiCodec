@echo off
chcp 65001 >nul
REM TaDiCodec Japanese Training Script (Quiet Mode - OpenJTalk warnings suppressed)
REM This script filters out OpenJTalk warnings for cleaner output

echo ========================================
echo TaDiCodec Japanese Fine-tuning (Quiet Mode)
echo ========================================
echo.
echo OpenJTalk warnings will be filtered out
echo Only training progress will be displayed
echo.

REM Set environment variables
set PYTHONPATH=.
set WORK_DIR=.
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM Start training with warning filtering
echo Starting training...
echo.
.venv\Scripts\python.exe bins\tts\train.py --config egs\tts\TaDiCodec\tadicodec_japanese_finetune.json --exp_name TaDiCodec_Japanese_JVS --resume --resume_type finetune --checkpoint_path .\ckpt\TaDiCodec 2>&1 | findstr /V "WARNING: convert_pos"

echo.
echo Training completed or terminated.
pause
