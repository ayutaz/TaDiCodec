@echo off
chcp 65001 >nul
REM TaDiCodec Japanese Training Resume Script
REM This script resumes training from the latest checkpoint

echo ========================================
echo Resume TaDiCodec Japanese Training
echo ========================================
echo.
echo This will resume training from the latest checkpoint
echo in logs\TaDiCodec_Japanese_JVS\checkpoint
echo.

REM Set environment variables
set PYTHONPATH=.
set WORK_DIR=.
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM Resume training
echo Resuming training...
echo.
.venv\Scripts\python.exe bins\tts\train.py --config egs\tts\TaDiCodec\tadicodec_japanese_finetune.json --exp_name TaDiCodec_Japanese_JVS --resume

echo.
echo Training completed or terminated.
pause
