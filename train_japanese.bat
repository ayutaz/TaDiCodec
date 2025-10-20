@echo off
chcp 65001 >nul
REM TaDiCodec Japanese Fine-tuning Training Script
REM This script trains the TaDiCodec model on Japanese JVS dataset

echo ========================================
echo TaDiCodec Japanese Fine-tuning
echo ========================================
echo.
echo Dataset: JVS Emilia (14,979 samples)
echo Batch size: 8
echo Gradient accumulation: 8
echo Effective batch size: 64
echo.
echo Features:
echo   - Japanese data augmentation: ENABLED
echo     * Pitch shift (+-4 semitones)
echo     * Speed perturbation (0.9x-1.1x)
echo     * Accent augmentation
echo     * Code switching (JP/EN)
echo   - Precomputed mel: Check config file
echo.
echo IMPORTANT:
echo   For best performance, run preprocess_japanese.bat first!
echo.
echo NOTE: You may see many OpenJTalk warnings during initialization.
echo   This is NORMAL and can be ignored. The warnings will stop after
echo   data loading completes (5-10 minutes). Use train_japanese_quiet.bat
echo   to suppress these warnings.
echo.

REM Set environment variables
set PYTHONPATH=.
set WORK_DIR=.
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM Run training
echo Starting training...
echo.
.venv\Scripts\python.exe bins\tts\train.py --config egs\tts\TaDiCodec\tadicodec_japanese_finetune.json --exp_name TaDiCodec_Japanese_JVS --resume_type finetune --checkpoint_path .\ckpt\TaDiCodec

echo.
echo Training completed or terminated.
pause
