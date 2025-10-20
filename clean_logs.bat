@echo off
chcp 65001 >nul
REM Clean up training logs and checkpoints
REM Use this to start fresh training

echo ========================================
echo Clean Training Logs
echo ========================================
echo.
echo WARNING: This will delete all training logs and checkpoints!
echo Location: logs\TaDiCodec_Japanese_JVS
echo.
echo Press Ctrl+C to cancel, or
pause

echo.
echo Cleaning logs...
if exist logs\TaDiCodec_Japanese_JVS (
    rmdir /s /q logs\TaDiCodec_Japanese_JVS
    echo Logs cleaned successfully!
) else (
    echo No logs found to clean.
)

echo.
echo Done!
pause
