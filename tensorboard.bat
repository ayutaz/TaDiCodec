@echo off
chcp 65001 >nul
REM TensorBoard Launch Script
REM View training progress at http://localhost:6006

echo ========================================
echo TensorBoard for TaDiCodec Training
echo ========================================
echo.
echo Starting TensorBoard...
echo Open your browser and go to: http://localhost:6006
echo.
echo Press Ctrl+C to stop TensorBoard
echo.

.venv\Scripts\tensorboard.exe --logdir .\logs\TaDiCodec_Japanese_JVS

pause
