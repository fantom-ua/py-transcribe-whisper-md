@echo off
title Transcribe Audio

echo ========================================
echo   MP3 Transcription (Whisper)
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install from https://python.org
    pause
    exit /b 1
)

python -c "import faster_whisper" >nul 2>&1
if errorlevel 1 (
    echo Installing faster-whisper...
    pip install faster-whisper
    echo.
)

python "%~dp0transcribe.py"

echo.
pause
