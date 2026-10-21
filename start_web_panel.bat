@echo off
REM Start the AI Agent Web Panel Server

echo ╔══════════════════════════════════════════════════════════╗
echo ║        AI Agent Web Panel Launcher                       ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo Installing dependencies...
    python -m pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

REM Check if Flask is installed
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Flask not found. Installing dependencies...
    python -m pip install -r requirements.txt
)

echo.
echo Starting web server...
echo.

REM Start the server
python web\server.py

pause
