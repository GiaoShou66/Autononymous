#!/bin/bash
# Start the AI Agent Web Panel Server

echo "╔══════════════════════════════════════════════════════════╗"
echo "║        AI Agent Web Panel Launcher                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Installing dependencies..."
    python -m pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Check if Flask is installed
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Flask not found. Installing dependencies..."
    python -m pip install -r requirements.txt
fi

echo ""
echo "Starting web server..."
echo ""

# Start the server
python web/server.py
