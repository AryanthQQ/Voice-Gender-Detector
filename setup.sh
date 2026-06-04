#!/bin/bash
# ============================================================
# setup.sh - One-click server setup for Voice Gender App
# Run: bash setup.sh
# ============================================================

echo "=== Voice Gender App - Server Setup ==="

# 1. Install Python dependencies
echo "[1/5] Installing Python packages..."
pip3 install -r requirements.txt

# 2. Create necessary directories
echo "[2/5] Creating directories..."
mkdir -p recordings models

# 3. Train ML models (if not already trained)
if [ ! -f "models/svm_model.pkl" ]; then
    echo "[3/5] Training ML models (this takes ~2 min)..."
    python3 train_model.py
else
    echo "[3/5] Models already exist, skipping training."
fi

# 4. Create .env if not exists
if [ ! -f ".env" ]; then
    echo "[4/5] Creating .env from template..."
    cp .env.example .env
    echo ">>> IMPORTANT: Edit .env and add your Telegram credentials! <<<"
else
    echo "[4/5] .env already exists."
fi

# 5. Start the app
echo "[5/5] Starting Voice Gender App on port 8000..."
nohup python3 main.py > app.log 2>&1 &
echo $! > app.pid
echo ""
echo "=== Setup Complete! ==="
echo "App running at: http://localhost:8000"
echo "Logs: tail -f app.log"
echo "Stop: kill \$(cat app.pid)"
