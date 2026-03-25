#!/bin/bash
# agentic-sense SenseVoice server — one-line setup
# Usage: curl -sSL <raw_url> | bash
#   or:  ./setup.sh

set -e

echo "🎤 Setting up SenseVoice server..."
echo ""

# ── Check dependencies ────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "❌ Python 3 not found. Install it first:"
  echo "   macOS: brew install python3"
  echo "   Ubuntu: sudo apt install python3 python3-pip python3-venv"
  exit 1
fi

if ! command -v ffmpeg &>/dev/null; then
  echo "⚠️  ffmpeg not found. Installing..."
  if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew &>/dev/null; then
      brew install ffmpeg
    else
      echo "❌ Install ffmpeg: brew install ffmpeg"
      exit 1
    fi
  elif command -v apt &>/dev/null; then
    sudo apt update && sudo apt install -y ffmpeg
  elif command -v dnf &>/dev/null; then
    sudo dnf install -y ffmpeg
  else
    echo "❌ Install ffmpeg manually"
    exit 1
  fi
fi

# ── Create venv ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "📦 Creating Python virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ── Install dependencies ──────────────────────────────────────────
echo "📦 Installing dependencies (this may take a few minutes on first run)..."
pip install -q --upgrade pip
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# ── Download models (first run only, ~900MB) ──────────────────────
echo ""
echo "🧠 Models will download on first run (~900MB total):"
echo "   • SenseVoiceSmall (~893MB)"
echo "   • FSMN-VAD (~1.6MB)"
echo ""

# ── Start server ──────────────────────────────────────────────────
echo "🚀 Starting SenseVoice server..."
echo ""
python3 "$SCRIPT_DIR/sensevoice-server.py" "$@"
