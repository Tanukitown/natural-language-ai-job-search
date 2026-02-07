#!/bin/bash
# Single-command launcher for the job search demo
set -e

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable not set"
    echo "Please set it with: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

# Check for uv, install if missing
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install dependencies and run demo
uv sync --quiet
uv run python demo.py
