# Single-command launcher for the job search demo (PowerShell)
$ErrorActionPreference = "Stop"

# Check for OpenAI API key
if (-not $env:OPENAI_API_KEY) {
    Write-Error "ERROR: OPENAI_API_KEY environment variable not set"
    Write-Host "Please set it with: `$env:OPENAI_API_KEY = 'your-key-here'"
    exit 1
}

# Check for uv, install if missing
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv package manager..."
    irm https://astral.sh/uv/install.ps1 | iex
}

# Install dependencies and run demo
uv sync --quiet
uv run python demo.py
