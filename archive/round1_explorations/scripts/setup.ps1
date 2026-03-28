# Setup script for Collaboration Circuits project (Windows PowerShell)
# Uses uv for fast, reliable Python package management

Write-Host "=== Collaboration Circuits Setup ===" -ForegroundColor Cyan

# Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..." -ForegroundColor Yellow
    irm https://astral.sh/uv/install.ps1 | iex
}

Write-Host "`nStep 1: Creating virtual environment..." -ForegroundColor Green
uv venv .venv --python 3.11

Write-Host "`nStep 2: Installing PyTorch with CUDA 12.1..." -ForegroundColor Green
# RTX 3080 works great with CUDA 12.1
.\.venv\Scripts\Activate.ps1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "`nStep 3: Installing project dependencies..." -ForegroundColor Green
uv pip install -e ".[dev]"

Write-Host "`nStep 4: Verifying CUDA setup..." -ForegroundColor Green
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "Run test: python scripts/test_setup.py" -ForegroundColor Yellow

