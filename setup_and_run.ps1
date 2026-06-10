$ErrorActionPreference = "Stop"

Write-Host "== Agent Workflow FastAPI setup ==" -ForegroundColor Cyan

if (!(Test-Path ".venv")) {
  Write-Host "Creating virtual environment..."
  python -m venv .venv
}

if (!(Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
}

Write-Host "Installing dependencies..."
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

$deepseekKey = Read-Host "Enter DeepSeek API Key"
$jimengKey = Read-Host "Enter Jimeng / Wanjie API Key, or press Enter to skip"
$dashscopeKey = Read-Host "Enter Alibaba Bailian DashScope API Key, or press Enter to skip"

$envText = Get-Content ".env" -Raw -Encoding UTF8
$envText = $envText -replace "(?m)^TEXT_API_KEY=.*$", "TEXT_API_KEY=$deepseekKey"
$envText = $envText -replace "(?m)^JIMENG_API_KEY=.*$", "JIMENG_API_KEY=$jimengKey"
$envText = $envText -replace "(?m)^DASHSCOPE_API_KEY=.*$", "DASHSCOPE_API_KEY=$dashscopeKey"
$envText = $envText -replace "(?m)^IMAGE_PROVIDER=.*$", "IMAGE_PROVIDER=qwen"
$envText = $envText -replace "(?m)^IMAGE_MODEL=.*$", "IMAGE_MODEL=qwen-image-2.0"
$envText = $envText -replace "(?m)^VIDEO_PROVIDER=.*$", "VIDEO_PROVIDER=qwen"
$envText = $envText -replace "(?m)^VIDEO_MODEL=.*$", "VIDEO_MODEL=happyhorse-1.0-i2v"
Set-Content ".env" $envText -Encoding UTF8

Write-Host ""
Write-Host "Config saved to .env. Starting server..." -ForegroundColor Green
Write-Host "Open http://127.0.0.1:8888" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow

.\.venv\Scripts\python.exe run.py
