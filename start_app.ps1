Write-Host "🌌 Starting Exoplanet Classifier..." -ForegroundColor Cyan
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Start Backend
Write-Host "🐍 Starting Backend API..." -ForegroundColor Green
$BackendPath = Join-Path $ScriptDir "backend"
Start-Process powershell -ArgumentList "-NoExit", "-WindowStyle", "Minimized", "-Command", "cd '$BackendPath'; python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload"

# Wait for backend
Start-Sleep 3

# Start Frontend  
Write-Host "⚛️ Starting Frontend..." -ForegroundColor Blue
$FrontendPath = Join-Path $ScriptDir "frontend"
Start-Process powershell -ArgumentList "-NoExit", "-WindowStyle", "Minimized", "-Command", "cd '$FrontendPath'; npm run dev"

# Wait for services to start
Start-Sleep 5

Write-Host "✅ Services Starting..." -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Frontend:  http://localhost:3000" -ForegroundColor Yellow
Write-Host "🔌 API:      http://localhost:8000" -ForegroundColor Yellow  
Write-Host "📚 Docs:     http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host ""

# Open browser
Write-Host "🚀 Opening browser..." -ForegroundColor Magenta
Start-Process "http://localhost:3000"

Write-Host "🎉 Exoplanet Classifier is running!" -ForegroundColor Green