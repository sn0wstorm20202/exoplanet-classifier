Write-Host "🌌 Starting Exoplanet Classifier Application..." -ForegroundColor Cyan
Write-Host ""

# Get the script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Start Backend API in new window
Write-Host "🐍 Starting Backend API on port 8000..." -ForegroundColor Green
$BackendPath = Join-Path $ScriptDir "backend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$BackendPath'; python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload" -WindowStyle Normal

# Wait a moment for backend to start
Start-Sleep -Seconds 3

# Start Frontend in new window  
Write-Host "⚛️  Starting Frontend on port 3000..." -ForegroundColor Blue
$FrontendPath = Join-Path $ScriptDir "frontend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$FrontendPath'; npm run dev" -WindowStyle Normal

# Wait a moment for frontend to start
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "✅ Application Starting..." -ForegroundColor Green
Write-Host ""
Write-Host "📱 Frontend:  http://localhost:3000" -ForegroundColor Yellow
Write-Host "🔌 Backend:   http://localhost:8000" -ForegroundColor Yellow  
Write-Host "📚 API Docs:  http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host ""

Write-Host "🌟 Opening web browser..." -ForegroundColor Magenta
Start-Process "http://localhost:3000"

Write-Host ""
Write-Host "🎉 Exoplanet Classifier is now running!" -ForegroundColor Green
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")