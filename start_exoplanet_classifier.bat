@echo off
echo 🌌 Starting Exoplanet Classifier Application...
echo.

REM Start Backend API in new window
echo 🐍 Starting Backend API on port 8000...
start "Exoplanet Backend API" powershell -NoExit -Command "cd '%~dp0backend'; python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start Frontend in new window
echo ⚛️  Starting Frontend on port 3000...
start "Exoplanet Frontend" powershell -NoExit -Command "cd '%~dp0frontend'; npm run dev"

REM Wait a moment for frontend to start
timeout /t 5 /nobreak >nul

echo.
echo ✅ Application Starting...
echo.
echo 📱 Frontend:  http://localhost:3000
echo 🔌 Backend:   http://localhost:8000
echo 📚 API Docs:  http://localhost:8000/docs
echo.
echo 🌟 Opening web browser...
start http://localhost:3000

echo.
echo Press any key to exit...
pause >nul