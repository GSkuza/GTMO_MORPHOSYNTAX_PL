@echo off
REM Local testing script for GTMØ Web Demo (Windows)

echo ====================================
echo GTMO Web Demo - Local Test
echo ====================================
echo.

REM Check if ANTHROPIC_API_KEY is set
if "%ANTHROPIC_API_KEY%"=="" (
    echo WARNING: ANTHROPIC_API_KEY not set
    echo Recommendations will be disabled
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "%CONTINUE%"=="y" exit /b 1
)

REM Start backend
echo 1. Starting backend (port 8000)...
cd demo_webapp\api

REM Check dependencies
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo    Installing dependencies...
    pip install -r requirements.txt
)

REM Start server
start "GTMO Backend" python main.py

REM Wait for backend to start
echo    Waiting for backend to start...
timeout /t 3 /nobreak >nul

REM Check if backend is running
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo    ERROR: Backend failed to start
    exit /b 1
)

echo    ✓ Backend running
echo.

REM Start frontend
cd ..\docs
echo 2. Starting frontend (port 8080)...
start "GTMO Frontend" python -m http.server 8080

timeout /t 2 /nobreak >nul

echo    ✓ Frontend running
echo.

REM Display info
echo ====================================
echo ✅ GTMO Web Demo is running!
echo.
echo URLs:
echo    Backend:  http://localhost:8000
echo    Frontend: http://localhost:8080
echo    API Docs: http://localhost:8000/docs
echo.
echo Test file: sample_document.txt
echo.
echo Press Ctrl+C in server windows to stop
echo ====================================
echo.

pause
