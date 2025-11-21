@echo off
echo Testing HerBERT embedding in JSON output...
cd /d D:\GTMO_MORPHOSYNTAX
call .venv\Scripts\activate.bat

echo.
echo [1] Analyzing test sentence...
python gtmo_general_text.py "C:\Users\grzeg\Desktop\RP KONSTYTUCJA 09112025.md"

echo.
echo [2] Checking if herbert_embedding is in JSON...
cd gtmo_results
for /f %%i in ('dir /b /od analysis_*') do set LASTDIR=%%i
cd %LASTDIR%

echo.
echo Checking first sentence file...
type sentence_001.json | findstr /C:"herbert_embedding"

echo.
echo.
echo If you see "herbert_embedding": [...], then it works!
pause
