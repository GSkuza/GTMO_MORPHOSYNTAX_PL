@echo off
echo ================================================================================
echo REGENERATE ANALYSIS WITH HERBERT EMBEDDINGS
echo ================================================================================
echo.

cd /d D:\GTMO_MORPHOSYNTAX
call .venv\Scripts\activate.bat

echo [1/2] Re-analyzing projekt_poselski_edited with HerBERT...
python gtmo_general_text.py "C:\Users\grzeg\Desktop\projekt poselski edited.md"

echo.
echo [2/2] Checking if herbert_embedding is in JSON...
cd gtmo_results
for /f %%i in ('dir /b /od analysis_*projekt_poselski_edited*') do set LASTDIR=%%i
cd %LASTDIR%

echo.
echo Checking sentence_001.json for herbert_embedding...
type sentence_001.json | findstr /C:"herbert_embedding"

echo.
echo.
echo ================================================================================
echo If you see "herbert_embedding": [...], then it works!
echo Now run: python hybrid_dse_predictor.py
echo ================================================================================
pause
