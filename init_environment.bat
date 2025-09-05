@echo off
REM GTMØ Analysis System - Environment Initialization Script for Windows
REM Usage: init_environment.bat

echo ================================================
echo GTMO Analysis System - Environment Setup
echo ================================================

REM Check Python version
echo.
echo Checking Python version...
python --version 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Create virtual environment
echo.
echo Creating virtual environment...
if exist .venv (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q .venv
)

python -m venv .venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo Virtual environment created successfully

REM Activate virtual environment
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo Virtual environment activated

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)
echo Requirements installed successfully

REM Download spaCy model
echo.
echo Downloading spaCy Polish model...
python -m spacy download pl_core_news_lg
if %errorlevel% neq 0 (
    echo Warning: Failed to download large model, trying small model...
    python -m spacy download pl_core_news_sm
    if %errorlevel% neq 0 (
        echo ERROR: Failed to download spaCy model
        pause
        exit /b 1
    )
)
echo spaCy model downloaded successfully

REM Create directory structure
echo.
echo Creating directory structure...
if not exist data mkdir data
if not exist output\gtmo_results mkdir output\gtmo_results
if not exist tests\fixtures mkdir tests\fixtures
echo Directories created successfully

REM Run tests
echo.
echo Running basic tests...
python -c "import morfeusz2; import spacy; import numpy; print('All basic imports successful!')"
if %errorlevel% neq 0 (
    echo ERROR: Basic tests failed
    pause
    exit /b 1
)

echo.
echo ================================================
echo Environment setup completed successfully!
echo ================================================
echo.
echo To activate the environment in the future, run:
echo   .venv\Scripts\activate.bat
echo.
echo To test the system, run:
echo   python gtmo_extended.py
echo.
echo To analyze files, run:
echo   python gtmo_file_loader.py data\ --output results.json
echo.
pause