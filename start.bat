@echo off
echo 🌿 Cotton Disease Prediction Application
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo ❌ Error: requirements.txt not found
    echo Please ensure you're in the correct directory
    pause
    exit /b 1
)

REM Check if cotton dataset exists
if not exist "cotton" (
    echo ❌ Error: cotton dataset folder not found
    echo Please ensure your dataset is in the 'cotton' folder
    pause
    exit /b 1
)

echo 📁 Dataset folder found
echo.

REM Install requirements if needed
echo 📦 Checking and installing required packages...
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install packages
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo ✅ Packages installed successfully
echo.

REM Start the application
echo 🚀 Starting the application...
echo The web interface will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python app.py

pause 