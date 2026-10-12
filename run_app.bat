@echo off
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
    echo Khong tim thay Python trong PATH. Hay cai Python va chon Add Python to PATH.
    pause
    exit /b 1
)
python src\faceautovn\main.py %*
if errorlevel 1 (
    echo.
    echo App da dung. Neu thieu thu vien, chay: python -m pip install -r requirements.txt
    pause
    exit /b 1
)
