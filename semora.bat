@echo off
title DayMark Server

echo =====================================
echo Starting DayMark Student OS
echo =====================================

REM Activate virtual environment
call venv\Scripts\activate

REM Get local IP address
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4"') do set IP=%%a
set IP=%IP: =%

echo.
echo =====================================
echo Server will be accessible at:
echo.
echo Laptop:  http://127.0.0.1:8000
echo Phone:   http://%IP%:8000
echo =====================================
echo.

REM Start FastAPI
uvicorn semora:app --host 0.0.0.0 --port 8000 --reload

pause