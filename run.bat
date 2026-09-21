@echo off
echo Starting BharatVault Server...
echo (waiting a few seconds for the server to finish starting before opening the browser...)
start /min cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8000"
uvicorn main:app --reload --port 8000
pause
