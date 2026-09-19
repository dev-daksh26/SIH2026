@echo off
echo Starting BharatVault Server...
start http://localhost:8000
uvicorn main:app --reload --port 8000
pause