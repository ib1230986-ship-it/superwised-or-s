@echo off
cd /d "%~dp0"
echo Server start ho raha hai... browser mein http://localhost:8000 kholein
python -m uvicorn main:app --port 8000
pause
