@echo off
:loop
echo [%time%] Starting server... >> crash_loop.log
C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 >> crash_loop.log 2>&1
echo [%time%] Server exited with code %errorlevel% >> crash_loop.log
timeout /t 2 >nul
goto loop
