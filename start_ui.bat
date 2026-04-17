@echo off
title ComfyUI Video Pipeline UI
echo.
echo  Starting ComfyUI Video Pipeline UI...
echo  Open your browser at: http://localhost:8501
echo  Press Ctrl+C to stop.
echo.
cd /d "%~dp0"
venv\Scripts\streamlit run app.py --server.port 8501 --server.headless false
pause
