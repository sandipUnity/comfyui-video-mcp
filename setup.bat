@echo off
echo =========================================
echo  ComfyUI Video MCP Setup (Skill Edition)
echo  Seedance 2.0 Skills x ComfyUI Pipeline
echo =========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.11+ from python.org
    pause
    exit /b 1
)

:: Check FFmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo WARNING: FFmpeg not found.
    echo Download: https://ffmpeg.org/download.html  ^|  Add ffmpeg/bin to PATH
    echo Montage compilation requires FFmpeg.
    echo.
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

:: Install dependencies
echo.
echo Installing Python dependencies...
pip install --upgrade pip --quiet
pip install -r requirements.txt

:: Create output directories
mkdir output\ideas 2>nul
mkdir output\videos 2>nul
mkdir output\montages 2>nul

echo.
echo =========================================
echo  Step 1: Choose your LLM provider
echo =========================================
echo.
echo OPTION A — No API key (Recommended to start):
echo   Install Ollama from https://ollama.ai
echo   Then run: ollama pull llama3.2
echo   Set in config.yaml: provider: "ollama"
echo.
echo OPTION B — Claude API (Best prompt quality):
echo   Get API key: https://console.anthropic.com
echo   Copy .env.example to .env and add your key
echo   Set in config.yaml: provider: "claude"
echo.
echo OPTION C — Offline / No LLM:
echo   Set in config.yaml: provider: "offline"
echo   Uses built-in skill templates (basic quality)
echo.
echo =========================================
echo  Step 2: Start ComfyUI
echo =========================================
echo   python main.py --listen
echo.
echo =========================================
echo  Step 3: Install ComfyUI custom nodes
echo =========================================
echo   Run: install_comfyui_nodes.bat
echo.
echo   Required nodes:
echo   - ComfyUI-AnimateDiff-Evolved
echo   - ComfyUI-VideoHelperSuite (VHS_VideoCombine)
echo   Optional for better quality:
echo   - ComfyUI-WanVideoWrapper   (Wan2.1 model)
echo   - ComfyUI-Impact             (particle effects)
echo.
echo =========================================
echo  Step 4: Register MCP with Claude Code
echo =========================================
echo   claude mcp add comfyui-video -- python "%CD%\server.py"
echo.
echo =========================================
echo  Setup Complete! Quick start:
echo =========================================
echo.
echo   generate_ideas('tokyo street at night, cyberpunk style')
echo   select_idea(2)
echo   generate_video()
echo   compile_montage(title='Tokyo Nights')
echo.
pause
