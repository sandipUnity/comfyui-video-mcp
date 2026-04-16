@echo off
echo =========================================
echo  ComfyUI Custom Nodes Installer
echo =========================================
echo.
echo This script clones required custom nodes into your ComfyUI installation.
echo.

set /p COMFYUI_PATH="Enter your ComfyUI path (e.g. C:\ComfyUI): "

if not exist "%COMFYUI_PATH%\custom_nodes" (
    echo ERROR: custom_nodes folder not found at %COMFYUI_PATH%
    echo Make sure ComfyUI is installed at that path.
    pause
    exit /b 1
)

cd /d "%COMFYUI_PATH%\custom_nodes"

echo.
echo Installing ComfyUI-AnimateDiff-Evolved...
if not exist "ComfyUI-AnimateDiff-Evolved" (
    git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git
) else (
    echo Already installed, updating...
    cd ComfyUI-AnimateDiff-Evolved && git pull && cd ..
)

echo.
echo Installing ComfyUI-VideoHelperSuite (VHS)...
if not exist "ComfyUI-VideoHelperSuite" (
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
) else (
    echo Already installed, updating...
    cd ComfyUI-VideoHelperSuite && git pull && cd ..
)

echo.
echo Installing ComfyUI-Advanced-ControlNet...
if not exist "ComfyUI-Advanced-ControlNet" (
    git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git
) else (
    echo Already installed, updating...
    cd ComfyUI-Advanced-ControlNet && git pull && cd ..
)

echo.
echo Installing ComfyUI-WanVideoWrapper (for Wan2.1)...
if not exist "ComfyUI-WanVideoWrapper" (
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
) else (
    echo Already installed, updating...
    cd ComfyUI-WanVideoWrapper && git pull && cd ..
)

echo.
echo Installing node dependencies...
cd "%COMFYUI_PATH%"
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

for %%n in (
    ComfyUI-AnimateDiff-Evolved
    ComfyUI-VideoHelperSuite
    ComfyUI-Advanced-ControlNet
    ComfyUI-WanVideoWrapper
) do (
    if exist "custom_nodes\%%n\requirements.txt" (
        echo Installing deps for %%n...
        pip install -r "custom_nodes\%%n\requirements.txt" --quiet
    )
)

echo.
echo =========================================
echo  Nodes installed!
echo =========================================
echo.
echo Now download models:
echo.
echo AnimateDiff motion modules -> ComfyUI\models\animatediff_models\
echo   mm_sd_v15_v2.ckpt (from HuggingFace: guoyww/animatediff)
echo.
echo SD checkpoints -> ComfyUI\models\checkpoints\
echo   v1-5-pruned-emaonly.safetensors (stable-diffusion-v1-5)
echo.
echo Wan2.1 models -> ComfyUI\models\diffusion_models\
echo   Wan2.1-T2V-1.3B (from HuggingFace: Wan-AI/Wan2.1-T2V-1.3B)
echo.
pause
