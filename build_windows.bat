@echo off
chcp 65001 >nul
echo ==========================================
echo AI Video Translator - Windows 打包脚本
echo ==========================================
echo.

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

:: 安装打包依赖
echo [1/5] 安装打包工具...
pip install pyinstaller -q
if errorlevel 1 (
    echo [错误] 安装 pyinstaller 失败
    pause
    exit /b 1
)

:: 安装项目依赖
echo [2/5] 安装项目依赖...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [警告] 部分依赖安装失败，继续尝试打包...
)

:: 安装 TTS 库
echo [3/5] 安装 TTS 库...
pip install qwen-tts -q
if errorlevel 1 (
    echo [警告] TTS 库安装失败，继续尝试打包...
)

:: 创建打包目录
echo [4/5] 准备打包目录...
if not exist "build" mkdir build
if not exist "dist" mkdir dist

:: 打包可执行文件
echo [5/5] 开始打包...
pyinstaller ^
    --name="AI_Video_Translator" ^
    --onefile ^
    --console ^
    --add-data "src;src" ^
    --add-data "requirements.txt;." ^
    --hidden-import=torch ^
    --hidden-import=transformers ^
    --hidden-import=whisper ^
    --hidden-import=demucs ^
    --hidden-import=soundfile ^
    --hidden-import=pydub ^
    --hidden-import=numpy ^
    --hidden-import=scipy ^
    --hidden-import=psutil ^
    --hidden-import=tqdm ^
    --hidden-import=qwen_tts ^
    --collect-all=torch ^
    --collect-all=transformers ^
    --collect-all=whisper ^
    --collect-all=demucs ^
    video_tool.py

if errorlevel 1 (
    echo [错误] 打包失败
    pause
    exit /b 1
)

echo.
echo ==========================================
echo 打包完成！
echo 输出文件: dist\AI_Video_Translator.exe
echo ==========================================
pause
