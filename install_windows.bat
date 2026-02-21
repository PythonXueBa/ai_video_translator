@echo off
chcp 65001 >nul
echo ==========================================
echo AI Video Translator - Windows 安装脚本
echo ==========================================
echo.

:: 检查管理员权限
net session >nul 2>&1
if errorlevel 1 (
    echo [提示] 需要管理员权限来安装 FFmpeg
    echo 请右键点击此文件，选择"以管理员身份运行"
    pause
    exit /b 1
)

echo [1/4] 检查系统环境...

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python
    echo 请从 https://python.org 下载并安装 Python 3.10+
    echo 安装时请勾选 "Add Python to PATH"
    pause
    exit /b 1
)
echo [OK] Python 已安装

:: 检查 FFmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [2/4] 安装 FFmpeg...
    echo 正在下载 FFmpeg...
    
    :: 下载 FFmpeg
    powershell -Command "Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile 'ffmpeg.zip'"
    
    if not exist "ffmpeg.zip" (
        echo [错误] 下载 FFmpeg 失败
        echo 请手动下载并安装 FFmpeg
        pause
        exit /b 1
    )
    
    echo 解压 FFmpeg...
    powershell -Command "Expand-Archive -Path 'ffmpeg.zip' -DestinationPath 'C:\ffmpeg' -Force"
    
    echo 添加 FFmpeg 到系统 PATH...
    setx /M PATH "C:\ffmpeg\ffmpeg-release-essentials\bin;%PATH%"
    
    del ffmpeg.zip
    echo [OK] FFmpeg 安装完成
) else (
    echo [OK] FFmpeg 已安装
)

echo [3/4] 安装 Python 依赖...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [错误] 安装依赖失败
    pause
    exit /b 1
)

echo [4/4] 安装 TTS 库...
pip install qwen-tts
if errorlevel 1 (
    echo [警告] TTS 库安装失败，尝试备用安装方式...
    pip install git+https://github.com/QwenLM/Qwen3-TTS.git
)

echo.
echo ==========================================
echo 安装完成！
echo ==========================================
echo.
echo 使用方法:
echo   python video_tool.py dub
echo.
echo 查看帮助:
echo   python video_tool.py --help
pause
