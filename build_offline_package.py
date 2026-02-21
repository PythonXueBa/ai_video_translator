#!/usr/bin/env python3
"""
AI Video Translator - 离线安装包构建脚本
构建包含所有依赖的 Windows 离线安装包
"""

import os
import sys
import subprocess
import shutil
import zipfile
from pathlib import Path

# 配置
APP_NAME = "AI_Video_Translator"
APP_VERSION = "1.0.0"
BUILD_DIR = "build_offline"
DIST_DIR = "dist_offline"

def run_command(cmd, description=""):
    """运行命令并显示输出"""
    if description:
        print(f"\n{'='*60}")
        print(f"{description}")
        print('='*60)

    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return False

    if result.stdout:
        print(result.stdout)
    return True

def clean_build():
    """清理构建目录"""
    print("\n清理构建目录...")
    dirs_to_clean = [BUILD_DIR, DIST_DIR, "build", "dist"]
    for d in dirs_to_clean:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"  删除: {d}")

def create_directory_structure():
    """创建目录结构"""
    print("\n创建目录结构...")
    dirs = [
        f"{BUILD_DIR}/{APP_NAME}",
        f"{BUILD_DIR}/{APP_NAME}/models",
        f"{BUILD_DIR}/{APP_NAME}/ffmpeg",
        f"{BUILD_DIR}/{APP_NAME}/python",
        f"{BUILD_DIR}/{APP_NAME}/src",
        f"{BUILD_DIR}/{APP_NAME}/data",
        DIST_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  创建: {d}")

def copy_source_files():
    """复制源代码文件"""
    print("\n复制源代码...")

    files_to_copy = [
        "video_tool.py",
        "gui_tool.py",
        "diagnose.py",
        "requirements.txt",
        "README.md",
    ]

    for f in files_to_copy:
        if os.path.exists(f):
            shutil.copy2(f, f"{BUILD_DIR}/{APP_NAME}/")
            print(f"  复制: {f}")

    # 复制 src 目录
    if os.path.exists("src"):
        for item in os.listdir("src"):
            src_path = os.path.join("src", item)
            dst_path = os.path.join(f"{BUILD_DIR}/{APP_NAME}/src", item)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"  复制: src/{item}")

def create_launcher():
    """创建启动器脚本"""
    print("\n创建启动器...")

    # GUI 启动器
    gui_launcher = f'''{APP_NAME}.bat
@echo off
chcp 65001 >nul
title AI Video Translator
cd /d "%~dp0"

echo ==========================================
echo AI Video Translator v{APP_VERSION}
echo ==========================================
echo.

:: 设置环境变量
set "APP_DIR=%~dp0"
set "PYTHON_DIR=%APP_DIR%python"
set "FFMPEG_DIR=%APP_DIR%ffmpeg\bin"
set "MODELS_DIR=%APP_DIR%models"

:: 添加 FFmpeg 到 PATH
set "PATH=%FFMPEG_DIR%;%PATH%"

:: 设置模型缓存目录
set "TRANSFORMERS_CACHE=%MODELS_DIR%"
set "HF_HOME=%MODELS_DIR%"
set "TORCH_HOME=%MODELS_DIR%"

:: 检查 Python
if exist "%PYTHON_DIR%\python.exe" (
    set "PYTHON=%PYTHON_DIR%\python.exe"
) else (
    echo [错误] 未找到 Python
    pause
    exit /b 1
)

:: 启动 GUI
echo 启动 AI Video Translator...
"%PYTHON%" gui_tool.py

pause
'''

    # CLI 启动器
    cli_launcher = f'''CLI_{APP_NAME}.bat
@echo off
chcp 65001 >nul
title AI Video Translator - CLI
cd /d "%~dp0"

:: 设置环境变量
set "APP_DIR=%~dp0"
set "PYTHON_DIR=%APP_DIR%python"
set "FFMPEG_DIR=%APP_DIR%ffmpeg\bin"
set "MODELS_DIR=%APP_DIR%models"

set "PATH=%FFMPEG_DIR%;%PATH%"
set "TRANSFORMERS_CACHE=%MODELS_DIR%"
set "HF_HOME=%MODELS_DIR%"
set "TORCH_HOME=%MODELS_DIR%"

if exist "%PYTHON_DIR%\python.exe" (
    set "PYTHON=%PYTHON_DIR%\python.exe"
) else (
    echo [错误] 未找到 Python
    pause
    exit /b 1
)

echo AI Video Translator v{APP_VERSION}
echo.
echo 使用方法:
echo   video_tool.py dub [视频文件] [选项]
echo.
echo 示例:
echo   video_tool.py dub video.mp4
echo   video_tool.py dub video.mp4 --source-lang en --target-lang zh
echo.

"%PYTHON%" video_tool.py --help

echo.
set /p CMD="输入命令: "
"%PYTHON%" %CMD%

pause
'''

    with open(f"{BUILD_DIR}/{APP_NAME}/{APP_NAME}.bat", "w", encoding="utf-8") as f:
        f.write(gui_launcher)
    print(f"  创建: {APP_NAME}.bat")

    with open(f"{BUILD_DIR}/{APP_NAME}/CLI_{APP_NAME}.bat", "w", encoding="utf-8") as f:
        f.write(cli_launcher)
    print(f"  创建: CLI_{APP_NAME}.bat")

def create_installer_script():
    """创建安装脚本"""
    print("\n创建安装脚本...")

    install_script = '''install.bat
@echo off
chcp 65001 >nul
title AI Video Translator - 安装程序
cd /d "%~dp0"

echo ==========================================
echo AI Video Translator v1.0.0 - 安装程序
echo ==========================================
echo.

:: 检查管理员权限
net session >nul 2>&1
if errorlevel 1 (
    echo [提示] 需要管理员权限来创建快捷方式
    echo 请右键点击此文件，选择"以管理员身份运行"
    echo.
)

set "INSTALL_DIR=%ProgramFiles%\AI_Video_Translator"
set "SHORTCUT_DIR=%ProgramData%\Microsoft\Windows\Start Menu\Programs"

echo 安装目录: %INSTALL_DIR%
echo.

:: 创建安装目录
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

:: 复制文件
echo [1/3] 复制文件...
xcopy /E /I /Y "AI_Video_Translator\*" "%INSTALL_DIR%\" >nul
echo 完成!

:: 创建快捷方式
echo [2/3] 创建快捷方式...
(
echo Set WshShell = CreateObject^("WScript.Shell"^)
echo Set oLink = WshShell.CreateShortcut^("%SHORTCUT_DIR%\AI Video Translator.lnk"^)
echo oLink.TargetPath = "%INSTALL_DIR%\AI_Video_Translator.bat"
echo oLink.WorkingDirectory = "%INSTALL_DIR%"
echo oLink.IconLocation = "%INSTALL_DIR%\icon.ico"
echo oLink.Save
) > "%TEMP%\CreateShortcut.vbs"

cscript //nologo "%TEMP%\CreateShortcut.vbs"
del "%TEMP%\CreateShortcut.vbs"
echo 完成!

:: 创建桌面快捷方式
(
echo Set WshShell = CreateObject^("WScript.Shell"^)
echo desktop = WshShell.SpecialFolders^("Desktop"^)
echo Set oLink = WshShell.CreateShortcut^(desktop ^& "\AI Video Translator.lnk"^)
echo oLink.TargetPath = "%INSTALL_DIR%\AI_Video_Translator.bat"
echo oLink.WorkingDirectory = "%INSTALL_DIR%"
echo oLink.IconLocation = "%INSTALL_DIR%\icon.ico"
echo oLink.Save
) > "%TEMP%\CreateDesktopShortcut.vbs"

cscript //nologo "%TEMP%\CreateDesktopShortcut.vbs"
del "%TEMP%\CreateDesktopShortcut.vbs"

echo [3/3] 安装完成!
echo.
echo ==========================================
echo 安装成功!
echo ==========================================
echo.
echo 启动方式:
echo   - 开始菜单: AI Video Translator
echo   - 桌面快捷方式
echo   - 安装目录: %INSTALL_DIR%
echo.
pause
'''

    with open(f"{BUILD_DIR}/install.bat", "w", encoding="utf-8") as f:
        f.write(install_script)
    print("  创建: install.bat")

def create_readme():
    """创建使用说明"""
    print("\n创建使用说明...")

    readme = '''README.txt
================================================================================
AI Video Translator v1.0.0 - 离线安装包
================================================================================

基于 Qwen3-TTS 的高质量多语言视频翻译与配音解决方案

--------------------------------------------------------------------------------
系统要求
--------------------------------------------------------------------------------
- Windows 10/11 64位
- 8GB+ 内存 (推荐 16GB)
- 10GB+ 磁盘空间
- NVIDIA GPU 推荐 (支持 CUDA)

--------------------------------------------------------------------------------
安装方法
--------------------------------------------------------------------------------

方法1: 直接运行 (绿色版)
1. 解压本压缩包到任意目录
2. 双击运行 "AI_Video_Translator.bat"
3. 首次运行会自动下载模型文件

方法2: 安装到系统
1. 右键点击 "install.bat"，选择"以管理员身份运行"
2. 等待安装完成
3. 从开始菜单或桌面快捷方式启动

--------------------------------------------------------------------------------
使用方法
--------------------------------------------------------------------------------

图形界面:
1. 双击 "AI_Video_Translator.bat"
2. 选择视频文件
3. 选择源语言和目标语言
4. 点击"开始 AI 配音"

命令行:
1. 双击 "CLI_AI_Video_Translator.bat"
2. 输入命令，例如:
   video_tool.py dub video.mp4
   video_tool.py dub video.mp4 --source-lang en --target-lang zh

--------------------------------------------------------------------------------
支持的语言
--------------------------------------------------------------------------------
- zh: 中文
- en: 英文
- ja: 日文
- ko: 韩文
- es: 西班牙文
- fr: 法文
- de: 德文
- ru: 俄文
- pt: 葡萄牙文
- it: 意大利文
- ar: 阿拉伯文
- hi: 印地文
- vi: 越南文
- th: 泰文
- id: 印尼文

--------------------------------------------------------------------------------
文件说明
--------------------------------------------------------------------------------
- AI_Video_Translator.bat    : 图形界面启动器
- CLI_AI_Video_Translator.bat: 命令行启动器
- install.bat                : 系统安装程序
- python/                    : 嵌入式 Python 环境
- ffmpeg/                    : FFmpeg 视频处理工具
- models/                    : AI 模型文件 (首次运行时下载)
- src/                       : 源代码

--------------------------------------------------------------------------------
常见问题
--------------------------------------------------------------------------------

Q: 首次启动很慢？
A: 首次运行需要下载 AI 模型文件 (约 5-10GB)，请保持网络连接。

Q: 可以使用离线模型吗？
A: 模型文件下载后会保存在 models/ 目录，之后可以离线使用。

Q: 支持哪些视频格式？
A: MP4, AVI, MKV, MOV, WMV, FLV 等常见格式。

Q: 处理速度如何？
A: 取决于硬件配置。GPU 加速比 CPU 快 5-10 倍。

--------------------------------------------------------------------------------
技术支持
--------------------------------------------------------------------------------
GitHub: https://github.com/PythonXueBa/ai_video_translator

================================================================================
'''

    with open(f"{BUILD_DIR}/{APP_NAME}/README.txt", "w", encoding="utf-8") as f:
        f.write(readme)
    print("  创建: README.txt")

def download_embedded_python():
    """下载嵌入式 Python"""
    print("\n下载嵌入式 Python...")

    python_url = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
    python_zip = f"{BUILD_DIR}/python.zip"

    # 使用 urllib 下载
    import urllib.request

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\r  进度: {percent:.1f}%", end="", flush=True)

    try:
        print(f"  下载: {python_url}")
        urllib.request.urlretrieve(python_url, python_zip, reporthook=report_progress)
        print("\n  下载完成!")

        # 解压
        python_dir = f"{BUILD_DIR}/{APP_NAME}/python"
        print(f"  解压到: {python_dir}")

        import zipfile
        with zipfile.ZipFile(python_zip, 'r') as zip_ref:
            zip_ref.extractall(python_dir)

        os.remove(python_zip)
        print("  解压完成!")

        # 修改 python310._pth 文件以启用 site-packages
        pth_file = os.path.join(python_dir, "python310._pth")
        if os.path.exists(pth_file):
            with open(pth_file, "r") as f:
                content = f.read()
            content = content.replace("#import site", "import site")
            with open(pth_file, "w") as f:
                f.write(content)
            print("  启用 site-packages")

        return True
    except Exception as e:
        print(f"\n  错误: {e}")
        return False

def install_dependencies():
    """安装 Python 依赖"""
    print("\n安装 Python 依赖...")

    python_dir = f"{BUILD_DIR}/{APP_NAME}/python"
    python_exe = os.path.join(python_dir, "python.exe")

    if not os.path.exists(python_exe):
        print(f"  错误: 未找到 {python_exe}")
        return False

    # 安装 pip
    print("  安装 pip...")
    get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
    get_pip_path = f"{BUILD_DIR}/get-pip.py"

    import urllib.request
    urllib.request.urlretrieve(get_pip_url, get_pip_path)

    subprocess.run([python_exe, get_pip_path], check=True)
    os.remove(get_pip_path)

    # 安装依赖
    pip_exe = os.path.join(python_dir, "Scripts", "pip.exe")

    packages = [
        "torch",
        "torchaudio",
        "transformers",
        "sentencepiece",
        "sacremoses",
        "openai-whisper",
        "demucs",
        "soundfile",
        "pydub",
        "numpy",
        "scipy",
        "psutil",
        "tqdm",
        "qwen-tts",
    ]

    for pkg in packages:
        print(f"  安装 {pkg}...")
        result = subprocess.run(
            [pip_exe, "install", "--no-cache-dir", pkg],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"    警告: {pkg} 安装失败")
            print(f"    {result.stderr[:200]}")

    return True

def download_ffmpeg():
    """下载 FFmpeg"""
    print("\n下载 FFmpeg...")

    ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    ffmpeg_zip = f"{BUILD_DIR}/ffmpeg.zip"

    import urllib.request
    import zipfile

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\r  进度: {percent:.1f}%", end="", flush=True)

    try:
        print(f"  下载: {ffmpeg_url}")
        urllib.request.urlretrieve(ffmpeg_url, ffmpeg_zip, reporthook=report_progress)
        print("\n  下载完成!")

        # 解压
        ffmpeg_temp = f"{BUILD_DIR}/ffmpeg_temp"
        print(f"  解压...")
        with zipfile.ZipFile(ffmpeg_zip, 'r') as zip_ref:
            zip_ref.extractall(ffmpeg_temp)

        # 找到解压后的目录并重命名
        for item in os.listdir(ffmpeg_temp):
            if item.startswith("ffmpeg-"):
                src = os.path.join(ffmpeg_temp, item)
                dst = f"{BUILD_DIR}/{APP_NAME}/ffmpeg"
                shutil.move(src, dst)
                break

        shutil.rmtree(ffmpeg_temp)
        os.remove(ffmpeg_zip)
        print("  FFmpeg 安装完成!")
        return True
    except Exception as e:
        print(f"\n  错误: {e}")
        return False

def create_package():
    """创建最终安装包"""
    print("\n创建安装包...")

    # 创建 ZIP 压缩包
    zip_name = f"{DIST_DIR}/{APP_NAME}_v{APP_VERSION}_Windows.zip"

    print(f"  创建: {zip_name}")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(f"{BUILD_DIR}"):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, BUILD_DIR)
                zipf.write(file_path, arcname)

    # 计算大小
    size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"  大小: {size_mb:.1f} MB")
    print(f"  完成!")

def main():
    print("=" * 60)
    print("AI Video Translator - 离线安装包构建工具")
    print("=" * 60)

    if sys.platform != "win32":
        print("\n警告: 此脚本用于构建 Windows 安装包")
        print("当前系统:", sys.platform)
        response = input("\n是否继续? (y/n): ")
        if response.lower() != 'y':
            return

    # 选择构建模式
    print("\n构建选项:")
    print("  1. 完整包 (包含 Python + FFmpeg + 依赖) - 约 3-5GB")
    print("  2. 精简包 (仅程序和启动器，需用户自行安装 Python) - 约 50MB")
    print("  3. 仅创建启动脚本")

    choice = input("\n请选择 (1/2/3): ").strip()

    if choice == "1":
        # 完整包
        clean_build()
        create_directory_structure()
        copy_source_files()
        create_launcher()
        create_installer_script()
        create_readme()

        # 下载嵌入式 Python
        if not download_embedded_python():
            print("\n错误: 下载 Python 失败")
            return

        # 安装依赖
        if not install_dependencies():
            print("\n错误: 安装依赖失败")
            return

        # 下载 FFmpeg
        if not download_ffmpeg():
            print("\n警告: 下载 FFmpeg 失败")

        create_package()

    elif choice == "2":
        # 精简包
        clean_build()
        create_directory_structure()
        copy_source_files()
        create_launcher()
        create_installer_script()
        create_readme()
        create_package()

    elif choice == "3":
        # 仅启动脚本
        create_launcher()
        create_installer_script()
        print("\n启动脚本已创建在当前目录")

    else:
        print("无效选择")
        return

    print("\n" + "=" * 60)
    print("构建完成!")
    print("=" * 60)
    print(f"\n输出目录: {DIST_DIR}/")
    print(f"\n使用方法:")
    print(f"  1. 解压 {APP_NAME}_v{APP_VERSION}_Windows.zip")
    print(f"  2. 运行 AI_Video_Translator.bat")

if __name__ == "__main__":
    main()
