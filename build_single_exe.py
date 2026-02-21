#!/usr/bin/env python3
"""
AI Video Translator - 单文件EXE构建脚本
构建包含所有依赖的独立可执行文件
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
BUILD_DIR = "build_exe"
DIST_DIR = "dist_exe"

def clean():
    """清理构建目录"""
    print("清理构建目录...")
    for d in [BUILD_DIR, DIST_DIR, "build", "dist"]:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"  删除: {d}")

def install_pyinstaller():
    """安装PyInstaller"""
    print("\n检查 PyInstaller...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "pyinstaller"],
        capture_output=True
    )
    if result.returncode != 0:
        print("安装 PyInstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

# 基础目录
base_dir = os.path.abspath(os.path.dirname(__file__))

def get_torch_lib_path():
    """获取torch库路径"""
    try:
        import torch
        return os.path.dirname(torch.__file__)
    except:
        return None

def get_ffmpeg_binary():
    """查找FFmpeg可执行文件"""
    ffmpeg_paths = [
        "ffmpeg.exe",
        "C:/ffmpeg/bin/ffmpeg.exe",
        "C:/Program Files/ffmpeg/bin/ffmpeg.exe",
    ]
    for path in ffmpeg_paths:
        if os.path.exists(path):
            return path
    return None

def create_spec_file():
    """创建PyInstaller spec文件"""
    print("\n创建 spec 文件...")

    # 查找FFmpeg
    ffmpeg_path = get_ffmpeg_binary()
    if ffmpeg_path:
        print(f"  FFmpeg路径: {ffmpeg_path}")
        ffmpeg_binaries = [(ffmpeg_path, '.')]
    else:
        print("  ⚠️ 未找到FFmpeg，将不包含在打包中")
        ffmpeg_binaries = []

    # 钩子路径
    hookspath = [os.path.join(base_dir, 'hooks')]

    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-
import sys
import os

block_cipher = None

# 基础目录
base_dir = r'{base_dir}'

# 添加的数据文件
added_files = [
    (r'{os.path.join(base_dir, "src")}', 'src'),
    (r'{os.path.join(base_dir, "video_tool.py")}', '.'),
    (r'{os.path.join(base_dir, "requirements.txt")}', '.'),
    (r'{os.path.join(base_dir, "README.md")}', '.'),
]

# 二进制文件 (FFmpeg等)
binaries = {ffmpeg_binaries}

# 隐藏导入 - 关键依赖
hidden_imports = [
    # 核心库
    'torch',
    'torchaudio',
    'torchvision',
    'transformers',
    'transformers.models',
    'transformers.models.m2m100',
    'transformers.models.m2m100.modeling_m2m100',
    'transformers.models.m2m100.tokenization_m2m100',
    'transformers.models.whisper',
    'transformers.models.whisper.modeling_whisper',
    'transformers.models.whisper.tokenization_whisper',
    # ASR
    'whisper',
    'whisper.model',
    'whisper.decoder',
    'whisper.tokenizer',
    'whisper.audio',
    'whisper.utils',
    # 音频处理
    'demucs',
    'demucs.model',
    'demucs.pretrained',
    'demucs.separate',
    'demucs.apply',
    'soundfile',
    'soundfile_compat',
    'pydub',
    'pydub.audio_segment',
    'pydub.effects',
    'librosa',
    'librosa.core',
    'librosa.feature',
    # 科学计算
    'numpy',
    'scipy',
    'scipy.signal',
    # 其他
    'psutil',
    'tqdm',
    'sentencepiece',
    'sacremoses',
    'einops',
    'tokenizers',
    'regex',
    'packaging',
    'yaml',
    'requests',
    'urllib3',
    'certifi',
    'charset_normalizer',
    # 项目模块
    'src.config',
    'src.analyzer',
    'src.extractor',
    'src.separator',
    'src.asr_module',
    'src.translator_m2m100',
    'src.tts_qwen3',
    'src.subtitle_handler',
    'src.subtitle_tts_engine',
    'src.merger',
    'src.video_processor',
    'src.performance_config',
    'src.memory_manager',
    'src.splitter',
]

# 排除不必要的包以减少体积
excludes = [
    'matplotlib',
    'PIL',
    'Pillow',
    'tkinter.test',
    'unittest',
    'pydoc',
    'pdb',
    'doctest',
    'test',
    'tests',
    '_pytest',
    'pytest',
    'mypy',
    'IPython',
    'jupyter',
    'notebook',
    'tornado',
    'zmq',
    'parso',
    'jedi',
    'sphinx',
    'alabaster',
    'babel',
    'docutils',
    'imagesize',
    'snowballstemmer',
    'sphinxcontrib',
    'sklearn',
    'pandas',
]

a = Analysis(
    [r'{os.path.join(base_dir, "ai_video_translator_cli.py")}'],
    pathex=[base_dir],
    binaries=binaries,
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath={hookspath},
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AI_Video_Translator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''

    with open(f"{APP_NAME}.spec", "w", encoding="utf-8") as f:
        f.write(spec_content)
    print("  创建: AI_Video_Translator.spec")

def build_exe():
    """构建EXE"""
    print("\n构建可执行文件...")
    print("这可能需要10-30分钟时间，请耐心等待...")
    print("正在收集依赖，请勿关闭窗口...")

    # 使用更激进的优化选项
    cmd = [
        sys.executable, "-m", "PyInstaller",
        f"{APP_NAME}.spec",
        "--clean",
        "--noconfirm",
        "--onefile",  # 单文件模式
    ]

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("❌ 构建失败")
        return False

    print("✅ 构建成功")
    return True

def create_portable_package():
    """创建便携版包"""
    print("\n创建便携版安装包...")

    os.makedirs(DIST_DIR, exist_ok=True)

    # 复制EXE
    exe_src = f"dist/{APP_NAME}.exe"
    exe_dst = f"{DIST_DIR}/{APP_NAME}.exe"

    if os.path.exists(exe_src):
        shutil.copy2(exe_src, exe_dst)
        print(f"  复制: {APP_NAME}.exe")

        # 获取文件大小
        size_mb = os.path.getsize(exe_dst) / (1024 * 1024)
        print(f"  大小: {size_mb:.1f} MB")
    else:
        print(f"❌ 未找到: {exe_src}")
        return False

    # 创建启动脚本
    launcher_bat = f'''{DIST_DIR}/启动_交互模式.bat
@echo off
chcp 65001 >nul
title AI Video Translator
cd /d "%~dp0"
AI_Video_Translator.exe
pause
'''
    with open(f"{DIST_DIR}/启动_交互模式.bat", "w", encoding="utf-8") as f:
        f.write(launcher_bat.replace(f'{DIST_DIR}/', ''))
    print("  创建: 启动_交互模式.bat")

    # 创建README
    readme = f'''{DIST_DIR}/README.txt
================================================================================
AI Video Translator v{APP_VERSION} - 便携版
================================================================================

📦 单文件可执行程序，无需安装Python和依赖

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
使用方法
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

方式1: 交互式菜单 (推荐)
  双击运行: 启动_交互模式.bat
  或命令行: AI_Video_Translator.exe

方式2: 直接命令
  AI_Video_Translator.exe <命令> [参数]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
命令列表
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  dub       - AI配音 (完整流程)
  separate  - 人声分离
  asr       - ASR语音识别
  translate - 翻译字幕
  tts       - TTS语音合成
  merge     - 合并音频
  replace   - 替换视频音轨
  silent    - 生成静音视频
  test      - 系统测试

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
使用示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. AI配音:
   AI_Video_Translator.exe dub video.mp4 --source-lang en --target-lang zh

2. 人声分离:
   AI_Video_Translator.exe separate video.mp4

3. ASR识别:
   AI_Video_Translator.exe asr video.mp4 --language en

4. 翻译字幕:
   AI_Video_Translator.exe translate subtitle.srt --source en --target zh

5. TTS合成:
   AI_Video_Translator.exe tts text.txt --language chinese

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
注意事项
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 首次运行需要下载AI模型 (约5-10GB)，请保持网络连接
2. 模型下载后会缓存，下次使用无需再下载
3. 需要NVIDIA GPU以获得最佳性能 (CPU模式较慢)
4. 输出文件保存在当前目录的 output/ 文件夹中

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
系统要求
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Windows 10/11 64位
- 8GB+ 内存 (推荐16GB)
- 10GB+ 磁盘空间
- NVIDIA GPU 推荐 (支持CUDA)

================================================================================
'''
    with open(f"{DIST_DIR}/README.txt", "w", encoding="utf-8") as f:
        f.write(readme.replace(f'{DIST_DIR}/', ''))
    print("  创建: README.txt")

    return True

def create_zip_package():
    """创建ZIP压缩包"""
    print("\n创建ZIP压缩包...")

    zip_name = f"{DIST_DIR}/{APP_NAME}_v{APP_VERSION}_便携版.zip"

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in os.listdir(DIST_DIR):
            if item.endswith('.zip'):
                continue
            item_path = os.path.join(DIST_DIR, item)
            zipf.write(item_path, item)
            print(f"  添加: {item}")

    size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"\n  压缩包大小: {size_mb:.1f} MB")
    print(f"  保存位置: {zip_name}")

    return zip_name

def main():
    print("=" * 60)
    print("AI Video Translator - 单文件EXE构建工具")
    print("=" * 60)
    print()

    # 检查平台
    if sys.platform != "win32":
        print("⚠️  警告: 此脚本用于构建 Windows EXE")
        print(f"   当前系统: {sys.platform}")
        response = input("\n是否继续? (y/n): ")
        if response.lower() != 'y':
            return

    # 检查依赖
    print("检查依赖...")
    try:
        import torch
        import transformers
        import whisper
        print("✅ 所有依赖已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请先安装依赖: pip install -r requirements.txt")
        return

    # 菜单
    print("\n构建选项:")
    print("  1. 构建单文件EXE (推荐)")
    print("  2. 清理构建文件")
    print("  3. 退出")

    choice = input("\n请选择 (1/2/3): ").strip()

    if choice == "1":
        # 完整构建流程
        clean()
        install_pyinstaller()
        create_spec_file()

        if build_exe():
            create_portable_package()
            create_zip_package()

            print("\n" + "=" * 60)
            print("✅ 构建完成!")
            print("=" * 60)
            print(f"\n输出目录: {DIST_DIR}/")
            print(f"\n文件列表:")
            for item in os.listdir(DIST_DIR):
                item_path = os.path.join(DIST_DIR, item)
                size = os.path.getsize(item_path) / (1024 * 1024)
                print(f"  - {item} ({size:.1f} MB)")

    elif choice == "2":
        clean()
        print("\n✅ 清理完成")

    elif choice == "3":
        print("\n退出")

    else:
        print("\n无效选择")

if __name__ == "__main__":
    main()
