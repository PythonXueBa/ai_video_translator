#!/usr/bin/env python3
"""
AI Video Translator - 精简版单文件EXE构建脚本
构建最小体积的独立可执行文件
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# 配置
APP_NAME = "AI_Video_Translator"
APP_VERSION = "1.0.0"

def clean():
    """清理构建目录"""
    print("清理构建目录...")
    for d in ["build", "dist", "__pycache__"]:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"  删除: {d}")

def get_site_packages():
    """获取site-packages路径"""
    import site
    return site.getsitepackages()[0]

def create_entry_point():
    """创建入口点文件"""
    print("创建入口点...")

    entry_code = '''#!/usr/bin/env python3
"""
AI Video Translator - 统一入口
"""

import sys
import os

# 设置环境变量
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ['PYTHONPATH'] = BASE_DIR
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# 添加路径
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

# 导入主程序
from video_tool import main

if __name__ == "__main__":
    main()
'''

    with open("entry_point.py", "w", encoding="utf-8") as f:
        f.write(entry_code)
    print("  创建: entry_point.py")

def build_minimal():
    """构建精简版EXE"""
    print("\n构建精简版EXE...")
    print("这可能需要10-20分钟...")

    # 基础命令
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--onefile",
        "--console",
        "--clean",
        "--noconfirm",
    ]

    # 添加数据文件
    cmd.extend(["--add-data", "src;src"])
    cmd.extend(["--add-data", "video_tool.py;."])

    # 关键隐藏导入
    hidden_imports = [
        # 核心
        "torch", "torchaudio", "torchvision",
        "transformers", "transformers.models",
        # ASR
        "whisper", "whisper.model", "whisper.tokenizer",
        # 音频
        "demucs", "soundfile", "pydub",
        # 计算
        "numpy", "scipy", "scipy.signal",
        # 其他
        "psutil", "tqdm", "sentencepiece", "sacremoses",
        "einops", "tokenizers", "regex",
        "yaml", "requests",
        # 项目模块
        "src.config", "src.analyzer", "src.extractor",
        "src.separator", "src.asr_module",
        "src.translator_m2m100", "src.tts_qwen3",
        "src.subtitle_handler", "src.subtitle_tts_engine",
        "src.merger", "src.video_processor",
        "src.performance_config", "src.memory_manager", "src.splitter",
    ]

    for imp in hidden_imports:
        cmd.extend(["--hidden-import", imp])

    # 排除不必要的包
    excludes = [
        "matplotlib", "PIL", "Pillow",
        "tkinter", "tkinter.test",
        "unittest", "pydoc", "pdb", "doctest",
        "test", "tests", "_pytest", "pytest",
        "IPython", "jupyter", "notebook",
        "tornado", "zmq", "sphinx",
        "sklearn", "pandas", "seaborn",
    ]

    for exc in excludes:
        cmd.extend(["--exclude-module", exc])

    # 入口文件
    cmd.append("entry_point.py")

    # 执行构建
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✅ 构建成功!")
        exe_path = f"dist/{APP_NAME}.exe"
        if os.path.exists(exe_path):
            size_mb = os.path.getsize(exe_path) / (1024 * 1024)
            print(f"  文件: {exe_path}")
            print(f"  大小: {size_mb:.1f} MB")
        return True
    else:
        print("\n❌ 构建失败")
        return False

def create_launcher():
    """创建启动器"""
    print("\n创建启动器...")

    # Windows启动器
    bat_content = '''@echo off
chcp 65001 >nul
title AI Video Translator
cd /d "%~dp0"
AI_Video_Translator.exe %*
'''

    with open("dist/启动.bat", "w", encoding="utf-8") as f:
        f.write(bat_content)
    print("  创建: 启动.bat")

def main():
    print("=" * 60)
    print("AI Video Translator - 精简版EXE构建工具")
    print("=" * 60)
    print()

    # 检查依赖
    print("检查依赖...")
    try:
        import torch
        import transformers
        print("✅ 依赖检查通过")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        return

    # 清理
    clean()

    # 创建入口点
    create_entry_point()

    # 构建
    if build_minimal():
        create_launcher()

        print("\n" + "=" * 60)
        print("✅ 构建完成!")
        print("=" * 60)
        print("\n输出文件:")
        print("  dist/AI_Video_Translator.exe - 主程序")
        print("  dist/启动.bat - Windows启动器")
        print("\n使用方法:")
        print("  AI_Video_Translator.exe dub video.mp4")
        print("  AI_Video_Translator.exe --help")

if __name__ == "__main__":
    main()
