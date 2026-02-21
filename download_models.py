#!/usr/bin/env python3
"""
AI Video Translator - 模型下载脚本
下载所有需要的AI模型到本地
"""

import os
import sys
import urllib.request
import json
from pathlib import Path

# 模型配置
MODELS = {
    "whisper": {
        "name": "Whisper (ASR)",
        "url": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34/small.pt",
        "filename": "small.pt",
        "size": "466 MB"
    },
    "m2m100_418m": {
        "name": "M2M100-418M (翻译)",
        "url": "https://huggingface.co/facebook/m2m100_418M/resolve/main/pytorch_model.bin",
        "filename": "m2m100_418m.bin",
        "size": "1.6 GB"
    },
    "m2m100_1.2b": {
        "name": "M2M100-1.2B (翻译)",
        "url": "https://huggingface.co/facebook/m2m100_1.2B/resolve/main/pytorch_model.bin",
        "filename": "m2m100_1.2b.bin",
        "size": "4.8 GB"
    },
}

# Hugging Face 镜像
HF_MIRROR = "https://hf-mirror.com"

def get_model_dir():
    """获取模型存储目录"""
    if sys.platform == "win32":
        base_dir = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    else:
        base_dir = os.path.expanduser("~/.cache")

    model_dir = os.path.join(base_dir, "ai_video_translator", "models")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def download_file(url, dest_path, description=""):
    """下载文件并显示进度"""
    print(f"下载 {description}...")
    print(f"  来源: {url}")
    print(f"  目标: {dest_path}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        size_mb = total_size / (1024 * 1024)
        downloaded_mb = downloaded / (1024 * 1024)
        print(f"\r  进度: {percent:.1f}% ({downloaded_mb:.1f}/{size_mb:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
        print("\n  下载完成!")
        return True
    except Exception as e:
        print(f"\n  下载失败: {e}")
        return False

def download_whisper_model(model_name="small"):
    """下载 Whisper 模型"""
    model_dir = get_model_dir()

    # Whisper 模型使用 transformers 自动下载
    print(f"Whisper 模型将通过 transformers 自动下载")
    print(f"模型将保存在: {model_dir}")
    return True

def download_m2m100_model(use_1_2b=True):
    """下载 M2M100 翻译模型"""
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    model_name = "facebook/m2m100_1.2B" if use_1_2b else "facebook/m2m100_418M"

    print(f"下载 M2M100 翻译模型: {model_name}")
    print("这可能需要几分钟时间，请耐心等待...")

    try:
        # 设置缓存目录
        model_dir = get_model_dir()
        os.environ["TRANSFORMERS_CACHE"] = model_dir
        os.environ["HF_HOME"] = model_dir

        # 下载模型
        print("正在下载模型文件...")
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)

        print(f"模型下载完成！保存在: {model_dir}")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        print("尝试使用镜像下载...")
        try:
            # 使用镜像
            os.environ["HF_ENDPOINT"] = HF_MIRROR
            model = M2M100ForConditionalGeneration.from_pretrained(model_name)
            tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            print("模型下载完成！")
            return True
        except Exception as e2:
            print(f"镜像下载也失败: {e2}")
            return False

def download_demucs_model():
    """下载 Demucs 人声分离模型"""
    print("Demucs 模型将通过 torch hub 自动下载")
    return True

def download_qwen3_tts_model():
    """下载 Qwen3-TTS 模型"""
    print("Qwen3-TTS 模型将通过 qwen-tts 自动下载")
    print("首次使用时需要联网下载")
    return True

def download_ffmpeg_windows():
    """下载 Windows 版 FFmpeg"""
    import zipfile

    model_dir = get_model_dir()
    ffmpeg_dir = os.path.join(model_dir, "ffmpeg")

    if os.path.exists(os.path.join(ffmpeg_dir, "ffmpeg.exe")):
        print("FFmpeg 已存在")
        return ffmpeg_dir

    print("下载 FFmpeg for Windows...")

    # FFmpeg 下载链接
    ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    zip_path = os.path.join(model_dir, "ffmpeg.zip")

    if download_file(ffmpeg_url, zip_path, "FFmpeg"):
        print("解压 FFmpeg...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)

        # 找到解压后的目录
        for item in os.listdir(model_dir):
            if item.startswith("ffmpeg-") and os.path.isdir(os.path.join(model_dir, item)):
                extracted_dir = os.path.join(model_dir, item)
                # 重命名为 ffmpeg
                if os.path.exists(ffmpeg_dir):
                    import shutil
                    shutil.rmtree(ffmpeg_dir)
                os.rename(extracted_dir, ffmpeg_dir)
                break

        os.remove(zip_path)
        print(f"FFmpeg 安装完成: {ffmpeg_dir}")
        return ffmpeg_dir
    else:
        print("FFmpeg 下载失败")
        return None

def setup_environment():
    """设置环境变量"""
    model_dir = get_model_dir()
    ffmpeg_dir = os.path.join(model_dir, "ffmpeg", "bin")

    # 添加 FFmpeg 到 PATH
    if os.path.exists(ffmpeg_dir):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        print(f"已添加 FFmpeg 到 PATH: {ffmpeg_dir}")

    # 设置模型缓存目录
    os.environ["TRANSFORMERS_CACHE"] = model_dir
    os.environ["HF_HOME"] = model_dir
    os.environ["TORCH_HOME"] = model_dir

def main():
    print("=" * 60)
    print("AI Video Translator - 模型下载工具")
    print("=" * 60)
    print()

    model_dir = get_model_dir()
    print(f"模型将下载到: {model_dir}")
    print()

    # 检查依赖
    try:
        import torch
        print(f"✓ PyTorch 已安装: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch 未安装，请先安装: pip install torch")
        return

    try:
        import transformers
        print(f"✓ Transformers 已安装: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers 未安装，请先安装: pip install transformers")
        return

    print()
    print("开始下载模型...")
    print("-" * 60)

    # 下载各模型
    results = []

    # 1. Whisper
    print("\n[1/4] Whisper ASR 模型")
    results.append(("Whisper", download_whisper_model()))

    # 2. M2M100
    print("\n[2/4] M2M100 翻译模型")
    results.append(("M2M100", download_m2m100_model(use_1_2b=True)))

    # 3. Demucs
    print("\n[3/4] Demucs 人声分离模型")
    results.append(("Demucs", download_demucs_model()))

    # 4. Qwen3-TTS
    print("\n[4/4] Qwen3-TTS 语音合成模型")
    results.append(("Qwen3-TTS", download_qwen3_tts_model()))

    # Windows 下载 FFmpeg
    if sys.platform == "win32":
        print("\n[额外] FFmpeg for Windows")
        ffmpeg_dir = download_ffmpeg_windows()
        if ffmpeg_dir:
            results.append(("FFmpeg", True))
        else:
            results.append(("FFmpeg", False))

    # 汇总
    print()
    print("=" * 60)
    print("下载结果汇总:")
    print("=" * 60)
    for name, success in results:
        status = "✓ 成功" if success else "✗ 失败"
        print(f"  {name}: {status}")

    print()
    print("注意: 部分模型会在首次使用时自动下载")
    print("模型保存位置:", model_dir)
    print()
    input("按回车键退出...")

if __name__ == "__main__":
    main()
