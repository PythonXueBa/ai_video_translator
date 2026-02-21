#!/usr/bin/env python3
"""
AI Video Translator - 安装脚本
支持 Windows/Linux/macOS
"""

from setuptools import setup, find_packages
import os

# 读取 README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# 读取 requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ai-video-translator",
    version="1.0.0",
    author="PythonXueBa",
    author_email="",
    description="基于 Qwen3-TTS 的高质量多语言 AI 视频翻译与配音解决方案",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PythonXueBa/ai_video_translator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Sound/Audio",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "tts": ["qwen-tts"],
        "gpu": ["torch[cuda]", "torchaudio[cuda]"],
    },
    entry_points={
        "console_scripts": [
            "ai-video-translator=video_tool:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
