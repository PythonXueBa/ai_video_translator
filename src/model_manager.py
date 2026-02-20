#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理模块 - 下载和管理所有离线模型
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import subprocess


class ModelManager:
    """模型管理器"""

    # 模型定义
    MODELS = {
        "demucs": {
            "name": "Demucs人声分离",
            "description": "人声分离模型，用于提取人声和背景音",
            "models": {
                "htdemucs": {
                    "description": "默认模型，平衡速度和质量",
                    "auto_download": True,  # 自动下载
                },
                "htdemucs_ft": {
                    "description": "微调版本，质量更好但更慢",
                    "auto_download": False,
                },
            },
        },
        "whisper": {
            "name": "Whisper语音识别",
            "description": "OpenAI Whisper ASR模型",
            "models": {
                "tiny": {
                    "size": "39 MB",
                    "description": "最快，适合测试",
                    "url": "openai/whisper-tiny",
                },
                "base": {
                    "size": "74 MB",
                    "description": "推荐，平衡速度和质量",
                    "url": "openai/whisper-base",
                },
                "small": {
                    "size": "244 MB",
                    "description": "更好质量，较慢",
                    "url": "openai/whisper-small",
                },
                "medium": {
                    "size": "769 MB",
                    "description": "高质量，慢",
                    "url": "openai/whisper-medium",
                },
                "large": {
                    "size": "1550 MB",
                    "description": "最佳质量，最慢",
                    "url": "openai/whisper-large-v3",
                },
            },
        },
        "translation": {
            "name": "翻译模型",
            "description": "M2M100多语言翻译(推荐)或MarianMT",
            "models": {
                "m2m100_1.2B": {
                    "name": "M2M100 1.2B",
                    "description": "Facebook M2M100 - 支持100种语言直接互译",
                    "repo": "facebook/m2m100_1.2B",
                    "size": "约4.5GB",
                    "languages": "100种语言",
                    "recommended": True,
                },
                "m2m100_418M": {
                    "name": "M2M100 418M",
                    "description": "Facebook M2M100 小模型",
                    "repo": "facebook/m2m100_418M",
                    "size": "约1.6GB",
                    "languages": "100种语言",
                },
                "zh-en": {
                    "name": "中文->英文",
                    "repo": "Helsinki-NLP/opus-mt-zh-en",
                    "size": "约300MB",
                },
                "en-zh": {
                    "name": "英文->中文",
                    "repo": "Helsinki-NLP/opus-mt-en-zh",
                    "size": "约300MB",
                },
                "en-ja": {
                    "name": "英文->日文",
                    "repo": "Helsinki-NLP/opus-mt-en-jap",
                    "size": "约300MB",
                },
                "ja-en": {
                    "name": "日文->英文",
                    "repo": "Helsinki-NLP/opus-mt-ja-en",
                    "size": "约300MB",
                },
            },
        },
        "tts": {
            "name": "TTS语音合成",
            "description": "Coqui TTS音色克隆模型",
            "models": {
                "xtts_v2": {
                    "name": "XTTS v2",
                    "description": "多语言音色克隆（推荐）",
                    "repo": "tts_models/multilingual/multi-dataset/xtts_v2",
                    "size": "约2-3GB",
                    "languages": [
                        "zh",
                        "en",
                        "ja",
                        "de",
                        "fr",
                        "es",
                        "it",
                        "pt",
                        "pl",
                        "tr",
                        "ru",
                        "nl",
                        "cs",
                        "ar",
                        "hu",
                        "ko",
                    ],
                },
                "tacotron2": {
                    "name": "Tacotron2",
                    "description": "英文TTS",
                    "size": "约100MB",
                },
            },
        },
    }

    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is None:
            self.models_dir = Path(__file__).parent.parent / "models"
        else:
            self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

    def list_models(self):
        """列出所有可用模型"""
        print("\n" + "=" * 60)
        print("可用模型列表")
        print("=" * 60)

        for category, info in self.MODELS.items():
            print(f"\n【{info['name']}】")
            print(f"  {info['description']}")
            print(f"  存储位置: {self.models_dir / category}")

            for model_name, model_info in info["models"].items():
                if isinstance(model_info, dict):
                    size = model_info.get("size", "")
                    desc = model_info.get("description", model_info.get("name", ""))
                    print(f"    - {model_name}: {desc} {size}")

    def download_whisper(self, model_size: str = "base"):
        """下载Whisper模型"""
        print(f"\n下载Whisper模型: {model_size}")
        print("=" * 60)

        try:
            import whisper

            print(f"正在下载 {model_size} 模型...")
            model = whisper.load_model(model_size)
            print(f"✓ Whisper {model_size} 模型已下载")
            return True
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            return False

    def download_m2m100(self, model_size: str = "1.2B"):
        """下载M2M100多语言翻译模型"""
        print(f"\n下载M2M100翻译模型: {model_size}")
        print("=" * 60)
        print("模型: facebook/m2m100_" + model_size)
        print("特点: 支持100种语言直接互译，无需中转英文")

        if model_size not in ["1.2B", "418M"]:
            print(f"错误: 不支持的模型大小 {model_size}")
            return False

        try:
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

            model_name = f"facebook/m2m100_{model_size}"
            print(f"\n正在下载 {model_name}...")
            print("模型大小: ~4.5GB (1.2B) 或 ~1.6GB (418M)")
            print("这可能需要20-40分钟，请保持网络连接稳定...")

            tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            model = M2M100ForConditionalGeneration.from_pretrained(model_name)

            print(f"✓ M2M100模型下载成功")
            print(f"✓ 支持100种语言直接互译")
            return True

        except Exception as e:
            print(f"✗ 下载失败: {e}")
            print("请检查网络连接或访问 https://huggingface.co/facebook/m2m100_1.2B")
            return False

    def download_translation(self, lang_pair: str = "zh-en"):
        """下载翻译模型"""
        print(f"\n下载翻译模型: {lang_pair}")
        print("=" * 60)

        if lang_pair not in self.MODELS["translation"]["models"]:
            print(f"错误: 不支持的语对 {lang_pair}")
            return False

        model_info = self.MODELS["translation"]["models"][lang_pair]
        repo = model_info["repo"]

        try:
            from transformers import MarianMTModel, MarianTokenizer

            print(f"正在下载 {repo}...")
            print("这可能需要5-10分钟...")

            tokenizer = MarianTokenizer.from_pretrained(repo)
            model = MarianMTModel.from_pretrained(repo)

            # 保存到本地
            save_path = self.models_dir / "translation" / lang_pair
            save_path.mkdir(parents=True, exist_ok=True)

            tokenizer.save_pretrained(save_path)
            model.save_pretrained(save_path)

            print(f"✓ 翻译模型已下载到: {save_path}")
            return True

        except Exception as e:
            print(f"✗ 下载失败: {e}")
            print("请检查网络连接或手动下载")
            return False

    def download_tts(self, model_name: str = "xtts_v2"):
        """下载TTS模型"""
        print(f"\n下载TTS模型: {model_name}")
        print("=" * 60)

        if model_name == "xtts_v2":
            try:
                from TTS.api import TTS

                print("正在下载 XTTS v2 模型...")
                print("模型大小约2-3GB，可能需要10-30分钟...")
                print("请保持网络连接稳定")

                model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

                print(f"✓ TTS模型已下载")
                return True

            except Exception as e:
                print(f"✗ 下载失败: {e}")
                print("请检查网络连接")
                return False
        else:
            print(f"不支持的TTS模型: {model_name}")
            return False

    def download_all(self):
        """下载所有推荐模型"""
        print("\n" + "=" * 60)
        print("下载所有推荐模型")
        print("=" * 60)
        print("注意: 这将下载约3-5GB的模型文件")
        print("请确保有足够的磁盘空间和稳定的网络连接")
        print("=" * 60)

        input("按 Enter 开始下载，或按 Ctrl+C 取消...")

        results = {}

        # Whisper
        results["whisper"] = self.download_whisper("base")

        # 翻译模型
        results["translation_zh-en"] = self.download_translation("zh-en")
        results["translation_en-zh"] = self.download_translation("en-zh")

        # TTS
        results["tts"] = self.download_tts("xtts_v2")

        # 总结
        print("\n" + "=" * 60)
        print("下载完成")
        print("=" * 60)
        for name, success in results.items():
            status = "✓" if success else "✗"
            print(f"{status} {name}")

    def check_models(self):
        """检查已安装的模型"""
        print("\n" + "=" * 60)
        print("检查模型状态")
        print("=" * 60)

        # 检查Whisper
        try:
            import whisper

            print("✓ Whisper: 已安装")
        except ImportError:
            print("✗ Whisper: 未安装 (pip install openai-whisper)")

        # 检查transformers
        try:
            import transformers

            print(f"✓ Transformers: 已安装 (v{transformers.__version__})")
        except ImportError:
            print("✗ Transformers: 未安装 (pip install transformers)")

        # 检查TTS
        try:
            import TTS

            print("✓ Coqui TTS: 已安装")
        except ImportError:
            print("✗ Coqui TTS: 未安装 (pip install TTS)")

        # 检查模型文件
        print("\n模型缓存目录:")
        cache_dirs = [
            Path.home() / ".cache" / "whisper",
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".local" / "share" / "tts",
        ]

        for cache_dir in cache_dirs:
            if cache_dir.exists():
                size = self._get_dir_size(cache_dir)
                print(f"  {cache_dir}: {size:.1f} MB")

    def _get_dir_size(self, path: Path) -> float:
        """获取目录大小(MB)"""
        total = 0
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
        return total / (1024 * 1024)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AI配音模型管理工具")
    parser.add_argument(
        "command",
        choices=[
            "list",
            "check",
            "download-whisper",
            "download-translation",
            "download-tts",
            "download-all",
        ],
        help="命令",
    )
    parser.add_argument("--model", help="模型名称")
    parser.add_argument("--lang-pair", default="zh-en", help="翻译语对")

    args = parser.parse_args()

    manager = ModelManager()

    if args.command == "list":
        manager.list_models()
    elif args.command == "check":
        manager.check_models()
    elif args.command == "download-whisper":
        manager.download_whisper(args.model or "base")
    elif args.command == "download-translation":
        manager.download_translation(args.lang_pair)
    elif args.command == "download-tts":
        manager.download_tts(args.model or "xtts_v2")
    elif args.command == "download-all":
        manager.download_all()


if __name__ == "__main__":
    main()
