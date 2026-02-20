#!/usr/bin/env python3
"""
M2M100 翻译模块
模型: facebook/m2m100_1.2B
支持100种语言直接互译
支持并行处理提升效率
优化显存管理
"""

from pathlib import Path
from typing import List, Optional
import gc
import torch

# 尝试导入性能配置
try:
    from .performance_config import get_parallel_config, get_performance_config
    from .memory_manager import get_free_gpu_memory, clear_gpu_cache
    HAS_PERF_CONFIG = True
except ImportError:
    HAS_PERF_CONFIG = False


class M2M100Translator:
    """M2M100多语言翻译器"""

    # 语言代码映射
    LANGUAGE_CODES = {
        "en": "en",
        "zh": "zh",
        "zh-cn": "zh",
        "ja": "ja",
        "ko": "ko",
        "es": "es",
        "fr": "fr",
        "de": "de",
        "ru": "ru",
        "ar": "ar",
    }

    def __init__(
        self,
        source_language: str = "en",
        target_language: str = "zh",
        device: Optional[str] = None,
        model_size: str = "1.2B",  # 418M or 1.2B
    ):
        self.source_lang = self._normalize_lang(source_language)
        self.target_lang = self._normalize_lang(target_language)
        self.model_size = model_size
        self.model_name = f"facebook/m2m100_{model_size}"

        # 自动检测设备，检查显存
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # 检查显存是否足够
        if device == "cuda" and HAS_PERF_CONFIG:
            free_mem = get_free_gpu_memory()
            required = 3.0 if model_size == "1.2B" else 1.5
            if free_mem < required:
                print(f"显存不足 ({free_mem:.1f}GB < {required}GB)，翻译使用CPU")
                device = "cpu"

        self.device = device
        self.model = None
        self.tokenizer = None

    def _normalize_lang(self, lang: str) -> str:
        """标准化语言代码"""
        lang = lang.lower().replace("-", "").replace("_", "")
        return self.LANGUAGE_CODES.get(lang, lang)

    def load_model(self):
        """加载M2M100模型"""
        if self.model is not None:
            return

        try:
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        except ImportError:
            print("错误: 未安装transformers")
            print("安装: pip install transformers")
            raise

        print(f"加载M2M100模型: {self.model_name}")
        print(f"设备: {self.device}")

        if self.model_size == "1.2B":
            print("模型大小约4.5GB，首次下载可能需要10-20分钟...")
        else:
            print("模型大小约1.6GB...")

        try:
            # 清理显存
            if self.device == "cuda":
                if HAS_PERF_CONFIG:
                    clear_gpu_cache()
                else:
                    gc.collect()
                    torch.cuda.empty_cache()

            self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)

            # 根据设备选择加载方式
            if self.device == "cuda":
                # 尝试使用float16减少显存
                try:
                    self.model = M2M100ForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                    ).to(self.device)
                except Exception:
                    # 回退到float32
                    self.model = M2M100ForConditionalGeneration.from_pretrained(
                        self.model_name
                    ).to(self.device)
            else:
                self.model = M2M100ForConditionalGeneration.from_pretrained(
                    self.model_name
                ).to(self.device)

            self.model.eval()
            print("✓ M2M100模型加载成功")
            print("✓ 支持100种语言直接互译")

        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            raise

    def unload_model(self):
        """卸载模型释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✓ 翻译模型已卸载")

    def translate(self, text: str) -> str:
        """翻译文本"""
        if not text or not text.strip():
            return ""

        self.load_model()

        try:
            # 设置源语言
            self.tokenizer.src_lang = self.source_lang

            # 编码
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            # 生成 - 指定目标语言
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.get_lang_id(self.target_lang),
                )

            # 解码
            result = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )[0]
            return result.strip()

        except Exception as e:
            print(f"翻译失败: {e}")
            return text

    def translate_segments(
        self, segments: List, batch_size: Optional[int] = None, use_parallel: bool = True
    ) -> List:
        """
        翻译多个片段

        Args:
            segments: 片段列表
            batch_size: 批处理大小（None则自动配置）
            use_parallel: 是否使用并行处理

        Returns:
            翻译后的片段列表
        """
        print(f"\n使用M2M100翻译 {len(segments)} 个片段...")
        print(f"{self.source_lang} -> {self.target_lang}")

        self.load_model()

        # 自动配置batch_size
        if batch_size is None and HAS_PERF_CONFIG:
            batch_size = get_parallel_config().translator_batch_size
        elif batch_size is None:
            batch_size = 8

        # 根据显存调整batch_size
        if self.device == "cuda" and HAS_PERF_CONFIG:
            free_mem = get_free_gpu_memory()
            if free_mem < 2.0:
                batch_size = min(batch_size, 4)
            elif free_mem < 4.0:
                batch_size = min(batch_size, 8)

        total = len(segments)

        # 收集有效文本
        valid_texts = []
        valid_indices = []
        for i, segment in enumerate(segments):
            if segment.text and segment.text.strip():
                valid_texts.append(segment.text)
                valid_indices.append(i)
            else:
                segment.translated_text = ""

        if not valid_texts:
            print("  无有效文本需要翻译")
            return segments

        print(f"  有效文本: {len(valid_texts)}/{total}")
        print(f"  批处理大小: {batch_size}")

        # 批量翻译
        translated_texts = self.batch_translate(valid_texts, batch_size)

        # 分配结果
        for idx, translated in zip(valid_indices, translated_texts):
            segments[idx].translated_text = translated

        # 显示示例
        example_count = min(3, len(valid_indices))
        for i in range(example_count):
            idx = valid_indices[i]
            print(f"    原文: {segments[idx].text[:50]}...")
            print(f"    译文: {segments[idx].translated_text[:50]}...")

        translated_count = sum(1 for seg in segments if hasattr(seg, 'translated_text') and seg.translated_text)
        print(f"\n✓ 翻译完成: {translated_count}/{total}")
        return segments

    def batch_translate(self, texts: List[str], batch_size: Optional[int] = None) -> List[str]:
        """
        批量翻译

        Args:
            texts: 文本列表
            batch_size: 批处理大小（None则自动配置）
        """
        self.load_model()

        # 自动配置batch_size
        if batch_size is None and HAS_PERF_CONFIG:
            batch_size = get_parallel_config().translator_batch_size
        elif batch_size is None:
            batch_size = 8

        results = []
        total = len(texts)

        # 设置源语言
        self.tokenizer.src_lang = self.source_lang

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            valid_texts = []
            valid_indices = []

            for j, text in enumerate(batch):
                if text and text.strip():
                    valid_texts.append(text)
                    valid_indices.append(j)

            if not valid_texts:
                results.extend([""] * len(batch))
                continue

            try:
                inputs = self.tokenizer(
                    valid_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)

                with torch.no_grad():
                    generated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=self.tokenizer.get_lang_id(
                            self.target_lang
                        ),
                    )

                decoded = self.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )

                batch_results = [""] * len(batch)
                for idx, result in zip(valid_indices, decoded):
                    batch_results[idx] = result.strip()

                results.extend(batch_results)

                # 定期清理显存
                if self.device == "cuda" and (i // batch_size) % 10 == 0:
                    torch.cuda.empty_cache()

                current = min(i + batch_size, total)
                if current % 20 == 0 or current == total:
                    print(f"  进度: {current}/{total} ({current * 100 // total}%)")

            except Exception as e:
                print(f"批量翻译失败 [{i}-{i + batch_size}]: {e}")
                results.extend([text for text in batch])

        return results


def get_translator(
    source_lang: str = "en",
    target_lang: str = "zh",
    device: Optional[str] = None,
    use_m2m100: bool = True,
    model_size: Optional[str] = None,
):
    """
    获取翻译器实例

    Args:
        source_lang: 源语言
        target_lang: 目标语言
        device: 计算设备
        use_m2m100: 是否使用M2M100
        model_size: 模型大小（None则自动配置）
    """
    # 自动配置模型大小
    if model_size is None and HAS_PERF_CONFIG:
        model_size = get_parallel_config().translator_model_size
    elif model_size is None:
        model_size = "418M"

    # 自动配置设备
    if device is None and HAS_PERF_CONFIG:
        device = get_parallel_config().device
    elif device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_m2m100:
        return M2M100Translator(
            source_language=source_lang,
            target_language=target_lang,
            device=device,
            model_size=model_size,
        )
    else:
        # 回退到OPUS-MT
        from translator_opus import OpusTranslator

        return OpusTranslator(
            source_language=source_lang,
            target_language=target_lang,
            device=device,
        )
