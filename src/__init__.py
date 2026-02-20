#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI配音系统 - 源代码包
"""

from .config import *
from .analyzer import MediaAnalyzer
from .extractor import AudioExtractor
from .separator import VocalSeparator
from .asr_module import WhisperASR, ASRResult
from .translator_m2m100 import M2M100Translator, get_translator
from .tts_qwen3 import Qwen3TTS, get_qwen3_tts
from .merger import AudioMerger
from .video_processor import VideoProcessor
from .subtitle_handler import SRTHandler, SubtitleEntry
from .performance_config import (
    PerformanceConfig,
    ParallelConfig,
    get_performance_config,
    get_parallel_config,
)
from .memory_manager import (
    GPUMemoryManager,
    get_memory_manager,
    clear_gpu_cache,
    get_free_gpu_memory,
    get_recommended_device,
)

__all__ = [
    "MediaAnalyzer",
    "AudioExtractor",
    "VocalSeparator",
    "WhisperASR",
    "ASRResult",
    "M2M100Translator",
    "get_translator",
    "Qwen3TTS",
    "get_qwen3_tts",
    "AudioMerger",
    "VideoProcessor",
    "SRTHandler",
    "SubtitleEntry",
    "PerformanceConfig",
    "ParallelConfig",
    "get_performance_config",
    "get_parallel_config",
    "GPUMemoryManager",
    "get_memory_manager",
    "clear_gpu_cache",
    "get_free_gpu_memory",
    "get_recommended_device",
]
