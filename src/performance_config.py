#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能配置模块 - 自动检测系统硬件并优化并行处理参数
支持动态调整 batch_size, num_workers, 并行策略等
优化显存管理和设备选择
"""

import os
import platform
import multiprocessing
from multiprocessing import cpu_count
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# 尝试导入psutil进行更精确的内存检测
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# 尝试导入torch检测GPU
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ComputeDevice(Enum):
    """计算设备类型"""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


class PerformanceLevel(Enum):
    """性能等级"""

    LOW = "low"  # < 8GB RAM, 无GPU或显存<4GB
    MEDIUM = "medium"  # 8-16GB RAM, 可能有GPU
    HIGH = "high"  # 16-32GB RAM, 有GPU且显存>=8GB
    ULTRA = "ultra"  # > 32GB RAM, 高端GPU


@dataclass
class SystemInfo:
    """系统信息"""

    cpu_count: int
    cpu_count_physical: int
    total_memory_gb: float
    available_memory_gb: float
    platform_system: str
    platform_machine: str
    has_gpu: bool
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    gpu_free_memory_gb: Optional[float] = None
    gpu_count: int = 0
    compute_device: ComputeDevice = ComputeDevice.CPU
    performance_level: PerformanceLevel = PerformanceLevel.MEDIUM


@dataclass
class ParallelConfig:
    """并行处理配置"""

    # 基础参数
    max_workers: int = 4
    batch_size: int = 8

    # ASR配置 - 使用large模型提升识别质量
    asr_model_size: str = "large"
    asr_batch_size: int = 1
    asr_parallel_segments: int = 2

    # 翻译配置 - 使用1.2B大模型提升翻译质量
    translator_model_size: str = "1.2B"
    translator_batch_size: int = 8
    translator_max_workers: int = 2

    # TTS配置 - 使用1.7B大模型提升合成质量
    tts_model_size: str = "1.7B"
    tts_max_workers: int = 1  # 强制串行避免显存冲突
    tts_batch_size: int = 1

    # 人声分离配置
    separator_model: str = "htdemucs"
    separator_device: str = "cpu"

    # 设备配置
    device: str = "cpu"
    dtype: str = "float32"
    use_fp16: bool = False

    # 内存优化
    enable_memory_efficient: bool = False
    max_memory_usage_ratio: float = 0.8

    # 显存安全阈值(GB)
    gpu_memory_safe_threshold: float = 2.0

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "asr_model_size": self.asr_model_size,
            "asr_batch_size": self.asr_batch_size,
            "asr_parallel_segments": self.asr_parallel_segments,
            "translator_model_size": self.translator_model_size,
            "translator_batch_size": self.translator_batch_size,
            "translator_max_workers": self.translator_max_workers,
            "tts_model_size": self.tts_model_size,
            "tts_max_workers": self.tts_max_workers,
            "tts_batch_size": self.tts_batch_size,
            "separator_model": self.separator_model,
            "separator_device": self.separator_device,
            "device": self.device,
            "dtype": self.dtype,
            "use_fp16": self.use_fp16,
            "enable_memory_efficient": self.enable_memory_efficient,
            "max_memory_usage_ratio": self.max_memory_usage_ratio,
            "gpu_memory_safe_threshold": self.gpu_memory_safe_threshold,
        }


class PerformanceConfig:
    """性能配置管理器"""

    def __init__(self):
        self._system_info: Optional[SystemInfo] = None
        self._config: Optional[ParallelConfig] = None

    @property
    def system_info(self) -> SystemInfo:
        """获取系统信息"""
        if self._system_info is None:
            self._system_info = self._detect_system()
        return self._system_info

    @property
    def config(self) -> ParallelConfig:
        """获取配置"""
        if self._config is None:
            self._config = self._auto_configure()
        return self._config

    def _detect_system(self) -> SystemInfo:
        """检测系统硬件"""
        # CPU信息
        cpu_cores = cpu_count()
        try:
            cpu_physical = psutil.cpu_count(logical=False) if HAS_PSUTIL else cpu_cores
        except:
            cpu_physical = cpu_cores

        # 内存信息
        if HAS_PSUTIL:
            mem = psutil.virtual_memory()
            total_mem_gb = mem.total / (1024**3)
            available_mem_gb = mem.available / (1024**3)
        else:
            # 估算
            total_mem_gb = 8.0
            available_mem_gb = total_mem_gb * 0.5

        # GPU信息
        has_gpu = False
        gpu_name = None
        gpu_memory_gb = None
        gpu_free_memory_gb = None
        gpu_count = 0
        compute_device = ComputeDevice.CPU

        if HAS_TORCH:
            # 检测CUDA
            if torch.cuda.is_available():
                has_gpu = True
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                try:
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (
                        1024**3
                    )
                    # 获取实际空闲显存
                    reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    gpu_free_memory_gb = max(0, gpu_memory_gb - reserved)
                except:
                    gpu_memory_gb = 8.0
                    gpu_free_memory_gb = gpu_memory_gb
                compute_device = ComputeDevice.CUDA

            # 检测Apple Silicon MPS
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                has_gpu = True
                gpu_name = "Apple Silicon"
                gpu_memory_gb = min(total_mem_gb * 0.5, 16.0)
                gpu_free_memory_gb = gpu_memory_gb
                compute_device = ComputeDevice.MPS

        # 确定性能等级 - 考虑实际空闲显存
        effective_gpu_memory = gpu_free_memory_gb or 0

        if has_gpu and effective_gpu_memory >= 8:
            if total_mem_gb >= 32:
                level = PerformanceLevel.ULTRA
            elif total_mem_gb >= 16:
                level = PerformanceLevel.HIGH
            else:
                level = PerformanceLevel.MEDIUM
        elif has_gpu and effective_gpu_memory >= 4:
            level = PerformanceLevel.MEDIUM
        elif total_mem_gb >= 16:
            level = PerformanceLevel.HIGH
        elif total_mem_gb >= 8:
            level = PerformanceLevel.MEDIUM
        else:
            level = PerformanceLevel.LOW

        return SystemInfo(
            cpu_count=cpu_cores,
            cpu_count_physical=cpu_physical,
            total_memory_gb=total_mem_gb,
            available_memory_gb=available_mem_gb,
            platform_system=platform.system(),
            platform_machine=platform.machine(),
            has_gpu=has_gpu,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            gpu_free_memory_gb=gpu_free_memory_gb,
            gpu_count=gpu_count,
            compute_device=compute_device,
            performance_level=level,
        )

    def _auto_configure(self) -> ParallelConfig:
        """根据系统信息自动配置"""
        info = self.system_info
        config = ParallelConfig()

        # 设备配置 - 检查实际可用显存
        if info.compute_device == ComputeDevice.CUDA:
            # 只有当空闲显存足够时才使用GPU
            if info.gpu_free_memory_gb and info.gpu_free_memory_gb >= 2.0:
                config.device = "cuda"
                config.use_fp16 = True
                config.dtype = "float16"
            else:
                print(
                    f"警告: GPU空闲显存不足 ({info.gpu_free_memory_gb:.1f}GB)，使用CPU"
                )
                config.device = "cpu"
                config.use_fp16 = False
                config.dtype = "float32"
        elif info.compute_device == ComputeDevice.MPS:
            config.device = "mps"
            config.use_fp16 = False
            config.dtype = "float32"
        else:
            config.device = "cpu"
            config.use_fp16 = False
            config.dtype = "float32"

        # 根据性能等级配置
        if info.performance_level == PerformanceLevel.ULTRA:
            config = self._configure_ultra(info, config)
        elif info.performance_level == PerformanceLevel.HIGH:
            config = self._configure_high(info, config)
        elif info.performance_level == PerformanceLevel.MEDIUM:
            config = self._configure_medium(info, config)
        else:
            config = self._configure_low(info, config)

        return config

    def _configure_ultra(
        self, info: SystemInfo, config: ParallelConfig
    ) -> ParallelConfig:
        """高端配置 (>32GB RAM, 高端GPU, 显存>=8GB)"""
        # 基础并行
        config.max_workers = min(info.cpu_count, 16)
        config.batch_size = 32

        # ASR - 使用大模型
        config.asr_model_size = "large" if info.has_gpu else "medium"
        config.asr_batch_size = 4
        config.asr_parallel_segments = min(info.cpu_count, 8)

        # 翻译 - 使用大模型
        config.translator_model_size = "1.2B"
        config.translator_batch_size = 32
        config.translator_max_workers = min(info.cpu_count // 2, 8)

        # TTS - 使用大模型，但GPU模式下强制串行
        config.tts_model_size = "1.7B"
        config.tts_max_workers = 1  # GPU模式强制串行
        config.tts_batch_size = 1

        # 分离
        config.separator_model = "htdemucs"
        config.separator_device = config.device

        # 显存安全阈值
        config.gpu_memory_safe_threshold = 4.0

        return config

    def _configure_high(
        self, info: SystemInfo, config: ParallelConfig
    ) -> ParallelConfig:
        """高配置 (16-32GB RAM, 有GPU, 显存>=8GB)"""
        config.max_workers = min(info.cpu_count, 8)
        config.batch_size = 16

        # ASR
        config.asr_model_size = "base"
        config.asr_batch_size = 2
        config.asr_parallel_segments = min(info.cpu_count, 4)

        # 翻译
        config.translator_model_size = "418M"
        config.translator_batch_size = 16
        config.translator_max_workers = min(info.cpu_count // 2, 4)

        # TTS - GPU模式强制串行
        config.tts_model_size = "1.7B"
        config.tts_max_workers = 1  # 强制串行
        config.tts_batch_size = 1

        config.separator_model = "htdemucs"
        config.separator_device = config.device

        # 显存安全阈值
        config.gpu_memory_safe_threshold = 3.0

        return config

    def _configure_medium(
        self, info: SystemInfo, config: ParallelConfig
    ) -> ParallelConfig:
        """中等配置 (8-16GB RAM, 可能有GPU但显存有限)"""
        # 基础并行
        config.max_workers = min(info.cpu_count, 6)
        config.batch_size = 8

        # ASR
        config.asr_model_size = "base"
        config.asr_batch_size = 1
        config.asr_parallel_segments = min(info.cpu_count // 2, 4)

        # 翻译
        config.translator_model_size = "418M"
        config.translator_batch_size = 8
        config.translator_max_workers = min(info.cpu_count // 2, 4)

        # TTS - 使用小模型
        config.tts_model_size = "0.6B"
        config.tts_max_workers = 1  # 强制串行
        config.tts_batch_size = 1

        # 分离
        config.separator_model = "htdemucs"
        config.separator_device = config.device

        # 内存优化
        config.enable_memory_efficient = True

        # 显存安全阈值
        config.gpu_memory_safe_threshold = 2.0

        return config

    def _configure_low(
        self, info: SystemInfo, config: ParallelConfig
    ) -> ParallelConfig:
        """低端配置 (<8GB RAM 或 显存<4GB)"""
        # 强制使用CPU
        config.device = "cpu"
        config.use_fp16 = False
        config.dtype = "float32"

        # 基础并行 - 保守设置
        config.max_workers = min(info.cpu_count // 2, 4)
        config.batch_size = 4

        # ASR - 使用小模型
        config.asr_model_size = "tiny"
        config.asr_batch_size = 1
        config.asr_parallel_segments = 2

        # 翻译 - 使用小模型
        config.translator_model_size = "418M"
        config.translator_batch_size = 4
        config.translator_max_workers = 2

        # TTS - 使用小模型
        config.tts_model_size = "0.6B"
        config.tts_max_workers = 1
        config.tts_batch_size = 1

        # 分离 - 使用CPU
        config.separator_model = "htdemucs"
        config.separator_device = "cpu"

        # 内存优化
        config.enable_memory_efficient = True
        config.max_memory_usage_ratio = 0.6

        # 显存安全阈值
        config.gpu_memory_safe_threshold = 0.0  # 不使用GPU

        return config

    def reconfigure_for_task(self, task: str) -> ParallelConfig:
        """
        为特定任务重新配置

        Args:
            task: 任务类型 (tts, asr, translation, separator)
        """
        config = self.config
        info = self.system_info

        # 根据任务调整配置
        if task == "tts":
            # TTS需要大量显存，检查是否需要降级
            if config.device == "cuda":
                free_mem = info.gpu_free_memory_gb or 0
                if free_mem < 4.0:
                    print("TTS任务: 显存不足，切换到CPU或小模型")
                    if free_mem < 2.0:
                        config.device = "cpu"
                    else:
                        config.tts_model_size = "0.6B"

        elif task == "separator":
            # 人声分离需要显存
            if config.device == "cuda":
                free_mem = info.gpu_free_memory_gb or 0
                if free_mem < 3.0:
                    print("人声分离: 显存不足，使用CPU")
                    config.separator_device = "cpu"

        return config

    def print_system_info(self):
        """打印系统信息"""
        info = self.system_info
        print("\n" + "=" * 60)
        print("系统性能检测")
        print("=" * 60)
        print(f"平台: {info.platform_system} ({info.platform_machine})")
        print(f"CPU: {info.cpu_count} 核 (物理: {info.cpu_count_physical})")
        print(
            f"内存: {info.total_memory_gb:.1f} GB (可用: {info.available_memory_gb:.1f} GB)"
        )

        if info.has_gpu:
            print(f"GPU: {info.gpu_name}")
            if info.gpu_memory_gb:
                print(
                    f"GPU显存: {info.gpu_memory_gb:.1f} GB (空闲: {info.gpu_free_memory_gb:.1f} GB)"
                )
        else:
            print("GPU: 无")

        print(f"计算设备: {info.compute_device.value}")
        print(f"性能等级: {info.performance_level.value}")
        print("=" * 60)

    def print_config(self):
        """打印配置"""
        config = self.config
        print("\n" + "=" * 60)
        print("自动优化配置")
        print("=" * 60)
        print(f"设备: {config.device} (dtype: {config.dtype}, fp16: {config.use_fp16})")
        print(f"最大并行工作进程: {config.max_workers}")
        print(f"默认批处理大小: {config.batch_size}")
        print("-" * 60)
        print(
            f"ASR模型: {config.asr_model_size}, 并行片段: {config.asr_parallel_segments}"
        )
        print(
            f"翻译模型: M2M100-{config.translator_model_size}, batch: {config.translator_batch_size}"
        )
        print(f"TTS模型: Qwen3-{config.tts_model_size}, 并行: {config.tts_max_workers}")
        print(f"人声分离: {config.separator_model}")
        print("-" * 60)
        print(f"内存优化模式: {'开启' if config.enable_memory_efficient else '关闭'}")
        print(f"显存安全阈值: {config.gpu_memory_safe_threshold} GB")
        print("=" * 60)

    def get_torch_dtype(self):
        """获取torch dtype"""
        if not HAS_TORCH:
            return None
        if self.config.dtype == "float16":
            return torch.float16
        elif self.config.dtype == "bfloat16":
            return torch.bfloat16
        else:
            return torch.float32

    def update_config(self, **kwargs):
        """更新配置参数"""
        config = self.config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# 全局单例
_performance_config: Optional[PerformanceConfig] = None


def get_performance_config() -> PerformanceConfig:
    """获取全局性能配置实例"""
    global _performance_config
    if _performance_config is None:
        _performance_config = PerformanceConfig()
    return _performance_config


def get_parallel_config() -> ParallelConfig:
    """获取并行配置"""
    return get_performance_config().config


def print_system_info():
    """打印系统信息"""
    get_performance_config().print_system_info()


def print_config():
    """打印配置"""
    get_performance_config().print_config()


# 便捷函数
def get_optimal_workers() -> int:
    """获取最优工作进程数"""
    return get_parallel_config().max_workers


def get_optimal_batch_size() -> int:
    """获取最优批处理大小"""
    return get_parallel_config().batch_size


def get_device() -> str:
    """获取计算设备"""
    return get_parallel_config().device


def is_gpu_available() -> bool:
    """检查GPU是否可用"""
    return get_performance_config().system_info.has_gpu


if __name__ == "__main__":
    # 测试
    config = get_performance_config()
    config.print_system_info()
    config.print_config()
    print("\n配置字典:")
    import json

    print(json.dumps(config.config.to_dict(), indent=2))
