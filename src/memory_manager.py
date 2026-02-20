#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU显存管理模块
提供显存监控、自动清理、设备切换等功能
"""

import gc
import os
import time
import threading
from typing import Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass

# 尝试导入torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# 尝试导入psutil
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class MemoryInfo:
    """内存信息"""
    gpu_total: float = 0.0
    gpu_used: float = 0.0
    gpu_free: float = 0.0
    ram_total: float = 0.0
    ram_available: float = 0.0
    ram_used_percent: float = 0.0


class GPUMemoryManager:
    """GPU显存管理器"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._last_check_time = 0
        self._check_interval = 0.5  # 最小检查间隔

    @property
    def is_cuda_available(self) -> bool:
        """检查CUDA是否可用"""
        return HAS_TORCH and torch.cuda.is_available()

    def get_memory_info(self) -> MemoryInfo:
        """获取内存信息"""
        info = MemoryInfo()

        # GPU信息
        if self.is_cuda_available:
            try:
                props = torch.cuda.get_device_properties(0)
                info.gpu_total = props.total_memory / (1024**3)

                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                info.gpu_used = allocated
                info.gpu_free = info.gpu_total - reserved

            except Exception:
                pass

        # RAM信息
        if HAS_PSUTIL:
            try:
                mem = psutil.virtual_memory()
                info.ram_total = mem.total / (1024**3)
                info.ram_available = mem.available / (1024**3)
                info.ram_used_percent = mem.percent
            except Exception:
                pass

        return info

    def get_gpu_memory_gb(self) -> float:
        """获取GPU总显存(GB)"""
        if not self.is_cuda_available:
            return 0.0
        try:
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            return 0.0

    def get_free_gpu_memory_gb(self) -> float:
        """获取GPU空闲显存(GB)"""
        if not self.is_cuda_available:
            return 0.0
        try:
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            total = self.get_gpu_memory_gb()
            return max(0, total - reserved)
        except Exception:
            return 0.0

    def check_memory_available(self, required_gb: float = 2.0) -> bool:
        """检查显存是否足够"""
        if not self.is_cuda_available:
            return False
        return self.get_free_gpu_memory_gb() >= required_gb

    def clear_cache(self):
        """清理GPU缓存"""
        gc.collect()

        if self.is_cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_recommended_device(self, task: str = "general") -> str:
        """
        根据当前显存状态推荐设备

        Args:
            task: 任务类型 (tts, asr, translation, separator)

        Returns:
            推荐的设备 ("cuda" 或 "cpu")
        """
        if not self.is_cuda_available:
            return "cpu"

        free_mem = self.get_free_gpu_memory_gb()
        total_mem = self.get_gpu_memory_gb()

        # 不同任务的最小显存需求
        min_memory = {
            "tts": 4.0,        # TTS需要较大显存
            "asr": 1.5,        # Whisper ASR
            "translation": 2.0,  # M2M100翻译
            "separator": 3.0,  # Demucs分离
            "general": 2.0,
        }

        required = min_memory.get(task, 2.0)

        # 检查RAM压力
        if HAS_PSUTIL:
            ram_info = psutil.virtual_memory()
            if ram_info.percent > 90:
                print(f"警告: 系统内存使用率过高 ({ram_info.percent}%)")

        if free_mem >= required:
            return "cuda"
        else:
            # 尝试清理缓存
            self.clear_cache()
            free_mem = self.get_free_gpu_memory_gb()

            if free_mem >= required:
                return "cuda"
            else:
                print(f"显存不足 (需要 {required}GB, 可用 {free_mem:.1f}GB)，使用CPU")
                return "cpu"

    def print_status(self):
        """打印显存状态"""
        info = self.get_memory_info()
        print("\n" + "=" * 50)
        print("内存状态")
        print("=" * 50)

        if self.is_cuda_available:
            print(f"GPU显存: {info.gpu_used:.1f}GB / {info.gpu_total:.1f}GB (空闲: {info.gpu_free:.1f}GB)")
        else:
            print("GPU: 不可用")

        if HAS_PSUTIL:
            print(f"系统内存: {info.ram_used_percent:.1f}% (可用: {info.ram_available:.1f}GB)")
        print("=" * 50)


@contextmanager
def gpu_memory_context(clear_after: bool = True):
    """
    GPU显存上下文管理器

    用法:
        with gpu_memory_context():
            # 执行GPU操作
            pass
        # 自动清理
    """
    manager = GPUMemoryManager()
    try:
        yield manager
    finally:
        if clear_after:
            manager.clear_cache()


def with_memory_check(min_memory_gb: float = 2.0, fallback_device: str = "cpu"):
    """
    装饰器：检查显存并在不足时切换设备

    用法:
        @with_memory_check(min_memory_gb=4.0)
        def my_function(device="cuda"):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = GPUMemoryManager()

            # 检查是否需要切换设备
            if 'device' in kwargs:
                if kwargs['device'] == 'cuda' and not manager.check_memory_available(min_memory_gb):
                    kwargs['device'] = fallback_device

            return func(*args, **kwargs)
        return wrapper
    return decorator


class MemoryMonitor:
    """内存监控器 - 用于长时间运行的任务"""

    def __init__(self, interval: float = 1.0, threshold: float = 0.9):
        """
        Args:
            interval: 检查间隔(秒)
            threshold: 内存使用阈值(0-1)
        """
        self.interval = interval
        self.threshold = threshold
        self._running = False
        self._thread = None
        self._callbacks = []

    def add_callback(self, callback: Callable):
        """添加内存警告回调"""
        self._callbacks.append(callback)

    def start(self):
        """开始监控"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止监控"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _monitor_loop(self):
        """监控循环"""
        manager = GPUMemoryManager()

        while self._running:
            try:
                info = manager.get_memory_info()

                # 检查GPU显存
                if manager.is_cuda_available and info.gpu_total > 0:
                    usage = info.gpu_used / info.gpu_total
                    if usage > self.threshold:
                        for callback in self._callbacks:
                            try:
                                callback("gpu", usage)
                            except Exception:
                                pass

                time.sleep(self.interval)

            except Exception:
                time.sleep(self.interval)


# 全局实例
_memory_manager: Optional[GPUMemoryManager] = None


def get_memory_manager() -> GPUMemoryManager:
    """获取全局内存管理器"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = GPUMemoryManager()
    return _memory_manager


# 便捷函数
def clear_gpu_cache():
    """清理GPU缓存"""
    get_memory_manager().clear_cache()


def get_free_gpu_memory() -> float:
    """获取空闲GPU显存(GB)"""
    return get_memory_manager().get_free_gpu_memory_gb()


def check_gpu_available(required_gb: float = 2.0) -> bool:
    """检查GPU是否可用且显存足够"""
    return get_memory_manager().check_memory_available(required_gb)


def get_recommended_device(task: str = "general") -> str:
    """获取推荐设备"""
    return get_memory_manager().get_recommended_device(task)


def print_memory_status():
    """打印内存状态"""
    get_memory_manager().print_status()


if __name__ == "__main__":
    # 测试
    print("测试GPU内存管理器")
    print("-" * 50)

    manager = GPUMemoryManager()
    manager.print_status()

    print(f"\n推荐设备 (TTS): {manager.get_recommended_device('tts')}")
    print(f"推荐设备 (ASR): {manager.get_recommended_device('asr')}")
    print(f"推荐设备 (翻译): {manager.get_recommended_device('translation')}")
