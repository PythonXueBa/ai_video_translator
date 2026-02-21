#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频合并模块
支持合并人声和背景音，调整音量比例
"""

import subprocess
from pathlib import Path
from typing import Optional, Union, List, Tuple

import numpy as np
import soundfile as sf

from .analyzer import MediaAnalyzer
from .logger import get_logger, log_execution

logger = get_logger(__name__)


class AudioMerger:
    """音频合并器"""

    @staticmethod
    def merge_with_volume(
        vocals_path: Union[str, Path],
        background_path: Union[str, Path],
        output_path: Union[str, Path],
        vocals_volume: float = 1.0,
        background_volume: float = 1.0,
        use_ffmpeg: bool = True,
    ) -> Path:
        """
        合并人声和背景音，可调整音量

        Args:
            vocals_path: 人声音频路径
            background_path: 背景音路径
            output_path: 输出路径
            vocals_volume: 人声音量倍率 (0.0-2.0)
            background_volume: 背景音量倍率 (0.0-2.0)
            use_ffmpeg: 是否使用FFmpeg（更快）还是Python（更精确）

        Returns:
            输出文件路径
        """
        vocals_path = Path(vocals_path)
        background_path = Path(background_path)
        output_path = Path(output_path)

        if not vocals_path.exists():
            raise FileNotFoundError(f"人声文件不存在: {vocals_path}")
        if not background_path.exists():
            raise FileNotFoundError(f"背景音文件不存在: {background_path}")

        logger.info(f"合并音频: 人声={vocals_path.name} (音量: {vocals_volume}x), 背景={background_path.name} (音量: {background_volume}x)")
        logger.debug(f"输出: {output_path.name}")

        if use_ffmpeg:
            return AudioMerger._merge_with_ffmpeg(
                vocals_path,
                background_path,
                output_path,
                vocals_volume,
                background_volume,
            )
        else:
            return AudioMerger._merge_with_python(
                vocals_path,
                background_path,
                output_path,
                vocals_volume,
                background_volume,
            )

    @staticmethod
    def _merge_with_ffmpeg(
        vocals_path: Path,
        background_path: Path,
        output_path: Path,
        vocals_vol: float,
        background_vol: float,
    ) -> Path:
        """使用FFmpeg合并音频"""
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(vocals_path),
            "-i",
            str(background_path),
            "-filter_complex",
            f"[0:a]volume={vocals_vol}[v];[1:a]volume={background_vol}[b];[v][b]amix=inputs=2:duration=longest",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"FFmpeg合并失败: {result.stderr}")
            raise RuntimeError(f"FFmpeg合并失败: {result.stderr}")

        logger.info(f"合并完成: {output_path}")
        return output_path

    @staticmethod
    def _merge_with_python(
        vocals_path: Path,
        background_path: Path,
        output_path: Path,
        vocals_vol: float,
        background_vol: float,
    ) -> Path:
        """使用Python合并音频（更精确）"""
        # 加载音频
        vocals, sr1 = sf.read(str(vocals_path), dtype="float32")
        background, sr2 = sf.read(str(background_path), dtype="float32")

        if sr1 != sr2:
            raise ValueError(f"采样率不匹配: {sr1} != {sr2}")

        # 确保相同长度
        min_len = min(len(vocals), len(background))
        vocals = vocals[:min_len]
        background = background[:min_len]

        # 调整音量并合并
        if vocals.ndim == 1:
            vocals = vocals.reshape(-1, 1)
        if background.ndim == 1:
            background = background.reshape(-1, 1)

        merged = vocals * vocals_vol + background * background_vol

        # 防止 clipping
        max_val = np.max(np.abs(merged))
        if max_val > 1.0:
            merged = merged / max_val * 0.95
            logger.warning(f"音频已归一化（峰值 {max_val:.2f}）")

        # 保存
        sf.write(str(output_path), merged, sr1, subtype="PCM_16")
        logger.info(f"合并完成: {output_path}")

        return output_path

    @staticmethod
    def merge_multiple(
        audio_files: List[Tuple[Union[str, Path], float]], output_path: Union[str, Path]
    ) -> Path:
        """
        合并多个音频文件

        Args:
            audio_files: [(文件路径, 音量倍率), ...]
            output_path: 输出路径

        Returns:
            输出文件路径
        """
        if len(audio_files) < 2:
            raise ValueError("至少需要2个音频文件")

        output_path = Path(output_path)

        # 加载第一个音频
        first_path, first_vol = audio_files[0]
        merged, sr = sf.read(str(first_path), dtype="float32")
        merged = merged * first_vol

        # 合并其余音频
        for audio_path, volume in audio_files[1:]:
            audio, sr2 = sf.read(str(audio_path), dtype="float32")
            if sr != sr2:
                raise ValueError(f"采样率不匹配: {sr} != {sr2}")

            min_len = min(len(merged), len(audio))
            if len(merged) > len(audio):
                # 扩展音频
                temp = np.zeros_like(merged)
                temp[:min_len] = audio[:min_len] * volume
            else:
                temp = np.zeros_like(audio)
                temp[:min_len] = merged[:min_len]
                merged = temp
                merged[:min_len] += audio[:min_len] * volume

        # 防止 clipping
        max_val = np.max(np.abs(merged))
        if max_val > 1.0:
            merged = merged / max_val * 0.95

        # 保存
        sf.write(str(output_path), merged, sr, subtype="PCM_16")
        logger.info(f"合并 {len(audio_files)} 个音频: {output_path}")

        return output_path
