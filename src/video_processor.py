#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频处理模块 - 音频替换和无声视频
"""

import subprocess
from pathlib import Path
from typing import Optional, Union

from .analyzer import MediaAnalyzer
from .logger import get_logger, log_execution

logger = get_logger(__name__)


class VideoProcessor:
    """视频处理器"""

    @staticmethod
    def replace_audio(
        video_path: Union[str, Path],
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        verify: bool = True,
    ) -> Path:
        """
        替换视频音频

        Args:
            video_path: 原视频路径
            audio_path: 新音频路径
            output_path: 输出路径（可选）
            verify: 是否验证

        Returns:
            输出视频路径
        """
        video_path = Path(video_path)
        audio_path = Path(audio_path)

        if not video_path.exists():
            raise FileNotFoundError(f"视频不存在: {video_path}")
        if not audio_path.exists():
            raise FileNotFoundError(f"音频不存在: {audio_path}")

        # 确定输出路径
        if output_path is None:
            output_path = (
                video_path.parent / f"{video_path.stem}_with_{audio_path.stem}{video_path.suffix}"
            )
        else:
            output_path = Path(output_path)

        logger.info(f"替换视频音频: 视频={video_path.name}, 音频={audio_path.name}")
        logger.debug(f"输出: {output_path.name}")

        # FFmpeg 命令: 替换音频
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),  # 输入视频
            "-i",
            str(audio_path),  # 输入音频
            "-c:v",
            "copy",  # 复制视频流（不重新编码）
            "-c:a",
            "aac",  # 音频编码为 AAC
            "-b:a",
            "320k",  # 音频码率
            "-shortest",  # 以较短者为准
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"FFmpeg 错误: {result.stderr}")
            raise RuntimeError(f"FFmpeg 错误: {result.stderr}")

        # 验证
        if verify and output_path.exists():
            info = MediaAnalyzer.analyze_video(output_path)
            audio_info = info.get("audio")
            if audio_info:
                logger.info(f"替换完成: {audio_info.duration:.2f}s")

        return output_path

    @staticmethod
    def adjust_audio_duration(
        audio_path: Union[str, Path],
        target_duration: float,
        output_path: Union[str, Path],
    ) -> Path:
        """
        调整音频时长以匹配目标时长

        Args:
            audio_path: 输入音频路径
            target_duration: 目标时长（秒）
            output_path: 输出路径

        Returns:
            调整后的音频路径
        """
        import soundfile as sf
        import numpy as np

        audio_path = Path(audio_path)
        output_path = Path(output_path)

        # 读取音频
        audio, sr = sf.read(str(audio_path), dtype='float32')
        current_duration = len(audio) / sr
        target_samples = int(target_duration * sr)

        logger.info(f"调整音频时长: {audio_path.name}, 当前={current_duration:.2f}s, 目标={target_duration:.2f}s")

        if len(audio) < target_samples:
            # 音频较短，用静音填充
            padding = np.zeros(target_samples - len(audio), dtype=np.float32)
            if audio.ndim == 1:
                adjusted = np.concatenate([audio, padding])
            else:
                padding = padding.reshape(-1, 1)
                adjusted = np.vstack([audio, padding])
            logger.info(f"操作: 填充静音 {len(padding) / sr:.2f}s")
        elif len(audio) > target_samples:
            # 音频较长，截断
            adjusted = audio[:target_samples]
            logger.info(f"操作: 截断 {(len(audio) - target_samples) / sr:.2f}s")
        else:
            adjusted = audio
            logger.info(f"操作: 无需调整")

        # 保存
        sf.write(str(output_path), adjusted, sr)
        logger.info(f"音频时长调整完成: {output_path.name}")

        return output_path

    @staticmethod
    def combine_audio_to_video(
        silent_video_path: Union[str, Path],
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        verify: bool = True,
    ) -> Path:
        """
        将音频与静音视频合成为最终视频

        Args:
            silent_video_path: 静音视频路径
            audio_path: 音频路径
            output_path: 输出路径
            verify: 是否验证

        Returns:
            输出视频路径
        """
        silent_video_path = Path(silent_video_path)
        audio_path = Path(audio_path)
        output_path = Path(output_path)

        if not silent_video_path.exists():
            raise FileNotFoundError(f"静音视频不存在: {silent_video_path}")
        if not audio_path.exists():
            raise FileNotFoundError(f"音频不存在: {audio_path}")

        logger.info(f"合成视频: 视频={silent_video_path.name}, 音频={audio_path.name}")
        logger.debug(f"输出: {output_path.name}")

        # FFmpeg 命令: 合成视频
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(silent_video_path),  # 输入静音视频
            "-i", str(audio_path),  # 输入音频
            "-c:v", "copy",  # 复制视频流
            "-c:a", "aac",  # 音频编码为 AAC
            "-b:a", "320k",  # 音频码率
            "-shortest",  # 以较短者为准
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"FFmpeg 错误: {result.stderr}")
            raise RuntimeError(f"FFmpeg 错误: {result.stderr}")

        # 验证
        if verify and output_path.exists():
            info = MediaAnalyzer.analyze_video(output_path)
            video_info = info.get("video")
            audio_info = info.get("audio")
            if video_info and audio_info:
                logger.info(f"视频合成完成: 视频 {video_info.duration:.2f}s, 音频 {audio_info.duration:.2f}s")

        return output_path

    @staticmethod
    def remove_audio(
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        verify: bool = True,
    ) -> Path:
        """
        移除视频音频（生成无声视频）

        Args:
            video_path: 原视频路径
            output_path: 输出路径（可选）
            verify: 是否验证

        Returns:
            无声视频路径
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"视频不存在: {video_path}")

        # 确定输出路径
        if output_path is None:
            output_path = (
                video_path.parent / f"{video_path.stem}_silent{video_path.suffix}"
            )
        else:
            output_path = Path(output_path)

        logger.info(f"生成无声视频: 输入={video_path.name}, 输出={output_path.name}")

        # FFmpeg 命令: 移除音频
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-c:v",
            "copy",  # 复制视频
            "-an",  # 禁用音频
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"FFmpeg 错误: {result.stderr}")
            raise RuntimeError(f"FFmpeg 错误: {result.stderr}")

        # 验证
        if verify and output_path.exists():
            info = MediaAnalyzer.analyze_video(output_path)
            audio_info = info.get("audio")
            if audio_info is None:
                logger.info(f"无声视频生成成功")
            else:
                logger.warning(f"视频仍包含音频")

        return output_path
