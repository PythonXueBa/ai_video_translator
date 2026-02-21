#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频音频分析模块
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union
from dataclasses import dataclass

from .logger import get_logger, log_execution

logger = get_logger(__name__)


@dataclass
class MediaInfo:
    """媒体文件信息"""

    path: Path
    duration: float
    codec: str
    sample_rate: int
    channels: int
    bit_rate: int

    def __str__(self) -> str:
        minutes = int(self.duration // 60)
        seconds = int(self.duration % 60)
        return (
            f"{self.path.name}\n"
            f"  时长: {minutes}:{seconds:02d} ({self.duration:.2f}s)\n"
            f"  编码: {self.codec}\n"
            f"  采样率: {self.sample_rate} Hz\n"
            f"  声道: {self.channels}\n"
            f"  码率: {self.bit_rate / 1000:.0f} kbps"
        )


class MediaAnalyzer:
    """媒体文件分析器"""

    @staticmethod
    def probe(media_path: Union[str, Path]) -> Dict:
        """使用 ffprobe 分析媒体文件"""
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(media_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"ffprobe 失败: {result.stderr}")

        return json.loads(result.stdout)

    @classmethod
    def analyze_video(cls, video_path: Union[str, Path]) -> Dict[str, MediaInfo]:
        """分析视频文件，返回视频流和音频流信息"""
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        data = cls.probe(video_path)
        format_info = data.get("format", {})
        streams = data.get("streams", [])

        result = {}

        for stream in streams:
            codec_type = stream.get("codec_type")

            if codec_type == "video":
                result["video"] = MediaInfo(
                    path=video_path,
                    duration=float(format_info.get("duration", 0)),
                    codec=stream.get("codec_name", "unknown"),
                    sample_rate=0,
                    channels=0,
                    bit_rate=int(stream.get("bit_rate", 0)),
                )
            elif codec_type == "audio":
                result["audio"] = MediaInfo(
                    path=video_path,
                    duration=float(format_info.get("duration", 0)),
                    codec=stream.get("codec_name", "unknown"),
                    sample_rate=int(stream.get("sample_rate", 44100)),
                    channels=int(stream.get("channels", 2)),
                    bit_rate=int(stream.get("bit_rate", 0)),
                )

        return result

    @classmethod
    def analyze_audio(cls, audio_path: Union[str, Path]) -> MediaInfo:
        """分析音频文件"""
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        data = cls.probe(audio_path)
        format_info = data.get("format", {})
        streams = data.get("streams", [])

        # 找到音频流
        audio_stream = None
        for stream in streams:
            if stream.get("codec_type") == "audio":
                audio_stream = stream
                break

        if not audio_stream:
            raise ValueError("未找到音频流")

        return MediaInfo(
            path=audio_path,
            duration=float(format_info.get("duration", 0)),
            codec=audio_stream.get("codec_name", "unknown"),
            sample_rate=int(audio_stream.get("sample_rate", 44100)),
            channels=int(audio_stream.get("channels", 2)),
            bit_rate=int(audio_stream.get("bit_rate", 0)),
        )


def check_ffmpeg() -> bool:
    """检查 FFmpeg 是否安装"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except:
        return False


def get_ffmpeg_version() -> str:
    """获取 FFmpeg 版本"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
        )
        return result.stdout.split("\n")[0]
    except:
        return "Unknown"
