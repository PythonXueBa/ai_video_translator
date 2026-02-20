#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频提取模块
"""

import subprocess
from pathlib import Path
from typing import Optional, Union

from .analyzer import MediaAnalyzer, MediaInfo


class AudioExtractor:
    """音频提取器"""

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        audio_format: str = "wav",
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_format = audio_format

    def extract(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        verify: bool = True,
    ) -> Path:
        """从视频中提取音频"""
        video_path = Path(video_path)

        # 分析视频
        info = MediaAnalyzer.analyze_video(video_path)
        video_info = info.get("video")
        audio_info = info.get("audio")

        if not audio_info:
            raise ValueError("视频中没有音频流")

        # 确定输出路径
        if output_path is None:
            output_path = video_path.with_suffix(f".{self.audio_format}")
        else:
            output_path = Path(output_path)

        # 使用原始参数或指定参数
        target_sr = self.sample_rate or audio_info.sample_rate
        target_ch = self.channels or audio_info.channels

        print(f"提取音频: {video_path.name}")
        print(f"  -> {output_path.name}")
        print(f"  采样率: {target_sr} Hz, 声道: {target_ch}")

        # 构建 FFmpeg 命令
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-map",
            "0:a:0",
            "-ar",
            str(target_sr),
            "-ac",
            str(target_ch),
        ]

        # 设置编码器
        if self.audio_format == "mp3":
            cmd.extend(["-c:a", "libmp3lame", "-q:a", "0"])
        elif self.audio_format == "flac":
            cmd.extend(["-c:a", "flac"])
        else:
            cmd.extend(["-c:a", "pcm_s16le"])

        cmd.append(str(output_path))

        # 执行
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg 错误: {result.stderr}")

        # 验证
        if verify and output_path.exists():
            output_info = MediaAnalyzer.analyze_audio(output_path)
            duration_diff = abs(audio_info.duration - output_info.duration)

            if duration_diff > 1.0:
                print(f"⚠ 警告: 时长不匹配 (差异 {duration_diff:.2f}s)")
            else:
                print(f"✓ 提取完成: {output_info.duration:.2f}s")

        return output_path
