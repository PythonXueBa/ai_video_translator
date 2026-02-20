#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频分割模块 - 基于静音检测的智能分段
确保分割点位于静音区域，保持总时长一致
"""

import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

import numpy as np
import soundfile as sf


@dataclass
class AudioSegment:
    """音频片段信息"""

    index: int
    start_time: float  # 秒
    end_time: float  # 秒
    duration: float  # 秒
    text: str = ""  # 识别文本
    translated_text: str = ""  # 翻译文本
    audio_path: Optional[Path] = None  # 片段音频路径

    def __str__(self):
        return f"片段{self.index}: {self.start_time:.2f}s - {self.end_time:.2f}s ({self.duration:.2f}s)"


class AudioSplitter:
    """音频分割器 - 基于静音检测"""

    def __init__(
        self,
        min_duration: float = 5.0,  # 最短5秒
        max_duration: float = 30.0,  # 最长30秒
        silence_threshold: float = 0.02,  # 静音能量阈值
        min_silence_duration: float = 0.3,  # 最短静音时长（秒）
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration

    def detect_silence_regions(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> List[Tuple[int, int]]:
        """
        检测静音区域

        Returns:
            静音区域的 (start_sample, end_sample) 列表
        """
        # 转换为单声道
        if audio_data.ndim > 1:
            audio_mono = np.mean(audio_data, axis=1)
        else:
            audio_mono = audio_data

        # 计算短时能量（使用滑动窗口）
        window_size = int(sample_rate * 0.02)  # 20ms窗口
        hop_size = int(sample_rate * 0.01)  # 10ms步长

        energy = []
        for i in range(0, len(audio_mono) - window_size, hop_size):
            window = audio_mono[i : i + window_size]
            # 使用RMS能量
            rms = np.sqrt(np.mean(window**2))
            energy.append(rms)

        energy = np.array(energy)

        # 能量低于阈值为静音
        is_silence = energy < self.silence_threshold

        # 找连续的静音区域
        silence_regions = []
        in_silence = False
        silence_start = 0

        for i, silent in enumerate(is_silence):
            if silent and not in_silence:
                silence_start = i
                in_silence = True
            elif not silent and in_silence:
                silence_end = i
                silence_duration = (
                    (silence_end - silence_start) * hop_size / sample_rate
                )

                # 只记录超过最小静音时长的区域
                if silence_duration >= self.min_silence_duration:
                    start_sample = silence_start * hop_size
                    end_sample = silence_end * hop_size
                    silence_regions.append((start_sample, end_sample))

                in_silence = False

        # 处理最后一个静音区域
        if in_silence:
            silence_end = len(is_silence)
            silence_duration = (silence_end - silence_start) * hop_size / sample_rate
            if silence_duration >= self.min_silence_duration:
                start_sample = silence_start * hop_size
                end_sample = min(silence_end * hop_size, len(audio_mono))
                silence_regions.append((start_sample, end_sample))

        return silence_regions

    def find_best_split_point(
        self,
        start_sample: int,
        end_sample: int,
        silence_regions: List[Tuple[int, int]],
        sample_rate: int,
    ) -> int:
        """
        在指定范围内找最佳分割点（优先选择静音区域的中间）
        """
        min_duration_samples = int(self.min_duration * sample_rate)
        max_duration_samples = int(self.max_duration * sample_rate)

        target_end = start_sample + max_duration_samples
        if target_end > end_sample:
            target_end = end_sample

        # 在目标范围内找静音区域
        best_point = None
        best_score = -1

        for silence_start, silence_end in silence_regions:
            # 静音区域必须在目标范围内
            if (
                silence_start >= start_sample + min_duration_samples
                and silence_end <= target_end
            ):
                # 计算得分（优先选择居中的点）
                center = (silence_start + silence_end) // 2
                duration_from_start = center - start_sample

                # 越接近 max_duration 得分越高，但不超过
                if (
                    self.min_duration * sample_rate
                    <= duration_from_start
                    <= max_duration_samples
                ):
                    score = duration_from_start
                    if score > best_score:
                        best_score = score
                        best_point = center

        # 如果没找到合适的静音点，在 max_duration 处分割
        if best_point is None:
            best_point = min(start_sample + max_duration_samples, end_sample)

        return best_point

    def split_audio(
        self,
        audio_path: Union[str, Path],
        output_dir: Union[str, Path],
        prefix: str = "segment",
    ) -> List[AudioSegment]:
        """
        分割音频为多个片段

        策略：
        1. 检测所有静音区域
        2. 在静音区域之间分割
        3. 确保每段 5-30 秒
        4. 保证总时长一致（从0到音频结束）

        Args:
            audio_path: 输入音频路径
            output_dir: 输出目录
            prefix: 片段文件名前缀

        Returns:
            片段信息列表
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n分割音频: {audio_path.name}")

        # 加载音频
        audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")
        total_samples = len(audio_data)
        total_duration = total_samples / sample_rate

        print(f"总时长: {total_duration:.2f}s")
        print(f"采样率: {sample_rate}Hz")
        print(f"总采样点: {total_samples}")

        # 检测静音区域
        silence_regions = self.detect_silence_regions(audio_data, sample_rate)
        print(f"检测到 {len(silence_regions)} 个静音区域")

        # 确定分割点
        split_points = [0]  # 从0开始
        current_pos = 0

        while current_pos < total_samples:
            # 找下一个分割点
            next_point = self.find_best_split_point(
                current_pos, total_samples, silence_regions, sample_rate
            )

            if next_point <= current_pos or next_point >= total_samples:
                break

            split_points.append(next_point)
            current_pos = next_point

        # 确保最后一个点是音频结束
        if split_points[-1] < total_samples:
            split_points.append(total_samples)

        print(f"分割点数量: {len(split_points)} (包括起点和终点)")

        # 提取并保存每个片段
        segment_list = []
        for i in range(len(split_points) - 1):
            start_sample = split_points[i]
            end_sample = split_points[i + 1]

            # 确保不越界
            end_sample = min(end_sample, total_samples)

            # 提取音频
            segment_audio = audio_data[start_sample:end_sample]

            # 保存片段
            segment_path = output_dir / f"{prefix}_{i:04d}.wav"
            sf.write(str(segment_path), segment_audio, sample_rate, subtype="PCM_16")

            # 计算时间
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            duration = end_time - start_time

            segment = AudioSegment(
                index=i,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                audio_path=segment_path,
            )
            segment_list.append(segment)

            print(f"  ✓ 片段{i}: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")

        # 验证总时长
        total_segment_duration = sum(seg.duration for seg in segment_list)
        print(f"\n验证:")
        print(f"  原始时长: {total_duration:.2f}s")
        print(f"  片段总时长: {total_segment_duration:.2f}s")
        print(f"  差异: {abs(total_duration - total_segment_duration):.3f}s")

        if abs(total_duration - total_segment_duration) > 0.1:
            print(f"  ⚠ 警告: 时长差异较大！")
        else:
            print(f"  ✓ 时长一致")

        return segment_list

    def merge_segments(
        self,
        segments: List[AudioSegment],
        output_path: Union[str, Path],
        crossfade_duration: float = 0.01,  # 10ms 淡入淡出
    ) -> Path:
        """
        合并多个音频片段

        Args:
            segments: 片段列表（每个片段需要有 audio_path）
            output_path: 输出路径
            crossfade_duration: 交叉淡入淡出时长（秒）

        Returns:
            合并后的音频路径
        """
        output_path = Path(output_path)

        print(f"\n合并 {len(segments)} 个片段...")

        all_audio = []
        sample_rate = None
        total_expected_duration = 0

        for i, segment in enumerate(segments):
            if not segment.audio_path or not segment.audio_path.exists():
                print(f"  ✗ 跳过片段{i}: 音频文件不存在")
                continue

            audio, sr = sf.read(str(segment.audio_path), dtype="float32")

            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                print(f"  ⚠ 片段{i} 采样率不匹配，跳过")
                continue

            # 验证时长
            actual_duration = len(audio) / sr
            expected_duration = segment.duration

            if abs(actual_duration - expected_duration) > 0.1:
                print(
                    f"  ⚠ 片段{i} 时长不匹配: 期望 {expected_duration:.2f}s, 实际 {actual_duration:.2f}s"
                )

            # 添加极短的淡入淡出避免爆音
            fade_samples = int(crossfade_duration * sample_rate)
            if len(audio) > fade_samples * 2:
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)

                if audio.ndim == 1:
                    audio[:fade_samples] *= fade_in
                    audio[-fade_samples:] *= fade_out
                else:
                    audio[:fade_samples] *= fade_in.reshape(-1, 1)
                    audio[-fade_samples:] *= fade_out.reshape(-1, 1)

            all_audio.append(audio)
            total_expected_duration += segment.duration

        if not all_audio:
            raise ValueError("没有可合并的音频")

        # 合并
        merged = np.concatenate(all_audio, axis=0)
        actual_total_duration = len(merged) / sample_rate

        # 保存
        sf.write(str(output_path), merged, sample_rate, subtype="PCM_16")

        print(f"✓ 合并完成: {output_path}")
        print(f"  期望总时长: {total_expected_duration:.2f}s")
        print(f"  实际总时长: {actual_total_duration:.2f}s")

        return output_path
