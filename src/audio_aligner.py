#!/usr/bin/env python3
"""
音频时间对齐模块

解决TTS配音与原始时间轴对齐的问题：
1. 当TTS音频短于目标时长时：使用静音填充
2. 当TTS音频长于目标时长时：使用智能时间拉伸（保留完整内容）
3. 保持音频质量和自然度

使用librosa的时间拉伸功能，避免直接截断音频
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Union, Optional, Tuple
import warnings

# 尝试导入librosa进行高质量音频处理
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    warnings.warn("librosa未安装，将使用基础音频处理方法。建议: pip install librosa")


class AudioAligner:
    """音频时间对齐器"""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

    def align_audio(
        self,
        audio_path: Union[str, Path],
        target_duration: float,
        output_path: Union[str, Path],
        method: str = "auto",
        preserve_pitch: bool = True,
    ) -> Path:
        """
        将音频对齐到目标时长

        Args:
            audio_path: 输入音频路径
            target_duration: 目标时长（秒）
            output_path: 输出路径
            method: 对齐方法 ("auto", "stretch", "pad", "truncate")
            preserve_pitch: 是否保持音调（时间拉伸时使用）

        Returns:
            输出文件路径
        """
        audio_path = Path(audio_path)
        output_path = Path(output_path)

        # 读取音频
        audio, sr = sf.read(str(audio_path), dtype="float32")

        # 如果是立体声，转换为单声道处理
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # 重采样到目标采样率
        if sr != self.sample_rate:
            if HAS_LIBROSA:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            else:
                # 简单重采样
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * self.sample_rate / sr))
            sr = self.sample_rate

        current_duration = len(audio) / sr
        target_samples = int(target_duration * sr)

        print(f"  对齐音频: {current_duration:.2f}s -> {target_duration:.2f}s")

        # 决定处理方法
        if method == "auto":
            if current_duration < target_duration:
                method = "pad"
            elif current_duration > target_duration * 1.1:  # 超过10%才拉伸
                method = "stretch"
            else:
                method = "pad"  # 小差异用静音填充

        # 执行对齐
        if method == "stretch" and current_duration > target_duration:
            aligned = self._time_stretch(audio, target_samples, preserve_pitch)
        elif method == "pad" and current_duration < target_duration:
            aligned = self._pad_silence(audio, target_samples)
        elif method == "truncate":
            aligned = self._truncate(audio, target_samples)
        else:
            # 默认：智能处理
            aligned = self._smart_align(audio, target_samples)

        # 保存
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), aligned, sr, subtype="PCM_16")

        actual_duration = len(aligned) / sr
        print(f"  ✓ 对齐完成: {actual_duration:.2f}s (目标: {target_duration:.2f}s)")

        return output_path

    def _time_stretch(
        self, audio: np.ndarray, target_samples: int, preserve_pitch: bool = True
    ) -> np.ndarray:
        """
        时间拉伸音频到目标长度

        使用librosa的phase vocoder进行高质量时间拉伸，
        保持音调不变，避免声音变调
        """
        if not HAS_LIBROSA:
            # 回退到简单方法
            return self._simple_stretch(audio, target_samples)

        current_samples = len(audio)
        rate = current_samples / target_samples  # 拉伸率

        print(f"  时间拉伸: 速率={rate:.3f}x")

        try:
            # 使用librosa的时间拉伸
            # 参数rate: >1表示加速（缩短），<1表示减速（延长）
            stretched = librosa.effects.time_stretch(
                audio,
                rate=rate
            )

            # 确保长度精确匹配
            if len(stretched) > target_samples:
                stretched = stretched[:target_samples]
            elif len(stretched) < target_samples:
                padding = np.zeros(target_samples - len(stretched), dtype=np.float32)
                stretched = np.concatenate([stretched, padding])

            return stretched

        except Exception as e:
            print(f"  警告: 时间拉伸失败 ({e})，使用简单拉伸")
            return self._simple_stretch(audio, target_samples)

    def _simple_stretch(self, audio: np.ndarray, target_samples: int) -> np.ndarray:
        """
        简单线性插值拉伸（不使用librosa时的回退方案）
        质量较低但总是可用
        """
        from scipy import signal

        current_samples = len(audio)

        # 使用线性插值
        stretched = signal.resample(audio, target_samples)

        return stretched.astype(np.float32)

    def _pad_silence(self, audio: np.ndarray, target_samples: int) -> np.ndarray:
        """用静音填充到目标长度"""
        current_samples = len(audio)
        padding_samples = target_samples - current_samples

        # 创建静音（使用渐变避免爆音）
        padding = np.zeros(padding_samples, dtype=np.float32)

        # 添加10ms淡入淡出
        fade_samples = min(int(0.01 * self.sample_rate), padding_samples // 2)
        if fade_samples > 0:
            fade_out = np.linspace(1, 0, fade_samples)
            audio[-fade_samples:] *= fade_out

        aligned = np.concatenate([audio, padding])

        print(f"  填充静音: +{padding_samples / self.sample_rate:.2f}s")

        return aligned

    def _truncate(self, audio: np.ndarray, target_samples: int) -> np.ndarray:
        """截断音频（不推荐，会丢失内容）"""
        print(f"  警告: 截断音频 {(len(audio) - target_samples) / self.sample_rate:.2f}s")

        # 添加淡出避免爆音
        truncated = audio[:target_samples].copy()
        fade_samples = min(int(0.05 * self.sample_rate), target_samples // 4)
        if fade_samples > 0:
            fade_out = np.linspace(1, 0, fade_samples)
            truncated[-fade_samples:] *= fade_out

        return truncated

    def _smart_align(self, audio: np.ndarray, target_samples: int) -> np.ndarray:
        """
        智能对齐：根据时长差异选择最佳策略

        - 差异 < 0.5s: 静音填充
        - 差异 0.5-2s: 轻微时间拉伸 + 填充
        - 差异 > 2s: 较大时间拉伸
        """
        current_samples = len(audio)
        diff_samples = abs(target_samples - current_samples)
        diff_seconds = diff_samples / self.sample_rate

        if current_samples < target_samples:
            # 音频较短：填充静音
            return self._pad_silence(audio, target_samples)
        else:
            # 音频较长：使用时间拉伸
            if HAS_LIBROSA and diff_seconds > 0.5:
                # 使用高质量时间拉伸
                return self._time_stretch(audio, target_samples)
            else:
                # 小差异直接截断（带淡出）
                return self._truncate(audio, target_samples)

    def align_segments(
        self,
        segment_files: list,
        segments_info: list,
        output_path: Union[str, Path],
        crossfade_ms: float = 10.0,
    ) -> Path:
        """
        对齐多个TTS片段到时间轴并合并

        Args:
            segment_files: TTS音频文件路径列表
            segments_info: 片段信息列表（包含start_time, end_time）
            output_path: 输出路径
            crossfade_ms: 交叉淡入淡出时长（毫秒）

        Returns:
            合并后的音频路径
        """
        output_path = Path(output_path)

        if not segment_files or not segments_info:
            raise ValueError("片段列表不能为空")

        # 计算总时长
        total_duration = max(seg.end_time for seg in segments_info)
        total_samples = int(total_duration * self.sample_rate)

        print(f"\n对齐 {len(segment_files)} 个TTS片段到时间轴...")
        print(f"  总时长: {total_duration:.2f}s")
        print(f"  采样率: {self.sample_rate}Hz")

        # 初始化输出数组
        timeline_audio = np.zeros(total_samples, dtype=np.float32)

        fade_samples = int(crossfade_ms / 1000 * self.sample_rate)

        for i, (tts_file, seg_info) in enumerate(zip(segment_files, segments_info)):
            if not Path(tts_file).exists():
                print(f"  警告: 片段 {i} 文件不存在，跳过")
                continue

            # 读取TTS音频
            tts_audio, sr = sf.read(str(tts_file), dtype="float32")

            # 转换为单声道
            if tts_audio.ndim > 1:
                tts_audio = np.mean(tts_audio, axis=1)

            # 重采样
            if sr != self.sample_rate:
                if HAS_LIBROSA:
                    tts_audio = librosa.resample(tts_audio, orig_sr=sr, target_sr=self.sample_rate)
                else:
                    from scipy import signal
                    tts_audio = signal.resample(tts_audio, int(len(tts_audio) * self.sample_rate / sr))

            # 计算目标时长和位置
            target_duration = seg_info.end_time - seg_info.start_time
            target_samples = int(target_duration * self.sample_rate)
            start_sample = int(seg_info.start_time * self.sample_rate)

            # 对齐音频到目标时长
            tts_duration = len(tts_audio) / self.sample_rate

            if abs(tts_duration - target_duration) > 0.1:  # 差异超过100ms需要调整
                # 使用时间拉伸对齐
                aligned_audio = self._time_stretch(tts_audio, target_samples)
            else:
                # 长度合适，直接使用
                if len(tts_audio) > target_samples:
                    aligned_audio = tts_audio[:target_samples]
                    # 添加淡出
                    if fade_samples > 0:
                        fade_out = np.linspace(1, 0, min(fade_samples, len(aligned_audio)))
                        aligned_audio[-len(fade_out):] *= fade_out
                else:
                    aligned_audio = np.zeros(target_samples, dtype=np.float32)
                    aligned_audio[:len(tts_audio)] = tts_audio

            # 放置到时间轴
            end_sample = start_sample + len(aligned_audio)

            if end_sample <= total_samples:
                # 添加淡入
                if fade_samples > 0 and len(aligned_audio) > fade_samples * 2:
                    fade_in = np.linspace(0, 1, fade_samples)
                    aligned_audio[:fade_samples] *= fade_in

                timeline_audio[start_sample:end_sample] = aligned_audio
            else:
                # 超出范围，截断
                available = total_samples - start_sample
                if available > 0:
                    timeline_audio[start_sample:] = aligned_audio[:available]

        # 保存
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), timeline_audio, self.sample_rate, subtype="PCM_16")

        actual_duration = len(timeline_audio) / self.sample_rate
        print(f"✓ 时间轴对齐完成: {actual_duration:.2f}s")

        return output_path


def align_audio_duration(
    audio_path: Union[str, Path],
    target_duration: float,
    output_path: Union[str, Path],
    sample_rate: int = 24000,
    method: str = "auto",
) -> Path:
    """
    便捷函数：对齐音频到目标时长

    Args:
        audio_path: 输入音频路径
        target_duration: 目标时长（秒）
        output_path: 输出路径
        sample_rate: 采样率
        method: 对齐方法

    Returns:
        输出文件路径
    """
    aligner = AudioAligner(sample_rate=sample_rate)
    return aligner.align_audio(audio_path, target_duration, output_path, method)


def stretch_audio_to_duration(
    audio_path: Union[str, Path],
    target_duration: float,
    output_path: Union[str, Path],
    sample_rate: int = 24000,
) -> Path:
    """
    便捷函数：使用时间拉伸调整音频时长（保持完整内容）

    Args:
        audio_path: 输入音频路径
        target_duration: 目标时长（秒）
        output_path: 输出路径
        sample_rate: 采样率

    Returns:
        输出文件路径
    """
    aligner = AudioAligner(sample_rate=sample_rate)
    return aligner.align_audio(
        audio_path, target_duration, output_path, method="stretch", preserve_pitch=True
    )
