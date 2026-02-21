#!/usr/bin/env python3
"""
字幕驱动TTS引擎 - 基于字幕时间轴的精准语音合成

核心功能：
1. 根据英文字幕切割原始音频获取参考音色
2. 计算目标时长（英文字幕时长）
3. 通过Qwen3-TTS参数调节语速匹配时长
4. 音频拉伸/压缩作为后备对齐方案
"""

import gc
import re
import math
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Union, Optional, Tuple, Dict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings
from pydub import AudioSegment as PydubAudioSegment

warnings.filterwarnings("ignore")

from .subtitle_handler import SubtitleEntry, SRTHandler
from .tts_qwen3 import Qwen3TTS, get_qwen3_tts, clear_gpu_memory, get_gpu_memory_info


@dataclass
class TTSConfig:
    """TTS配置参数"""
    # 语速调节参数 - 默认禁用，保持原始克隆音色和语速
    speed_adjustment: bool = False  # 禁用语速调节，保持自然语速
    target_duration_tolerance: float = 0.5  # 目标时长容差（秒）
    max_speed_ratio: float = 1.5  # 最大加速比例
    min_speed_ratio: float = 0.7  # 最大减速比例

    # 音频后处理参数 - 禁用拉伸，保持自然音色
    enable_stretching: bool = False  # 禁用音频拉伸/压缩
    stretching_threshold: float = 0.3  # 拉伸阈值（差异超过此值才拉伸）

    # 音色克隆参数
    use_voice_clone: bool = True  # 启用音色克隆
    reference_audio_padding: float = 0.5  # 参考音频前后留白（秒），增加以获取更多音色上下文

    # 生成参数
    default_sample_rate: int = 24000
    max_retries: int = 3  # 最大重试次数


@dataclass
class SubtitleTTSTask:
    """字幕TTS任务"""
    index: int
    english_text: str
    chinese_text: str
    target_duration: float  # 目标时长（基于英文字幕）
    start_time: float  # 开始时间（秒）
    end_time: float  # 结束时间（秒）
    reference_audio_path: Optional[Path] = None  # 参考音频路径
    output_path: Optional[Path] = None  # 输出路径
    generated_duration: float = 0.0  # 实际生成时长
    speed_ratio: float = 1.0  # 使用的语速比例
    success: bool = False
    error_message: str = ""


class AudioProcessor:
    """音频处理工具类"""

    @staticmethod
    def extract_segment(
        input_audio: Union[str, Path],
        output_path: Union[str, Path],
        start_time: float,
        end_time: float,
        padding: float = 0.2,
    ) -> Path:
        """
        从音频中提取指定时间段的片段

        Args:
            input_audio: 输入音频路径
            output_path: 输出路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            padding: 前后留白时间（秒）

        Returns:
            输出音频路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 添加padding，但确保不超出音频边界
        # 增加padding以获取更多上下文，改善音色克隆效果
        actual_start = max(0, start_time - padding * 2)
        duration = (end_time - start_time) + padding * 4

        try:
            # 使用pydub提取片段
            audio = PydubAudioSegment.from_file(str(input_audio))

            # 转换为毫秒
            start_ms = int(actual_start * 1000)
            end_ms = int((actual_start + duration) * 1000)

            # 确保不超出边界
            end_ms = min(end_ms, len(audio))

            segment = audio[start_ms:end_ms]
            segment.export(str(output_path), format="wav")

            return output_path
        except Exception as e:
            print(f"  警告: 音频提取失败 {e}，尝试使用soundfile")
            # 后备方案：使用soundfile
            try:
                audio_data, sample_rate = sf.read(str(input_audio), dtype="float32")

                start_sample = int(actual_start * sample_rate)
                end_sample = int((end_time + padding) * sample_rate)
                end_sample = min(end_sample, len(audio_data))

                segment = audio_data[start_sample:end_sample]
                sf.write(str(output_path), segment, sample_rate)

                return output_path
            except Exception as e2:
                print(f"  错误: 音频提取完全失败 {e2}")
                raise

    @staticmethod
    def stretch_audio(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_duration: float,
        method: str = "wsola",  # wsola 或 phase_vocoder
    ) -> float:
        """
        拉伸/压缩音频到目标时长

        Args:
            input_path: 输入音频路径
            output_path: 输出路径
            target_duration: 目标时长（秒）
            method: 拉伸方法

        Returns:
            实际拉伸比例
        """
        output_path = Path(output_path)

        try:
            # 加载音频
            audio, sample_rate = sf.read(str(input_path), dtype="float32")

            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            current_duration = len(audio) / sample_rate

            if current_duration <= 0:
                return 1.0

            # 计算拉伸比例
            stretch_ratio = target_duration / current_duration

            # 如果差异很小，不需要拉伸
            if abs(stretch_ratio - 1.0) < 0.05:
                sf.write(str(output_path), audio, sample_rate)
                return 1.0

            # 使用librosa进行时间拉伸
            try:
                import librosa

                # 限制拉伸比例
                stretch_ratio = np.clip(stretch_ratio, 0.5, 2.0)

                if method == "phase_vocoder":
                    # 使用相位声码器
                    stft = librosa.stft(audio)
                    stretched_stft = librosa.phase_vocoder(
                        stft, rate=1.0/stretch_ratio, hop_length=512
                    )
                    stretched = librosa.istft(stretched_stft)
                else:
                    # 使用WSOLA（默认）
                    stretched = librosa.effects.time_stretch(
                        audio, rate=1.0/stretch_ratio
                    )

                sf.write(str(output_path), stretched, sample_rate)
                return stretch_ratio

            except ImportError:
                print("  警告: librosa未安装，使用简单重采样")
                # 简单重采样（会改变音调）
                new_length = int(len(audio) * stretch_ratio)
                indices = np.linspace(0, len(audio) - 1, new_length)
                stretched = np.interp(indices, np.arange(len(audio)), audio)
                sf.write(str(output_path), stretched, sample_rate)
                return stretch_ratio

        except Exception as e:
            print(f"  音频拉伸失败: {e}")
            # 复制原文件
            import shutil
            shutil.copy(str(input_path), str(output_path))
            return 1.0

    @staticmethod
    def adjust_audio_duration(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_duration: float,
        tolerance: float = 0.1,
    ) -> Tuple[float, float]:
        """
        调整音频时长到目标值

        Returns:
            (原始时长, 调整后时长)
        """
        output_path = Path(output_path)

        # 获取当前时长
        audio, sample_rate = sf.read(str(input_path), dtype="float32")
        current_duration = len(audio) / sample_rate

        # 检查是否需要调整
        if abs(current_duration - target_duration) <= tolerance:
            sf.write(str(output_path), audio, sample_rate)
            return current_duration, current_duration

        # 拉伸音频
        stretch_ratio = AudioProcessor.stretch_audio(
            input_path, output_path, target_duration
        )

        # 验证结果
        adjusted_audio, _ = sf.read(str(output_path), dtype="float32")
        adjusted_duration = len(adjusted_audio) / sample_rate

        return current_duration, adjusted_duration


class SubtitleTTSEngine:
    """字幕驱动TTS引擎"""

    def __init__(
        self,
        tts_model: Optional[Qwen3TTS] = None,
        config: Optional[TTSConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or TTSConfig()
        self.audio_processor = AudioProcessor()

        # 初始化TTS模型
        if tts_model:
            self.tts = tts_model
        else:
            self.tts = get_qwen3_tts(device=device)

        self._model_lock = threading.Lock()

    def prepare_tasks(
        self,
        english_srt_path: Union[str, Path],
        chinese_srt_path: Union[str, Path],
        original_audio_path: Union[str, Path],
        output_dir: Union[str, Path],
        reference_audio_dir: Optional[Union[str, Path]] = None,
        preprocess_subtitles: bool = True,
    ) -> List[SubtitleTTSTask]:
        """
        准备TTS任务列表

        Args:
            english_srt_path: 英文字幕路径
            chinese_srt_path: 中文字幕路径
            original_audio_path: 原始音频路径（用于提取参考音色）
            output_dir: TTS输出目录
            reference_audio_dir: 参考音频保存目录（可选）
            preprocess_subtitles: 是否预处理字幕（合并短字幕、分割长字幕）

        Returns:
            TTS任务列表
        """
        print("\n" + "=" * 60)
        print("准备字幕驱动TTS任务")
        print("=" * 60)

        # 解析字幕
        english_entries = SRTHandler.parse(Path(english_srt_path))
        chinese_entries = SRTHandler.parse(Path(chinese_srt_path))

        # 预处理字幕
        if preprocess_subtitles:
            print("\n预处理字幕...")
            print(f"  原始英文字幕: {len(english_entries)} 条")
            print(f"  原始中文字幕: {len(chinese_entries)} 条")

            # 合并过短的字幕，优化避免将一个声音分成两个
            english_entries = SRTHandler.merge_short_entries(
                english_entries, min_duration=0.5, max_gap=1.0
            )

            # 重新同步中文字幕（如果数量变化）
            if len(english_entries) != len(chinese_entries):
                print(f"  合并后英文字幕: {len(english_entries)} 条")
                print(f"  注意：中文字幕数量不匹配，将使用英文字幕时间轴")
                # 使用中文字幕内容，但英文字幕时间轴
                if len(chinese_entries) > 0:
                    chinese_texts = [e.text for e in chinese_entries]
                    # 重新分配文本到合并后的时间轴 - 按比例分配避免重复
                    new_chinese_entries = []
                    total_texts = len(chinese_texts)
                    total_en = len(english_entries)

                    for i, en_entry in enumerate(english_entries):
                        # 计算当前条目应该分配的中文字幕范围
                        start_idx = int(i * total_texts / total_en)
                        end_idx = int((i + 1) * total_texts / total_en)

                        # 确保不超出范围且不重复
                        if i == total_en - 1:  # 最后一个条目包含剩余所有
                            end_idx = total_texts

                        combined_text = " ".join(chinese_texts[start_idx:end_idx])
                        # 关键：使用英文字幕的时间轴
                        new_chinese_entries.append(SubtitleEntry(
                            index=en_entry.index,
                            start_time=en_entry.start_time,
                            end_time=en_entry.end_time,
                            text=combined_text,
                        ))

                    chinese_entries = new_chinese_entries

        if len(english_entries) != len(chinese_entries):
            print(f"警告: 英文字幕({len(english_entries)})和中文字幕({len(chinese_entries)})数量不匹配")
            # 取最小值
            min_count = min(len(english_entries), len(chinese_entries))
            english_entries = english_entries[:min_count]
            chinese_entries = chinese_entries[:min_count]

        print(f"字幕条目数: {len(english_entries)}")
        print(f"  平均时长: {sum(e.duration for e in english_entries) / len(english_entries):.2f}s")
        print(f"  最短时长: {min(e.duration for e in english_entries):.2f}s")
        print(f"  最长时长: {max(e.duration for e in english_entries):.2f}s")

        # 准备目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if reference_audio_dir:
            reference_audio_dir = Path(reference_audio_dir)
            reference_audio_dir.mkdir(parents=True, exist_ok=True)

        # 创建任务列表
        tasks = []
        for i, (en_entry, zh_entry) in enumerate(zip(english_entries, chinese_entries)):
            task = SubtitleTTSTask(
                index=i,
                english_text=en_entry.text,
                chinese_text=zh_entry.text,
                target_duration=en_entry.end_seconds - en_entry.start_seconds,
                start_time=en_entry.start_seconds,
                end_time=en_entry.end_seconds,
                output_path=output_dir / f"tts_segment_{i:04d}.wav",
            )

            # 提取参考音频
            if self.config.use_voice_clone and reference_audio_dir:
                try:
                    ref_path = reference_audio_dir / f"ref_segment_{i:04d}.wav"
                    self.audio_processor.extract_segment(
                        original_audio_path,
                        ref_path,
                        task.start_time,
                        task.end_time,
                        padding=self.config.reference_audio_padding,
                    )
                    task.reference_audio_path = ref_path
                except Exception as e:
                    print(f"  警告: 片段{i}参考音频提取失败: {e}")

            tasks.append(task)

        print(f"✓ 创建了 {len(tasks)} 个TTS任务")

        # 统计时长信息
        total_target_duration = sum(t.target_duration for t in tasks)
        print(f"目标总时长: {total_target_duration:.2f}s")
        print(f"平均片段时长: {total_target_duration / len(tasks):.2f}s")

        return tasks

    def _estimate_duration(
        self, text: str, language: str = "zh", speed_ratio: float = 1.0
    ) -> float:
        """
        估算TTS生成音频的时长

        Args:
            text: 文本内容
            language: 语言
            speed_ratio: 语速比例

        Returns:
            估算时长（秒）
        """
        if not text or not text.strip():
            return 0.0

        # 中文字符和英文单词的估算
        if language == "zh":
            # 中文：每个字约0.25-0.3秒
            char_count = len([c for c in text if not c.isspace()])
            base_duration = char_count * 0.28
        else:
            # 英文：每个词约0.3-0.4秒
            word_count = len(text.split())
            base_duration = word_count * 0.35

        # 应用语速比例
        estimated = base_duration / speed_ratio

        return max(estimated, 0.5)  # 最少0.5秒

    def _calculate_speed_ratio(
        self, text: str, target_duration: float, language: str = "zh"
    ) -> float:
        """
        计算需要的语速比例

        Args:
            text: 文本内容
            target_duration: 目标时长
            language: 语言

        Returns:
            语速比例
        """
        if target_duration <= 0:
            return 1.0

        # 估算正常语速下的时长
        normal_duration = self._estimate_duration(text, language, speed_ratio=1.0)

        if normal_duration <= 0:
            return 1.0

        # 计算需要的语速比例
        speed_ratio = normal_duration / target_duration

        # 限制在合理范围内
        speed_ratio = np.clip(
            speed_ratio,
            self.config.min_speed_ratio,
            self.config.max_speed_ratio,
        )

        return speed_ratio

    def _synthesize_with_duration_control(
        self,
        task: SubtitleTTSTask,
        language: str = "zh",
        max_retries: int = 3,
    ) -> bool:
        """
        语音合成 - 保持自然语速和完整情绪音色

        Args:
            task: TTS任务
            language: 语言
            max_retries: 最大重试次数

        Returns:
            是否成功
        """
        if not task.chinese_text or not task.chinese_text.strip():
            # 生成静音
            samples = int(task.target_duration * self.config.default_sample_rate)
            silence = np.zeros(samples, dtype=np.float32)
            sf.write(str(task.output_path), silence, self.config.default_sample_rate)
            task.generated_duration = task.target_duration
            task.success = True
            return True

        # 准备生成参数 - 使用原始参考音频，保持自然语速
        ref_audio = task.reference_audio_path
        if not ref_audio or not ref_audio.exists():
            ref_audio = None

        # 不使用语速调节，保持speed_ratio=1.0（自然语速）
        task.speed_ratio = 1.0

        for attempt in range(max_retries):
            try:
                # 直接使用原始文本，不添加语速控制标记
                text_to_synthesize = task.chinese_text

                # 生成音频 - 不调节语速，保持自然音色
                self.tts.synthesize(
                    text=text_to_synthesize,
                    reference_audio=ref_audio,
                    output_path=task.output_path,
                    language=language,
                    speed_ratio=1.0,  # 自然语速
                )

                # 检查生成时长
                generated_audio, sr = sf.read(str(task.output_path), dtype="float32")
                task.generated_duration = len(generated_audio) / sr

                task.success = True
                return True

            except Exception as e:
                print(f"  尝试 {attempt + 1}/{max_retries} 失败: {e}")
                if attempt < max_retries - 1:
                    clear_gpu_memory()
                else:
                    task.error_message = str(e)
                    # 生成静音作为后备
                    samples = int(task.target_duration * self.config.default_sample_rate)
                    silence = np.zeros(samples, dtype=np.float32)
                    sf.write(str(task.output_path), silence, self.config.default_sample_rate)
                    task.generated_duration = task.target_duration
                    task.success = False

        return task.success

    def _apply_speed_control(self, text: str, speed_ratio: float) -> str:
        """
        应用语速控制（通过文本标记）- 当前禁用，保持自然语速

        Args:
            text: 原始文本
            speed_ratio: 语速比例

        Returns:
            原始文本（不添加语速标记）
        """
        # 禁用语速控制，直接返回原始文本
        # 保持克隆后的完整情绪音色和自然语速
        return text

    def process_tasks(
        self,
        tasks: List[SubtitleTTSTask],
        language: str = "zh",
        show_progress: bool = True,
    ) -> List[SubtitleTTSTask]:
        """
        处理所有TTS任务 - 保持自然语速和完整情绪音色

        Args:
            tasks: 任务列表
            language: 语言
            show_progress: 是否显示进度

        Returns:
            处理后的任务列表
        """
        total = len(tasks)
        success_count = 0

        print(f"\n开始处理 {total} 个TTS任务...")
        print(f"设备: {self.tts.device}")
        print(f"音色克隆: {'启用' if self.config.use_voice_clone else '禁用'}")
        print(f"语速调节: 禁用（保持自然语速）")

        for i, task in enumerate(tasks):
            if show_progress:
                print(f"\n[{i+1}/{total}] 片段 {task.index}")
                print(f"  中文: {task.chinese_text[:50]}...")
                print(f"  参考时长: {task.target_duration:.2f}s")

            # 处理任务
            success = self._synthesize_with_duration_control(task, language)

            if success:
                success_count += 1
                if show_progress:
                    print(f"  ✓ 生成时长: {task.generated_duration:.2f}s (自然语速)")
            else:
                if show_progress:
                    print(f"  ✗ 失败: {task.error_message}")

            # 定期清理显存
            if self.tts.device == "cuda" and (i + 1) % 10 == 0:
                clear_gpu_memory()

        # 统计结果
        print(f"\n{'=' * 60}")
        print(f"TTS处理完成: {success_count}/{total}")

        total_target = sum(t.target_duration for t in tasks)
        total_generated = sum(t.generated_duration for t in tasks)

        print(f"参考总时长: {total_target:.2f}s")
        print(f"生成总时长: {total_generated:.2f}s")
        print(f"语速: 自然语速（保持克隆音色）")

        return tasks

    def merge_task_audio(
        self,
        tasks: List[SubtitleTTSTask],
        output_path: Union[str, Path],
        crossfade_ms: int = 10,
    ) -> Path:
        """
        合并所有任务的音频

        Args:
            tasks: 任务列表
            output_path: 输出路径
            crossfade_ms: 交叉淡入淡出时长（毫秒）

        Returns:
            合并后的音频路径
        """
        output_path = Path(output_path)
        print(f"\n合并 {len(tasks)} 个音频片段...")

        # 按时间顺序排序
        sorted_tasks = sorted(tasks, key=lambda t: t.index)

        # 使用pydub合并
        combined = None
        sample_rate = None

        for i, task in enumerate(sorted_tasks):
            if not task.output_path or not task.output_path.exists():
                print(f"  ✗ 跳过片段{i}: 文件不存在")
                continue

            try:
                segment = PydubAudioSegment.from_file(str(task.output_path))

                if combined is None:
                    combined = segment
                    sample_rate = segment.frame_rate
                else:
                    # 添加极短的交叉淡入淡出
                    if crossfade_ms > 0 and len(combined) > crossfade_ms and len(segment) > crossfade_ms:
                        combined = combined.append(segment, crossfade=crossfade_ms)
                    else:
                        combined += segment

            except Exception as e:
                print(f"  ✗ 片段{i}加载失败: {e}")

        if combined is None:
            raise ValueError("没有可合并的音频")

        # 导出
        combined.export(str(output_path), format="wav")

        # 验证时长
        final_audio, sr = sf.read(str(output_path), dtype="float32")
        final_duration = len(final_audio) / sr

        print(f"✓ 合并完成: {output_path}")
        print(f"  总时长: {final_duration:.2f}s")

        return output_path


def create_aligned_tts(
    english_srt_path: Union[str, Path],
    chinese_srt_path: Union[str, Path],
    original_audio_path: Union[str, Path],
    output_dir: Union[str, Path],
    final_output_path: Union[str, Path],
    device: Optional[str] = None,
    language: str = "zh",
) -> Path:
    """
    创建与英文字幕时长对齐的中文TTS

    这是主入口函数，封装完整流程：
    1. 准备任务（提取参考音频）
    2. 生成带时长控制的TTS
    3. 合并音频

    Args:
        english_srt_path: 英文字幕路径
        chinese_srt_path: 中文字幕路径
        original_audio_path: 原始音频路径
        output_dir: 输出目录
        final_output_path: 最终合并音频路径
        device: 计算设备
        language: 语言

    Returns:
        最终音频路径
    """
    print("\n" + "=" * 60)
    print("字幕对齐TTS生成")
    print("=" * 60)

    # 创建引擎
    config = TTSConfig(
        speed_adjustment=True,
        enable_stretching=True,
        use_voice_clone=True,
    )
    engine = SubtitleTTSEngine(config=config, device=device)

    # 准备目录
    output_dir = Path(output_dir)
    ref_dir = output_dir / "reference_segments"
    tts_dir = output_dir / "tts_segments"

    # 准备任务
    tasks = engine.prepare_tasks(
        english_srt_path=english_srt_path,
        chinese_srt_path=chinese_srt_path,
        original_audio_path=original_audio_path,
        output_dir=tts_dir,
        reference_audio_dir=ref_dir,
    )

    if not tasks:
        raise ValueError("没有可处理的任务")

    # 处理任务
    processed_tasks = engine.process_tasks(tasks, language=language)

    # 合并音频
    final_path = engine.merge_task_audio(processed_tasks, final_output_path)

    print(f"\n✓ 字幕对齐TTS生成完成: {final_path}")

    return final_path
