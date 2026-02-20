#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR语音识别模块 - 集成Whisper
支持多种模型和自动语言检测
支持并行处理提升效率
"""

import sys
from pathlib import Path
from typing import List, Optional, Union, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing

# 添加asr目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "asr"))

import whisper
import torch

from .splitter import AudioSegment

# 尝试导入性能配置
try:
    from .performance_config import get_parallel_config, get_performance_config
    HAS_PERF_CONFIG = True
except ImportError:
    HAS_PERF_CONFIG = False


@dataclass
class ASRResult:
    """ASR识别结果"""

    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0


class WhisperASR:
    """Whisper ASR识别器"""

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        language: Optional[str] = None,
    ):
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.model = None

    def load_model(self):
        """加载Whisper模型"""
        if self.model is None:
            print(f"加载Whisper模型: {self.model_size} ({self.device})")
            self.model = whisper.load_model(self.model_size, device=self.device)
            print("✓ 模型加载成功")

    def transcribe(
        self, audio_path: Union[str, Path], task: str = "transcribe"
    ) -> List[ASRResult]:
        """
        识别音频文件

        Args:
            audio_path: 音频文件路径
            task: "transcribe"(转录) 或 "translate"(翻译成英文)

        Returns:
            ASR结果列表
        """
        self.load_model()

        audio_path = Path(audio_path)
        print(f"\n识别音频: {audio_path.name}")

        result = self.model.transcribe(
            str(audio_path), language=self.language, task=task, verbose=False
        )

        results = []
        for segment in result["segments"]:
            asr_result = ASRResult(
                text=segment["text"].strip(),
                start_time=segment["start"],
                end_time=segment["end"],
                confidence=segment.get("avg_logprob", 0.0),
            )
            results.append(asr_result)

        return results

    def transcribe_segments(
        self, segments: List[AudioSegment], task: str = "transcribe",
        max_workers: Optional[int] = None,
        use_parallel: bool = True,
    ) -> List[AudioSegment]:
        """
        批量识别音频片段（支持并行处理）

        Args:
            segments: 音频片段列表
            task: 任务类型
            max_workers: 最大并行工作数（None则自动配置）
            use_parallel: 是否使用并行处理

        Returns:
            填充了text的片段列表
        """
        self.load_model()

        total = len(segments)
        print(f"\n开始识别 {total} 个片段...")

        # 自动配置并行参数
        if max_workers is None and HAS_PERF_CONFIG:
            max_workers = get_parallel_config().asr_parallel_segments
        elif max_workers is None:
            max_workers = min(multiprocessing.cpu_count() // 2, 4)

        # 检查是否启用并行
        if not use_parallel or max_workers <= 1:
            return self._transcribe_segments_sequential(segments, task)

        print(f"并行处理: {max_workers} 个工作进程")

        # 使用线程池并行处理（Whisper模型加载后可以多线程推理）
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for i, segment in enumerate(segments):
                if not segment.audio_path or not segment.audio_path.exists():
                    segment.text = ""
                    continue

                future = executor.submit(
                    self._transcribe_single_segment,
                    segment=segment,
                    task=task,
                    index=i,
                )
                futures[future] = i

            # 收集结果
            completed = 0
            for future in as_completed(futures):
                i = futures[future]
                try:
                    idx, text, success = future.result()
                    segments[idx].text = text
                    completed += 1

                    if completed % 5 == 0 or completed == total:
                        print(f"  进度: {completed}/{total} ({completed * 100 // total}%)")

                except Exception as e:
                    i = futures[future]
                    print(f"  ✗ 片段{i}: 识别失败 - {e}")
                    segments[i].text = ""

        return segments

    def _transcribe_single_segment(
        self,
        segment: AudioSegment,
        task: str,
        index: int,
    ) -> tuple:
        """
        单个片段识别（内部方法，用于并行）

        Returns:
            (index, text, success)
        """
        try:
            result = self.model.transcribe(
                str(segment.audio_path),
                language=self.language,
                task=task,
                verbose=False
            )

            texts = [seg["text"].strip() for seg in result["segments"]]
            text = " ".join(texts)

            if text:
                return (index, text, True)
            else:
                return (index, "", False)

        except Exception as e:
            print(f"  ✗ 片段{index}: {e}")
            return (index, "", False)

    def _transcribe_segments_sequential(
        self, segments: List[AudioSegment], task: str = "transcribe"
    ) -> List[AudioSegment]:
        """顺序识别音频片段（备用方法）"""
        total = len(segments)

        for i, segment in enumerate(segments):
            if not segment.audio_path or not segment.audio_path.exists():
                print(f"  ✗ 片段{i}: 音频文件不存在")
                continue

            try:
                results = self.transcribe(segment.audio_path, task)
                if results:
                    # 合并多个片段的文本
                    segment.text = " ".join([r.text for r in results])
                    print(f"  ✓ 片段{i}: {segment.text[:50]}...")
                else:
                    segment.text = ""
                    print(f"  ⚠ 片段{i}: 无识别结果")
            except Exception as e:
                print(f"  ✗ 片段{i}: 识别失败 - {e}")
                segment.text = ""

        return segments

    def transcribe_with_segments(
        self,
        audio_path: Union[str, Path],
        segments: List[AudioSegment],
        task: str = "transcribe",
    ) -> List[AudioSegment]:
        """
        对整个音频识别，然后匹配到各个片段
        适用于长音频，更准确
        """
        self.load_model()

        audio_path = Path(audio_path)
        print(f"\n识别完整音频: {audio_path.name}")

        result = self.model.transcribe(
            str(audio_path), language=self.language, task=task, verbose=False
        )

        # 将识别结果分配到各个片段
        for segment in segments:
            segment_texts = []
            for asr_seg in result["segments"]:
                asr_start = asr_seg["start"]
                asr_end = asr_seg["end"]

                # 检查是否与当前片段有重叠
                if asr_start < segment.end_time and asr_end > segment.start_time:
                    segment_texts.append(asr_seg["text"].strip())

            segment.text = " ".join(segment_texts)
            print(f"  ✓ 片段{segment.index}: {segment.text[:50]}...")

        return segments

    def save_srt(
        self,
        segments: List[AudioSegment],
        output_path: Union[str, Path],
        text_field: str = "text",
    ):
        """保存为SRT字幕文件"""
        output_path = Path(output_path)

        with open(output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, 1):
                text = getattr(segment, text_field, segment.text)
                if not text:
                    continue

                start = self._format_time(segment.start_time)
                end = self._format_time(segment.end_time)

                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")

        print(f"✓ SRT字幕已保存: {output_path}")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间为SRT格式 HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
