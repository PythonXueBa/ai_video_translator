#!/usr/bin/env python3
"""
字幕处理模块 - 支持SRT格式读写和翻译
"""

import re
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class SubtitleEntry:
    """字幕条目"""

    index: int
    start_time: str  # SRT格式: HH:MM:SS,mmm
    end_time: str  # SRT格式: HH:MM:SS,mmm
    text: str  # 原文
    translated_text: str = ""  # 译文

    @property
    def start_seconds(self) -> float:
        """转换为秒"""
        return self._time_to_seconds(self.start_time)

    @property
    def end_seconds(self) -> float:
        """转换为秒"""
        return self._time_to_seconds(self.end_time)

    @staticmethod
    def _time_to_seconds(time_str: str) -> float:
        """SRT时间格式转秒"""
        time_str = time_str.replace(",", ".")
        parts = time_str.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    @staticmethod
    def seconds_to_time(seconds: float) -> str:
        """秒转SRT时间格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @property
    def duration(self) -> float:
        """获取字幕持续时间"""
        return self.end_seconds - self.start_seconds

    def adjust_timing(self, offset: float = 0.0, scale: float = 1.0):
        """调整时间轴

        Args:
            offset: 时间偏移（秒）
            scale: 时间缩放比例
        """
        self.start_time = self.seconds_to_time(
            self.start_seconds * scale + offset
        )
        self.end_time = self.seconds_to_time(
            self.end_seconds * scale + offset
        )


class SRTHandler:
    """SRT字幕处理器"""

    @staticmethod
    def parse(srt_path: Path) -> List[SubtitleEntry]:
        """解析SRT文件"""
        entries = []
        content = srt_path.read_text(encoding="utf-8")

        # 分割条目 (用空行分隔)
        blocks = re.split(r"\n\s*\n", content.strip())

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue

            try:
                # 第一行是序号
                index = int(lines[0].strip())

                # 第二行是时间
                time_line = lines[1].strip()
                match = re.match(
                    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
                    time_line,
                )
                if not match:
                    continue

                start_time = match.group(1)
                end_time = match.group(2)

                # 剩余行是文本
                text = "\n".join(lines[2:]).strip()

                entries.append(
                    SubtitleEntry(
                        index=index, start_time=start_time, end_time=end_time, text=text
                    )
                )
            except Exception as e:
                print(f"解析字幕条目失败: {e}")
                continue

        return entries

    @staticmethod
    def write(
        entries: List[SubtitleEntry], output_path: Path, use_translated: bool = False
    ):
        """写入SRT文件"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for i, entry in enumerate(entries, 1):
                # 使用译文或原文
                text = (
                    entry.translated_text
                    if use_translated and entry.translated_text
                    else entry.text
                )

                if not text:
                    continue

                f.write(f"{i}\n")
                f.write(f"{entry.start_time} --> {entry.end_time}\n")
                f.write(f"{text}\n\n")

        print(f"✓ SRT字幕已保存: {output_path}")

    @staticmethod
    def translate_srt(
        input_path: Path,
        output_path: Path,
        translator,
        source_lang: str = "en",
        target_lang: str = "zh",
    ):
        """翻译SRT文件"""
        print(f"\n翻译字幕: {input_path.name}")
        print(f"{source_lang} → {target_lang}")

        # 解析原文字幕
        entries = SRTHandler.parse(input_path)
        print(f"共 {len(entries)} 个字幕条目")

        # 翻译每个条目
        translator.load_model()

        translated_count = 0
        for i, entry in enumerate(entries):
            if not entry.text:
                continue

            try:
                entry.translated_text = translator.translate(entry.text)
                translated_count += 1

                if i < 3:
                    print(f"\n  原文: {entry.text[:60]}...")
                    print(f"  译文: {entry.translated_text[:60]}...")
            except Exception as e:
                print(f"  ✗ 条目{i}翻译失败: {e}")
                entry.translated_text = entry.text

        print(f"\n✓ 翻译完成: {translated_count}/{len(entries)}")

        # 写入译文SRT
        SRTHandler.write(entries, output_path, use_translated=True)

        return entries

    @staticmethod
    def merge_short_entries(
        entries: List[SubtitleEntry],
        min_duration: float = 1.5,
        max_gap: float = 0.5,
    ) -> List[SubtitleEntry]:
        """
        合并过短的字幕条目

        Args:
            entries: 原始字幕条目
            min_duration: 最小时长（秒）
            max_gap: 最大合并间隔（秒）

        Returns:
            合并后的字幕条目
        """
        if not entries:
            return entries

        merged = []
        current = entries[0]

        for i in range(1, len(entries)):
            next_entry = entries[i]

            # 计算当前条目的时长
            current_duration = current.duration

            # 计算与下一个条目的间隔
            gap = next_entry.start_seconds - current.end_seconds

            # 如果当前条目太短且与下一个条目间隔小，则合并
            if current_duration < min_duration and gap <= max_gap:
                # 合并文本
                current.text += " " + next_entry.text
                if current.translated_text and next_entry.translated_text:
                    current.translated_text += " " + next_entry.translated_text
                # 更新结束时间
                current.end_time = next_entry.end_time
            else:
                merged.append(current)
                current = next_entry

        # 添加最后一个条目
        merged.append(current)

        # 重新编号
        for i, entry in enumerate(merged, 1):
            entry.index = i

        return merged

    @staticmethod
    def split_long_entries(
        entries: List[SubtitleEntry],
        max_duration: float = 10.0,
        max_chars: int = 80,
    ) -> List[SubtitleEntry]:
        """
        分割过长的字幕条目

        Args:
            entries: 原始字幕条目
            max_duration: 最大时长（秒）
            max_chars: 最大字符数

        Returns:
            分割后的字幕条目
        """
        result = []

        for entry in entries:
            duration = entry.duration
            text = entry.text
            translated = entry.translated_text

            # 检查是否需要分割
            need_split = duration > max_duration or len(text) > max_chars

            if not need_split:
                result.append(entry)
                continue

            # 计算分割数
            num_splits = max(
                int(duration / max_duration) + 1,
                len(text) // max_chars + 1,
            )

            # 分割文本（按句子或字符数）
            split_texts = SRTHandler._split_text(text, num_splits)
            split_translated = SRTHandler._split_text(translated, num_splits) if translated else split_texts

            # 计算每个片段的时长
            total_duration = duration
            segment_duration = total_duration / len(split_texts)

            # 创建新的条目
            for i, (split_text, split_trans) in enumerate(zip(split_texts, split_translated)):
                start_sec = entry.start_seconds + i * segment_duration
                end_sec = min(start_sec + segment_duration, entry.end_seconds)

                new_entry = SubtitleEntry(
                    index=0,  # 稍后重新编号
                    start_time=SubtitleEntry.seconds_to_time(start_sec),
                    end_time=SubtitleEntry.seconds_to_time(end_sec),
                    text=split_text,
                    translated_text=split_trans,
                )
                result.append(new_entry)

        # 重新编号
        for i, entry in enumerate(result, 1):
            entry.index = i

        return result

    @staticmethod
    def _split_text(text: str, num_parts: int) -> List[str]:
        """将文本分割成多个部分"""
        if num_parts <= 1:
            return [text]

        # 尝试按句子分割
        sentences = re.split(r'([.!?。！？]+)', text)
        sentences = [s for s in sentences if s.strip()]

        if len(sentences) >= num_parts:
            # 按句子分割
            result = []
            part_size = len(sentences) // num_parts
            for i in range(num_parts):
                start = i * part_size
                end = start + part_size if i < num_parts - 1 else len(sentences)
                part = "".join(sentences[start:end]).strip()
                if part:
                    result.append(part)
            return result if result else [text]

        # 按字符数平均分割
        chars_per_part = len(text) // num_parts
        result = []
        for i in range(num_parts):
            start = i * chars_per_part
            end = start + chars_per_part if i < num_parts - 1 else len(text)
            part = text[start:end].strip()
            if part:
                result.append(part)

        return result if result else [text]
