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
