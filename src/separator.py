#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人声分离模块 - 合并背景音轨
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

from .analyzer import MediaAnalyzer
from .config import STEM_MAPPING
from .logger import get_logger, log_execution

logger = get_logger(__name__)


class VocalSeparator:
    """人声分离器 - 输出人声和合并的背景音"""

    def __init__(self, model: str = "htdemucs", device: Optional[str] = None):
        self.model_name = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.source_names: List[str] = []

    def _load_model(self):
        """加载 Demucs 模型"""
        if self.model is None:
            logger.info(f"加载模型: {self.model_name}")
            model = get_model(self.model_name)
            if model is None:
                logger.error(f"无法加载模型: {self.model_name}")
                raise RuntimeError(f"无法加载模型: {self.model_name}")
            self.model = model
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"模型加载成功")
            # 获取音源名称
            if hasattr(self.model, "sources"):
                self.source_names = list(self.model.sources)
                logger.debug(f"音源: {self.source_names}")
            else:
                # 默认音源
                self.source_names = ["drums", "bass", "other", "vocals"]
                logger.debug(f"音源: {self.source_names} (默认)")

    def _load_audio(self, audio_path: Path) -> tuple:
        """加载音频文件"""
        logger.info(f"加载音频: {audio_path.name}")
        audio_data, sr = sf.read(str(audio_path), dtype="float32")

        # 转换为 [channels, time] 格式
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)
        else:
            audio_data = audio_data.T

        # 转换为 torch tensor
        wav = torch.from_numpy(audio_data)

        logger.info(f"音频加载成功: {wav.shape[1] / sr:.2f}s @ {sr}Hz")
        return wav, sr

    def _apply_model(self, wav: torch.Tensor) -> torch.Tensor:
        """应用分离模型"""
        if self.model is None:
            logger.error("模型未加载")
            raise RuntimeError("模型未加载")

        logger.info("分离音轨中...")

        # 添加 batch 维度
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)

        with torch.no_grad():
            sources = apply_model(self.model, wav, device=self.device, progress=True)

        logger.info("分离完成")
        return sources

    def _merge_background(
        self, sources: torch.Tensor, source_names: List[str]
    ) -> torch.Tensor:
        """
        合并背景音轨

        Args:
            sources: 分离后的音轨 [batch, sources, channels, time]
            source_names: 音轨名称列表

        Returns:
            合并后的背景音 [channels, time]
        """
        # 找到背景音轨的索引
        background_indices = []
        vocals_index = None

        for i, name in enumerate(source_names):
            if name in STEM_MAPPING["background"]:
                background_indices.append(i)
            elif name in STEM_MAPPING["vocals"]:
                vocals_index = i

        # 合并背景音
        background = torch.zeros_like(sources[0, 0])
        for idx in background_indices:
            background += sources[0, idx]

        return background

    def separate(
        self,
        audio_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """
        分离音频为人声和背景

        Returns:
            {'vocals': Path, 'background': Path}
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        # 分析输入音频
        input_info = MediaAnalyzer.analyze_audio(audio_path)
        logger.info(f"输入音频: {audio_path.name}, 时长: {input_info.duration:.2f}s")

        # 设置输出目录
        if output_dir is None:
            out_path = (
                audio_path.parent / "separated" / self.model_name / audio_path.stem
            )
        else:
            out_path = Path(output_dir) / self.model_name / audio_path.stem

        out_path.mkdir(parents=True, exist_ok=True)
        output_dir = out_path
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"使用设备: {self.device}")

        # 加载模型和音频
        self._load_model()
        wav, sr = self._load_audio(audio_path)

        # 应用模型
        sources = self._apply_model(wav)

        # 保存结果
        logger.info("保存分离结果...")
        result = {}

        # 确保有音源名称
        if not self.source_names:
            self.source_names = ["drums", "bass", "other", "vocals"]

        # 保存人声
        vocals_idx = self.source_names.index("vocals")
        vocals = sources[0, vocals_idx].cpu().numpy()
        vocals_path = output_dir / "vocals.wav"
        sf.write(str(vocals_path), vocals.T, sr, subtype="PCM_16")
        result["vocals"] = vocals_path
        logger.info(f"人声保存: {vocals_path.name}")

        # 合并并保存背景音
        background = self._merge_background(sources, self.source_names)
        background = background.cpu().numpy()
        background_path = output_dir / "background.wav"
        sf.write(str(background_path), background.T, sr, subtype="PCM_16")
        result["background"] = background_path
        logger.info(f"背景音保存: {background_path.name}")

        return result

    def verify(self, original_path: Path, separated_files: Dict[str, Path]) -> bool:
        """验证分离结果"""
        logger.info("验证分离结果...")

        original_info = MediaAnalyzer.analyze_audio(original_path)
        logger.debug(f"原始音频: {original_info.duration:.2f}s")

        all_valid = True

        for name, path in separated_files.items():
            if not path.exists():
                logger.error(f"{name}: 文件不存在")
                all_valid = False
                continue

            info = MediaAnalyzer.analyze_audio(path)
            duration_diff = abs(original_info.duration - info.duration)

            if duration_diff > 1.0:
                logger.warning(f"{name}: 时长不匹配 (差异 {duration_diff:.2f}s)")
                all_valid = False
            else:
                logger.debug(f"{name}: {info.duration:.2f}s (差异 {duration_diff:.3f}s)")

        return all_valid
