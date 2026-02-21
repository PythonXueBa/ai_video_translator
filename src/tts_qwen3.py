#!/usr/bin/env python3
"""
Qwen3-TTS 语音合成模块
支持语音克隆和情感控制
优化显存管理和错误处理
"""

import gc
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Union, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings

warnings.filterwarnings("ignore")

# 全局模型缓存（避免重复加载）
_MODEL_CACHE = {}
_MODEL_LOCK = threading.Lock()

# 尝试导入性能配置
try:
    from .performance_config import get_parallel_config, get_performance_config
    HAS_PERF_CONFIG = True
except ImportError:
    HAS_PERF_CONFIG = False


def get_gpu_memory_info() -> dict:
    """获取GPU显存信息"""
    if not torch.cuda.is_available():
        return {"total": 0, "used": 0, "free": 0, "available": 0}

    try:
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        free = total - reserved

        return {
            "total": total,
            "used": allocated,
            "reserved": reserved,
            "free": free,
            "available": free
        }
    except Exception:
        return {"total": 0, "used": 0, "free": 0, "available": 0}


def clear_gpu_memory():
    """清理GPU显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def check_gpu_memory_available(required_gb: float = 2.0) -> bool:
    """检查GPU显存是否足够"""
    info = get_gpu_memory_info()
    return info["free"] >= required_gb


class Qwen3TTS:
    """Qwen3-TTS语音合成器 - 优化版"""

    # 支持的模型列表
    SUPPORTED_MODELS = {
        "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    }

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
    ):
        self.model_name = model_name
        self.device = self._validate_device(device)
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self.model = None
        self.sample_rate = 24000
        self._model_loaded = False

    def _validate_device(self, device: str) -> str:
        """验证并返回有效设备"""
        if device == "cuda" and not torch.cuda.is_available():
            print("警告: CUDA不可用，切换到CPU")
            return "cpu"
        return device

    def _check_memory_before_load(self) -> bool:
        """加载前检查显存"""
        if self.device != "cuda":
            return True

        info = get_gpu_memory_info()
        # 1.7B模型约需要4-6GB显存
        required = 6.0 if "1.7B" in self.model_name else 3.0

        if info["free"] < required:
            print(f"警告: 显存不足 (需要 {required}GB, 可用 {info['free']:.1f}GB)")
            # 尝试清理
            clear_gpu_memory()
            info = get_gpu_memory_info()

            if info["free"] < required:
                print("显存仍不足，尝试使用CPU...")
                self.device = "cpu"
                self.dtype = torch.float32
                return True

        return True

    def load_model(self):
        """加载Qwen3-TTS模型"""
        if self._model_loaded and self.model is not None:
            return

        # 检查显存
        self._check_memory_before_load()

        # 尝试从缓存获取
        cache_key = f"{self.model_name}_{self.device}"
        with _MODEL_LOCK:
            if cache_key in _MODEL_CACHE:
                self.model = _MODEL_CACHE[cache_key]
                self._model_loaded = True
                print(f"✓ 从缓存加载Qwen3-TTS模型")
                return

        # 尝试不同的导入方式
        Qwen3TTSModel = None

        # 方式1: qwen_tts
        try:
            from qwen_tts import Qwen3TTSModel as Model
            Qwen3TTSModel = Model
            print("使用 qwen_tts 库")
        except ImportError:
            pass

        # 方式2: qwen-tts (pip包名)
        if Qwen3TTSModel is None:
            try:
                from qwen_tts.api import Qwen3TTSModel as Model
                Qwen3TTSModel = Model
                print("使用 qwen_tts.api 库")
            except ImportError:
                pass

        # 方式3: 尝试其他可能的导入
        if Qwen3TTSModel is None:
            try:
                import qwen_tts
                if hasattr(qwen_tts, 'Qwen3TTSModel'):
                    Qwen3TTSModel = qwen_tts.Qwen3TTSModel
                    print("使用 qwen_tts.Qwen3TTSModel")
            except ImportError:
                pass

        if Qwen3TTSModel is None:
            print("=" * 60)
            print("错误: 未找到Qwen3-TTS库")
            print("=" * 60)
            print("请安装:")
            print("  pip install qwen-tts")
            print("或")
            print("  pip install git+https://github.com/QwenLM/Qwen3-TTS.git")
            print("=" * 60)
            raise ImportError("未安装qwen-tts库")

        print(f"加载Qwen3-TTS模型: {self.model_name}")
        print(f"设备: {self.device}, 数据类型: {self.dtype}")
        print("首次下载可能需要几分钟...")

        try:
            if self.device == "cuda":
                clear_gpu_memory()

            # 加载模型
            load_kwargs = {
                "pretrained_model_name_or_path": self.model_name,
                "device_map": self.device,
            }

            # 只在支持的情况下使用dtype
            try:
                load_kwargs["dtype"] = self.dtype
            except TypeError:
                pass

            # 尝试不同的加载方式
            try:
                self.model = Qwen3TTSModel.from_pretrained(**load_kwargs)
            except TypeError:
                # 可能不支持某些参数
                self.model = Qwen3TTSModel.from_pretrained(
                    self.model_name,
                    device_map=self.device
                )

            if self.device == "cuda":
                torch.cuda.synchronize()

            # 缓存模型
            with _MODEL_LOCK:
                _MODEL_CACHE[cache_key] = self.model

            self._model_loaded = True
            print("✓ Qwen3-TTS模型加载成功")

            # 显示显存状态
            if self.device == "cuda":
                info = get_gpu_memory_info()
                print(f"显存使用: {info['used']:.1f}GB / {info['total']:.1f}GB")

        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            self._model_loaded = False
            raise

    def unload_model(self):
        """卸载模型释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
            self._model_loaded = False

            # 从缓存移除
            cache_key = f"{self.model_name}_{self.device}"
            with _MODEL_LOCK:
                if cache_key in _MODEL_CACHE:
                    del _MODEL_CACHE[cache_key]

            clear_gpu_memory()
            print("✓ TTS模型已卸载")

    def synthesize(
        self,
        text: str,
        reference_audio: Union[str, Path],
        output_path: Union[str, Path],
        language: str = "zh",
        ref_text: Optional[str] = None,
        speed_ratio: float = 1.0,
        emotion: Optional[str] = None,
    ) -> Path:
        """
        合成语音（带音色克隆和语速控制）

        Args:
            text: 要合成的文本
            reference_audio: 参考音频路径（用于克隆音色）
            output_path: 输出路径
            language: 语言代码 (zh/en/ja/ko/de/fr/ru/pt/es/it)
            ref_text: 参考音频的文本（可选）
            speed_ratio: 语速比例 (0.5-2.0, 1.0为正常语速)
            emotion: 情感标记 (可选)
        """
        self.load_model()
        output_path = Path(output_path)

        if not text or not text.strip():
            return self._generate_silence(output_path, 1.0)

        print(f"TTS合成: {text[:50]}...")
        if speed_ratio != 1.0:
            print(f"  语速比例: {speed_ratio:.2f}x")

        try:
            # 参考音频检查
            ref_audio_path = Path(reference_audio) if reference_audio else None
            if ref_audio_path and not ref_audio_path.exists():
                print(f"警告: 参考音频不存在: {reference_audio}")
                ref_audio_path = None

            # 构建生成参数
            generate_kwargs = {
                "text": text,
                "language": language.capitalize(),
            }

            # 如果有参考音频，使用音色克隆
            if ref_audio_path:
                generate_kwargs["ref_audio"] = str(ref_audio_path)

            # 添加语速控制参数（如果模型支持）
            if speed_ratio != 1.0:
                # 尝试不同的语速参数名
                for param_name in ["speed", "speed_ratio", "speaking_rate", "rate"]:
                    generate_kwargs[param_name] = speed_ratio

            # 添加情感参数（如果指定）
            if emotion:
                generate_kwargs["emotion"] = emotion

            # 尝试不同的生成方式
            wavs = None
            sr = self.sample_rate

            try:
                # 方式1: 使用generate_voice_clone（带音色克隆）
                if ref_audio_path and hasattr(self.model, 'generate_voice_clone'):
                    clone_kwargs = generate_kwargs.copy()
                    clone_kwargs["x_vector_only_mode"] = True
                    wavs, sr = self.model.generate_voice_clone(**clone_kwargs)
            except (TypeError, AttributeError, Exception) as e:
                pass

            if wavs is None:
                try:
                    # 方式2: 使用标准generate
                    if hasattr(self.model, 'generate'):
                        wavs, sr = self.model.generate(**generate_kwargs)
                except Exception as e:
                    pass

            if wavs is None:
                try:
                    # 方式3: 最简调用
                    wavs, sr = self.model.generate(text=text)
                except Exception as e:
                    raise RuntimeError(f"所有生成方式都失败: {e}")

            # 保存音频
            if isinstance(wavs, (list, tuple)):
                audio_data = wavs[0]
            else:
                audio_data = wavs

            # 确保音频数据是numpy数组
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()

            sf.write(str(output_path), audio_data, sr)

            # 清理
            del wavs
            if self.device == "cuda":
                torch.cuda.empty_cache()

            print(f"✓ 合成完成: {output_path.name}")
            return output_path

        except Exception as e:
            print(f"✗ 合成失败: {e}")
            # 根据文本长度估算静音时长
            estimated_duration = max(1.0, len(text) * 0.15)
            if speed_ratio > 0:
                estimated_duration /= speed_ratio
            return self._generate_silence(output_path, estimated_duration)

    def _generate_silence(self, output_path: Path, duration: float) -> Path:
        """生成静音文件"""
        samples = int(duration * self.sample_rate)
        silence = np.zeros(samples, dtype=np.float32)
        sf.write(str(output_path), silence, self.sample_rate)
        return output_path

    def synthesize_with_emotion(
        self,
        text: str,
        reference_audio: Union[str, Path],
        output_path: Union[str, Path],
        language: str = "zh",
        emotion: str = "自然",
        speaking_rate: str = "正常",
        speed_ratio: float = 1.0,
    ) -> Path:
        """
        带情感和语速控制的语音合成

        Args:
            text: 要合成的文本
            reference_audio: 参考音频路径
            output_path: 输出路径
            language: 语言
            emotion: 情感 (自然/高兴/悲伤/愤怒/惊讶等)
            speaking_rate: 语速描述 (慢速/正常/快速)
            speed_ratio: 语速比例 (0.5-2.0)
        """
        # 将描述性语速转换为比例
        rate_map = {
            "慢速": 0.8,
            "正常": 1.0,
            "快速": 1.2,
        }

        if speaking_rate in rate_map and speed_ratio == 1.0:
            speed_ratio = rate_map[speaking_rate]

        return self.synthesize(
            text=text,
            reference_audio=reference_audio,
            output_path=output_path,
            language=language,
            speed_ratio=speed_ratio,
            emotion=emotion if emotion != "自然" else None,
        )

    def _synthesize_single_safe(
        self,
        text: str,
        reference_audio: Union[str, Path],
        output_path: Union[str, Path],
        language: str = "zh",
        index: int = 0,
    ) -> tuple:
        """
        安全的单个语音合成（带错误处理）
        """
        output_path = Path(output_path)

        if not text or not text.strip():
            self._generate_silence(output_path, 2.0)
            return (index, output_path, True)

        try:
            # 检查显存
            if self.device == "cuda":
                if not check_gpu_memory_available(1.0):
                    print(f"  警告: 片段{index}显存不足，清理中...")
                    clear_gpu_memory()

            self.synthesize(
                text=text,
                reference_audio=reference_audio,
                output_path=output_path,
                language=language,
            )
            return (index, output_path, True)
        except Exception as e:
            print(f"  ✗ 片段{index}: {e}")
            self._generate_silence(output_path, max(1.0, len(text) * 0.15))
            return (index, output_path, False)

    def batch_synthesize(
        self,
        texts: List[str],
        reference_audio: Union[str, Path],
        output_dir: Union[str, Path],
        language: str = "zh",
        prefix: str = "tts",
        max_workers: Optional[int] = None,
        use_parallel: bool = True,
    ) -> List[Path]:
        """
        批量合成语音

        重要：GPU模式下强制使用串行处理，避免显存冲突
        """
        self.load_model()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        total = len(texts)
        print(f"\n批量合成 {total} 个语音片段...")
        print(f"参考音频: {reference_audio}")
        print(f"设备: {self.device}")

        # GPU模式强制串行处理（多线程会导致显存冲突）
        if self.device == "cuda":
            print("GPU模式: 使用串行处理避免显存冲突")
            use_parallel = False
            max_workers = 1

        # 自动配置并行参数
        if max_workers is None and HAS_PERF_CONFIG:
            max_workers = get_parallel_config().tts_max_workers
        elif max_workers is None:
            max_workers = 1

        # CPU模式也限制并行数
        if max_workers > 2:
            max_workers = 2

        # 检查是否启用并行
        if not use_parallel or max_workers <= 1:
            return self._batch_synthesize_sequential(
                texts, reference_audio, output_dir, language, prefix
            )

        print(f"CPU并行处理: {max_workers} 个工作线程")

        # 准备输出路径
        output_paths = [None] * total
        success_count = 0

        # 使用线程池并行处理（仅CPU模式）
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for i, text in enumerate(texts):
                output_path = output_dir / f"{prefix}_{i:04d}.wav"
                future = executor.submit(
                    self._synthesize_single_safe,
                    text=text,
                    reference_audio=reference_audio,
                    output_path=output_path,
                    language=language,
                    index=i,
                )
                futures[future] = i

            # 收集结果
            for future in as_completed(futures):
                try:
                    idx, path, success = future.result()
                    output_paths[idx] = path
                    if success:
                        success_count += 1

                    # 显示进度
                    completed = sum(1 for p in output_paths if p is not None)
                    if completed % 5 == 0 or completed == total:
                        print(f"  进度: {completed}/{total} ({completed * 100 // total}%)")

                except Exception as e:
                    idx = futures[future]
                    print(f"  ✗ 片段{idx}处理异常: {e}")
                    output_path = output_dir / f"{prefix}_{idx:04d}.wav"
                    self._generate_silence(output_path, 2.0)
                    output_paths[idx] = output_path

        # 确保所有路径都有值
        for i, path in enumerate(output_paths):
            if path is None:
                output_paths[i] = output_dir / f"{prefix}_{i:04d}.wav"

        print(f"✓ 批量合成完成: {success_count}/{total}")
        return output_paths

    def _batch_synthesize_sequential(
        self,
        texts: List[str],
        reference_audio: Union[str, Path],
        output_dir: Union[str, Path],
        language: str = "zh",
        prefix: str = "tts",
    ) -> List[Path]:
        """顺序批量合成"""
        output_dir = Path(output_dir)
        output_paths = []

        for i, text in enumerate(texts):
            output_path = output_dir / f"{prefix}_{i:04d}.wav"

            if not text or not text.strip():
                self._generate_silence(output_path, 2.0)
                output_paths.append(output_path)
                continue

            try:
                self.synthesize(
                    text=text,
                    reference_audio=reference_audio,
                    output_path=output_path,
                    language=language,
                )
                output_paths.append(output_path)

                # 定期清理显存
                if self.device == "cuda" and (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()

                # 显示进度
                if i < 3 or (i + 1) % 10 == 0:
                    print(f"  ✓ [{i+1}/{len(texts)}] {text[:40]}...")

            except Exception as e:
                print(f"  ✗ 片段{i}: {e}")
                self._generate_silence(output_path, max(1.0, len(text) * 0.15))
                output_paths.append(output_path)

        print(f"✓ 批量合成完成: {len(output_paths)}/{len(texts)}")
        return output_paths


def get_qwen3_tts(
    model_size: str = "1.7B",
    device: Optional[str] = None,
) -> Qwen3TTS:
    """
    获取Qwen3-TTS实例

    Args:
        model_size: 模型大小 (0.6B 或 1.7B)
        device: 计算设备 (cuda/cpu)

    Returns:
        Qwen3TTS实例
    """
    # 自动配置设备
    if device is None:
        if HAS_PERF_CONFIG:
            device = get_parallel_config().device
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # 检查显存决定是否使用小模型
    if device == "cuda":
        info = get_gpu_memory_info()
        if info["free"] < 6.0 and model_size == "1.7B":
            print(f"显存不足({info['free']:.1f}GB)，自动切换到0.6B模型")
            model_size = "0.6B"
        if info["free"] < 3.0:
            print("显存严重不足，切换到CPU模式")
            device = "cpu"

    model_name = Qwen3TTS.SUPPORTED_MODELS.get(model_size, Qwen3TTS.SUPPORTED_MODELS["1.7B"])

    # 根据设备选择数据类型
    dtype = torch.float16 if device == "cuda" else torch.float32

    return Qwen3TTS(model_name=model_name, device=device, dtype=dtype)
