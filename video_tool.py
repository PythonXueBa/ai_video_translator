#!/usr/bin/env python3
"""
音视频AI工具箱 - 统一入口
功能：视频转音频、人声分离、AI配音、音频合并、模块测试

特性:
- 自动检测系统硬件并优化并行处理参数
- 支持GPU加速（CUDA/Apple Silicon）
- 智能内存管理和模型选择
- 完善的显存监控和自动降级

用法:
    python video_tool.py [命令] [选项]

命令:
    convert     视频转音频
    test        测试所有模块
    separate    人声分离
    merge       合并音频
    replace     替换视频音频
    dub         AI配音(英文→中文)

示例:
    # 视频转音频
    python video_tool.py convert video.mp4 -o audio.mp3

    # 测试所有模块
    python video_tool.py test

    # 人声分离
    python video_tool.py separate video.mp4

    # 合并音频
    python video_tool.py merge -v vocals.wav -b background.wav

    # 替换视频音频
    python video_tool.py replace video.mp4 -a new_audio.wav

    # AI配音(完整流程)
    python video_tool.py dub video.mp4
"""

import argparse
import sys
import subprocess
import gc
import logging
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

# 优化CUDA内存分配
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 导入显存管理
try:
    from src.memory_manager import (
        GPUMemoryManager,
        clear_gpu_cache,
        get_free_gpu_memory,
        print_memory_status,
    )

    HAS_MEMORY_MANAGER = True
except ImportError:
    HAS_MEMORY_MANAGER = False

import torch


# 日志系统
def setup_logger(name: str = "video_tool", log_dir: str = "logs") -> logging.Logger:
    """配置日志系统"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"dub_{timestamp}.log"

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


logger = setup_logger()


def log_step(step: str, status: str, details: str = ""):
    """记录步骤日志"""
    msg = f"[{step}] {status}"
    if details:
        msg += f" | {details}"
    logger.info(msg)
    print(msg)


def log_time(start: float, step: str):
    """记录耗时"""
    elapsed = time.time() - start
    logger.info(f"[{step}] 耗时: {elapsed:.1f}秒")


def clear_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# 导入性能配置模块
from src.performance_config import (
    ParallelConfig,
    get_performance_config,
    get_parallel_config,
    print_system_info,
    print_config,
)


def _get_optimal_tts_workers(config: ParallelConfig, num_segments: int) -> int:
    """
    根据GPU显存和片段数量动态计算最优TTS并行数

    重要：GPU模式下强制返回1，避免显存冲突
    """
    # GPU模式强制串行
    if config.device == "cuda":
        # 再次检查显存
        if HAS_MEMORY_MANAGER:
            free_mem = get_free_gpu_memory()
            if free_mem < 3.0:
                print(f"  警告: 剩余显存 {free_mem:.1f}GB 不足，建议使用CPU")
        return 1

    # CPU模式可以根据片段数量适当并行
    if num_segments <= 10:
        return 1
    elif num_segments <= 50:
        return min(2, config.tts_max_workers)
    else:
        return min(2, config.tts_max_workers)


def cmd_convert(args):
    """视频转音频"""
    from src.config import OUTPUT_DIR
    from src.extractor import AudioExtractor

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ 文件不存在: {input_path}")
        return 1

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / f"{input_path.stem}.{args.format}"

    print(f"转换视频到音频:")
    print(f"  输入: {input_path}")
    print(f"  输出: {output_path}")
    print(f"  格式: {args.format}")

    try:
        import soundfile as sf
        import numpy as np

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ar",
            str(args.sample_rate),
            "-ac",
            str(args.channels),
        ]

        if args.format == "mp3":
            cmd.extend(["-c:a", "libmp3lame"])
            if args.quality:
                cmd.extend(["-b:a", args.quality])
        elif args.format == "m4a":
            cmd.extend(["-c:a", "aac"])
        elif args.format == "ogg":
            cmd.extend(["-c:a", "libvorbis"])
        else:
            cmd.extend(["-c:a", "pcm_s16le"])

        cmd.append(str(output_path))
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"✗ 转换失败: {result.stderr}")
            return 1

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ 转换完成: {output_path} ({size_mb:.2f} MB)")
        return 0

    except Exception as e:
        print(f"✗ 错误: {e}")
        return 1


def cmd_test(args):
    """测试所有模块"""
    print("=" * 60)
    print("AI配音系统 - 模块测试")
    print("=" * 60)

    # 显示系统信息和性能配置
    print("\n[系统信息]")
    print_system_info()
    print("\n[性能配置]")
    print_config()

    # 显示显存状态
    if HAS_MEMORY_MANAGER:
        print("\n[显存状态]")
        print_memory_status()

    def test_module(name, import_path):
        print(f"\n[测试] {name}")
        try:
            __import__(import_path, fromlist=[""])
            print(f"  ✓ {name}")
            return True
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            return False

    results = []
    results.append(test_module("FFmpeg", "src.analyzer"))
    results.append(test_module("音频提取", "src.extractor"))
    results.append(test_module("人声分离", "src.separator"))
    results.append(test_module("ASR识别", "src.asr_module"))
    results.append(test_module("翻译模块(M2M100)", "src.translator_m2m100"))
    results.append(test_module("TTS合成", "src.tts_qwen3"))
    results.append(test_module("字幕处理", "src.subtitle_handler"))
    results.append(test_module("音频合并", "src.merger"))
    results.append(test_module("视频处理", "src.video_processor"))
    results.append(test_module("性能配置", "src.performance_config"))
    results.append(test_module("显存管理", "src.memory_manager"))

    print("\n" + "=" * 60)
    passed = sum(results)
    print(f"测试结果: {passed}/{len(results)} 通过")

    if passed == len(results):
        print("✓ 所有模块可用!")
        return 0
    else:
        print(f"✗ {len(results) - passed} 个模块失败")
        return 1


def cmd_separate(args):
    """人声分离"""
    from src.extractor import AudioExtractor
    from src.separator import VocalSeparator
    from src.config import OUTPUT_DIR

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ 文件不存在: {input_path}")
        return 1

    output_dir = OUTPUT_DIR / f"{input_path.stem}_separated"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"人声分离: {input_path.name}")

    # 提取音频
    audio_path = output_dir / "audio.wav"
    if input_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        extractor = AudioExtractor()
        extractor.extract(input_path, audio_path)
    else:
        audio_path = input_path

    # 检查显存决定设备
    device = args.device or "cpu"
    if device == "cuda" and HAS_MEMORY_MANAGER:
        free_mem = get_free_gpu_memory()
        if free_mem < 3.0:
            print(f"显存不足 ({free_mem:.1f}GB)，使用CPU")
            device = "cpu"

    # 分离人声
    separator = VocalSeparator(device=device)
    separated = separator.separate(audio_path, output_dir / "separated")

    print(f"\n✓ 分离完成:")
    print(f"  人声: {separated['vocals']}")
    print(f"  背景: {separated['background']}")

    # 清理显存
    clear_memory()
    return 0


def cmd_merge(args):
    """合并音频"""
    from src.merger import AudioMerger
    from src.config import OUTPUT_DIR

    if not args.vocals or not args.background:
        print("✗ 需要指定人声和背景音频文件")
        return 1

    vocals_path = Path(args.vocals)
    background_path = Path(args.background)

    if not vocals_path.exists():
        print(f"✗ 人声文件不存在: {vocals_path}")
        return 1
    if not background_path.exists():
        print(f"✗ 背景文件不存在: {background_path}")
        return 1

    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / "final_dubbed_zh.wav"

    print(f"合并音频:")
    print(f"  人声: {vocals_path} (音量: {args.vocals_vol})")
    print(f"  背景: {background_path} (音量: {args.background_vol})")

    merger = AudioMerger()
    merger.merge_with_volume(
        vocals_path,
        background_path,
        output_path,
        vocals_volume=args.vocals_vol,
        background_volume=args.background_vol,
    )

    print(f"\n✓ 合并完成: {output_path}")
    return 0


def cmd_replace(args):
    """替换视频音频"""
    from src.video_processor import VideoProcessor
    from src.config import OUTPUT_DIR

    video_path = Path(args.input)
    audio_path = Path(args.audio)

    if not video_path.exists():
        print(f"✗ 视频文件不存在: {video_path}")
        return 1
    if not audio_path.exists():
        print(f"✗ 音频文件不存在: {audio_path}")
        return 1

    output_path = OUTPUT_DIR / f"{video_path.stem}_replaced{video_path.suffix}"

    print(f"替换视频音频:")
    print(f"  视频: {video_path}")
    print(f"  音频: {audio_path}")

    processor = VideoProcessor()
    processor.replace_audio(video_path, audio_path, output_path)

    print(f"\n✓ 替换完成: {output_path}")
    return 0


def cmd_asr(args):
    """ASR语音识别"""
    from src.asr_module import WhisperASR
    from src.subtitle_handler import SRTHandler, SubtitleEntry
    from src.config import OUTPUT_DIR

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ 文件不存在: {input_path}")
        return 1

    output_dir = OUTPUT_DIR / f"{input_path.stem}_asr"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查显存
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and HAS_MEMORY_MANAGER:
        free_mem = get_free_gpu_memory()
        if free_mem < 1.5:
            print(f"显存不足 ({free_mem:.1f}GB)，使用CPU")
            device = "cpu"

    print(f"ASR语音识别: {input_path.name}")
    print(f"  模型: {args.model}")
    print(f"  语言: {args.language}")
    print(f"  设备: {device}")

    asr = WhisperASR(model_size=args.model, device=device, language=args.language)
    asr.load_model()
    results = asr.transcribe(input_path)

    # 保存字幕
    subtitle_entries = []
    for i, result in enumerate(results):
        entry = SubtitleEntry(
            index=i + 1,
            start_time=SubtitleEntry.seconds_to_time(result.start_time),
            end_time=SubtitleEntry.seconds_to_time(result.end_time),
            text=result.text,
        )
        subtitle_entries.append(entry)

    srt_path = output_dir / f"{input_path.stem}.srt"
    SRTHandler.write(subtitle_entries, srt_path)

    # 保存文本
    txt_path = output_dir / f"{input_path.stem}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for entry in subtitle_entries:
            f.write(entry.text + "\n")

    print(f"\n✓ ASR完成: {len(results)} 个片段")
    print(f"  字幕: {srt_path}")
    print(f"  文本: {txt_path}")

    clear_memory()
    return 0


def cmd_translate(args):
    """文本/字幕翻译"""
    from src.translator_m2m100 import M2M100Translator
    from src.subtitle_handler import SRTHandler
    from src.config import OUTPUT_DIR

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ 文件不存在: {input_path}")
        return 1

    output_dir = OUTPUT_DIR / "translated"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查显存
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and HAS_MEMORY_MANAGER:
        free_mem = get_free_gpu_memory()
        if free_mem < 2.0:
            print(f"显存不足 ({free_mem:.1f}GB)，使用CPU")
            device = "cpu"

    print(f"翻译: {input_path.name}")
    print(f"  {args.source} -> {args.target}")
    print(f"  设备: {device}")
    print(f"  模型: M2M100")

    # 使用M2M100翻译器
    translator = M2M100Translator(
        source_language=args.source,
        target_language=args.target,
        device=device,
        model_size="418M",  # 使用418M模型，平衡速度和质量
    )
    translator.load_model()

    if input_path.suffix.lower() == ".srt":
        # 翻译SRT字幕
        entries = SRTHandler.parse(input_path)
        for entry in entries:
            if entry.text:
                entry.translated_text = translator.translate(entry.text)
        output_path = output_dir / f"{input_path.stem}_{args.target}.srt"
        SRTHandler.write(entries, output_path, use_translated=True)
    else:
        # 翻译文本文件
        with open(input_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        translated = [translator.translate(text) for text in texts]
        output_path = output_dir / f"{input_path.stem}_{args.target}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            for text in translated:
                f.write(text + "\n")

    print(f"\n✓ 翻译完成: {output_path}")

    clear_memory()
    return 0


def cmd_tts(args):
    """TTS语音合成"""
    from src.tts_qwen3 import get_qwen3_tts
    from src.config import OUTPUT_DIR

    input_path = Path(args.input)
    ref_audio = Path(args.reference) if args.reference else None

    if not input_path.exists():
        print(f"✗ 文件不存在: {input_path}")
        return 1

    if ref_audio and not ref_audio.exists():
        print(f"✗ 参考音频不存在: {ref_audio}")
        return 1

    output_dir = OUTPUT_DIR / "tts_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取文本
    if input_path.suffix.lower() == ".txt":
        with open(input_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    elif input_path.suffix.lower() == ".srt":
        from src.subtitle_handler import SRTHandler

        entries = SRTHandler.parse(input_path)
        texts = [e.text for e in entries if e.text]
    else:
        print("✗ 不支持的输入格式，请使用 .txt 或 .srt 文件")
        return 1

    # 检查显存
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and HAS_MEMORY_MANAGER:
        free_mem = get_free_gpu_memory()
        if free_mem < 2.0:
            print(f"显存不足 ({free_mem:.1f}GB)，使用CPU")
            device = "cpu"

    print(f"TTS语音合成: {input_path.name}")
    print(f"  文本数量: {len(texts)}")
    print(f"  语言: {args.language}")
    print(f"  设备: {device}")

    tts = get_qwen3_tts(model_size=args.model, device=device)
    tts.load_model()

    output_paths = tts.batch_synthesize(
        texts=texts,
        reference_audio=ref_audio,
        output_dir=output_dir,
        language=args.language,
        prefix="tts",
        max_workers=1,
        use_parallel=False,
    )

    # 合并音频
    import soundfile as sf
    import numpy as np

    all_audio = []
    for path in output_paths:
        if path.exists():
            audio, sr = sf.read(str(path), dtype="float32")
            all_audio.append(audio)

    if all_audio:
        merged = np.concatenate(all_audio, axis=0)
        merged_path = output_dir / "merged_tts.wav"
        sf.write(str(merged_path), merged, 24000)
        print(f"\n✓ TTS完成: {len(texts)} 个片段")
        print(f"  输出目录: {output_dir}")
        print(f"  合并音频: {merged_path}")

    tts.unload_model()
    clear_memory()
    return 0


def cmd_silent(args):
    """生成静音视频"""
    from src.video_processor import VideoProcessor
    from src.config import OUTPUT_DIR

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ 文件不存在: {input_path}")
        return 1

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / f"{input_path.stem}_silent{input_path.suffix}"

    print(f"生成静音视频: {input_path.name}")

    processor = VideoProcessor()
    processor.remove_audio(input_path, output_path)

    print(f"\n✓ 完成: {output_path}")
    return 0


def cmd_dub(args):
    """AI配音(英文→中文)"""
    from src.config import OUTPUT_DIR
    from src.extractor import AudioExtractor
    from src.separator import VocalSeparator
    from src.asr_module import WhisperASR
    from src.translator_m2m100 import M2M100Translator
    from src.tts_qwen3 import Qwen3TTS, get_qwen3_tts
    from src.merger import AudioMerger
    from src.video_processor import VideoProcessor
    from src.subtitle_handler import SRTHandler, SubtitleEntry
    from src.analyzer import MediaAnalyzer
    import soundfile as sf
    import numpy as np

    input_video = Path(args.input) if args.input else Path("data/SpongeBob SquarePants_en.mp4")

    if not input_video.exists():
        print(f"✗ 视频文件不存在: {input_video}")
        return 1

    # 处理时间范围参数
    start_time = args.start_time
    duration = args.duration

    # 如果需要切割视频
    if start_time > 0 or duration > 0:
        print(f"\n[预处理] 切割视频片段...")
        print(f"  开始时间: {start_time:.1f}s")
        if duration > 0:
            print(f"  时长: {duration:.1f}s")

        # 使用ffmpeg切割视频
        segment_video = OUTPUT_DIR / f"{input_video.stem}_segment.mp4"
        if duration > 0:
            cmd = [
                "ffmpeg", "-y", "-i", str(input_video),
                "-ss", str(start_time), "-t", str(duration),
                "-c:v", "copy", "-c:a", "copy",
                str(segment_video)
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-i", str(input_video),
                "-ss", str(start_time),
                "-c:v", "copy", "-c:a", "copy",
                str(segment_video)
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"✗ 视频切割失败: {result.stderr}")
            return 1

        print(f"  ✓ 视频片段已保存: {segment_video}")
        input_video = segment_video

    output_dir = OUTPUT_DIR / f"{input_video.stem}_zh_dubbed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取性能配置
    perf_config = get_performance_config()
    config = get_parallel_config()

    print("=" * 60)
    print("AI配音 - 英文→中文")
    print("=" * 60)

    # 显示系统信息和配置
    perf_config.print_system_info()
    perf_config.print_config()

    # 显示显存状态
    if HAS_MEMORY_MANAGER:
        print_memory_status()

    # 记录日志
    logger.info(f"=== 开始AI配音任务 ===")
    logger.info(f"输入: {input_video}")
    logger.info(f"输出: {output_dir}")
    logger.info(
        f"ASR: {config.asr_model_size} | 翻译: {config.translator_model_size} | TTS: {config.tts_model_size}"
    )

    total_start = time.time()

    print(f"\n输入: {input_video.name}")
    print(f"输出: {output_dir}")

    # ==================== 步骤1: 人声分离 ====================
    step_start = time.time()
    print("\n[1/6] 人声分离...")

    # 检查显存
    sep_device = config.separator_device
    if sep_device == "cuda" and HAS_MEMORY_MANAGER:
        free_mem = get_free_gpu_memory()
        if free_mem < 3.0:
            print(f"  显存不足 ({free_mem:.1f}GB)，人声分离使用CPU")
            sep_device = "cpu"

    extractor = AudioExtractor()
    audio_path = output_dir / "extracted.wav"
    extractor.extract(input_video, audio_path)

    # 获取原始音频时长
    original_audio_info = MediaAnalyzer.analyze_audio(audio_path)
    original_duration = original_audio_info.duration
    print(f"  原始音频时长: {original_duration:.2f}s")

    separator = VocalSeparator(device=sep_device)
    separated = separator.separate(audio_path, output_dir / "separated")
    vocals_path = separated["vocals"]
    background_path = separated["background"]

    # 释放分离器显存
    del separator
    del separated
    clear_memory()

    step_time = time.time() - step_start
    print(f"✓ 人声分离完成 ({step_time:.1f}秒)")
    logger.info(f"[1/6] 人声分离完成 | 耗时: {step_time:.1f}秒")

    # ==================== 步骤2: ASR识别 ====================
    step_start = time.time()
    print("\n[2/6] ASR识别...")

    # 检查显存
    asr_device = config.device
    if asr_device == "cuda" and HAS_MEMORY_MANAGER:
        free_mem = get_free_gpu_memory()
        if free_mem < 1.5:
            print(f"  显存不足 ({free_mem:.1f}GB)，ASR使用CPU")
            asr_device = "cpu"

    asr = WhisperASR(model_size=config.asr_model_size, device=asr_device, language="en")
    asr.load_model()
    results = asr.transcribe(vocals_path)

    # 释放ASR显存
    del asr
    clear_memory()

    step_time = time.time() - step_start
    print(f"✓ 识别完成: {len(results)} 个片段 ({step_time:.1f}秒)")
    logger.info(f"[2/6] ASR识别完成 | 片段: {len(results)} | 耗时: {step_time:.1f}秒")

    # 创建字幕
    subtitle_entries = []
    for i, result in enumerate(results):
        entry = SubtitleEntry(
            index=i + 1,
            start_time=SubtitleEntry.seconds_to_time(result.start_time),
            end_time=SubtitleEntry.seconds_to_time(result.end_time),
            text=result.text,
        )
        subtitle_entries.append(entry)

    SRTHandler.write(subtitle_entries, output_dir / "english.srt")

    # ==================== 步骤3: 翻译 ====================
    step_start = time.time()
    print("\n[3/6] 翻译...")
    print(f"  模型: M2M100-418M")
    print(f"  批处理大小: {config.translator_batch_size}")
    print(f"  设备: {config.device}")

    # 检查显存
    trans_device = config.device
    if trans_device == "cuda" and HAS_MEMORY_MANAGER:
        free_mem = get_free_gpu_memory()
        if free_mem < 2.0:
            print(f"  显存不足 ({free_mem:.1f}GB)，翻译使用CPU")
            trans_device = "cpu"

    # 使用M2M100翻译器
    translator = M2M100Translator(
        source_language="en",
        target_language="zh",
        device=trans_device,
        model_size="418M",  # 使用418M模型，平衡质量和速度
    )
    translator.load_model()

    # 翻译每个字幕条目
    for entry in subtitle_entries:
        if entry.text:
            entry.translated_text = translator.translate(entry.text)

    SRTHandler.write(subtitle_entries, output_dir / "chinese.srt", use_translated=True)

    # 释放翻译器显存
    del translator
    clear_memory()

    step_time = time.time() - step_start
    print(f"✓ 翻译完成 ({step_time:.1f}秒)")
    logger.info(
        f"[3/6] 翻译完成 | 片段: {len(subtitle_entries)} | 耗时: {step_time:.1f}秒"
    )

    # ==================== 步骤4: TTS合成（字幕驱动） ====================
    step_start = time.time()
    print("\n[4/6] TTS合成（字幕驱动，时长对齐）...")

    # 显示显存状态
    if HAS_MEMORY_MANAGER:
        print_memory_status()

    # 检查显存决定TTS配置
    tts_device = config.device
    tts_model_size = config.tts_model_size

    if tts_device == "cuda" and HAS_MEMORY_MANAGER:
        free_mem = get_free_gpu_memory()
        print(f"  可用显存: {free_mem:.1f}GB")

        if free_mem < 2.0:
            print(f"  显存严重不足，TTS使用CPU")
            tts_device = "cpu"
        elif free_mem < 4.0 and tts_model_size == "1.7B":
            print(f"  显存不足以运行1.7B模型，切换到0.6B")
            tts_model_size = "0.6B"

    print(f"  模型: Qwen3-{tts_model_size}")
    print(f"  设备: {tts_device}")
    print(f"  策略: 根据英文字幕时长生成中文TTS")

    # 使用新的字幕驱动TTS引擎
    from src.subtitle_tts_engine import (
        SubtitleTTSEngine,
        TTSConfig,
        create_aligned_tts,
    )

    # 保存字幕文件
    english_srt_path = output_dir / "english.srt"
    chinese_srt_path = output_dir / "chinese.srt"
    SRTHandler.write(subtitle_entries, english_srt_path)
    SRTHandler.write(subtitle_entries, chinese_srt_path, use_translated=True)

    # 创建TTS配置
    tts_config = TTSConfig(
        speed_adjustment=True,  # 启用语速调节
        enable_stretching=True,  # 启用音频拉伸
        use_voice_clone=True,  # 启用音色克隆
        target_duration_tolerance=0.15,  # 时长容差0.15秒
        max_speed_ratio=1.4,  # 最大加速1.4倍
        min_speed_ratio=0.75,  # 最大减速0.75倍
    )

    # 创建TTS引擎
    tts_engine = SubtitleTTSEngine(
        config=tts_config,
        device=tts_device,
    )

    # 准备输出目录
    tts_output_dir = output_dir / "tts_output"
    tts_output_dir.mkdir(parents=True, exist_ok=True)

    # 准备任务（提取参考音频片段）
    print("\n  [4.1] 根据字幕切割参考音频...")
    ref_segments_dir = tts_output_dir / "reference_segments"
    tts_segments_dir = tts_output_dir / "tts_segments"

    # 根据命令行参数调整配置
    if args.no_voice_clone:
        tts_engine.config.use_voice_clone = False
        print("  音色克隆: 禁用")

    if args.no_speed_adjust:
        tts_engine.config.speed_adjustment = False
        print("  语速调节: 禁用")

    tasks = tts_engine.prepare_tasks(
        english_srt_path=english_srt_path,
        chinese_srt_path=chinese_srt_path,
        original_audio_path=vocals_path,  # 使用分离的人声作为参考
        output_dir=tts_segments_dir,
        reference_audio_dir=ref_segments_dir,
        preprocess_subtitles=True,  # 启用字幕预处理
    )

    # 处理TTS任务（带时长控制）
    print("\n  [4.2] 生成时长对齐的中文TTS...")
    processed_tasks = tts_engine.process_tasks(
        tasks=tasks,
        language="chinese",
        show_progress=True,
    )

    # 释放TTS显存
    tts_engine.tts.unload_model()
    clear_memory()

    # 合并TTS音频
    print("\n  [4.3] 合并TTS音频片段...")
    merged_tts_path = output_dir / "merged_tts_aligned.wav"
    tts_engine.merge_task_audio(
        tasks=processed_tasks,
        output_path=merged_tts_path,
        crossfade_ms=10,
    )

    # 验证最终时长
    merged_info = MediaAnalyzer.analyze_audio(merged_tts_path)
    print(f"\n  TTS音频对齐完成: {merged_info.duration:.2f}s (目标: {original_duration:.2f}s)")

    # 不再强制调整TTS音频时长到原视频时长，保持自然语速
    # 只处理差异过大的情况（超过5秒才进行轻微调整）
    duration_diff = abs(merged_info.duration - original_duration)
    if duration_diff > 5.0:
        print(f"  注意: TTS音频时长 ({merged_info.duration:.2f}s) 与原视频 ({original_duration:.2f}s) 差异较大")
        print(f"  保持自然语速，仅添加静音填充到视频时长...")

        # 使用静音填充而不是拉伸音频
        from src.subtitle_tts_engine import AudioProcessor
        import numpy as np

        # 加载现有TTS音频
        tts_audio, sr = sf.read(str(merged_tts_path), dtype="float32")
        current_duration = len(tts_audio) / sr

        if current_duration < original_duration:
            # 在末尾添加静音，使时长匹配视频
            silence_samples = int((original_duration - current_duration) * sr)
            silence = np.zeros(silence_samples, dtype=np.float32)

            # 合并音频和静音
            if tts_audio.ndim == 1:
                final_audio = np.concatenate([tts_audio, silence])
            else:
                silence = np.zeros((silence_samples, tts_audio.shape[1]), dtype=np.float32)
                final_audio = np.concatenate([tts_audio, silence], axis=0)

            final_aligned_path = output_dir / "merged_tts_final.wav"
            sf.write(str(final_aligned_path), final_audio, sr)
            merged_tts_path = final_aligned_path

            # 验证调整后的时长
            final_info = MediaAnalyzer.analyze_audio(merged_tts_path)
            print(f"  调整后时长: {final_info.duration:.2f}s (添加静音填充)")
        else:
            print(f"  保持原TTS音频时长: {current_duration:.2f}s")

    step_time = time.time() - step_start
    logger.info(
        f"[4/6] TTS合成完成 | 片段: {len(processed_tasks)} | 耗时: {step_time:.1f}秒"
    )

    # ==================== 步骤5: 音频处理 ====================
    step_start = time.time()
    print("\n[5/6] 音频处理...")

    # 修复：确保使用正确对齐的TTS音频合并配音和背景音
    merger = AudioMerger()
    final_audio = output_dir / "final_dubbed_zh.wav"
    merger.merge_with_volume(
        merged_tts_path,  # 使用已时间轴对齐的TTS音频
        background_path,
        final_audio,
        vocals_volume=1.5,  # 提高人声音量
        background_volume=0.6,  # 降低背景音量
    )
    
    # 检查合并后音频长度
    import soundfile as sf
    final_audio_data, final_sr = sf.read(str(final_audio))
    final_duration = len(final_audio_data) / final_sr
    print(f"  最终音频时长: {final_duration:.2f}s")

    # 如果需要，调整最终音频时长（使用静音填充或截断）
    if abs(final_duration - original_duration) > 0.5:  # 时长差异超过0.5秒
        print(f"  调整音频时长从 {final_duration:.2f}s 到 {original_duration:.2f}s")
        adjusted_final_path = output_dir / "final_dubbed_zh_adjusted.wav"

        if final_duration < original_duration:
            # 添加静音填充
            silence_samples = int((original_duration - final_duration) * final_sr)
            silence = np.zeros(silence_samples, dtype=np.float32)
            adjusted_audio = np.concatenate([final_audio_data, silence])
        else:
            # 截断音频
            end_sample = int(original_duration * final_sr)
            adjusted_audio = final_audio_data[:end_sample]

        sf.write(str(adjusted_final_path), adjusted_audio, final_sr)

        # 用调整后的文件替换原文件
        final_audio.unlink()
        adjusted_final_path.rename(final_audio)
        print(f"  ✓ 音频时长调整完成")

    # 初始化processor变量供后续使用
    processor = VideoProcessor()

    step_time = time.time() - step_start
    print(f"✓ 音频处理完成 ({step_time:.1f}秒)")
    logger.info(f"[5/6] 音频处理完成 | 耗时: {step_time:.1f}秒")

    # ==================== 步骤6: 视频合成 ====================
    step_start = time.time()
    print("\n[6/6] 视频合成...")

    # 生成静音视频
    silent_video_path = output_dir / f"{input_video.stem}_silent{input_video.suffix}"
    processor.remove_audio(input_video, silent_video_path)

    # 合成最终视频
    output_video = output_dir / f"{input_video.stem}_中文配音{input_video.suffix}"
    processor.replace_audio(silent_video_path, final_audio, output_video)

    total_time = time.time() - total_start
    step_time = time.time() - step_start

    print("\n" + "=" * 60)
    print("✓ AI配音完成!")
    print("=" * 60)
    print(f"总耗时: {total_time:.1f}秒 ({total_time / 60:.1f}分钟)")
    print(f"输出文件:")
    print(f"  最终视频: {output_video.name}")
    print(f"  静音视频: {silent_video_path.name}")
    print(f"  字幕文件: chinese.srt, english.srt")
    print(f"  配音音频: final_dubbed_zh.wav")
    print(f"  原始音频: extracted.wav ({original_duration:.2f}s)")
    print(f"  位置: {output_dir}")

    logger.info(f"=== AI配音完成 ===")
    logger.info(f"总耗时: {total_time:.1f}秒 ({total_time / 60:.1f}分钟)")
    logger.info(f"输出: {output_video}")

    # 最终清理
    clear_memory()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="音视频AI工具箱",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 视频转音频
    python video_tool.py convert video.mp4 -o audio.mp3 -f mp3

    # 测试所有模块
    python video_tool.py test

    # 人声分离
    python video_tool.py separate video.mp4

    # 合并音频
    python video_tool.py merge -v vocals.wav -b background.wav

    # 替换视频音频
    python video_tool.py replace video.mp4 -a new_audio.wav

    # AI配音(完整流程)
    python video_tool.py dub video.mp4
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # convert 命令
    convert_parser = subparsers.add_parser("convert", help="视频转音频")
    convert_parser.add_argument("input", help="输入视频文件")
    convert_parser.add_argument("-o", "--output", help="输出文件路径")
    convert_parser.add_argument("-f", "--format", default="wav", help="输出格式")
    convert_parser.add_argument("-q", "--quality", help="音频质量")
    convert_parser.add_argument("-r", "--sample-rate", type=int, default=44100)
    convert_parser.add_argument("-c", "--channels", type=int, default=2)
    convert_parser.add_argument("--device", choices=["cuda", "cpu"], help="计算设备")
    convert_parser.set_defaults(func=cmd_convert)

    # test 命令
    test_parser = subparsers.add_parser("test", help="测试所有模块")
    test_parser.set_defaults(func=cmd_test)

    # separate 命令
    separate_parser = subparsers.add_parser("separate", help="人声分离")
    separate_parser.add_argument("input", help="输入文件")
    separate_parser.add_argument("--device", choices=["cuda", "cpu"], help="计算设备")
    separate_parser.set_defaults(func=cmd_separate)

    # merge 命令
    merge_parser = subparsers.add_parser("merge", help="合并音频")
    merge_parser.add_argument("-v", "--vocals", required=True, help="人声文件")
    merge_parser.add_argument("-b", "--background", required=True, help="背景文件")
    merge_parser.add_argument("-o", "--output", help="输出文件路径(默认: final_dubbed_zh.wav)")
    merge_parser.add_argument("--vocals-vol", type=float, default=1.0, help="人声音量")
    merge_parser.add_argument(
        "--background-vol", type=float, default=0.8, help="背景音量"
    )
    merge_parser.set_defaults(func=cmd_merge)

    # replace 命令
    replace_parser = subparsers.add_parser("replace", help="替换视频音频")
    replace_parser.add_argument("input", help="输入视频")
    replace_parser.add_argument("-a", "--audio", required=True, help="新音频文件")
    replace_parser.set_defaults(func=cmd_replace)

    # dub 命令
    dub_parser = subparsers.add_parser("dub", help="AI配音(英文→中文)")
    dub_parser.add_argument(
        "input", nargs="?", help="输入视频(默认: data/SpongeBob SquarePants_en.mp4)"
    )
    dub_parser.add_argument(
        "--legacy-tts",
        action="store_true",
        help="使用传统TTS模式（不按字幕时长对齐）",
    )
    dub_parser.add_argument(
        "--no-speed-adjust",
        action="store_true",
        help="禁用语速调节",
    )
    dub_parser.add_argument(
        "--no-voice-clone",
        action="store_true",
        help="禁用音色克隆",
    )
    dub_parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="开始时间（秒，默认0）",
    )
    dub_parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="转换时长（秒，默认0表示转换全部）",
    )
    dub_parser.set_defaults(func=cmd_dub)

    # asr 命令 - 语音识别
    asr_parser = subparsers.add_parser("asr", help="ASR语音识别")
    asr_parser.add_argument("input", help="输入视频/音频文件")
    asr_parser.add_argument("-m", "--model", default="small", help="模型大小")
    asr_parser.add_argument("-l", "--language", default="en", help="语言")
    asr_parser.add_argument("--device", choices=["cuda", "cpu"], help="计算设备")
    asr_parser.set_defaults(func=cmd_asr)

    # translate 命令 - 翻译
    translate_parser = subparsers.add_parser("translate", help="翻译文本/字幕")
    translate_parser.add_argument("input", help="输入文件(.txt/.srt)")
    translate_parser.add_argument("-s", "--source", default="en", help="源语言")
    translate_parser.add_argument("-t", "--target", default="zh", help="目标语言")
    translate_parser.add_argument("-m", "--model", default="base", help="模型大小")
    translate_parser.add_argument("--device", choices=["cuda", "cpu"], help="计算设备")
    translate_parser.set_defaults(func=cmd_translate)

    # tts 命令 - 语音合成
    tts_parser = subparsers.add_parser("tts", help="TTS语音合成")
    tts_parser.add_argument("input", help="输入文本文件(.txt/.srt)")
    tts_parser.add_argument("-l", "--language", default="chinese", help="语言")
    tts_parser.add_argument("-m", "--model", default="0.6B", help="模型大小")
    tts_parser.add_argument("-r", "--reference", help="参考音频(用于音色克隆)")
    tts_parser.add_argument("--device", choices=["cuda", "cpu"], help="计算设备")
    tts_parser.set_defaults(func=cmd_tts)

    # silent 命令 - 生成静音视频
    silent_parser = subparsers.add_parser("silent", help="生成静音视频")
    silent_parser.add_argument("input", help="输入视频文件")
    silent_parser.add_argument("-o", "--output", help="输出文件路径")
    silent_parser.set_defaults(func=cmd_silent)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
