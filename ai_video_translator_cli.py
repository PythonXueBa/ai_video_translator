#!/usr/bin/env python3
"""
AI Video Translator - 统一CLI入口
支持所有功能的终端交互界面
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# 确保可以导入src模块
if getattr(sys, 'frozen', False):
    # 运行在PyInstaller打包环境
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)

# 设置环境变量
os.environ['PYTHONPATH'] = BASE_DIR
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 导入功能模块
from src.config import Config
from src.analyzer import MediaAnalyzer
from src.extractor import AudioExtractor
from src.separator import VocalSeparator
from src.asr_module import ASRModule
from src.translator_m2m100 import TranslatorM2M100
from src.tts_qwen3 import TTSQwen3
from src.subtitle_handler import SRTHandler
from src.merger import AudioMerger
from src.video_processor import VideoProcessor
from src.performance_config import PerformanceConfig


def print_banner():
    """打印程序横幅"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║           AI Video Translator - 智能视频翻译工具              ║
║                                                              ║
║     支持: ASR语音识别 | 机器翻译 | AI配音 | 人声分离         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def print_menu():
    """打印主菜单"""
    print("""
【主菜单】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 1. 🎬 AI配音 (完整流程)     - 视频翻译配音一键完成
 2. 🎵 人声分离              - 分离人声和背景音乐
 3. 📝 ASR语音识别           - 语音转文字生成字幕
 4. 🌐 翻译字幕              - 翻译字幕文件
 5. 🔊 TTS语音合成           - 文字转语音
 6. 🔀 合并音频              - 合并人声和背景音
 7. 📹 视频处理              - 替换音轨/生成静音视频
 8. ⚙️  系统测试             - 测试所有模块
 9. ❓ 帮助                  - 显示使用说明
 0. 🚪 退出

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)


def get_input(prompt, default=None):
    """获取用户输入"""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()


def get_bool_input(prompt, default=True):
    """获取布尔输入"""
    default_str = "Y/n" if default else "y/N"
    user_input = input(f"{prompt} [{default_str}]: ").strip().lower()
    if not user_input:
        return default
    return user_input in ['y', 'yes', '是', '1', 'true']


def cmd_dub_interactive():
    """交互式AI配音"""
    print("\n【AI配音】")
    print("-" * 50)

    # 视频文件
    video = get_input("视频文件路径", "data/SpongeBob SquarePants_en.mp4")
    if not os.path.exists(video):
        print(f"❌ 文件不存在: {video}")
        return

    # 语言设置
    print("\n支持的语言: en/zh/ja/ko/es/fr/de/ru/pt/it/ar/hi/vi/th/id")
    source_lang = get_input("源语言", "en")
    target_lang = get_input("目标语言", "zh")

    # 时间范围
    print("\n时间范围设置 (0 = 完整视频)")
    start_time = get_input("开始时间(秒)", "0")
    duration = get_input("处理时长(秒)", "0")

    # 选项
    voice_clone = get_bool_input("启用音色克隆", True)
    speed_adjust = get_bool_input("启用语速调节", False)

    print("\n" + "=" * 50)
    print("开始AI配音流程...")
    print("=" * 50)

    # 构建命令
    cmd = [
        sys.executable, "video_tool.py", "dub", video,
        "--source-lang", source_lang,
        "--target-lang", target_lang,
        "--start-time", start_time,
        "--duration", duration
    ]

    if not voice_clone:
        cmd.append("--no-voice-clone")
    if not speed_adjust:
        cmd.append("--no-speed-adjust")

    # 执行
    subprocess.run(cmd, cwd=BASE_DIR)


def cmd_separate_interactive():
    """交互式人声分离"""
    print("\n【人声分离】")
    print("-" * 50)

    input_file = get_input("输入文件(视频或音频)")
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        return

    device = get_input("计算设备 (cuda/cpu)", "cuda")

    print("\n开始人声分离...")
    subprocess.run([
        sys.executable, "video_tool.py", "separate", input_file,
        "--device", device
    ], cwd=BASE_DIR)


def cmd_asr_interactive():
    """交互式ASR识别"""
    print("\n【ASR语音识别】")
    print("-" * 50)

    input_file = get_input("输入文件(视频或音频)")
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        return

    print("\n模型大小: tiny/base/small/medium/large")
    model = get_input("模型大小", "small")
    language = get_input("语言代码", "en")
    device = get_input("计算设备 (cuda/cpu)", "cuda")

    print("\n开始语音识别...")
    subprocess.run([
        sys.executable, "video_tool.py", "asr", input_file,
        "--model", model,
        "--language", language,
        "--device", device
    ], cwd=BASE_DIR)


def cmd_translate_interactive():
    """交互式翻译"""
    print("\n【翻译字幕】")
    print("-" * 50)

    input_file = get_input("字幕文件路径(.srt)")
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        return

    print("\n支持的语言: en/zh/ja/ko/es/fr/de/ru/pt/it/ar/hi/vi/th/id")
    source = get_input("源语言", "en")
    target = get_input("目标语言", "zh")
    device = get_input("计算设备 (cuda/cpu)", "cuda")

    print("\n开始翻译...")
    subprocess.run([
        sys.executable, "video_tool.py", "translate", input_file,
        "--source", source,
        "--target", target,
        "--device", device
    ], cwd=BASE_DIR)


def cmd_tts_interactive():
    """交互式TTS"""
    print("\n【TTS语音合成】")
    print("-" * 50)

    input_file = get_input("输入文件(.txt或.srt)")
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        return

    language = get_input("语言 (chinese/english/japanese等)", "chinese")

    reference = get_input("参考音频路径(音色克隆,可选)", "")
    if reference and not os.path.exists(reference):
        print(f"⚠️ 参考音频不存在，将使用默认音色")
        reference = ""

    device = get_input("计算设备 (cuda/cpu)", "cuda")

    print("\n开始语音合成...")
    cmd = [
        sys.executable, "video_tool.py", "tts", input_file,
        "--language", language,
        "--device", device
    ]
    if reference:
        cmd.extend(["--reference", reference])

    subprocess.run(cmd, cwd=BASE_DIR)


def cmd_merge_interactive():
    """交互式合并音频"""
    print("\n【合并音频】")
    print("-" * 50)

    vocals = get_input("人声文件路径")
    if not os.path.exists(vocals):
        print(f"❌ 文件不存在: {vocals}")
        return

    background = get_input("背景音文件路径")
    if not os.path.exists(background):
        print(f"❌ 文件不存在: {background}")
        return

    output = get_input("输出文件路径", "final_dubbed.wav")
    vocals_vol = get_input("人声音量倍数", "1.5")
    bg_vol = get_input("背景音量倍数", "0.6")

    print("\n开始合并音频...")
    subprocess.run([
        sys.executable, "video_tool.py", "merge",
        "--vocals", vocals,
        "--background", background,
        "--output", output,
        "--vocals-vol", vocals_vol,
        "--background-vol", bg_vol
    ], cwd=BASE_DIR)


def cmd_video_interactive():
    """交互式视频处理"""
    print("\n【视频处理】")
    print("-" * 50)

    print("\n1. 替换视频音轨")
    print("2. 生成静音视频")
    choice = get_input("选择功能 (1/2)", "1")

    if choice == "1":
        video = get_input("视频文件路径")
        audio = get_input("音频文件路径")
        output = get_input("输出文件路径", "output_replaced.mp4")

        if not os.path.exists(video) or not os.path.exists(audio):
            print("❌ 文件不存在")
            return

        print("\n开始替换音轨...")
        subprocess.run([
            sys.executable, "video_tool.py", "replace", video,
            "--audio", audio,
            "--output", output
        ], cwd=BASE_DIR)

    elif choice == "2":
        video = get_input("视频文件路径")
        output = get_input("输出文件路径", "output_silent.mp4")

        if not os.path.exists(video):
            print("❌ 文件不存在")
            return

        print("\n开始生成静音视频...")
        subprocess.run([
            sys.executable, "video_tool.py", "silent", video,
            "--output", output
        ], cwd=BASE_DIR)


def cmd_test():
    """系统测试"""
    print("\n【系统测试】")
    print("=" * 50)
    subprocess.run([sys.executable, "video_tool.py", "test"], cwd=BASE_DIR)


def cmd_help():
    """显示帮助"""
    print("""
【使用帮助】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 AI配音完整流程:
   1. 人声分离 - 分离人声和背景音
   2. ASR识别 - 将语音转为文字字幕
   3. 机器翻译 - 翻译字幕为目标语言
   4. TTS合成 - 将翻译后的文字转为语音
   5. 音频合并 - 合并新的人声和原背景音
   6. 视频合成 - 将新音频合成到视频

📌 支持的格式:
   视频: MP4, AVI, MKV, MOV, WMV, FLV
   音频: WAV, MP3, M4A, OGG, FLAC
   字幕: SRT

📌 快捷键:
   Ctrl+C - 取消当前操作
   Enter  - 确认默认选项

📌 输出位置:
   所有输出文件保存在 output/ 目录

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)


def main():
    """主函数"""
    print_banner()

    # 检查是否是直接运行某个命令
    if len(sys.argv) > 1:
        # 直接调用 video_tool.py
        subprocess.run([sys.executable, "video_tool.py"] + sys.argv[1:], cwd=BASE_DIR)
        return

    # 交互式菜单
    while True:
        print_menu()
        choice = get_input("请选择功能 (0-9)")

        if choice == '0':
            print("\n感谢使用 AI Video Translator，再见！")
            break
        elif choice == '1':
            cmd_dub_interactive()
        elif choice == '2':
            cmd_separate_interactive()
        elif choice == '3':
            cmd_asr_interactive()
        elif choice == '4':
            cmd_translate_interactive()
        elif choice == '5':
            cmd_tts_interactive()
        elif choice == '6':
            cmd_merge_interactive()
        elif choice == '7':
            cmd_video_interactive()
        elif choice == '8':
            cmd_test()
        elif choice == '9':
            cmd_help()
        else:
            print("\n❌ 无效选择，请重新输入")

        input("\n按回车键继续...")
        print("\n" * 2)


if __name__ == "__main__":
    main()
