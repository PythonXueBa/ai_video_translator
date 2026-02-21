"""
配置模块
"""

import platform
from pathlib import Path
from multiprocessing import cpu_count

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

SYSTEM = platform.system().lower()
IS_WINDOWS = SYSTEM == "windows"
IS_LINUX = SYSTEM == "linux"
IS_MACOS = SYSTEM == "darwin"

CPU_COUNT = cpu_count()
DEFAULT_WORKERS = max(1, CPU_COUNT - 1)

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 2
DEFAULT_FORMAT = "wav"
DEFAULT_MODEL = "htdemucs"

STEM_MAPPING = {
    "vocals": ["vocals"],
    "background": ["drums", "bass", "other"],
}

SUPPORTED_AUDIO_FORMATS = ["wav", "mp3", "flac", "m4a", "ogg", "aac", "wma"]
SUPPORTED_VIDEO_FORMATS = [
    "mp4",
    "avi",
    "mov",
    "mkv",
    "flv",
    "wmv",
    "webm",
    "m4v",
    "3gp",
]
SUPPORTED_FORMATS = SUPPORTED_AUDIO_FORMATS + SUPPORTED_VIDEO_FORMATS
VIDEO_EXTENSIONS = {f".{ext}" for ext in SUPPORTED_VIDEO_FORMATS}
AUDIO_EXTENSIONS = {f".{ext}" for ext in SUPPORTED_AUDIO_FORMATS}
