# AI Video Translator

基于 Qwen3-TTS 的高质量多语言 AI 视频翻译与配音解决方案

**支持平台**: Windows | Linux | macOS

---

## 功能特性

- **多语言翻译** - 支持 18 种语言互译（中/英/日/韩/法/德/西/俄/葡/意/阿/印/越/泰/印尼等）
- **视频转音频** - 支持多种格式 (MP3, WAV, M4A, OGG)
- **人声分离** - 基于 Demucs，分离人声和背景音
- **ASR 语音识别** - 基于 Whisper，支持多语言
- **机器翻译** - 基于 M2M100-1.2B，离线高质量翻译
- **TTS 语音合成** - 基于 Qwen3-TTS，支持音色克隆，保持自然语速和情绪
- **智能资源管理** - 自动检测 GPU 显存，动态优化配置
- **时间轴对齐** - 精确的时间戳同步，确保配音与原视频同步
- **时间范围选择** - 支持指定视频片段进行配音转换
- **音量自动优化** - 人声音量增强，背景音量自动平衡

---

## 快速开始

### Windows 用户（推荐）

下载离线安装包，双击运行即可，无需安装 Python 或其他依赖。

### 从源码安装

```bash
# 1. 克隆仓库
git clone https://github.com/PythonXueBa/ai_video_translator.git
cd ai_video_translator

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装 TTS 库
pip install qwen-tts
```

---

## 使用方法

### 基本用法

```bash
# 英文转中文（默认）
python video_tool.py dub

# 指定视频文件
python video_tool.py dub video.mp4

# 指定时间范围（从第10秒开始，处理30秒）
python video_tool.py dub video.mp4 --start-time 10 --duration 30
```

### 多语言翻译

```bash
# 日文转中文
python video_tool.py dub japanese_video.mp4 --source-lang ja --target-lang zh

# 英文转西班牙文
python video_tool.py dub video.mp4 --target-lang es

# 中文转英文
python video_tool.py dub chinese_video.mp4 --source-lang zh --target-lang en
```

### 高级选项

```bash
# 禁用音色克隆（使用默认音色）
python video_tool.py dub video.mp4 --no-voice-clone

# 禁用语速调节
python video_tool.py dub video.mp4 --no-speed-adjust

# 完整示例
python video_tool.py dub video.mp4 --source-lang en --target-lang zh --start-time 0 --duration 60
```

---

## 命令详解

### AI 配音 `dub`

完整的视频翻译配音流程。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入视频文件 | data/SpongeBob SquarePants_en.mp4 |
| `--source-lang` | 源语言代码 | en |
| `--target-lang` | 目标语言代码 | zh |
| `--start-time` | 开始时间（秒） | 0 |
| `--duration` | 处理时长（秒） | 0（完整视频） |
| `--no-voice-clone` | 禁用音色克隆 | - |
| `--no-speed-adjust` | 禁用语速调节 | - |

### 支持的语言代码

| 代码 | 语言 | 代码 | 语言 |
|------|------|------|------|
| zh | 中文 | en | 英文 |
| ja | 日文 | ko | 韩文 |
| es | 西班牙文 | fr | 法文 |
| de | 德文 | ru | 俄文 |
| pt | 葡萄牙文 | it | 意大利文 |
| ar | 阿拉伯文 | hi | 印地文 |
| vi | 越南文 | th | 泰文 |
| id | 印尼文 | - | - |

---

## 完整 AI 配音流程

```
视频输入
    │
    ▼
┌─────────────────────────┐
│ 0. 视频切割 (FFmpeg)    │
│    支持指定时间范围     │
│    输出: segment.mp4    │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 1. 人声分离 (Demucs)    │
│    输出: vocals.wav     │
│          background.wav │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 2. ASR识别 (Whisper)    │
│    语音转文字           │
│    输出: source.srt     │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 3. 翻译 (M2M100-1.2B)   │
│    高质量离线翻译       │
│    输出: target.srt     │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 4. TTS合成 (Qwen3-TTS)  │
│    音色克隆 + 自然语速  │
│    保持完整情绪音色     │
│    输出: tts_*.wav      │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 5. 音频处理 + 视频合成  │
│    人声增强+背景平衡    │
│    输出: 配音视频.mp4   │
└─────────────────────────┘
```

---

## 输出文件

运行 `python video_tool.py dub video.mp4` 后生成：

```
output/视频名_zh_dubbed/
├── 视频名_中文配音.mp4    # 最终配音视频
├── 视频名_silent.mp4     # 静音视频
├── chinese.srt            # 中文字幕文件
├── english.srt            # 英文字幕文件
├── final_dubbed_zh.wav    # 最终配音音频
├── merged_tts.wav         # TTS音频
├── extracted.wav          # 提取的原始音频
├── separated/
│   ├── vocals.wav         # 分离的人声
│   └── background.wav     # 分离的背景音
└── tts_output/            # TTS合成文件
    └── tts_*.wav
```

---

## 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Windows 10+ / Linux / macOS | Windows 11 / Ubuntu 22.04 |
| Python | 3.8+ | 3.10+ |
| FFmpeg | 4.0+ | 5.0+ |
| 磁盘空间 | 15GB+ | 30GB+ |
| 内存 | 8GB | 16GB+ |
| GPU 显存 | 6GB | 12GB+ |

---

## 模型规格

| 组件 | 模型 | 规格 | 显存需求 |
|------|------|------|----------|
| ASR | Whisper | base / small | 1GB |
| 翻译 | M2M100 | 1.2B | 4-6GB |
| TTS | Qwen3-TTS | 1.7B | 4-6GB |
| 人声分离 | Demucs | htdemucs | 2-3GB |

**总计**: 约 12-16GB 显存（可自动降级到 CPU）

---

## 离线安装包制作

### Windows 离线包

使用 PyInstaller 打包：

```bash
# 安装打包工具
pip install pyinstaller

# 创建 spec 文件并打包
pyinstaller --name="AI_Video_Translator" \
    --onefile \
    --windowed \
    --add-data "src;src" \
    --hidden-import=torch \
    --hidden-import=transformers \
    video_tool.py
```

打包后的可执行文件位于 `dist/AI_Video_Translator.exe`

---

## 许可证

MIT License

---

## 问题反馈

如有问题，请在 GitHub Issues 中提交：
https://github.com/PythonXueBa/ai_video_translator/issues
