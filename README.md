# AI Video Translator

基于 Qwen3-TTS 的高质量多语言 AI 视频翻译与配音工具

## 功能特性

- **AI配音** - 完整的视频翻译配音流程（人声分离→ASR→翻译→TTS→合并）
- **人声分离** - 基于 Demucs 分离人声和背景音
- **ASR识别** - 基于 Whisper 语音转文字/字幕
- **机器翻译** - 基于 M2M100-1.2B 支持100种语言互译
- **TTS合成** - 基于 Qwen3-TTS (1.7B/0.6B)，支持音色克隆和语速调节
- **时间范围选择** - 支持指定视频片段处理
- **音量优化** - 自动平衡人声和背景音量
- **完整日志系统** - 详细的处理日志记录

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

## 支持平台

- ✅ Linux
- ✅ macOS
- ✅ Windows

## 安装

### 环境要求

- Python 3.9+
- FFmpeg（系统依赖）
- CUDA（可选，用于GPU加速）

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/pythonxueba/ai-video-translator.git
cd ai-video-translator

# 安装依赖
pip install -r requirements.txt

# 安装 Qwen-TTS（语音合成）
pip install qwen-tts

# 验证安装
python video_tool.py test
```

### 安装 FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
下载并安装 FFmpeg，添加到系统 PATH

---

## 功能使用指南

### 1. AI配音（完整流程）

将视频从一种语言配音成另一种语言：

```bash
# 英文视频转中文配音
python video_tool.py dub video.mp4

# 指定时间范围（从第10秒开始，处理30秒）
python video_tool.py dub video.mp4 --start-time 10 --duration 30

# 日文转中文
python video_tool.py dub japanese_video.mp4 --source-lang ja --target-lang zh

# 中文转英文
python video_tool.py dub chinese_video.mp4 --source-lang zh --target-lang en

# 禁用音色克隆（使用默认音色）
python video_tool.py dub video.mp4 --no-voice-clone

# 禁用语速调节
python video_tool.py dub video.mp4 --no-speed-adjust
```

**支持的源/目标语言：**
- `zh` - 中文
- `en` - 英文
- `ja` - 日文
- `ko` - 韩文
- `es` - 西班牙文
- `fr` - 法文
- `de` - 德文
- `ru` - 俄文
- `pt` - 葡萄牙文
- `it` - 意大利文
- `ar` - 阿拉伯文
- `hi` - 印地文
- `vi` - 越南文
- `th` - 泰文
- `id` - 印尼文

---

### 2. 视频转音频

从视频中提取音频：

```bash
# 提取为 WAV 格式（默认）
python video_tool.py convert video.mp4

# 提取为 MP3
python video_tool.py convert video.mp4 -f mp3

# 指定输出路径
python video_tool.py convert video.mp4 -o output/audio.wav

# 指定采样率和声道
python video_tool.py convert video.mp4 -r 48000 -c 2
```

---

### 3. 人声分离

将音频/视频中的人声和背景音分离：

```bash
# 分离视频中的音频
python video_tool.py separate video.mp4

# 分离音频文件
python video_tool.py separate audio.wav

# 指定处理设备（cuda/cpu）
python video_tool.py separate video.mp4 --device cuda
```

**输出文件：**
- `vocals.wav` - 人声
- `background.wav` - 背景音（鼓声+贝斯+其他）

---

### 4. ASR语音识别（语音转字幕）

将音频/视频转换为字幕文件：

```bash
# 识别视频生成字幕
python video_tool.py asr video.mp4

# 识别音频
python video_tool.py asr audio.wav

# 指定语言和模型
python video_tool.py asr video.mp4 -l en -m small

# 模型选项: tiny, base, small, medium, large
```

**输出文件：**
- `english.srt` - SRT字幕文件
- `english.txt` - 纯文本文件

---

### 5. 字幕/文本翻译

翻译字幕文件或文本文件：

```bash
# 翻译 SRT 字幕文件
python video_tool.py translate subtitles.srt -s en -t zh

# 翻译文本文件
python video_tool.py translate text.txt -s en -t zh

# 指定模型和设备
python video_tool.py translate text.txt -s ja -t zh --device cuda
```

**输出文件：**
- `subtitles_zh.srt` - 翻译后的字幕
- `text_zh.txt` - 翻译后的文本

---

### 6. TTS语音合成

将文本转换为语音：

```bash
# 合成文本文件
python video_tool.py tts text.txt

# 合成字幕文件
python video_tool.py tts subtitles.srt

# 指定语言
python video_tool.py tts text.txt -l chinese

# 使用参考音频进行音色克隆
python video_tool.py tts text.txt -r reference_audio.wav

# 指定模型大小 (1.7B 或 0.6B)
python video_tool.py tts text.txt -m 1.7B
```

**输出文件：**
- `tts_segment_*.wav` - 分段音频
- `merged_tts.wav` - 合并后的完整音频

---

### 7. 音频合并

合并人声和背景音：

```bash
# 基础合并
python video_tool.py merge -v vocals.wav -b background.wav

# 调整音量
python video_tool.py merge -v vocals.wav -b background.wav --vocals-vol 1.5 --background-vol 0.6

# 指定输出路径
python video_tool.py merge -v vocals.wav -b background.wav -o output.wav
```

---

### 8. 替换视频音频

用新音频替换视频的音轨：

```bash
# 替换音频
python video_tool.py replace video.mp4 -a new_audio.wav
```

---

### 9. 生成静音视频

移除视频中的音频：

```bash
# 生成静音视频
python video_tool.py silent video.mp4

# 指定输出路径
python video_tool.py silent video.mp4 -o silent_video.mp4
```

---

### 10. 测试所有模块

验证安装和模块功能：

```bash
python video_tool.py test
```

---

## 完整工作流程示例

### 示例1：完整的视频翻译流程

```bash
# 步骤1: 提取音频
python video_tool.py convert input.mp4 -o audio.wav

# 步骤2: 人声分离
python video_tool.py separate audio.wav

# 步骤3: ASR识别（生成英文字幕）
python video_tool.py asr vocals.wav -l en

# 步骤4: 翻译字幕（英转中）
python video_tool.py translate english.srt -s en -t zh

# 步骤5: TTS合成（使用原声克隆）
python video_tool.py tts chinese.srt -r vocals.wav -l chinese

# 步骤6: 合并音频
python video_tool.py merge -v merged_tts.wav -b background.wav --vocals-vol 1.5 --background-vol 0.6

# 步骤7: 替换视频音频
python video_tool.py replace input.mp4 -a final_dubbed_zh.wav
# 注意: replace命令会直接修改输入视频，建议先备份
```

### 示例2：一键完成

```bash
# 使用 dub 命令一键完成所有步骤
python video_tool.py dub input.mp4 --source-lang en --target-lang zh

# 处理视频片段（从第0秒开始，处理120秒）
python video_tool.py dub input.mp4 --start-time 0 --duration 120 --source-lang en --target-lang zh
```

### 实际生成示例

基于 SpongeBob SquarePants 英文视频生成2分钟中文配音：

```bash
python video_tool.py dub data/SpongeBob\ SquarePants_en.mp4 \
  --start-time 0 --duration 120 \
  --source-lang en --target-lang zh
```

**生成结果：**
- 输入时长：120秒（2分钟）
- 总处理时间：约5.4分钟（Tesla T4 GPU）
- 识别片段：24个语音片段
- 输出文件：
  - `SpongeBob SquarePants_en_segment_中文配音.mp4` (7.9MB)
  - `chinese.srt` - 中文字幕
  - `english.srt` - 英文字幕
  - `final_dubbed_zh.wav` - 配音音频

**处理流程耗时：**
1. 人声分离：7.6秒
2. ASR识别：6.7秒（24个片段）
3. 翻译：26.1秒
4. TTS合成：277.8秒（24个片段，音色克隆）
5. 音频合并：0.4秒
6. 视频合成：3秒

---

## 项目结构

```
ai-video-translator/
├── video_tool.py          # 主程序入口
├── requirements.txt       # Python依赖
├── README.md             # 项目说明
├── .gitignore            # Git忽略文件
├── src/                  # 核心模块
│   ├── __init__.py       # 包初始化
│   ├── config.py         # 项目配置
│   ├── logger.py         # 日志系统
│   ├── analyzer.py       # 媒体文件分析
│   ├── extractor.py      # 音频提取
│   ├── separator.py      # 人声分离(Demucs)
│   ├── asr_module.py     # ASR识别(Whisper)
│   ├── translator_m2m100.py  # 翻译(M2M100)
│   ├── tts_qwen3.py      # TTS合成(Qwen3)
│   ├── subtitle_handler.py   # 字幕处理
│   ├── subtitle_tts_engine.py  # 字幕驱动TTS
│   ├── merger.py         # 音频合并
│   ├── video_processor.py    # 视频处理
│   ├── performance_config.py # 性能配置
│   ├── memory_manager.py     # 显存管理
│   ├── splitter.py       # 音频分割
│   └── audio_aligner.py  # 音频对齐
├── data/                 # 示例数据
└── logs/                 # 日志文件
```

---

## 性能优化

### GPU加速

程序会自动检测并使用GPU加速。如需强制使用CPU：

```bash
python video_tool.py dub video.mp4 --device cpu
```

### 内存管理

- 自动检测GPU显存并调整模型
- 支持显存不足时自动降级到CPU
- 支持模型卸载释放显存

---

## 日志系统

所有操作都会记录到 `logs/` 目录：

```
logs/
├── ai_video_translator_YYYYMMDD_HHMMSS.log
```

日志格式：`时间 | 模块名 | 级别 | 消息`

---

## 常见问题

### Q: 显存不足怎么办？
A: 程序会自动检测并切换到CPU模式，或选择更小的模型。

### Q: 支持哪些视频格式？
A: mp4, avi, mov, mkv, flv, wmv, webm, m4v, 3gp

### Q: 支持哪些音频格式？
A: wav, mp3, flac, m4a, ogg, aac, wma

### Q: 如何提高翻译质量？
A: 使用更大的翻译模型（1.2B），或手动校对字幕后再合成。

### Q: TTS模型如何选择？
A: 默认使用1.7B模型（质量更好），显存不足时可切换至0.6B模型：
```bash
python video_tool.py tts text.txt -m 0.6B
```

### Q: replace命令会覆盖原视频吗？
A: 是的，replace命令会直接修改输入视频。如需保留原视频，请先复制备份。

### Q: 2分钟视频需要多久处理时间？
A: 在 Tesla T4 GPU 上约需5-6分钟，具体时间取决于：
- 语音片段数量（2分钟视频约20-30个片段）
- TTS模型大小（1.7B比0.6B慢但质量更好）
- 是否启用音色克隆

### Q: 生成的字幕可以编辑吗？
A: 可以，所有生成的字幕都保存为SRT格式，可用任何文本编辑器修改：
```bash
# 编辑中文字幕后重新合成
python video_tool.py tts chinese.srt -r vocals.wav -l chinese
```

---

## 许可证

MIT License

---

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

GitHub: [@pythonxueba](https://github.com/pythonxueba)
