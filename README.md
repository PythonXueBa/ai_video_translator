# AI Video Translator

基于 Qwen3-TTS 的高质量英文→中文 AI 视频翻译与配音解决方案

**支持平台**: Windows | Linux | macOS | Android/Termux

---

## 功能特性

- **视频转音频** - 支持多种格式 (MP3, WAV, M4A, OGG)
- **人声分离** - 基于 Demucs，分离人声和背景音
- **ASR 语音识别** - 基于 Whisper，支持多语言
- **机器翻译** - 基于 M2M100，离线翻译
- **TTS 语音合成** - 基于 Qwen3-TTS，支持音色克隆
- **智能资源管理** - 自动检测 GPU 显存，动态优化配置
- **时间轴对齐** - 精确的时间戳同步，确保配音与原视频同步

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 查看功能

```bash
python video_tool.py --help
```

### 3. 测试模块

```bash
python video_tool.py test
```

### 4. AI 配音

```bash
python video_tool.py dub data/gdot_En.mp4
```

---

## Demo 示例

### 前后对比视频

| 原始视频 (英文) | 翻译后视频 (中文配音) |
|----------------|----------------------|
| [gdot_En.mp4](data/gdot_En.mp4) | [gdot_En_中文配音.mp4](output/gdot_En_zh_dubbed/gdot_En_中文配音.mp4) |

**原始视频**: 8分40秒英文演讲视频  
**处理后**: 完整中文AI配音，保留原始画面和背景音

---

## 命令详解

### 1. 视频转音频 `convert`

将视频文件转换为音频文件。

```bash
python video_tool.py convert video.mp4 -o audio.mp3 -f mp3 -q 320k
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入视频文件 | 必填 |
| `-o, --output` | 输出文件路径 | 自动生成 |
| `-f, --format` | 输出格式 (wav/mp3/m4a/ogg) | wav |
| `-q, --quality` | 音频质量 (如 320k) | - |
| `-r, --sample-rate` | 采样率 | 44100 |
| `-c, --channels` | 声道数 | 2 |

### 2. 人声分离 `separate`

将音频分离为人声和背景音。

```bash
python video_tool.py separate video.mp4 --device cuda
```

| 参数 | 说明 |
|------|------|
| `input` | 输入文件 (视频或音频) |
| `--device` | 计算设备 (cuda/cpu) |

### 3. ASR语音识别 `asr`

将语音识别为文字并生成字幕。

```bash
python video_tool.py asr video.mp4 -m small -l en
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入视频/音频文件 | 必填 |
| `-m, --model` | 模型大小 (tiny/base/small/medium/large) | small |
| `-l, --language` | 语言代码 | en |
| `--device` | 计算设备 (cuda/cpu) | 自动 |

### 4. 翻译 `translate`

翻译文本或字幕文件。

```bash
python video_tool.py translate input.srt -s en -t zh
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入文件 (.txt/.srt) | 必填 |
| `-s, --source` | 源语言 | en |
| `-t, --target` | 目标语言 | zh |
| `-m, --model` | 模型大小 | base |
| `--device` | 计算设备 (cuda/cpu) | 自动 |

### 5. TTS语音合成 `tts`

将文本合成为语音。

```bash
python video_tool.py tts text.txt -l chinese -r reference.wav
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入文件 (.txt/.srt) | 必填 |
| `-l, --language` | 语言 | chinese |
| `-m, --model` | 模型大小 (0.6B/1.7B) | 0.6B |
| `-r, --reference` | 参考音频(音色克隆) | - |
| `--device` | 计算设备 (cuda/cpu) | 自动 |

### 6. 合并音频 `merge`

将人声和背景音合并，生成最终配音音频。

```bash
# 使用默认输出文件名 (final_dubbed_zh.wav)
python video_tool.py merge -v vocals.wav -b background.wav

# 指定输出文件名
python video_tool.py merge -v vocals.wav -b background.wav -o final_dubbed_zh.wav

# 调整音量
python video_tool.py merge -v vocals.wav -b background.wav --vocals-vol 1.2 --background-vol 0.8
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-v, --vocals` | 人声文件 | 必填 |
| `-b, --background` | 背景文件 | 必填 |
| `-o, --output` | 输出文件路径 | final_dubbed_zh.wav |
| `--vocals-vol` | 人声音量倍数 | 1.0 |
| `--background-vol` | 背景音量倍数 | 0.8 |

### 7. 替换视频音频 `replace`

替换视频中的音轨。

```bash
python video_tool.py replace video.mp4 -a new_audio.wav
```

### 8. 生成静音视频 `silent`

移除视频中的音频，生成静音视频。

```bash
python video_tool.py silent video.mp4 -o silent.mp4
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入视频文件 | 必填 |
| `-o, --output` | 输出文件路径 | 自动生成 |

### 9. AI 配音 `dub`

完整的英文→中文配音流程。

```bash
python video_tool.py dub /data/gdot_En.mp4
```

---

## 完整 AI 配音流程

```
视频输入
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
│    英文语音转文字       │
│    输出: english.srt    │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 3. 翻译 (M2M100)        │
│    en → zh              │
│    输出: chinese.srt    │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 4. TTS合成 (Qwen3-TTS)  │
│    音色克隆 + 时间轴对齐 │
│    输出: tts_*.wav      │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 5. 音频处理 + 视频合成  │
│    输出: 中文配音.mp4   │
└─────────────────────────┘
```

---

## 输出文件

运行 `python video_tool.py dub video.mp4` 后生成：

```
output/视频名_zh_dubbed/
├── 视频名_中文配音.mp4    # 最终中文配音视频
├── 视频名_silent.mp4     # 静音视频（可用于其他音频合成）
├── chinese.srt            # 中文字幕文件
├── english.srt            # 英文字幕文件
├── final_dubbed_zh.wav    # 最终配音音频（人声+背景音）
├── merged_tts.wav         # TTS音频（时间轴对齐）
├── extracted.wav          # 提取的原始音频
├── separated/
│   ├── vocals.wav         # 分离的人声
│   └── background.wav     # 分离的背景音
└── tts_output/            # TTS合成文件
    └── tts_*.wav
```

---

## 前后对比示例

### 原始视频
- 视频: `gdot_En.mp4`
- 音频: 英文原声
- 时长: 8分40秒 (520.10秒)

### 处理后输出
- 视频: `gdot_En_中文配音.mp4`
- 音频: 中文AI配音（保留原视频画面）
- 时长: 8分40秒 (520.10秒)

### 音频处理详情

| 文件 | 说明 | 时长 | 状态 |
|------|------|------|------|
| `extracted.wav` | 提取的原始英文音频 | 520.10秒 | ✓ 已生成 |
| `merged_tts.wav` | TTS中文配音（时间轴对齐） | 520.10秒 | ✓ 已生成 |
| `final_dubbed_zh.wav` | 合成后的配音+背景音 | 520.10秒 | ✓ 已生成 |

### 单独使用各功能示例

```bash
# 假设输入视频为 data/gdot_En.mp4，输出目录为 output/gdot_En_zh_dubbed/

# 步骤1: 提取音频
python video_tool.py convert data/gdot_En.mp4 -o output/gdot_En_zh_dubbed/extracted.wav

# 步骤2: 人声分离（输出到 separated/ 子目录）
python video_tool.py separate data/gdot_En.mp4
# 生成: output/gdot_En_separated/separated/vocals.wav
#       output/gdot_En_separated/separated/background.wav

# 步骤3: 语音识别（基于人声）
python video_tool.py asr output/gdot_En_separated/vocals.wav -l en
# 生成: output/gdot_En_separated_asr/gdot_En_separated.srt

# 步骤4: 翻译（英文 -> 中文）
python video_tool.py translate output/gdot_En_separated_asr/gdot_En_separated.srt -s en -t zh
# 生成: output/translated/gdot_En_separated_zh.srt

# 步骤5: TTS合成（带音色克隆）
python video_tool.py tts output/translated/gdot_En_separated_zh.srt -l chinese -r output/gdot_En_separated/vocals.wav
# 生成: output/tts_output/merged_tts.wav

# 步骤6: 合并音频（TTS人声 + 背景音）
# 方式1: 使用默认输出文件名 final_dubbed_zh.wav
python video_tool.py merge -v output/tts_output/merged_tts.wav -b output/gdot_En_separated/separated/background.wav

# 方式2: 指定输出路径并调整音量
python video_tool.py merge -v output/tts_output/merged_tts.wav -b output/gdot_En_separated/separated/background.wav -o output/gdot_En_zh_dubbed/final_dubbed_zh.wav --vocals-vol 1.0 --background-vol 0.8

# 步骤7: 生成静音视频（移除原视频音轨）
python video_tool.py silent data/gdot_En.mp4 -o output/gdot_En_silent.mp4

# 步骤8: 替换视频音频（将配音音频合成到静音视频）
python video_tool.py replace output/gdot_En_silent.mp4 -a output/gdot_En_zh_dubbed/final_dubbed_zh.wav
```

---

## 项目结构

```
ai_video_translator/
├── video_tool.py          # 主脚本 (统一入口)
├── diagnose.py            # 系统诊断脚本
├── README.md              # 项目说明
├── requirements.txt       # 依赖列表
├── .gitignore             # Git 忽略规则
├── data/                  # 输入视频目录
│   └── (放置视频文件)
├── output/                # 输出目录
├── logs/                  # 日志目录
└── src/                   # 核心模块
    ├── __init__.py
    ├── config.py          # 配置管理
    ├── analyzer.py        # 媒体分析
    ├── extractor.py       # 音频提取
    ├── separator.py       # 人声分离
    ├── asr_module.py      # ASR 识别
    ├── translator_m2m100.py # 翻译模块
    ├── tts_qwen3.py       # TTS 合成
    ├── subtitle_handler.py # 字幕处理
    ├── merger.py          # 音频合并
    ├── video_processor.py # 视频合成
    ├── performance_config.py # 性能配置
    ├── memory_manager.py  # 显存管理
    ├── model_manager.py   # 模型管理
    └── splitter.py        # 音频分割
```

---

## 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| Python | 3.8+ | 3.10+ |
| FFmpeg | 4.0+ | 5.0+ |
| 磁盘空间 | 10GB+ | 20GB+ |
| 内存 | 8GB | 16GB+ |
| GPU 显存 | 4GB | 8GB+ |

---

## 性能配置

系统会自动检测硬件并优化配置：

- **GPU 检测**: 自动识别 CUDA / Apple Silicon
- **显存管理**: 动态监控显存，自动降级到 CPU
- **模型选择**: 根据显存大小选择合适的模型
- **并行优化**: GPU 模式强制串行避免冲突

### 模型规格

| 组件 | 模型 | 规格 |
|------|------|------|
| ASR | Whisper | small / medium |
| 翻译 | M2M100 | 418M |
| TTS | Qwen3-TTS | 0.6B / 1.7B |
| 人声分离 | Demucs | htdemucs |

## 依赖说明

核心依赖：

- **torch** - 深度学习框架
- **transformers** - Hugging Face 模型库
- **openai-whisper** - 语音识别
- **demucs** - 人声分离
- **soundfile** - 音频读写
- **pydub** - 音频处理

安装 TTS 库：

```bash
# 方式1: Qwen3-TTS (推荐)
pip install qwen-tts

# 方式2: Coqui TTS (可选)
pip install TTS
```

---

## 许可证

MIT License

