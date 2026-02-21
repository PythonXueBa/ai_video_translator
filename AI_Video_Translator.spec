# -*- mode: python ; coding: utf-8 -*-

import sys
import os

block_cipher = None

# 添加数据文件
added_files = [
    ('src', 'src'),
    ('requirements.txt', '.'),
    ('README.md', '.'),
]

# 隐藏导入
hidden_imports = [
    'torch',
    'torchaudio',
    'transformers',
    'whisper',
    'demucs',
    'soundfile',
    'pydub',
    'numpy',
    'scipy',
    'psutil',
    'tqdm',
    'qwen_tts',
    'sentencepiece',
    'sacremoses',
    'einops',
    'librosa',
]

a = Analysis(
    ['video_tool.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AI_Video_Translator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
