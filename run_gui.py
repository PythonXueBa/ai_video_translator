#!/usr/bin/env python3
"""
AI Video Translator - GUI启动器
直接运行图形界面
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui_tool import main

if __name__ == "__main__":
    main()
