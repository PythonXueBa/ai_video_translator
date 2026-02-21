#!/usr/bin/env python3
"""
AI Video Translator - GUI工具
提供图形界面选择功能
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import os
import sys
import threading
import queue


class AIVideoTranslatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Video Translator - 智能视频翻译")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        # 设置样式
        self.style = ttk.Style()
        self.style.configure('Title.TLabel', font=('Microsoft YaHei', 16, 'bold'))
        self.style.configure('Subtitle.TLabel', font=('Microsoft YaHei', 10))
        self.style.configure('Action.TButton', font=('Microsoft YaHei', 11))

        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)

        self.create_widgets()
        self.output_queue = queue.Queue()
        self.check_queue()

    def create_widgets(self):
        # 标题
        title_label = ttk.Label(
            self.main_frame,
            text="AI Video Translator",
            style='Title.TLabel'
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 5))

        subtitle_label = ttk.Label(
            self.main_frame,
            text="基于 Qwen3-TTS 的高质量多语言视频翻译与配音",
            style='Subtitle.TLabel'
        )
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))

        # 分隔线
        ttk.Separator(self.main_frame, orient='horizontal').grid(
            row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )

        # 视频文件选择
        ttk.Label(self.main_frame, text="视频文件:").grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        self.video_path = tk.StringVar()
        self.video_entry = ttk.Entry(
            self.main_frame, textvariable=self.video_path, width=50
        )
        self.video_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(
            self.main_frame, text="浏览...", command=self.browse_video
        ).grid(row=3, column=2, sticky=tk.W)

        # 源语言选择
        ttk.Label(self.main_frame, text="源语言:").grid(
            row=4, column=0, sticky=tk.W, pady=5
        )
        self.source_lang = ttk.Combobox(
            self.main_frame,
            values=[
                "en - 英文",
                "zh - 中文",
                "ja - 日文",
                "ko - 韩文",
                "es - 西班牙文",
                "fr - 法文",
                "de - 德文",
                "ru - 俄文",
                "pt - 葡萄牙文",
                "it - 意大利文",
                "ar - 阿拉伯文",
                "hi - 印地文",
                "vi - 越南文",
                "th - 泰文",
                "id - 印尼文"
            ],
            width=20,
            state="readonly"
        )
        self.source_lang.set("en - 英文")
        self.source_lang.grid(row=4, column=1, sticky=tk.W, padx=5)

        # 目标语言选择
        ttk.Label(self.main_frame, text="目标语言:").grid(
            row=5, column=0, sticky=tk.W, pady=5
        )
        self.target_lang = ttk.Combobox(
            self.main_frame,
            values=[
                "zh - 中文",
                "en - 英文",
                "ja - 日文",
                "ko - 韩文",
                "es - 西班牙文",
                "fr - 法文",
                "de - 德文",
                "ru - 俄文",
                "pt - 葡萄牙文",
                "it - 意大利文",
                "ar - 阿拉伯文",
                "hi - 印地文",
                "vi - 越南文",
                "th - 泰文",
                "id - 印尼文"
            ],
            width=20,
            state="readonly"
        )
        self.target_lang.set("zh - 中文")
        self.target_lang.grid(row=5, column=1, sticky=tk.W, padx=5)

        # 时间范围
        ttk.Label(self.main_frame, text="时间范围:").grid(
            row=6, column=0, sticky=tk.W, pady=5
        )
        time_frame = ttk.Frame(self.main_frame)
        time_frame.grid(row=6, column=1, sticky=tk.W, padx=5)

        ttk.Label(time_frame, text="开始(秒):").pack(side=tk.LEFT)
        self.start_time = ttk.Entry(time_frame, width=8)
        self.start_time.insert(0, "0")
        self.start_time.pack(side=tk.LEFT, padx=(5, 15))

        ttk.Label(time_frame, text="时长(秒):").pack(side=tk.LEFT)
        self.duration = ttk.Entry(time_frame, width=8)
        self.duration.insert(0, "0")
        self.duration.pack(side=tk.LEFT, padx=5)
        ttk.Label(time_frame, text="(0=完整视频)").pack(side=tk.LEFT, padx=(5, 0))

        # 选项
        options_frame = ttk.LabelFrame(self.main_frame, text="选项", padding="10")
        options_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        self.voice_clone = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="启用音色克隆",
            variable=self.voice_clone
        ).pack(side=tk.LEFT, padx=10)

        self.speed_adjust = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame,
            text="启用语速调节",
            variable=self.speed_adjust
        ).pack(side=tk.LEFT, padx=10)

        # 分隔线
        ttk.Separator(self.main_frame, orient='horizontal').grid(
            row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )

        # 快速功能按钮
        ttk.Label(self.main_frame, text="快速功能:").grid(
            row=9, column=0, sticky=tk.W, pady=5
        )

        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.grid(row=9, column=1, columnspan=2, sticky=tk.W, pady=5)

        ttk.Button(
            btn_frame,
            text="系统测试",
            command=self.run_test
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="查看帮助",
            command=self.show_help
        ).pack(side=tk.LEFT, padx=5)

        # 主操作按钮
        action_frame = ttk.Frame(self.main_frame)
        action_frame.grid(row=10, column=0, columnspan=3, pady=20)

        self.run_btn = ttk.Button(
            action_frame,
            text="开始 AI 配音",
            command=self.run_dubbing,
            style='Action.TButton'
        )
        self.run_btn.pack()

        # 进度条
        self.progress = ttk.Progressbar(
            self.main_frame,
            mode='indeterminate',
            length=400
        )
        self.progress.grid(row=11, column=0, columnspan=3, pady=10)
        self.progress.grid_remove()

        # 状态标签
        self.status_var = tk.StringVar(value="就绪")
        self.status_label = ttk.Label(
            self.main_frame,
            textvariable=self.status_var,
            foreground="blue"
        )
        self.status_label.grid(row=12, column=0, columnspan=3, pady=5)

        # 输出日志
        ttk.Label(self.main_frame, text="输出日志:").grid(
            row=13, column=0, sticky=tk.W, pady=(10, 5)
        )

        self.log_text = scrolledtext.ScrolledText(
            self.main_frame,
            height=15,
            wrap=tk.WORD,
            state='disabled'
        )
        self.log_text.grid(row=14, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        self.main_frame.rowconfigure(14, weight=1)

        # 清空日志按钮
        ttk.Button(
            self.main_frame,
            text="清空日志",
            command=self.clear_log
        ).grid(row=15, column=0, sticky=tk.W, pady=5)

    def browse_video(self):
        filename = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            self.video_path.set(filename)

    def get_lang_code(self, lang_str):
        """从语言字符串中提取语言代码"""
        return lang_str.split(" - ")[0]

    def log(self, message):
        """添加日志"""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')

    def clear_log(self):
        """清空日志"""
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')

    def run_command(self, cmd, description):
        """在后台线程运行命令"""
        def run():
            try:
                self.output_queue.put(("status", f"正在{description}..."))
                self.output_queue.put(("progress", True))

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='ignore'
                )

                for line in process.stdout:
                    self.output_queue.put(("log", line.strip()))

                process.wait()

                if process.returncode == 0:
                    self.output_queue.put(("status", f"{description}完成！"))
                else:
                    self.output_queue.put(("status", f"{description}失败！"))

            except Exception as e:
                self.output_queue.put(("log", f"错误: {str(e)}"))
                self.output_queue.put(("status", f"{description}出错！"))
            finally:
                self.output_queue.put(("progress", False))
                self.output_queue.put(("done", None))

        threading.Thread(target=run, daemon=True).start()

    def check_queue(self):
        """检查队列更新GUI"""
        try:
            while True:
                msg_type, msg = self.output_queue.get_nowait()

                if msg_type == "log":
                    self.log(msg)
                elif msg_type == "status":
                    self.status_var.set(msg)
                elif msg_type == "progress":
                    if msg:
                        self.progress.grid()
                        self.progress.start()
                        self.run_btn.configure(state='disabled')
                    else:
                        self.progress.stop()
                        self.progress.grid_remove()
                        self.run_btn.configure(state='normal')
                elif msg_type == "done":
                    messagebox.showinfo("完成", self.status_var.get())

        except queue.Empty:
            pass

        self.root.after(100, self.check_queue)

    def run_dubbing(self):
        """运行AI配音"""
        video = self.video_path.get()
        if not video:
            # 使用默认视频
            video = "data/SpongeBob SquarePants_en.mp4"
            if not os.path.exists(video):
                messagebox.showerror("错误", "请选择视频文件或放置默认视频到 data/ 目录")
                return

        source_lang = self.get_lang_code(self.source_lang.get())
        target_lang = self.get_lang_code(self.target_lang.get())
        start_time = self.start_time.get()
        duration = self.duration.get()

        cmd = [
            sys.executable, "video_tool.py", "dub", video,
            "--source-lang", source_lang,
            "--target-lang", target_lang,
            "--start-time", start_time,
            "--duration", duration
        ]

        if not self.voice_clone.get():
            cmd.append("--no-voice-clone")
        if not self.speed_adjust.get():
            cmd.append("--no-speed-adjust")

        self.log(f"执行命令: {' '.join(cmd)}")
        self.run_command(cmd, "AI配音")

    def run_test(self):
        """运行系统测试"""
        cmd = [sys.executable, "video_tool.py", "test"]
        self.log("开始系统测试...")
        self.run_command(cmd, "系统测试")

    def show_help(self):
        """显示帮助"""
        help_text = """AI Video Translator 使用帮助

1. 选择视频文件 - 点击"浏览..."选择要翻译的视频
2. 选择源语言和目标语言
3. 设置时间范围（可选，0表示完整视频）
4. 点击"开始 AI 配音"

输出文件将保存在 output/ 目录中

支持的视频格式: MP4, AVI, MKV, MOV, WMV, FLV
支持的语言: 中文、英文、日文、韩文、法文、德文、西班牙文等18种语言

注意事项:
- 首次运行需要下载模型文件，请保持网络连接
- 处理时间较长，请耐心等待
- 建议使用 GPU 加速处理
"""
        messagebox.showinfo("帮助", help_text)


def main():
    root = tk.Tk()
    app = AIVideoTranslatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
