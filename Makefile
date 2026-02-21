# AI Video Translator - Makefile

.PHONY: help install install-gpu test clean build build-windows build-linux build-macos

help:
	@echo "AI Video Translator - 构建工具"
	@echo ""
	@echo "可用命令:"
	@echo "  make install       - 安装依赖"
	@echo "  make install-gpu   - 安装 GPU 版本依赖"
	@echo "  make test          - 运行测试"
	@echo "  make build         - 构建安装包"
	@echo "  make build-windows - 构建 Windows 可执行文件"
	@echo "  make build-linux   - 构建 Linux 安装包"
	@echo "  make build-macos   - 构建 macOS 安装包"
	@echo "  make clean         - 清理构建文件"

install:
	pip install -r requirements.txt
	pip install qwen-tts

install-gpu:
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	pip install -r requirements.txt
	pip install qwen-tts

test:
	python video_tool.py test

clean:
	rm -rf build dist *.spec
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf *.egg-info
	rm -rf output/*
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

build-windows:
	@echo "构建 Windows 可执行文件..."
	pyinstaller AI_Video_Translator.spec
	@echo "构建完成: dist/AI_Video_Translator.exe"

build-linux:
	@echo "构建 Linux 安装包..."
	python setup.py sdist bdist_wheel
	@echo "构建完成: dist/"

build-macos:
	@echo "构建 macOS 应用..."
	pyinstaller --name="AI_Video_Translator" \
		--onefile \
		--console \
		--add-data "src:src" \
		video_tool.py
	@echo "构建完成: dist/AI_Video_Translator"

build: clean build-linux
	@echo "构建完成"

# 开发模式安装
dev:
	pip install -e .

# 发布到 PyPI
publish:
	python setup.py sdist bdist_wheel
	twine upload dist/*
