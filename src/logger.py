#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志模块
为整个AI视频翻译项目提供统一的日志配置
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""

    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
    }
    RESET = '\033[0m'

    def __init__(self, fmt: str, datefmt: str = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        # 保存原始levelname
        orig_levelname = record.levelname

        if self.use_colors and sys.stdout.isatty():
            # 添加颜色
            color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{color}{record.levelname}{self.RESET}"

        result = super().format(record)

        # 恢复原始levelname
        record.levelname = orig_levelname

        return result


class LoggerManager:
    """日志管理器 - 单例模式"""

    _instance: Optional['LoggerManager'] = None
    _initialized: bool = False

    def __new__(cls) -> 'LoggerManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if LoggerManager._initialized:
            return

        self.log_dir: Path = Path("logs")
        self.log_level: int = logging.DEBUG
        self.console_level: int = logging.INFO
        self.max_bytes: int = 10 * 1024 * 1024  # 10MB
        self.backup_count: int = 5
        self.use_colors: bool = True
        self._log_file: Optional[Path] = None

        LoggerManager._initialized = True

    def setup(
        self,
        log_dir: str = "logs",
        log_level: int = logging.DEBUG,
        console_level: int = logging.INFO,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        use_colors: bool = True,
    ) -> 'LoggerManager':
        """配置日志管理器"""
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self.console_level = console_level
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.use_colors = use_colors

        # 创建日志目录
        self.log_dir.mkdir(parents=True, exist_ok=True)

        return self

    def get_log_file(self) -> Path:
        """获取当前日志文件路径"""
        if self._log_file is None:
            # 确保日志目录存在
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_file = self.log_dir / f"ai_video_translator_{timestamp}.log"
        return self._log_file

    def create_logger(self, name: str) -> logging.Logger:
        """为指定模块创建日志记录器"""
        logger = logging.getLogger(name)

        # 如果已经配置过，直接返回
        if logger.handlers:
            return logger

        logger.setLevel(self.log_level)
        logger.propagate = False

        # 文件处理器 - 使用RotatingFileHandler进行日志轮转
        log_file = self.get_log_file()
        fh = RotatingFileHandler(
            log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding="utf-8"
        )
        fh.setLevel(self.log_level)

        # 控制台处理器
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(self.console_level)

        # 格式化器
        file_formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )

        console_formatter = ColoredFormatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            "%H:%M:%S",
            use_colors=self.use_colors
        )

        fh.setFormatter(file_formatter)
        ch.setFormatter(console_formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger


# 全局日志管理器实例
_logger_manager = LoggerManager()


def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    use_colors: bool = True,
) -> LoggerManager:
    """
    配置全局日志系统

    Args:
        log_dir: 日志目录
        log_level: 文件日志级别
        console_level: 控制台日志级别
        max_bytes: 单个日志文件最大大小
        backup_count: 保留的备份文件数量
        use_colors: 是否在控制台使用颜色

    Returns:
        LoggerManager实例
    """
    return _logger_manager.setup(
        log_dir=log_dir,
        log_level=log_level,
        console_level=console_level,
        max_bytes=max_bytes,
        backup_count=backup_count,
        use_colors=use_colors,
    )


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志记录器

    Args:
        name: 日志记录器名称（通常使用__name__）

    Returns:
        配置好的Logger实例
    """
    return _logger_manager.create_logger(name)


def get_log_file() -> Path:
    """获取当前日志文件路径"""
    return _logger_manager.get_log_file()


class LogContext:
    """日志上下文管理器 - 用于记录代码块执行时间"""

    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.INFO):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = datetime.now().timestamp()
        self.logger.log(self.level, f"[{self.operation}] 开始")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = datetime.now().timestamp() - self.start_time
        if exc_type is None:
            self.logger.log(self.level, f"[{self.operation}] 完成 | 耗时: {elapsed:.2f}秒")
        else:
            self.logger.error(f"[{self.operation}] 失败 | 耗时: {elapsed:.2f}秒 | 错误: {exc_val}")
        return False


def log_execution(logger: logging.Logger, operation: str, level: int = logging.INFO):
    """
    装饰器/上下文管理器 - 记录函数执行时间

    用法:
        @log_execution(logger, "数据处理")
        def process_data():
            pass

        # 或作为上下文管理器
        with log_execution(logger, "数据处理"):
            process_data()
    """
    return LogContext(logger, operation, level)


# 便捷函数
def log_step(logger: logging.Logger, step: str, status: str, details: str = ""):
    """记录步骤日志"""
    msg = f"[{step}] {status}"
    if details:
        msg += f" | {details}"
    logger.info(msg)


def log_progress(logger: logging.Logger, current: int, total: int, operation: str = ""):
    """记录进度"""
    percent = (current / total * 100) if total > 0 else 0
    msg = f"进度: {current}/{total} ({percent:.1f}%)"
    if operation:
        msg = f"[{operation}] {msg}"
    logger.info(msg)
