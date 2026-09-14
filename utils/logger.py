"""日志工具：同时输出到控制台与日志文件。"""

import logging
import os


def setup_logger(name, log_dir, filename=None):
    """创建一个同时写控制台和文件的 logger。

    Args:
        name: logger 名称（同时作为默认日志文件名）
        log_dir: 日志目录，不存在会自动创建
        filename: 日志文件名，默认 "{name}.log"

    Returns:
        logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)
    filename = filename or f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:  # 避免重复调用时重复添加 handler
        return logger

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(os.path.join(log_dir, filename), encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
