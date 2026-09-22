"""应用日志配置。"""

import logging


def configure_logging() -> None:
    """配置控制台日志；重复调用时不会重复添加处理器。"""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
