import logging
import os
from datetime import datetime

class LoggerGenerator:
    """
    通用日志生成器
    传入日志目录路径，返回一个配置好的 logger 实例
    """
    # 用于缓存已创建的 logger，避免重复配置
    _logger_cache = {}

    @staticmethod
    def get_logger(log_dir, name=None, console_output=True):
        """
        获取一个配置好的 logger

        :param log_dir: 日志文件夹路径
        :param name: logger 名称（默认为调用模块名）
        :param console_output: 是否同时输出到控制台
        :return: logging.Logger 实例
        """
        # 如果没有指定 name，使用调用者的模块名
        if name is None:
            import inspect
            caller_frame = inspect.currentframe().f_back
            name = caller_frame.f_globals.get("__name__", "unknown")

        # 检查缓存，避免重复添加 handler
        if name in LoggerGenerator._logger_cache:
            return LoggerGenerator._logger_cache[name]

        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)

        # 创建 logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # 捕获所有级别
        logger.propagate = False  # 防止向上传播，避免重复日志

        # 避免重复添加 handler
        if logger.hasHandlers():
            logger.handlers.clear()

        # 日志文件名：按天分割+精确区分
        log_filename = os.path.join(
            log_dir,
            f"app_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        )

        # 文件处理器 - 记录所有级别日志到文件
        file_handler = logging.FileHandler(log_filename, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.DEBUG)

        # 控制台处理器（可选）
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)  # 控制台只显示 INFO 及以上

        # 格式化器
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        if console_output:
            console_handler.setFormatter(formatter)

        # 添加处理器
        logger.addHandler(file_handler)
        if console_output:
            logger.addHandler(console_handler)

        # 缓存 logger
        LoggerGenerator._logger_cache[name] = logger

        return logger


# 使用示例
if __name__ == "__main__":
    # 指定日志目录
    log_directory = "./logs"

    # 获取 logger
    logger = LoggerGenerator.get_logger(log_directory, name="model", console_output=True)

    # 写日志（所有级别都会写入文件）
    logger.debug("这是 debug 信息")
    logger.info("这是 info 信息")
    logger.warning("这是 warning 信息")
    logger.error("这是 error 信息")
    logger.critical("这是 critical 信息")