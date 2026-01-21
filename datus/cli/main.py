#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
"""
Datus-CLI: 为数据工程师打造的AI驱动SQL命令行界面
CLI应用程序的主入口点
"""

import argparse

# 导入版本信息
from datus import __version__
# 导入REPL模式的CLI实现
from datus.cli.repl import DatusCLI
# 导入Windows策略设置工具
from datus.utils.async_utils import setup_windows_policy
# 导入数据库类型常量
from datus.utils.constants import DBType
# 导入日志配置工具
from datus.utils.loggings import configure_logging, get_logger

# 获取当前模块的日志记录器
logger = get_logger(__name__)


class ArgumentParser:
    """命令行参数解析器类

    负责解析CLI应用程序的所有命令行参数，包括数据库连接配置、
    日志设置、Web界面选项等
    """

    def __init__(self):
        """初始化参数解析器

        创建argparse.ArgumentParser实例并设置所有命令行参数
        """
        self.parser = argparse.ArgumentParser(description="Datus: AI驱动的SQL命令行界面")
        self._setup_arguments()

    def _setup_arguments(self):
        """设置所有命令行参数"""
        # 添加版本参数 -v/--version
        self.parser.add_argument("-v", "--version", action="version", version=f"Datus CLI {__version__}")

        # 数据库连接设置
        self.parser.add_argument(
            "--db_type",
            dest="db_type",
            choices=[DBType.SQLITE, DBType.SNOWFLAKE, DBType.DUCKDB],
            default=DBType.SQLITE,
            help="要连接的数据库类型",
        )
        self.parser.add_argument(
            "--db_path", dest="db_path", type=str, help="数据库文件路径 (适用于SQLite/DuckDB)"
        )

        # 通用设置
        self.parser.add_argument(
            "--history_file",
            dest="history_file",
            type=str,
            default=None,
            help="历史文件路径 (默认: {agent.home}/history)",
        )
        self.parser.add_argument(
            "--config",
            dest="config",
            type=str,
            help="配置文件路径 (默认: ./conf/agent.yml > {agent.home}/conf/agent.yml)",
        )
        self.parser.add_argument("--debug", action="store_true", help="启用调试日志")
        self.parser.add_argument("--no_color", dest="no_color", action="store_true", help="禁用彩色输出")
        # storage_path参数已废弃 - 数据路径现在固定在 {agent.home}/data

        self.parser.add_argument(
            "--namespace",
            type=str,
            help="数据库命名空间或基准测试命名空间",
        )

        self.parser.add_argument("--database", type=str, help="默认连接的数据库", default="")

        # LLM追踪设置
        self.parser.add_argument(
            "--save_llm_trace",
            action="store_true",
            help="启用将LLM输入/输出追踪保存到YAML文件",
        )

        # Web界面设置
        self.parser.add_argument(
            "--web",
            action="store_true",
            help="启动基于Web的Streamlit聊天机器人界面",
        )

        self.parser.add_argument(
            "--port",
            type=int,
            default=8501,
            help="Web界面端口 (默认: 8501)",
        )

        self.parser.add_argument(
            "--host",
            type=str,
            default="localhost",
            help="Web界面主机 (默认: localhost)",
        )

    def parse_args(self):
        return self.parser.parse_args()


class Application:
    """CLI应用程序主类

    负责协调整个CLI应用程序的执行流程，包括参数解析、
    日志配置和启动相应的界面(CLI或Web)
    """

    def __init__(self):
        """初始化应用程序

        创建ArgumentParser实例用于解析命令行参数
        """
        self.arg_parser = ArgumentParser()

    def run(self):
        """运行CLI应用程序的主要流程"""
        # 解析命令行参数
        args = self.arg_parser.parse_args()

        # 配置日志系统
        configure_logging(args.debug, console_output=False)

        # 检查是否提供了命名空间参数
        if not args.namespace:
            self.arg_parser.parser.print_help()
            return

        # 根据参数决定启动Web界面还是CLI界面
        if args.web:
            self._run_web_interface(args)
        else:
            # 启动REPL模式的CLI
            cli = DatusCLI(args)
            cli.run()

    def _run_web_interface(self, args):
        """启动Streamlit Web界面"""
        # 动态导入Web界面模块以避免依赖问题
        from datus.cli.web import run_web_interface

        # 启动Web界面
        run_web_interface(args)


def main():
    """控制台脚本的入口点

    这是CLI应用程序的main函数，创建Application实例并运行
    """
    app = Application()
    app.run()


if __name__ == "__main__":
    # Windows平台特定的策略设置
    setup_windows_policy()
    # 启动应用程序
    main()
