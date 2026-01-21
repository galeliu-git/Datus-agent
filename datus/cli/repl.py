# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Datus-CLI REPL (Read-Eval-Print Loop) 实现
此模块提供CLI的主要交互式Shell功能
"""

from __future__ import annotations

import sys
import threading
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# 导入prompt_toolkit用于构建交互式命令行界面
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style, merge_styles, style_from_pygments_cls
# 导入rich用于美化控制台输出
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from datus.agent.workflow_runner import WorkflowRunner

# 导入CLI工具函数
from datus.cli._cli_utils import prompt_input
# 导入各种命令处理器
from datus.cli.agent_commands import AgentCommands
from datus.cli.autocomplete import AtReferenceCompleter, CustomPygmentsStyle, CustomSqlLexer, SubagentCompleter
from datus.cli.bi_dashboard import BiDashboardCommands
from datus.cli.chat_commands import ChatCommands
from datus.cli.context_commands import ContextCommands
from datus.cli.metadata_commands import MetadataCommands
from datus.cli.sub_agent_commands import SubAgentCommands
# 导入配置管理
from datus.configuration.agent_config_loader import configuration_manager, load_agent_config
# 导入动作历史记录相关
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import SQLContext
# 导入数据库连接器
from datus.tools.db_tools import BaseSqlConnector
from datus.tools.db_tools.db_manager import db_manager_instance
# 导入常量
from datus.utils.constants import SYS_SUB_AGENTS, DBType, SQLType
# 导入异常处理设置
from datus.utils.exceptions import setup_exception_handler
# 导入日志记录器
from datus.utils.loggings import get_logger
# 导入SQL类型解析工具
from datus.utils.sql_utils import parse_sql_type

# 获取当前模块的日志记录器
logger = get_logger(__name__)


class CommandType(Enum):
    """用户输入的命令类型枚举

    定义了CLI支持的所有命令类型，用于命令解析和执行
    """

    SQL = "sql"  # 常规SQL语句
    TOOL = "tool"  # !命令 (工具/工作流)
    CONTEXT = "context"  # @命令 (上下文浏览器)
    CHAT = "chat"  # /命令 (聊天)
    INTERNAL = "internal"  # .命令 (CLI控制)
    EXIT = "exit"  # 退出命令 (exit/quit)


class DatusCLI:
    """Main REPL for the Datus CLI application."""

    def __init__(self, args):
        """使用给定的参数初始化CLI

        Args:
            args: 从命令行解析得到的参数对象
        """
        self.args = args
        # 创建控制台输出对象，用于美化输出
        self.console = Console(log_path=False)
        # 控制台列宽设置，用于智能表格显示
        self.console_column_width = 16
        # 当前选中的目录路径
        self.selected_catalog_path = ""
        # 是否为Streamlit模式(用于Web界面)
        self.streamlit_mode = False
        # 当前选中的目录数据
        self.selected_catalog_data = {}

        # 设置统一的异常处理器
        setup_exception_handler(
            console_logger=self.console.print, prefix_wrap_func=lambda x: f"[bold red]{x}[/bold red]"
        )
        # 数据库连接器，类型注解
        self.db_connector: BaseSqlConnector

        # AI智能体相关状态
        self.agent = None  # 智能体实例
        self.agent_initializing = False  # 是否正在初始化
        self.agent_ready = False  # 是否已准备就绪
        self._workflow_runner: WorkflowRunner | None = None  # 工作流运行器

        # 计划模式支持 (Plan Mode)
        self.plan_mode_active = False

        # 首先加载智能体配置以初始化路径管理器
        self.agent_config = load_agent_config(**vars(self.args))
        self.configuration_manager = configuration_manager()

        # 设置历史文件路径
        if args.history_file:
            history_file = Path(args.history_file).expanduser().resolve()
        else:
            from datus.utils.path_manager import get_path_manager

            history_file = get_path_manager().history_file_path()
        history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = FileHistory(str(history_file))
        # 自动补全器
        self.at_completer: AtReferenceCompleter
        # 初始化提示会话
        self._init_prompt_session()

        # 最后执行的SQL语句和结果
        self.last_sql = None
        self.last_result = None

        # 动作历史记录管理器，用于跟踪所有CLI操作
        self.actions = ActionHistoryManager()

        # 初始化CLI上下文用于状态管理
        from datus.cli.cli_context import CliContext

        self.cli_context = CliContext(
            current_db_name=getattr(args, "database", ""),
            current_catalog=getattr(args, "catalog", ""),
            current_schema=getattr(args, "schema", ""),
        )
        # 数据库管理器实例
        self.db_manager = db_manager_instance(self.agent_config.namespaces)

        # 从agentic_nodes初始化可用的子智能体(排除'chat')，并包含内置子智能体
        self.available_subagents = set(SYS_SUB_AGENTS)
        if hasattr(self.agent_config, "agentic_nodes") and self.agent_config.agentic_nodes:
            self.available_subagents.update(name for name in self.agent_config.agentic_nodes.keys() if name != "chat")

        # 在cli_context创建后初始化命令处理器
        self.agent_commands = AgentCommands(self, self.cli_context)  # 智能体命令处理器
        self.chat_commands = ChatCommands(self)  # 聊天命令处理器
        self.context_commands = ContextCommands(self)  # 上下文命令处理器
        self.metadata_commands = MetadataCommands(self)  # 元数据命令处理器
        self.sub_agent_commands = SubAgentCommands(self)  # 子智能体命令处理器
        self.bi_dashboard_commands = BiDashboardCommands(self)  # BI仪表板命令处理器

        # 可用命令字典 - 在处理器初始化后创建
        self.commands = {
            # 工具命令 (!前缀)
            "!sl": self.agent_commands.cmd_schema_linking,
            "!schema_linking": self.agent_commands.cmd_schema_linking,
            "!sm": self.agent_commands.cmd_search_metrics,
            "!search_metrics": self.agent_commands.cmd_search_metrics,
            "!sq": self.agent_commands.cmd_search_reference_sql,
            "!search_sql": self.agent_commands.cmd_search_reference_sql,
            "!save": self.agent_commands.cmd_save,
            "!bash": self._cmd_bash,
            # 上下文命令 (@前缀)
            "@catalog": self.context_commands.cmd_catalog,
            "@subject": self.context_commands.cmd_subject,
            # 内部命令 (.前缀)
            ".clear": self.chat_commands.cmd_clear_chat,
            ".chat_info": self.chat_commands.cmd_chat_info,
            ".compact": self.chat_commands.cmd_compact,
            ".sessions": self.chat_commands.cmd_list_sessions,
            ".databases": self.metadata_commands.cmd_list_databases,
            ".database": self.metadata_commands.cmd_switch_database,
            ".tables": self.metadata_commands.cmd_tables,
            ".schemas": self.metadata_commands.cmd_schemas,
            ".schema": self.metadata_commands.cmd_switch_schema,
            ".table_schema": self.metadata_commands.cmd_table_schema,
            ".indexes": self.metadata_commands.cmd_indexes,
            ".namespace": self._cmd_switch_namespace,
            ".subagent": self.sub_agent_commands.cmd,
            ".mcp": self._cmd_mcp,
            ".bootstrap-bi": self.bi_dashboard_commands.cmd,
            ".help": self._cmd_help,
            ".exit": self._cmd_exit,
            ".quit": self._cmd_exit,
        }

        # 在后台启动智能体初始化
        self._async_init_agent()
        # 初始化数据库连接
        self._init_connection()

    @property
    def workflow_runner(self) -> WorkflowRunner:
        """工作流运行器属性

        Returns:
            WorkflowRunner: 智能体的工作流运行器实例

        Raises:
            RuntimeError: 当智能体未初始化时抛出
        """
        if not self.check_agent_available():
            raise RuntimeError("智能体未初始化，无法创建工作流运行器。")
        if not self._workflow_runner:
            self._workflow_runner = self.agent.create_workflow_runner()
        return self._workflow_runner

    def _create_custom_key_bindings(self):
        """为REPL创建自定义键绑定"""
        kb = KeyBindings()

        @kb.add("tab")
        def _(event):
            """Tab键触发自动补全功能，不用于导航"""
            buffer = event.app.current_buffer

            if buffer.complete_state:
                # 如果菜单已经打开，则关闭它
                buffer.complete_next()
            else:
                # 如果菜单未完成，则触发补全
                buffer.start_completion(select_first=False)

        @kb.add("s-tab")
        def _(event):
            """Shift+Tab: 切换计划模式的开关"""
            self.plan_mode_active = not self.plan_mode_active

            # 清除当前输入缓冲区并强制退出当前提示
            buffer = event.app.current_buffer
            buffer.reset()

            # 强制提示符退出并重新启动新前缀
            # 这将导致主循环重新生成提示符
            buffer.validation_state = None
            event.app.exit()

            # 显示模式变更消息
            if self.plan_mode_active:
                self.console.print("[bold green]计划模式已激活![/]")
                self.console.print("[dim]输入您的规划任务并按Enter键生成计划[/]")
            else:
                self.console.print("[yellow]计划模式已停用[/]")

        @kb.add("enter")
        def _(event):
            """
            Enter键:
                如果补全菜单打开，应用高亮项目(如果有)或关闭菜单；否则执行
            """
            buffer = event.app.current_buffer

            if buffer.complete_state:
                # 如果有当前高亮的补全项，则应用它
                cs = buffer.complete_state
                comp = cs.current_completion
                if comp is not None:
                    buffer.apply_completion(comp)
                else:
                    # 没有高亮项目(例如select_first=False)。关闭菜单并按正常Enter处理
                    buffer.cancel_completion()
                    buffer.validate_and_handle()
                return

            # 当没有补全菜单时执行正常的Enter行为
            buffer.validate_and_handle()

        @kb.add("c-o")
        def _(event):
            """显示操作详情"""
            event.app.exit(result="_open_chat_sql_details")

        return kb

    def _get_prompt_text(self):
        """根据模式获取当前提示文本"""
        if self.plan_mode_active:
            return "[计划模式] Datus> "
        else:
            return "Datus> "

    def _update_prompt(self):
        """更新提示显示(在模式变更时调用)"""
        # 提示符将在主循环的下一次迭代中更新
        # 这是prompt_toolkit的PromptSession的限制
        # 要获得即时反馈，我们可以强制重绘，但会很复杂

    def _init_prompt_session(self):
        """初始化提示会话并设置自定义键绑定"""
        # 使用自定义键绑定设置提示会话
        self.session = PromptSession(
            history=self.history,
            auto_suggest=AutoSuggestFromHistory(),
            lexer=PygmentsLexer(CustomSqlLexer),
            completer=self.create_combined_completer(),
            multiline=False,
            key_bindings=self._create_custom_key_bindings(),
            enable_history_search=True,
            search_ignore_case=True,
            style=merge_styles(
                [
                    style_from_pygments_cls(CustomPygmentsStyle),
                    Style.from_dict(
                        {
                            "prompt": "ansigreen bold",
                        }
                    ),
                ]
            ),
            complete_while_typing=True,
        )

    # 创建组合补全器
    def create_combined_completer(self):
        """创建组合补全器: SubagentCompleter + AtReferenceCompleter + SqlCompleter"""
        from datus.cli.autocomplete import SQLCompleter

        sql_completer = SQLCompleter()
        self.at_completer = AtReferenceCompleter(self.agent_config)  # 路由器补全器
        subagent_completer = SubagentCompleter(self.agent_config)  # 子智能体补全器

        # 使用merge_completers来组合补全器
        from prompt_toolkit.completion import merge_completers

        return merge_completers(
            [
                subagent_completer,  # 子智能体补全器(最高优先级)
                self.at_completer,  # @引用补全器
                sql_completer,  # SQL关键字补全器(最低优先级)
            ]
        )

    def run(self):
        """运行REPL主循环"""
        self._print_welcome()

        while True:
            try:
                # 获取动态提示文本
                prompt_text = self._get_prompt_text()

                # 获取用户输入
                user_input_raw = self.session.prompt(
                    message=prompt_text,
                )
                if user_input_raw is None:
                    continue
                if user_input_raw == "_open_chat_sql_details":
                    if not self.streamlit_mode and self.chat_commands and self.chat_commands.last_actions:
                        from datus.cli.screen.action_display_app import ChatApp

                        app = ChatApp(self.chat_commands.last_actions)
                        app.run()
                    continue
                user_input = user_input_raw.strip()

                if not user_input:
                    continue

                # 解析并执行命令
                cmd_type, cmd, args = self._parse_command(user_input)
                if cmd_type == CommandType.EXIT:
                    return True

                # 根据类型执行命令
                if cmd_type == CommandType.SQL:
                    self._execute_sql(user_input)
                elif cmd_type == CommandType.TOOL:
                    self._execute_tool_command(cmd, args)
                elif cmd_type == CommandType.CONTEXT:
                    self._execute_context_command(cmd, args)
                elif cmd_type == CommandType.CHAT:
                    self._execute_chat_command(args, subagent_name=cmd)
                elif cmd_type == CommandType.INTERNAL:
                    self._execute_internal_command(cmd, args)

            except KeyboardInterrupt:
                continue
            except EOFError:
                return 0
            except Exception as e:
                # 检查这是否是退出事件(用于计划模式切换)
                if "exit" in str(e).lower() and "app" in str(e).lower():
                    # 这是Shift+Tab切换的预期行为，继续循环
                    continue
                logger.error(f"错误: {str(e)}")
                self.console.print(f"[bold red]错误:[/] {str(e)}")

    def _async_init_agent(self):
        """在后台线程中异步初始化智能体"""
        if self.agent_initializing or self.agent_ready:
            return

        # 在Streamlit模式下跳过后台初始化以避免向量数据库冲突
        if hasattr(self, "streamlit_mode") and self.streamlit_mode:
            return

        self.agent_initializing = True
        self.console.print("[dim]在后台初始化AI功能...[/]")

        # 在单独的线程中启动初始化
        thread = threading.Thread(target=self._background_init_agent)
        thread.daemon = True  # 守护线程会在主线程退出时退出
        thread.start()

    def _background_init_agent(self):
        """在后台线程中初始化智能体的函数"""
        try:
            # 基于CLI参数创建模拟args对象
            from argparse import Namespace

            agent_args = Namespace(
                temperature=0.7,
                top_p=0.9,
                max_tokens=8000,
                workflow="reflection",
                max_steps=20,
                debug=self.args.debug,
                load_cp=False,
                components=["metrics", "metadata", "table_lineage", "document"],
            )

            from datus.agent.agent import Agent

            self.agent = Agent(agent_args, self.agent_config)

            self.agent_ready = True
            self.agent_initializing = False

            self.agent_commands.update_agent_reference()
            self._pre_load_storage()
            self._workflow_runner = self.agent.create_workflow_runner()
            # self.console.print("[dim]智能体在后台成功初始化[/]")
        except Exception as e:
            self.console.print(f"[bold red]错误:[/]在后台初始化智能体失败: {str(e)}")
            logger.error(f"[bold red]在后台初始化智能体失败: {e}")
            self.agent_initializing = False
            self.agent = None

    def _pre_load_storage(self):
        """预加载向量数据库以避免不必要的打印"""
        if self.at_completer:
            self.at_completer.reload_data()

    def check_agent_available(self):
        """检查智能体是否可用，如果仍在初始化则通知用户

        Returns:
            bool: 智能体是否可用
        """
        if self.agent_ready and self.agent:
            return True
        elif self.agent_initializing:
            self.console.print(
                "[yellow]AI功能仍在后台初始化中。请稍后再试。[/]"
            )
            return False
        else:
            self.console.print("[bold red]错误:[/] AI功能不可用。智能体初始化失败。")
            return False

    def _cmd_list_namespaces(self):
        """列出所有可用的命名空间"""
        table = Table(show_header=True, header_style="bold green")
        table.add_column("命名空间")
        for namespace in self.agent_config.namespaces.keys():
            if self.agent_config.current_namespace == namespace:
                table.add_row(f"[bold green]{namespace}[/]")
            else:
                table.add_row(namespace)
        self.console.print(table)
        return

    def _cmd_mcp(self, args):
        """处理MCP(模型配置协议)相关命令"""
        from datus.cli.mcp_commands import MCPCommands

        MCPCommands(self).cmd_mcp(args)

    def _smart_display_table(
        self,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
    ) -> None:
        """智能表格显示，通过限制列数和截断内容来处理宽表格

        Args:
            data: 表示表格行的字典列表
            columns: 要显示的列，如果不提供，则显示所有列
        """
        if not data:
            self.console.print("[yellow]没有数据显示[/]")
            return

        if columns:
            all_columns_list = columns
        else:
            # 获取所有唯一的列名
            all_columns_list = []
            for row in data:
                all_columns_list.extend(list(row.keys()))
        # 根据终端宽度计算最大列数
        max_columns = max(4, self.console.width // self.console_column_width)

        # 智能列选择：根据终端宽度显示前端+后端+省略号
        if len(all_columns_list) > max_columns:
            show_back = max_columns // 2
            show_front = max_columns - show_back  # -1为省略号留出空间

            # 选择要显示的列
            front_columns = all_columns_list[:show_front]
            back_columns = all_columns_list[-show_back:] if show_back > 0 else []
            display_columns = front_columns + ["..."] + back_columns
        else:
            display_columns = all_columns_list

        # 根据列数计算动态列宽
        # 启用折叠后，我们可以使用更窄的列并在屏幕上容纳更多内容
        num_display_columns = len([col for col in display_columns if col != "..."])
        if num_display_columns <= 2:
            # 对于1-2列，使用中等宽度(内容会在需要时折叠)
            dynamic_column_width = max(25, self.console.width // max(2, num_display_columns) - 4)
        elif num_display_columns <= 4:
            # 对于3-4列，使用紧凑宽度
            dynamic_column_width = max(20, self.console.width // num_display_columns - 3)
        elif num_display_columns <= 8:
            # 对于5-8列，使用窄宽度(内容会在需要时折叠)
            dynamic_column_width = max(18, self.console.width // num_display_columns - 2)
        else:
            # 对于很多列，使用默认紧凑宽度
            dynamic_column_width = self.console_column_width

        table = Table(show_header=True, header_style="bold green")

        # 添加具有宽度约束和溢出折叠的列
        for col in display_columns:
            if col == "...":
                table.add_column(col, width=5, justify="center")
            else:
                # 使用动态列宽，并为长内容启用折叠
                table.add_column(col, width=dynamic_column_width, overflow="fold", no_wrap=False)

        # 添加具有截断内容的行
        for row in data:
            row_values: List[Any] = []
            for col in display_columns:
                if col == "...":
                    row_values.append("...")
                else:
                    row_value = row.get(col)
                    if isinstance(row_value, datetime):
                        row_value = row_value.strftime("%Y-%m-%d %H:%M:%S")
                    elif isinstance(row_value, date):
                        row_value = row_value.strftime("%Y-%m-%d")
                    else:
                        row_value = str(row_value)
                    row_values.append(row_value)
            table.add_row(*row_values)

        self.console.print(table)

    def reset_session(self):
        self.chat_commands.update_chat_node_tools()
        if self.at_completer:
            # Perhaps we should reload the data here.
            self.at_completer.reload_data()

    def _cmd_switch_namespace(self, args: str):
        if args.strip() == "":
            self._cmd_list_namespaces()
        elif self.agent_config.current_namespace == args.strip():
            self.console.print(
                (
                    f"[yellow]It's now under the namespace [bold]{self.agent_config.current_namespace}[/]"
                    " and doesn't need to be switched[/]"
                )
            )
            self._cmd_list_namespaces()
            return
        else:
            self.agent_config.current_namespace = args.strip()
            name, self.db_connector = self.db_manager.first_conn_with_name(self.agent_config.current_namespace)
            db_name = self.db_connector.database_name
            db_logic_name = name or self.agent_config.current_namespace
            self.cli_context.update_database_context(
                catalog=self.db_connector.catalog_name,
                db_name=db_name,
                schema=self.db_connector.schema_name,
                db_logic_name=db_logic_name,
            )
            self.reset_session()
            self.chat_commands.update_chat_node_tools()
            self.console.print(f"[bold green]Namespace changed to: {self.agent_config.current_namespace}[/]")

    def _parse_command(self, text: str) -> Tuple[CommandType, str, str]:
        """
        Parse the command and determine its type.

        Returns:
            Tuple containing (command_type, command, arguments)
        """
        text = text.strip()

        # Remove trailing semicolons (common in SQL)
        if text.endswith(";"):
            text = text[:-1].strip()

        # Exit commands
        if text.lower() in [".exit", ".quit", "exit", "quit"]:
            return CommandType.EXIT, "", ""

        # Tool commands (!prefix)
        if text.startswith("!"):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            return CommandType.TOOL, cmd, args

        # Context commands (@prefix)
        if text.startswith("@"):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            return CommandType.CONTEXT, cmd, args

        # Chat commands (/prefix)
        if text.startswith("/"):
            message = text[1:].strip()
            parts = message.split(maxsplit=1)
            if len(parts) > 1:
                # Check if first part is a valid subagent
                potential_subagent = parts[0]
                if potential_subagent in self.available_subagents:
                    # Sub-agent syntax: /subagent_name message
                    subagent_name = potential_subagent
                    actual_message = parts[1]
                    return CommandType.CHAT, subagent_name, actual_message
                else:
                    # Regular chat: /message (first part is not a valid subagent)
                    return CommandType.CHAT, "", message
            else:
                # Regular chat: /message
                return CommandType.CHAT, "", message

        # Internal commands (.prefix)
        if text.startswith("."):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            return CommandType.INTERNAL, cmd, args

        # Determine if text is SQL or chat using parse_sql_type
        try:
            # Get current database dialect from agent_config.db_type (set from current namespace)
            dialect = self.agent_config.db_type if self.agent_config.db_type else "snowflake"
            sql_type = parse_sql_type(text, dialect)

            # If parse_sql_type returns a valid SQL type (not UNKNOWN), treat as SQL
            if sql_type != SQLType.UNKNOWN:
                return CommandType.SQL, "", text
            else:
                return CommandType.CHAT, "", text.strip()
        except Exception:
            # If any exception occurs, treat as chat
            return CommandType.CHAT, "", text.strip()

    def _execute_sql(self, sql: str, system: bool = False):
        """Execute a SQL query and display results."""
        logger.debug(f"Executing SQL query: '{sql}'")

        # Create action for SQL execution
        sql_action = ActionHistory.create_action(
            role=ActionRole.USER,
            action_type="sql_execution",
            messages=f"Executing SQL: {sql[:100]}..." if len(sql) > 100 else f"Executing SQL: {sql}",
            input_data={"sql": sql, "system": system},
            status=ActionStatus.PROCESSING,
        )
        self.actions.add_action(sql_action)

        try:
            if not self.db_connector:
                error_msg = "No database connection. Please initialize a connection first."
                self.console.print(f"[bold red]Error:[/] {error_msg}")

                # Update action with error
                self.actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.FAILED,
                    output={"error": error_msg},
                    messages=f"SQL execution failed: {error_msg}",
                )
                return

            # Execute the query
            import time

            start_time = time.time()
            result = self.db_connector.execute(input_params={"sql_query": sql}, result_format="arrow")
            end_time = time.time()
            exec_time = end_time - start_time

            if not result:
                error_msg = "No result from the query."
                self.console.print(f"[bold red]Error:[/] {error_msg}")

                # Update action with error
                self.actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.FAILED,
                    output={"error": error_msg},
                    messages=f"SQL execution failed: {error_msg}",
                )
                return

            # Save for later reference
            self.last_sql = sql
            self.last_result = result

            # Display results and update action
            if result.success:
                if not hasattr(result.sql_return, "column_names"):
                    if result.row_count is not None and result.row_count > 0:
                        # Update action with success
                        self.actions.update_action_by_id(
                            sql_action.action_id,
                            status=ActionStatus.SUCCESS,
                            output={
                                "row_count": result.row_count,
                                "execution_time": exec_time,
                                "success": True,
                            },
                            messages=f"SQL executed successfully: {result.row_count} rows in {exec_time:.2f}s",
                        )
                        self.console.print(f"[dim]Update {result.sql_return} rows in {exec_time:.2f} seconds[/]")
                    elif result.sql_return:
                        self.console.print(f"[dim]SQL execution successful in {exec_time:.2f} seconds[/]")
                        # Update action with success
                        self.actions.update_action_by_id(
                            sql_action.action_id,
                            status=ActionStatus.SUCCESS,
                            output={
                                "row_count": 0,
                                "execution_time": exec_time,
                                "success": True,
                            },
                            messages=f"SQL executed successfully in {exec_time:.2f}s",
                        )
                    else:
                        error_msg = (
                            f"Query execution failed - received string instead of Arrow data:"
                            f" {result.error or 'Unknown error'}"
                        )
                        self.console.print(f"[bold red]Error:[/] {error_msg}")

                        # Update action with error
                        self.actions.update_action_by_id(
                            sql_action.action_id,
                            status=ActionStatus.FAILED,
                            output={"error": error_msg, "result_type_error": True},
                            messages=f"Result format error: {error_msg}",
                        )
                    return
                # Convert Arrow data to list of dictionaries for smart display
                rows = result.sql_return.to_pylist()
                self._smart_display_table(data=rows, columns=result.sql_return.column_names)

                row_count = result.sql_return.num_rows
                self.console.print(f"[dim]Returned {row_count} rows in {exec_time:.2f} seconds[/]")

                # Update action with success
                self.actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.SUCCESS,
                    output={
                        "row_count": row_count,
                        "execution_time": exec_time,
                        "columns": result.sql_return.column_names,
                        "success": True,
                    },
                    messages=f"SQL executed successfully: {row_count} rows in {exec_time:.2f}s",
                )
                workflow_ready = self._workflow_runner and self._workflow_runner.workflow_ready
                if not system and workflow_ready:  # Add to sql context if not system command
                    new_record = SQLContext(
                        sql_query=sql,
                        sql_return=str(result.sql_return),
                        row_count=row_count,
                        explanation=f"Manual sql: Returned {row_count} rows in {exec_time:.2f} seconds",
                    )
                    self.workflow_runner.workflow.context.sql_contexts.append(new_record)

            else:
                error_msg = result.error or "Unknown SQL error"
                self.console.print(f"[bold red]SQL Error:[/] {error_msg}")

                # Update action with SQL error
                self.actions.update_action_by_id(
                    sql_action.action_id,
                    status=ActionStatus.FAILED,
                    output={"error": error_msg, "sql_error": True},
                    messages=f"SQL error: {error_msg}",
                )
                workflow_ready = self._workflow_runner and self._workflow_runner.workflow_ready
                if not system and workflow_ready:  # Add to sql context if not system command
                    new_record = SQLContext(
                        sql_query=sql,
                        sql_return=str(result.error) if result.error else "Unknown error",
                        row_count=0,
                        explanation="Manual sql",
                    )
                    self._workflow_runner.workflow.context.sql_contexts.append(new_record)
        except Exception as e:
            logger.error(f"SQL execution error: {str(e)}")
            self.console.print(f"[bold red]Error:[/] {str(e)}")

            # Update action with exception
            self.actions.update_action_by_id(
                sql_action.action_id,
                status=ActionStatus.FAILED,
                output={"error": str(e), "exception": True},
                messages=f"SQL execution exception: {str(e)}",
            )

    def _execute_tool_command(self, cmd: str, args: str):
        """Execute a tool command (! prefix)."""
        if cmd in self.commands:
            self.commands[cmd](args)
        else:
            self.console.print(f"[bold red]Unknown command:[/] {cmd}")

    def _execute_context_command(self, cmd: str, args: str):
        """Execute a context command (@ prefix)."""
        if cmd in self.commands:
            self.commands[cmd](args)
        else:
            self.console.print(f"[bold red]Unknown command:[/] {cmd}")

    def _execute_chat_command(self, message: str, subagent_name: str = None):
        """Execute a chat command (/ prefix) using ChatAgenticNode."""
        self.chat_commands.execute_chat_command(message, plan_mode=self.plan_mode_active, subagent_name=subagent_name)

    def _execute_internal_command(self, cmd: str, args: str):
        """Execute an internal command (. prefix)."""
        logger.debug(f"Executing internal command: '{cmd}' with args: '{args}'")
        if cmd in self.commands:
            self.commands[cmd](args)
        else:
            self.console.print(f"[bold red]Unknown command:[/] {cmd}")

    def _wait_for_agent_available(self, max_attempts=5, delay=1):
        """Wait for the agent to become available, with timeout."""
        if self.check_agent_available():
            return True

        self.console.print("[yellow]Waiting for the agent to initialize...[/]")

        import time

        for _ in range(max_attempts):
            time.sleep(delay)
            if self.check_agent_available():
                return True

        self.console.print("[bold red]Agent initialization timed out. Try again later.[/]")
        return False

    def _cmd_bash(self, args: str):
        """Execute a bash command."""
        # Define a whitelist of allowed commands
        whitelist = ["pwd", "ls", "cat", "head", "tail", "echo"]

        if not args.strip():
            self.console.print("[yellow]Please provide a bash command.[/]")
            return

        # Parse the command to check against whitelist
        cmd_parts = args.split()
        base_cmd = cmd_parts[0]

        if base_cmd not in whitelist:
            self.console.print(
                f"[bold red]Security:[/] Command '{base_cmd}' not in whitelist. Allowed: {', '.join(whitelist)}"
            )
            return

        try:
            # Execute the command
            import subprocess

            result = subprocess.run(args, shell=True, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                if result.stdout:
                    self.console.print(result.stdout)
            else:
                self.console.print(f"[bold red]Command failed with code {result.returncode}:[/]\n{result.stderr}")

        except subprocess.TimeoutExpired:
            self.console.print("[bold red]Error:[/] Command timed out after 10 seconds.")
        except Exception as e:
            self.console.print(f"[bold red]Error:[/] {str(e)}")

    def _cmd_help(self, args: str):
        """Display help information with aligned command explanations."""
        CMD_WIDTH = 30
        lines = []
        lines.append("[bold green]Datus-CLI Help[/]\n")
        lines.append("[bold]SQL Commands:[/]")
        lines.append(f"    {'<sql>':<{CMD_WIDTH}}Execute SQL query directly\n")

        lines.append("[bold]Tool Commands (! prefix):[/]")
        tool_cmds = [
            # ("!run <query>", "Run a natural language query with live workflow status display"),
            ("!sl/!schema_linking", "Schema linking: show list of recommended tables and values"),
            ("!sm/!search_metrics", "Use natural language to search for corresponding metrics"),
            ("!sq/!search_sql", "Use natural language to search for reference SQL"),
            # ("!gen", "Generate SQL, optionally with table constraints"),
            # ("!fix <description>", "Fix the last SQL query"),
            ("!save", "Save the last result to a file"),
            ("!bash <command>", "Execute a bash command (limited to safe commands)"),
            # remove this when sub agent is ready
            # ("!reason", "Run SQL reasoning with streaming output"),
            # ("!compare", "Compare SQL results with streaming output"),
        ]
        for cmd, desc in tool_cmds:
            lines.append(f"    {cmd:<{CMD_WIDTH}}{desc}")
        lines.append("")

        lines.append("[bold]Context Commands (@ prefix):[/]")
        context_cmds = [
            ("@catalog", "Display database catalog"),
            ("@subject", "Display Semantic Model, Metrics etc."),
        ]
        for cmd, desc in context_cmds:
            lines.append(f"    {cmd:<{CMD_WIDTH}}{desc}")
        lines.append("")

        lines.append("[bold]Chat Commands (/ prefix):[/]")
        chat_cmds = [
            ("/<message>", "Chat with the AI assistant"),
        ]
        for cmd, desc in chat_cmds:
            lines.append(f"    {cmd:<{CMD_WIDTH}}{desc}")
        lines.append("")

        lines.append("[bold]Internal Commands (. prefix):[/]")
        internal_cmds = [
            (".help", "Display this help message"),
            (".exit, .quit", "Exit the CLI"),
            (".clear", "Clear console and chat session"),
            (".chat_info", "Show current chat session information"),
            (".compact", "Compact chat session by summarizing conversation history"),
            (".sessions", "List all stored SQLite sessions with detailed information"),
            (".bootstrap_bi", "Extract BI dashboard assets to assemble sub-agent context"),
            (".databases", "List all databases"),
            (".database database_name", "Switch current database"),
            (".tables", "List all tables"),
            (".schemas", "List all schemas or show detailed schema information"),
            (".schema schema_name", "Switch current schema"),
            (".table_schema table_name", "Show table field details"),
            (".indexes table_name", "Show indexes for a table"),
            (".namespace namespace", "Switch current namespace"),
            (".mcp", "Manage MCP (Model Configuration Protocol) servers"),
            ("     .mcp list", "List all MCP servers"),
            (
                "     .mcp add --transport \\[stdio/sse/http] <name> <command> \\[args1 args2 ...]",
                "Add a new MCP server configuration",
            ),
            ("     .mcp remove <name>", "Remove an MCP server configuration"),
            ("     .mcp check <name>", "Check connectivity to an MCP server"),
            ("     .mcp call <server.tool> \\[params]", "Call a tool on an MCP server"),
            ("     .mcp filter", "Manage tool filters for MCP servers"),
            (
                "       .mcp filter set <server> \\[--allowed tool1,tool2] "
                + "\\[--blocked tool3,tool4] \\[--enabled true/false]",
                "Set tool filter",
            ),
            ("       .mcp filter get <server>", "Get current tool filter configuration"),
            ("       .mcp filter remove <server>", "Remove tool filter configuration"),
        ]
        for cmd, desc in internal_cmds:
            lines.append(f"    {cmd:<{CMD_WIDTH}}{desc}")
        help_text = "\n".join(lines)
        self.console.print(help_text)

    def _cmd_exit(self, args: str):
        """Exit the CLI."""
        if self.db_connector:
            try:
                # Close the connection
                self.db_connector.close()
            except Exception as e:
                logger.warning(f"Database connection closed failed, reason:{e}")
        sys.exit(0)

    def catalogs_callback(self, selected_path: str = "", selected_data: Optional[Dict[str, Any]] = None):
        if not selected_path:
            return
        self.selected_catalog_path = selected_path
        self.selected_catalog_data = selected_data

    def _print_welcome(self):
        """Print the welcome message."""
        welcome_text = """
[bold green]Datus[/] - [bold]AI-powered SQL command-line interface[/]
Type '.help' for a list of commands or '.exit' to quit.
"""
        self.console.print(welcome_text)

        namespace = getattr(self.args, "namespace", "")
        # TODO use default namespace if not set
        if namespace:
            self.console.print(f"Namespace [bold green]{namespace}[/] selected")
        else:
            self.console.print("[yellow]Warning: No namespace selected, please use .namespace to select a namespace[/]")
        # Display connection info
        if self.db_connector:
            db_info = f"Connected to [bold green]{self.agent_config.db_type}[/]"
            if self.cli_context.current_db_name:
                db_info += f" using database [bold]{self.cli_context.current_db_name}[/]"

            self.console.print(db_info)

            # Show CLI context summary
            context_summary = self.cli_context.get_context_summary()
            if context_summary != "No context available":
                self.console.print(f"[dim]Context: {context_summary}[/]")

            self.console.print("Type SQL statements or use ! @ . commands to interact.")
        else:
            self.console.print("[yellow]Warning: No database connection initialized.[/]")

    def prompt_input(self, message: str, default: str = "", choices: list = None, multiline: bool = False):
        """
        Unified input method using prompt_toolkit to avoid conflicts with rich.Prompt.ask().

        Args:
            message: The prompt message to display
            default: Default value if user presses Enter without input
            choices: List of valid choices (validates input)
            multiline: Whether to allow multiline input

        Returns:
            User input string or default value
        """
        return prompt_input(
            self.console, message, default=default, choices=choices, multiline=multiline, style=self.session.style
        )

    def _init_connection(self):
        """Initialize database connection."""
        current_namespace = self.agent_config.current_namespace
        if not self.cli_context.current_db_name:
            db_name, self.db_connector = self.db_manager.first_conn_with_name(current_namespace)
            if self.db_connector.dialect in (DBType.SQLITE, DBType.DUCKDB):
                self.cli_context.update_database_context(db_name=self.db_connector.database_name, db_logic_name=db_name)
            else:
                self.cli_context.update_database_context(
                    catalog=self.db_connector.catalog_name,
                    db_name=self.db_connector.database_name,
                    db_logic_name=db_name or self.db_connector.database_name or self.agent_config.current_namespace,
                )
        else:
            self.db_connector = self.db_manager.get_conn(current_namespace, self.cli_context.current_db_name)
            if self.db_connector.dialect in (DBType.SQLITE, DBType.DUCKDB):
                self.cli_context.update_database_context(
                    db_name=self.db_connector.database_name, db_logic_name=self.cli_context.current_db_name
                )
        if not self.db_connector:
            self.console.print("[bold red]Error:[/] No database connection.")
            return

        # Test the connection, ff there is an exception, it will be handled by unified exception handling.
        connection_result = self.db_connector.test_connection()
        logger.debug(f"Connection test result: {connection_result}")
