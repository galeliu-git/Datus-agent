# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Web界面的聊天执行和流式处理

处理：
- 流式聊天执行
- 操作格式化显示
- SQL和响应提取
"""

import asyncio
from typing import List, Optional, Tuple

import structlog

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

logger = structlog.get_logger(__name__)


class ChatExecutor:
    """执行带有流式支持的聊天命令"""

    def __init__(self):
        """初始化聊天执行器"""
        self.last_actions = []  # 存储最后一次执行的操作

    def execute_chat_stream(self, user_message: str, cli, current_subagent: Optional[str] = None):
        """执行带有流式支持的聊天命令 - 重用chat_commands逻辑

        Args:
            user_message: 用户输入的聊天消息
            cli: DatusCLI实例
            current_subagent: 当前子智能体名称（可选）

        Yields:
            str: 流式返回的格式化消息
        """
        # 检查CLI和聊天命令是否已初始化
        if not cli or not cli.chat_commands:
            yield "错误：请先加载配置！"
            return

        try:
            # 步骤1：使用chat_commands逻辑获取节点和输入
            # 解析@上下文，提取表、指标和SQL相关信息
            at_tables, at_metrics, at_sqls = cli.at_completer.parse_at_context(user_message)

            # 步骤2：重用chat_commands的节点管理逻辑
            # 判断是否需要创建新节点（根据子智能体变化）
            need_new_node = cli.chat_commands._should_create_new_node(current_subagent)

            # 步骤3：在Web模式下禁用压缩以避免阻塞
            if need_new_node:
                # 创建新的聊天节点
                current_node = cli.chat_commands._create_new_node(current_subagent)
                # 更新当前节点和子智能体名称
                cli.chat_commands.current_node = current_node
                cli.chat_commands.current_subagent_name = current_subagent if current_subagent else None
                # 如果没有指定子智能体，将其设为聊天节点
                if not current_subagent:
                    cli.chat_commands.chat_node = current_node
            else:
                # 重用现有节点
                current_node = cli.chat_commands.current_node

            # 步骤4：使用chat_commands的共享方法创建节点输入
            node_input, _ = cli.chat_commands.create_node_input(
                user_message, current_node, at_tables, at_metrics, at_sqls, plan_mode=False
            )

            # 步骤5：流式执行并去重
            incremental_actions = []  # 累积的操作列表
            seen_thinking_content = set()  # 跟踪唯一的思考内容（不带前缀）
            last_message = None  # 跟踪上一条消息以避免连续重复

            async def collect_actions():
                """从流中收集所有操作"""
                nonlocal last_message

                # 设置当前节点的输入
                current_node.input = node_input
                # 异步流式执行节点操作
                async for action in current_node.execute_stream(cli.actions):
                    incremental_actions.append(action)
                    # 格式化为流式消息
                    formatted = self.format_action_for_stream(action)

                    # 跳过空消息
                    if not formatted:
                        continue

                    # 去重：跳过与上一条消息相同的内容
                    if formatted == last_message:
                        continue

                    # 对于思考消息，检查去除表情符号和前缀后的内容
                    if formatted.startswith("💭Thinking:"):
                        # 提取实际内容（去掉"💭Thinking: "前缀）
                        thinking_content = formatted[11:].strip()  # 移除"💭Thinking: "

                        # 如果之前已经见过这个确切的思考内容，则跳过
                        if thinking_content in seen_thinking_content:
                            continue

                        seen_thinking_content.add(thinking_content)

                    last_message = formatted
                    yield formatted

            # 步骤6：使用适当的事件循环处理执行异步生成器
            loop = asyncio.new_event_loop()

            try:
                # 设置当前节点的输入
                current_node.input = node_input

                async def run_stream():
                    """包装器，迭代异步生成器直到完成"""
                    try:
                        # 异步流式执行节点操作
                        async for action in current_node.execute_stream(cli.actions):
                            # 跳过正在处理中的工具操作
                            if action.role == ActionRole.TOOL and action.status == ActionStatus.PROCESSING:
                                continue
                            incremental_actions.append(action)
                            yield action
                        # 存储所有累积的操作供调用者访问
                        self.last_actions = incremental_actions
                    except StopAsyncIteration:
                        pass

                async_gen = run_stream()
                # 循环获取异步生成器的结果
                while True:
                    try:
                        result = loop.run_until_complete(async_gen.__anext__())
                        yield result
                    except StopAsyncIteration:
                        break

            finally:
                # 清理：关闭事件循环
                loop.close()

        except Exception as e:
            # 步骤7：错误处理
            logger.exception(f"执行错误: {e}")
            yield f"错误: {str(e)}"

    def format_action_for_stream(self, action: ActionHistory) -> str:
        """格式化ActionHistory以便流式显示详细信息

        Args:
            action: 要格式化的操作历史对象

        Returns:
            str: 格式化后的显示字符串
        """
        if action.role == ActionRole.TOOL:
            function_name = action.function_name() or "未知"

            # 提取输入参数用于显示
            input_preview = ""
            if action.input and isinstance(action.input, dict):
                # 显示关键参数（限制前100个字符）
                params = {k: v for k, v in action.input.items() if k != "function_name"}
                if params:
                    param_str = str(params)[:100]
                    if len(str(params)) > 100:
                        param_str += "..."
                    input_preview = f" ({param_str})"

            if action.status == ActionStatus.SUCCESS:
                # 为成功的工具显示输出预览
                output_preview = ""
                if action.output:
                    if isinstance(action.output, dict):
                        # 显示第一个键值对或长度
                        if "result" in action.output:
                            result_str = str(action.output["result"])[:80]
                            output_preview = (
                                f" → {result_str}..." if len(str(action.output["result"])) > 80 else f" → {result_str}"
                            )
                        elif len(action.output) > 0:
                            first_key = list(action.output.keys())[0]
                            output_preview = f" → {first_key}: ..."
                    else:
                        output_str = str(action.output)[:80]
                        output_preview = f" → {output_str}..." if len(str(action.output)) > 80 else f" → {output_str}"

                return f"✓工具调用: {function_name}{input_preview}{output_preview}"
            elif action.status == ActionStatus.PROCESSING:
                return f"⟳工具调用: {function_name}{input_preview}..."
            else:
                return f"✗工具调用: {function_name}{input_preview}"
        elif action.role == ActionRole.ASSISTANT and action.messages:
            # 显示LLM思考过程及简要预览
            message = action.messages.strip()

            # 移除由openai_compatible.py添加的"Thinking: "前缀
            if message.startswith("Thinking: "):
                message = message[10:].strip()  # 移除"Thinking: "

            # 跳过空消息或非常通用的消息
            if not message or message.lower() in ["thinking...", "processing...", ""]:
                return ""

            # 截断长消息
            if len(message) > 100:
                message = message[:100] + "..."

            return f"💭思考: {message}"
        return ""

    def extract_sql_and_response(self, actions: List[ActionHistory], cli) -> Tuple[Optional[str], Optional[str]]:
        """使用现有逻辑从操作中提取SQL和清理响应

        Args:
            actions: 操作历史列表
            cli: DatusCLI实例

        Returns:
            Tuple[Optional[str], Optional[str]]: 提取的SQL和响应元组
        """
        if not actions:
            return None, None

        # 获取最后一个操作
        final_action = actions[-1]
        # 检查最后一个操作是否成功且有输出
        if not (
            final_action.output
            and isinstance(final_action.output, dict)
            and final_action.status == ActionStatus.SUCCESS
        ):
            return None, None

        # 提取SQL和响应
        sql = final_action.output.get("sql")
        response = final_action.output.get("response")

        # 处理None响应
        if response is None:
            return sql, None

        # 处理字典响应（已解析）
        if isinstance(response, dict):
            return sql, response.get("raw_output", str(response))

        # 仅处理字符串响应
        if not isinstance(response, str):
            return sql, str(response)

        # 使用ChatCommands提取SQL和输出
        extracted_sql, extracted_output = None, response
        if cli and cli.chat_commands:
            extracted_sql, extracted_output = cli.chat_commands._extract_sql_and_output_from_content(response)
            sql = sql or extracted_sql

        # 确定清理后的输出
        if sql:
            return sql, extracted_output or response

        if isinstance(extracted_output, dict):
            return None, extracted_output.get("raw_output", str(extracted_output))

        # Try to parse response as Python literal (only for strings)
        try:
            import ast

            response_dict = ast.literal_eval(response)
            if isinstance(response_dict, dict):
                return None, response_dict.get("raw_output", response)
        except (ValueError, SyntaxError):
            pass

        return None, response
