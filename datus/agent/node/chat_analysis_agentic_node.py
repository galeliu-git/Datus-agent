# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
支持预测工具的聊天数据分析智能节点实现

该模块提供了 ChatAgenticNode 的专门实现，包含
用于对话数据分析的预测和趋势分析功能。
"""
from typing import AsyncGenerator, Optional, override

from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.chat_analysis_agentic_node_models import ChatAnalysisNodeInput, ChatAnalysisNodeResult
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool import ContextSearchTools, DBFuncTool, trans_to_function_tool
from datus.tools.data_analysis_tools.data_analysis_tools import DataAnalysisTools
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ChatAnalysisAgenticNode(ChatAgenticNode):
    """
    支持数据库、文件系统和预测工具的聊天专用智能节点

    该节点提供灵活的数据分析聊天功能，包含：
    - 基于命名空间的数据库 MCP 服务器选择
    - 默认文件系统 MCP 服务器
    - 趋势预测和分析工具
    - 流式响应生成
    - 基于会话的对话管理
    """

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[ChatAnalysisNodeInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
    ):
        """
        将 ChatAnalysisAgenticNode 初始化为专门的 ChatAgenticNode

        Args:
            node_id: 节点唯一标识符
            description: 人类可读的节点描述
            node_type: 节点类型（应为 'chat_analysis'）
            input_data: 聊天分析输入数据
            agent_config: 智能体配置
            tools: 工具列表（将在 setup_tools 中填充）
        """

        # 使用 node_name="chat_analysis" 调用父类构造函数
        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools,
        )

        logger.debug(
            f"ChatAnalysisAgenticNode initialized: {self.agent_config.current_namespace} {self.agent_config.current_database}"
        )

    def setup_input(self, workflow: Workflow) -> dict:
        """
        从工作流上下文设置聊天分析输入

        使用任务和上下文数据中的用户消息创建 ChatAnalysisNodeInput

        Args:
            workflow: 包含上下文和任务的工作流实例

        Returns:
            包含成功状态和消息的字典
        """
        # 如果任务指定了不同的数据库，更新数据库连接
        task_database = workflow.task.database_name
        if task_database and self.db_func_tool and task_database != self.db_func_tool.connector.database_name:
            logger.debug(
                f"根据工作流任务，将数据库连接从 '{self.db_func_tool.connector.database_name}' "
                f"更新为 '{task_database}'"
            )
            self._update_database_connection(task_database)

        # 从工作流元数据中读取 plan_mode
        plan_mode = workflow.metadata.get("plan_mode", False)
        auto_execute_plan = workflow.metadata.get("auto_execute_plan", False)

        # 如果尚未设置，创建 ChatAnalysisNodeInput
        if not self.input:
            self.input = ChatAnalysisNodeInput(
                user_message=workflow.task.task,
                external_knowledge=workflow.task.external_knowledge,
                catalog=workflow.task.catalog_name,
                database=workflow.task.database_name,
                db_schema=workflow.task.schema_name,
                schemas=workflow.context.table_schemas,
                metrics=workflow.context.metrics,
                reference_sql=None,
                plan_mode=plan_mode,
                auto_execute_plan=auto_execute_plan,
                prompt_version=self.node_config.get("prompt_version"),
            )
        else:
            # 使用工作流数据更新现有输入
            self.input.user_message = workflow.task.task
            self.input.external_knowledge = workflow.task.external_knowledge
            self.input.catalog = workflow.task.catalog_name
            self.input.database = workflow.task.database_name
            self.input.db_schema = workflow.task.schema_name
            self.input.schemas = workflow.context.table_schemas
            self.input.metrics = workflow.context.metrics

        return {"success": True, "message": "已从工作流准备聊天分析输入"}

    def update_context(self, workflow: Workflow) -> dict:
        """
        使用聊天分析结果更新工作流上下文

        如果结果中存在 SQL 和分析结果，则将它们存储到工作流上下文中

        Args:
            workflow: 要更新的工作流实例

        Returns:
            包含成功状态和消息的字典
        """
        if not self.result:
            return {"success": False, "message": "没有结果来更新上下文"}

        result = self.result

        try:
            # 如果存在，存储 SQL（从 ChatAgenticNode 继承）
            if hasattr(result, "sql") and result.sql:
                from datus.schemas.node_models import SQLContext

                # 如果可用，从响应中提取 SQL 结果
                sql_result = ""
                if hasattr(result, "response") and result.response:
                    _, sql_result = self._extract_sql_and_output_from_response({"content": result.response})
                    sql_result = sql_result or ""

                new_record = SQLContext(
                    sql_query=result.sql,
                    explanation=result.response if hasattr(result, "response") else "",
                    sql_return=sql_result,
                )
                workflow.context.sql_contexts.append(new_record)

            # 如果存在，存储分析结果
            if hasattr(result, "analysis_results") and result.analysis_results:
                from datus.schemas.node_models import AnalysisContext

                analysis_record = AnalysisContext(
                    analysis_type=getattr(result, "analysis_type", "general"),
                    results=result.analysis_results,
                    insights=getattr(result, "data_insights", []),
                )
                workflow.context.analysis_contexts.append(analysis_record)

            # 如果存在，存储预测数据
            if hasattr(result, "prediction_data") and result.prediction_data:
                from datus.schemas.node_models import PredictionContext

                prediction_record = PredictionContext(
                    model_type=result.prediction_data.get("model_type", "unknown"),
                    predictions=result.prediction_data.get("predictions", []),
                    parameters=result.prediction_data.get("parameters", {}),
                )
                workflow.context.prediction_contexts.append(prediction_record)

            return {"success": True, "message": "已更新聊天分析上下文"}
        except Exception as e:
            logger.error(f"更新聊天分析上下文失败: {e}")
            return {"success": False, "message": str(e)}

    @override
    def setup_tools(self):
        """
        使用默认数据库连接初始化所有工具，包括预测工具

        扩展父类 setup_tools，包含趋势预测工具
        """
        # 调用父类设置以初始化数据库和其他标准工具
        super().setup_tools()

        # 将数据分析工具添加到工具列表
        self._setup_data_analysis_tools()


    def _setup_data_analysis_tools(self):
        """Setup 数据分析工具"""
        try:
            from datus.tools.data_analysis_tools.data_analysis_tools import DataAnalysisTools

            self.data_analysis_tool = DataAnalysisTools()

            # 将数据分析工具转换为函数工具
            # 添加预测分析工具
            if not hasattr(self, 'tools') or self.tools is None:
                self.tools = []

            self.tools.append(trans_to_function_tool(self.data_analysis_tool.predict_trend))
            logger.debug(f"已添加数据分析工具: predict_trend")

            # Setup 语义工具（包含维度归因分析等）
            self._setup_semantic_tools()
        except Exception as e:
            logger.error(f"设置数据分析工具失败: {e}")

    def _setup_semantic_tools(self):
        """Setup 语义工具（包含维度归因分析等）"""
        try:
            # 创建语义工具实例，支持从节点配置中获取 adapter_type
            from datus.tools.func_tool.semantic_tools import SemanticTools

            adapter_type = self.node_config.get("adapter_type", "metricflow") if hasattr(self, 'node_config') else None

            semantic_tools = SemanticTools(
                agent_config=self.agent_config,
                sub_agent_name="chat_analysis_system",
                adapter_type=adapter_type
            )

            # 获取所有可用的语义工具
            semantic_tools_list = semantic_tools.available_tools()
            if semantic_tools_list:
                # 将所有语义工具添加到工具列表
                self.tools.extend(semantic_tools_list)
                tool_names = [tool.name for tool in semantic_tools_list]
                logger.debug(f"已添加 {len(semantic_tools_list)} 个语义工具: {tool_names}")
            else:
                logger.debug("未找到语义层适配器，跳过语义工具设置")

        except Exception as e:
            logger.error(f"设置语义工具失败: {e}")