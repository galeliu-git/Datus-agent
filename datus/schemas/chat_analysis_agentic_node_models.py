# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
聊天分析智能节点的架构模型

该模块定义了 ChatAnalysisAgenticNode 的输入和输出模型，
为基于聊天的数据分析交互（包含预测工具）提供结构化验证。
"""

from typing import Optional, List, Dict, Any

from pydantic import AliasChoices, Field

from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.node_models import Metric, ReferenceSql, TableSchema


class ChatAnalysisNodeInput(BaseInput):
    """
    ChatAnalysisAgenticNode 交互的输入模型

    扩展了 ChatNodeInput，增加了用于数据分析和预测的额外字段
    """

    user_message: str = Field(..., description="用户的数据分析聊天消息输入")
    catalog: Optional[str] = Field(default=None, description="用于上下文的数据库目录")
    database: Optional[str] = Field(default=None, description="用于上下文的数据库名称")
    db_schema: Optional[str] = Field(default=None, description="用于上下文的数据库架构")
    max_turns: int = Field(default=30, description="每次交互的最大对话轮数")
    external_knowledge: Optional[str] = Field(default="", description="外部知识")
    workspace_root: Optional[str] = Field(default=None, description="文件系统 MCP 服务器的根目录路径")
    prompt_version: Optional[str] = Field(default=None, description="提示词模板版本")
    schemas: Optional[list[TableSchema]] = Field(default=None, description="要使用的架构列表")
    metrics: Optional[list[Metric]] = Field(default=None, description="要使用的指标列表")
    reference_sql: Optional[list[ReferenceSql]] = Field(
        default=None,
        description="参考 SQL 片段，用于重用/调整",
        validation_alias=AliasChoices("reference_sql", "historical_sql"),
    )
    plan_mode: bool = Field(default=False, description="是否为计划模式交互")
    auto_execute_plan: bool = Field(
        default=False, description="是否在没有用户确认的情况下自动执行计划（用于工作流/基准测试）"
    )
    # 分析专用新字段
    analysis_type: Optional[str] = Field(
        default="general",
        description="分析类型：general, trend, forecast, statistical, comparative"
    )
    data_source: Optional[str] = Field(
        default=None,
        description="分析数据源（表名、查询结果或手动输入）"
    )
    prediction_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="预测工具参数（freq, pred_length 等）"
    )

    class Config:
        populate_by_name = True


class ChatAnalysisNodeResult(BaseResult):
    """
    ChatAnalysisAgenticNode 交互的结果模型

    扩展了 ChatNodeResult，增加了用于分析结果的额外字段
    """

    response: str = Field(..., description="AI 助手的响应")
    sql: Optional[str] = Field(default=None, description="生成的或在响应中引用的 SQL 查询")
    tokens_used: int = Field(default=0, description="此交互中使用的总令牌数")
    # 分析专用新字段
    analysis_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="数据分析的结构化结果"
    )
    prediction_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="趋势预测工具的预测结果"
    )
    data_insights: Optional[List[str]] = Field(
        default=None,
        description="分析的关键洞察"
    )
