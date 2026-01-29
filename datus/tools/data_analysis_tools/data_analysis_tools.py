# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
from typing import Any, List, Optional
import json

from datus.tools.func_tool.base import FuncToolResult
from datus.tools.data_analysis_tools.trend_predictor import TrendPredictor


class DataAnalysisTools:
    """数据分析工具类，提供各种数据分析方法"""

    def __init__(self):
        """初始化预测工具"""
        pass

    def set_tool_context(self, tool_context):
        """设置工具上下文（如果需要）"""
        self.tool_context = tool_context

    def predict_trend(
        self,
        data: List[float],
        freq: str = "DAY",
        pred_length: int = 5
    ) -> FuncToolResult:
        """
        统一趋势预测：基于历史数据预测未来趋势值。

        This method：
        1. 分析历史数据序列的模式和规律
        2. 根据时间频率和预测长度计算未来趋势
        3. 生成带有置信区间的预测结果

        Args:
            data: 历史数据列表，只需输入值即可，如：[1,2,3,4,5,6]
            freq: 时间频率，支持 DAY, WEEK, MONTH, QUARTER, YEAR
            pred_length: 预测长度，表示要预测多少个时间单位

        Returns:
            TrendPredictionResult，包含：
            - model_type: 使用的预测模型类型
            - predictions: 预测的未来值列表
            - freq: 使用的时间频率
            - pred_length: 预测长度
            - input_data_length: 输入历史数据的长度

        Example:
            result = await tool.predict_trend(
                data=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                freq="DAY",
                pred_length=5
            )
            # 预测未来5天的数据趋势
        """
        try:
            # 验证输入参数
            if not data or len(data) < 2:
                return FuncToolResult(
                    success=0,
                    error="数据长度至少需要2个数据点",
                    result=None
                )

            valid_freqs = ["DAY", "WEEK", "MONTH", "QUARTER", "YEAR"]
            if freq not in valid_freqs:
                return FuncToolResult(
                    success=0,
                    error=f"无效的时间频率，支持的频率: {valid_freqs}",
                    result=None
                )

            if pred_length <= 0:
                return FuncToolResult(
                    success=0,
                    error="预测长度必须大于0",
                    result=None
                )

            # 创建预测器并执行预测
            predictor = TrendPredictor(data=data, freq=freq, pred_length=pred_length)
            predictions = predictor.forecast()

            # 准备返回结果
            result = {
                "model_type": predictor.model,
                "predictions": predictions,
                "freq": freq,
                "pred_length": pred_length,
                "input_data_length": len(data)
            }

            return FuncToolResult(
                success=1,
                error=None,
                result=result
            )

        except Exception as e:
            return FuncToolResult(
                success=0,
                error=f"预测过程中发生错误: {str(e)}",
                result=None
            )