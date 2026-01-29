# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


class TrendPredictor:
    """趋势预测算法，支持多种时间颗粒度的数据预测

    该算法使用XGBoost回归器进行时间序列预测，支持不同时间颗粒度，
    并在数据量不足时自动降级到朴素预测方法。
    """
    MIN_DATA_DIC = {
        "DAY": 20, "WEEK": 14, "MONTH": 12, "QUARTER": 8, "YEAR": 5
    }  # 不同时间颗粒度最小数据量

    LAG_LIST_DIC = {
        "DAY": [1, 3, 7], "WEEK": [1, 2, 4],
        "MONTH": [1, 3, 6], "QUARTER": [1, 2, 4], "YEAR": [1, 3]
    }  # 滞后特征

    ROLLING_LIST_DIC = {
        "DAY": [3, 7], "WEEK": [2, 4],
        "MONTH": [3, 6], "QUARTER": [2, 4], "YEAR": [2, 3]
    }  # 滚动特征

    def __init__(self, data: list[float], freq: str, pred_length: int):
        """初始化趋势预测器

        Args:
            data: 历史数据列表
            freq: 时间频率，支持 DAY, WEEK, MONTH, QUARTER, YEAR
            pred_length: 预测长度
        """
        self.data = data
        self.freq = freq
        self.model = self._determine_model()
        self.pred_length = pred_length

    def _determine_model(self):
        """为了保证预测稳定性，根据数据量判断是否模型降级"""
        min_length = TrendPredictor.MIN_DATA_DIC.get(self.freq)
        if len(self.data) < min_length:
            return "NAIVE"
        else:
            return "XGB"

    def _preprocess_data(self):
        """模型训练样本构建"""
        series = pd.Series(self.data)
        df = pd.DataFrame({'y': series})
        df["delta"] = df["y"].diff()

        # 添加滞后特征
        for lag in TrendPredictor.LAG_LIST_DIC.get(self.freq):
            df[f"lag_{lag}"] = df["y"].shift(lag)
            df[f"delta_lag_{lag}"] = df["delta"].shift(lag)

        # 添加滚动特征
        for roll in TrendPredictor.ROLLING_LIST_DIC.get(self.freq):
            df[f"roll_mean_delta_{roll}"] = df["delta"].rolling(roll, min_periods=1).mean().shift(1)
            df[f"roll_std_delta_{roll}"] = df["delta"].rolling(roll, min_periods=1).std().shift(1)

        df_train = df.dropna().reset_index(drop=True)
        feature_cols = [c for c in df_train.columns if c not in ["y", "delta"]]
        x_train = df_train[feature_cols]
        y_train = df_train["delta"]
        return x_train, y_train

    def _construct_new_point(self):
        """构建预测点的特征"""
        y_values = np.array(self.data)
        delta_values = np.diff(y_values, prepend=y_values[0])
        x_pred = {}

        # 滞后特征
        for lag in TrendPredictor.LAG_LIST_DIC.get(self.freq):
            x_pred[f"lag_{lag}"] = y_values[-lag]
            x_pred[f"delta_lag_{lag}"] = delta_values[-lag]

        # 滚动特征
        for roll in TrendPredictor.ROLLING_LIST_DIC.get(self.freq):
            window = delta_values[-roll:]
            x_pred[f"roll_mean_delta_{roll}"] = np.mean(window)
            x_pred[f"roll_std_delta_{roll}"] = np.std(window)

        return pd.DataFrame([x_pred])

    def _predict_naively(self):
        """数据量不足时，调用降级预测算法"""
        steps = self.pred_length
        if len(self.data) < 2:
            return [self.data[-1]] * steps
        else:
            diffs = [self.data[i] - self.data[i - 1] for i in range(1, len(self.data))]
            mean_diff = sum(diffs) / len(diffs)
            last = self.data[-1]
            result = [last + mean_diff * i for i in range(1, steps + 1)]
            return result

    def _construct_model(self):
        """训练XGBoost预测模型"""
        train_x, train_y = self._preprocess_data()

        # 预测模型训练
        model = XGBRegressor(
            max_depth=5,
            learning_rate=0.03,
            n_estimators=100
        )
        model.fit(train_x, train_y)
        return model

    @staticmethod
    def _softplus(x):
        """数值稳定版softplus，避免大正值出现overflow警告"""
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def forecast(self):
        """滚动预测主方法"""
        if self.model == "NAIVE":
            return self._predict_naively()
        else:
            result = []
            model = self._construct_model()
            # 滚动预测
            for _ in range(0, self.pred_length):
                new_point = self._construct_new_point()
                pred_residual = model.predict(new_point)[0]
                pred_y = self.data[-1] + pred_residual
                self.data.append(pred_y)
                result.append(self._softplus(pred_y))
            return result