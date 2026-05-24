"""评估模块: 数据解析、预测、绘图"""
from .data import parse_xml, build_features
from .predict import (sliding_window_predict, find_event_windows,
                      predict_window, interactive_predict,
                      get_control_vector, get_context_features,
                      CONTROL_DIM, CONTEXT_DIM)
from .plot import (plot_results, plot_event_window, plot_full_patient_timeline)


def compute_metrics(predictions, targets):
    """计算评估指标"""
    import numpy as np
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    step_rmse = np.sqrt(np.mean((predictions - targets) ** 2, axis=0))
    step_mae = np.mean(np.abs(predictions - targets), axis=0)
    within_20pct = np.abs(predictions - targets) / np.maximum(targets, 1) < 0.20
    pct_within_20 = np.mean(within_20pct) * 100
    within_15_15 = (np.abs(predictions - targets) <= 15) | (
        np.abs(predictions - targets) / np.maximum(targets, 1) <= 0.15
    )
    pct_zone_ab = np.mean(within_15_15) * 100
    mard = np.mean(np.abs(predictions - targets) / np.maximum(targets, 1)) * 100
    return {
        'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MARD': mard,
        '% within 20%': pct_within_20, '% Clarke A+B': pct_zone_ab,
        'step_RMSE': step_rmse, 'step_MAE': step_mae,
    }
