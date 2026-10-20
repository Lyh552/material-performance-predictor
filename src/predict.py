"""唯一对外推理接口：加载每个目标的 GPR 模型并返回预测区间。"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import FEATURE_COLS, MODEL_PATHS, TARGET_COLS


def _predict_one(params: dict[str, Any], artifact: dict[str, Any]) -> dict[str, float]:
    """使用一个目标的已保存模型生成点预测和名义 90% 区间。"""
    feature_cols = list(artifact.get("feature_cols", FEATURE_COLS))
    if feature_cols != FEATURE_COLS:
        raise ValueError("模型输入结构已过期，请运行 pipeline.py 重新训练双变量模型。")
    missing = [column for column in feature_cols if column not in params]
    if missing:
        raise ValueError(f"缺少推理参数: {missing}")
    input_values = {column: float(params[column]) for column in feature_cols}
    if not np.isfinite(list(input_values.values())).all():
        raise ValueError("推理参数必须是有限数值。")
    for column, bounds in artifact.get("feature_bounds", {}).items():
        if not bounds[0] <= input_values[column] <= bounds[1]:
            raise ValueError(f"{column}超出训练范围 {bounds}，当前不支持外推。")
    row = pd.DataFrame([input_values], columns=feature_cols)
    model = artifact["model"]
    if artifact.get("model_type", "gpr") == "gpr":
        value, std = model.predict(row, return_std=True)
        interval_z = float(artifact.get("interval_z", 1.6448536269514722))
        spread = float(np.asarray(std).reshape(-1)[0]) * interval_z
    else:
        value = model.predict(row)
        spread = 0.0
    center = float(np.asarray(value).reshape(-1)[0])
    if not np.isfinite([center, spread]).all() or spread < 0:
        raise ValueError("模型生成了无效预测值或预测区间。")
    result = {"value": center, "lower": center - spread, "upper": center + spread}
    if not np.isfinite(list(result.values())).all():
        raise ValueError("预测结果超出可表示范围。")
    return result


def _load_target_artifacts() -> dict[str, dict[str, Any]]:
    """从配置路径加载全部目标模型，不依赖当前工作目录。"""
    from src.version_store import runtime_bundle
    bundle = runtime_bundle()
    artifacts: dict[str, dict[str, Any]] = {}
    for target_col in TARGET_COLS:
        model_path = Path(bundle["model_paths"][target_col])
        artifacts[target_col] = joblib.load(model_path)
    return artifacts


def predict(
    params: dict,
    *,
    model_artifact: dict | None = None,
) -> dict[str, dict[str, float]]:
    """返回所有目标的预测结果；界面、评估和其他调用方均使用此接口。"""
    private_path = params.get("_model_path")
    if private_path is not None:
        artifact = joblib.load(Path(private_path))
        target_col = artifact.get("target_col")
        if not target_col:
            raise ValueError("折内模型缺少 target_col，无法确定输出归属。")
        return {target_col: _predict_one(params, artifact)}

    if model_artifact is None:
        artifacts = _load_target_artifacts()
    elif "model" in model_artifact:
        target_col = model_artifact.get("target_col")
        if not target_col:
            raise ValueError("模型 artifact 缺少 target_col。")
        # 评估阶段会传入单个目标的折内模型，直接返回该目标即可。
        return {target_col: _predict_one(params, model_artifact)}
    else:
        artifacts = model_artifact

    results: dict[str, dict[str, float]] = {}
    for target_col in TARGET_COLS:
        if target_col not in artifacts:
            raise ValueError(f"缺少目标模型：{target_col}")
        results[target_col] = _predict_one(params, artifacts[target_col])
    return results


if __name__ == "__main__":
    print(predict({"铝相对剂量": 11.0, "反应温度": 180.0}))
