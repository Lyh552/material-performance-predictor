"""页面只读验证适配层：不训练、不选择模型、不改写实验数据。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_COLS, TARGET_COLS


def check_source(source, data, models, metadata):
    """拒绝把被改写的源数据或其他训练版本与当前模型混用。"""
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != metadata["data_sha256"]:
        raise ValueError("训练源文件已变化，请先完成独立审核与重训；当前页面停止预测。")
    if len(data) != metadata["n_samples"]:
        raise ValueError("页面数据行数与训练记录不一致，已停止预测。")
    keys = ("data_sha256", "n_samples", "feature_cols", "feature_bounds", "code_sha256", "model_settings")
    for target in TARGET_COLS:
        if models[target].get("target_col") != target or any(
            models[target].get(key) != metadata.get(key) for key in keys
        ):
            raise ValueError("模型与训练记录不一致，请恢复同一版本文件后再使用。")


def load_validation(path, metadata, data):
    """校验主指标并由逐行折外预测复算，损坏或错版报告不展示数字。"""
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if result["metadata"] != metadata or metadata["primary_validation"] != "condition":
        raise ValueError("验证报告与当前训练版本不一致。")
    n_conditions = len(data[FEATURE_COLS].drop_duplicates())
    if result["n_conditions"] != n_conditions:
        raise ValueError("验证报告中的实验条件数不一致。")
    rows = [row for row in result["summary"] if row["scheme"] == "condition"]
    expected = {(target, model) for target in TARGET_COLS for model in ("gpr", "linear", "quadratic")}
    if len(rows) != len(expected) or {(r["target"], r["model"]) for r in rows} != expected:
        raise ValueError("留一条件汇总缺失或重复。")
    for row in rows:
        records = sorted((p for p in result["predictions"] if p["scheme"] == "condition"
                          and p["target"] == row["target"] and p["model"] == row["model"]),
                         key=lambda p: p["row_index"])
        if [p["row_index"] for p in records] != list(range(len(data))):
            raise ValueError("留一条件逐行记录不完整或重复。")
        observed = np.array([p["observed"] for p in records], dtype=float)
        predicted = np.array([p["predicted"] for p in records], dtype=float)
        if not np.isfinite([observed, predicted]).all() or not np.array_equal(
            observed, data[row["target"]].to_numpy(dtype=float)
        ):
            raise ValueError("验证实测值与训练源文件不一致或预测无效。")
        delta = predicted - observed
        total = np.sum((observed - observed.mean()) ** 2)
        actual = {"mae": np.mean(abs(delta)), "rmse": np.sqrt(np.mean(delta ** 2)),
                  "r2": 1 - np.sum(delta ** 2) / total if total > 0 else None}
        if row["n_folds"] != n_conditions:
            raise ValueError("留一条件折数与源数据不一致。")
        if row["model"] == "gpr":
            lower = np.array([p["lower"] for p in records], dtype=float)
            upper = np.array([p["upper"] for p in records], dtype=float)
            if not np.isfinite([lower, upper]).all() or not ((lower <= predicted) & (predicted <= upper)).all():
                raise ValueError("验证区间无效。")
            actual.update(coverage=np.mean((lower <= observed) & (observed <= upper)),
                          mean_width=np.mean(upper - lower))
        for key, value in actual.items():
            if (value is None and row[key] is not None) or (value is not None and
                    (row[key] is None or not np.isfinite(row[key]) or not np.isclose(row[key], value, rtol=1e-9, atol=1e-9))):
                raise ValueError("留一条件指标无法从逐行结果复算。")
        folds = [f for f in result["folds"] if f["scheme"] == "condition"
                 and f["target"] == row["target"] and f["model"] == row["model"]]
        if sorted(f["fold"] for f in folds) != list(range(1, n_conditions + 1)):
            raise ValueError("留一条件逐折记录缺失或重复。")
        seen = []
        for fold in folds:
            indices = fold["test_indices"]
            train = fold["train_indices"]
            if not indices or sorted(train + indices) != list(range(len(data))):
                raise ValueError("训练折与测试折未完整分离。")
            conditions = data.iloc[indices][FEATURE_COLS].drop_duplicates()
            if len(conditions) != 1 or conditions.to_dict("records") != fold["held_out_conditions"]:
                raise ValueError("留出条件与逐折记录不符。")
            condition = conditions.iloc[0].to_numpy(dtype=float)
            expected_indices = np.flatnonzero((data[FEATURE_COLS].to_numpy() == condition).all(axis=1)).tolist()
            if sorted(indices) != expected_indices or any(records[i]["fold"] != fold["fold"] for i in indices):
                raise ValueError("同一实验条件没有整组留出。")
            seen.extend(indices)
            train_x, test_x = data.iloc[train][FEATURE_COLS], data.iloc[indices][FEATURE_COLS]
            outside = (test_x.lt(train_x.min()) | test_x.gt(train_x.max()))
            if fold["outside_features"] != [c for c in FEATURE_COLS if outside[c].any()] or fold["constant_training_features"] != [c for c in FEATURE_COLS if train_x[c].nunique() < 2]:
                raise ValueError("逐折支持边界与训练数据不一致。")
            if not isinstance(fold["warnings"], list):
                raise ValueError("优化警告记录格式错误。")
            difference = delta[indices]
            variance = np.sum((observed[indices] - observed[indices].mean()) ** 2)
            scores = {"mae": np.mean(abs(difference)), "rmse": np.sqrt(np.mean(difference ** 2)),
                      "r2": 1 - np.sum(difference ** 2) / variance if variance > 0 else None}
            for key, score in scores.items():
                if (score is None and fold[key] is not None) or (score is not None and
                        (fold[key] is None or not np.isfinite(fold[key]) or not np.isclose(score, fold[key], rtol=1e-9, atol=1e-9))):
                    raise ValueError("逐折误差无法从逐行结果复算。")
        if sorted(seen) != list(range(len(data))):
            raise ValueError("逐折测试记录重复或缺失。")
    return result


def support_note(data, dose, temperature):
    """区分已测组合与范围内未测组合，不把矩形范围等同实验支持。"""
    values = data[FEATURE_COLS].to_numpy(dtype=float)
    matches = np.isclose(values, [dose, temperature], rtol=0, atol=1e-8).all(axis=1)
    if matches.any():
        return f"已有 {int(matches.sum())} 条同条件实测记录；当前数值仍是模型估计，不是新增实测。"
    return "当前组合没有直接实测：虽在两输入的取值范围内，仍依赖加性模型假设，不能视为已验证条件。"


def primary_table(result):
    """仅输出留一条件指标，不把其他验证指标送入展示表。"""
    names = {"gpr": "GPR（主模型）", "linear": "线性（对照）", "quadratic": "加性二次（对照）"}
    rows = []
    for item in result["summary"]:
        if item["scheme"] != "condition":
            continue
        rows.append({"目标": item["target"], "模型": names[item["model"]], "折数": item["n_folds"],
                     "MAE": item["mae"], "RMSE": item["rmse"], "R²": item["r2"],
                     "90%区间经验覆盖率": item.get("coverage"), "平均区间宽度": item.get("mean_width")})
    return pd.DataFrame(rows)
