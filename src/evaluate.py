"""独立折内拟合的三套验证；离线保留困难折，不能替代线上范围限制。"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut

from src.config import FEATURE_COLS, TARGET_COLS
from src.modeling import build_models


SCHEMES = {"row": "留一行", "dose": "留一剂量", "condition": "留一条件（主指标）"}
INTERVAL_Z = 1.6448536269514722


def metrics(y, predicted):
    """整体指标合并折外预测计算；单样本或常量目标的 R² 不定义。"""
    y = np.asarray(y, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "mae": float(mean_absolute_error(y, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(y, predicted))),
        "r2": float(r2_score(y, predicted)) if len(y) > 1 and np.var(y) > 0 else None,
    }


def make_splits(X, scheme):
    """按数值剂量或完整条件分组，整组留出，不打乱记录。"""
    if scheme == "row":
        return list(LeaveOneOut().split(X))
    if scheme not in SCHEMES:
        raise ValueError("未知验证方式。")
    columns = FEATURE_COLS[:1] if scheme == "dose" else FEATURE_COLS
    groups = pd.factorize(pd.MultiIndex.from_frame(X[columns]))[0]
    if np.unique(groups).size < 2:
        raise ValueError(f"{SCHEMES[scheme]}至少需要两个组。")
    return list(LeaveOneGroupOut().split(X, groups=groups))


def fit_recorded(model, X, y):
    """保留所有优化警告，不用验证结果修改固定的模型约束。"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    notices = []
    for item in caught:
        original = str(item.message)
        if "lower bound" in original:
            message = "核参数触及优化下界，需谨慎解释。"
        elif "upper bound" in original:
            message = "核参数触及优化上界，需谨慎解释。"
        else:
            message = "模型拟合产生警告，可能未充分收敛，需谨慎解释。"
        notices.append({"category": item.category.__name__, "message": message})
    if hasattr(model.steps[-1][1], "kernel_"):
        kernel = model.steps[-1][1].kernel_
        for notice in notices:
            notice["fitted_kernel"] = str(kernel)
    return model, notices


def evaluate(data):
    """返回汇总、逐折与逐行折外结果，支持复算与分组审计。"""
    X = data[FEATURE_COLS].astype(float).reset_index(drop=True)
    if not np.isfinite(data[[*FEATURE_COLS, *TARGET_COLS]].to_numpy(dtype=float)).all():
        raise ValueError("交叉验证拒绝缺失或非有限值。")
    summary, folds, predictions = [], [], []
    for scheme in SCHEMES:
        splits = make_splits(X, scheme)
        for target in TARGET_COLS:
            y = data[target].to_numpy(dtype=float)
            for name in build_models():
                oof = np.full(len(X), np.nan)
                lower = np.full(len(X), np.nan)
                upper = np.full(len(X), np.nan)
                seen = np.zeros(len(X), dtype=int)
                for fold, (train, test) in enumerate(splits, start=1):
                    model = build_models()[name]
                    model, notices = fit_recorded(model, X.iloc[train], y[train])
                    if name == "gpr":
                        mean, std = model.predict(X.iloc[test], return_std=True)
                        lower[test] = mean - INTERVAL_Z * std
                        upper[test] = mean + INTERVAL_Z * std
                        if not (np.isfinite(lower[test]).all() and np.isfinite(upper[test]).all()
                                and (lower[test] <= mean).all() and (mean <= upper[test]).all()):
                            raise ValueError("折外预测区间无效。")
                    else:
                        mean = model.predict(X.iloc[test])
                    if not np.isfinite(mean).all():
                        raise ValueError("折外预测无效。")
                    oof[test] = mean
                    seen[test] += 1
                    train_X, test_X = X.iloc[train], X.iloc[test]
                    outside = (test_X.lt(train_X.min()) | test_X.gt(train_X.max()))
                    constant = [c for c in FEATURE_COLS if train_X[c].nunique() < 2]
                    item = {
                        "scheme": scheme, "target": target, "model": name, "fold": fold,
                        "train_indices": train.tolist(), "test_indices": test.tolist(),
                        "held_out_conditions": test_X.drop_duplicates().to_dict("records"),
                        "outside_features": [c for c in FEATURE_COLS if outside[c].any()],
                        "constant_training_features": constant, "warnings": notices,
                        "scaler_mean": model.steps[0][1].mean_.tolist(),
                        "scaler_scale": model.steps[0][1].scale_.tolist(),
                        **metrics(y[test], mean),
                    }
                    if name == "gpr":
                        item.update(coverage=float(np.mean((y[test] >= lower[test]) & (y[test] <= upper[test]))),
                                    mean_width=float(np.mean(upper[test] - lower[test])))
                    folds.append(item)
                    for index in test:
                        predictions.append({
                            "scheme": scheme, "target": target, "model": name, "fold": fold,
                            "row_index": int(index), "observed": float(y[index]),
                            "predicted": float(oof[index]),
                            "lower": float(lower[index]) if name == "gpr" else None,
                            "upper": float(upper[index]) if name == "gpr" else None,
                            "outside_train_range": bool(outside.loc[index].any()),
                        })
                if not (seen == 1).all() or not np.isfinite(oof).all():
                    raise RuntimeError("折外覆盖审计失败：每行必须且只能被留出一次。")
                item = {"scheme": scheme, "target": target, "model": name,
                        "n_folds": len(splits), **metrics(y, oof)}
                if name == "gpr":
                    item.update(coverage=float(np.mean((y >= lower) & (y <= upper))),
                                mean_width=float(np.mean(upper - lower)))
                summary.append(item)
    return {"summary": summary, "folds": folds, "predictions": predictions,
            "n_conditions": int(len(X.drop_duplicates()))}


def format_number(value):
    return "不定义" if value is None else f"{value:.6f}"


def report_lines(results):
    """人类可读报告与机器可复算明细同时保留。"""
    lines = ["## 三套折外验证", "", "**主指标是留一条件；留一行存在重复实验信息泄漏，指标偏乐观，不用于对外宣传。**",
             "整体 R² 由合并的折外预测计算，不平均各折 R²。", "",
             "| 目标 | 模型 | 验证 | MAE | RMSE | R² | 名义90%覆盖率 | 平均区间宽度 |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for item in results["summary"]:
        cells = [item["target"], item["model"], SCHEMES[item["scheme"]]]
        cells += [format_number(item[k]) for k in ("mae", "rmse", "r2")]
        cells += [format_number(item.get(k)) if item["model"] == "gpr" else "不适用"
                  for k in ("coverage", "mean_width")]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "### 主模型结论", ""]
    for item in results["summary"]:
        if item["scheme"] != "condition" or item["model"] != "gpr":
            continue
        if item["r2"] is None or item["r2"] < 0:
            lines.append(f"- {item['target']}：留一条件未证明可靠的新条件泛化能力；即使只作粗略预测，也不能忽略该结果。")
        if item["coverage"] < 0.90:
            lines.append(f"- {item['target']}：本次折外覆盖率低于名义 90%，现有区间不能作为可信覆盖承诺。")
    lines += ["", "## 留一条件逐折误差", "",
              "| 目标 | 模型 | 折 | 留出条件（剂量,温度） | MAE | RMSE | R² | 覆盖率 | 平均宽度 | 支持边界 |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for item in results["folds"]:
        if item["scheme"] != "condition":
            continue
        conditions = "、".join(str(tuple(c[f] for f in FEATURE_COLS)) for c in item["held_out_conditions"])
        notes = []
        if item["outside_features"]:
            notes.append("超出折内范围：" + "、".join(item["outside_features"]))
        if item["constant_training_features"]:
            notes.append("训练折无变化：" + "、".join(item["constant_training_features"]) + "，无法学习相应响应")
        cells = [item["target"], item["model"], str(item["fold"]), conditions]
        cells += [format_number(item[k]) for k in ("mae", "rmse", "r2")]
        cells += [format_number(item.get(k)) if item["model"] == "gpr" else "不适用"
                  for k in ("coverage", "mean_width")]
        lines.append("| " + " | ".join(cells + ["；".join(notes) or "范围内未测条件"]) + " |")
    lines += ["", "## 解释边界", "",
              "- 离线验证保留全部困难折，包括超出折内训练范围的测试记录；不意味着线上允许外推。",
              "- 留出 200℃ 条件时，训练数据没有温度变化，不能认为模型已经学会温度响应。",
              "- 温度只有剂量 11.0 处有对照；加性假设不能证明不存在交互，也不能辨识交互。",
              "- 区间含观测噪声，覆盖率是本次折外经验结果，不是独立校准或未来覆盖保证。",
              f"- 当前有 {results['n_conditions']} 个实验条件，不能将重复行数理解为同等数量的独立条件。",
              "- 单折只有少量重复记录，R² 对组内方差很敏感；同时查看原单位下的 MAE、RMSE 和总体折外 R²。",
              "- 尚无实验编号与批次依赖资料，按条件分组不代表已经排除全部实验相关性。",
              "- 新增实测数据需审核后重新训练和验证，不保证每次增加数据都提升预测能力。",
              "", "## 折内优化警告", ""]
    for item in results["folds"]:
        for notice in item["warnings"]:
            lines.append(f"- {SCHEMES[item['scheme']]} / {item['target']} / {item['model']} / 折 {item['fold']}："
                         + notice["message"] + f" 拟合核：`{notice.get('fitted_kernel', '不适用')}`")
    if not any(i["warnings"] for i in results["folds"]):
        lines.append("本次折内拟合没有产生优化警告。")
    return lines
