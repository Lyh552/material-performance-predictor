"""候选版本训练和同条件共同测试；不修改任何正式模型文件。"""

from importlib.metadata import version
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, FEATURE_COLS, TARGET_COLS
from src.evaluate import evaluate, fit_recorded, metrics, report_lines, INTERVAL_Z
from src.modeling import build_models
from src.version_store import digest, read_json, write_json

SETTINGS = {"random_seed": 42, "length_scale_bounds": [0.35, 10.0],
            "noise_level_bounds": [1e-4, 10.0], "n_restarts_optimizer": 2,
            "kernel": "逐维 RBF 相加再加观测噪声；无交互项"}


def records_frame(records):
    if len({r["record_id"] for r in records}) != len(records):
        raise ValueError("训练快照记录标识重复。")
    data = pd.DataFrame([r["values"] for r in records])[[*FEATURE_COLS, *TARGET_COLS]].astype(float)
    if len(data) < 3 or not np.isfinite(data.to_numpy()).all():
        raise ValueError("快照必须有至少三条完整有限数值记录。")
    return data


def train_candidate(store, job, request):
    folder = store.root / "jobs" / job / "candidate"
    folder.mkdir()
    records = request["records"]
    data = records_frame(records)
    if any(data[c].nunique() < 2 for c in FEATURE_COLS):
        raise ValueError("候选全量数据的两个输入都必须有变化。")
    baseline = read_json(store.verify("initial") / "training_metadata.json")
    if baseline["model_settings"] != SETTINGS or baseline["code_sha256"]["src/modeling.py"] != digest(PROJECT_ROOT / "src/modeling.py"):
        raise ValueError("模型结构或边界发生变化，禁止在当前固定方案下自动训练。")
    source = folder / "training.csv"
    data.to_csv(source, index=False, encoding="utf-8-sig")
    write_json(folder / "records.json", records)
    environment = {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "streamlit")}
    environment["python"] = sys.version.split()[0]
    if environment["scikit-learn"] != "1.5.2" or environment["streamlit"] != "1.39.0" or sys.version_info[:2] != (3, 12):
        raise ValueError("运行环境不符合锁定版本；不自动升级或降级。")
    final_folder = store.version_dir(job)
    metadata = {"data_path": str(final_folder / "training.csv"), "data_sha256": digest(source),
                "n_samples": len(data), "feature_cols": FEATURE_COLS,
                "feature_bounds": {c: [float(data[c].min()), float(data[c].max())] for c in FEATURE_COLS},
                "sklearn_version": environment["scikit-learn"], "environment": environment,
                "code_sha256": {n: digest(PROJECT_ROOT / n) for n in ("pipeline.py", "src/modeling.py", "src/evaluate.py", "src/training.py", "src/config.py", "src/records.py", "src/jobs.py", "src/version_store.py")},
                "model_settings": SETTINGS, "primary_validation": "condition", "version_id": job,
                "validation_status": "候选三套折外验证；仅留一条件对外说明",
                "row_index_note": "加载后零基顺序；对应records.json中的记录标识，不是原文件物理行号。"}
    print(f"候选数据：{len(data)}条；冻结后新增审核不进入本次训练。", flush=True)
    results = evaluate(data)
    results["metadata"] = metadata
    print("三套独立折内验证完成，开始全量拟合。", flush=True)
    final_warnings = []
    for target in TARGET_COLS:
        fitted = {}
        for name, model in build_models().items():
            fitted[name], notices = fit_recorded(model, data[FEATURE_COLS], data[target].to_numpy())
            final_warnings.extend({"target": target, "model": name, **notice} for notice in notices)
        artifact = {**metadata, "model": fitted["gpr"], "model_type": "gpr", "target_col": target,
                    "interval_z": INTERVAL_Z, "comparison_models": {k: v for k, v in fitted.items() if k != "gpr"}}
        joblib.dump(artifact, folder / f"model_{target}.joblib")
    write_json(folder / "training_metadata.json", metadata)
    write_json(folder / "cross_validation.json", results)
    write_json(folder / "full_fit_warnings.json", final_warnings)
    pd.DataFrame(results["predictions"]).to_csv(folder / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    lines = ["# 候选模型验证", "", "尚未正式启用；不是精度提升承诺。", ""] + report_lines(results)
    lines += ["", "## 全量拟合警告", str(final_warnings)]
    (folder / "evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(folder / "version.json", {"id": job, "parent": request["against"], "source": "training.csv",
                                        "created": request["created"], "kind": "已审核实测候选"})
    # 先按页面同一套标准完成语义检查，再封存，不修改其他版本。
    from app.validation_view import check_source, load_validation
    check_source(source, data, {t: joblib.load(folder / f"model_{t}.joblib") for t in TARGET_COLS}, metadata)
    load_validation(folder / "cross_validation.json", metadata, data)
    return folder


def compare_datasets(old_records, candidate_records, against, candidate):
    """每折双方从各自训练数据剔除整个测试条件，测试记录为同一并集。"""
    old = records_frame(old_records)
    new = records_frame(candidate_records)
    union = {}
    for record in [*old_records, *candidate_records]:
        identifier = record["record_id"]
        if identifier in union and any(float(union[identifier]["values"][c]) != float(record["values"][c]) for c in [*FEATURE_COLS, *TARGET_COLS]):
            raise ValueError("同一记录标识在新旧快照中出现不同数值。")
        union[identifier] = record
    testing = list(union.values())
    test_data = records_frame(testing)
    conditions = test_data[FEATURE_COLS].drop_duplicates().to_numpy()
    predictions, folds, summaries = [], [], []
    for target in TARGET_COLS:
        for name in build_models():
            for side, training, source_records in (("old", old, old_records), ("candidate", new, candidate_records)):
                all_predicted = np.full(len(test_data), np.nan)
                low, high = all_predicted.copy(), all_predicted.copy()
                for fold, condition in enumerate(conditions, 1):
                    train_mask = ~(training[FEATURE_COLS].to_numpy() == condition).all(axis=1)
                    test_indices = np.flatnonzero((test_data[FEATURE_COLS].to_numpy() == condition).all(axis=1))
                    train_indices = np.flatnonzero(train_mask)
                    train_x = training.iloc[train_indices][FEATURE_COLS]
                    test_x = test_data.iloc[test_indices][FEATURE_COLS]
                    item = {"target": target, "model": name, "side": side, "fold": fold,
                            "condition": condition.tolist(), "train_ids": [source_records[i]["record_id"] for i in train_indices],
                            "test_ids": [testing[i]["record_id"] for i in test_indices], "warnings": [], "error": None,
                            "outside_features": [c for c in FEATURE_COLS if (test_x[c].lt(train_x[c].min()) | test_x[c].gt(train_x[c].max())).any()],
                            "constant_training_features": [c for c in FEATURE_COLS if train_x[c].nunique() < 2]}
                    try:
                        if len(train_x) < 2:
                            raise ValueError("训练折不足两行，不能比较。")
                        model, item["warnings"] = fit_recorded(build_models()[name], train_x, training.iloc[train_indices][target].to_numpy())
                        item["scaler_mean"] = model.steps[0][1].mean_.tolist()
                        item["scaler_scale"] = model.steps[0][1].scale_.tolist()
                        if name == "gpr":
                            mean, std = model.predict(test_x, return_std=True)
                            low[test_indices], high[test_indices] = mean - INTERVAL_Z * std, mean + INTERVAL_Z * std
                            if not np.isfinite([low[test_indices], high[test_indices]]).all() or not (low[test_indices] <= high[test_indices]).all():
                                raise ValueError("区间无效，不能比较。")
                        else:
                            mean = model.predict(test_x)
                        if not np.isfinite(mean).all():
                            raise ValueError("预测结果无效，不能比较。")
                        all_predicted[test_indices] = mean
                        item.update(metrics(test_data.iloc[test_indices][target], mean))
                    except (ValueError, np.linalg.LinAlgError) as error:
                        item["error"] = f"该折不可比较：{error}"
                    folds.append(item)
                observed = test_data[target].to_numpy()
                complete = bool(np.isfinite(all_predicted).all())
                summary = {"target": target, "model": name, "side": side, "complete": complete, "n_test": len(test_data)}
                summary.update(metrics(observed, all_predicted) if complete else {"mae": None, "rmse": None, "r2": None})
                if name == "gpr":
                    summary.update(coverage=float(np.mean((low <= observed) & (observed <= high))) if complete else None,
                                   mean_width=float(np.mean(high - low)) if complete else None)
                summaries.append(summary)
                for i, record in enumerate(testing):
                    valid = bool(np.isfinite(all_predicted[i]))
                    predictions.append({"record_id": record["record_id"], "target": target, "model": name, "side": side,
                                        "condition": test_data.iloc[i][FEATURE_COLS].tolist(), "observed": float(observed[i]),
                                        "predicted": float(all_predicted[i]) if valid else None,
                                        "lower": float(low[i]) if valid and name == "gpr" else None,
                                        "upper": float(high[i]) if valid and name == "gpr" else None})
    return {"against": against, "candidate": candidate, "summary": summaries, "folds": folds, "predictions": predictions,
            "note": "不同训练数据方案的同口径内部验证，不是两个全量模型在独立未来实验上的直接比赛；批次依赖未完全排除。"}


def comparison_risks(comparison):
    risks = []
    for target in TARGET_COLS:
        pair = {r["side"]: r for r in comparison["summary"] if r["target"] == target and r["model"] == "gpr"}
        old, new = pair["old"], pair["candidate"]
        if not old["complete"] or not new["complete"]:
            risks.append(f"{target}：共同测试有不可比较折，不能宣称改进。")
            continue
        if new["r2"] is None or new["r2"] < 0:
            risks.append(f"{target}：候选共同测试R²为负或不定义，新条件泛化证据不足。")
        if new["coverage"] < 0.90:
            risks.append(f"{target}：区间覆盖不足名义90%。")
        for key, label in (("mae", "MAE"), ("rmse", "RMSE")):
            if new[key] > old[key] + 1e-9:
                risks.append(f"{target}：候选{label}比旧方案增加。")
        if new["coverage"] < old["coverage"] - 1e-9:
            risks.append(f"{target}：候选区间覆盖率下降；区间变窄不等于更准确。")
    return risks
