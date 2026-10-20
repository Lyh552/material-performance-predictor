"""只从独立公式生成合成数据，初始化公开演示；不读取任何真实实验文件。"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FEATURE_COLS, TARGET_COLS
from src.evaluate import evaluate, fit_recorded, report_lines, INTERVAL_Z
from src.modeling import build_models
from src.training import SETTINGS
from src.version_store import Store, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-only", action="store_true", required=True)
    parser.parse_args()
    if not (ROOT / "PUBLIC_SYNTHETIC_ONLY").is_file():
        raise ValueError("缺少公开合成目录标记，禁止在真实项目运行。")
    if (ROOT / "local_store/registry.sqlite3").exists() or (ROOT / "models/training_metadata.json").exists():
        raise ValueError("演示已经初始化，不覆盖已有版本。需要重建时使用一份新的公开包副本。")
    if sys.version_info[:2] != (3, 12) or version("scikit-learn") != "1.5.2":
        raise ValueError("请使用Python3.12与scikit-learn1.5.2，不自动更改环境。")
    rows = []
    for dose in (2., 4., 6., 8.):
        for temperature in (140., 180., 220.):
            for repeat in (-1, 1):
                rows.append({FEATURE_COLS[0]: dose, FEATURE_COLS[1]: temperature,
                             TARGET_COLS[0]: 120. + 14. * dose - 1.2 * dose ** 2 + .12 * (temperature - 140.) + repeat * 3.,
                             TARGET_COLS[1]: int(260 + 16 * dose - 2 * dose ** 2 + .45 * (temperature - 140) + repeat * 8)})
    data = pd.DataFrame(rows)
    for directory in ("models", "reports", "data/raw"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/raw/synthetic_demo.csv"
    data.to_csv(source, index=False, encoding="utf-8-sig")
    metadata = {"data_path": str(source), "data_sha256": digest(source), "n_samples": len(data),
                "feature_cols": FEATURE_COLS, "feature_bounds": {f: [float(data[f].min()), float(data[f].max())] for f in FEATURE_COLS},
                "sklearn_version": version("scikit-learn"), "primary_validation": "condition", "model_settings": SETTINGS,
                "code_sha256": {n: digest(ROOT / n) for n in ("pipeline.py", "src/modeling.py", "src/evaluate.py", "src/training.py", "src/config.py", "src/records.py", "src/jobs.py", "src/version_store.py")},
                "environment": {n: version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "streamlit")},
                "data_notice": "独立公式生成的24条合成数据，非真实实验，不作为科研性能证据。"}
    result = evaluate(data)
    result["metadata"] = metadata
    for target in TARGET_COLS:
        models = {}
        for name, model in build_models().items():
            models[name], _ = fit_recorded(model, data[FEATURE_COLS], data[target].to_numpy())
        joblib.dump({**metadata, "target_col": target, "model": models["gpr"], "model_type": "gpr",
                     "interval_z": INTERVAL_Z, "comparison_models": {k: v for k, v in models.items() if k != "gpr"}},
                    ROOT / "models" / f"model_{target}.joblib")
    def save(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    save(ROOT / "models/training_metadata.json", metadata)
    save(ROOT / "reports/cross_validation.json", result)
    pd.DataFrame(result["predictions"]).to_csv(ROOT / "reports/oof_predictions.csv", index=False, encoding="utf-8-sig")
    (ROOT / "reports/evaluation.md").write_text("# 合成流程演示\n\n不作为科研性能结论。\n" + "\n".join(report_lines(result)), encoding="utf-8")
    store = Store(ROOT / "local_store")
    store.initialize()
    print("已生成24条合成数据，12个条件；完成三套验证及独立版本初始化。未读取真实实验数据。")


if __name__ == "__main__":
    main()
