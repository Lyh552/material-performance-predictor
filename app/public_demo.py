"""公开录屏专用入口：只接受独立合成项目，不展示编号、批次或内部路径。"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from app.data_explorer import raw_chart, slice_chart, oof_chart, mae_chart, trend_frames, error_frames
from app.validation_view import check_source, load_validation, primary_table
from src.config import FEATURE_COLS, TARGET_COLS, UNIT_MAP
from src.predict import predict
from src.version_store import Store

st.set_page_config(page_title="合成数据公开演示", layout="wide")
st.title("材料性能趋势预测 · 合成数据演示")
st.warning("本页全部使用独立公式生成的合成数据，非课题组真实实验，不作为科研精度证据。")
if not (ROOT / "PUBLIC_SYNTHETIC_ONLY").is_file():
    st.error("当前不是已隔离的公开合成项目，停止展示。")
    st.stop()
store = Store(ROOT / "local_store")
try:
    bundle = store.bundle()
    metadata = bundle["metadata"]
    if metadata.get("data_notice") != "独立公式生成的24条合成数据，非真实实验，不作为科研性能证据。":
        raise ValueError("不是合成版本")
    models = {t: joblib.load(p) for t, p in bundle["model_paths"].items()}
    data = pd.read_csv(bundle["source"])[[*FEATURE_COLS, *TARGET_COLS]]
    check_source(bundle["source"], data, models, metadata)
    validation = load_validation(bundle["directory"] / "cross_validation.json", metadata, data)
except (OSError, ValueError, KeyError):
    st.error("合成演示尚未初始化或校验失败；请先按README运行合成初始化。不显示内部路径。")
    st.stop()
st.caption(f"合成训练记录：{len(data)}条 · 完整条件：{len(data[FEATURE_COLS].drop_duplicates())}组 · 不公开真实原始数据")
prediction, graph, method, workflow = st.tabs(["预测演示", "数据图解", "验证与边界", "更新流程"])
bounds = metadata["feature_bounds"]
with prediction:
    dose = st.number_input("模拟剂量（a.u.）", min_value=2., max_value=8., value=5., step=.1)
    temperature = st.number_input("模拟温度（℃）", min_value=140., max_value=220., value=180., step=20.)
    result = predict({FEATURE_COLS[0]: dose, FEATURE_COLS[1]: temperature}, model_artifact=models)
    for target, column in zip(TARGET_COLS, st.columns(2)):
        item = result[target]
        with column:
            st.metric(f"{target}（{UNIT_MAP[target]}）", f'{item["value"]:.1f}')
            st.write(f'名义90%区间：{item["lower"]:.1f}–{item["upper"]:.1f}')
    st.info("模型估计不是实际测量；区间不保证未来覆盖，不推断因果或精确最优配方。")
with graph:
    target = st.selectbox("图解目标", TARGET_COLS)
    raw = data.copy()
    raw["版本编号"] = "合成演示"
    raw["温度分组"] = raw[FEATURE_COLS[1]].map(lambda t: f"{t:g} ℃")
    # 标识只在计算层关联行，公开图、悬停和表格均不输出此字段。
    raw["记录标识"] = [f"internal:{i}" for i in range(len(raw))]
    raw["标识说明"] = "合成数据"
    st.subheader("合成分布与趋势")
    st.altair_chart(raw_chart(raw, target), use_container_width=True, theme=None)
    curves, current = trend_frames("合成演示", target, dose, temperature, bounds, models)
    st.caption("蓝线：模型估计；阴影：名义90%预测区间；黄色圆点：固定条件匹配的合成点；白色菱形：当前输入。")
    for curve, feature, fixed_feature, fixed in zip(curves, FEATURE_COLS, FEATURE_COLS[::-1], (temperature, dose)):
        st.altair_chart(slice_chart(curve, raw.loc[raw[fixed_feature] == fixed], current, feature, dose, temperature, target),
                        use_container_width=True, theme=None)
    points, grouped, _ = error_frames(validation, raw, target)
    st.subheader("合成数据的留一条件误差")
    st.altair_chart(oof_chart(points, target), use_container_width=True, theme=None)
    st.altair_chart(mae_chart(grouped, target), use_container_width=True, theme=None)
    st.caption("所有图形都是合成流程示例，不是未公开研究的数据或性能结果。")
with method:
    st.dataframe(primary_table(validation), hide_index=True, use_container_width=True)
    st.info("按完整条件留出验证；留一行有重复信息泄漏。合成数据验证只说明流程可运行，不能证明真实模型准确。")
with workflow:
    st.subheader("实测更新采用人工确认，不自动启用")
    st.markdown("字段检查 → 待审核 → 人工确认 → 冻结快照 → 候选训练 → 同口径比较 → 确认启用或回滚")
    st.caption("公开录屏只展示流程说明，不展示实验编号、批次、审核记录、来源路径或私密模型。原实现代码保留在仓库中供流程复现。")
st.caption("原始实验数据因课题组授权限制未公开。仓库提供字段模板与数据字典；完整数据请联系作者，并以课题组授权为准。")
