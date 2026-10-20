"""只读数据图解：合成观测、全量模型趋势和留一条件折外误差分开展示。"""

import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app.validation_view import check_source
from src.config import FEATURE_COLS, TARGET_COLS, UNIT_MAP
from src.predict import predict


def axis_title(name):
    return f"{name}（{UNIT_MAP[name]}）"


def snapshot_frame(bundle, data, models):
    """必须与已解析整套版本一致，不能从实验库拼接待审核记录。"""
    check_source(bundle["source"], data, models, bundle["metadata"])
    records = json.loads((bundle["directory"] / "records.json").read_text(encoding="utf-8"))
    frame = pd.DataFrame([r["values"] for r in records])[[*FEATURE_COLS, *TARGET_COLS]]
    if len(frame) != len(data) or not np.array_equal(frame.to_numpy(), data.to_numpy()):
        raise ValueError("图解记录与当前训练快照不一致，停止展示。")
    ids = [r["record_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("训练快照中的记录标识重复，停止展示。")
    frame.insert(0, "记录标识", ids)
    frame.insert(0, "版本编号", bundle["id"])
    frame["标识说明"] = ["迁移标识，非原实验编号" if i.startswith("baseline:") else "已审核实验记录" for i in ids]
    frame["温度分组"] = frame[FEATURE_COLS[1]].map(lambda t: f"{t:g} ℃")
    return frame


def default_value(bounds, preferred):
    low, high = bounds
    return float(preferred if low <= preferred <= high else (low + high) / 2)


def trend_frames(version, target, dose, temperature, bounds, models):
    """各101个等距原接口预测，当前输入独立计算，不插值、不裁剪。"""
    frames = []
    for feature in FEATURE_COLS:
        rows = []
        for value in np.linspace(*bounds[feature], 101):
            params = {FEATURE_COLS[0]: dose, FEATURE_COLS[1]: temperature}
            params[feature] = float(value)
            item = predict(params, model_artifact=models)[target]
            rows.append({"版本编号": version, "目标": target, **params,
                         "预测值": item["value"], "区间下界": item["lower"], "区间上界": item["upper"]})
        frames.append(pd.DataFrame(rows))
    current = predict({FEATURE_COLS[0]: dose, FEATURE_COLS[1]: temperature}, model_artifact=models)[target]
    return frames, current


@st.cache_data(show_spinner=False, max_entries=16)
def cached_trends(identity, target, dose, temperature, bounds_items, _models):
    # 完整版本路径、编号及数据指纹入键；不同隔离库的initial也不会共享缓存。
    return trend_frames(identity[1], target, dose, temperature, dict(bounds_items), _models)


def error_frames(validation, raw, target):
    """只取已校验报告中的完整留一条件GPR记录，不挑选容易的折。"""
    predictions = sorted((p for p in validation["predictions"] if p["scheme"] == "condition"
                          and p["model"] == "gpr" and p["target"] == target), key=lambda p: p["row_index"])
    if [p["row_index"] for p in predictions] != list(range(len(raw))):
        raise ValueError("留一条件逐行结果不完整，已隐藏误差图。")
    folds = {f["fold"]: f for f in validation["folds"] if f["scheme"] == "condition"
             and f["model"] == "gpr" and f["target"] == target}
    rows = []
    for p in predictions:
        source = raw.iloc[p["row_index"]]
        fold = folds[p["fold"]]
        rows.append({"版本编号": source["版本编号"], "目标": target, "记录标识": source["记录标识"],
                     "剂量": source[FEATURE_COLS[0]], "温度": source[FEATURE_COLS[1]],
                     "条件": f'{source[FEATURE_COLS[0]]:g} / {source[FEATURE_COLS[1]]:g}℃',
                     "折": p["fold"], "合成观测值": p["observed"], "折外预测值": p["predicted"],
                     "区间下界": p["lower"], "区间上界": p["upper"],
                     "误差": p["predicted"] - p["observed"],
                     "绝对误差": abs(p["predicted"] - p["observed"]),
                     "折内测试范围": "超训练折范围" if p["outside_train_range"] else "训练折范围内",
                     "恒定训练特征": "、".join(fold["constant_training_features"]) or "无",
                     "优化警告数": len(fold["warnings"])})
    points = pd.DataFrame(rows)
    grouped = points.groupby(["版本编号", "目标", "条件", "折"], sort=False).agg(
        MAE=("绝对误差", "mean"), 记录数=("记录标识", "size")).reset_index()
    warnings = pd.DataFrame([{"折": f["fold"], "超训练折范围特征": "、".join(f["outside_features"]) or "无",
                              "恒定训练特征": "、".join(f["constant_training_features"]) or "无",
                              "优化警告": json.dumps(f["warnings"], ensure_ascii=False)} for f in folds.values()])
    return points, grouped, warnings


def style_chart(chart, description):
    """局部深色样式，不改变全局主题和其他页面。"""
    return chart.properties(height=300, description=description).configure(
        background="#101c2b", font="Microsoft YaHei, sans-serif"
    ).configure_axis(labelColor="#d3dce7", titleColor="#f2f5f9", gridColor="#344254",
                     labelFontSize=12, titleFontSize=13).configure_legend(
        labelColor="#d3dce7", titleColor="#f2f5f9", orient="bottom", labelLimit=260
    ).configure_view(stroke="#344254")


def raw_chart(raw, target):
    return style_chart(alt.Chart(raw.drop(columns=["记录标识", "标识说明"], errors="ignore")).mark_point(size=85, opacity=0.85).encode(
        x=alt.X(f"{FEATURE_COLS[0]}:Q", title=axis_title(FEATURE_COLS[0]), scale=alt.Scale(zero=False)),
        y=alt.Y(f"{target}:Q", title=axis_title(target), scale=alt.Scale(zero=True)),
        color=alt.Color("温度分组:N", title="合成观测温度", scale=alt.Scale(scheme="tableau10")),
        shape=alt.Shape("温度分组:N", title="合成观测温度"),
        tooltip=["版本编号:N", *[alt.Tooltip(f"{f}:Q", format=".6g")
                  for f in [*FEATURE_COLS, target]]]
    ).interactive(), "全部合成观测记录，不平滑、不连接、不移动重叠点。")


def slice_chart(curve, observations, current, feature, dose, temperature, target):
    tooltip = [alt.Tooltip(f"{f}:Q", format=".6g") for f in [*FEATURE_COLS, "预测值", "区间下界", "区间上界"]]
    base = alt.Chart(curve).encode(x=alt.X(f"{feature}:Q", title=axis_title(feature), scale=alt.Scale(zero=False)))
    band = base.mark_area(color="#76b7ed", opacity=0.22).encode(
        y=alt.Y("区间下界:Q", title=axis_title(target), scale=alt.Scale(zero=True)), y2="区间上界:Q", tooltip=tooltip)
    line = base.mark_line(color="#76b7ed", strokeWidth=2).encode(y="预测值:Q", tooltip=tooltip)
    actual = alt.Chart(observations.drop(columns=["记录标识", "标识说明"], errors="ignore")).mark_point(shape="circle", color="#f5b644", size=95).encode(
        x=f"{feature}:Q", y=f"{target}:Q", tooltip=[*[alt.Tooltip(f"{f}:Q", format=".6g")
                                                  for f in [*FEATURE_COLS, target]]])
    marker = pd.DataFrame([{FEATURE_COLS[0]: dose, FEATURE_COLS[1]: temperature,
                            "预测值": current["value"], "区间下界": current["lower"], "区间上界": current["upper"]}])
    point = alt.Chart(marker).mark_point(shape="diamond", color="#ffffff", filled=True, size=120).encode(
        x=f"{feature}:Q", y="预测值:Q", tooltip=tooltip)
    return style_chart(band + line + actual + point,
                       "蓝线为模型估计，阴影为名义90%预测区间，黄色圆点为固定条件匹配的合成观测，白色菱形为当前输入。")


def oof_chart(points, target):
    low = min(points["合成观测值"].min(), points["折外预测值"].min(), points["区间下界"].min())
    high = max(points["合成观测值"].max(), points["折外预测值"].max(), points["区间上界"].max())
    reference = alt.Chart(pd.DataFrame({"参考": [low, high]})).mark_line(
        color="#9aa9b9", strokeDash=[6, 4]).encode(x="参考:Q", y="参考:Q")
    base = alt.Chart(points.drop(columns=["记录标识", "标识说明"], errors="ignore")).encode(
        x=alt.X("合成观测值:Q", title=f"合成观测{axis_title(target)}", scale=alt.Scale(domain=[low, high])),
        color=alt.Color("条件:N", title="剂量 / 温度", scale=alt.Scale(scheme="tableau10")))
    interval = base.mark_rule(opacity=0.35).encode(y="区间下界:Q", y2="区间上界:Q")
    scatter = base.mark_point(size=85).encode(
        y=alt.Y("折外预测值:Q", title=f"折外预测{axis_title(target)}", scale=alt.Scale(domain=[low, high])),
        shape=alt.Shape("折内测试范围:N", title="折内测试范围"),
        tooltip=["条件:N", "折:O", "折内测试范围:N", "恒定训练特征:N",
                 *[alt.Tooltip(f"{f}:Q", format=".6g") for f in ["合成观测值", "折外预测值", "误差", "区间下界", "区间上界"]]])
    return style_chart(reference + interval + scatter, "留一条件折外预测，虚线为预测等于合成观测；竖线为名义90%区间。所有困难折均保留。")


def mae_chart(grouped, target):
    return style_chart(alt.Chart(grouped).mark_bar(color="#76b7ed").encode(
        x=alt.X("条件:N", title="留出条件：剂量 / 温度", sort=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("MAE:Q", title=f"平均绝对误差（{UNIT_MAP[target]}）", scale=alt.Scale(zero=True)),
        tooltip=["条件:N", "折:O", "记录数:Q", alt.Tooltip("MAE:Q", format=".6g")]
    ), "按完整留出条件计算MAE，越高代表该条件误差越大；未删除困难条件。")


def table_download(frame, label, filename):
    with st.expander(label):
        st.dataframe(frame, hide_index=True, use_container_width=True)
        st.download_button("下载" + label, frame.to_csv(index=False).encode("utf-8-sig"),
                           file_name=filename, mime="text/csv", key="graph_download_" + filename)


def render_explorer(bundle, data, models, validation, validation_error):
    with st.container(key="explorer_panel"):
        st.subheader("数据图解 · 合成观测、趋势与误差")
        try:
            raw = snapshot_frame(bundle, data, models)
        except (OSError, ValueError, KeyError, TypeError):
            st.error("图解模型或训练快照未能通过一致性校验；停止展示，不使用旧版本代替。")
            return
        st.caption(f'当前版本：{bundle["id"]} · {len(raw)}条训练记录 · '
                   f'{len(raw[FEATURE_COLS].drop_duplicates())}个条件。只显示该版本训练快照。')
        st.warning("新条件预测尚不可靠。曲线是加性模型估计，不是实验机理；不能辨识剂量与温度交互，也不输出精确最优剂量。")
        target = st.selectbox("图解目标", TARGET_COLS, key="graph_target")
        bounds = {f: tuple(bundle["metadata"]["feature_bounds"][f]) for f in FEATURE_COLS}
        identity = (str(bundle["directory"].resolve()), bundle["id"], bundle["metadata"]["data_sha256"])
        if st.session_state.get("graph_version") != identity:
            for key, f, preferred in (("graph_dose", FEATURE_COLS[0], 11.0),
                                      ("graph_temperature", FEATURE_COLS[1], 180.0)):
                st.session_state[key] = default_value(bounds[f], preferred)
            st.session_state["graph_version"] = identity
        st.markdown("### 1．合成观测数据如何分布")
        st.altair_chart(raw_chart(raw, target), use_container_width=True, theme=None)
        st.caption("颜色与点形共同区分温度。全部记录原坐标保留，重叠点可在下表逐条查看；不抖动、不连线、不平滑。迁移标识不冒充原实验编号。")
        table_download(raw.drop(columns="温度分组"), "查看合成观测原始数值", "graph_observations.csv")
        st.markdown("### 2．输入变化时，模型如何变化")
        st.caption("以下输入仅控制本页，不改变预测页参数。每条曲线使用范围内101个等距点。")
        dose = st.number_input("图解固定剂量（a.u.）", min_value=float(bounds[FEATURE_COLS[0]][0]),
                               max_value=float(bounds[FEATURE_COLS[0]][1]), key="graph_dose", format="%.4f")
        temperature = st.number_input("图解固定温度（℃）", min_value=float(bounds[FEATURE_COLS[1]][0]),
                                      max_value=float(bounds[FEATURE_COLS[1]][1]), key="graph_temperature", format="%.2f")
        try:
            curves, current = cached_trends(identity, target, dose, temperature, tuple(bounds.items()), _models=models)
        except (ValueError, KeyError, TypeError):
            st.error("当前版本无法生成有效趋势及区间，已停止趋势展示。")
            return
        st.caption("图例：蓝线＝模型估计，非合成观测；蓝色阴影＝名义90%预测区间，不保证未来覆盖；黄色圆点＝合成观测；白色菱形＝当前输入。")
        st.write(f'当前输入模型估计：{current["value"]:.1f} {UNIT_MAP[target]}；'
                 f'名义90%预测区间 {current["lower"]:.1f} – {current["upper"]:.1f}。')
        for index, (feature, fixed_feature, fixed_value) in enumerate(
                ((FEATURE_COLS[0], FEATURE_COLS[1], temperature), (FEATURE_COLS[1], FEATURE_COLS[0], dose))):
            observations = raw.loc[raw[fixed_feature] == fixed_value]
            st.markdown(f"**固定{fixed_feature} {fixed_value:g} {UNIT_MAP[fixed_feature]}，改变{feature}**")
            st.altair_chart(slice_chart(curves[index], observations, current, feature, dose, temperature, target),
                            use_container_width=True, theme=None)
            if observations.empty:
                st.info("此固定条件没有直接合成观测切片，图中只有模型估计；范围内不代表有实验支持。")
            else:
                st.caption(f"仅叠加固定条件严格匹配的{len(observations)}条合成观测，不投影其他条件。")
            table_download(curves[index], f"查看{feature}切片数值", f"graph_slice_{index}.csv")
        if any((curve["区间下界"] < 0).any() for curve in curves) or current["lower"] < 0:
            st.warning("模型区间含负下界，原值完整保留；它不代表负比容量或负循环圈数具有物理意义。")
        st.caption("合成数据只用于演示。曲线不是新增合成观测，不说明已学会完整温度响应，也不能据曲线峰值选定最优剂量。")
        st.markdown("### 3．留一条件预测可能偏差多大")
        if validation is None:
            st.warning(validation_error or "当前版本验证报告不可用，已隐藏误差图，不使用旧报告。")
            return
        try:
            points, grouped, warnings = error_frames(validation, raw, target)
        except (KeyError, TypeError, ValueError):
            st.warning("当前版本逐行或逐折结果不完整，已隐藏误差图。")
            return
        st.caption("只用留一条件GPR折外预测，不是全量模型拟合值；每次将完整剂量＋温度组合留出。")
        st.altair_chart(oof_chart(points, target), use_container_width=True, theme=None)
        st.caption("虚线表示预测等于合成观测，离线越远误差越大；颜色区分留出条件、点形区分是否超训练折范围，竖线表示名义90%区间。")
        st.altair_chart(mae_chart(grouped, target), use_container_width=True, theme=None)
        st.caption("柱高为各条件平均绝对误差，不平均逐折R²；所有困难条件均保留。困难折保留，不宣称已学会真实温度响应。")
        table_download(points, "查看逐行折外数值", "graph_oof.csv")
        table_download(grouped, "查看逐条件误差", "graph_condition_mae.csv")
        with st.expander("查看折内范围、支持不足与优化警告"):
            st.dataframe(warnings, hide_index=True, use_container_width=True)
        st.caption("图解改善的是表达，不是精度。未审核、驳回和预测生成记录不进入图中训练快照。历史批次信息不足，不能声称排除全部批次相关性。")
