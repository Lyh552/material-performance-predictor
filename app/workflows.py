"""实测录入、审核和模型生命周期页面；操作必须显式提交。"""

from io import BytesIO
import json
import hashlib

import pandas as pd
import streamlit as st

from src.config import FEATURE_COLS, TARGET_COLS, UNIT_MAP
from src.records import TEMPLATE_FIELDS, VALUE_FIELDS, validate_record, preview_file, template_bytes
from src.jobs import create_job, list_jobs
from src.version_store import read_json

STATUS = {"pending": "待审核", "approved": "已通过", "rejected": "已驳回",
          "queued": "等待执行", "running": "运行中", "succeeded": "已完成", "failed": "失败或中断"}


def action_error(error):
    message = str(error)
    st.error(message if isinstance(error, ValueError) and any("\u4e00" <= c <= "\u9fff" for c in message)
             else "操作未完成，正式版本未切换。请检查本地文件与日志。")


def review_action(store, identifier, revision, approve):
    try:
        store.review(identifier, revision, approve, st.session_state.get("review_actual_" + identifier, False),
                     st.session_state.get("review_reason_" + identifier, ""))
        st.session_state["record_message"] = "审核已生效，原记录与审核原因已留存。"
    except ValueError as error:
        st.session_state["record_error"] = str(error)


def switch_action(store, selected, active_id, rollback):
    try:
        store.switch(selected, st.session_state.get("switch_reason_" + selected, ""),
                     st.session_state.get("switch_confirm_" + selected, False), active_id, rollback=rollback)
        st.session_state["version_changed_message"] = f"已{'回滚' if rollback else '启用'}版本 {selected}；新实测记录仍保留。"
    except (ValueError, OSError) as error:
        st.session_state["version_error"] = str(error)


def input_fields(prefix, values=None, lock_id=False):
    values = values or {}
    result = {}
    for key in TEMPLATE_FIELDS:
        label = f"{key}（{UNIT_MAP[key]}）" if key in UNIT_MAP else key
        result[key] = st.text_input(label, value=str(values.get(key, "")), key=f"{prefix}_{key}",
                                    disabled=lock_id and key == "实验编号")
    return result


def entries_notice(store, entries, current_data):
    existing = {r["id"]: r for r in store.list_records()}
    numbers = current_data[VALUE_FIELDS].to_numpy(dtype=float).tolist()
    for record in existing.values():
        numbers.append([record["payload"][c] for c in VALUE_FIELDS])
    for entry in entries:
        value = entry["values"]
        if value["实验编号"] in existing:
            st.warning(f"编号 {value['实验编号']} 已存在：相同内容会忽略重复提交，不同内容将整批拒绝。")
        vector = [value[c] for c in VALUE_FIELDS]
        if vector in numbers:
            st.warning(f"编号 {value['实验编号']} 的四项数值与其他记录完全相同，请核实是否真实重复实验；不会自动删除。")
        numbers.append(vector)
        if any(not current_data[c].min() <= value[c] <= current_data[c].max() for c in FEATURE_COLS):
            st.warning(f"编号 {value['实验编号']} 超出当前模型训练范围。可以审核，但旧模型仍拒绝外推；只有启用新版本后才扩展范围。")


def render_records(store, current_data):
    with st.container(key="records_panel"):
        st.subheader("实测数据 · 先审核，再训练")
        if st.session_state.get("record_message"):
            st.success(st.session_state.pop("record_message"))
        if st.session_state.get("record_error"):
            st.error(st.session_state.pop("record_error"))
        st.button("刷新实测列表", key="refresh_records")
        st.caption("仅供本机单人使用。实验编号对应一次实际测量；程序记录来源声明，不认证实验真实性。初始合成记录不会被编辑。")
        st.download_button("下载CSV导入模板", template_bytes(), file_name="experiment_template.csv", mime="text/csv")
        mode = st.radio("录入方式", ["逐条填写", "文件导入"], horizontal=True, key="entry_mode")
        fingerprint = "form"
        blob, filename, sheet = None, "", None
        if mode == "逐条填写":
            with st.form("new_experiment_form"):
                values = input_fields("new")
                checked = st.form_submit_button("检查单条记录")
            if checked:
                st.session_state.pop("entry_preview", None)
                try:
                    st.session_state["entry_preview"] = {"fingerprint": fingerprint, "entries": [{"values": validate_record(values), "source_row": None}]}
                except ValueError as error:
                    action_error(error)
        else:
            uploaded = st.file_uploader("选择实测CSV或XLSX", type=["csv", "xlsx"], key="experiment_upload")
            if uploaded is not None:
                blob, filename = uploaded.getvalue(), uploaded.name
                try:
                    if filename.lower().endswith(".xlsx"):
                        sheets = pd.ExcelFile(BytesIO(blob)).sheet_names
                        sheet = st.selectbox("明确选择工作表", ["请选择工作表", *sheets], key="upload_sheet")
                        if sheet == "请选择工作表":
                            sheet = None
                    fingerprint = hashlib.sha256(blob).hexdigest() + ":" + str(sheet)
                    if st.button("检查导入", key="inspect_upload"):
                        st.session_state.pop("entry_preview", None)
                        entries, errors = preview_file(blob, filename, sheet)
                        if errors:
                            st.error("本次导入存在错误，整批未提交。请修正后重新检查。")
                            st.dataframe(pd.DataFrame(errors), hide_index=True, use_container_width=True)
                        else:
                            st.session_state["entry_preview"] = {"fingerprint": fingerprint, "entries": entries}
                except Exception as error:
                    message = str(error)
                    st.error(message if isinstance(error, ValueError) and any("\u4e00" <= c <= "\u9fff" for c in message)
                             else "无法读取文件，请检查格式、模板表头及工作表选择。")
            else:
                fingerprint = None
        preview = st.session_state.get("entry_preview")
        if preview and preview["fingerprint"] == fingerprint:
            st.markdown("**待提交快照：请核对以下数值，而不是尚未重新检查的表单改动。**")
            st.dataframe(pd.DataFrame([{**e["values"], "原始行号": e["source_row"]} for e in preview["entries"]]), hide_index=True, use_container_width=True)
            entries_notice(store, preview["entries"], current_data)
            actual = st.checkbox("我确认以上数据来自实际测量，不是模型预测值", key="submit_actual")
            if st.button("提交到待审核区", key="submit_records"):
                try:
                    count = store.submit(preview["entries"], actual, blob, filename, sheet)
                    st.success(f"新增 {count} 条待审核记录；尚未参与训练。重复提交不新增记录。")
                except (ValueError, OSError) as error:
                    action_error(error)
        records = store.list_records()
        st.markdown("### 新实测记录")
        if records:
            st.dataframe(pd.DataFrame([{**r["payload"], "审核状态": STATUS[r["status"]], "修订号": r["revision"]} for r in records]), hide_index=True, use_container_width=True)
        else:
            st.info("尚无新增实测。初始合成记录仅作为只读训练基线，不伪造实验编号或批次。")
        pending = [r for r in records if r["status"] == "pending"]
        if pending:
            st.markdown("### 审核与待审核修改")
            selected = st.selectbox("选择待审核实验", [r["id"] for r in pending], key="review_id")
            record = next(r for r in pending if r["id"] == selected)
            st.dataframe(pd.DataFrame([record["payload"]]), hide_index=True, use_container_width=True)
            st.caption("来源：" + json.dumps(record["provenance"], ensure_ascii=False))
            with st.expander("修改这条待审核记录"):
                with st.form("edit_pending_" + selected + str(record["revision"])):
                    values = input_fields("edit_" + selected + str(record["revision"]), record["payload"], lock_id=True)
                    if st.form_submit_button("保存待审核修改"):
                        try:
                            store.edit_pending(selected, values, record["revision"])
                            st.success("修改已保存，请刷新实测列表后审核最新修订。")
                        except ValueError as error:
                            action_error(error)
            reason = st.text_area("审核原因（通过或驳回均必填）", key="review_reason_" + selected)
            confirm = st.checkbox("审核确认：已核对来源，为实际测量且不是模型生成", key="review_actual_" + selected)
            st.button("审核通过", key="approve_record", on_click=review_action,
                      args=(store, selected, record["revision"], True))
            st.button("驳回记录", key="reject_record", on_click=review_action,
                      args=(store, selected, record["revision"], False))
        with st.expander("查看录入与审核事件"):
            events = [e for e in store.events() if "模型" not in e["action"] and "任务" not in e["action"] and "比较" not in e["action"]]
            st.dataframe(pd.DataFrame(events), hide_index=True, use_container_width=True)


def comparison_table(comparison):
    labels = {"old": "旧训练方案", "candidate": "候选训练方案"}
    names = {"gpr": "GPR（主模型）", "linear": "线性对照", "quadratic": "加性二次对照"}
    return pd.DataFrame([{"目标": r["target"], "模型": names[r["model"]], "方案": labels[r["side"]],
                          "可完整比较": "是" if r["complete"] else "否，保留失败折", "共同测试行数": r["n_test"],
                          **{label: r.get(key) for label, key in (("MAE", "mae"), ("RMSE", "rmse"), ("R²", "r2"), ("区间覆盖率", "coverage"), ("平均宽度", "mean_width"))}}
                         for r in comparison["summary"]])


def render_versions(store, active_id):
    with st.container(key="versions_panel"):
        st.subheader("模型版本 · 候选不等于正式启用")
        if st.session_state.get("version_changed_message"):
            st.success(st.session_state.pop("version_changed_message"))
        if st.session_state.get("version_error"):
            st.error(st.session_state.pop("version_error"))
        st.caption(f"当前版本：{active_id}。录入与审核不会自动训练，训练也不会自动启用。")
        if st.button("创建候选训练", key="train_candidate"):
            try:
                job = create_job(store)
                st.success(f"任务 {job} 已创建或正在运行。快照已冻结；当前模型不变。")
            except Exception as error:
                action_error(error)
        st.button("刷新任务与版本", key="refresh_versions")
        jobs = list_jobs(store)
        if jobs:
            st.dataframe(pd.DataFrame([{k: STATUS.get(r[k], r[k]) if k == "status" else r[k]
                                       for k in ("id", "kind", "status", "created", "updated", "message")} for r in jobs]).rename(columns={"id": "任务", "kind": "类型", "status": "状态", "created": "创建时间", "updated": "更新时间", "message": "进展"}),
                         hide_index=True, use_container_width=True)
            log_job = st.selectbox("查看任务日志", [r["id"] for r in jobs], key="log_job")
            logfile = store.root / "jobs" / log_job / "worker.log"
            if logfile.exists():
                with st.expander("工作进程日志（刷新查看更新）"):
                    st.text(logfile.read_text(encoding="utf-8", errors="replace")[-12000:])
        versions = store.list_versions()
        selected = st.selectbox("选择模型版本", [r["id"] for r in versions], key="selected_version")
        st.dataframe(pd.DataFrame(versions).rename(columns={"id": "版本", "created": "创建时间", "parent": "创建时基准版本"}), hide_index=True, use_container_width=True)
        try:
            directory = store.verify(selected)
            metadata = read_json(directory / "training_metadata.json")
            st.write(f"版本训练记录数：{metadata['n_samples']}；训练范围：{metadata['feature_bounds']}")
            st.download_button("下载该版本完整验证报告", (directory / "evaluation.md").read_bytes(), file_name="evaluation.md")
            comparison = None
            if selected != "initial":
                try:
                    comparison = store.comparison(selected)
                except (OSError, ValueError) as error:
                    st.warning(str(error) if isinstance(error, ValueError) else "比较文件缺失或无法读取，不能据此启用。")
                if comparison:
                    st.caption(comparison["note"])
                    st.caption(f"共同测试比较的旧版本：{comparison['against']}。这是内部交叉验证，不是独立未来实验。")
                    st.dataframe(comparison_table(comparison), hide_index=True, use_container_width=True)
                    risks = comparison.get("risks", [])
                    if risks:
                        st.warning("\n\n".join(risks))
                    st.info("即使数字变好，也不保证未来实验改进；区间变窄不能单独当作提升依据。批次信息仅追溯，未完全排除批次依赖。")
                    with st.expander("同口径比较的逐折记录"):
                        st.json(comparison["folds"])
                    st.download_button("下载共同测试逐行结果", pd.DataFrame(comparison["predictions"]).to_csv(index=False).encode("utf-8-sig"), file_name="comparison_predictions.csv")
                if st.button("针对当前版本重新比较", key="recompare"):
                    job = create_job(store, kind="compare", candidate=selected)
                    st.success(f"比较任务 {job} 已创建或正在运行；不会重训全量模型或自动启用。")
            if selected != active_id:
                stale = not comparison or comparison["against"] != active_id
                if selected != "initial" and stale:
                    st.warning("比较缺失或当前版本已经变化：不能直接启用，请先重新比较。历史回滚不需要重新比较。")
                reason = st.text_area("启用或回滚原因（必填）", key="switch_reason_" + selected)
                confirmed = st.checkbox("我已阅读指标、退步项与数据支持风险，确认整体切换该版本", key="switch_confirm_" + selected)
                st.button("确认启用所选候选", key="promote_version", disabled=selected == "initial" or stale,
                          on_click=switch_action, args=(store, selected, active_id, False))
                historical = {"initial"} | {json.loads(e["detail"])["to"] for e in store.events() if e["action"] in ("启用模型", "回滚模型")}
                st.caption("回滚仅限曾正式启用的历史版本；未启用候选应使用上方启用操作。")
                st.button("确认回滚到历史版本", key="rollback_version", disabled=selected not in historical,
                          on_click=switch_action, args=(store, selected, active_id, True))
        except Exception as error:
            action_error(error)
        with st.expander("查看版本操作历史"):
            st.dataframe(pd.DataFrame([e for e in store.events() if "模型" in e["action"] or "比较" in e["action"] or "任务" in e["action"]]), hide_index=True, use_container_width=True)
