"""跨进程单任务训练；进程中断不会改变当前版本。"""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import traceback
import uuid

from src.config import PROJECT_ROOT
from src.version_store import Store, now, read_json, write_json, digest

# 保留子进程句柄，避免任务状态先更新而日志文件仍占用时过早清理。
_PROCESSES = {}


def wait_job_exit(identifier, timeout=60):
    process = _PROCESSES.get(identifier)
    if process is not None:
        process.wait(timeout=timeout)
        _PROCESSES.pop(identifier, None)


@contextmanager
def process_lock(root, blocking=True):
    path = Path(root) / "worker.lock"
    with open(path, "a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def recover_interrupted(store):
    """只有拿到进程锁才判定遗留任务已中断，不根据页面刷新误判。"""
    try:
        with process_lock(store.root, blocking=False):
            with store.connection(write=True) as con:
                rows = con.execute("SELECT * FROM jobs WHERE status IN ('running','queued')").fetchall()
                for row in rows:
                    from datetime import datetime
                    if time.time() - datetime.fromisoformat(row["updated"]).timestamp() < 60:
                        continue
                    con.execute("UPDATE jobs SET status='failed',updated=?,message=? WHERE id=?", (now(), "工作进程已中断；当前模型未改变，可创建新任务。", row["id"]))
                    store.event(con, "任务中断恢复", {"job": row["id"]})
    except (OSError, BlockingIOError):
        pass


def list_jobs(store):
    for identifier, process in list(_PROCESSES.items()):
        if process.poll() is not None:
            _PROCESSES.pop(identifier, None)
    recover_interrupted(store)
    with store.connection() as con:
        return [dict(r) for r in con.execute("SELECT * FROM jobs ORDER BY created DESC")]


def create_job(store, kind="train", candidate=None, source=None, spawn=True, reject_existing=False):
    if kind != "train" and source is not None:
        raise ValueError("重新比较任务不能接收外部训练文件。")
    store.initialize()
    recover_interrupted(store)
    with store.connection(write=True) as con:
        running = con.execute("SELECT id FROM jobs WHERE status IN ('queued','running')").fetchone()
        if running:
            if reject_existing:
                raise ValueError("已有训练或比较任务运行，本次不启动第二个任务。")
            return running[0]
        against = con.execute("SELECT value FROM settings WHERE key='active'").fetchone()[0]
        records = store.snapshot_records(con) if kind == "train" else None
        if kind == "train" and source is not None:
            source_hash = digest(source)
            from contextlib import redirect_stdout
            from io import StringIO
            import numpy as np
            from src.config import FEATURE_COLS, TARGET_COLS
            from src.load_data import load_dataset
            from src.training import records_frame
            with redirect_stdout(StringIO()):
                provided = load_dataset(source, strict=True, log_path=store.root / "cli_loading.md")
            expected = records_frame(records)
            if digest(source) != source_hash:
                raise ValueError("--data文件在检查期间变化，停止创建候选任务。")
            if provided.shape[0] != len(expected) or not np.array_equal(provided[[*FEATURE_COLS, *TARGET_COLS]].to_numpy(dtype=float), expected.to_numpy()):
                raise ValueError("--data内容必须与初始数据加已审核实测快照一致；新数据请先导入并审核，不能绕过审核训练。")
        if kind == "compare":
            if not candidate:
                raise ValueError("缺少候选版本。")
            store.verify(candidate)
        elif kind != "train":
            raise ValueError("未知任务类型。")
        identifier = "v" + uuid.uuid4().hex[:16]
        token = uuid.uuid4().hex
        folder = store.root / "jobs" / identifier
        folder.mkdir()
        request = {"id": identifier, "kind": kind, "created": now(), "against": against,
                   "candidate": candidate, "records": records, "token": token,
                   "cli_source": {"path": str(Path(source).resolve()), "sha256": source_hash} if source else None}
        write_json(folder / "request.json", request)
        request_hash = digest(folder / "request.json")
        con.execute("INSERT INTO jobs VALUES(?,?, 'queued',?,?,NULL,?,?)", (identifier, kind, now(), now(), "已冻结快照，等待工作进程。", request_hash))
        store.event(con, "创建候选训练" if kind == "train" else "创建重新比较", {"job": identifier, "against": against, "candidate": candidate, "request_sha256": request_hash})
    if spawn:
        try:
            with open(folder / "worker.log", "wb") as output:
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT / ".deps"), str(PROJECT_ROOT), env.get("PYTHONPATH", "")])
                process = subprocess.Popen([sys.executable, "-m", "src.jobs", "--store", str(store.root), "--job", identifier],
                                           cwd=PROJECT_ROOT, stdout=output, stderr=subprocess.STDOUT, env=env,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                _PROCESSES[identifier] = process
            with store.connection(write=True) as con:
                con.execute("UPDATE jobs SET pid=? WHERE id=?", (process.pid, identifier))
        except OSError:
            with store.connection(write=True) as con:
                con.execute("UPDATE jobs SET status='failed',updated=?,message=? WHERE id=?", (now(), "无法启动工作进程；当前模型未改变。", identifier))
            raise
    return identifier


def run_job(store, identifier):
    from src.training import train_candidate, compare_datasets, comparison_risks
    folder = store.root / "jobs" / identifier
    with process_lock(store.root):
        with store.connection() as con:
            existing = con.execute("SELECT status FROM jobs WHERE id=?", (identifier,)).fetchone()
        if not existing or existing[0] != "queued":
            print("任务不再等待执行，忽略重复启动。", flush=True)
            return
        try:
            with store.connection(write=True) as con:
                job = con.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
                if not job or job["status"] != "queued":
                    raise ValueError("任务不再等待执行，不重复训练。")
                if digest(folder / "request.json") != job["token"]:
                    raise ValueError("冻结请求指纹错误，拒绝训练。")
                request = read_json(folder / "request.json")
                con.execute("UPDATE jobs SET status='running',pid=?,updated=?,message=? WHERE id=?", (os.getpid(), now(), "正在验证训练，当前模型保持不变。", identifier))
            old_folder = store.verify(request["against"])
            if request["kind"] == "train":
                staging = train_candidate(store, identifier, request)
                candidate_id, records = identifier, request["records"]
            else:
                candidate_id = request["candidate"]
                records = read_json(store.verify(candidate_id) / "records.json")
            print("开始同条件共同测试比较。", flush=True)
            comparison = compare_datasets(read_json(old_folder / "records.json"), records, request["against"], candidate_id)
            comparison["risks"] = comparison_risks(comparison)
            write_json(folder / "comparison.json", comparison)
            import pandas as pd
            pd.DataFrame(comparison["predictions"]).to_csv(folder / "comparison_predictions.csv", index=False, encoding="utf-8-sig")
            if request["kind"] == "train":
                store.seal(staging)
                staging.rename(store.version_dir(candidate_id))
            with store.connection(write=True) as con:
                if request["kind"] == "train":
                    con.execute("INSERT INTO versions VALUES(?,?,?)", (candidate_id, now(), request["against"]))
                store.event(con, "比较完成", {"version": candidate_id, "against": request["against"], "job": identifier,
                                          "sha256": digest(folder / "comparison.json")})
                con.execute("UPDATE jobs SET status='succeeded',updated=?,message=? WHERE id=?", (now(), "候选验证与比较完成，尚未启用。", identifier))
            print(f"任务 {identifier} 完成；当前模型未切换。", flush=True)
        except BaseException as error:
            with store.connection(write=True) as con:
                con.execute("UPDATE jobs SET status='failed',updated=?,message=? WHERE id=?", (now(), "任务失败或中断，当前模型未改变。请查看日志。", identifier))
                store.event(con, "任务失败", {"job": identifier, "error_type": type(error).__name__})
            print(f"任务失败：{error if isinstance(error, ValueError) else '运行异常，详细堆栈供排查，当前版本未改变。'}", flush=True)
            traceback.print_exc()
            raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="内部候选训练工作进程")
    parser.add_argument("--store", required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    run_job(Store(args.store), args.job)
