"""本机SQLite版本库：数据审核、不可变快照及原子模型切换。"""

from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import sqlite3
import uuid

import pandas as pd

from src.config import PROJECT_ROOT, FEATURE_COLS, TARGET_COLS
from src.records import validate_record


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def now():
    return datetime.now(timezone.utc).isoformat()


def store_root():
    return Path(os.environ.get("MATERIAL_STORE", str(PROJECT_ROOT / "local_store"))).resolve()


def runtime_bundle():
    """每次调用只解析一次当前版本；已有版本库损坏时绝不静默回退。"""
    store = Store()
    if store.db.exists():
        return store.bundle()
    from src.config import MODEL_PATHS, TRAINING_METADATA_PATH, DATA_PATH
    return {"id": "legacy", "directory": PROJECT_ROOT / "models", "source": Path(DATA_PATH),
            "metadata": read_json(TRAINING_METADATA_PATH), "model_paths": MODEL_PATHS}


class Store:
    def __init__(self, root=None):
        self.root = Path(root).resolve() if root else store_root()
        self.db = self.root / "registry.sqlite3"

    @contextmanager
    def connection(self, write=False):
        connection = sqlite3.connect(self.db, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def event(self, con, action, detail):
        con.execute("INSERT INTO events(created,action,detail) VALUES(?,?,?)", (now(), action, json.dumps(detail, ensure_ascii=False)))

    def initialize(self):
        """归档原模型，不重训；原文件与原报告一个字节也不修改。"""
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("versions", "uploads", "jobs"):
            (self.root / name).mkdir(exist_ok=True)
        with self.connection() as con:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS experiments(id TEXT PRIMARY KEY,payload TEXT NOT NULL,status TEXT NOT NULL,
                revision INTEGER NOT NULL,provenance TEXT NOT NULL,created TEXT NOT NULL,updated TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS versions(id TEXT PRIMARY KEY,created TEXT NOT NULL,parent TEXT);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,created TEXT NOT NULL,action TEXT NOT NULL,detail TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,kind TEXT NOT NULL,status TEXT NOT NULL,created TEXT NOT NULL,
                updated TEXT NOT NULL,pid INTEGER,message TEXT NOT NULL,token TEXT NOT NULL);
            CREATE UNIQUE INDEX IF NOT EXISTS one_job ON jobs((1)) WHERE status IN ('queued','running');
            """)
        with self.connection(write=True) as con:
            if con.execute("SELECT value FROM settings WHERE key='active'").fetchone():
                return
            from src.config import TRAINING_METADATA_PATH, MODEL_PATHS
            from src.load_data import load_dataset
            import joblib
            from app.validation_view import check_source, load_validation
            metadata = read_json(TRAINING_METADATA_PATH)
            source = Path(metadata["data_path"])
            with redirect_stdout(StringIO()):
                data = load_dataset(source, strict=True, log_path=self.root / "initial_loading.md")
            models = {t: joblib.load(p) for t, p in MODEL_PATHS.items()}
            check_source(source, data, models, metadata)
            load_validation(PROJECT_ROOT / "reports/cross_validation.json", metadata, data)
            staging = self.root / "versions" / (".initial-" + uuid.uuid4().hex)
            staging.mkdir()
            shutil.copy2(source, staging / ("source" + source.suffix.lower()))
            for target, path in MODEL_PATHS.items():
                shutil.copy2(path, staging / f"model_{target}.joblib")
            shutil.copy2(TRAINING_METADATA_PATH, staging / "training_metadata.json")
            for name in ("cross_validation.json", "oof_predictions.csv", "evaluation.md"):
                shutil.copy2(PROJECT_ROOT / "reports" / name, staging / name)
            records = [{"record_id": f"baseline:{metadata['data_sha256']}:{i}",
                        "values": {c: float(row[c]) for c in [*FEATURE_COLS, *TARGET_COLS]},
                        "provenance": {"kind": "迁移标识，原实验编号与批次未提供", "source_index": i}}
                       for i, row in data.iterrows()]
            write_json(staging / "records.json", records)
            write_json(staging / "version.json", {"id": "initial", "source": "source" + source.suffix.lower(),
                       "parent": None, "created": now(), "kind": "原模型归档，未重训"})
            self.seal(staging)
            destination = self.root / "versions/initial"
            if destination.exists():
                self.verify("initial")
                # 中断遗留的归档只允许与本次完全相同的训练源恢复。
                if read_json(destination / "training_metadata.json") != metadata:
                    raise ValueError("已有初始归档不匹配，不能自动覆盖。")
            else:
                staging.rename(destination)
            con.execute("INSERT OR IGNORE INTO versions VALUES(?,?,?)", ("initial", now(), None))
            con.execute("INSERT INTO settings VALUES('active','initial')")
            self.event(con, "初始化", {"version": "initial", "source_sha256": metadata["data_sha256"]})

    @staticmethod
    def seal(folder):
        files = {p.relative_to(folder).as_posix(): digest(p) for p in folder.rglob("*") if p.is_file() and p.name != "manifest.json"}
        write_json(folder / "manifest.json", files)

    def version_dir(self, version):
        if not version or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in version):
            raise ValueError("无效的版本标识。")
        return self.root / "versions" / version

    def verify(self, version):
        directory = self.version_dir(version)
        manifest = read_json(directory / "manifest.json")
        required = {"version.json", "records.json", "training_metadata.json", "cross_validation.json", "evaluation.md"}
        required.update(f"model_{t}.joblib" for t in TARGET_COLS)
        if not required.issubset(manifest):
            raise ValueError("版本清单不完整，禁止使用。")
        for name, expected in manifest.items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()) or not path.is_file() or digest(path) != expected:
                raise ValueError("版本文件缺失或指纹不匹配，禁止使用。")
        descriptor = read_json(directory / "version.json")
        if descriptor["id"] != version or descriptor["source"] not in manifest:
            raise ValueError("版本来源描述不完整。")
        return directory

    def active_id(self):
        with self.connection() as con:
            row = con.execute("SELECT value FROM settings WHERE key='active'").fetchone()
        if not row:
            raise ValueError("版本库尚未完成初始化。")
        return row[0]

    def bundle(self, version=None):
        version = version or self.active_id()
        directory = self.verify(version)
        descriptor = read_json(directory / "version.json")
        return {"id": version, "directory": directory, "source": directory / descriptor["source"],
                "metadata": read_json(directory / "training_metadata.json"),
                "model_paths": {t: directory / f"model_{t}.joblib" for t in TARGET_COLS}}

    def list_records(self, status=None):
        with self.connection() as con:
            rows = con.execute("SELECT * FROM experiments" + (" WHERE status=?" if status else "") + " ORDER BY created,id",
                               (status,) if status else ()).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"]), "provenance": json.loads(row["provenance"])} for row in rows]

    def submit(self, entries, actual, blob=None, filename="", sheet=None):
        if actual is not True:
            raise ValueError("只有声明为实际测量的数据才能提交；模型预测值不得入库。")
        if not entries:
            raise ValueError("没有可提交记录。")
        checked = [validate_record(item["values"]) for item in entries]
        if blob is not None:
            from src.records import preview_file
            parsed, errors = preview_file(blob, filename, sheet)
            if errors or len(parsed) != len(entries) or any(
                p["values"] != values or p["source_row"] != entry.get("source_row")
                for p, values, entry in zip(parsed, checked, entries)
            ):
                raise ValueError("导入文件有错误或提交不完整，整批未写入；请重新预览修正。")
        ids = [r["实验编号"] for r in checked]
        if len(set(ids)) != len(ids):
            raise ValueError("本次提交实验编号重复，整批未写入。")
        sha = hashlib.sha256(blob).hexdigest() if blob is not None else None
        with self.connection(write=True) as con:
            additions = []
            for original, values in zip(entries, checked):
                previous = con.execute("SELECT payload FROM experiments WHERE id=?", (values["实验编号"],)).fetchone()
                if previous:
                    if json.loads(previous[0]) != values:
                        raise ValueError("实验编号已存在且内容不同，整批未写入。")
                    continue
                additions.append((original, values))
            if blob is not None:
                suffix = Path(filename).suffix.lower()
                if suffix not in (".csv", ".xlsx"):
                    raise ValueError("无效上传文件格式。")
                path = self.root / "uploads" / (sha + suffix)
                if not path.exists():
                    path.write_bytes(blob)
                elif digest(path) != sha:
                    raise ValueError("已存上传文件指纹错误。")
            for original, values in additions:
                provenance = {"actual_declared": True, "sha256": sha, "filename": Path(filename).name,
                              "sheet": sheet, "source_row": original.get("source_row")}
                con.execute("INSERT INTO experiments VALUES(?,?, 'pending',1,?,?,?)",
                            (values["实验编号"], json.dumps(values, ensure_ascii=False), json.dumps(provenance, ensure_ascii=False), now(), now()))
                self.event(con, "提交实测待审核", {"id": values["实验编号"], "values": values, "provenance": provenance})
        return len(additions)

    def edit_pending(self, experiment_id, values, revision):
        values = validate_record(values)
        if values["实验编号"] != experiment_id:
            raise ValueError("修改待审核记录时不能更换编号。")
        with self.connection(write=True) as con:
            row = con.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            if not row or row["status"] != "pending" or row["revision"] != revision:
                raise ValueError("记录已经变化或不再待审核，请刷新。")
            con.execute("UPDATE experiments SET payload=?,revision=revision+1,updated=? WHERE id=?",
                        (json.dumps(values, ensure_ascii=False), now(), experiment_id))
            self.event(con, "修改待审核记录", {"id": experiment_id, "before": json.loads(row["payload"]), "after": values})

    def review(self, experiment_id, revision, approve, actual, reason):
        if not reason.strip() or (approve and actual is not True):
            raise ValueError("审核必须填写原因；通过还需独立确认是真实测量。")
        with self.connection(write=True) as con:
            row = con.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            if not row or row["status"] != "pending" or row["revision"] != revision:
                raise ValueError("记录已经变化或已审核，请刷新后重试。")
            if approve:
                validate_record(json.loads(row["payload"]))
                if json.loads(row["provenance"]).get("actual_declared") is not True:
                    raise ValueError("缺少实际测量声明，不能通过。")
            status = "approved" if approve else "rejected"
            con.execute("UPDATE experiments SET status=?,revision=revision+1,updated=? WHERE id=?", (status, now(), experiment_id))
            self.event(con, "审核通过" if approve else "审核驳回", {"id": experiment_id, "revision": revision, "reason": reason.strip(), "actual_confirmed": actual})

    def snapshot_records(self, con):
        records = read_json(self.verify("initial") / "records.json")
        for row in con.execute("SELECT * FROM experiments WHERE status='approved' ORDER BY created,id"):
            values = validate_record(json.loads(row["payload"]))
            provenance = json.loads(row["provenance"])
            if provenance.get("actual_declared") is not True:
                raise ValueError("已审核记录缺少实测声明，停止冻结。")
            records.append({"record_id": "experiment:" + row["id"], "values": values,
                            "provenance": provenance, "reviewed_revision": row["revision"]})
        return records

    def list_versions(self):
        with self.connection() as con:
            return [dict(r) for r in con.execute("SELECT * FROM versions ORDER BY created DESC")]

    def events(self):
        with self.connection() as con:
            return [dict(r) for r in con.execute("SELECT * FROM events ORDER BY id DESC")]

    def switch(self, version, reason, confirmed, expected_active, rollback=False):
        if not reason.strip() or confirmed is not True:
            raise ValueError("必须填写原因并再次确认风险。")
        with self.connection(write=True) as con:
            active = con.execute("SELECT value FROM settings WHERE key='active'").fetchone()[0]
            if active != expected_active:
                raise ValueError("当前版本已变化，请刷新并重新比较。")
            if active == version:
                raise ValueError("该版本已经启用。")
            if not con.execute("SELECT 1 FROM versions WHERE id=?", (version,)).fetchone():
                raise ValueError("版本尚未完成训练。")
            directory = self.verify(version)
            if rollback:
                previous = [json.loads(r[0]).get("to") for r in con.execute("SELECT detail FROM events WHERE action IN ('启用模型','回滚模型')")]
                if version != "initial" and version not in previous:
                    raise ValueError("只能回滚到曾正式启用的有效历史版本。")
            else:
                comparison = self.comparison(version)
                if comparison["against"] != active or comparison["candidate"] != version:
                    raise ValueError("比较绑定的旧版本已变化，必须重新比较。")
            # 再做语义核对，不能仅凭存在模型文件就切换。
            import joblib
            from app.validation_view import check_source, load_validation
            bundle = self.bundle(version)
            metadata = bundle["metadata"]
            records = read_json(directory / "records.json")
            data = pd.DataFrame([r["values"] for r in records])[[*FEATURE_COLS, *TARGET_COLS]]
            models = {t: joblib.load(p) for t, p in bundle["model_paths"].items()}
            check_source(bundle["source"], data, models, metadata)
            load_validation(directory / "cross_validation.json", metadata, data)
            con.execute("UPDATE settings SET value=? WHERE key='active'", (version,))
            self.event(con, "回滚模型" if rollback else "启用模型", {"from": active, "to": version, "reason": reason.strip(), "risk_confirmed": True})

    def comparison(self, version):
        self.verify(version)
        with self.connection() as con:
            row = con.execute("SELECT detail FROM events WHERE action='比较完成' ORDER BY id DESC").fetchall()
        for item in row:
            detail = json.loads(item[0])
            if detail["version"] == version:
                path = self.root / "jobs" / detail["job"] / "comparison.json"
                if digest(path) != detail["sha256"]:
                    raise ValueError("比较文件指纹错误，禁止启用。")
                return read_json(path)
        raise ValueError("该版本尚无有效比较报告。")
