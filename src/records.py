"""实测记录的统一检查和无副作用导入预览。"""

from io import BytesIO, StringIO
import csv
import math
from pathlib import Path

import pandas as pd

from src.config import FEATURE_COLS, TARGET_COLS
from src.load_data import map_column_name

TEXT_FIELDS = ["实验编号", "批次", "来源说明"]
VALUE_FIELDS = [*FEATURE_COLS, *TARGET_COLS]
TEMPLATE_FIELDS = [*TEXT_FIELDS[:2], *VALUE_FIELDS, "来源说明", "备注"]


def validate_record(record):
    """只接受完整实际测量，不推断或填补缺失值。"""
    result = {}
    for key in TEXT_FIELDS:
        value = record.get(key)
        if value is None or pd.isna(value) or not str(value).strip():
            raise ValueError(f"{key}不能为空。")
        result[key] = str(value).strip()
    if result["实验编号"].startswith("baseline:"):
        raise ValueError("实验编号不能使用系统迁移编号前缀。")
    for key in VALUE_FIELDS:
        try:
            value = float(record[key])
        except (KeyError, ValueError, TypeError):
            raise ValueError(f"{key}必须是完整数值。") from None
        if not math.isfinite(value):
            raise ValueError(f"{key}必须是有限数值。")
        result[key] = value
    if result[FEATURE_COLS[0]] <= 0:
        raise ValueError("铝相对剂量必须大于零。")
    if result[TARGET_COLS[0]] < 0:
        raise ValueError("比容量不能为负数。")
    cycles = result[TARGET_COLS[1]]
    if cycles < 0 or not cycles.is_integer():
        raise ValueError("循环圈数必须是非负整数。")
    note = record.get("备注", "")
    result["备注"] = "" if note is None or pd.isna(note) else str(note).strip()
    return result


def preview_file(blob, filename, sheet=None):
    """保留物理行号；XLSX必须明确选择表，不自动拼接工作表。"""
    extension = Path(filename).suffix.lower()
    if extension == ".xlsx":
        excel = pd.ExcelFile(BytesIO(blob))
        if sheet not in excel.sheet_names:
            raise ValueError("请选择文件中实际存在的工作表。")
        raw = pd.read_excel(excel, sheet_name=sheet, dtype=object, header=None)
        if raw.empty:
            raise ValueError("文件没有表头和实验记录。")
        frame = raw.iloc[1:].copy()
        frame.columns = raw.iloc[0].tolist()
        frame.index = range(len(frame))
        source_rows = list(range(2, len(frame) + 2))
    elif extension == ".csv":
        try:
            text = blob.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = blob.decode("gb18030")
        reader = csv.reader(StringIO(text), strict=True)
        headers = next(reader, None)
        if not headers:
            raise ValueError("文件没有表头。")
        rows, source_rows = [], []
        while True:
            start_row = reader.line_num + 1
            row = next(reader, None)
            if row is None:
                break
            if len(row) != len(headers):
                raise ValueError(f"原始第{start_row}行列数与表头不一致；整批未提交。")
            rows.append(row)
            source_rows.append(start_row)
        frame = pd.DataFrame(rows, columns=headers)
    else:
        raise ValueError("仅支持CSV和XLSX，不接收模型文件或其他格式。")
    # 只做明确的列名映射，不猜表头、不填补合并单元格。
    columns = [str(c).strip() if str(c).strip() in TEMPLATE_FIELDS
               else map_column_name(c) or str(c).strip() for c in frame.columns]
    if len(set(columns)) != len(columns):
        raise ValueError("列名映射后重复，请按模板修正。")
    frame.columns = columns
    missing = [key for key in [*TEXT_FIELDS, *VALUE_FIELDS] if key not in columns]
    if missing:
        raise ValueError("缺少必需列：" + "、".join(missing))
    if frame.empty:
        raise ValueError("文件没有实验记录。")
    records, errors, seen = [], [], set()
    for index, source in frame.iterrows():
        row_number = source_rows[int(index)]
        try:
            record = validate_record(source.to_dict())
            if record["实验编号"] in seen:
                raise ValueError("同一导入文件中实验编号重复。")
            seen.add(record["实验编号"])
            records.append({"values": record, "source_row": row_number, "sheet": sheet})
        except ValueError as error:
            errors.append({"原始行号": row_number, "问题": str(error)})
    return records, errors


def template_bytes():
    return (",".join(TEMPLATE_FIELDS) + "\n").encode("utf-8-sig")
