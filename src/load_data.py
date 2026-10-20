"""健壮的数据加载层：处理未知格式的真实实验数据。

设计目标（对应「明天拿到格式未知的数据」这一前提）：
1. 编码不确定          → utf-8-sig / gbk / gb18030 依次尝试
2. 表头行不在第 0 行    → 扫描前若干行，找与 COLUMN_MAPPING 命中最多的一行作为表头
3. 有全空行、全空列    → 读取后剔除
4. 列名和预想不一样    → 全部经 COLUMN_MAPPING 映射，不匹配的保留原名并告警
5. 数字被存成文本      → 规范化后按比例判定是否强转数值
6. 全角字符、不可见字符 → 统一规范化
7. 缺失值              → 目标列缺失丢行、特征缺失中位数填补，动作全部落盘
8. 合并单元格          → 只在「未映射的非数值列」上尝试向前填充并逐条记录

本模块只负责把数据变成可信的 DataFrame，不做任何建模。
"""

from __future__ import annotations

import re
import csv
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.config import (
    CLEANING_LOG_PATH,
    COLUMN_MAPPING,
    FEATURE_COLS,
    NUMERIC_DETECT_RATIO,
    TARGET_COLS,
)


# ------------------------------------------------------------------
# 规范化工具
# ------------------------------------------------------------------

# 常被误当空白的不可见字符
_INVISIBLE = {
    "\u200b", "\u200c", "\u200d", "\u200e", "\u200f",
    "\ufeff", "\u00a0", "\u2028", "\u2029", "\u2060",
}
# 允许从数值里剥离的单位后缀（仅剥离完全相同的后缀）：
# 只在整列后缀一致时才处理，避免误删有效字符。
_UNIT_SUFFIXES = ("℃", "°C", "°c", "%", "nm", "Å", "mV", "mA", "h", "g/L")


def _normalize_text(value: Any) -> Any:
    """去空白、全角转半角、清不可见字符；非字符串原样返回。"""
    if not isinstance(value, str):
        return value
    text = value
    for char in _INVISIBLE:
        text = text.replace(char, "")
    # 全角空格
    text = text.replace("\u3000", " ")
    # 全角 ASCII（！-～）转为半角
    text = "".join(
        chr(ord(ch) - 0xFEE0) if 0xFF01 <= ord(ch) <= 0xFF5E else ch for ch in text
    )
    # 全角标点在数字里常见：． ，－
    text = text.replace("．", ".").replace("，", ",").replace("－", "-")
    # NFKC 进一步折叠兼容字符（如 ③、①）
    text = unicodedata.normalize("NFKC", text)
    return text.strip()


def _mapping_key(name: Any) -> str:
    """生成用于匹配 COLUMN_MAPPING 的键：规范化 + 去空白 + 小写。"""
    if name is None:
        return ""
    text = str(_normalize_text(str(name)))
    text = re.sub(r"\s+", "", text)
    text = text.replace("（", "(").replace("）", ")")
    return text.lower()


_MAPPING_LOOKUP = {_mapping_key(alias): standard for alias, standard in COLUMN_MAPPING.items()}


def map_column_name(original: Any) -> str | None:
    """把真实列名映射为标准字段名；无法映射时返回 None。"""
    key = _mapping_key(original)
    if key in _MAPPING_LOOKUP:
        return _MAPPING_LOOKUP[key]
    # 去掉括号内的单位后再试一次，例如「比容量(mAh/g)」
    stripped = re.sub(r"\([^)]*\)", "", key)
    if stripped and stripped in _MAPPING_LOOKUP:
        return _MAPPING_LOOKUP[stripped]
    return None


# ------------------------------------------------------------------
# 日志
# ------------------------------------------------------------------

class CleaningLog:
    """收集本次加载的全部处理动作，运行结束后写入 Markdown。"""

    def __init__(self, source: Path | None = None) -> None:
        self.source = source
        self.entries: list[tuple[str, str]] = []

    def add(self, category: str, message: str) -> None:
        self.entries.append((category, message))
        print(f"[{category}] {message}")

    def write(self, path: Path | None = None) -> Path:
        path = path or CLEANING_LOG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# 数据清洗日志",
            "",
            f"- 生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
            f"- 数据文件：`{self.source.name if self.source else '未指定'}`",
            "",
        ]
        if not self.entries:
            lines.extend(["本次加载未触发任何清洗动作。", ""])
        else:
            current = None
            for category, message in self.entries:
                if category != current:
                    lines.extend([f"## {category}", ""])
                    current = category
                lines.append(f"- {message}")
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path


# ------------------------------------------------------------------
# 读取
# ------------------------------------------------------------------

def _sniff_delimiter(path: Path, encoding: str) -> str:
    """猜测分隔符：逗号/分号/制表符/竖线中取出现最多的一种。"""
    try:
        with open(path, "r", encoding=encoding, errors="strict") as handle:
            sample = [handle.readline() for _ in range(15)]
    except Exception:
        return ","
    candidates = [",", ";", "\t", "|"]
    scores = {delimiter: sum(line.count(delimiter) for line in sample) for delimiter in candidates}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else ","


def _read_ragged_csv(path: Path, encoding: str) -> tuple[pd.DataFrame, str]:
    """宽松读取：按最大列数给每行补齐，保留字段数不一致的行（标题行、合并行）。"""
    delimiter = _sniff_delimiter(path, encoding)
    rows: list[list[str]] = []
    with open(path, "r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            rows.append(list(row))
    if not rows:
        raise ValueError(f"文件为空或无法解析：{path}")
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    return pd.DataFrame(padded, dtype=object), delimiter


def _read_csv_once(path: Path, encoding: str) -> tuple[pd.DataFrame, str]:
    """优先用 pandas 快速读取；字段数不一致时退回宽松读取器，避免丢行。"""
    try:
        frame = pd.read_csv(path, encoding=encoding, header=None, dtype=object)
        return frame, ""
    except pd.errors.ParserError:
        frame, delimiter = _read_ragged_csv(path, encoding)
        return frame, delimiter


def _read_csv_with_fallback(path: Path, log: CleaningLog) -> pd.DataFrame:
    """依次尝试常见中文编码读取 CSV，并容忍字段数不一致的行。"""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            frame, fallback_mode = _read_csv_once(path, encoding)
        except UnicodeDecodeError as error:
            last_error = error
            continue
        except pd.errors.EmptyDataError as error:
            raise ValueError(f"文件为空或无法解析：{path}") from error
        log.add("读取", f"CSV 编码识别为 `{encoding}`，原始行数 {len(frame)}")
        if fallback_mode:
            log.add(
                "读取",
                f"字段数不一致，已启用宽松读取器（分隔符 {fallback_mode!r}）并按最大列数补齐，未丢弃任何行",
            )
        return frame
    raise ValueError(f"无法用 utf-8-sig / utf-8 / gbk / gb18030 解码文件：{path}（{last_error}）")


def _read_excel_all_sheets(path: Path, log: CleaningLog) -> pd.DataFrame:
    """读取 xlsx，合并多个工作表；每一行都带上来源表名，便于排查。"""
    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    frames = []
    for name, frame in sheets.items():
        if frame.empty:
            continue
        frame = frame.copy()
        frame.insert(0, "__sheet__", name)
        frames.append(frame)
    if not frames:
        raise ValueError(f"Excel 文件没有可用工作表：{path}")
    if len(frames) > 1:
        log.add("读取", f"检测到 {len(frames)} 个工作表，已纵向合并：{', '.join(s.name for s in sheets.values())}")
    combined = pd.concat(frames, ignore_index=True)
    log.add("读取", f"Excel 读取完成，原始行数 {len(combined)}")
    return combined


def _guess_header_row(raw: pd.DataFrame) -> int:
    """在前若干行里找命中 COLUMN_MAPPING 最多的一行作为表头。"""
    best_row, best_hits = 0, 0
    limit = min(15, len(raw))
    for row_index in range(limit):
        hits = sum(1 for value in raw.iloc[row_index] if map_column_name(value) is not None)
        if hits > best_hits:
            best_row, best_hits = row_index, hits
    return best_row if best_hits >= 2 else 0


def _promote_header(raw: pd.DataFrame, header_row: int, log: CleaningLog) -> pd.DataFrame:
    """把指定行提升为列名，其上的标题行丢弃。"""
    if header_row > 0:
        log.add("表头", f"检测到前 {header_row} 行为标题/说明行，已跳过（表头位于第 {header_row + 1} 行）")
    header_values = raw.iloc[header_row].tolist()
    body = raw.iloc[header_row + 1:].reset_index(drop=True)
    names: list[str] = []
    seen: dict[str, int] = {}
    for index, value in enumerate(header_values):
        # 空单元格、NaN、纯空白一律生成占位列名，避免出现 "nan" 这种假列名
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            candidate = ""
        else:
            text = _normalize_text(value)
            candidate = "" if text is None else str(text).strip()
        if candidate == "" or candidate.lower() in {"nan", "none", "nat"}:
            candidate = f"未命名列{index + 1}"
        if candidate in seen:
            seen[candidate] += 1
            candidate = f"{candidate}_{seen[candidate]}"
        else:
            seen[candidate] = 1
        names.append(candidate)
    body.columns = names
    return body


def read_raw_table(path: str | Path, log: CleaningLog | None = None) -> pd.DataFrame:
    """读取任意格式文件，返回带列名的原始表（未映射、未清洗）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在：{path}")
    log = log or CleaningLog(path)

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        raw = _read_excel_all_sheets(path, log)
        if "__sheet__" in raw.columns:
            raw = raw.drop(columns=["__sheet__"])
    else:
        raw = _read_csv_with_fallback(path, log)

    header_row = _guess_header_row(raw)
    table = _promote_header(raw, header_row, log)
    return table


# ------------------------------------------------------------------
# 规范化与清洗
# ------------------------------------------------------------------

def _drop_empty(table: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """剔除全空行与全空列（纯空格、全角空格、NaN 均视为空）。"""
    def is_blank_value(value: Any) -> bool:
        if isinstance(value, str):
            return _normalize_text(value) == ""
        return pd.isna(value)

    blank_rows = [index for index in table.index if all(is_blank_value(value) for value in table.loc[index])]
    if blank_rows:
        table = table.drop(index=blank_rows)
        log.add("空行", f"剔除全空行 {len(blank_rows)} 行（原索引：{blank_rows[:10]}）")
    blank_cols = [column for column in table.columns if all(is_blank_value(value) for value in table[column])]
    if blank_cols:
        table = table.drop(columns=blank_cols)
        log.add("空列", f"剔除全空列 {len(blank_cols)} 列：{blank_cols}")
    return table.reset_index(drop=True)


def _clean_cells(table: pd.DataFrame) -> pd.DataFrame:
    """对每个单元格做文本规范化。"""
    return table.map(_normalize_text)


def _try_numeric(series: pd.Series) -> tuple[pd.Series | None, int]:
    """尝试把一列转成数值，返回 (结果列, 转换失败的非空行数)。"""
    non_null = series.notna() & (series.astype(str).str.strip() != "")
    if not non_null.any():
        return None, 0
    converted = pd.to_numeric(series, errors="coerce")
    failed = int((non_null & converted.isna()).sum())
    return converted, failed


def _strip_units(series: pd.Series) -> pd.Series:
    """剥掉数值尾部统一出现的单位后缀（仅当整列后缀一致时生效）。"""
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return series
    for suffix in _UNIT_SUFFIXES:
        if non_null.str.endswith(suffix).mean() > 0.9:
            return series.astype(str).str.replace(re.escape(suffix), "", regex=True).str.strip()
    return series


# 逐格清理：只保留「前面是纯数字、后面跟一个已知单位」的写法，
# 这样即使同一列里有的带单位、有的不带，也能逐个处理。
_NUMERIC_WITH_UNIT = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:℃|°C|°c|%|nm|Å|mV|mA|A·h|mAh|Ah|g/L|g|h)\s*$"
)


def _clean_numeric_text(value: Any) -> Any:
    """单值层面清理数字：去千分位、去百分号、剥离单一单位后缀。"""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return value
    text = str(value).strip()
    if text == "":
        return value
    # 千分位与百分号
    stripped = text.replace(",", "").replace("%", "").strip()
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", stripped):
        return stripped
    match = _NUMERIC_WITH_UNIT.match(text)
    if match:
        return match.group(1)
    # 去掉千分位后仍带单位的情况
    match = _NUMERIC_WITH_UNIT.match(stripped)
    if match:
        return match.group(1)
    return value


def _coerce_numeric_columns(
    table: pd.DataFrame,
    protected: Sequence[str],
    log: CleaningLog,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """对疑似数值列做数值化，并记录失败行数。返回 (表, 最终 dtype 字典)。"""
    result = table.copy()
    dtypes: dict[str, str] = {}
    for column in result.columns:
        series = result[column]
        if pd.api.types.is_numeric_dtype(series):
            dtypes[column] = str(series.dtype)
            continue
        non_null = series.notna() & (series.astype(str).str.strip() != "")
        total = int(non_null.sum())
        if total == 0:
            dtypes[column] = str(series.dtype)
            continue

        # 第一次：直接转换
        converted, failed = _try_numeric(series)
        ratio = 1 - failed / total if converted is not None else 0.0

        # 第二次：只要存在失败就逐格清理（去千分位/百分号/单位后缀）后再转换，
        # 取更优结果，避免「大部分能转就放过」导致个别带单位的单元格被静默丢掉。
        if failed > 0:
            cleaned = series.map(_clean_numeric_text)
            cleaned_converted, cleaned_failed = _try_numeric(cleaned)
            cleaned_ratio = 1 - cleaned_failed / total if cleaned_converted is not None else 0.0
            if cleaned_ratio > ratio:
                log.add(
                    "数值化",
                    f"列「{column}」为文本型数字（含单位/全角/百分号等），清理后转换，"
                    f"仍无法转换 {cleaned_failed} 行",
                )
                converted, failed, ratio = cleaned_converted, cleaned_failed, cleaned_ratio

        if converted is not None and ratio >= NUMERIC_DETECT_RATIO:
            result[column] = converted
            dtypes[column] = str(converted.dtype)
            if failed > 0:
                log.add("数值化", f"列「{column}」有 {failed} 行无法转为数值，已置为空值待后续处理")
        else:
            dtypes[column] = str(result[column].dtype)
            if column in protected:
                log.add(
                    "数值化",
                    f"列「{column}」是必需字段但无法转为数值（无法转换 {failed} 行），请在预处理阶段确认",
                )
    return result, dtypes


def _apply_mapping(
    table: pd.DataFrame,
    log: CleaningLog,
    source_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """按 COLUMN_MAPPING 重命名列；未映射列保留原名并告警。"""
    rename: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    original_names = list(source_names) if source_names else list(table.columns)
    for column, original in zip(table.columns, original_names):
        mapped = map_column_name(original)
        if mapped:
            rename[column] = mapped
        records.append({"original": str(original), "mapped": mapped or str(original), "is_mapped": bool(mapped)})
    renamed = table.rename(columns=rename)
    unmapped = [record["original"] for record in records if not record["is_mapped"]]
    if unmapped:
        log.add("映射", f"以下 {len(unmapped)} 列未在 COLUMN_MAPPING 中找到，已保留原始列名：{unmapped}")
    # 若映射后出现重名列，追加后缀避免覆盖
    if renamed.columns.has_duplicates:
        seen: dict[str, int] = {}
        new_columns = []
        for column in renamed.columns:
            if column in seen:
                seen[column] += 1
                new_columns.append(f"{column}_{seen[column]}")
            else:
                seen[column] = 1
                new_columns.append(column)
        renamed.columns = new_columns
        log.add("映射", f"检测到映射后重名列，已自动加后缀：{list(new_columns)}")
    return renamed, records


def _fill_merged_like(
    table: pd.DataFrame,
    mapped_fields: set[str],
    log: CleaningLog,
) -> pd.DataFrame:
    """处理疑似合并单元格：对「非特征、非目标」列向前填充并逐列记录。

    输入特征与目标列绝不向前填充——那会凭空造数据，比留空更危险。
    """
    protected = {*FEATURE_COLS, *TARGET_COLS}
    result = table.copy()
    for column in result.columns:
        if column in protected:
            continue
        series = result[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        missing = int(series.isna().sum() + (series.astype(str).str.strip() == "").sum())
        if missing == 0:
            continue
        # 只在「非空值很少、且像分组标签」时启用，避免误填数据列
        non_null = int(series.notna().sum())
        if non_null == 0 or non_null > missing:
            continue
        filled = series.replace("", np.nan).ffill()
        filled_count = int(filled.notna().sum() - series.notna().sum())
        if filled_count > 0:
            result[column] = filled
            log.add(
                "合并单元格",
                f"列「{column}」疑似合并单元格（非空 {non_null} 个、空 {missing} 个），已向前填充 {filled_count} 处",
            )
    return result


def _print_mapping_table(
    records: Sequence[dict[str, Any]],
    table: pd.DataFrame,
) -> None:
    """打印「原列名 | 映射后字段名 | 数据类型 | 缺失数 | 唯一值数」表格。"""
    headers = ["原列名", "映射后字段名", "数据类型", "缺失数", "唯一值数"]
    rows: list[list[str]] = []
    for record in records:
        column = record["mapped"]
        if column not in table.columns:
            continue
        series = table[column]
        rows.append(
            [
                record["original"],
                f"{column}" + ("" if record["is_mapped"] else " (未映射)"),
                str(series.dtype),
                str(int(series.isna().sum())),
                str(int(series.nunique(dropna=True))),
            ]
        )
    widths = [max(len(str(cell)) for cell in [header] + [row[i] for row in rows]) for i, header in enumerate(headers)]

    def fmt(row: Sequence[str]) -> str:
        # 中文宽度按 2 计，简单左对齐即可读
        cells = []
        for index, cell in enumerate(row):
            text = str(cell)
            pad = widths[index] - len(text) + sum(1 for ch in text if ord(ch) > 127)
            cells.append(text + " " * max(pad, 1))
        return "  ".join(cells)

    print("\n映射结果表：")
    print(fmt(headers))
    print("  ".join("-" * max(width, 6) for width in widths))
    for row in rows:
        print(fmt(row))
    print()


class MissingRequiredColumnsError(ValueError):
    """必需列缺失时抛出的明确错误。"""


def _validate_required_columns(table: pd.DataFrame, mapping_records: Sequence[dict[str, Any]]) -> None:
    """检查 FEATURE_COLS 与 TARGET_COLS 是否齐全；缺失时给出可执行的中文提示。"""
    available = list(table.columns)
    missing_features = [column for column in FEATURE_COLS if column not in available]
    missing_targets = [column for column in TARGET_COLS if column not in available]
    if not missing_features and not missing_targets:
        return

    mapped_pairs = [f"{r['original']}→{r['mapped']}" for r in mapping_records if r["is_mapped"]]
    original_names = [r["original"] for r in mapping_records]
    hints = [
        "请在 config.py 的 COLUMN_MAPPING 中补充「真实列名 → 标准字段名」的映射。",
        "当前可用的原始列名：" + "、".join(original_names),
        "已被映射的标准字段：" + ("、".join(mapped_pairs) if mapped_pairs else "无"),
        "当前 FEATURE_COLS = " + str(list(FEATURE_COLS)) + "；TARGET_COLS = " + str(list(TARGET_COLS)),
    ]
    parts = []
    for column in missing_targets:
        parts.append(f"缺少目标列『{column}』")
    for column in missing_features:
        parts.append(f"缺少输入特征列『{column}』")
    raise MissingRequiredColumnsError("。".join(parts) + "。\n" + "\n".join(hints))


# ------------------------------------------------------------------
# 对外主函数
# ------------------------------------------------------------------

def load_dataset(
    path: str | Path,
    *,
    require_columns: bool = True,
    log_path: Path | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """读取并清洗实验数据，返回映射到标准字段名的 DataFrame。

    参数
    ----
    path             : CSV / XLSX 文件路径
    require_columns  : True 时对 FEATURE_COLS 与 TARGET_COLS 做强制检查
    log_path         : 清洗日志输出路径，默认 reports/data_cleaning_log.md
    strict           : True 时拒绝必需字段缺失或非有限数值，不填补或丢行
    """
    path = Path(path)
    log = CleaningLog(path)

    table = read_raw_table(path, log)
    table = _drop_empty(table, log)
    table = _clean_cells(table)
    source_names = list(table.columns)
    table, mapping_records = _apply_mapping(table, log, source_names=source_names)

    mapped_fields = {record["mapped"] for record in mapping_records if record["is_mapped"]}
    table = _fill_merged_like(table, mapped_fields, log)

    protected = [*FEATURE_COLS, *TARGET_COLS]
    table, dtypes = _coerce_numeric_columns(table, protected, log)

    if strict:
        # 验证前不得使用全体样本的中位数填补，也不能静默丢弃目标缺失行。
        _validate_required_columns(table, mapping_records)
        values = table[protected].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError("严格模式：必需字段存在缺失、非数值或非有限数值，已停止，未填补或丢弃记录。")
        table[protected] = values

    # ---- 缺失处理：目标列丢行，特征列中位数填补 ----
    drop_mask = pd.Series(False, index=table.index)
    for column in TARGET_COLS:
        if column in table.columns:
            column_missing = table[column].isna()
            if column_missing.any():
                drop_mask = drop_mask | column_missing
    if drop_mask.any():
        table = table.loc[~drop_mask].reset_index(drop=True)
        log.add("缺失值", f"因目标值缺失丢弃 {int(drop_mask.sum())} 行")

    for column in FEATURE_COLS:
        if column not in table.columns:
            continue
        missing = int(table[column].isna().sum())
        if missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(table[column]):
            fill_value = float(table[column].median())
            table[column] = table[column].fillna(fill_value)
            log.add("缺失值", f"列 {column} 填补 {missing} 个缺失值（中位数 {fill_value:.6g}）")
        else:
            mode_series = table[column].mode(dropna=True)
            fill_value = mode_series.iloc[0] if not mode_series.empty else ""
            table[column] = table[column].fillna(fill_value)
            log.add("缺失值", f"列 {column} 非数值，改用众数填补 {missing} 个缺失值（取值 {fill_value}）")

    if require_columns:
        _validate_required_columns(table, mapping_records)

    _print_mapping_table(mapping_records, table)
    written = log.write(log_path)
    print(f"清洗日志：{written}")
    print(f"载入完成：{len(table)} 行 × {len(table.columns)} 列\n")
    return table


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法：python -m src.load_data <数据文件路径>")
        raise SystemExit(1)
    frame = load_dataset(sys.argv[1])
    print(frame.head().to_string())
