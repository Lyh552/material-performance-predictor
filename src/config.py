"""项目的统一配置：字段名、随机种子、目录和输出文件均从这里读取。"""

from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLS: list[str] = ["铝相对剂量", "反应温度"]
TARGET_COLS: list[str] = ["比容量", "循环稳定性"]
UNIT_MAP: dict[str, str] = {
    # 输入变量
    "化学组分": "未注明",
    "铝相对剂量": "a.u.",
    "反应温度": "℃",
    # 输出变量（真实数据的量纲）
    "比容量": "mAh/g",
    "循环稳定性": "圈",
}

# 只陈述文件来源，不由程序自动判定实验真实性。
DATA_SOURCE_LABEL = "独立公式生成的合成演示数据，非真实实验"
DATA_STATUS_NOTE = "小样本趋势参考；泛化能力以留一条件验证为准"
SAMPLE_TYPE_COL = "样品类型"
NOTE_COL = "备注"
SAMPLE_TYPE_VALUE = "同一体系"
NOTE_VALUE = "合成数据，仅用于验证流程"
RANDOM_SEED = 42

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

SYNTHETIC_DATA_PATH = DATA_RAW / "synthetic_demo.csv"
# 页面读取最近一次训练所用文件，不再把占位数据与新模型混用。
TRAINING_METADATA_PATH = MODELS_DIR / "training_metadata.json"
DATA_PATH = (
    Path(json.loads(TRAINING_METADATA_PATH.read_text(encoding="utf-8"))["data_path"])
    if TRAINING_METADATA_PATH.exists()
    else PROJECT_ROOT / "data/raw/synthetic_demo.csv"
)
MODEL_PATHS = {
    target: MODELS_DIR / f"model_{target}.joblib" for target in TARGET_COLS
}
PLOT_PATHS = {
    target: FIGURES_DIR / f"pred_vs_true_{target}.png" for target in TARGET_COLS
}
COMBINED_PLOT_PATH = FIGURES_DIR / "pred_vs_true_all_targets.png"
# 旧路径保留，避免历史文件或外部脚本读取时报错；新流程不再使用它们。
GPR_MODEL_PATH = MODELS_DIR / "gpr_model.joblib"
RIDGE_MODEL_PATH = MODELS_DIR / "ridge_model.joblib"
EVALUATION_REPORT_PATH = PROJECT_ROOT / "reports" / "evaluation.md"
PREDICTION_PLOT_PATH = FIGURES_DIR / "pred_vs_true.png"


# ============ 候选变量取舍（新增） ============
# 待评估是否纳入模型的候选列。列名只在这里声明，其他文件不得硬编码。
# 「温度」为老师可能补充的列；数据中不存在时审计会标记为未提供，不影响主流程。
CANDIDATE_COLS: list[str] = ["温度"]

# 方差/稀疏判定阈值（按最大占比 = 众数出现次数 / 非空样本数）
CONSTANT_UNIQUE_COUNT = 1
SPARSE_MAX_SHARE = 0.80      # 最大占比 >= 0.80 → 稀疏变量（极低方差）
LOW_MAX_SHARE = 0.60         # 0.60 <= 最大占比 < 0.80 → 低方差变量

# 差异组数阈值（差异组数 = 取值不等于众数的样品数）
DIFF_N_ALLOW = 8             # >= 8 → 允许进入主模型
DIFF_N_ROBUST = 4            # 4~7 → 仅作稳健性检验；<= 3 → 不纳入

# 报告与中间产物路径
FEATURE_AUDIT_PATH = PROJECT_ROOT / "reports" / "feature_audit.md"
CLEAN_DATA_PATH = DATA_PROCESSED / "clean.csv"

# ============ 真实数据列名映射层（新增） ============
# 作用：把「真实数据文件里的列名」映射到「内部标准字段名」。
# 规则：所有列名判断都走这张表，代码里不出现任何具体中文列名。
# 匹配时会忽略大小写、首尾空格、全角字符与不可见字符。
COLUMN_MAPPING: dict[str, str] = {
    # ---- 输入特征 ----
    "化学组分": "化学组分",
    "组分": "化学组分",
    "成分": "化学组分",
    "组分比例": "化学组分",
    "配比": "化学组分",
    "化学组成": "化学组分",
    "composition": "化学组分",
    "component": "化学组分",
    # ---- 目标：比容量 ----
    "比容量": "比容量",
    "容量": "比容量",
    "放电比容量": "比容量",
    "充电比容量": "比容量",
    "specific capacitance": "比容量",
    "specific capacity": "比容量",
    "capacity": "比容量",
    # ---- 目标：循环稳定性 ----
    "循环稳定性": "循环稳定性",
    "循环稳定性%": "循环稳定性",
    "循环稳定性能": "循环稳定性",
    "循环性能": "循环稳定性",
    "容量保持率": "循环稳定性",
    "循环稳定性（圈数，≥95）": "循环稳定性",
    "循环圈数": "循环稳定性",
    "capacity retention": "循环稳定性",
    "cycling stability": "循环稳定性",
    "retention": "循环稳定性",
    # ---- 双变量输入的列名映射 ----
    "铝相对剂量(a.u.)": "铝相对剂量",
    "铝相对剂量": "铝相对剂量",
    "铝剂量": "铝相对剂量",
    "al_dose": "铝相对剂量",
    "反应温度（℃）": "反应温度",
    "反应温度": "反应温度",
    "反应温度(℃)": "反应温度",
    # ---- 候选变量：温度 ----
    "温度": "温度",
    "沉积温度": "温度",
    "生长温度": "温度",
    "temperature": "温度",
    # ---- 辅助列 ----
    "样品类型": "样品类型",
    "样品编号": "样品编号",
    "批次": "批次",
    "备注": "备注",
    "说明": "备注",
}

# 读取真实数据时的清洗开关
CLEANING_LOG_PATH = PROJECT_ROOT / "reports" / "data_cleaning_log.md"
# 判定「疑似数值列」的门槛：规范化后可解析为数字的非空占比
NUMERIC_DETECT_RATIO = 0.60
