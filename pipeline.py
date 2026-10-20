"""阶段4：审核快照的候选训练入口，不直接覆盖当前正式版本。"""

import argparse
from pathlib import Path

from src.jobs import create_job, run_job
from src.version_store import Store, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path, help='须与初始数据加已审核快照一致的CSV/XLSX；不绕过审核')
    parser.add_argument('--store', type=Path, help='独立版本库；测试时指定临时目录')
    args = parser.parse_args()
    store = Store(args.store)
    job = create_job(store, source=args.data.resolve(), spawn=False, reject_existing=True)
    with store.connection() as con:
        status = con.execute('SELECT status FROM jobs WHERE id=?', (job,)).fetchone()[0]
    if status != 'queued':
        print(f'已有任务 {job} 正在运行，本次不重复训练。')
        return
    run_job(store, job)
    result = read_json(store.version_dir(job) / 'cross_validation.json')
    for row in result['summary']:
        scheme = {'row': '留一行（偏乐观，不对外宣传）', 'dose': '留一剂量', 'condition': '留一条件（主指标）'}[row['scheme']]
        text = f"{row['target']} {row['model']} {scheme} MAE={row['mae']:.6f} RMSE={row['rmse']:.6f} R²={row['r2']}"
        if row['model'] == 'gpr':
            text += f" 名义90%覆盖率={row['coverage']:.6f} 平均宽度={row['mean_width']:.6f}"
        print(text)
    print(f'候选版本 {job} 已保存；当前正式版本仍为 {store.active_id()}，启用须在模型版本页人工确认。')


if __name__ == '__main__':
    main()
