"""WD14 批量推理脚本(增量版)。

输入: data/metadata.jsonl                  ← source of truth(爬虫维护)
输出: data/metadata_with_wd14.jsonl        ← 派生文件,带 wd14_raw 字段
失败: data/wd14_failed.jsonl               ← 失败记录,下次会自动重试

用法:
    python run_wd14_batch.py --limit 10    # sanity check,只跑 10 张
    python run_wd14_batch.py               # 跑全量(增量,跳过已有的)
    python run_wd14_batch.py --force-all   # 忽略缓存,全部重跑

设计:
- 增量: 已有 wd14_raw 的图跳过(通过 id 匹配缓存)
- 原子写: 每张图跑完写临时文件再 rename,中途崩不会损坏原文件
- 容错: 单张图失败记录到 failed 文件,不中断整体流程
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from tagger import predict_tags_raw

# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────

ROOT = Path(__file__).parent
META_SRC    = ROOT / "data" / "metadata.jsonl"
META_OUT    = ROOT / "data" / "metadata_with_wd14.jsonl"
FAILED_LOG  = ROOT / "data" / "wd14_failed.jsonl"

# WD14 元数据(写进每条 record 方便溯源)
MODEL_VERSION = "wd-v1-4-moat-tagger-v2"
MIN_CONF      = 0.05


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    """读 jsonl 文件,返回 record 列表。文件不存在返回 []。"""
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def atomic_write_jsonl(records: list[dict], path: Path):
    """原子写:先写临时文件,再 rename 覆盖。中途崩不会损坏原文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)   # rename 是原子操作


def append_failed(record_id: int, image_path: str, error: str):
    """追加一条失败记录。"""
    FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "id":         record_id,
        "image_path": image_path,
        "error":      error,
        "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def run(limit: Optional[int] = None, force_all: bool = False, save_every: int = 10):
    """跑 WD14 批量推理(增量)。
    
    Args:
        limit: 只跑前 N 张(sanity check 用)
        force_all: 忽略缓存,所有图都重跑
        save_every: 每 N 张写一次盘(权衡 IO 和容错)
    """
    
    # ── 1. 读源数据 ──
    if not META_SRC.exists():
        print(f"❌ 源文件不存在: {META_SRC}")
        sys.exit(1)
    
    all_records = load_jsonl(META_SRC)
    print(f"📂 源 metadata: {len(all_records)} 条")
    
    # ── 2. 读缓存(已有 wd14_raw 的) ──
    cached_wd14: dict[int, dict] = {}
    if not force_all and META_OUT.exists():
        existing = load_jsonl(META_OUT)
        for rec in existing:
            if "wd14_raw" in rec:
                cached_wd14[rec["id"]] = rec["wd14_raw"]
        print(f"💾 已有 WD14 缓存: {len(cached_wd14)} 张")
    
    # ── 3. 决定要跑的 records ──
    if limit:
        # sanity check: 优先选还没跑过的前 N 张
        candidates = [r for r in all_records if r["id"] not in cached_wd14]
        to_process = candidates[:limit]
        print(f"🧪 Sanity check 模式: 只跑 {len(to_process)} 张")
    else:
        to_process = all_records   # 走全量,内部判断跳过
    
    # ── 4. 准备输出 records(保持源 metadata 的顺序) ──
    # 先把所有 records 装好,有缓存的填上,没缓存的稍后跑
    output_records = []
    for rec in all_records:
        rec_copy = dict(rec)   # 浅拷贝,不动源数据
        if rec["id"] in cached_wd14:
            rec_copy["wd14_raw"] = cached_wd14[rec["id"]]
        output_records.append(rec_copy)
    
    # 用 id → index 映射方便后续更新
    id_to_idx = {rec["id"]: i for i, rec in enumerate(output_records)}
    
    # ── 5. 跑推理 ──
    need_run = [r for r in to_process if r["id"] not in cached_wd14 or force_all]
    
    if limit:
        need_run = need_run[:limit]
    
    if not need_run:
        print("✨ 所有图都已经有 WD14 结果,无需运行")
        return
    
    print(f"🚀 本次需要跑: {len(need_run)} 张")
    print(f"💾 每 {save_every} 张写盘一次\n")
    
    success = 0
    failed  = 0
    t_start = time.time()
    
    for i, rec in enumerate(need_run, 1):
        rec_id = rec["id"]
        img_path = rec.get("local_image_path")
        
        # 文件存在性检查
        if not img_path or not Path(img_path).exists():
            failed += 1
            append_failed(rec_id, img_path or "", "image file not found")
            print(f"  [{i}/{len(need_run)}] id={rec_id}  ⚠️  文件不存在")
            continue
        
        # 跑模型
        try:
            raw = predict_tags_raw(img_path, min_conf=MIN_CONF)
            raw["model_version"] = MODEL_VERSION
            raw["min_conf"]      = MIN_CONF
            
            # 写到 output_records 对应位置
            idx = id_to_idx[rec_id]
            output_records[idx]["wd14_raw"] = raw
            
            success += 1
            
            # 进度打印
            elapsed = time.time() - t_start
            avg_per_img = elapsed / i
            eta_sec = avg_per_img * (len(need_run) - i)
            eta_min = eta_sec / 60
            print(f"  [{i}/{len(need_run)}] id={rec_id}  ✅  "
                  f"({len(raw['general'])} general, "
                  f"{len(raw['character'])} character)  "
                  f"ETA: {eta_min:.1f} min")
            
        except Exception as e:
            failed += 1
            append_failed(rec_id, img_path, str(e))
            print(f"  [{i}/{len(need_run)}] id={rec_id}  ❌  {e}")
            continue
        
        # 周期性写盘
        if i % save_every == 0:
            atomic_write_jsonl(output_records, META_OUT)
    
    # ── 6. 最终写盘 ──
    atomic_write_jsonl(output_records, META_OUT)
    
    # ── 7. 总结 ──
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"✨ 完成!")
    print(f"   成功: {success}")
    print(f"   失败: {failed}")
    print(f"   耗时: {total_time/60:.1f} 分钟  "
          f"(平均 {total_time/max(success,1):.1f} 秒/张)")
    print(f"   输出: {META_OUT}")
    if failed > 0:
        print(f"   失败日志: {FAILED_LOG}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WD14 批量推理(增量)")
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 张(sanity check)")
    parser.add_argument("--force-all", action="store_true",
                        help="忽略缓存,所有图重跑")
    parser.add_argument("--save-every", type=int, default=10,
                        help="每 N 张写盘一次(默认 10)")
    args = parser.parse_args()
    
    run(limit=args.limit, force_all=args.force_all, save_every=args.save_every)