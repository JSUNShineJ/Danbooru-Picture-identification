"""WD14 vs Danbooru 人工标注的对照评估。
产出:不同阈值下的召回率/精确率/F1 曲线 + 关键 tag 的检出分析。
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent)) 
from tagger import predict_tags
from config import META_PATH


def load_wd14_results(jsonl_path: str = None, num_samples: int = None, seed: int = 42):
    """直接读已有的 metadata_with_wd14.jsonl,不需要重新跑推理。
    
    每行格式:
    {
        "id": ...,
        "tag_string_general": "tag1 tag2 ...",     # danbooru 人工标注
        "wd14_raw": {
            "general": [[tag, conf], ...],          # WD14 输出(带置信度)
            "character": [...],
            "rating": {...},
        }
    }
    """
    if jsonl_path is None:
        # 默认路径,改成你的实际路径
        from config import DATA_DIR
        jsonl_path = DATA_DIR / "metadata_with_wd14.jsonl"
    
    print(f"📂 读取 {jsonl_path} ...")
    
    results = []
    skipped = 0
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            record = json.loads(line)
            
            # 必须同时有 danbooru 标注和 WD14 输出
            danbooru_str = record.get("tag_string_general") or ""
            wd14_raw = record.get("wd14_raw")
            
            if not danbooru_str or not wd14_raw:
                skipped += 1
                continue
            
            results.append({
                "id":         record["id"],
                "danbooru":   set(danbooru_str.split()),
                "wd14_raw":   wd14_raw["general"],         # [[tag, conf], ...]
                "image_path": record.get("local_image_path"),
            })
    
    print(f"   ✅ 加载 {len(results)} 条(跳过 {skipped} 条缺数据的)")
    
    # 抽样(如果指定)
    if num_samples and num_samples < len(results):
        random.seed(seed)
        results = random.sample(results, num_samples)
        print(f"   📊 随机抽样 {len(results)} 条")
    
    return results

def compute_metrics(results, thresholds):
    """对每个阈值算 P/R/F1。"""
    rows = []
    for thresh in thresholds:
        recalls, precisions = [], []
        for r in results:
            wd14_set = {t for t, c in r["wd14_raw"] if c >= thresh}
            common = r["danbooru"] & wd14_set
            
            if r["danbooru"]:
                recalls.append(len(common) / len(r["danbooru"]))
            if wd14_set:
                precisions.append(len(common) / len(wd14_set))
        
        avg_r = np.mean(recalls)    if recalls    else 0
        avg_p = np.mean(precisions) if precisions else 0
        f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) else 0
        
        rows.append({
            "threshold": thresh,
            "recall": avg_r,
            "precision": avg_p,
            "f1": f1,
        })
    
    return pd.DataFrame(rows)


def plot_pr_curve(df, save_path="wd14_pr_curve.png"):
    """画 P/R/F1 三条曲线。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df["threshold"], df["recall"]    * 100, "-o", label="Recall",    linewidth=2)
    ax.plot(df["threshold"], df["precision"] * 100, "-s", label="Precision", linewidth=2)
    ax.plot(df["threshold"], df["f1"]        * 100, "-^", label="F1",        linewidth=2)
    
    # 标 F1 最高点
    best_idx = df["f1"].idxmax()
    best = df.iloc[best_idx]
    ax.axvline(best["threshold"], color="gray", linestyle="--", alpha=0.5)
    ax.annotate(
        f"Best F1: {best['f1']*100:.1f}%\n@ threshold={best['threshold']:.2f}",
        xy=(best["threshold"], best["f1"]*100),
        xytext=(best["threshold"]+0.05, best["f1"]*100-10),
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )
    
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("WD14 vs Danbooru: Precision/Recall/F1 vs Threshold", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"💾 图表已保存: {save_path}")


def analyze_key_tags(results, threshold: float, top_n: int = 20):
    """分析每个 tag 的召回率,找出 WD14 最容易漏的 tag。"""
    tag_stats = defaultdict(lambda: {"in_danbooru": 0, "in_both": 0})
    
    for r in results:
        wd14_set = {t for t, c in r["wd14_raw"] if c >= threshold}
        for tag in r["danbooru"]:
            tag_stats[tag]["in_danbooru"] += 1
            if tag in wd14_set:
                tag_stats[tag]["in_both"] += 1
    
    # 转 DataFrame,只看出现过 >= 5 次的 tag(避免低频噪声)
    rows = []
    for tag, stats in tag_stats.items():
        if stats["in_danbooru"] >= 20:
            recall = stats["in_both"] / stats["in_danbooru"]
            rows.append({
                "tag": tag,
                "count": stats["in_danbooru"],
                "recall": recall,
            })
    
    df = pd.DataFrame(rows)
    
    print(f"\n🔻 WD14 召回率最低的 {top_n} 个常见 tag(@ threshold={threshold}):")
    print(df.nsmallest(top_n, "recall").to_string(index=False))
    
    print(f"\n🔺 WD14 召回率最高的 {top_n} 个常见 tag:")
    print(df.nlargest(top_n, "recall").to_string(index=False))
    
    return df

def tag_recall(results, target_tag: str, threshold: float = 0.30):
    """简单粗暴:某个 tag 的召回率 + 置信度分布。"""
    import numpy as np
    
    total = 0
    hit = 0
    confs = []
    
    for r in results:
        if target_tag not in r["danbooru"]:
            continue
        
        total += 1
        
        # 找 WD14 给这个 tag 的置信度
        wd14_dict = {t: c for t, c in r["wd14_raw"]}
        conf = wd14_dict.get(target_tag, 0.0)
        confs.append(conf)
        
        if conf >= threshold:
            hit += 1
    
    if total == 0:
        print(f"⚠️ '{target_tag}' 在数据集里没出现")
        return
    
    recall = hit / total
    confs = np.array(confs)
    
    print(f"🔍 '{target_tag}':")
    print(f"   样本数:     {total}")
    print(f"   召回 (≥{threshold}): {hit} ({recall*100:.1f}%)")
    print(f"   置信度均值: {confs.mean():.3f}")
    print(f"   置信度中位: {np.median(confs):.3f}")
    print(f"   有输出 (>0): {(confs > 0).sum()} ({(confs > 0).sum()/total*100:.1f}%)")
    print()

if __name__ == "__main__":
    
    results = load_wd14_results()  # 用全部数据
    
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    metrics_df = compute_metrics(results, thresholds)
    print("\n📊 阈值-精度对照表:")
    print(metrics_df.to_string(index=False))
    
    plot_pr_curve(metrics_df)
    
    best_threshold = metrics_df.loc[metrics_df["f1"].idxmax(), "threshold"]
    tag_df = analyze_key_tags(results, threshold=best_threshold)
    
    
    for tag in ["claw", "claws", "dragon_claw", "tail", "dragon_tail", "horns", "dragon_horns"]:
        tag_recall(results, tag, threshold=0.30)
