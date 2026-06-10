"""一次性脚本: 构建 Danbooru wiki tag 的 embedding 索引。

用途: 给 tag_resolver 提供"语义兜底"能力——当 GPT 输出的 tag 在
     by_tag/alias/base 三个词典都不命中时,用 embedding 检索找近义 tag。

输入: HuggingFace 数据集 isek-ai/danbooru-wiki-2024
     (180k 条,字段含 tag/title/other_names/category/is_locked/is_deleted)

输出:
  - data/wiki_emb_index.faiss     # FAISS 索引(~1.1 GB)
  - data/wiki_emb_tags.pkl        # tag 字符串列表(顺序对应 FAISS 索引行号)
  - data/wiki_emb_failed.jsonl    # 失败记录(便于重跑)

用法:
    python build_wiki_emb_index.py             # 全量构建
    python build_wiki_emb_index.py --limit 100 # sanity check
    python build_wiki_emb_index.py --resume    # 断点续跑(从 .partial 缓存)

设计:
- 方案 A: 只 embedding tag 字符串,不带 body
- batch + 重试 + 周期性保存 partial 缓存(crash 不会全白跑)
- 跳过 is_deleted / is_locked 的 tag
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import faiss
from datasets import load_dataset
from openai import OpenAI

from config import (
    OPENAI_API_KEY, EMBED_MODEL,
    BATCH_SIZE, SLEEP_BETWEEN, DATA_DIR,
)


# ── 输出路径 ─────────────────────────────────
WIKI_EMB_INDEX_PATH = DATA_DIR / "wiki_emb_index.faiss"
WIKI_EMB_TAGS_PATH  = DATA_DIR / "wiki_emb_tags.pkl"
WIKI_EMB_FAILED_LOG = DATA_DIR / "wiki_emb_failed.jsonl"
WIKI_EMB_PARTIAL    = DATA_DIR / "wiki_emb_partial.pkl"   # 增量缓存

WIKI_DATASET = "isek-ai/danbooru-wiki-2024"

client = OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────
# Embedding API + 重试
# ─────────────────────────────────────────────

def embed_batch(texts: list[str]) -> list[list[float]]:
    """单次 batch 调用,带 3 次指数退避重试。"""
    for attempt in range(3):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as e:
            print(f"   ⚠️ 第 {attempt+1} 次失败: {e}")
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def append_failed(tags: list[str], error: str):
    """记录失败的 tag 批次。"""
    WIKI_EMB_FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(WIKI_EMB_FAILED_LOG, "a", encoding="utf-8") as f:
        for t in tags:
            entry = {
                "tag":       t,
                "error":     error,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────
# 数据准备
# ─────────────────────────────────────────────

def load_wiki_tags() -> list[str]:
    """从 HF 加载 wiki,过滤 is_deleted/is_locked,去重,返回 tag 列表。"""
    print("📚 加载 Danbooru wiki dataset...")
    ds = load_dataset(WIKI_DATASET, split="train")
    print(f"   原始: {len(ds)} 条")

    tags = []
    seen = set()
    skipped_deleted = 0
    skipped_locked = 0
    skipped_empty = 0

    for row in ds:
        if row.get("is_deleted"):
            skipped_deleted += 1
            continue
        if row.get("is_locked"):
            skipped_locked += 1
            continue
        tag = (row.get("tag") or "").strip()
        if not tag:
            skipped_empty += 1
            continue
        if tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)

    print(f"   过滤后: {len(tags)} 条")
    print(f"   (跳过 deleted={skipped_deleted}, locked={skipped_locked}, "
          f"empty={skipped_empty})")
    return tags


# ─────────────────────────────────────────────
# 增量缓存
# ─────────────────────────────────────────────

def load_partial() -> dict[str, list[float]]:
    """读断点续跑缓存,返回 {tag: embedding} 字典。"""
    if not WIKI_EMB_PARTIAL.exists():
        return {}
    with open(WIKI_EMB_PARTIAL, "rb") as f:
        return pickle.load(f)


def save_partial(cache: dict[str, list[float]]):
    """保存断点缓存(原子写,中途崩不会损坏)。"""
    WIKI_EMB_PARTIAL.parent.mkdir(parents=True, exist_ok=True)
    tmp = WIKI_EMB_PARTIAL.with_suffix(WIKI_EMB_PARTIAL.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(cache, f)
    tmp.replace(WIKI_EMB_PARTIAL)


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def run(limit: int = None, resume: bool = False, save_every: int = 50):
    """构建 wiki embedding 索引。

    Args:
        limit:      只跑前 N 个 tag(sanity check)
        resume:     从 partial 缓存继续
        save_every: 每 N 个 batch 写一次 partial 缓存
    """
    # ── 1. 加载 tag 列表 ──
    tags = load_wiki_tags()
    if limit:
        tags = tags[:limit]
        print(f"🧪 Sanity check: 只跑前 {limit} 个 tag")

    # ── 2. 读 partial 缓存 ──
    cache = {}
    if resume:
        cache = load_partial()
        print(f"💾 加载 partial 缓存: {len(cache)} 条")

    to_embed = [t for t in tags if t not in cache]
    if not to_embed:
        print("✨ 所有 tag 都已在缓存,跳过 embedding 阶段")
    else:
        print(f"🧠 需要 embedding: {len(to_embed)} 条 (batch={BATCH_SIZE})")

        # ── 3. batch embedding ──
        t_start = time.time()
        total_batches = (len(to_embed) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in range(total_batches):
            s = batch_idx * BATCH_SIZE
            batch = to_embed[s : s + BATCH_SIZE]

            try:
                vecs = embed_batch(batch)
                for t, v in zip(batch, vecs):
                    cache[t] = v
            except Exception as e:
                append_failed(batch, str(e))
                print(f"   ❌ batch {batch_idx+1} 失败,已记录到 failed log")
                continue

            # 进度
            done = (batch_idx + 1) * BATCH_SIZE
            done = min(done, len(to_embed))
            elapsed = time.time() - t_start
            eta_sec = elapsed / (batch_idx + 1) * (total_batches - batch_idx - 1)
            print(f"   batch {batch_idx+1}/{total_batches}  "
                  f"({done}/{len(to_embed)})  "
                  f"ETA: {eta_sec/60:.1f} min")

            # 周期写盘
            if (batch_idx + 1) % save_every == 0:
                save_partial(cache)
                print(f"   💾 partial 缓存已保存 ({len(cache)} 条)")

            if batch_idx + 1 < total_batches:
                time.sleep(SLEEP_BETWEEN)

        # 最终保存 partial
        save_partial(cache)
        print(f"💾 partial 缓存最终保存 ({len(cache)} 条)")

    # ── 4. 按 tags 顺序拿向量,建 FAISS ──
    print("\n🔧 构建 FAISS 索引...")
    valid_tags = [t for t in tags if t in cache]
    if len(valid_tags) != len(tags):
        print(f"   ⚠️ {len(tags) - len(valid_tags)} 个 tag 没成功 embedding,被跳过")

    vectors = np.array([cache[t] for t in valid_tags], dtype=np.float32)
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print(f"   索引大小: {index.ntotal} 条向量,维度 {dim}")

    # ── 5. 保存索引 + tag list ──
    print("\n💾 保存到磁盘...")
    WIKI_EMB_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(WIKI_EMB_INDEX_PATH))
    with open(WIKI_EMB_TAGS_PATH, "wb") as f:
        pickle.dump(valid_tags, f)
    print(f"   ✅ FAISS 索引: {WIKI_EMB_INDEX_PATH}")
    print(f"   ✅ Tag 列表:   {WIKI_EMB_TAGS_PATH}")
    print(f"\n🎉 完成,共索引 {index.ntotal} 个 wiki tag")
    print(f"   你可以现在删 partial 缓存: rm {WIKI_EMB_PARTIAL}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 Danbooru wiki tag embedding 索引")
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 个 tag(sanity check)")
    parser.add_argument("--resume", action="store_true",
                        help="从 partial 缓存继续")
    parser.add_argument("--save-every", type=int, default=50,
                        help="每 N 个 batch 写一次缓存(默认 50)")
    args = parser.parse_args()

    run(limit=args.limit, resume=args.resume, save_every=args.save_every)