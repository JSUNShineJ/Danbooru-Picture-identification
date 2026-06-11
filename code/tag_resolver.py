"""Tag resolver: GPT 输出的 raw tag → Danbooru wiki 真实候选 tag。

变化点 vs 旧版:
1. ❌ 移除 local_vocab 和 exists_local 加分
   原因: 本地数据量小,作为统计信号无参考价值,会让搜索偏向已爬过的 tag
   现在: 完全以 Danbooru wiki 作为权威词表

2. ✅ 加 wiki category 匹配加分(+1.0)
   原因: wiki 的 category 字段(general/character/copyright)是天然的"类型校验"
   作用: 过滤掉类型错配的候选,如用户在 general 类输"龙娘",
        wiki 里 dragon_girl(category=general) 加分,
        而 dragon_girl_(某角色)(category=character) 不加

3. ✅ Embedding 兜底
   原因: GPT 偶尔输出格式不规范的 tag(如复合词、非标拼写),
        三个词典都查不到,但语义合理
   作用: candidates 为空时,用 OpenAI embedding 查 wiki tag FAISS 索引,
        返回 top-K 近似候选

Score 公式(满分 5.2):
  +2.0  tag == key                       (完全相同)
  +1.2  tag_base(tag) == key 且 tag != key  (去括号后相同,不重复加)
  +1.0  suffix in context_tags           (上下文相关)
  +1.0  wiki category 匹配               (类型对齐)

兜底候选: 三个词典都不命中时由 embedding 引入,score 主要靠 category 匹配
        (大概率拿 0~1.0),自然排在主路径候选后面。
"""

from collections import defaultdict
from pathlib import Path
import pickle
import re

import numpy as np

# faiss / openai 是可选的:wiki_emb 索引不存在时降级,不强依赖
try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


TAG_FIELDS = {
    "general":   "tag_string_general",
    "character": "tag_string_character",
    "copyright": "tag_string_copyright",
}


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def norm_text(text: str) -> str:
    """规范化: 小写 + 空格转下划线 + 去首尾空格。"""
    return re.sub(r"\s+", "_", (text or "").strip().lower())


def tag_base(tag: str) -> str:
    """去掉 _(xxx) 尾巴。如 dragon_girl_(monster) → dragon_girl"""
    return re.sub(r"_\(.+\)$", "", tag)


# ─────────────────────────────────────────────
# 词典构建(Danbooru wiki)
# ─────────────────────────────────────────────

def build_danbooru_lookup(ds):
    """从 wiki dataset 构建三个查找词典:
       - by_tag:        {tag: row_dict}      精确命中
       - alias_to_tags: {alias: [tag, ...]}  别名/title/other_names 命中
       - base_to_tags:  {base: [tag, ...]}   去括号后的 base 命中
    """
    by_tag = {}
    alias_to_tags = defaultdict(list)
    base_to_tags  = defaultdict(list)

    for row in ds:
        tag = norm_text(row.get("tag") or row.get("title") or "")
        if not tag:
            continue

        by_tag[tag] = dict(row) | {"tag": tag}
        base_to_tags[tag_base(tag)].append(tag)

        aliases = {tag, norm_text(row.get("title") or ""), tag_base(tag)}
        for name in row.get("other_names") or []:
            aliases.add(norm_text(name))

        for alias in aliases:
            if alias:
                alias_to_tags[alias].append(tag)

    return by_tag, dict(alias_to_tags), dict(base_to_tags)


# ─────────────────────────────────────────────
# Wiki embedding 兜底索引
# ─────────────────────────────────────────────

def _load_wiki_embedding(index_path: Path, tags_path: Path):
    """加载 wiki embedding 索引和 tag 列表。任一文件不存在则返回 None。"""
    if not (index_path.exists() and tags_path.exists()):
        return None
    if not _HAS_FAISS:
        print("⚠️ faiss 未安装,跳过 wiki embedding 兜底加载")
        return None
    index = faiss.read_index(str(index_path))
    with open(tags_path, "rb") as f:
        tags = pickle.load(f)
    print(f"   ✅ wiki embedding 索引: {index.ntotal} 条")
    return {"index": index, "tags": tags}


# ─────────────────────────────────────────────
# Resolver 构建入口
# ─────────────────────────────────────────────

def build_resolver(records, danbooru_ds,
                   wiki_emb_index_path: Path = None,
                   wiki_emb_tags_path: Path  = None,
                   openai_api_key: str = None,
                   embed_model: str = "text-embedding-3-small"):
    """构建 resolver。

    Args:
        records:  本地图库记录(目前 *不使用*,保留参数兼容旧调用)
        danbooru_ds:           HuggingFace 数据集(必需)
        wiki_emb_index_path:   wiki embedding FAISS 索引路径(可选,兜底用)
        wiki_emb_tags_path:    wiki embedding 对应 tag 列表(可选)
        openai_api_key:        OpenAI key(兜底时调 embedding API 需要)
        embed_model:           embedding model(默认与 indexer 一致)

    Returns:
        resolver 字典,可丢给 resolve_tag / resolve_parsed_tags 使用。
    """
    by_tag, alias_to_tags, base_to_tags = build_danbooru_lookup(danbooru_ds)

    wiki_emb = None
    if wiki_emb_index_path and wiki_emb_tags_path:
        wiki_emb = _load_wiki_embedding(wiki_emb_index_path, wiki_emb_tags_path)

    openai_client = None
    if wiki_emb and openai_api_key and _HAS_OPENAI:
        openai_client = OpenAI(api_key=openai_api_key)

    return {
        "danbooru_by_tag":   by_tag,
        "danbooru_alias":    alias_to_tags,
        "danbooru_base":     base_to_tags,
        "wiki_emb":          wiki_emb,           # None 表示无兜底
        "openai_client":     openai_client,
        "embed_model":       embed_model,
    }


# ─────────────────────────────────────────────
# Embedding 兜底
# ─────────────────────────────────────────────

def _embedding_fallback(key: str, resolver: dict,
                        k: int = 5, threshold: float = 0.5) -> list[str]:
    """三词典都不命中时,用 OpenAI embedding 查 wiki tag FAISS 索引。

    Returns:
        top-K 个 cosine sim >= threshold 的 wiki tag 字符串。
        资源不全(无索引/无 client)时返回空 list。
    """
    wiki_emb = resolver.get("wiki_emb")
    client   = resolver.get("openai_client")
    if not wiki_emb or not client:
        return []

    try:
        resp = client.embeddings.create(
            model=resolver["embed_model"],
            input=[key],
        )
        q_vec = np.array([resp.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(q_vec)
        scores, indices = wiki_emb["index"].search(q_vec, k)
    except Exception as e:
        print(f"⚠️ embedding 兜底失败 (key={key!r}): {e}")
        return []

    out = []
    for idx, sim in zip(indices[0], scores[0]):
        if sim < threshold:
            continue
        if 0 <= idx < len(wiki_emb["tags"]):
            out.append(wiki_emb["tags"][idx])
    return out


# ─────────────────────────────────────────────
# 单 raw tag 解析
# ─────────────────────────────────────────────

def resolve_tag(raw, category, resolver,
                context_tags=None, limit=5,
                fallback_k: int = 5,
                fallback_threshold: float = 0.5):
    """把一个 raw tag 解析为 wiki 候选列表(按 score 降序,top-limit)。

    Args:
        raw:       GPT 输出的 raw tag(可能不规范)
        category:  用户意图分类(general/character/copyright)
        resolver:  build_resolver 的产物
        context_tags: 上下文 tag 集合(主要是 copyright,影响 suffix 加分)
        limit:     返回 top-N(默认 5)
        fallback_k:        embedding 兜底召回数
        fallback_threshold: embedding 兜底距离阈值

    Returns:
        list of candidate dicts:
        {"raw": str, "tag": str, "category": str,
         "wiki_category": str, "from_fallback": bool, "score": float}
    """
    key = norm_text(raw)
    if not key:
        return []

    by_tag       = resolver["danbooru_by_tag"]
    aliases      = resolver["danbooru_alias"]
    base_to_tags = resolver["danbooru_base"]
    context_tags = context_tags or set()

    # ── Stage 1: 三词典查找 ──
    candidates = set()
    if key in by_tag:
        candidates.add(key)
    candidates.update(aliases.get(key, []))
    candidates.update(base_to_tags.get(key, []))
    from_fallback = {t: False for t in candidates}

    # ── Stage 2: 词典空时,embedding 兜底 ──
    if not candidates:
        fb = _embedding_fallback(key, resolver,
                                 k=fallback_k, threshold=fallback_threshold)
        for t in fb:
            candidates.add(t)
            from_fallback[t] = True

    # ── Stage 3: 打分 ──
    ranked = []
    for tag in candidates:
        base = tag_base(tag)
        suffix = tag[len(base) + 2 : -1] if tag.startswith(base + "_(") else ""

        # wiki 里这个 tag 的 category(可能不存在,fallback 引入的也可能查得到)
        wiki_row = by_tag.get(tag, {})
        wiki_category = wiki_row.get("category", "")

        score = 0.0
        score += 2.0 if tag == key else 0.0
        score += 1.2 if base == key and tag != key else 0.0
        score += 1.0 if suffix and suffix in context_tags else 0.0
        score += 1.0 if wiki_category == category else 0.0

        ranked.append({
            "raw":           raw,
            "tag":           tag,
            "category":      category,
            "wiki_category": wiki_category,
            "from_fallback": from_fallback.get(tag, False),
            "score":         score,
        })

    # 排序: score 降序; 同分 tiebreaker(短优先 + 字母序),保证可重现
    return sorted(
        ranked,
        key=lambda x: (-x["score"], len(x["tag"]), x["tag"]),
    )[:limit]


# ─────────────────────────────────────────────
# 批量解析(parsed query → resolved candidates)
# ─────────────────────────────────────────────

def resolve_parsed_tags(parsed, resolver):
    """对整个 parsed query 调用 resolve_tag。

    parsed 形如(parse_query 输出):
    {
      "positive": {"general": [...], "character": [...], "copyright": [...]},
      "negative": {...},
      "semantic": str
    }
    """
    out = {
        "positive": {"general": [], "character": [], "copyright": []},
        "negative": {"general": [], "character": [], "copyright": []},
        "semantic": parsed.get("semantic") or "",
    }

    # 上下文: 用 positive.copyright 解析后的 tag 集合,
    # 给后续 general/character 的 suffix 匹配提供线索
    # (例如 search "原神角色甘雨", copyright=genshin_impact, 后面 character=ganyu_(genshin_impact)
    #  的 suffix "genshin_impact" 就能匹配上下文)
    context = set()
    for tag in parsed.get("positive", {}).get("copyright", []):
        context.update(c["tag"] for c in resolve_tag(tag, "copyright", resolver))

    for polarity in ("positive", "negative"):
        for cat in ("general", "character", "copyright"):
            for raw in parsed.get(polarity, {}).get(cat, []):
                candidates = resolve_tag(raw, cat, resolver, context_tags=context)
                if candidates:
                    out[polarity][cat].append(candidates)

    return out


# ─────────────────────────────────────────────
# 工具:扁平化输出(下游用)
# ─────────────────────────────────────────────

def flatten_top(groups):
    """每组取 top-1 tag。"""
    return [group[0]["tag"] for group in groups if group]


def group_hits(groups, tag_set):
    """每组任一候选命中 tag_set 即计 1。"""
    return sum(1 for group in groups if any(c["tag"] in tag_set for c in group))