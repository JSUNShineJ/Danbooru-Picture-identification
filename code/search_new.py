"""根据自然语言查询,从 FAISS 索引里搜出最相似的图片。

两个入口:
  - simple_search(): GPT 把 query 改写成 tag 串,直接 embedding 检索(baseline)
  - smart_search():  GPT 结构化解析 → 向量召回 → 合并 manual+WD14 tag → αβγ 加权重排

WD14 接入(方案 B):
  - FAISS 索引仍只用人工 tag(indexer 产物),不重建
  - 启动时额外加载 metadata_with_wd14.jsonl,按 id 合并 WD14 raw 到 records
  - 评分时基于 "manual ∪ WD14(>=阈值)" 的合并 tag 字典做加权
"""

import json
import pickle
from pathlib import Path

import numpy as np
import faiss
from datasets import load_dataset
from openai import OpenAI
from collections import defaultdict

from config import (
    OPENAI_API_KEY, EMBED_MODEL, CHAT_MODEL,
    INDEX_PATH, META_PKL, WD14_META_PATH,
    WIKI_EMB_INDEX_PATH, WIKI_EMB_TAGS_PATH,
    SEARCH_RECALL_K, WD14_QUERY_THRESHOLD,
    FIELD_WEIGHTS, ALPHA_VECTOR, BETA_POSITIVE, GAMMA_NEG,
)
from tag_resolver import (
    build_resolver,
    flatten_top,
    group_hits,
    resolve_parsed_tags,
)


client = OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────
# 启动加载:FAISS 索引 + records + WD14 数据
# ─────────────────────────────────────────────

print("📦 加载 FAISS 索引...")
_index = faiss.read_index(str(INDEX_PATH))

with open(META_PKL, "rb") as f:
    _records = pickle.load(f)
print(f"   ✅ {_index.ntotal} 条向量,{len(_records)} 条元数据")


def _load_wd14_data() -> dict[int, dict]:
    """读 metadata_with_wd14.jsonl,返回 {post_id: wd14_raw} 字典。

    文件不存在时返回空 dict(degrade 到纯人工 tag 模式,搜索仍可用,
    只是失去了 WD14 的补召回能力)。
    """
    if not WD14_META_PATH.exists():
        print(f"⚠️ 未找到 WD14 数据 {WD14_META_PATH},将仅使用人工 tag")
        return {}
    wd14_by_id = {}
    with open(WD14_META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "wd14_raw" in rec:
                wd14_by_id[rec["id"]] = rec["wd14_raw"]
    return wd14_by_id


print("🔖 加载 WD14 推理结果...")
_wd14_by_id = _load_wd14_data()
print(f"   ✅ {len(_wd14_by_id)} 张图带 WD14 tag")

print("📚 加载 Danbooru wiki tag 表...")
_danbooru_ds = load_dataset("isek-ai/danbooru-wiki-2024", split="train")
_tag_resolver = build_resolver(
    _records, _danbooru_ds,
    wiki_emb_index_path=WIKI_EMB_INDEX_PATH,
    wiki_emb_tags_path=WIKI_EMB_TAGS_PATH,
    openai_api_key=OPENAI_API_KEY,
    embed_model=EMBED_MODEL,
)
print("   ✅ Danbooru tag resolver ready")


# ─────────────────────────────────────────────
# 工具:tag 字符串 → set,manual+WD14 合并视图
# ─────────────────────────────────────────────

def _split_tags(s: str) -> set[str]:
    """tag_string_xxx 字段是空格分隔字符串,转 set。"""
    return set((s or "").split())


def build_merged_tags(record: dict) -> dict[str, dict[str, float]]:
    """构造一张图的 "合并 tag 视图"(策略 1: max 合并)。

    返回:
    {
        "general":   {tag: conf, ...},
        "character": {tag: conf, ...},
        "copyright": {tag: conf, ...},
    }

    合并规则:
      - 人工 tag → conf = 1.0(视为 ground truth)
      - WD14 tag(conf >= WD14_QUERY_THRESHOLD)→ 用 WD14 真实 conf
      - 同一 tag 两边都有 → 取 max(必然 1.0)
      - WD14 无 copyright 字段,该字段只来自人工

    入参 record 是 _records 里的一条(只含人工 tag),WD14 数据从
    模块级 _wd14_by_id[record["id"]] 取。
    """
    merged = {
        "general":   {},
        "character": {},
        "copyright": {},
    }

    # ── 人工 tag(conf = 1.0) ──
    for t in _split_tags(record.get("tag_string_general")):
        merged["general"][t] = 1.0
    for t in _split_tags(record.get("tag_string_character")):
        merged["character"][t] = 1.0
    for t in _split_tags(record.get("tag_string_copyright")):
        merged["copyright"][t] = 1.0

    # ── WD14 tag(conf >= 阈值,与人工取 max) ──
    wd14 = _wd14_by_id.get(record["id"])
    if wd14:
        for tag, conf in wd14.get("general", []):
            if conf >= WD14_QUERY_THRESHOLD:
                prev = merged["general"].get(tag, 0.0)
                merged["general"][tag] = max(prev, conf)
        for tag, conf in wd14.get("character", []):
            if conf >= WD14_QUERY_THRESHOLD:
                prev = merged["character"].get(tag, 0.0)
                merged["character"][tag] = max(prev, conf)
        # WD14 无 copyright,跳过

    return merged


# ─────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────

def embed_query(query: str) -> np.ndarray:
    """把自然语言查询转成向量(归一化后,用于 IP 内积 = 余弦)。"""
    resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
    vec = np.array([resp.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(vec)
    return vec


# ─────────────────────────────────────────────
# Baseline: simple_search(GPT 改写 query → embedding 直查)
# ─────────────────────────────────────────────

REWRITE_PROMPT = """You are a danbooru tag expert. Convert the user's natural language query into danbooru tags.

Rules:
- Output ONLY the tags, separated by spaces, no other text
- Use underscores for multi-word tags (e.g., "from_behind" not "from behind")
- Use lowercase
- Common tag patterns:
  - person count: 1girl, 1boy, 2girls, ...
  - viewpoint: from_behind, from_above, from_side, looking_back
  - hair: pink_hair, long_hair, short_hair, twintails
  - fantasy: dragon_girl, monster_girl, kemonomimi
  - composition: solo, multiple_girls

Examples:
User: a girl with pink hair seen from behind
Output: 1girl pink_hair from_behind

User: 龙娘从背后看
Output: 1girl dragon_girl from_behind

User: two girls smiling
Output: 2girls smile

Now convert this query:
"""


def rewrite_query(natural_query: str) -> str:
    """把自然语言转成 tag 串(simple_search 专用)。"""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user",   "content": natural_query},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def simple_search(query: str, top_k: int = 5, use_rewrite: bool = True) -> list[dict]:
    """Baseline:GPT 改写 → embedding → FAISS 直查,无后排序。"""
    if use_rewrite:
        rewritten = rewrite_query(query)
        print(f"  🔄 改写: '{query}' → '{rewritten}'")
        embed_input = rewritten
    else:
        embed_input = query

    q_vec = embed_query(embed_input)
    scores, indices = _index.search(q_vec, top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        meta = _records[idx].copy()
        meta["_rank"]  = rank
        meta["_score"] = float(score)
        results.append(meta)
    return results


# 兼容别名(保留旧调用代码不报错)
search = simple_search


# ─────────────────────────────────────────────
# Smart Search:结构化解析(GPT) → 召回 → 合并tag → αβγ 重排
# ─────────────────────────────────────────────

PARSE_PROMPT = """You are a danbooru tag expert. Parse the user's query into a structured filter.

# Output format
You MUST output valid JSON:
{
  "positive": {
    "general":   [list of general/concept tags],
    "character": [list of character tags],
    "copyright": [list of copyright/series tags]
  },
  "negative": {
    "general":   [...],
    "character": [...],
    "copyright": [...]
  },
  "semantic": "free-text description"
}

# Tag categorization
- character: specific character names (hatsune_miku, kagamine_rin, ...)
- copyright: series/franchise names (vocaloid, blue_archive, genshin_impact, ...)
- general: everything else (1girl, from_behind, pink_hair, dragon_girl, smile, ...)

# Rules
1. All tags lowercase, underscores for multi-word
2. NEVER invent fake tags (no_xxx, etc.)
3. Interpret intent:
   - "要 X" / "want X" → positive
   - "不要 X" / "no X" → negative
   - ONLY assign to negative if the user explicitly uses negation words
    ("not", "no", "without", "exclude", "don't want", "不要", "排除", "没有", "去掉").
    If no negation word is present, it MUST go to positive.
4. If a category has no tags, use empty list []


# Examples
User: 异色皮肤
Output: {"positive": {"general": ["colored_skin", "multicolored_skin"], "character": [], "copyright": []}, "negative": {...empty...}, "semantic": "person with colored skin"}

User: 单只眼睛
Output: {"positive": {"general": ["one_eye_covered", "monoculus"], "character": [], "copyright": []}, "negative": {...empty...}, "semantic": "single eye"}

User: 粉色头发的龙娘,从背后看
Output: {"positive": {"general": ["1girl", "dragon_girl", "pink_hair", "from_behind"], "character": [], "copyright": []}, "negative": {"general": [], "character": [], "copyright": []}, "semantic": "dragon girl with pink hair from behind"}

User: 初音未来穿校服
Output: {"positive": {"general": ["1girl", "school_uniform"], "character": ["hatsune_miku"], "copyright": ["vocaloid"]}, "negative": {"general": [], "character": [], "copyright": []}, "semantic": "Hatsune Miku wearing school uniform"}

User: 龙娘,不要异色皮肤
Output: {"positive": {"general": ["1girl", "dragon_girl"], "character": [], "copyright": []}, "negative": {"general": ["colored_skin", "multicolored_skin"], "character": [], "copyright": []}, "semantic": "dragon girl with normal skin"}

User: blue archive characters, not miku
Output: {"positive": {"general": [], "character": [], "copyright": ["blue_archive"]}, "negative": {"general": [], "character": ["hatsune_miku"], "copyright": []}, "semantic": "Blue Archive characters"}

Now parse:
"""


def parse_query(natural_query: str) -> dict:
    """GPT 结构化解析自然语言查询。"""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": PARSE_PROMPT},
            {"role": "user",   "content": natural_query},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError as e:
        print(f"⚠️ GPT 输出 JSON 解析失败: {e}")
        parsed = {}

    def get_category(d, key):
        sub = d.get(key) or {}
        return {
            "general":   sub.get("general")   or [],
            "character": sub.get("character") or [],
            "copyright": sub.get("copyright") or [],
        }

    return {
        "positive": get_category(parsed, "positive"),
        "negative": get_category(parsed, "negative"),
        "semantic": parsed.get("semantic") or natural_query,
    }


# ─────────────────────────────────────────────
# GPT 二选:从 resolver 给出的候选里挑(类 RAG)
# ─────────────────────────────────────────────

SELECT_PROMPT = """You are a danbooru tag expert. Pick the most semantically appropriate tag(s)
from the candidate lists provided for each user-intended concept.

# Input
You'll receive:
- Original user query
- For each polarity (positive/negative) × category (general/character/copyright),
  a list of "concept groups". Each group represents one raw concept the user mentioned,
  with up to 5 candidate tags retrieved from a verified Danbooru tag vocabulary.

# Your task
For EACH group, select tag(s) that best match the user's intent. Rules:
- Multi-select allowed: if multiple candidates all fit, keep all of them.
- Empty allowed: if NONE of the candidates fit semantically, return an empty list.
- DO NOT invent tags outside the candidates (the candidates are the only valid vocabulary).
- Preserve the group order; output the same number of groups as input.

# Output
You MUST output valid JSON, mirroring the input structure but with each group replaced
by a list of selected tag strings:
{
  "positive": {
    "general":   [["selected_tag", ...], ...],
    "character": [...],
    "copyright": [...]
  },
  "negative": { ... same shape ... }
}

# Example
Input:
Query: "粉色头发的龙娘从背后"
positive.general:
  - "龙娘" → [dragon_girl, dragon_horns, dragon, kemonomimi, monster_girl]
  - "粉色头发" → [pink_hair, light_pink_hair, hair_color, multicolored_hair, two-tone_hair]
  - "从背后" → [from_behind, looking_back, back, back_focus, facing_away]
positive.character: (empty)
positive.copyright: (empty)
negative.*: (empty)

Output:
{"positive": {"general": [["dragon_girl"], ["pink_hair"], ["from_behind", "looking_back"]], "character": [], "copyright": []}, "negative": {"general": [], "character": [], "copyright": []}}

Note: "from_behind" and "looking_back" are both selected because the user's intent
("从背后") is compatible with either or both. "dragon_horns" wasn't selected because
the user said 龙娘 (dragon girl), not specifically about horns.

Now select:
"""


def _format_resolved_for_selection(resolved: dict) -> str:
    """把 resolver 输出格式化成给 GPT 看的文本。"""
    lines = []
    for polarity in ("positive", "negative"):
        for cat in ("general", "character", "copyright"):
            groups = resolved[polarity][cat]
            if not groups:
                lines.append(f"{polarity}.{cat}: (empty)")
                continue
            lines.append(f"{polarity}.{cat}:")
            for group in groups:
                if not group:
                    continue
                # group 来自 resolver,每项是 dict 含 "raw" 和 "tag"
                raw = group[0].get("raw", "?")
                cands = [c["tag"] for c in group]
                lines.append(f"  - \"{raw}\" → [{', '.join(cands)}]")
    return "\n".join(lines)


def select_from_candidates(query: str, resolved: dict) -> dict:
    """让 GPT 从 resolver 候选里挑出最贴切的 tag(可多选/可弃选)。

    输入: resolver 输出(positive/negative × category × groups of cand dicts)
    输出: 同结构,但 groups 里每项是 list[str](GPT 选定的 tag)
    """
    user_msg = f"Query: \"{query}\"\n\n{_format_resolved_for_selection(resolved)}"

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SELECT_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        selected = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError as e:
        print(f"⚠️ GPT 二选输出 JSON 解析失败: {e},回退到 resolver flatten_top")
        # fallback: 按 flatten_top 行为,每组取第 1 名
        def fallback(groups):
            return [[g[0]["tag"]] for g in groups if g]
        return {
            "positive": {
                "general":   fallback(resolved["positive"]["general"]),
                "character": fallback(resolved["positive"]["character"]),
                "copyright": fallback(resolved["positive"]["copyright"]),
            },
            "negative": {
                "general":   fallback(resolved["negative"]["general"]),
                "character": fallback(resolved["negative"]["character"]),
                "copyright": fallback(resolved["negative"]["copyright"]),
            },
        }

    # 检查 GPT 是否选了候选外的 tag(理论上不该发生,但容错)
    def validate(groups_selected, groups_original):
        out = []
        for sel, orig in zip(groups_selected, groups_original):
            allowed = {c["tag"] for c in orig}
            valid = [t for t in sel if t in allowed]
            if len(valid) != len(sel):
                extra = set(sel) - allowed
                print(f"⚠️ GPT 二选包含候选外 tag,已丢弃: {extra}")
            out.append(valid)
        # 如果 GPT 漏返了某组,补空
        while len(out) < len(groups_original):
            out.append([])
        return out

    def get_cat(d, polarity, cat):
        sub = (d.get(polarity) or {})
        return sub.get(cat) or []

    return {
        "positive": {
            "general":   validate(get_cat(selected, "positive", "general"),
                                  resolved["positive"]["general"]),
            "character": validate(get_cat(selected, "positive", "character"),
                                  resolved["positive"]["character"]),
            "copyright": validate(get_cat(selected, "positive", "copyright"),
                                  resolved["positive"]["copyright"]),
        },
        "negative": {
            "general":   validate(get_cat(selected, "negative", "general"),
                                  resolved["negative"]["general"]),
            "character": validate(get_cat(selected, "negative", "character"),
                                  resolved["negative"]["character"]),
            "copyright": validate(get_cat(selected, "negative", "copyright"),
                                  resolved["negative"]["copyright"]),
        },
    }


# ─────────────────────────────────────────────
# 评分:公式 4-b,基于 merged_tags 用 WD14 conf 加权
# ─────────────────────────────────────────────

def _group_field_score(groups: list, merged_field: dict[str, float],
                       field_weight: float) -> float:
    """单字段的命中加权和(按 resolver group 评分,保留多候选容错)。

    每个 query raw tag → 一个 group。group 元素既可能是 resolver 的 candidate dict
    (含 "tag" key),也可能是 GPT 二选后的纯 tag str。
    评分规则:
      - 同一 group 内任一候选命中 merged_field → 用命中候选的最高 conf 计分
      - 一个 group 内都不命中 → 0 贡献

    举例: raw="红发" → group=[red_hair, redhead, red_long_hair]
         图人工标了 redhead → 该 group 贡献 1.0 * field_weight
         图人工标了 red_hair → 该 group 贡献 1.0 * field_weight
         两者都没有但 WD14 有 redhead@0.6 → 贡献 0.6 * field_weight
    """
    score = 0.0
    for group in groups:
        if not group:
            continue
        best_conf = 0.0
        for cand in group:
            tag = cand["tag"] if isinstance(cand, dict) else cand
            conf = merged_field.get(tag)
            if conf is not None and conf > best_conf:
                best_conf = conf
        score += best_conf * field_weight
    return score


def _score_candidate(record: dict, positive: dict, negative: dict,
                     vec_sim: float) -> dict:
    """对一个候选算 αβγ 综合得分,返回打分明细。

    positive/negative 是 resolver 的输出形态:
      {"general":   [[cand_dict, ...], ...],   # 每个 raw → 一组候选
       "character": [...],
       "copyright": [...]}

    record 是 _records 里的一条(纯人工 tag);合并 tag 通过 build_merged_tags 拿。
    """
    merged = build_merged_tags(record)

    pos_score = (
        _group_field_score(positive["general"],   merged["general"],   FIELD_WEIGHTS["general"]) +
        _group_field_score(positive["character"], merged["character"], FIELD_WEIGHTS["character"]) +
        _group_field_score(positive["copyright"], merged["copyright"], FIELD_WEIGHTS["copyright"])
    )
    neg_score = (
        _group_field_score(negative["general"],   merged["general"],   FIELD_WEIGHTS["general"]) +
        _group_field_score(negative["character"], merged["character"], FIELD_WEIGHTS["character"]) +
        _group_field_score(negative["copyright"], merged["copyright"], FIELD_WEIGHTS["copyright"])
    )

    final = (
        ALPHA_VECTOR  * vec_sim
        + BETA_POSITIVE * pos_score
        - GAMMA_NEG     * neg_score
    )
    return {
        "_vec_sim":      float(vec_sim),
        "_pos_score":    float(pos_score),
        "_neg_score":    float(neg_score),
        "_final_score":  float(final),
    }


# ─────────────────────────────────────────────
# Smart Search 主入口
# ─────────────────────────────────────────────

def smart_search(
    query: str,
    top_k: int = 5,
    recall_k: int = SEARCH_RECALL_K,
    verbose: bool = True,
    use_gpt_select: bool = True,
) -> list[dict]:
    """结构化解析 + 召回 + 合并 tag 重排。

    Args:
        use_gpt_select: True 时启用 GPT 二选(从 resolver 候选里挑);
                        False 时直接用 resolver flatten_top(每组取第 1 名)。
    """
    # ── 1. GPT-1: 自然语言 → 结构化 raw tag ──
    parsed = parse_query(query)

    # ── 2. Resolver: 每个 raw tag → top-5 候选(从 Danbooru wiki) ──
    resolved = resolve_parsed_tags(parsed, _tag_resolver)
    semantic = resolved["semantic"]

    if verbose:
        print(f"🔍 查询: \"{query}\"")
        print(f"   📋 Resolver 候选(每组 top-5):")
        for polarity in ("positive", "negative"):
            for cat in ("general", "character", "copyright"):
                groups = resolved[polarity][cat]
                if groups:
                    print(f"      {polarity}.{cat}:")
                    for g in groups:
                        if g:
                            print(f"         \"{g[0].get('raw', '?')}\" → "
                                  f"{[c['tag'] for c in g]}")

    # ── 3. GPT-2: 从候选里挑(类 RAG) ──
    if use_gpt_select:
        selected = select_from_candidates(query, resolved)
        pos = selected["positive"]
        neg = selected["negative"]
        if verbose:
            print(f"   🎯 GPT 二选结果:")
            for polarity, sel in (("positive", pos), ("negative", neg)):
                for cat in ("general", "character", "copyright"):
                    if sel[cat]:
                        flat = [t for g in sel[cat] for t in g]
                        print(f"      {polarity}.{cat}: {flat}")
    else:
        # fallback: 每组取 top-1(行为同旧版 flatten_top)
        def to_groups(field):
            return [[g[0]["tag"]] for g in field if g]
        pos = {
            "general":   to_groups(resolved["positive"]["general"]),
            "character": to_groups(resolved["positive"]["character"]),
            "copyright": to_groups(resolved["positive"]["copyright"]),
        }
        neg = {
            "general":   to_groups(resolved["negative"]["general"]),
            "character": to_groups(resolved["negative"]["character"]),
            "copyright": to_groups(resolved["negative"]["copyright"]),
        }

    if verbose:
        print(f"   📝 semantic: {semantic}\n")

    # ── 4. 构造 embedding 文本: 选定的 positive tags + semantic ──
    all_positive = []
    for cat in ("general", "character", "copyright"):
        for group in pos[cat]:
            all_positive.extend(group)   # group 是 list[str]
    embed_text = " ".join(all_positive) + ". " + semantic
    q_vec = embed_query(embed_text)

    # ── 5. FAISS 召回 ──
    scores, indices = _index.search(q_vec, recall_k)

    # ── 6. 逐候选打分 ──
    candidates = []
    for idx, vec_sim in zip(indices[0], scores[0]):
        rec = _records[idx]
        score_detail = _score_candidate(rec, pos, neg, float(vec_sim))

        result = rec.copy()
        result.update(score_detail)
        candidates.append(result)

    # ── 7. 排序 + 截断 ──
    candidates.sort(key=lambda x: x["_final_score"], reverse=True)
    results = candidates[:top_k]
    for rank, item in enumerate(results, start=1):
        item["_rank"] = rank

    if verbose:
        print(f"📊 召回 {len(candidates)} 张候选 → 返回 Top-{len(results)}")
        print(f"   公式: α·vec + β·pos − γ·neg "
              f"(α={ALPHA_VECTOR}, β={BETA_POSITIVE}, γ={GAMMA_NEG})")
        print(f"   {'rank':<5}{'id':<10}{'final':<8}{'vec':<8}{'pos':<8}{'neg':<8}")
        for item in results:
            print(f"   {item['_rank']:<5}{item.get('id', '?'):<10}"
                  f"{item['_final_score']:<8.3f}{item['_vec_sim']:<8.3f}"
                  f"{item['_pos_score']:<8.3f}{item['_neg_score']:<8.3f}")
        print()

    return results


# ─────────────────────────────────────────────
# WD14-only search(ablation 第三臂)
#
# 纯 WD14 检索:只用 WD14 自动标签做倒排检索,完全不碰人工标注。
# 这是 smart_search(human ∪ WD14 合并)、simple_search(embedding baseline)
# 之外的第三条臂,用来隔离演示 WD14 的检索能力,也是 ablation 对照组。
#
# 复用模块已加载的 _wd14_by_id 与 _records,不重读文件。
# 倒排索引 tag -> [(post_id, conf)] 在 import 时构建一次;
# 查询阈值在检索时过滤,扫阈值不需要重建索引。
# ─────────────────────────────────────────────

print("🔧 构建 WD14 倒排索引...")
_wd14_inverted = defaultdict(list)   # tag -> [(post_id, conf), ...]
for _rec in _records:
    _wd14 = _wd14_by_id.get(_rec["id"])
    if not _wd14:
        continue
    for _tag, _conf in _wd14.get("general", []):
        _wd14_inverted[_tag].append((_rec["id"], _conf))
print(f"   ✅ {len(_wd14_inverted)} 个 WD14 tag 进入倒排索引")

_records_by_id = {rec["id"]: rec for rec in _records}


def _normalize_tag(tag: str) -> str:
    return tag.strip().lower().replace(" ", "_")


def wd14_known_tags(query_tags: list[str]) -> dict:
    """报告哪些 query tag 在 WD14 词表里。

    暴露词表覆盖缺口:比如 dragon_claw 在人工 tag 里有、但不在 WD14 的
    9083 词词表里,纯 WD14 检索它结构上就搜不到。
    """
    return {t: (_normalize_tag(t) in _wd14_inverted) for t in query_tags}


def wd14_search(query_tags: list[str], top_k: int = 5, threshold=None,
                mode: str = "and", aggregate: str = "sum",
                verbose: bool = True) -> list[dict]:
    """纯 WD14 倒排检索。

    query_tags : WD14 风格 tag,如 ["dragon_girl", "white_hair"]
                 (自然语言请先经 parse_query / rewrite_query 转成 tag)
    threshold  : 单 tag 置信度下限,默认 WD14_QUERY_THRESHOLD
    mode       : "and" 图须含全部 query tag(均 >= 阈值);"or" 含其一即可
    aggregate  : "sum" 命中 tag 置信度求和作排序分;"mean" 取均值
                 (AND 模式下两者排序等价,差异只在 OR 模式)

    返回与 simple_search / smart_search 同构的 record 列表(含 _rank/_score),
    便于 notebook 用同一套展示逻辑。
    """
    thr = WD14_QUERY_THRESHOLD if threshold is None else threshold
    norm_tags = [_normalize_tag(t) for t in query_tags]
    n_query = len(norm_tags)

    hits = defaultdict(lambda: {"matched": 0, "conf_sum": 0.0, "tags": {}})
    for tag in norm_tags:
        for pid, conf in _wd14_inverted.get(tag, []):
            if conf < thr:
                continue
            h = hits[pid]
            h["matched"] += 1
            h["conf_sum"] += conf
            h["tags"][tag] = conf

    results = []
    for pid, h in hits.items():
        if mode == "and" and h["matched"] < n_query:
            continue
        score = h["conf_sum"] / h["matched"] if aggregate == "mean" else h["conf_sum"]
        rec = _records_by_id.get(pid)
        if rec is None:
            continue
        item = rec.copy()
        item["_score"] = float(score)
        item["_n_matched"] = h["matched"]
        item["_matched_tags"] = {t: round(c, 4) for t, c in h["tags"].items()}
        results.append(item)

    # 主排序: 命中分;tie-break: danbooru score(人气)
    results.sort(key=lambda r: (r["_score"], r.get("score", 0)), reverse=True)
    results = results[:top_k]
    for rank, item in enumerate(results, start=1):
        item["_rank"] = rank

    if verbose:
        miss = [t for t, ok in wd14_known_tags(query_tags).items() if not ok]
        if miss:
            print(f"⚠️ 不在 WD14 词表的 query tag(永不命中): {miss}")
        print(f"🔖 WD14 检索 [{mode}] thr={thr}: {norm_tags}")
        print(f"   {'rank':<5}{'id':<10}{'score':<8}{'matched':<8}")
        for item in results:
            print(f"   {item['_rank']:<5}{item.get('id','?'):<10}"
                  f"{item['_score']:<8.3f}{item['_n_matched']:<8}")
        print()

    return results

