"""根据自然语言查询,从 FAISS 索引里搜出最相似的图片。"""

import pickle
from pathlib import Path

import json

import numpy as np
import faiss
from openai import OpenAI

from config import (
    OPENAI_API_KEY, EMBED_MODEL,
    INDEX_PATH, META_PKL,
    CHAT_MODEL, SEARCH_RECALL_K
)


client = OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────
# 加载索引和元数据(模块级,导入时只跑一次)
# ─────────────────────────────────────────────

print("📦 加载 FAISS 索引...")
_index = faiss.read_index(str(INDEX_PATH))

with open(META_PKL, "rb") as f:
    _records = pickle.load(f)

print(f"   ✅ {_index.ntotal} 条向量,{len(_records)} 条元数据")


### 下面是smart search的代码,目前还在测试阶段,接口可能会变动

# ─────────────────────────────────────────────
# 查询解析(GPT 结构化输出)
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
   - Ambiguous → prefer positive
4. If a category has no tags, use empty list []

# Examples
User: 粉色头发的龙娘,从背后看
Output: {"positive": {"general": ["1girl", "dragon_girl", "pink_hair", "from_behind"], "character": [], "copyright": []}, "negative": {"general": [], "character": [], "copyright": []}, "semantic": "dragon girl with pink hair from behind"}

User: 初音未来穿校服
Output: {"positive": {"general": ["1girl", "school_uniform"], "character": ["hatsune_miku"], "copyright": ["vocaloid"]}, "negative": {"general": [], "character": [], "copyright": []}, "semantic": "Hatsune Miku wearing school uniform"}

User: 龙娘,异色皮肤
Output: {"positive": {"general": ["1girl", "dragon_girl", "colored_skin"], "character": [], "copyright": []}, "negative": {"general": [], "character": [], "copyright": []}, "semantic": "dragon girl with colored skin"}

User: 龙娘,不要异色皮肤
Output: {"positive": {"general": ["1girl", "dragon_girl"], "character": [], "copyright": []}, "negative": {"general": ["colored_skin", "multicolored_skin"], "character": [], "copyright": []}, "semantic": "dragon girl with normal skin"}

User: blue archive characters, not miku
Output: {"positive": {"general": [], "character": [], "copyright": ["blue_archive"]}, "negative": {"general": [], "character": ["hatsune_miku"], "copyright": []}, "semantic": "Blue Archive characters"}

Now parse:
"""

def parse_query(natural_query: str) -> dict:
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
    
    # 兜底:确保结构完整
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
# Smart Search:解析 + 召回 + 过滤
# ─────────────────────────────────────────────

def smart_search(
    query: str,
    top_k: int = 5,
    recall_k: int = SEARCH_RECALL_K,
    verbose: bool = True,
) -> list[dict]:
    parsed = parse_query(query)
    pos = parsed["positive"]
    neg = parsed["negative"]
    semantic = parsed["semantic"]
    
    if verbose:
        print(f"🔍 查询: \"{query}\"")
        print(f"   ✅ positive:")
        print(f"      general:   {pos['general']}")
        print(f"      character: {pos['character']}")
        print(f"      copyright: {pos['copyright']}")
        print(f"   ❌ negative:")
        print(f"      general:   {neg['general']}")
        print(f"      character: {neg['character']}")
        print(f"      copyright: {neg['copyright']}")
        print(f"   📝 semantic: {semantic}\n")
    
    # ── embedding 召回 ──
    # 把所有 positive tag 拍平拼成文本
    all_positive = pos["general"] + pos["character"] + pos["copyright"]
    embed_text = " ".join(all_positive) + ". " + semantic
    q_vec = embed_query(embed_text)
    
    scores, indices = _index.search(q_vec, recall_k)
    
    # ── 分类别硬过滤 ──
    results = []
    stats = {"total": 0, "miss_pos": 0, "hit_neg": 0}
    
    for idx, score in zip(indices[0], scores[0]):
        stats["total"] += 1
        meta = _records[idx]
        
        # 三个字段分别建 set
        general_set   = set((meta.get("tag_string_general")   or "").split())
        character_set = set((meta.get("tag_string_character") or "").split())
        copyright_set = set((meta.get("tag_string_copyright") or "").split())
        
        # 必须包含所有 positive(分别在对应字段)
        if not (
            all(t in general_set   for t in pos["general"]) and
            all(t in character_set for t in pos["character"]) and
            all(t in copyright_set for t in pos["copyright"])
        ):
            stats["miss_pos"] += 1
            continue
        
        # 不能包含任何 negative
        if (
            any(t in general_set   for t in neg["general"]) or
            any(t in character_set for t in neg["character"]) or
            any(t in copyright_set for t in neg["copyright"])
        ):
            stats["hit_neg"] += 1
            continue
        
        result = meta.copy()
        result["_rank"] = len(results) + 1
        result["_score"] = float(score)
        results.append(result)
        
        if len(results) >= top_k:
            break
    
    if verbose:
        print(f"📊 召回 {stats['total']} 张候选")
        print(f"   - 缺少 positive 被过滤: {stats['miss_pos']}")
        print(f"   - 命中 negative 被过滤: {stats['hit_neg']}")
        print(f"   - 最终返回: {len(results)} 张\n")
    
    return results