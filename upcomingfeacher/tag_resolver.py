from collections import Counter, defaultdict
import re

TAG_FIELDS = {
    "general": "tag_string_general",
    "character": "tag_string_character",
    "copyright": "tag_string_copyright",
}

def norm_text(text: str) -> str:
    return re.sub(r"\s+", "_", (text or "").strip().lower())

def tag_base(tag: str) -> str:
    return re.sub(r"_\(.+\)$", "", tag)

def build_local_vocab(records):
    vocab = {cat: Counter() for cat in TAG_FIELDS}
    for meta in records:
        for cat, field in TAG_FIELDS.items():
            for tag in (meta.get(field) or "").split():
                vocab[cat][tag] += 1
    return vocab

def build_danbooru_lookup(ds):
    by_tag = {}
    alias_to_tags = defaultdict(list)
    base_to_tags = defaultdict(list)

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

def build_resolver(records, danbooru_ds):
    by_tag, alias_to_tags, base_to_tags = build_danbooru_lookup(danbooru_ds)
    return {
        "local_vocab": build_local_vocab(records),
        "danbooru_by_tag": by_tag,
        "danbooru_alias": alias_to_tags,
        "danbooru_base": base_to_tags,
    }

def resolve_tag(raw, category, resolver, context_tags=None, limit=3):
    key = norm_text(raw)
    if not key:
        return []

    by_tag = resolver["danbooru_by_tag"]
    aliases = resolver["danbooru_alias"]
    base_to_tags = resolver["danbooru_base"]
    local_vocab = resolver["local_vocab"]
    context_tags = context_tags or set()

    candidates = set()
    if key in by_tag:
        candidates.add(key)
    candidates.update(aliases.get(key, []))
    candidates.update(base_to_tags.get(key, []))

    ranked = []
    for tag in candidates:
        base = tag_base(tag)
        suffix = tag[len(base) + 2 : -1] if tag.startswith(base + "_(") else ""
        exists_local = tag in local_vocab.get(category, {})

        score = 0
        score += 2.0 if tag == key else 0
        score += 1.2 if base == key else 0
        score += 1.0 if suffix and suffix in context_tags else 0
        score += 0.8 if exists_local else 0

        ranked.append({
            "raw": raw,
            "tag": tag,
            "category": category,
            "exists_local": exists_local,
            "score": score,
        })

    return sorted(ranked, key=lambda x: x["score"], reverse=True)[:limit]

def resolve_parsed_tags(parsed, resolver):
    out = {
        "positive": {"general": [], "character": [], "copyright": []},
        "negative": {"general": [], "character": [], "copyright": []},
        "semantic": parsed.get("semantic") or "",
    }

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

def flatten_top(groups):
    return [group[0]["tag"] for group in groups if group]

def group_hits(groups, tag_set):
    return sum(1 for group in groups if any(c["tag"] in tag_set for c in group))
