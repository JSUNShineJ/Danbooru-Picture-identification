"""Danbooru 增量爬虫。
按 config 里的 tag 列表爬图,自动去重,支持断点续爬。
"""

import json
import time
from pathlib import Path
from typing import Optional

import requests

from config import (
    DANBOORU_BASE_URL, CRAWL_SLEEP,
    CRAWL_TAGS, LIMIT_PER_TAG, POSTS_PER_PAGE,
    IMAGE_DIR, META_PATH, PROGRESS_PATH,
    RATING_FILTER, SCORE_MIN,
    DANBOORU_USERNAME, DANBOORU_API_KEY, 
)
# ─────────────────────────────────────────────
# 全局 session(带登录认证 + 持久化 cookie)
# ─────────────────────────────────────────────
_session = requests.Session()
_session.headers.update({
    "User-Agent": "picsearch-crawler/0.1 (educational project)"
})

if DANBOORU_USERNAME and DANBOORU_API_KEY:
    _session.auth = (DANBOORU_USERNAME, DANBOORU_API_KEY)
    print(f"✅ 已启用 danbooru 登录认证({DANBOORU_USERNAME})")
else:
    print("⚠️ 未配置 danbooru 账号,可能遇到 CDN 限制")

# ─────────────────────────────────────────────
# 进度文件读写
# ─────────────────────────────────────────────

def load_progress() -> dict:
    """读已有进度。
    返回格式:{ "dragon_girl": {"last_id": 123, "fetched_count": 50}, ... }
    """
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress: dict):
    """把进度写回磁盘。"""
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def load_existing_ids() -> set:
    """读 metadata.jsonl,把已经爬过的 ID 全部装到一个 set 里,用来去重。"""
    ids = set()
    if not META_PATH.exists():
        return ids
    
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except Exception:
                continue
    return ids


# ─────────────────────────────────────────────
# 调 Danbooru API
# ─────────────────────────────────────────────

def fetch_posts_page(tag: str, before_id: Optional[int] = None) -> list[dict]:
    """从 danbooru 拉一页 posts。"""
    query_parts = [tag]
    if RATING_FILTER:
        query_parts.append(f"rating:{RATING_FILTER}")
    if SCORE_MIN is not None:
        query_parts.append(f"score:>{SCORE_MIN}")
    full_query = " ".join(query_parts)
    
    params = {"tags": full_query, "limit": POSTS_PER_PAGE}
    if before_id is not None:
        params["page"] = f"b{before_id}"
    
    # 用 session 而不是 requests.get
    resp = _session.get(
        f"{DANBOORU_BASE_URL}/posts.json",
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def download_image(url: str, save_path: Path) -> bool:
    """下载图片。"""
    try:
        # 用 session,自动带上认证和 cookie
        with _session.get(url, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"  ⚠️ 下载失败: {url} ({e})")
        return False
   


# ─────────────────────────────────────────────
# 读取tag信息
# ─────────────────────────────────────────────
def extract_metadata(post: dict, image_path: str, source_tag: str) -> dict:
    """从 danbooru 的 post 里挑出我们关心的字段。"""
    return {
        "id":     post["id"],
        "rating": post.get("rating"),
        "score":  post.get("score"),
        "tag_string_general":   post.get("tag_string_general", ""),
        "tag_string_character": post.get("tag_string_character", ""),
        "tag_string_copyright": post.get("tag_string_copyright", ""),
        "tag_string_artist":    post.get("tag_string_artist", ""),
        "file_url":             post.get("file_url"),
        "local_image_path":     image_path,
        "crawled_tag":          source_tag,
    }


# ─────────────────────────────────────────────
# 单个 tag 的爬取
# ─────────────────────────────────────────────

def crawl_tag(tag: str, progress: dict, existing_ids: set) -> int:
    """爬一个 tag,本次新增 LIMIT_PER_TAG 张,返回实际新增数量。"""
    
    # 读这个 tag 的历史进度(只用 last_id 作为分页起点)
    tag_state = progress.get(tag, {"last_id": None, "fetched_count": 0})
    last_id   = tag_state["last_id"]
    already   = tag_state["fetched_count"]
    
    # 本次目标:新增 LIMIT_PER_TAG 张
    need = LIMIT_PER_TAG
    
    print(f"[{tag}] 已有累计 {already} 张,本次目标新增 {need} 张")
    
    new_count = 0
    
    with open(META_PATH, "a", encoding="utf-8") as meta_file:
        while new_count < need:
            posts = fetch_posts_page(tag, before_id=last_id)
            
            if not posts:
                print(f"[{tag}] 没有更多数据了,停止")
                break
            
            for post in posts:
                if new_count >= need:
                    break
                
                post_id = post["id"]
                last_id = post_id
                
                if post_id in existing_ids:
                    continue
                if not post.get("file_url"):
                    continue

                # 白名单过滤:只下载常规图片格式,排除 ugoira(.zip)、视频(.mp4)、动图(.gif)等
                ALLOWED_EXTS = {"jpg", "jpeg", "png", "webp"}
                ext = post.get("file_ext", "").lower()  # 优先用 danbooru 返回的字段,比解析 URL 可靠
                if not ext:
                    # 兜底:从 URL 推断
                    ext = post["file_url"].split(".")[-1].split("?")[0].lower()

                if ext not in ALLOWED_EXTS:
                    print(f"  ⏭️  跳过 id={post_id}: 不支持的格式 .{ext}")
                    continue

                save_path = IMAGE_DIR / f"{post_id}.{ext}"
                
                meta = extract_metadata(post, str(save_path), tag)
                meta_file.write(json.dumps(meta, ensure_ascii=False) + "\n")
                meta_file.flush()
                
                existing_ids.add(post_id)
                new_count += 1
                
                print(f"  ✅ [{tag}] {new_count}/{need}  id={post_id}")
                
                time.sleep(CRAWL_SLEEP)
            
            # 每页爬完保存进度
            progress[tag] = {
                "last_id":       last_id,
                "fetched_count": already + new_count,
            }
            save_progress(progress)
    
    return new_count

# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

def crawl_all():
    """按 CRAWL_TAGS 挨个爬。"""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    progress     = load_progress()
    existing_ids = load_existing_ids()
    
    print(f"📂 已爬过的图片:{len(existing_ids)} 张\n")
    
    total_new = 0
    for tag in CRAWL_TAGS:
        total_new += crawl_tag(tag, progress, existing_ids)
    
    print(f"\n🎉 全部完成,本次新增 {total_new} 张")


if __name__ == "__main__":
    crawl_all()
