import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── 路径 ──────────────────────────────
BASE_DIR   = Path(__file__).parent       # 项目根目录
DATA_DIR   = BASE_DIR / "data"
IMAGE_DIR  = DATA_DIR / "images"
META_PATH  = DATA_DIR / "metadata.jsonl"
INDEX_PATH = DATA_DIR / "faiss.index"
META_PKL   = DATA_DIR / "index_meta.pkl"
EMBED_CACHE   = DATA_DIR / "embeddings.pkl"

# ── WD14 推理结果路径 ─────────────────
WD14_META_PATH = DATA_DIR / "metadata_with_wd14.jsonl"

# ── Wiki embedding 索引路径(由 build_wiki_emb_index.py 产出) ──
WIKI_EMB_INDEX_PATH = DATA_DIR / "wiki_emb_index.faiss"
WIKI_EMB_TAGS_PATH  = DATA_DIR / "wiki_emb_tags.pkl"

# ── OpenAI ────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","Your_Key_here")
EMBED_MODEL    = "text-embedding-3-small"
CHAT_MODEL     = "gpt-4o-mini"

# ── 索引参数 ──────────────────────────
BATCH_SIZE     = 100
SLEEP_BETWEEN  = 0.5

# ── 爬虫参数 ──────────────────────────
DANBOORU_USERNAME = os.getenv("DANBOORU_USERNAME")
DANBOORU_API_KEY  = os.getenv("DANBOORU_API_KEY")
DANBOORU_BASE_URL = "https://danbooru.donmai.us"
CRAWL_SLEEP       = 0.6

# ── WD14 Tagger 配置 ────────────────────
WD14_MODEL_REPO  = "SmilingWolf/wd-v1-4-moat-tagger-v2"
WD14_MODEL_FILE  = "model.onnx"
WD14_LABEL_FILE  = "selected_tags.csv"

WD14_THRESHOLD_GENERAL   = 0.3
WD14_THRESHOLD_CHARACTER = 0.2

# ── 检索阶段阈值 ──────────────────────
WD14_QUERY_THRESHOLD = 0.25      # 搜索时 WD14 conf 阈值
SEARCH_RECALL_K = 1000           # FAISS 召回候选数

# ── Wiki embedding 兜底参数 ───────────
WIKI_FALLBACK_K = 5              # 兜底召回的 top-K
WIKI_FALLBACK_THRESHOLD = 0.5    # cosine sim 下限(低于不返回)

# ── 字段权重(character > copyright > general) ──
FIELD_WEIGHTS = {
    "character": 0.25,
    "copyright": 0.18,
    "general":   0.10,
}

# ── 混合评分权重: final = α·vec + β·pos − γ·neg ──
ALPHA_VECTOR  = 1.0
BETA_POSITIVE = 0.5
GAMMA_NEG     = 0.3

# ── 爬虫 ─────────────────────────────
CRAWL_TAGS = [
    'animated',
]

RATING_FILTER = "g,s"
SCORE_MIN     = 5
LIMIT_PER_TAG = 5
POSTS_PER_PAGE = 100

PROGRESS_PATH = DATA_DIR / "crawl_progress.json"
