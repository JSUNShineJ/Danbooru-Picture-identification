import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 召回阶段拿多少候选(留余量给后续过滤)
SEARCH_RECALL_K = 1000

# ── 路径 ──────────────────────────────
BASE_DIR   = Path(__file__).parent       # 项目根目录
DATA_DIR   = BASE_DIR / "data"
IMAGE_DIR  = DATA_DIR / "images"
META_PATH  = DATA_DIR / "metadata.jsonl"
INDEX_PATH = DATA_DIR / "faiss.index"
META_PKL   = DATA_DIR / "index_meta.pkl"
EMBED_CACHE   = DATA_DIR / "embeddings.pkl" 

# ── OpenAI ────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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

# 阈值:置信度高于这个值的 tag 才被采纳
WD14_THRESHOLD_GENERAL   = 0.25    # 普通 tag(default 0.35)
WD14_THRESHOLD_CHARACTER = 0.2   # 角色 tag(更严格)

# 要爬取的 tag 列表,每个 tag 爬多少张
CRAWL_TAGS = [
    
    'animated',  # 想爬什么直接加，这里的逻辑是or，也就是只要有其中之一的tag就会爬取  
    
   # 想爬什么直接加，这里的逻辑是or，也就是只要有其中之一的tag就会爬取
]
#默认的metatag设置
RATING_FILTER = "g,s"     # None 表示不过滤
SCORE_MIN     = 5         # None 表示不过滤
LIMIT_PER_TAG = 5       # 每个 tag 目标爬取数量
POSTS_PER_PAGE = 100          # danbooru API 每页最多 200,设 100 比较稳

# 进度文件:记录每个 tag 爬到哪一页/哪个 ID
PROGRESS_PATH = DATA_DIR / "crawl_progress.json"
