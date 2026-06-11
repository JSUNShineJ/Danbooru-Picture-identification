"""WD14 Tagger:给图片自动打 danbooru 风格的 tag。

模型:SmilingWolf/wd-v1-4-moat-tagger-v2

设计:
- predict_tags_raw: 跑模型,返回所有 conf >= MIN_CONF 的 tag(实验用)
- predict_tags: 在 raw 基础上按业务阈值过滤(日常用)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from config import (
    WD14_MODEL_REPO, WD14_MODEL_FILE, WD14_LABEL_FILE,
    WD14_THRESHOLD_GENERAL, WD14_THRESHOLD_CHARACTER,
)


# raw 阶段保留的最低置信度:足够低不丢信号,又能控制文件大小
# 大约保留每张图 50-200 个 tag(相比模型输出的 6000 个)
MIN_CONF = 0.05


# ─────────────────────────────────────────────
# 模型懒加载
# ─────────────────────────────────────────────

_session: Optional[ort.InferenceSession] = None
_tags_df: Optional[pd.DataFrame] = None
_input_size: int = 448


def _load_model():
    """下载并加载 WD14 模型(只跑一次)。"""
    global _session, _tags_df, _input_size
    
    if _session is not None:
        return
    
    print("📦 加载 WD14 模型(首次会下载约 300MB)...")
    
    model_path = hf_hub_download(WD14_MODEL_REPO, WD14_MODEL_FILE)
    label_path = hf_hub_download(WD14_MODEL_REPO, WD14_LABEL_FILE)
    
    # Mac 上优先 CoreML,fallback CPU
    _session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    
    _tags_df = pd.read_csv(label_path)
    _input_size = _session.get_inputs()[0].shape[1]
    
    print(f"   ✅ 模型加载完成 (provider: {_session.get_providers()[0]})")
    print(f"   📊 标签词表: {len(_tags_df)} 个")
    print(f"   🖼️  输入尺寸: {_input_size}x{_input_size}")


# ─────────────────────────────────────────────
# 图像预处理
# ─────────────────────────────────────────────

def _preprocess(image_path: str) -> np.ndarray:
    """图片 → 模型输入格式(448x448, BGR, float32)。"""
    img = Image.open(image_path).convert("RGB")
    
    w, h = img.size
    size = max(w, h)
    square = Image.new("RGB", (size, size), (255, 255, 255))
    square.paste(img, ((size - w) // 2, (size - h) // 2))
    
    square = square.resize((_input_size, _input_size), Image.BICUBIC)
    
    arr = np.array(square, dtype=np.float32)
    arr = arr[:, :, ::-1]   # RGB → BGR
    arr = np.expand_dims(arr, 0)
    
    return arr


# ─────────────────────────────────────────────
# 核心:raw 推理
# ─────────────────────────────────────────────

def predict_tags_raw(image_path: str, min_conf: float = MIN_CONF) -> dict:
    """跑一次模型,返回所有 conf >= min_conf 的 tag。
    
    用于批量推理 + 阈值扫描:一次推理,任意阈值实验都是字典操作。
    
    返回:
    {
        "general":   [(tag, conf), ...],   # 按 conf 降序
        "character": [(tag, conf), ...],
        "rating":    {"general": 0.8, "sensitive": 0.1, ...},   # 4 个分级全保留
    }
    """
    _load_model()
    
    arr = _preprocess(image_path)
    input_name = _session.get_inputs()[0].name
    probs = _session.run(None, {input_name: arr})[0][0]
    
    general_tags = []
    character_tags = []
    rating_probs = {}
    
    for i, conf in enumerate(probs):
        row = _tags_df.iloc[i]
        category = row["category"]
        name = row["name"]
        c = float(conf)
        
        if category == 9:
            rating_probs[name] = c   # rating 只有 4 个,全保留
        elif category == 0 and c >= min_conf:
            general_tags.append((name, c))
        elif category == 4 and c >= min_conf:
            character_tags.append((name, c))
    
    general_tags.sort(key=lambda x: -x[1])
    character_tags.sort(key=lambda x: -x[1])
    
    return {
        "general":   general_tags,
        "character": character_tags,
        "rating":    rating_probs,
    }


# ─────────────────────────────────────────────
# 日常接口:raw + 业务阈值过滤
# ─────────────────────────────────────────────

def predict_tags(
    image_path: str,
    threshold_general:   float = WD14_THRESHOLD_GENERAL,
    threshold_character: float = WD14_THRESHOLD_CHARACTER,
) -> dict:
    """对一张图预测 tag(已按业务阈值过滤)。
    
    内部 = predict_tags_raw + 阈值过滤。日常调用走这里。
    """
    raw = predict_tags_raw(image_path)
    return {
        "general":   [(t, c) for t, c in raw["general"]   if c >= threshold_general],
        "character": [(t, c) for t, c in raw["character"] if c >= threshold_character],
        "rating":    raw["rating"],
    }


# ─────────────────────────────────────────────
# 批量预测(保留兼容)
# ─────────────────────────────────────────────

def predict_tags_batch(
    image_paths: list[str],
    threshold_general:   float = WD14_THRESHOLD_GENERAL,
    threshold_character: float = WD14_THRESHOLD_CHARACTER,
    progress_every: int = 50,
) -> list[dict]:
    """批量预测(过滤版)。
    
    注:批量实验请直接循环调 predict_tags_raw 并把 raw 结果存盘,
       这样后续阈值扫描不用重跑模型。
    """
    _load_model()
    
    results = []
    total = len(image_paths)
    
    for i, path in enumerate(image_paths, 1):
        try:
            r = predict_tags(path, threshold_general, threshold_character)
            results.append(r)
        except Exception as e:
            print(f"  ⚠️ 失败: {path} ({e})")
            results.append({"general": [], "character": [], "rating": {}})
        
        if i % progress_every == 0 or i == total:
            print(f"  进度: {i}/{total}")
    
    return results


# ─────────────────────────────────────────────
# CLI 测试
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from config import IMAGE_DIR
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        first_img = next(iter(Path(IMAGE_DIR).glob("*")))
        test_path = str(first_img)
    
    print(f"\n🖼️  测试图片: {test_path}\n")
    
    result = predict_tags(test_path)
    
    print("📋 General tags:")
    for tag, conf in result["general"][:20]:
        print(f"   {conf:.3f}  {tag}")
    
    if result["character"]:
        print("\n👤 Character tags:")
        for tag, conf in result["character"]:
            print(f"   {conf:.3f}  {tag}")
    
    print("\n🔖 Rating:")
    for rating, conf in sorted(result["rating"].items(), key=lambda x: -x[1]):
        print(f"   {conf:.3f}  {rating}")