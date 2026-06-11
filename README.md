# PicSearch

Natural-language semantic image search over a Danbooru-style corpus, built with OpenAI embeddings, FAISS, and a WD14 vision tagger.

You describe what you want in plain English. The system parses the query, grounds it against real Danbooru tags, and retrieves the most relevant images — including images that carry no human annotations at all.

## How it works

PicSearch runs three retrieval strategies over the same corpus, so each can be compared against the others:

- **smart** — a GPT model parses the query into structured positive / negative / semantic fields. The resulting tags are embedded with `text-embedding-3-small` and matched against a FAISS index built from each image's tags (human tags ∪ WD14 tags). Scoring is weighted by field: character > copyright > general.
- **simple** — a plain embedding baseline: query → embedding → FAISS top-K. Kept as a control to measure what the query parsing actually adds.
- **wd14-only** — an inverted index built directly from WD14 tags, with no embeddings and no FAISS. This is the only arm that can retrieve images with no human tags, which is the main reason WD14 was added.

## WD14 auto-tagging

Images crawled from arbitrary sources don't come with Danbooru metadata, so they can't be searched by tag. To fix that, PicSearch runs the **WD14 tagger** (`wd-v1-4-moat-tagger-v2`, ViT-based, ONNX / CPU) to predict Danbooru-style tags directly from pixels.

- Full inference has been run over the entire corpus (2,662 images). Output is cached in `metadata_with_wd14.jsonl` so it doesn't need to be recomputed.
- Thresholds are deliberately asymmetric — 0.45 at indexing, 0.25 at query time, 0.30 for evaluation. The pipeline is recall-first: WD14 over-tags on purpose, and the downstream GPT re-select step acts as the precision filter.
- A threshold sweep was run to choose those values; the precision/recall curve is in `wd14_pr_curve.png`. Evaluation uses F1 rather than accuracy, since the task is sparse multi-label and accuracy is misleading when negative labels dominate.

The net effect: the corpus goes from "searchable only if it has human tags" to "searchable regardless of source."

## RAG tag grounding

A GPT model will happily invent tags that don't exist in the Danbooru vocabulary. To prevent that, query parsing runs a generate → ground → re-select loop:

1. GPT proposes candidate tags for the query.
2. Each candidate is grounded against the Danbooru wiki dataset (`isek-ai/danbooru-wiki-2024`) — exact match if the tag exists, nearest embedding candidates if it doesn't.
3. GPT re-selects from the real, grounded options.

`Build wiki db index.py` builds the wiki embedding index used in step 2. The grounding loop itself runs at query time inside `search_new.py`.

## Pipeline

**Crawler (`crawler.py`)** — incremental crawler on the Danbooru API. Supports checkpoint resumption, tag combinations, and rating/score filtering. Re-running only fetches new images, no duplicates.

**Indexer (`indexer.py`)** — concatenates each image's tags into a text string, embeds them with `text-embedding-3-small`, and stores them in a FAISS `IndexFlatIP` index. Embeddings are cached by post ID, so re-running only calls the API for new images.

**Search (`search_new.py`)** — the three retrieval arms above, plus the GPT query parser and the RAG grounding loop.

## Project structure

```
.
├── code/
│   ├── crawler.py                  # incremental Danbooru crawler
│   ├── indexer.py                  # builds the FAISS index + embedding cache
│   ├── search_new.py               # smart / simple / wd14-only search
│   ├── config.py                   # settings (read API keys from env)
│   ├── run_wd14_batch.py           # batch WD14 inference over the corpus
│   ├── Build wiki db index.py      # builds the Danbooru wiki grounding index
│   ├── tagger.py                   # tag pictures with WD14
│   ├── tag_resolver.py             # RAG systerm built to gain real tag output
│   └── searchresult.ipynb          # run queries + view results inline
├── data/
│   ├── metadata_with_wd14.jsonl    # WD14 tagging output (cached)
├── WD14_threshold_test/
│   ├── WD14-threshold-test.py      # threshold sweep experiment
│   ├── wd14_pr_curve.png           # precision/recall curve from the sweep
├── upcomingfeacher/                # work-in-progress features
├── requirements.txt
└── README.md
```

## Usage

Install dependencies, then set your OpenAI API key as an environment variable (don't hardcode it — `config.py` reads from the environment). Danbooru credentials are optional.

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

python code/crawler.py                  # crawl images into data/
python code/indexer.py                  # build the FAISS index
python code/run_wd14_batch.py           # tag images with WD14
python "code/Build wiki db index.py"    # build the wiki grounding index
```

Then open `code/searchresult.ipynb` to run queries and view results inline.

## Roadmap

- **Empty-tag diagnostic** — count how many of the total images have no human tags, to quantify how much retrieval the WD14-only arm actually adds.
- **Hard negation** — negation is currently a soft score penalty, so a query like "girls without exposed skin" can still surface excluded tags. Move excluded tags to a hard rerank filter.
- **Scale** — `IndexFlatIP` is fine at the current size; switch to an HNSW index once the corpus grows.
