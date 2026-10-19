# -*- coding: utf-8 -*-
"""SuiYan AI - structured product store + knowledge store on Chroma.

Two separate collections in the same persist directory:
  - "suiyan_products"  : one document per SKU, with scalar metadata for
                         post-filtering (Chroma metadata does not support lists)
  - "suiyan_knowledge" : cleaned knowledge documents (teammate PDFs, Xinhua
                         guide, makeup knowledge, shade data)

Retrieval strategy (product store):
  1. vector search top-N
  2. python-side post filter on skin_type / tone_match
  3. graceful degradation: if filtered < 3 results, relax the filter
"""
import json
import os
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from suiyan_schemas import SkinProfile

# ---------------------------------------------------------------------------
# Paths & constants (relative to the RAGProject root when run from there)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
KNOWLEDGE_DIR = os.path.join(DATA_DIR, "knowledge")
PRODUCTS_JSON = os.path.join(DATA_DIR, "products.json")

PERSIST_DIR = os.path.join(BASE_DIR, "suiyan_chroma.db")
PRODUCT_COLLECTION = "suiyan_products"
KNOWLEDGE_COLLECTION = "suiyan_knowledge"

EMBED_MODEL = "text-embedding-v4"


def get_embeddings() -> DashScopeEmbeddings:
    return DashScopeEmbeddings(model=EMBED_MODEL)


# ---------------------------------------------------------------------------
# Product store
# ---------------------------------------------------------------------------
def load_products(path: str = PRODUCTS_JSON) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["products"]


def product_to_text(p: Dict) -> str:
    """Serialize one SKU into a retrieval-friendly Chinese text block."""
    shade_lines = []
    for s in p.get("shades", []):
        tone = s.get("tone") or ""
        desc = s.get("desc") or ""
        shade_lines.append(f"{s['shade_id']}({tone},{desc})")
    shades_str = "; ".join(shade_lines) if shade_lines else "无色号(护肤品)"

    nickname = p.get("nickname", "")
    return (
        f"【产品】{p['name']}" + (f"（{nickname}）" if nickname else "") + "\n"
        f"【类别】{p['category']}  【系列】{p['series']}\n"
        f"【价格】¥{p['price']} / {p.get('size', '')}\n"
        f"【质地】{p.get('texture', '')}\n"
        f"【适合肤质】{'、'.join(p.get('suitable_skin', []))}\n"
        f"【适配肤色】{'、'.join(p.get('tone_match', []))}\n"
        f"【色号】{shades_str}\n"
        f"【功效】{p.get('benefits', '')}\n"
        f"【用法】{p.get('usage_note', '')}"
    )


class SuiyanProductStore:
    """Vector index over the structured product catalog with post-filtering."""

    def __init__(self, embedding=None):
        self.embedding = embedding or get_embeddings()
        self.store = Chroma(
            collection_name=PRODUCT_COLLECTION,
            embedding_function=self.embedding,
            persist_directory=PERSIST_DIR,
        )
        self._products: List[Dict] = []
        self._by_id: Dict[str, Dict] = {}
        self._by_name: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def ingest(self, force: bool = False) -> int:
        """Build the product index. Returns number of documents in store.

        force=True wipes and rebuilds; default skips if already populated.
        """
        if force:
            self.store.reset()
        elif self.store._collection.count() > 0:
            self._load_cache()
            return self.store._collection.count()

        self._products = load_products()
        texts, metadatas, ids = [], [], []
        for p in self._products:
            texts.append(product_to_text(p))
            metadatas.append({
                # Chroma metadata must be scalar: join lists with '|'
                "product_id": p["product_id"],
                "name": p["name"],
                "category": p["category"],
                "series": p["series"],
                "suitable_skin": "|".join(p.get("suitable_skin", [])),
                "tone_match": "|".join(p.get("tone_match", [])),
                "price": int(p.get("price", 0)),
                "is_color": 1 if p.get("shades") else 0,
            })
            ids.append(p["product_id"])
            self._by_id[p["product_id"]] = p
            self._by_name[p["name"]] = p

        self.store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        n = self.store._collection.count()
        print(f"[product_store] ingested {len(texts)} SKUs, collection size={n}")
        return n

    def _load_cache(self):
        """Reload product dicts for post-filtering without re-embedding."""
        self._products = load_products()
        for p in self._products:
            self._by_id[p["product_id"]] = p
            self._by_name[p["name"]] = p

    # ------------------------------------------------------------------
    # Filtered retrieval with graceful degradation
    # ------------------------------------------------------------------
    def search(self, profile: SkinProfile, k: int = 8) -> List[Dict]:
        """Retrieve with an auto-built query from the profile alone."""
        query = (
            f"{profile.skin_type}肌熟龄 {'、'.join(profile.concerns) if profile.concerns else '日常'} "
            f"{profile.occasion} {profile.skin_tone} 粉底口红面霜底妆"
        )
        return self.search_text(query, profile, k=k)

    def search_text(self, query: str, profile: SkinProfile, k: int = 8) -> List[Dict]:
        """Retrieve with a caller-provided query (e.g. enriched by voice
        request), applying the same skin/tone post-filtering."""
        n_fetch = min(k * 3, 30)
        docs = self.store.similarity_search(query, k=n_fetch)

        skin = profile.skin_type
        tone = profile.undertone
        # map undertone to tone_match vocabulary
        tone_alts = {
            "冷调": ["冷白", "中性"],
            "暖调": ["暖黄", "自然", "中性"],
            "中性调": ["中性", "自然", "冷白", "暖黄"],
        }.get(tone, ["自然", "中性"])

        strict, relaxed = [], []
        for d in docs:
            m = d.metadata
            p = self._by_id.get(m.get("product_id"))
            if p is None:
                continue
            skins = m.get("suitable_skin", "").split("|")
            tones = m.get("tone_match", "").split("|")
            skin_ok = (skin in skins) or (skin == "敏感肌" and "敏感肌" in skins)
            tone_ok = any(t in tones for t in tone_alts)
            if skin_ok and tone_ok:
                strict.append(p)
            elif skin_ok:
                relaxed.append(p)

        picked = strict[:k]
        if len(picked) < 3:  # graceful degradation: relax tone filter
            for p in relaxed:
                if p not in picked:
                    picked.append(p)
            if len(picked) < 3:  # second fallback: raw vector order
                for d in docs:
                    p = self._by_id.get(d.metadata.get("product_id"))
                    if p and p not in picked:
                        picked.append(p)
        picked = picked[:k]
        return self._ensure_categories(picked, k)

    def _ensure_categories(self, picked: List[Dict], k: int) -> List[Dict]:
        """Category quota: every plan needs lip makeup (lipstick/gloss) and a
        base product. If missing, run a dedicated vector query to backfill."""
        def has_cat(cat_set):
            return any(p.get("category") in cat_set for p in picked)

        backfills = []
        if not has_cat({"口红", "唇釉"}):
            backfills.append(("口红 唇釉 豆沙 显气色 显白", {"口红", "唇釉"}))
        if not has_cat({"粉底", "气垫"}):
            backfills.append(("粉底 气垫 底妆 养肤", {"粉底", "气垫"}))
        for query, cats in backfills:
            docs = self.store.similarity_search(query, k=12)
            for d in docs:
                p = self._by_id.get(d.metadata.get("product_id"))
                if p and p.get("category") in cats and p not in picked:
                    picked.append(p)
                    break  # one backfill per missing category is enough
        return picked

    # ------------------------------------------------------------------
    # Helpers for anti-hallucination checks
    # ------------------------------------------------------------------
    def shade_exists(self, product_name: str, shade: Optional[str]) -> bool:
        p = self._by_name.get(product_name)
        if p is None or not shade:
            return False
        return any(s["shade_id"] == shade for s in p.get("shades", []))


# ---------------------------------------------------------------------------
# Knowledge store
# ---------------------------------------------------------------------------
class SuiyanKnowledgeStore:
    """Vector index over cleaned knowledge documents (txt files)."""

    def __init__(self, embedding=None):
        self.embedding = embedding or get_embeddings()
        self.store = Chroma(
            collection_name=KNOWLEDGE_COLLECTION,
            embedding_function=self.embedding,
            persist_directory=PERSIST_DIR,
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""],
            length_function=len,
        )

    def ingest(self, force: bool = False) -> int:
        """Ingest every .txt file in data/knowledge. Idempotent per file via
        a simple marker file (suiyan_kb_ingested.flag)."""
        flag = os.path.join(PERSIST_DIR, "suiyan_kb_ingested.flag")
        if not force and os.path.exists(flag):
            return self.store._collection.count()
        if force:
            self.store.reset()

        texts, metadatas = [], []
        for fn in sorted(os.listdir(KNOWLEDGE_DIR)):
            if not fn.endswith(".txt"):
                continue
            path = os.path.join(KNOWLEDGE_DIR, fn)
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            if not raw.strip():
                continue
            chunks = self.spliter.split_text(raw)
            texts.extend(chunks)
            metadatas.extend({"source": fn, "layer": "knowledge"} for _ in chunks)

        if texts:
            self.store.add_texts(texts=texts, metadatas=metadatas)
        os.makedirs(PERSIST_DIR, exist_ok=True)
        with open(flag, "w", encoding="utf-8") as f:
            f.write("done")
        n = self.store._collection.count()
        print(f"[knowledge_store] ingested {len(texts)} chunks, collection size={n}")
        return n

    def search(self, query: str, k: int = 4):
        return self.store.similarity_search(query, k=k)
