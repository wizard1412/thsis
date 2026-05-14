"""
RAG Retriever for LLM-Lasso Feature Scoring  (PD / MDS-UPDRS adaptation)
==========================================================================
Adapts the RAG pipeline from Zhang et al. (2025) LLM-Lasso for the
Parkinson's disease finger-tapping domain.

The original repo uses ChromaDB + OMIM + PubMed (gene-centric).
This version replaces OMIM with a local PDF corpus and keeps PubMed
retrieval, which works without any API key.

Two retrieval sources (both optional, independently toggled):
  1. LOCAL  — PDF/TXT papers in knowledge_base/ → FAISS vector index
  2. PUBMED — live PubMed queries via LangChain's PubmedQueryRun

Usage
-----
    # One-time index build (for local source):
    python rag_retriever.py build

    # Test retrieval:
    python rag_retriever.py query period_quartile_range num_peaks

    # Check status:
    python rag_retriever.py check

Dependencies
------------
    pip install faiss-cpu sentence-transformers pymupdf
    pip install langchain langchain-community    # for PubMed source
"""

import re
import pickle
import shutil
import tempfile
import time
from typing import Optional
import numpy as np
from pathlib import Path

# ── optional deps — fail gracefully ───────────────────────────────────────
try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from langchain_community.tools.pubmed.tool import PubmedQueryRun
    HAS_PUBMED = True
except ImportError:
    HAS_PUBMED = False

from llm_lasso_config import PROJECT_ROOT, OUTPUT_DIR, FEATURE_DESCRIPTIONS

# ── Paths ──────────────────────────────────────────────────────────────────
KB_DIR      = PROJECT_ROOT / "knowledge_base"
INDEX_DIR   = OUTPUT_DIR   / "rag_index"
INDEX_FILE  = INDEX_DIR    / "faiss.index"
CHUNKS_FILE = INDEX_DIR    / "chunks.pkl"

# ── Hyper-parameters ────────────────────────────────────────────────────────
CHUNK_SIZE    = 400   # reverted: 1000 caused mds_updrs.txt to dominate context
CHUNK_OVERLAP = 80
TOP_K_LOCAL   = 2     # passages from local index per feature
TOP_K_PUBMED  = 1     # PubMed results per query type per feature
EMBED_MODEL   = "all-MiniLM-L6-v2"

# Toggle retrieval sources here
USE_LOCAL_INDEX    = True   # requires knowledge_base/ + built index
USE_PUBMED         = True   # live PubMed queries (needs internet + langchain)
PER_FEATURE        = False  # True = retrieve per feature (precise, ~30x slower)
                             # False = one batch query for all features (fast)
                             # NOTE: PER_FEATURE=True causes LLM to assign low penalties
                             # to too many features simultaneously, hurting discrimination.

CATEGORY_DESCRIPTION = "MDS-UPDRS Item 3.4 Finger Tapping in Parkinson's disease"


# ═══════════════════════════════════════════════════════════════════════════
# Text utilities
# ═══════════════════════════════════════════════════════════════════════════

def _split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Recursive character-level splitter that respects natural text boundaries.
    Mirrors LangChain's RecursiveCharacterTextSplitter logic used in the
    original LLM-Lasso repo (chunking.py).
    Priority: paragraph → sentence → word → character.
    """
    separators = ["\n\n", "\n", ". ", " "]

    def _split(t: str, seps: list) -> list:
        if not seps or len(t) <= size:
            return [t] if len(t.strip()) > 50 else []
        sep = seps[0]
        parts = t.split(sep)
        chunks, current = [], ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = part
        if current.strip():
            chunks.append(current.strip())
        # Recursively split any chunk still too large
        result = []
        for c in chunks:
            if len(c) > size:
                result.extend(_split(c, seps[1:]))
            elif len(c) > 50:
                result.append(c)
        return result

    raw = _split(text, separators)

    # Apply overlap: each chunk includes the tail of the previous chunk
    if overlap <= 0 or len(raw) <= 1:
        return raw

    overlapped = [raw[0]]
    for i in range(1, len(raw)):
        tail = overlapped[-1][-overlap:]
        overlapped.append(tail + " " + raw[i] if tail else raw[i])
    return overlapped


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Source 1: Local PDF/TXT corpus → FAISS
# ═══════════════════════════════════════════════════════════════════════════

def _read_pdf(path: Path) -> str:
    if not HAS_PYMUPDF:
        raise ImportError("pip install pymupdf")
    doc = fitz.open(str(path))
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def load_corpus(kb_dir: Path = KB_DIR) -> list[dict]:
    if not kb_dir.exists():
        print(f"[RAG-Local] knowledge_base/ not found: {kb_dir}")
        return []
    docs = []
    # Files excluded from indexing (criteria already in prompt template)
    EXCLUDE_FROM_INDEX = {"mds_updrs_item34_domain_knowledge.txt"}

    for path in sorted(kb_dir.iterdir()):
        if path.name in EXCLUDE_FROM_INDEX:
            print(f"[RAG-Local] Skipped (excluded): {path.name}")
            continue
        try:
            if path.suffix.lower() == ".pdf":
                text = _read_pdf(path)
            elif path.suffix.lower() == ".txt":
                text = path.read_text(encoding="utf-8", errors="ignore")
            else:
                continue
            docs.append({"source": path.name, "text": text})
            print(f"[RAG-Local] Loaded {path.suffix.upper()}: {path.name}  ({len(text):,} chars)")
        except Exception as e:
            print(f"[RAG-Local] Failed {path.name}: {e}")
    return docs


class LocalIndexRetriever:
    """FAISS-backed retriever over local PDF/TXT corpus."""

    def __init__(self, embed_model: str = EMBED_MODEL):
        self._embed_model_name = embed_model
        self._model  = None
        self._index  = None
        self._chunks: list[dict] = []

    @property
    def model(self):
        if self._model is None:
            if not HAS_ST:
                raise ImportError("pip install sentence-transformers")
            print(f"[RAG-Local] Loading embedding model: {self._embed_model_name}")
            self._model = SentenceTransformer(self._embed_model_name)
        return self._model

    def is_ready(self) -> bool:
        return self._index is not None and len(self._chunks) > 0

    def build_index(self, force: bool = False) -> None:
        if not HAS_FAISS:
            raise ImportError("pip install faiss-cpu")
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        if not force and INDEX_FILE.exists() and CHUNKS_FILE.exists():
            print("[RAG-Local] Loading cached index...")
            # Copy to a temp ASCII path before reading (Windows non-ASCII workaround)
            with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
                tmp_path = tmp.name
            shutil.copy(str(INDEX_FILE), tmp_path)
            self._index = faiss.read_index(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)
            with open(CHUNKS_FILE, "rb") as f:
                self._chunks = pickle.load(f)
            print(f"[RAG-Local] Ready — {len(self._chunks):,} chunks, dim={self._index.d}")
            return

        docs = load_corpus()
        if not docs:
            print("[RAG-Local] No documents found.")
            return

        all_chunks = []
        for doc in docs:
            for chunk in _split_text(doc["text"]):
                all_chunks.append({"source": doc["source"], "text": chunk})
        print(f"[RAG-Local] {len(all_chunks):,} chunks. Embedding...")

        embeddings = self.model.encode(
            [c["text"] for c in all_chunks],
            show_progress_bar=True, batch_size=64, normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype="float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        # FAISS C++ backend cannot handle non-ASCII paths on Windows.
        # Write to a temp file in the system temp dir (ASCII path), then move.
        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
            tmp_path = tmp.name
        faiss.write_index(index, tmp_path)
        shutil.move(tmp_path, str(INDEX_FILE))

        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump(all_chunks, f)

        self._index, self._chunks = index, all_chunks
        print(f"[RAG-Local] Index cached → {INDEX_DIR}")

    def _try_load(self) -> bool:
        if self.is_ready():
            return True
        if INDEX_FILE.exists() and CHUNKS_FILE.exists():
            self.build_index()
        return self.is_ready()

    def _query(self, query: str, top_k: int) -> list[str]:
        """Run one FAISS query, return deduplicated passage strings."""
        q_emb = np.array(
            self.model.encode([query], normalize_embeddings=True), dtype="float32"
        )
        scores, indices = self._index.search(q_emb, top_k)
        passages = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                passages.append((score, self._chunks[idx]))
        return passages

    def retrieve_single(self, feature: str, top_k: int = TOP_K_LOCAL) -> list[str]:
        """
        Per-feature retrieval using three query types (mirrors original LLM-Lasso).
        Returns list of (score, chunk) tuples.
        """
        desc = FEATURE_DESCRIPTIONS.get(feature, feature.replace("_", " "))
        queries = [
            # (1) Category query — what is MDS-UPDRS finger tapping?
            f"MDS-UPDRS finger tapping Parkinson's disease severity assessment",
            # (2) Feature query — what is this feature?
            f"{desc} kinematic feature Parkinson disease bradykinesia",
            # (3) Interaction query — how does this feature relate to severity?
            f"{desc} relevance to {CATEGORY_DESCRIPTION}",
        ]
        all_passages = []
        for q in queries:
            all_passages.extend(self._query(q, top_k))
        return all_passages

    def retrieve(self, features: list[str], top_k: int = TOP_K_LOCAL) -> str:
        if not self._try_load():
            return ""

        if PER_FEATURE:
            all_passages, seen = [], set()
            for feat in features:
                for score, chunk in self.retrieve_single(feat, top_k):
                    text = _clean_text(chunk["text"])
                    if text not in seen:
                        seen.add(text)
                        all_passages.append((score, chunk["source"], text))
            # Sort by similarity, keep top passages overall
            all_passages.sort(key=lambda x: x[0], reverse=True)
            passages = [
                f'[{src} | sim={sc:.2f}]\n{txt}'
                for sc, src, txt in all_passages[:top_k * len(features)]
            ]
        else:
            # Batch mode (original behaviour)
            parts = [
                f"{f}: {FEATURE_DESCRIPTIONS.get(f, f.replace('_', ' '))}"
                for f in features
            ]
            query = (
                f"Retrieve information about "
                f"{', '.join(FEATURE_DESCRIPTIONS.get(f, f) for f in features)}'s "
                f"relevance to {CATEGORY_DESCRIPTION}."
            )
            raw = self._query(query, top_k)
            seen, passages = set(), []
            for score, chunk in raw:
                text = _clean_text(chunk["text"])
                if text not in seen:
                    seen.add(text)
                    passages.append(f'[{chunk["source"]} | sim={score:.2f}]\n{text}')

        if not passages:
            return ""
        return "--- Local Literature ---\n" + "\n\n".join(passages)


# ═══════════════════════════════════════════════════════════════════════════
# Source 2: PubMed live retrieval
# Adapted from LLM-Lasso: src/llm_lasso/llm_penalty/rag/pubMed_RAG_process.py
# ═══════════════════════════════════════════════════════════════════════════

def _pubmed_safe(tool, query: str) -> str:
    """Run PubMed query, return "" on failure or no-result."""
    try:
        result = tool.invoke(query)
        if not result or "No good PubMed result" in result:
            return ""
        return result.strip()
    except Exception as e:
        print(f"[RAG-PubMed] Query failed: {e}")
        return ""


def _pubmed_retrieve_single(tool, feature: str,
                            category: str = CATEGORY_DESCRIPTION) -> list[str]:
    """
    Three query types per feature, mirroring original LLM-Lasso
    pubMed_RAG_process.py (category / feature / interaction).
    Returns list of non-empty result strings.
    """
    desc = FEATURE_DESCRIPTIONS.get(feature, feature.replace("_", " "))
    queries = [
        # (1) Category query
        (f"finger tapping Parkinson disease MDS-UPDRS severity", "category"),
        # (2) Feature query
        (f"{desc} Parkinson disease bradykinesia", "feature"),
        # (3) Interaction query — most informative
        (f"{desc} relevance to {category}", "interaction"),
    ]
    results = []
    for q, label in queries:
        text = _pubmed_safe(tool, q)
        if text:
            results.append(f"[PubMed | {label} | {feature}]\n{text}")
        time.sleep(1)
    return results


def pubmed_retrieve_batch(features: list[str], category: str = CATEGORY_DESCRIPTION) -> str:
    """
    Retrieve PubMed abstracts for a batch of features.

    PER_FEATURE=True  (recommended): three query types per feature, deduplicated.
    PER_FEATURE=False: one batch-level query (fast, less precise).

    Mirrors the LLM-Lasso pubMed_RAG_process.py pipeline adapted for PD features.
    """
    if not HAS_PUBMED:
        print("[RAG-PubMed] langchain_community not installed — pip install langchain-community")
        return ""

    tool    = PubmedQueryRun()
    results = []
    seen    = set()

    def _add(text: str):
        if text and text not in seen:
            seen.add(text)
            results.append(text)

    if PER_FEATURE:
        for feat in features:
            for entry in _pubmed_retrieve_single(tool, feat, category):
                _add(entry)
    else:
        feat_terms = " ".join(
            FEATURE_DESCRIPTIONS.get(f, f.replace("_", " ")) for f in features[:5]
        )
        text = _pubmed_safe(tool, f"{category} {feat_terms}")
        _add(f"[PubMed | batch]\n{text}" if text else "")

    if not results:
        return ""
    return "--- PubMed Abstracts ---\n" + "\n\n".join(results)


# ═══════════════════════════════════════════════════════════════════════════
# Combined RAGRetriever
# ═══════════════════════════════════════════════════════════════════════════

class RAGRetriever:
    """
    Combines local FAISS index and PubMed live retrieval.
    Returns a formatted context string to prepend to the LLM scoring prompt.
    """

    def __init__(self):
        self._local = LocalIndexRetriever() if USE_LOCAL_INDEX else None

    def is_ready(self) -> bool:
        local_ok  = (self._local is not None and self._local.is_ready()) if USE_LOCAL_INDEX else True
        pubmed_ok = HAS_PUBMED if USE_PUBMED else True
        return local_ok or pubmed_ok

    def build_index(self, force: bool = False) -> None:
        """Build/load the local FAISS index (call once before scoring)."""
        if self._local is not None:
            self._local.build_index(force=force)
        if USE_PUBMED and not HAS_PUBMED:
            print("[RAG] PubMed source requested but langchain_community not installed.")
            print("      pip install langchain langchain-community")

    def retrieve(self, features: list[str]) -> str:
        """
        Retrieve domain context for a batch of features.
        Returns "" if no sources are available (safe fallback).
        """
        parts = []

        if USE_LOCAL_INDEX and self._local is not None:
            local_ctx = self._local.retrieve(features)
            if local_ctx:
                parts.append(local_ctx)

        if USE_PUBMED and HAS_PUBMED:
            pubmed_ctx = pubmed_retrieve_batch(features)
            if pubmed_ctx:
                parts.append(pubmed_ctx)

        if not parts:
            return ""

        return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════════════

_retriever: Optional[RAGRetriever] = None


def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
        _retriever.build_index()
    return _retriever


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"

    if cmd == "build":
        r = RAGRetriever()
        r.build_index(force="--force" in sys.argv)

    elif cmd == "query":
        feats = sys.argv[2:] or ["period_quartile_range", "num_peaks", "periodEntropy"]
        r = get_retriever()
        print(r.retrieve(feats))

    elif cmd == "check":
        print(f"knowledge_base : {KB_DIR}  exists={KB_DIR.exists()}")
        if KB_DIR.exists():
            print(f"  files: {[f.name for f in KB_DIR.iterdir()]}")
        print(f"FAISS index    : {INDEX_FILE.exists()}")
        if CHUNKS_FILE.exists():
            with open(CHUNKS_FILE, "rb") as f:
                chunks = pickle.load(f)
            print(f"  chunks: {len(chunks):,}")
        print(f"PubMed source  : USE_PUBMED={USE_PUBMED}, available={HAS_PUBMED}")
