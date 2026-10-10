"""Create frozen BGE-M3 embeddings for Intel SKU metadata.

Requires sentence-transformers and downloads BAAI/bge-m3 on first run.
Run from repository root: python scripts/data/build_intel_sku_embeddings.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / "configs/intel_sku/raid_aggregate.json").read_text())

def main():
    d = CFG["data"]
    meta = pd.read_csv(ROOT / d["sku_metadata_path"])
    sku_col = d["sku_column"]
    fields = d["retrieval_feature_columns"]
    missing = set([sku_col, *fields]) - set(meta.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    # Dataset-specific serialization: Intel has structured fields, not the
    # paper's category/brand/title/description schema.
    texts = [
        f"[PRODUCT_OFFERING] {row[fields[0]]} "
        f"[GENERATION] {row[fields[1]]} [ASP_GROUP] {row[fields[2]]}"
        for _, row in meta.iterrows()
    ]
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("BAAI/bge-m3")
    emb = encoder.encode(texts, batch_size=16, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    emb_path = ROOT / d["embedding_path"]
    index_path = ROOT / d["embedding_index_path"]
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    index_path.write_text(json.dumps({
        "skus": meta[sku_col].astype(str).tolist(),
        "model": "BAAI/bge-m3", "normalized": True,
        "serialization": "[PRODUCT_OFFERING] ... [GENERATION] ... [ASP_GROUP] ..."
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {emb.shape} embeddings to {emb_path}")
    print(f"Saved SKU order to {index_path}")

if __name__ == "__main__":
    main()
