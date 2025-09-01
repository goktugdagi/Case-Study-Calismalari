# part4c_chroma_recommend.py


import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

# ChromaDB'nin kalıcı istemcisi (PersistentClient) import edilir, eski sürümler için fallback yapılır.
try:
    from chromadb import PersistentClient

except ImportError:
    import chromadb
    PersistentClient = chromadb.PersistentClient  # type: ignore

def log(m: str): print(time.strftime("[%H:%M:%S]"), m, flush=True)

def l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray: # Numpy array'ini L2 normuna göre normalize eder.
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 2 and x.shape[0] == 1:  # (1, d) -> (d,)
        x = x[0]
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)

def get_emb_by_id(col, item_id: int) -> np.ndarray:
    # ChromaDB koleksiyonundan bir item'ın embedding'ini alır ve normalize eder.
    # Koleksiyondan tek id'nin embedding'ini al; yoksa KeyError.

    r = col.get(ids=[str(item_id)], include=["embeddings"])
    embs = r.get("embeddings", None)
    # embs tipik olarak list; bazı sürümlerde np.array olabilir

    if embs is None:
        raise KeyError(f"Embedding yok: id={item_id}")
    
    # list ise uzunluk kontrolü
    if isinstance(embs, list) and len(embs) == 0:
        raise KeyError(f"Embedding yok: id={item_id}")
    
    # np.array ise size kontrolü
    if hasattr(embs, "size") and getattr(embs, "size", 0) == 0:
        raise KeyError(f"Embedding yok: id={item_id}")
    
    e = np.asarray(embs[0], dtype=np.float32)

    return l2(e)

def query_candidates(col, q_emb: np.ndarray, top_m: int = 200):
    # Koleksiyonda top_m adayını (ids + embeddings) getir, normalize et.
    qr = col.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_m,
        include=["embeddings"]   
    )

    ids = [int(x) for x in qr["ids"][0]]  # ids default geliyor
    embs = np.asarray(qr.get("embeddings", [[]])[0], dtype=np.float32)

    return ids, l2(embs) if len(embs) else np.empty((0, q_emb.shape[-1]), dtype=np.float32)

def collection_has_id(col, item_id: int) -> bool:
    r = col.get(ids=[str(item_id)], include=[])

    return len(r.get("ids", [])) > 0

def pick_any_intersection_id(col_t, col_g, sample_n: int = 1000) -> int:
    #İki koleksiyonda da mevcut geçerli bir id döndür.
    
    try:
        ids_t = col_t.peek(sample_n)["ids"]

    except Exception:
        ids_t = col_t.get(limit=sample_n).get("ids", [])

    try:
        ids_g = set(col_g.peek(sample_n)["ids"])

    except Exception:
        ids_g = set(col_g.get(limit=sample_n).get("ids", []))

    inter = [int(x) for x in ids_t if x in ids_g]

    if not inter:
        raise RuntimeError("Koleksiyonların kesişiminde id bulunamadı; yüklemeyi kontrol edin.")
    
    return inter[0]

def fuse_scores(q_t: np.ndarray, q_g: np.ndarray,
                cand_ids: np.ndarray,
                E_t: dict, E_g: dict,
                alpha: float):
    # Text ve graph embedding benzerliklerini alpha ile birleştirerek fusion skorunu hesaplar.

    sims_t, sims_g = [], []

    for i in cand_ids:
        et = E_t.get(i, None); eg = E_g.get(i, None)
        st = float(q_t @ et) if et is not None else -1.0
        sg = float(q_g @ eg) if eg is not None else -1.0
        sims_t.append(st); sims_g.append(sg)

    sims_t = np.asarray(sims_t, dtype=np.float32)
    sims_g = np.asarray(sims_g, dtype=np.float32)
    score = alpha * sims_g + (1.0 - alpha) * sims_t
    return score, sims_t, sims_g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persist_dir", type=str, default="vector_db")
    ap.add_argument("--col_text", type=str, default="items_text")
    ap.add_argument("--col_graph", type=str, default="items_graph")
    ap.add_argument("--query_item_id", type=int, required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--top_m", type=int, default=200)
    ap.add_argument("--save_csv", type=str, default="")
    args = ap.parse_args()

    client = PersistentClient(path=str(Path(args.persist_dir)))
    col_t = client.get_or_create_collection(name=args.col_text,  metadata={"hnsw:space":"cosine"})
    col_g = client.get_or_create_collection(name=args.col_graph, metadata={"hnsw:space":"cosine"})

    qid = int(args.query_item_id)
    # Geçerli mi? Değilse otomatik geçerli bir id seç
    # Sorgu id'si koleksiyonlarda yoksa, otomatik olarak ortak bir id seçilir.
    if not (collection_has_id(col_t, qid) and collection_has_id(col_g, qid)):
        alt_id = pick_any_intersection_id(col_t, col_g, sample_n=1000)
        log(f"id={qid} koleksiyonlardan birinde yok. Otomatik olarak id={alt_id} ile devam ediliyor.")
        qid = alt_id

    log(f"Sorgu id={qid} | k={args.k} | alpha={args.alpha} | top_m={args.top_m}")

    # 1) Sorgu embedding'leri
    q_t = get_emb_by_id(col_t, qid)
    q_g = get_emb_by_id(col_g, qid)

    # 2) Adaylar
    ids_t, cand_t = query_candidates(col_t, q_t, top_m=args.top_m)
    ids_g, cand_g = query_candidates(col_g, q_g, top_m=args.top_m)

    # 3) Birleşim + haritalar
    union_ids = sorted(set(ids_t) | set(ids_g))
    E_t = {int(i): e for i, e in zip(ids_t, cand_t)}
    E_g = {int(i): e for i, e in zip(ids_g, cand_g)}

    # 4) Fusion skorları
    score, st, sg = fuse_scores(q_t, q_g, np.array(union_ids, dtype=np.int64), E_t, E_g, alpha=args.alpha)

    # 5) Sıralama ve self'i çıkar
    order = np.argsort(-score)
    ranked_ids = np.array(union_ids, dtype=np.int64)[order]
    ranked_score = score[order]; ranked_st = st[order]; ranked_sg = sg[order]
    mask = ranked_ids != qid
    ranked_ids = ranked_ids[mask]; ranked_score = ranked_score[mask]
    ranked_st = ranked_st[mask];   ranked_sg = ranked_sg[mask]

    k = min(args.k, len(ranked_ids))
    out = pd.DataFrame({
        "item_id": ranked_ids[:k],
        "score": ranked_score[:k].round(6),
        "sim_text": ranked_st[:k].round(6),
        "sim_graph": ranked_sg[:k].round(6),
    })
    print(out.to_string(index=False))

    if args.save_csv:
        p = Path(args.save_csv); p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False, encoding="utf-8")
        log(f"Kaydedildi: {p.resolve()}")

if __name__ == "__main__":
    main()

# Kodu çalıştırmak için
# python .\part4c_chroma_recommend.py --persist_dir ".\vector_db" --query_item_id 320682 --k 10 --alpha 0.5 --save_csv ".\outputs_part4a\recos_320682_chroma.csv"













