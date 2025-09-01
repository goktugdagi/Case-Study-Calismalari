# Bu dosya, metin ve davranış (graph) embeddinglerini ChromaDB vektör veritabanına yükler.
# İki koleksiyon oluşturur, istenirse sadece ortak item'ları yükler, batch ile ekler ve sağlık kontrolü yapar.
# Parametreler ve dosya yolları komut satırından kontrol edilebilir.
# Çıktı: .vector_db altında kalıcı ChromaDB deposu.

# Gerekli kütüphaneler
import argparse, time, shutil
from pathlib import Path
import numpy as np, pandas as pd
import chromadb

# ChromaDB'nin kalıcı istemcisi (PersistentClient) import etme eğer eski sürüm ise fallback yapılır.
try:
    from chromadb import PersistentClient

except ImportError:
    # Eski sürümler için otomatik fallback
    PersistentClient = chromadb.PersistentClient  # type: ignore

def log(msg: str):
    print(time.strftime("[%H:%M:%S]"), msg, flush=True)

def load_emb_and_ids(npy_path: Path, id_csv: Path, id_col: str = "item_id"):
    # Embedding ve id dosyaları yüklenir.
    # Boyut uyuşmazlığı varsa kırma yapılır.

    if not npy_path.exists():
        raise FileNotFoundError(f"Bulunamadı: {npy_path}")
    
    if not id_csv.exists():
        raise FileNotFoundError(f"Bulunamadı: {id_csv}")

    E = np.load(npy_path)  # (N, d)
    df = pd.read_csv(id_csv)

    if id_col not in df.columns:
        raise ValueError(f"{id_csv} içinde '{id_col}' yok. Var olanlar: {list(df.columns)}")
    
    ids = df[id_col].astype("int64").to_numpy()

    if len(ids) != len(E):
        log(f"id sayısı ({len(ids)}) != embedding satırları ({len(E)}). Kırpma yapılacak.")
        n = min(len(ids), len(E))
        ids, E = ids[:n], E[:n]
    
    return ids, E.astype("float32", copy=False)

def intersect_by_ids(ids_a: np.ndarray, E_a: np.ndarray, ids_b: np.ndarray, E_b: np.ndarray):
    # Embedding ve id kaynağının kesişimini bulur ve ortak embeddingleri döndürür.

    map_a = {int(x): i for i, x in enumerate(ids_a)}
    idx_a, idx_b, inter = [], [], []

    for j, ib in enumerate(ids_b):
        ib = int(ib)

        if ib in map_a:
            idx_a.append(map_a[ib])
            idx_b.append(j)
            inter.append(ib)

    idx_a = np.asarray(idx_a, dtype=np.int64)
    idx_b = np.asarray(idx_b, dtype=np.int64)
    inter = np.asarray(inter, dtype=np.int64)

    return inter, E_a[idx_a], E_b[idx_b]

def batched(iterable, batch_size: int):
    # Veriyi batch'lere böler 
    for i in range(0, len(iterable), batch_size):
        yield slice(i, i + batch_size)

def ensure_collection(client, name: str, metric: str = "cosine"):
    # ChromaDB'de koleksiyon oluşturur veya varsa alır, benzerlik metriğini ayarlar.
    # metric: "cosine" / "l2" / "ip"

    meta = {"hnsw:space": metric}
    
    try:
        col = client.get_or_create_collection(name=name, metadata=meta)
    
    except TypeError:
        # Eski API'lerde metadata key'i değişmiş olabilir
        col = client.get_or_create_collection(name=name, metadata=meta)
    return col

def upsert_vectors(collection, ids: np.ndarray, E: np.ndarray, batch: int, source_tag: str = ""):
    # Embeddingleri ve id'leri ChromaDB koleksiyonuna batch halinde ekler (upsert/add).

    n = len(ids)

    for sl in batched(range(n), batch):
        s, e = sl.start, min(sl.stop, n)
        ids_str = [str(x) for x in ids[s:e].tolist()]
        vecs = E[s:e].tolist()
        metas = [{"source": source_tag}] * (e - s)

        
        if hasattr(collection, "upsert"):
            collection.upsert(ids=ids_str, embeddings=vecs, metadatas=metas)
        else:
            
            try:
                collection.add(ids=ids_str, embeddings=vecs, metadatas=metas)
            
            except Exception:
                collection.delete(ids=ids_str)
                collection.add(ids=ids_str, embeddings=vecs, metadatas=metas)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persist_dir", type=str, default="vector_db",
                    help="Chroma kalıcı depo klasörü")
    ap.add_argument("--reset", action="store_true",
                    help="persist_dir içeriğini sıfırla (DİKKAT: siler)")
    ap.add_argument("--metric", type=str, default="cosine", choices=["cosine","l2","ip"])
    ap.add_argument("--batch", type=int, default=4096)

    # text
    ap.add_argument("--text_emb_npy", type=str, required=True)
    ap.add_argument("--text_id_csv",  type=str, required=True)

    # Graph
    ap.add_argument("--graph_emb_npy", type=str, required=True)
    ap.add_argument("--graph_id_csv",  type=str, required=True)

    ap.add_argument("--only_intersection", action="store_true",
                    help="Sadece her iki kaynakta da bulunan item'ları yükle")

    ap.add_argument("--col_text", type=str, default="items_text")
    ap.add_argument("--col_graph", type=str, default="items_graph")

    args = ap.parse_args()

    persist_dir = Path(args.persist_dir)
    if args.reset and persist_dir.exists():
        shutil.rmtree(persist_dir)
        log(f"{persist_dir.resolve()} silindi.")

    persist_dir.mkdir(parents=True, exist_ok=True)

    # Yükleme
    text_ids, text_E = load_emb_and_ids(Path(args.text_emb_npy), Path(args.text_id_csv))
    graph_ids, graph_E = load_emb_and_ids(Path(args.graph_emb_npy), Path(args.graph_id_csv))

    log(f"Text:  ids={len(text_ids)} emb={text_E.shape}")
    log(f"Graph: ids={len(graph_ids)} emb={graph_E.shape}")

    if args.only_intersection:
        inter_ids, text_E, graph_E = intersect_by_ids(text_ids, text_E, graph_ids, graph_E)
        text_ids = graph_ids = inter_ids
        log(f"Kesişim: {len(inter_ids)} item")

    # Chroma client
    client = PersistentClient(path=str(persist_dir))
    col_text  = ensure_collection(client, args.col_text,  metric=args.metric)
    col_graph = ensure_collection(client, args.col_graph, metric=args.metric)

    # Upsert
    log("items_text başlıyor...")
    upsert_vectors(col_text, text_ids, text_E, batch=args.batch, source_tag="part3_text")
    log(f"items_text: {col_text.count()}")

    log("items_graph başlıyor...")
    upsert_vectors(col_graph, graph_ids, graph_E, batch=args.batch, source_tag="part2_graph")
    log(f"items_graph: {col_graph.count()}")

    # Kontrol: aynı item'ı her iki koleksiyonda var mı?
    some_id = str(text_ids[0]) if len(text_ids) else None
    if some_id:
        try:
            q1 = col_text.get(ids=[some_id])
            q2 = col_graph.get(ids=[some_id])
            log(f"örnek id={some_id} | text_hit={len(q1.get('ids', []))} | graph_hit={len(q2.get('ids', []))}")
        except Exception as e:
            log(f"örnek get başarısız: {e}")

    log("Tamamlandı.")

if __name__ == "__main__":
    main()



# Kodu çalıştırmak için

# python .\part4b_chroma_build.py --persist_dir ".\vector_db" --reset --metric cosine --batch 4096 --text_emb_npy ".\emb_out_text\sbert_mMiniLM_embeddings.npy" --text_id_csv ".\emb_out_text\sbert_mMiniLM_id_map.csv" --graph_emb_npy ".\outputs_part2a_lightgcn\item_embeddings_lightgcn.npy" --graph_id_csv ".\outputs_part2a_lightgcn\item_id_map.csv" --only_intersection

# --reset ilk çalıştırmada klasörü temizler (sonraki çalıştırmalarda gerekmez).

# --only_intersection iki kaynağın ortak olduğu ilanları yükler; fusion için garanti kolaylık.

# Çıkış: .\vector_db\ altında kalıcı Chroma deposu.



