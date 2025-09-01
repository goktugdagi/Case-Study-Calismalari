# Bu dosya, metin ve davranış (graph) embeddinglerini birleştirerek (late fusion) bir item için en benzer diğer item'ları önerir.
# Kesişimdeki item'lar için embeddingleri yükler, alpha ile ağırlıklı skor hesaplar, top-k sonucu ekrana ve dosyaya yazar.
# Parametreler ve dosya yolları komut satırından kontrol edilebilir.

import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch

def log(msg: str):
    print(time.strftime("[%H:%M:%S]"), msg, flush=True)

def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def load_ids(csv_path: Path, id_col: str = "item_id") -> np.ndarray: # csv den item_id okunur
    df = pd.read_csv(csv_path)

    if id_col not in df.columns:
        raise ValueError(f"{csv_path} içinde '{id_col}' bulunamadı. Var olanlar: {list(df.columns)}")
    
    ids = df[id_col].values
    return ids

def align_by_intersection(ids_a: np.ndarray, ids_b: np.ndarray):
    # """
    # İki id dizisinin kesişimini ve her biri için index eşlemelerini döndür.
    # """
    set_a = {int(x): i for i, x in enumerate(ids_a)}
    idx_a, idx_b, inter = [], [], []

    for j, idb in enumerate(ids_b):
        idb = int(idb)

        if idb in set_a:
            idx_a.append(set_a[idb])
            idx_b.append(j)
            inter.append(idb)

    return np.array(idx_a), np.array(idx_b), np.array(inter, dtype=np.int64)

def to_torch(x: np.ndarray, device: str) -> torch.Tensor: # torch a çeviri
    return torch.from_numpy(x).to(device=device, dtype=torch.float32)

class FusionRecommender:
    # text ve Graph embeddinglerini normalize eder, tensöre çevirir ve id eşleştirmelerini oluşturur.
    def __init__(self,
                 text_emb: np.ndarray, graph_emb: np.ndarray,
                 item_ids: np.ndarray,
                 device: str = "cuda"):
        # """
        # text_emb: (N, d_t)
        # graph_emb: (N, d_g)
        # item_ids: (N,)
        # device: "cuda" or "cpu"
        # """
        assert text_emb.shape[0] == graph_emb.shape[0] == len(item_ids)
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

        # L2 normalize güvence altına al
        text_emb = l2_normalize_np(text_emb).astype(np.float32)
        graph_emb = l2_normalize_np(graph_emb).astype(np.float32)

        # Torch tensörleri
        self.T_text  = to_torch(text_emb,  self.device)   # (N, d_t)
        self.T_graph = to_torch(graph_emb, self.device)   # (N, d_g)
        self.ids = np.array(item_ids)
        self.id2idx = {int(i): k for k, i in enumerate(self.ids)}

        if self.device == "cuda":
            torch.set_float32_matmul_precision("high")
            log(f"CUDA device: {torch.cuda.get_device_name(0)}")

        log(f"T_text:  {tuple(self.T_text.shape)} | T_graph: {tuple(self.T_graph.shape)} | N={len(self.ids)}")

    @torch.inference_mode()
    def recommend(self, query_item_id: int, k: int = 10, alpha: float = 0.5, exclude_self: bool = True):
        # Sorgu item için top-k öneri döndürür
        # Skorları alpha ile birleştirir, 
        # Sonuçları DataFrame olarak verir.
        # Tek bir item_id için top-k öneri döndürür.
        # Skor: alpha*sim_graph + (1-alpha)*sim_text
        # Dönen: pandas.DataFrame [item_id, score, sim_text, sim_graph]

        if int(query_item_id) not in self.id2idx:
            raise KeyError(f"query_item_id {query_item_id} sistemde yok (kesişimde olmayabilir).")

        qi = self.id2idx[int(query_item_id)]

        # Sorgu vektörleri (1, d)
        q_text  = self.T_text[qi:qi+1]     # (1, d_t)
        q_graph = self.T_graph[qi:qi+1]    # (1, d_g)

        # Benzerlikler (N,)
        sim_text  = torch.matmul(self.T_text,  q_text.T).squeeze(1)   # cosine ≈ dot
        sim_graph = torch.matmul(self.T_graph, q_graph.T).squeeze(1)

        score = alpha * sim_graph + (1.0 - alpha) * sim_text

        if exclude_self:
            score[qi] = -1e9

        # Top-k
        k = min(k, len(self.ids))
        vals, idxs = torch.topk(score, k)
        idxs = idxs.detach().cpu().numpy()
        vals = vals.detach().cpu().numpy()

        # İsteğe bağlı: ayrı sim'leri de raporla
        sim_t_vals  = sim_text[idxs].detach().cpu().numpy()
        sim_g_vals  = sim_graph[idxs].detach().cpu().numpy()

        rec_ids = self.ids[idxs]
        df = pd.DataFrame({
            "item_id": rec_ids.astype(np.int64),
            "score": vals.astype(np.float32),
            "sim_text": sim_t_vals.astype(np.float32),
            "sim_graph": sim_g_vals.astype(np.float32),
        })
        return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_emb_npy", type=str, required=True)
    ap.add_argument("--text_id_csv",  type=str, required=True)
    ap.add_argument("--graph_emb_npy", type=str, required=True)
    ap.add_argument("--graph_id_csv",  type=str, required=True)
    ap.add_argument("--id_col", type=str, default="item_id")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--query_item_id", type=int, default=None)
    ap.add_argument("--save_csv", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    args = ap.parse_args()

    # Yollar
    text_emb_npy  = Path(args.text_emb_npy)
    text_id_csv   = Path(args.text_id_csv)
    graph_emb_npy = Path(args.graph_emb_npy)
    graph_id_csv  = Path(args.graph_id_csv)

    for p in [text_emb_npy, text_id_csv, graph_emb_npy, graph_id_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Bulunamadı: {p}")

    # Yükle
    log("Yükleniyor...")
    text_ids  = load_ids(text_id_csv, id_col=args.id_col)
    graph_ids = load_ids(graph_id_csv, id_col=args.id_col)
    text_emb  = np.load(text_emb_npy)   # (Nt, dt)
    graph_emb = np.load(graph_emb_npy)  # (Ng, dg)

    # Kesişim
    idx_t, idx_g, inter_ids = align_by_intersection(text_ids, graph_ids)
    if len(inter_ids) == 0:
        raise RuntimeError("Kesişim boş! id eşlemesini kontrol et.")
    
    log(f"Kesişimdeki item sayısı: {len(inter_ids)} / text={len(text_ids)} / graph={len(graph_ids)}")

    text_emb  = text_emb[idx_t]
    graph_emb = graph_emb[idx_g]

    # Recommender
    rec = FusionRecommender(text_emb, graph_emb, inter_ids, device=args.device)

    # Sorgu
    if args.query_item_id is None:
        qid = int(inter_ids[0])
        log(f"[Uyarı] --query_item_id verilmedi, örnek olarak {qid} kullanılıyor.")

    else:
        qid = int(args.query_item_id)

    log(f"Sorgu item_id: {qid} | alpha={args.alpha} | k={args.k}")
    out_df = rec.recommend(qid, k=args.k, alpha=args.alpha, exclude_self=True)

    # Ekran özeti
    print(out_df.head(args.k).to_string(index=False))

    # Kaydet
    if args.save_csv:
        out_path = Path(args.save_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        log(f"Kaydedildi: {out_path.resolve()}")

if __name__ == "__main__":
    main()


# python part4a_fusion_recommender.py `
#   --text_emb_npy  ".\emb_out_text\sbert_mMiniLM_embeddings.npy" `
#   --text_id_csv   ".\emb_out_text\sbert_mMiniLM_id_map.csv" `
#   --graph_emb_npy ".\emb_out_graph\item_embeddings.npy" `
#   --graph_id_csv  ".\emb_out_graph\item_id_map.csv" `
#   --alpha 0.5 --k 10 --query_item_id 123456 `
#   --save_csv ".\outputs_part4a\recos_123456.csv"
# alpha: davranış verisi (graph) güvenilir ise ↑ (örn. 0.7); metin daha açıklayıcı ise ↓ (örn. 0.3).
# CUDA kapatmak istersen --device cpu.

##############################3 TEK SATIRDA ÇALIŞTIRMA ÖRNEĞİ  ################################

# Fusion recommender (late-fusion)
# python .\part4a_fusion_recommender.py --text_emb_npy ".\emb_out_text\sbert_mMiniLM_embeddings.npy" --text_id_csv ".\emb_out_text\sbert_mMiniLM_id_map.csv" --graph_emb_npy ".\outputs_part2a_lightgcn\item_embeddings_lightgcn.npy" --graph_id_csv ".\outputs_part2a_lightgcn\item_id_map.csv" --alpha 0.5 --k 10 --save_csv ".\outputs_part4a\recos_sample.csv"



