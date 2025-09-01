# lightgcn
import os, json, math, random, sys
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import trange
from scipy.sparse import coo_matrix # Seyrek matris işlemleri için

# ========================
# 0) Ayarlar
# ========================
EVENTS_CSV   = os.getenv("EVENTS_CSV", "user_event_data.csv")
ITEMS_CSV    = os.getenv("ITEMS_CSV",  "item_information.csv")
OUTPUT_DIR   = os.getenv("OUTPUT_DIR", "outputs_part2a_lightgcn")

# Ağırlık Parametreleri
WEIGHT_CLICK    = float(os.getenv("WEIGHT_CLICK", "1.0"))
WEIGHT_PURCHASE = float(os.getenv("WEIGHT_PURCHASE", "3.0"))
HALF_LIFE_DAYS  = float(os.getenv("HALF_LIFE_DAYS", "30"))   # 0 -> decay yok
SESSION_BOOST   = float(os.getenv("SESSION_BOOST", "1.1"))

# LightGCN hiperparametreleri
EMBED_DIM   = int(os.getenv("EMBED_DIM", "64"))
N_LAYERS    = int(os.getenv("N_LAYERS", "3"))     # Propagation katmanı
EPOCHS      = int(os.getenv("EPOCHS", "20"))
BATCH_USERS = int(os.getenv("BATCH_USERS", "1024"))
# LightGCN ve BPR loss gibi öneri algoritmalarında, modelin kullanıcıya gerçekten ilgilenmediği ürünleri ayırt edebilmesi için negatif örnekler gereklidir.
# NEG_PER_POS -> her pozitif örnek için kaç tane negatif örnekle karşılaştırma yapılacağını belirler ve öneri modelinin daha iyi ayrım yapmasını sağlar.
# Bu negatif örnekler, modelin "kullanıcı pozitif ürünü negatif ürüne tercih etsin" diye öğrenmesini sağlar.
NEG_PER_POS = int(os.getenv("NEG_PER_POS", "1")) # 
LR          = float(os.getenv("LR", "0.01"))
SEED        = int(os.getenv("SEED", "42"))

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def time_decay_factor(ts: pd.Timestamp, ref: pd.Timestamp, half_life_days: float) -> float:
    # Son etkileşimlere daha fazla önem verme.
    # ts -> olay zamanı
    # ref -> en güncel zaman
    # half_life_days -> yarı ömür gün
    # delta_days -> etkileşim ile ref arasındaki gün farkı

    if half_life_days <= 0 or pd.isna(ts):
        return 1.0
    delta_days = (ref - ts).total_seconds() / (3600*24)
    if delta_days < 0:
        return 1.0
    return 0.5 ** (delta_days / half_life_days)

def base_event_weight(etype: str) -> float:
    # Olay tipine göre ağırlık (click<purchase).
    et = str(etype).strip().lower()
    return WEIGHT_PURCHASE if et == "purchase" else WEIGHT_CLICK

def load_interactions(events_path: str) -> pd.DataFrame:
    # Etkileşimleri yükler, ağırlıklarını hesaplar ve aynı kullanıcı-ürün için ağırlıkları toplar.
    if not os.path.exists(events_path):
        raise FileNotFoundError(events_path)
    df = pd.read_csv(events_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
    if "ds_search_id" not in df.columns: df["ds_search_id"] = None
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # ref time (en yeni)
    ref = pd.to_datetime(df["timestamp"].max(), utc=True)

    # ağırlık: base * session_boost * time_decay
    w = []
    for r in df.itertuples(index=False):
        w0 = base_event_weight(r.event_type)
        if getattr(r, "ds_search_id", None) not in (None, "", "nan"):
            w0 *= SESSION_BOOST
        w0 *= time_decay_factor(getattr(r, "timestamp", pd.NaT), ref, HALF_LIFE_DAYS)
        w.append(float(w0))
    df["weight"] = w

    # aynı (user,item) için biriktir
    grp = df.groupby(["client_id", "item_id"], as_index=False)["weight"].sum()
    return grp  # columns: client_id, item_id, weight


def make_id_mappings(interactions: pd.DataFrame) -> Tuple[Dict[str,int], Dict[str,int]]:
    # client_id ve item_id için ardışık indeksler üretir.
    # Kullanıcı ve ürünleri indekslere map eder.

    users = interactions["client_id"].astype(str).unique().tolist()
    items = interactions["item_id"].astype(str).unique().tolist()
    user2idx = {u:i for i,u in enumerate(users)}
    item2idx = {it:i for i,it in enumerate(items)}
    return user2idx, item2idx

def build_weighted_adj(interactions: pd.DataFrame,
                       user2idx: Dict[str,int],
                       item2idx: Dict[str,int]) -> Tuple[coo_matrix, np.ndarray, np.ndarray, np.ndarray]:
    
    # Kullanıcı-ürün etkileşimlerinden ağırlıklı komşuluk matrix oluşturur.
    # Komşuluk matrisi, bir grafikteki düğümler arasındaki bağlantıları (kenarları) gösteren bir matristir.

    U = len(user2idx); I = len(item2idx)
    rows_u, cols_i, vals = [], [], []
    for r in interactions.itertuples(index=False):
        u = user2idx[str(r.client_id)]
        i = item2idx[str(r.item_id)]
        rows_u.append(u); cols_i.append(i); vals.append(float(r.weight))

    R = coo_matrix((vals, (rows_u, cols_i)), shape=(U, I), dtype=np.float32)

    # A = [[0, R],
    #      [R^T, 0]]  → tek adımda coo üret
    row = np.concatenate([R.row,         R.col + U])
    col = np.concatenate([R.col + U,     R.row     ])
    dat = np.concatenate([R.data,        R.data    ]).astype(np.float32)

    A = coo_matrix((dat, (row, col)), shape=(U+I, U+I), dtype=np.float32)
    return A, R.row, R.col, np.array(vals, dtype=np.float32)

def normalize_adj(A: coo_matrix) -> torch.sparse.FloatTensor:
    # Komşuluk matrix'i normalize eder ve PyTorch sparse tensor'a çevirir.
    
    A = A.tocoo(copy=False)

    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0.0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)

    row = A.row
    col = A.col
    data = A.data * d_inv_sqrt[row] * d_inv_sqrt[col]

    i = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    v = torch.tensor(data, dtype=torch.float32)
    n = A.shape[0]
    return torch.sparse_coo_tensor(i, v, (n, n))


# LightGCN modeli
class LightGCN(nn.Module):
    # Kullanıcı ve ürün embedding'lerini başlatır.
    # propagate: Katmanlar boyunca embedding'leri yayar ve ortalamasını döndürür.
    
    def __init__(self, num_users: int, num_items: int, embed_dim: int, n_layers: int, adj_hat: torch.sparse.FloatTensor):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.adj_hat = adj_hat.coalesce()

        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Katmanlar boyunca gömme yayılımı; katman ortalaması döner.

        x0 = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)  # [U+I, D]
        outs = [x0]
        x = x0
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.adj_hat, x)
            outs.append(x)
        x_final = torch.mean(torch.stack(outs, dim=0), dim=0)  # [U+I, D]
        users_final = x_final[:self.num_users]
        items_final = x_final[self.num_users:]
        return users_final, items_final

    def forward(self):
        return self.propagate()

def bpr_loss(u_e, i_pos_e, i_neg_e):
    # Bayesian Personalized Ranking loss
    # BPR loss fonksiyonu: Pozitif ve negatif örnekler arasındaki ayrımı maksimize eder.

    # L = -log(σ( (u·i_pos) - (u·i_neg) ))
    pos_scores = torch.sum(u_e * i_pos_e, dim=1)
    neg_scores = torch.sum(u_e * i_neg_e, dim=1)
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))



def build_user_pos_dict(interactions: pd.DataFrame,
                        user2idx: Dict[str,int],
                        item2idx: Dict[str,int]) -> Dict[int, Dict[int, float]]:
    # Her kullanıcı için pozitif etkileşimde bulunduğu ürünleri ve ağırlıklarını tutar.
    # user_idx -> {item_idx: weight}

    up = {}
    for r in interactions.itertuples(index=False):
        u = user2idx[str(r.client_id)]
        i = item2idx[str(r.item_id)]
        w = float(r.weight)
        up.setdefault(u, {})
        up[u][i] = up[u].get(i, 0.0) + w
    return up

def sample_batch(user_pos: Dict[int, Dict[int, float]], num_items: int,
                 batch_users: int, neg_per_pos: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Batch için pozitif ve negatif örnekler seçer. Pozitifler ağırlığa göre, negatifler rastgele seçilir.

    users, pos_items, neg_items = [], [], []
    all_items = np.arange(num_items)
    chosen_users = random.sample(list(user_pos.keys()), k=min(batch_users, len(user_pos)))
    for u in chosen_users:
        pos_dict = user_pos[u]
        items = np.array(list(pos_dict.keys()), dtype=np.int64)
        weights = np.array([pos_dict[i] for i in items], dtype=np.float64)
        probs = weights / weights.sum()
        # 1 pozitif örnek seç
        i_pos = np.random.choice(items, p=probs)
        # NEG_PER_POS adet negatif
        for _ in range(neg_per_pos):
            while True:
                j = np.random.randint(0, num_items)
                if j not in pos_dict:
                    users.append(u); pos_items.append(i_pos); neg_items.append(j)
                    break
    return np.array(users), np.array(pos_items), np.array(neg_items)


def main():
    print("Yükleme & ağırlık hesapları...")
    inter = load_interactions(EVENTS_CSV)  # client_id, item_id, weight

    print("ID mapping...")
    user2idx, item2idx = make_id_mappings(inter)
    U, I = len(user2idx), len(item2idx)
    print(f"Users: {U}, Items: {I}, Interactions: {len(inter)}")

    print("A (weighted) ve Ĥ normalizasyonu...")
    A, rrow, rcol, rvals = build_weighted_adj(inter, user2idx, item2idx)
    adj_hat = normalize_adj(A)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    adj_hat = adj_hat.to(device)

    model = LightGCN(U, I, EMBED_DIM, N_LAYERS, adj_hat).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    user_pos = build_user_pos_dict(inter, user2idx, item2idx)

    print("Eğitim başlıyor...")
    for epoch in range(1, EPOCHS+1):
        model.train()
        total = 0.0
        steps = max(1, len(user_pos)//BATCH_USERS)
        for _ in trange(steps, desc=f"Epoch {epoch:02d}", leave=False):
            u, i_pos, i_neg = sample_batch(user_pos, I, BATCH_USERS, NEG_PER_POS)
            u = torch.from_numpy(u).long().to(device)
            i_pos = torch.from_numpy(i_pos).long().to(device)
            i_neg = torch.from_numpy(i_neg).long().to(device)

            u_e_all, i_e_all = model()  # propagate (katman ort.)
            loss = bpr_loss(u_e_all[u], i_e_all[i_pos], i_e_all[i_neg])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        print(f"[Epoch {epoch:02d}] bpr_loss: {total/steps:.4f}")

    print("Nihai embedding'ler çıkarılıyor...")
    model.eval()
    with torch.no_grad():
        Ue, Ie = model()  # [U,D], [I,D]
        Ue = Ue.detach().cpu().numpy()
        Ie = Ie.detach().cpu().numpy()

    # Kaydet
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Ters mapping
    inv_user = {v:k for k,v in user2idx.items()}
    inv_item = {v:k for k,v in item2idx.items()}

    user_df = pd.DataFrame(Ue)
    user_df.insert(0, "client_id", [inv_user[i] for i in range(U)])
    item_df = pd.DataFrame(Ie)
    item_df.insert(0, "item_id", [inv_item[i] for i in range(I)])

    user_csv = os.path.join(OUTPUT_DIR, f"user_embeddings_lightgcn_{run_id}.csv")
    item_csv = os.path.join(OUTPUT_DIR, f"item_embeddings_lightgcn_{run_id}.csv")
    user_df.to_csv(user_csv, index=False, encoding="utf-8")
    item_df.to_csv(item_csv, index=False, encoding="utf-8")

    ckpt = os.path.join(OUTPUT_DIR, f"lightgcn_{run_id}.pt")
    torch.save({"state_dict": model.state_dict(),
                "meta": {"U":U, "I":I, "EMBED_DIM":EMBED_DIM, "N_LAYERS":N_LAYERS}}, ckpt)

    # Parametre günlüğü
    with open(os.path.join(OUTPUT_DIR, f"part2a_lightgcn_params_{run_id}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "EVENTS_CSV": os.path.abspath(EVENTS_CSV),
            "ITEMS_CSV": os.path.abspath(ITEMS_CSV),
            "WEIGHT_CLICK": WEIGHT_CLICK,
            "WEIGHT_PURCHASE": WEIGHT_PURCHASE,
            "HALF_LIFE_DAYS": HALF_LIFE_DAYS,
            "SESSION_BOOST": SESSION_BOOST,
            "EMBED_DIM": EMBED_DIM,
            "N_LAYERS": N_LAYERS,
            "EPOCHS": EPOCHS,
            "BATCH_USERS": BATCH_USERS,
            "NEG_PER_POS": NEG_PER_POS,
            "LR": LR,
            "device": str(device)
        }, f, ensure_ascii=False, indent=2)

    print("\nÇıktılar:")
    print("  -", item_csv)
    print("  -", user_csv)
    print("  -", ckpt)
    print("Tamamlandı.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())