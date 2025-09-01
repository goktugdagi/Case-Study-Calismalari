import os, glob, json, math, random, argparse
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix # Büyük ve çoğu sıfır olan matrisleri tutmak için.

OUT_DIR = "outputs_part2b"
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def latest(path_glob: str) -> str | None:
    # Belirtilen özellikle eşleşen dosyaları bul.
    files = glob.glob(path_glob)
    if not files: return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def find_existing_embeddings() -> tuple[str | None, str | None]:

    # Önce part2b çıkışları, sonra part2a_lightgcn klasörü
    # Kullanıcı embedding dosyasını bulma
    cand_user = latest(os.path.join(OUT_DIR, "user_emb_train_*.csv")) \
                or latest(os.path.join("outputs_part2a_lightgcn", "user_embeddings_lightgcn_*.csv"))
    
    # Ürün embedding dosyasını bul.
    cand_item = latest(os.path.join(OUT_DIR, "item_emb_train_*.csv")) \
                or latest(os.path.join("outputs_part2a_lightgcn", "item_embeddings_lightgcn_*.csv"))
    return cand_user, cand_item


def _stdname(s: str) -> str:
    # Sütun adını normalize eder
    return (s.replace("\ufeff","").strip().lower()
              .replace(" ", "").replace("-", "")
              .replace("\t","").replace("\r","").replace("\n",""))


def standardize_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Tüm sütun adlarını normalize edip orijinal adlarıyla eşleştirir.
    norm = {_stdname(c): c for c in df.columns}


    def pick(cands): 
        # Alternatif sütun adlarından uygun olanı seçer.
        for c in cands:
            if _stdname(c) in norm: return norm[_stdname(c)]
        return None
    
    # Kullanıcı, ürün ve zaman sütunlarını bulmaya çalışır.
    u = pick(["client_id","clientid","user_id","userid","user"])
    i = pick(["item_id","itemid","ad_id","ilan_id","item"])
    t = pick(["timestamp","event_time","datetime","date","time"])
    miss = []
    if u is None: 
        miss.append("client_id")
    if i is None: 
        miss.append("item_id")
    if t is None: 
        miss.append("timestamp")

    if miss: # Eksik sütun varsa hata verir.
        raise KeyError(f"Events CSV kolonları eksik: {miss} | mevcut: {list(df.columns)}")
    
    return df.rename(columns={u:"client_id", i:"item_id", t:"timestamp"})


def time_split(events_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = pd.read_csv(events_csv)
    df = standardize_event_columns(df)

    df["client_id"] = df["client_id"].astype(str)
    df["item_id"]   = df["item_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # Eksik verileri atar, kullanıcı ve zaman sırasına göre sıralar.
    df = df.dropna(subset=["client_id","item_id","timestamp"]).sort_values(["client_id","timestamp"])
    
    # Her kullanıcının son etkileşimini bulur.
    last  = df.groupby("client_id").tail(1)

    # Son etkileşimler hariç kalanlar eğitim verisi yapar.
    train = pd.concat([df, last]).drop_duplicates(keep=False)

    # Test verisi: her kullanıcının son etkileşimi.
    test  = last[["client_id","item_id"]].copy()

    return train, test

def load_embeddings(user_csv: str, item_csv: str):

    # Kullanıcı ve ürün embedding dosyalarını okur.
    users = pd.read_csv(user_csv); items = pd.read_csv(item_csv)

    # Embedding değerlerini numpy array'e çevirir.
    U = users.drop(columns=["client_id"]).to_numpy(np.float32)
    I = items.drop(columns=["item_id"]).to_numpy(np.float32)

    # Kullanıcı ve ürün id'lerini listeye çevirir.
    uid = users["client_id"].astype(str).tolist()
    iid = items["item_id"].astype(str).tolist()

    return uid, iid, U, I

def eval_user_item(user_csv: str, item_csv: str, events_csv: str,
                   K_list=[10,20,50]) -> dict[str, float]:
    
    # Embedding'leri ve id'leri yükleme.
    uid, iid, U, I = load_embeddings(user_csv, item_csv)

    # Eğitim ve test verisini ayarlama
    train, test = time_split(events_csv)

    # Sadece embedding'lerde olan kullanıcı ve ürünlerle çalış.
    uid_set, iid_set = set(uid), set(iid)
    train = train[train["client_id"].isin(uid_set) & train["item_id"].isin(iid_set)]
    test  = test [test ["client_id"].isin(uid_set) & test ["item_id"].isin(iid_set)]

    # Id'den index'e mappingler yapar.
    u2i = {u:i for i,u in enumerate(uid)}
    i2i = {it:i for i,it in enumerate(iid)}

    # Her kullanıcı için eğitimde gördüğü ürünleri tutar.
    seen: Dict[int, Set[int]] = {}
    for r in train.itertuples(index=False):
        seen.setdefault(u2i[r.client_id], set()).add(i2i[r.item_id])

    # Her kullanıcı için testteki hedef ürünü tutar.    
    targets: Dict[int,int] = {}
    for r in test.itertuples(index=False):
        targets[u2i[r.client_id]] = i2i[r.item_id]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    U_t = torch.tensor(U, device=device)
    I_t = torch.tensor(I, device=device).t()
    with torch.no_grad():
        scores = (U_t @ I_t).cpu().numpy()

    topK = max(K_list)
    recs = {k: [] for k in K_list}
    ndcgs = {k: [] for k in K_list}
    mrrs = {k: [] for k in K_list}
    for u_idx, tgt in targets.items():

        # Kullanıcının eğitimde gördüğü ürünlerin skorunu düşür.
        for it in seen.get(u_idx, []): scores[u_idx, it] = -1e9

        # En yüksek skorlu K ürünü bulur.
        idx = np.argpartition(-scores[u_idx], kth=topK)[:topK]
        idx = idx[np.argsort(-scores[u_idx, idx])]

        for K in K_list:
        # Her kullanıcı için K öneride başarı, ndcg ve mrr hesaplar.
            pred = idx[:K].tolist()
            hit = 1.0 if tgt in pred else 0.0
            recs[K].append(hit)
            if tgt in pred:
                rank = pred.index(tgt) + 1
                ndcgs[K].append(1.0 / math.log2(rank + 1))
                mrrs[K].append(1.0 / rank)
            else:
                ndcgs[K].append(0.0); mrrs[K].append(0.0)

    out = {f"Recall@{K}": float(np.mean(recs[K])) for K in K_list}
    out.update({f"NDCG@{K}": float(np.mean(ndcgs[K])) for K in K_list})
    out.update({f"MRR@{K}":  float(np.mean(mrrs[K]))  for K in K_list})
    return out

#  LightGCN
def interactions_weighted(events_csv: str,
                          w_click=1.0, 
                          w_purchase=3.0, 
                          half_life=30.0, 
                          session_boost=1.1) -> pd.DataFrame:
    
    df = pd.read_csv(events_csv)

    # Sütun adlarını standartlaştır (client_id, item_id, timestamp).
    df = standardize_event_columns(df)

    # Olay tipini normalize eder (click/purchase).
    df["event_type"] = df.get("event_type", "click").astype(str).str.lower()

    # Zaman sütununu datetime'a çevirir.
    df["timestamp"]  = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # En yeni zamanı referans olarak alır
    ref = pd.to_datetime(df["timestamp"].max(), utc=True)

    # Zamanla azalan ağırlık fonksiyonu
    def decay(ts):
        if half_life <= 0 or pd.isna(ts): return 1.0
        d = (ref - ts).total_seconds()/(3600*24)
        return 1.0 if d < 0 else 0.5 ** (d/half_life)
    
    def base_w(e): # Etkileişim tipine göre ağırlık.
        return w_purchase if e=="purchase" else w_click
    
    # Etkileişim tipine göre ağırlık.
    w = []
    for r in df.itertuples(index=False):
        w0 = base_w(getattr(r,"event_type","click"))

        # Oturum id varsa ağırlığı artır.
        if getattr(r,"ds_search_id",None) not in (None,"","nan"): 
            w0 *= session_boost

        # Zamanla azalan ağırlık uygulanır.
        w0 *= decay(getattr(r,"timestamp",pd.NaT))
        w.append(float(w0))

    df["weight"] = w

    # Aynı kullanıcı-ürün için ağırlıkları topla.
    g = df.groupby(["client_id","item_id"], as_index=False)["weight"].sum()

    # Id'leri string'e çevir.
    g["client_id"] = g["client_id"].astype(str); g["item_id"] = g["item_id"].astype(str)

    return g

def make_maps(inter: pd.DataFrame):

    # Tüm kullanıcıları al.
    users = inter["client_id"].astype(str).unique().tolist()

    # Tüm ürünleri al.
    items = inter["item_id"].astype(str).unique().tolist()
    
    # Kullanıcı ve ürünlerden index mappingleri döndür.
    return {u:i for i,u in enumerate(users)}, {it:i for i,it in enumerate(items)}

def build_adj_hat(inter: pd.DataFrame, u2i: dict, i2i: dict) -> torch.sparse.FloatTensor:

    # Kullanıcı ve ürün sayısı
    U, I = len(u2i), len(i2i)

    ru, ci, val = [], [], []

    # Her etkileşim için kullanıcı ve ürün indekslerini ve ağırlığı eklre.
    for r in inter.itertuples(index=False):
        ru.append(u2i[r.client_id]); ci.append(i2i[r.item_id]); val.append(float(r.weight))

    # Seyrek Kullanıcı-ürün etkileşim matrisi
    R = coo_matrix((val, (ru, ci)), shape=(U, I), dtype=np.float32).tocoo()

    # Bipartite graph için komşuluk matrix oluştur.
    row = np.concatenate([R.row, R.col + U]); col = np.concatenate([R.col + U, R.row])
    dat = np.concatenate([R.data, R.data]).astype(np.float32)

    # Toplam komşuluk matrix.
    A = coo_matrix((dat, (row, col)), shape=(U+I, U+I), dtype=np.float32).tocoo()

    # Her düğümün derecesini bulur ve sıfır olanlara 1 verir.
    deg = np.asarray(A.sum(axis=1)).ravel(); deg[deg==0] = 1.0

    d_inv = 1.0/np.sqrt(deg)

    # Veriyi normalize eder.
    data = A.data * d_inv[A.row] * d_inv[A.col]

    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    val = torch.tensor(data, dtype=torch.float32)

    return torch.sparse_coo_tensor(idx, val, (U+I, U+I))

class LightGCN(nn.Module):
    def __init__(self, U, I, dim, layers, adj):

        # Kulllanıcı ve ürün embeddingleri
        super().__init__()
        self.U, self.I = U, I
        self.user = nn.Embedding(U, dim); self.item = nn.Embedding(I, dim)

        # Embeddinleri normal dağılım ile başlatma
        nn.init.normal_(self.user.weight, std=0.01); nn.init.normal_(self.item.weight, std=0.01)
        self.layers = layers; self.adj = adj.coalesce()


    def propagate(self):
        # Kullanıcı ve ürün embeddinglerini birleştirme.
        x0 = torch.cat([self.user.weight, self.item.weight], dim=0)
        outs = [x0]; x = x0

        # Katmanler botyunca embeddingleri yayar.
        for _ in range(self.layers):
            x = torch.sparse.mm(self.adj, x); outs.append(x)

        x_final = torch.mean(torch.stack(outs, dim=0), dim=0) # KAtman embeddinglerinin ortalamasını alır.

        return x_final[:self.U], x_final[self.U:]
    
    def forward(self): 
        return self.propagate()

def bpr_loss(u, ip, ineg):
    # Pozitif ve negatif skorları hesaplama
    pos = torch.sum(u*ip, dim=1) 
    neg = torch.sum(u*ineg, dim=1)

    # Pozitif ve negatif skorlar arasındaki farkı log-loss ile cezalandırır.
    return -torch.mean(torch.log(torch.sigmoid(pos - neg) + 1e-10))

def train_once(events_csv: str, dim=64, layers=3, epochs=10, lr=0.01, batch_users=1024, neg_per_pos=1):
    inter = interactions_weighted(events_csv)

    u2i, i2i = make_maps(inter); U, I = len(u2i), len(i2i) # Kullanıcı ve ürün mappinglerini oluşturma.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # CUDA kontrolü
    model = LightGCN(U, I, dim, layers, build_adj_hat(inter, u2i, i2i).to(device)).to(device)# Modeli ve komşuluk matrix'i oluştur.
    opt = torch.optim.Adam(model.parameters(), lr=lr) # optimizer ayarlama.

    # user-> {item:weight}
    # Her kullanıcı için pozitif etkileşimde bulunduğu ürünleri ve ağırlıklarını tut.
    up: Dict[int, Dict[int,float]] = {}
    for r in inter.itertuples(index=False):
        u = u2i[r.client_id]; it = i2i[r.item_id]; w = float(r.weight)
        up.setdefault(u, {}); up[u][it] = up[u].get(it, 0.0) + w

    # Batch sayısı ayarlama
    steps = max(1, len(up)//batch_users)
    for ep in range(1, epochs+1):
        model.train(); total=0.0

        for _ in range(steps):

            
            users, poss, negs = [], [], []
            choose = random.sample(list(up.keys()), k=min(batch_users, len(up)))
            
            # Pozitif örnekleri ağırlığa göre seçme.
            for u in choose:
                items = np.array(list(up[u].keys()), dtype=np.int64)
                w = np.array([up[u][i] for i in items], dtype=np.float64)
                p = w / w.sum()
                ip = np.random.choice(items, p=p)

                # Negatif örnekleri rastgele seçme.
                for _ in range(neg_per_pos):
                    while True:
                        j = np.random.randint(0, I)
                        if j not in up[u]: 
                            users.append(u) 
                            poss.append(ip) 
                            negs.append(j) 
                            break

            if not users: 
                continue

            u = torch.tensor(users, dtype=torch.long, device=device)
            ip = torch.tensor(poss,  dtype=torch.long, device=device)
            ineg = torch.tensor(negs,  dtype=torch.long, device=device)
            Ue, Ie = model() 

            # loss'u hesapla.
            loss = bpr_loss(Ue[u], Ie[ip], Ie[ineg])
            opt.zero_grad(set_to_none=True) 
            loss.backward() 
            opt.step()
            total += float(loss.item())

        # Ekrana yazdır
        print(f"[train] epoch {ep} loss {total/max(1,steps):.4f}")

    model.eval()
    with torch.no_grad(): 
        Ue, Ie = model() 
        Ue = Ue.cpu().numpy() 
        Ie = Ie.cpu().numpy()

    inv_u = {v:k for k,v in u2i.items()} 
    inv_i = {v:k for k,v in i2i.items()}

    u_df = pd.DataFrame(Ue); u_df.insert(0, "client_id", [inv_u[i] for i in range(U)])
    i_df = pd.DataFrame(Ie); i_df.insert(0, "item_id",   [inv_i[i] for i in range(I)])
    rid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    u_csv = os.path.join(OUT_DIR, f"user_emb_train_{rid}.csv")
    i_csv = os.path.join(OUT_DIR, f"item_emb_train_{rid}.csv")

    u_df.to_csv(u_csv, index=False, encoding="utf-8")
    i_df.to_csv(i_csv, index=False, encoding="utf-8")

    return u_csv, i_csv

# -------------------- grid-search  --------------------
def grid_search(events_csv: str, 
                K=10, 
                dims=(32,64,128), 
                layers=(2,3), 
                negs=(1,2), 
                epochs=8,
                lr=0.01, 
                batch_users=1024) -> dict:
    # En iyi sonucu ve skorunu tutacak değişkenler. Tüm sonuçlar all_res listesinde birikecek.
    best = None 
    best_score = -1.0 
    all_res = []

    for d in dims: # Embedding boyutları için döngü 
        for L in layers: # Katman sayısı için döngü
            for n in negs:# Negatif örnek sayısı için döngü
                print(f"\n[GRID] dim={d}, L={L}, neg={n}, epochs={epochs}")

                u_csv, i_csv = train_once(events_csv, 
                                          dim=d, 
                                          layers=L, 
                                          epochs=epochs, 
                                          lr=lr, 
                                          batch_users=batch_users, 
                                          neg_per_pos=n) # Bu parametrelerle modeli eğit, embedding dosyalarını al.
                
                res = eval_user_item(u_csv, i_csv, events_csv, K_list=[K]) # Eğitim sonrası embedding'lerle değerlendirme metriklerini hesapla.
                score = res.get(f"Recall@{K}",0.0) + res.get(f"NDCG@{K}",0.0) # Recall@K ve NDCG@K toplamını skor olarak kullan.
                all_res.append({"dim":d,"layers":L,"neg":n, **res}) # Sonuçları parametrelerle birlikte all_res listesine ekle.
 
                if score > best_score: # Eğer skor önceki en iyiden yüksekse, en iyi sonucu güncelle.
                    best_score = score 
                    best = {"dim":d,"layers":L,"neg":n,"metrics":res} 

    rid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") # Zaman damgası oluştur

    with open(os.path.join(OUT_DIR, f"grid_summary_{rid}.json"), "w", encoding="utf-8") as f:
        json.dump({"K":K, "best":best, "all":all_res}, f, ensure_ascii=False, indent=2)

    print("\n=== GRID RESULTS ===")

    for r in all_res: # Tüm grid-search sonuçlarını ve en iyi konfigürasyonu JSON dosyasına kaydet.
        print(f"dim={r['dim']} L={r['layers']} neg={r['neg']} | " +
              " ".join([f"{k}:{v:.4f}" for k,v in r.items() if k.startswith('Recall') or k.startswith('NDCG')]))
        
    print("\n=== BEST CONFIG ===")
    print(best)

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events_csv", required=True)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--neg_per_pos", type=int, default=1)
    ap.add_argument("--batch_users", type=int, default=1024)
    ap.add_argument("--grid", action="store_true", help="Sonunda küçük bir grid-search de yap")
    args = ap.parse_args()

    # 1) varsa mevcut embedding'lerle EVAL, yoksa TRAIN->EVAL
    u_csv, i_csv = find_existing_embeddings()
    if u_csv and i_csv:
        print(f"[INFO] Mevcut embedding bulundu:\n  user={u_csv}\n  item={i_csv}\n>>> EVAL...")
    else:
        print("[INFO] Mevcut embedding yok. TRAIN -> EVAL...")
        u_csv, i_csv = train_once(args.events_csv, dim=args.dim, layers=args.layers,
                                  epochs=args.epochs, lr=0.01,
                                  batch_users=args.batch_users, neg_per_pos=args.neg_per_pos)
    K_list = sorted(set([args.K, max(5, min(20, 2*args.K)), 50]))
    res = eval_user_item(u_csv, i_csv, args.events_csv, K_list=K_list)
    print("\n=== EVAL ===")
    for k,v in res.items(): print(f"{k}: {v:.4f}")
    rid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with open(os.path.join(OUT_DIR, f"pipeline_eval_{rid}.json"), "w", encoding="utf-8") as f:
        json.dump({"emb_user":u_csv,"emb_item":i_csv,"K_list":K_list,"metrics":res}, f, ensure_ascii=False, indent=2)

    # 2) İstenirse grid-search
    if args.grid:
        best = grid_search(args.events_csv, K=args.K, epochs=max(8, args.epochs//1),
                           dims=(32,64,128), layers=(2,3), negs=(1,2),
                           lr=0.01, batch_users=args.batch_users)
        with open(os.path.join(OUT_DIR, f"pipeline_best_{rid}.json"), "w", encoding="utf-8") as f:
            json.dump({"best":best}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    raise SystemExit(main())

# # 1) varsa mevcut embedding ile EVAL, yoksa TRAIN->EVAL
# python .\part2-b.py --events_csv ".\user_event_data.csv" --K 10

# part2-a dan embedding geldiyse direk bunu çalıştır.
# # 2) Aynı ama sonunda küçük grid-search de yap
# python .\part2-b.py --events_csv ".\user_event_data.csv" --K 10 --grid (bunu çalıştır)

# # 3) Mevcut embedding yoksa, TRAIN için varsayılanları override ederek
# python .\part2-b.py --events_csv ".\user_event_data.csv" --dim 64 --layers 3 --epochs 10 --neg_per_pos 1 --K 10