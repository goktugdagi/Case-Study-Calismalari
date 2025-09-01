# Amaç : Kullacıların ilanlarla olan etkileşimlerini(click/purchase) bir Bipartite Graph olarak moedellemek.
# Bu Graph üzerinden çeşitli istatitikler ve özetler üretmek, en iyi user ve itemları belirlemek ve sonuçları dosyalara kaydetmek.


# Kütüphane importları
import os, json
from typing import Optional
import pandas as pd
import networkx as nx # Graph için
import matplotlib.pyplot as plt


EVENTS_CSV = os.getenv("EVENTS_CSV", "user_event_data.csv")
ITEMS_CSV  = os.getenv("ITEMS_CSV",  "item_information.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR",  "outputs_part1b")
TOP_N      = int(os.getenv("TOP_N", "20"))

WEIGHT_CLICK    = float(os.getenv("WEIGHT_CLICK", "1.0")) # Tıklama ağırlığı
WEIGHT_PURCHASE = float(os.getenv("WEIGHT_PURCHASE", "3.0")) # Başvurma ağırlığı
HALF_LIFE_DAYS  = float(os.getenv("HALF_LIFE_DAYS", "30")) # yarı ömür, gün cinsinden
SESSION_BOOST   = float(os.getenv("SESSION_BOOST", "1.1")) # birden fazla arama yaptıksa biraz daha ağırlık ver

os.makedirs(OUTPUT_DIR, exist_ok=True)


def time_decay_factor(ts: pd.Timestamp, ref: pd.Timestamp, half_life_days: float) -> float:
    # Son etkileşimlere daha fazla önem verme.
    # ts -> olay zamanı
    # ref -> en güncel zaman
    # half_life_days -> yarı ömür gün
    # delta_days -> etkileşim ile ref arasındaki gün farkı

    # exp decay: 0.5 ** (delta_gün / yarı_ömür_gün)
    if half_life_days <= 0 or pd.isna(ts): return 1.0

    delta_days = (ref - ts).total_seconds() / (3600 * 24)
    if delta_days < 0: 
        return 1.0
    
    return 0.5 ** (delta_days / half_life_days)

def base_event_weight(event_type: str) -> float:
    # Olay tipine göre ağırlık (click<purchase).
    return WEIGHT_PURCHASE if str(event_type).lower().strip() == "purchase" else WEIGHT_CLICK

def load_data(events_path: str, items_path: str):
    # CSV'leri yükle, temel temizlik yap.
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Bulunamadı: {events_path}")
    events = pd.read_csv(events_path)
    events.columns = [c.strip().lower() for c in events.columns]

    # normalize
    events["event_type"] = events["event_type"].astype(str).str.strip().str.lower()
    events["timestamp"]  = pd.to_datetime(events["timestamp"], errors="coerce", utc=True)
    if "ds_search_id" not in events.columns:
        events["ds_search_id"] = None
    events = events[["event_type","client_id","item_id","ds_search_id","timestamp"]]

    items = None
    if os.path.exists(items_path):
        items = pd.read_csv(items_path)
        items.columns = [c.strip().lower() for c in items.columns]
        if "item_id" not in items.columns:
            items = None
    return events, items

def build_bipartite_graph(events: pd.DataFrame, items: Optional[pd.DataFrame]) -> nx.Graph:
    # Kullanıcı–İlan bipartite grafını kurar 
    # kenarlarda ağırlık ve sayaçları biriktir.
    # item(ilan) düğümlerine açıklama ve pozisyon ekler
    B = nx.Graph()
    users = events["client_id"].astype(str).unique().tolist()
    items_ids = events["item_id"].astype(str).unique().tolist()
    B.add_nodes_from(users, bipartite="users")
    B.add_nodes_from(items_ids, bipartite="items")

    ref_time = pd.to_datetime(events["timestamp"].max(), utc=True)

    for row in events.itertuples(index=False):
        etype = str(row.event_type)
        u = str(row.client_id)
        i = str(row.item_id)
        dsid = getattr(row, "ds_search_id", None)
        ts = getattr(row, "timestamp", pd.NaT)

        w = base_event_weight(etype)
        if dsid not in (None,"","nan"):
            w *= SESSION_BOOST
        w *= time_decay_factor(ts, ref_time, HALF_LIFE_DAYS)

        if B.has_edge(u,i):
            B[u][i]["weight"] += float(w)
            B[u][i]["count"]  += 1
            if etype in ("click","purchase"):
                B[u][i][f"{etype}_count"] = B[u][i].get(f"{etype}_count",0) + 1
        else:
            attrs = {
                "weight": float(w),
                "count": 1,
                "click_count": 1 if etype=="click" else 0,
                "purchase_count": 1 if etype=="purchase" else 0,
            }
            if dsid not in (None,"","nan"):
                attrs["last_ds_search_id"] = str(dsid)
            if isinstance(ts, pd.Timestamp) and not pd.isna(ts):
                attrs["last_timestamp_utc"] = ts.isoformat()
            B.add_edge(u,i,**attrs)

    # İlan düğümlerine hafif metadata
    if items is not None:
        items = items.drop_duplicates("item_id")
        for r in items.itertuples(index=False):
            item_id = str(r.item_id)
            if item_id in B and B.nodes[item_id].get("bipartite") == "items":
                if hasattr(r,"pozisyon_adi"):
                    B.nodes[item_id]["pozisyon_adi"] = str(getattr(r,"pozisyon_adi"))
                if hasattr(r,"item_id_aciklama"):
                    desc = str(getattr(r,"item_id_aciklama"))
                    B.nodes[item_id]["desc_len"] = len(desc)

    return B

def pct(vs, p):
    # istatistiksel özetlerde, derece ve ağırlık gibi metriklerin dağılımını anlamak için
    # Verilen bir sayı dizisi yada arrayi yüzdelik değerini döndürür.
    # vs -> sayıların bulunduğu liste veya array.
    # p -> Yüzdelik değer
    if not vs: 
        return 0.0
    
    s = sorted(vs)
    k = int(round((p/100.0)*(len(s)-1)))
    return float(s[k])

def analyze(B: nx.Graph) -> tuple[dict, dict, dict]:
    # Özet metrikler, derece ve ağırlık persentilleri ve ek istatistikler.
    users = [n for n,d in B.nodes(data=True) if d.get("bipartite")=="users"]
    items = [n for n,d in B.nodes(data=True) if d.get("bipartite")=="items"]
    deg_users = [B.degree(u) for u in users]
    deg_items = [B.degree(i) for i in items]
    weights   = [dat.get("weight",0.0) for _,_,dat in B.edges(data=True)]

    summary = {
        "kullanıcı sayısı": len(users),
        "ilan sayısı": len(items),
        "kenar sayısı": B.number_of_edges(),
        "ortalama kullanıcı derecesi": round(sum(deg_users)/len(users),3) if users else 0.0,
        "ortalama ilan derecesi": round(sum(deg_items)/len(items),3) if items else 0.0,
        "bileşen sayısı": nx.number_connected_components(B),
        "weight_min": round(min(weights),6) if weights else 0.0,
        "weight_mean": round(sum(weights)/len(weights),6) if weights else 0.0,
        "weight_max": round(max(weights),6) if weights else 0.0,
    }

    extra = {
        "kullanıcı derece median": float(pd.Series(deg_users).median()) if deg_users else 0.0,
        "kullanıcı derece p90": pct(deg_users,90), # Kullanıcıların %90' ı en fazla kaç ilana tıklamış 
        "kullanıcı derece p99": pct(deg_users,99), # Kullanıcıların %99' ı en fazla kaç ilana tıklamış
        "ilan derece median": float(pd.Series(deg_items).median()) if deg_items else 0.0,
        # İlanın popülerlik dağılımını gösteriyor. 
        "ilan derece p90": pct(deg_items,90), # İlanların %90'ı kaç kullanıcıya ulaştığını gösteriyor.
        "ilan derece p99": pct(deg_items,99),
    }

    weight_stats = {
        "weight_median": float(pd.Series(weights).median()) if weights else 0.0,
        "weight_p90": float(pd.Series(weights).quantile(0.90)) if weights else 0.0,
        "weight_p99": float(pd.Series(weights).quantile(0.99)) if weights else 0.0,
        "weight_std": float(pd.Series(weights).std()) if weights else 0.0,
    }

    return summary, extra, weight_stats

def top_tables(B: nx.Graph, top_n: int = 20):
    # Top-N ilan ve kullanıcı tablolarını degree & strength(güçlü) döndürür.
    users = [n for n,d in B.nodes(data=True) if d.get("bipartite")=="users"]
    items = [n for n,d in B.nodes(data=True) if d.get("bipartite")=="items"]

    item_strength = []
    for i in items:
        s = sum(B[i][nbr]["weight"] for nbr in B.neighbors(i))
        item_strength.append((i, B.degree(i), s, B.nodes[i].get("pozisyon_adi","")))
    top_items_df = pd.DataFrame(item_strength, columns=["item_id","degree","strength","pozisyon_adi"]).sort_values(["strength","degree"], ascending=False).head(top_n).reset_index(drop=True)

    user_strength = []
    for u in users:
        s = sum(B[u][nbr]["weight"] for nbr in B.neighbors(u))
        user_strength.append((u, B.degree(u), s))

    top_users_df = pd.DataFrame(user_strength, columns=["client_id","degree","strength"]).sort_values(["strength","degree"], ascending=False).head(top_n).reset_index(drop=True)

    return top_items_df, top_users_df

def main():
    print("CSV'ler yükleniyor...")
    events, items = load_data(EVENTS_CSV, ITEMS_CSV)

    print("Bipartite graf kuruluyor...")
    B = build_bipartite_graph(events, items)

    print("Analiz...")
    summary, extra, weight_stats = analyze(B)

    print("Top-N tablolar...")
    top_items_df, top_users_df = top_tables(B, TOP_N)

    print("Kaydetme...")
    # JSON özet
    with open(os.path.join(OUTPUT_DIR, "part1b_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "extra": extra, "weight_stats": weight_stats}, f, ensure_ascii=False, indent=2)

    # CSV tablolar
    top_items_df.to_csv(os.path.join(OUTPUT_DIR, "top20_items.csv"), index=False, encoding="utf-8")
    top_users_df.to_csv(os.path.join(OUTPUT_DIR, "top20_users.csv"), index=False, encoding="utf-8")

    # TXT hızlı özet
    with open(os.path.join(OUTPUT_DIR, "part1b_summary.txt"), "w", encoding="utf-8") as f:

        f.write("=== Özet Metrikler ===\n")
        for k,v in summary.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== Ek İstatistikler ===\n")

        for k,v in extra.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== Ağırlık İstatistikleri ===\n")

        for k,v in weight_stats.items():
            f.write(f"{k}: {v}\n")

    # Konsola kısa özet ve ilk satırlar
    print("Metrikler:")
    for k,v in summary.items():
        print(f"  - {k}: {v}")
    print("Ek İstatistikler:")

    for k,v in extra.items():
        print(f"  - {k}: {v}")
    print("Ağırlık İstatistikleri:")

    for k,v in weight_stats.items():
        print(f"  - {k}: {v}")

    print("\n Top İlanlar (ilk 5 satır):")
    print(top_items_df.head(5).to_string(index=False))

    print("\nTop Kullanıcılar (ilk 5 satır):")
    print(top_users_df.head(5).to_string(index=False))

    print(f"\nTamamlandı. Çıktılar: {OUTPUT_DIR}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())