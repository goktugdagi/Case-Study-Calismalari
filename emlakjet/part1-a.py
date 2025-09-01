import os
import sys
from typing import Optional
from datetime import datetime, timezone
import json
import pandas as pd
import networkx as nx


EVENTS_CSV = os.getenv("EVENTS_CSV", "user_event_data.csv")
ITEMS_CSV  = os.getenv("ITEMS_CSV",  "item_information.csv")

# Kenar ağırlıkları için katsayılar
# Veri dengesiz olduğu için ağırlıklandırdım.
WEIGHT_CLICK    = float(os.getenv("WEIGHT_CLICK", "1.0"))   # tıklama
WEIGHT_PURCHASE = float(os.getenv("WEIGHT_PURCHASE", "3.0")) # başvuru

# Zaman çürütmesi yarı-ömür, gün. 0 → çürütme yok
HALF_LIFE_DAYS = float(os.getenv("HALF_LIFE_DAYS", "30"))

# Aynı arama oturumu (ds_search_id) varsa küçük bir bonus
# Belirli bir aramadayken aynı bağlamda gördüğü ilanlar birbirinin alternatifi olabilir bu yüzden daha güçlü etkileşimolabilir.
# Kenar ağırlığını arttırır.

SESSION_BOOST = float(os.getenv("SESSION_BOOST", "1.1"))

# Çıktı dosyası
# Düğüm, kenar ve öznitelikleri tutar.

GRAPHML_PATH = os.getenv("GRAPHML_PATH", "bipartite_user_item.graphml")

# ÇIKTI / KAYIT: Özet metrikleri ve parametreleri kaydetmek için
OUTPUT_DIR       = os.getenv("OUTPUT_DIR", "outputs_part1a")
GRAPHML_PATH     = os.getenv("GRAPHML_PATH", os.path.join(OUTPUT_DIR, "bipartite_user_item.graphml"))
SUMMARY_BASENAME = os.getenv("SUMMARY_BASENAME", "graph_summary")
PARAMS_BASENAME  = os.getenv("PARAMS_BASENAME",  "run_params")

def ensure_dir(path: str) -> None:
    """Klasör yoksa oluşturur."""
    os.makedirs(path, exist_ok=True)

def load_data(events_path: str, items_path: str) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    #CSV dosyalarını yükle ve temel temizlik yap.
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Bulunamadı: {events_path}")

    events = pd.read_csv(events_path)
    events.columns = [c.strip().lower() for c in events.columns]

    # event_type normalize
    events["event_type"] = events["event_type"].astype(str).str.strip().str.lower()

    # timestamp → UTC Datetime
    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce", utc=True)

    # ds_search_id opsiyonel ise yoksa kolonu ekle
    if "ds_search_id" not in events.columns:
        events["ds_search_id"] = None

    # Sadece gerekli kolonları tut
    events = events[["event_type", "client_id", "item_id", "ds_search_id", "timestamp"]]

    # İlan bilgileri
    items = None
    if os.path.exists(items_path):
        items = pd.read_csv(items_path)
        items.columns = [c.strip().lower() for c in items.columns]
        if "item_id" not in items.columns:
            print("item_information.csv içinde 'item_id' yok. İlan eklenmeyecek.")
            items = None

    return events, items

# Amaç : Eski etkileşimlerin ağırlığını kademeli olarak azaltmak. Yani yakın zamanda olan click/pırchase daha önemli.
# Böylece trendleri yakalama imkanı artar.
# ts -> Etkilişimin gerçekleştiği zaman
# ref -> Kıyaslanan referans zaman
# half_life_days -> Yarı ömür süresi(gün cinsinden)
# delta_days -> gün cinsinden olayın ne kadar eski olduğu

def time_decay_factor(ts: pd.Timestamp, ref: pd.Timestamp, half_life_days: float) -> float:
    # exponential decay: factor = 0.5 ** (delta_gün / half_life_days)
    if half_life_days <= 0 or pd.isna(ts):
        return 1.0
    
    delta_days = (ref - ts).total_seconds() / (3600 * 24)
    if delta_days < 0:  # gelecekteki tarihler için decay uygulama
        return 1.0
    
    return 0.5 ** (delta_days / half_life_days)

# Kullanıcı eylemlerini (click/purchase) sayısal ağırlığa çevirir.
# Böylece Graph kenar ağırlığını hesaplarken her olayı farklı yorumlar.
# Çok tıklanan ama başvuru yapılmayan ilanların etkisini azaltır.

def base_event_weight(event_type: str) -> float:
    # Olay tipine göre taban ağırlık.
    if event_type == "purchase":
        return WEIGHT_PURCHASE
    
    return WEIGHT_CLICK  # default: click veya diğerleri

# Kullanıcı-İlan etkileşimi kayıtları events ve ilan bilgileri items.
# Kullanıcı düğümleri client_id ve item_id 
# Kenarlar -> weight, count, click_count, purchase_count, (varsa) last_ds_search_id, last_timestamp_utc öznitelikleri bulunur.


def build_bipartite_graph(events: pd.DataFrame, items: Optional[pd.DataFrame]) -> nx.Graph:
    # """
    # Kullanıcı–İlan bipartite grafını kurar.
    # - Düğümler: client_id (users), item_id (items)
    # - Kenar: kullanıcı–ilan etkileşimi
    # - Ağırlık: event_type taban ağırlığı ×  session boost × zaman çürütmesi
    # """
    B = nx.Graph()

    # Düğüm kümeleri: kullanıcılar ve ilanlar
    users = events["client_id"].astype(str).unique().tolist()
    items_ids = events["item_id"].astype(str).unique().tolist()

    B.add_nodes_from(users, bipartite="users")
    B.add_nodes_from(items_ids, bipartite="items")

    # Time decay için referans: en yeni timestamp
    ref_time = pd.to_datetime(events["timestamp"].max(), utc=True)

    # Her bir etkileşimi dolaş ve uygun kenar ağırlığını biriktir
    for row in events.itertuples(index=False):
        etype = str(row.event_type)                # click / purchase
        u = str(row.client_id)                     # kullanıcı düğümü
        i = str(row.item_id)                       # ilan düğümü
        dsid = getattr(row, "ds_search_id", None)  # oturum arama
        ts = getattr(row, "timestamp", pd.NaT)     # zaman

        # 1)  ağırlık (event_type)
        w = base_event_weight(etype)

        # 2) oturum bonusu : ds_search_id dolu ise
        if dsid not in (None, "", "nan"):
            w *= SESSION_BOOST

        # 3) zaman çürütmesi
        w *= time_decay_factor(ts, ref_time, HALF_LIFE_DAYS)

        # Kenarı oluşturma ve güncelle
        if B.has_edge(u, i):
            B[u][i]["weight"] += float(w)
            B[u][i]["count"] += 1
            if etype in ("click", "purchase"):
                B[u][i][f"{etype}_count"] = B[u][i].get(f"{etype}_count", 0) + 1
        else:
            attrs = {
                "weight": float(w),
                "count": 1,
                "click_count": 1 if etype == "click" else 0,
                "purchase_count": 1 if etype == "purchase" else 0,
            }
            if dsid not in (None, "", "nan"):
                attrs["last_ds_search_id"] = str(dsid)

            if isinstance(ts, pd.Timestamp) and not pd.isna(ts):
                attrs["last_timestamp_utc"] = ts.isoformat()

            B.add_edge(u, i, **attrs)

    # İlan düğümlerine başlık, açıklama uzunluğu vs ekle.
    if items is not None:
        items = items.drop_duplicates("item_id")
        for r in items.itertuples(index=False):
            item_id = str(r.item_id)
            if item_id in B and B.nodes[item_id].get("bipartite") == "items":
                if hasattr(r, "pozisyon_adi"):
                    B.nodes[item_id]["pozisyon_adi"] = str(getattr(r, "pozisyon_adi"))
                if hasattr(r, "item_id_aciklama"):
                    desc = str(getattr(r, "item_id_aciklama"))
                    B.nodes[item_id]["desc_len"] = len(desc)

    return B

def graph_summary(B: nx.Graph) -> dict:
    # Basit özet metrikler: düğüm/kenar sayıları, ortalama derece, ağırlık istatistikleri.

    users = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "users"]
    items = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "items"]

    num_users = len(users) # Toplam kullanıcı sayısı
    num_items = len(items) # Toplam ilan sayısı
    num_edges = B.number_of_edges() # Toplma kenar sayısı(Kullanıcı-İlan bağlantısı)

    # Ortalama derece
    avg_deg_users = sum(dict(B.degree(users)).values()) / num_users if num_users else 0.0
    avg_deg_items = sum(dict(B.degree(items)).values()) / num_items if num_items else 0.0

    # Kenar ağırlıklarının temel istatistikleri
    weights = [d.get("weight", 0.0) for _, _, d in B.edges(data=True)] # Tüm kenarların ağırlık listesi
    w_min = min(weights) if weights else 0.0 # kenar ağırlıklarının minimum değeri
    w_max = max(weights) if weights else 0.0 # kenar ağırlıklarının maksimum değeri
    w_mean = float(sum(weights) / len(weights)) if weights else 0.0 # kenar ağırlıklarının ortalama değeri, threshold için kullanılabilir.

    return {
        "kullanıcı sayısı": num_users,
        "ilan sayısı": num_items,
        "kenar sayısı": num_edges,
        "ortalama kullanıcı derecesi": round(avg_deg_users, 3),
        "ortalama ilan derecesi": round(avg_deg_items, 3),
        "edge_weight_min": round(w_min, 6),
        "edge_weight_max": round(w_max, 6),
        "edge_weight_mean": round(w_mean, 6),
    }

def save_run_outputs(summary: dict, params: dict, out_dir: str, graphml_path: str) -> None:
    # Özet metrikleri (CSV+JSON) ve parametreleri (JSON) kaydet.
    ensure_dir(out_dir)
    # UTC bazlı zaman bilgisi
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Summary → CSV + JSON
    summary_df = pd.DataFrame([summary])
    summary_csv = os.path.join(out_dir, f"{SUMMARY_BASENAME}_{run_id}.csv")
    summary_json = os.path.join(out_dir, f"{SUMMARY_BASENAME}_{run_id}.json")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Parametreler → JSON
    params_out = {
        "run_id": run_id,
        "graphml_path": os.path.abspath(graphml_path),
        "params": params,
    }
    params_json = os.path.join(out_dir, f"{PARAMS_BASENAME}_{run_id}.json")
    with open(params_json, "w", encoding="utf-8") as f:
        json.dump(params_out, f, ensure_ascii=False, indent=2)

    print(f"Özet metrikler kaydedildi:\n    - {summary_csv}\n    - {summary_json}")
    print(f"Çalıştırma parametreleri kaydedildi:\n    - {params_json}")



def main() -> int:
    print("CSV'ler yükleniyor...")
    events, items = load_data(EVENTS_CSV, ITEMS_CSV)

    print("Bipartite (Kullanıcı–İlan) grafı oluşturuluyor...")
    B = build_bipartite_graph(events, items)

    print("Özet metrikler:")
    summary = graph_summary(B)
    for k, v in summary.items():
        print(f"  - {k}: {v}")

    # Çıkış klasörü garanti altına alınsın
    ensure_dir(OUTPUT_DIR)

    # GraphML'e kaydet
    print(f" GraphML kaydediliyor: {GRAPHML_PATH}")
    try:
        nx.write_graphml(B, GRAPHML_PATH)
        print(">>> GraphML tamam.")
    except Exception as e:
        print(f"GraphML kaydedilemedi: {e}")

    # Özet metrikleri ve parametreleri kaydet
    run_params = {
        "EVENTS_CSV": os.path.abspath(EVENTS_CSV),
        "ITEMS_CSV": os.path.abspath(ITEMS_CSV),
        "WEIGHT_CLICK": WEIGHT_CLICK,
        "WEIGHT_PURCHASE": WEIGHT_PURCHASE,
        "HALF_LIFE_DAYS": HALF_LIFE_DAYS,
        "SESSION_BOOST": SESSION_BOOST,
        "OUTPUT_DIR": os.path.abspath(OUTPUT_DIR),
    }
    save_run_outputs(summary, run_params, OUTPUT_DIR, GRAPHML_PATH)

    return 0

if __name__ == "__main__":
    sys.exit(main())