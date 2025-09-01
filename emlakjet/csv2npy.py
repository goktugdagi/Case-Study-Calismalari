# part4-a-fusion-recommender.py için csv dosyasını npy'ye çeviriyor. İlk bunu çalıştır daha sonra çeviri olduktan sonra part4a_fusion_Recommender.py çalıştır.

import argparse, re
from pathlib import Path
import pandas as pd, numpy as np

def auto_pick_csv(folder: Path) -> Path:
    cands = sorted(folder.glob("item_embeddings_lightgcn_*.csv"))
    if not cands:
        raise FileNotFoundError(f"{folder} içinde item_embeddings_lightgcn_*.csv bulunamadı.")
    return cands[-1]  # en yenisi

ap = argparse.ArgumentParser()
ap.add_argument("--csv", type=str, default="", help="LightGCN item embedding CSV yolu")
ap.add_argument("--folder", type=str, default="outputs_part2b_lightgcn", help="CSV dosyasını arayacağımız klasör")
ap.add_argument("--id_col", type=str, default="item_id")
ap.add_argument("--out_npy", type=str, default="", help="Çıkış .npy yolu (boşsa klasöre kaydeder)")
ap.add_argument("--out_id_map", type=str, default="", help="Çıkış id_map.csv (boşsa klasöre kaydeder)")
args = ap.parse_args()

folder = Path(args.folder)
csv_path = Path(args.csv) if args.csv else auto_pick_csv(folder)
df = pd.read_csv(csv_path)

if args.id_col not in df.columns:
    raise ValueError(f"{csv_path} içinde {args.id_col} yok. Var olanlar: {list(df.columns)}")

ids = df[args.id_col].astype("int64").to_numpy()
E = df.drop(columns=[args.id_col]).to_numpy(dtype="float32")

out_npy = Path(args.out_npy) if args.out_npy else folder / "item_embeddings_lightgcn.npy"
out_map = Path(args.out_id_map) if args.out_id_map else folder / "item_id_map.csv"

np.save(out_npy, E)
pd.DataFrame({args.id_col: ids}).to_csv(out_map, index=False, encoding="utf-8")

print("[OK] npy:", out_npy.resolve())
print("[OK] id_map:", out_map.resolve())
print("[SHAPE]", E.shape)

# Kodu çalıştırmak için
# python .\csv2npy.py --folder ".\outputs_part2a_lightgcn"

