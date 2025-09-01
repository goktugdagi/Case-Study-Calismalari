# Gerekli kütüphaeler
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer

def log(s: str):
    print(time.strftime("[%H:%M:%S]"), s, flush=True)

def load_input(input_parquet: str, input_csv: str, id_col: str, text_col: str) -> pd.DataFrame:

    if input_parquet and Path(input_parquet).exists():
        df = pd.read_parquet(input_parquet)
        src = input_parquet

    elif input_csv and Path(input_csv).exists():
        df = pd.read_csv(input_csv)
        src = input_csv

    else:
        raise FileNotFoundError("Girdi dosyası bulunamadı. input_parquet veya input_csv verin.")

    for c in [id_col, text_col]:
        if c not in df.columns:
            raise ValueError(f"Eksik kolon: {c}. Var olanlar: {list(df.columns)}")

    # Metni boş olanları ele ve duplike kayıtları çıkart
    df = df[df[text_col].fillna("").str.strip() != ""].copy()
    df = df[[id_col, text_col]].drop_duplicates(subset=[id_col]).reset_index(drop=True)

    log(f"Girdi yüklendi: {src} (rows={len(df)})")
    return df

def get_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def maybe_enable_fp16(model: SentenceTransformer) -> bool:
    # dtype float32 olarak ayarla
    try:
        first = model._first_module()  # Transformer modülü
        if hasattr(first, "auto_model"):
            first.auto_model = first.auto_model.to(dtype=torch.float16)
            return True
        
    except Exception:
        pass

    return False

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", type=str, default="cleaned_item_texts.parquet",
                    help="Temizlenmiş metin dosyası (parquet)")
    ap.add_argument("--input_csv", type=str, default="",
                    help="Alternatif: temizlenmiş metin dosyası (csv)")
    ap.add_argument("--id_col", type=str, default="item_id")
    ap.add_argument("--text_col", type=str, default="text_clean")

    ap.add_argument("--model", type=str,
                    default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    help="Türkçe destekli SBERT benzeri model")
    ap.add_argument("--batch_size", type=int, default=32, help="Encode batch size (GPU VRAM'a göre ayarlayın)")
    ap.add_argument("--normalize", action="store_true", help="L2 normalize (cosine=dot eşitliği için)")
    ap.add_argument("--fp16", action="store_true", help="Mümkünse yarı hassasiyet (float16) ile encode")
    ap.add_argument("--max_seq_len", type=int, default=256, help="Maks. token uzunluğu (SBERT default ~384)")

    ap.add_argument("--output_dir", type=str, default="emb_out_text",
                    help="Çıktı klasörü")
    ap.add_argument("--out_prefix", type=str, default="sbert_mMiniLM",
                    help="Çıktı dosya ön eki")

    args = ap.parse_args()

    # Çıktı klasörü
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aygıt
    device = get_device(prefer_cuda=True)
    log(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()} | device: {device}")
    
    if device == "cuda":
        log(f"CUDA device: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")

    # Veri
    df = load_input(args.input_parquet, args.input_csv, args.id_col, args.text_col)
    texts = df[args.text_col].astype(str).tolist()
    ids = df[args.id_col].tolist()

    # Model
    log(f"Model yükleniyor: {args.model}")
    model = SentenceTransformer(args.model, device=device)

    # Maks token uzunluğu
    try:
        model.max_seq_length = args.max_seq_len
    except Exception:
        pass

    # FP16 mümkünse
    if args.fp16 and device == "cuda":
        ok = maybe_enable_fp16(model)
        log("FP16: " + ("AKTİF" if ok else "desteklenmiyor, FP32 ile devam"))

    # Encode
    log(f"Encode başlıyor... (n={len(texts)}, batch={args.batch_size}, normalize={args.normalize})")

    try:

        embeddings = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False  
        )
    except TypeError:
        
        embeddings = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    if args.normalize:
        embeddings = l2_normalize(embeddings)

    dim = embeddings.shape[1]
    log(f"Emb shape: {embeddings.shape} (dim={dim})")

    # Kaydet
    id_map_path = out_dir / f"{args.out_prefix}_id_map.csv"
    npy_path = out_dir / f"{args.out_prefix}_embeddings.npy"
    parquet_path = out_dir / f"{args.out_prefix}_embeddings.parquet"

    pd.DataFrame({args.id_col: ids}).to_csv(id_map_path, index=False, encoding="utf-8")
    log(f"ID haritası kaydedildi: {id_map_path.resolve()}")

    np.save(npy_path, embeddings.astype(np.float32))
    log(f"Numpy kaydedildi: {npy_path.resolve()}")

    df_emb = pd.DataFrame(embeddings.astype(np.float32))
    df_emb.insert(0, args.id_col, ids)

    try:
        df_emb.to_parquet(parquet_path, index=False)
        log(f"Parquet kaydedildi: {parquet_path.resolve()}")

    except Exception as e:
        log(f"Parquet kaydı başarısız: {e}")

    # Özet
    log("İlk 3 id & norm kontrolü:")
    norms = np.linalg.norm(embeddings[:3], axis=1)

    for i in range(min(3, len(ids))):
        print(f"  {ids[i]}  ||  ||v||={norms[i]:.4f}")

    log("Tamamlandı.")

if __name__ == "__main__":
    main()



########################333  ÇALIŞTIRMA ÖRNEĞİ   #############################


# TEK SATIRDA ÇALIŞTIRMAK İÇİN
# python part3-a-embed.py --input_csv ".\cleaned_item_texts.csv" --normalize --batch_size 32 --fp16 --max_seq_len 256