# Gerekli kütüphaneler
import argparse
from pathlib import Path
import re
import regex as re2  # daha güçlü unicode desteği için
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# --- Basit TR stopword list  ---
TR_STOPWORDS = {
    "ve","veya","ile","ama","fakat","ancak","ise","de","da","ki","mi","mu","mü",
    "bu","şu","o","bir","her","çok","az","daha","en","gibi","için","üzere","ile",
    "diye","yada","ya","hem","ile","çünkü","eğer","ise","ya","ile","şayet","ki"
}

# --- Regex kalıpları ---
RE_URL = re2.compile(r"""(?i)\b((?:https?://|www\.)\S+)\b""", re2.UNICODE)
RE_EMAIL = re2.compile(r"""(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b""", re2.UNICODE)
RE_PHONE = re2.compile(r"""(?x)
    (?:\+?\d{1,3}[\s\-\.]?)?      # ülke kodu
    (?:\(?\d{3}\)?[\s\-\.]?)?     # alan kodu
    \d{3}[\s\-\.]?\d{2,4}         # numara
""", re2.UNICODE)
RE_MULTISPACE = re2.compile(r"\s+", re2.UNICODE)

# Türkçe karakterleri koruyarak basit noktalama temizliği
PUNCTS_TO_SPACE = r"""[\"'`^~_|<>\[\]{}()=+*#@:;.,!?–—-]"""

def strip_html(text: str) -> str: # html temizliği
    return BeautifulSoup(text, "lxml").get_text(separator=" ", strip=True)

def normalize_numbers(text: str) -> str:
    # 3+ basamaklı sayıları <NUM> ile maskele 
    # Tek basamaklıları korumak isteyebiliriz; burada basit strateji:
    text = re2.sub(r"\b\d{3,}\b", "<NUM>", text)
    # 1-2 basamaklı ama uzun zincirlerin parçası olanları da normalize etme:
    return text

def remove_or_mask_patterns(text: str) -> str: # URL, e-posta ve telefonları maskele.
    text = RE_URL.sub("<URL>", text)
    text = RE_EMAIL.sub("<EMAIL>", text)
    # Telefonları çok agresif maskelememek istersen, bu satırı kapatabilirsin:
    text = RE_PHONE.sub("<PHONE>", text)
    return text

def basic_punct_and_space(text: str) -> str: # Noktalama işaretlerini ve fazla boşlukları temizler.
    text = re2.sub(PUNCTS_TO_SPACE, " ", text)
    text = RE_MULTISPACE.sub(" ", text)
    return text.strip()

def apply_stopwords(text: str) -> str: # Türkçe stopword'leri metinden çıkarır.
    tokens = text.split()
    kept = [t for t in tokens if t not in TR_STOPWORDS]
    return " ".join(kept)

def clean_text(
    text: str,
    lower: bool = True,
    strip_html_flag: bool = True,
    mask_patterns: bool = True,
    normalize_num: bool = True,
    remove_punct: bool = True,
    use_stopwords: bool = False
) -> str:
    if not isinstance(text, str):
        return ""
    t = text

    # 1) HTML temizliği
    if strip_html_flag:
        t = strip_html(t)

    # 2) Unicode/boşluk normalize
    t = t.replace("\u00A0", " ")  # non-breaking space

    # 3) URL/e-posta/telefon maskesi
    if mask_patterns:
        t = remove_or_mask_patterns(t)

    # 4) Sayı normalizasyonu
    if normalize_num:
        t = normalize_numbers(t)

    # 5) Noktalama ve boşluk
    if remove_punct:
        t = basic_punct_and_space(t)

    # 6) lower
    if lower:
        t = t.lower()

    # 7) (Opsiyonel) Stopwords
    if use_stopwords:
        t = apply_stopwords(t)

    return t

def build_text(row, title_col: str, desc_col: str) -> str: # Başlık ve açıklamayı birleştirir, ikisi de varsa araya nokta koyar.
    title = str(row.get(title_col, "") or "")
    desc  = str(row.get(desc_col, "") or "")
    
    if title and desc:
        return f"{title} . {desc}"
    return title or desc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items_csv", type=str, required=True,
                    help="item_information.csv yolu (Windows için raw string önerilir)")
    ap.add_argument("--id_col", type=str, default="item_id", help="İlan ID kolonu")
    ap.add_argument("--title_col", type=str, default="pozisyon_adi", help="Başlık kolonu")
    ap.add_argument("--desc_col", type=str, default="item_id_aciklama", help="Açıklama kolonu")

    ap.add_argument("--output_csv", type=str, default="cleaned_item_texts.csv",
                    help="Temiz metinlerin kaydedileceği CSV")
    ap.add_argument("--output_parquet", type=str, default="cleaned_item_texts.parquet",
                    help="Aynı verinin Parquet çıktısı (embedding aşamasında hızlı okunur)")

    # Temizlik bayrakları
    ap.add_argument("--lower", action="store_true", help="Küçük harfe çevir (default: False)")
    ap.add_argument("--apply_stopwords", action="store_true", help="TR stopwords uygula (default: False)")
    ap.add_argument("--min_len", type=int, default=5, help="Min. temiz metin uzunluğu (kelime)")

    args = ap.parse_args()

    items_path = Path(args.items_csv)
    if not items_path.exists():
        raise FileNotFoundError(f"Bulunamadı: {items_path}")

    print(f"[INFO] Yükleniyor: {items_path}")
    df = pd.read_csv(items_path)

    # Girdi kolon kontrolü
    for c in [args.id_col, args.title_col, args.desc_col]:
        if c not in df.columns:
            raise ValueError(f"Kolon eksik: {c}. Var olanlar: {list(df.columns)}")

    # Ham metin oluşturma
    tqdm.pandas(desc="Metin birleştiriliyor")
    df["text_raw"] = df.progress_apply(
        lambda r: build_text(r, args.title_col, args.desc_col), axis=1
    )

    # Temizlik
    print("Temizleme başlıyor...")
    df["text_clean"] = df["text_raw"].progress_apply(
        lambda t: clean_text(
            t,
            lower=args.lower,
            strip_html_flag=True,
            mask_patterns=True,
            normalize_num=True,
            remove_punct=True,
            use_stopwords=args.apply_stopwords
        )
    )

    # Boş ve kısa metinleri filtreleme
    def word_count(s: str) -> int:
        return len(s.split()) if isinstance(s, str) else 0
    
    df["wc"] = df["text_clean"].apply(word_count)
    before = len(df)
    df = df[df["wc"] >= args.min_len].copy()
    after = len(df)
    print(f"[Kısa metin filtresi: {before-after} satır çıkarıldı (min_len={args.min_len})")

    # Sadece gerekli kolonlar
    out_cols = [args.id_col, "text_raw", "text_clean", "wc"]
    df_out = df[out_cols].reset_index(drop=True)

    # Kaydet
    csv_path = Path(args.output_csv)
    parquet_path = Path(args.output_parquet)

    df_out.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"CSV kaydedildi: {csv_path.resolve()} (rows={len(df_out)})")

    try:
        df_out.to_parquet(parquet_path, index=False)
        print(f"Parquet kaydedildi: {parquet_path.resolve()} (rows={len(df_out)})")

    except Exception as e:
        print(f"Parquet kaydedilemedi: {e}")

    # Hızlı önizleme
    print("\nİlk 5 satır:")

    with pd.option_context("display.max_colwidth", 120):
        print(df_out.head(5).to_string(index=False))

if __name__ == "__main__":
    main()


##################################  KOD ÇALIŞTIRMA ÖRNEKLERİ  ##################################
# python part3-a.py --items_csv "item_information.csv" --lower --min_len 5
##################################  KOD ÇALIŞTIRMA ÖRNEKLERİ  ##################################
# Kolon isimlerin farklıysa (ör. title, description) argümanlarla geçebilirsin:
# python clean_texts.py --items_csv ".\item_information.csv" --id_col item_id --title_col pozisyon_adi --desc_col item_id_aciklama --lower
