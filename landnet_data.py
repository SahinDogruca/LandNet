"""
landnet_data.py — LARD tarzi (LARD_V2_XPlane vb.) veri kumeleri icin
PyTorch Dataset/DataLoader. landnet.py'deki LandNet modeliyle birlikte
kullanilmak uzere tasarlanmistir.

Beklenen tablo semasi (CSV ya da parquet):
    image, height, width, type, original_dataset, scenario, airport, runway,
    time_to_landing, weather, night, time, yaw, pitch, roll, slant_distance,
    along_track_distance, height_above_runway, lateral_path_angle,
    vertical_path_angle, watermark_height, runway_in_cone,
    x_TR,y_TR,x_TL,y_TL,x_BL,y_BL,x_BR,y_BR

'image' sutunu ya bir DOSYA YOLU (str) ya da HuggingFace `datasets` tarzi
{'bytes': b'...', 'path': ...} sozlugu olabilir -- ikisi de otomatik
algilanir.

============================== KRITIK NOTLAR ==============================
1) ACI KONVANSIYONU: landnet.py'deki lard_raw_angles_to_R() ve
   LARD_ANGLE_OFFSETS_DEG'e bakiniz. pitch_raw sutununun ~90 derece
   ofsetli oldugu HIPOTEZ ediliyor (tek ornek + fiziksel muhakemeyle).
   Bu dosyadaki sanity_check_angle_convention() fonksiyonunu (train.py
   icinde de cagirilir) EGITIME BASLAMADAN ONCE mutlaka calistirin.

2) RUNWAY ID: CSV'deki 'runway' sutunu sifir-dolgusuz olabilir (ornegin
   '1'), ama runways_db_V2_XPlane.json anahtarlari sifir-dolgulu ('01').
   landnet.get_runway_object_points() bunu normalize_runway_id() ile
   otomatik dener; yine de her ihtimale karsi bu dosyadaki
   `report_runway_db_coverage()` ile veri kumenizin ne kadarinin GERCEK
   DB'den (fallback degil) geometriyle eslesdigini kontrol edin.

3) GEOMETRIK AUGMENTASYON (rastgele crop/zoom) VARSAYILAN OLARAK KAPALIDIR
   (DataConfig.geometric_aug=False). Sub-0.1 derece hedefinde, K/kose
   noktalarinin YANLIS guncellenmesi SESSIZCE modeli bozar ve fark etmesi
   zordur. Once fotometrik-augmentasyon-sadece bir calisma ile saglam bir
   taban olusturmanizi, geometrik augmentasyonu SADECE bu taban dogrulandiktan
   sonra, kucuk bir alt-kumede gorsel dogrulama yaparak acmanizi oneririm.
=============================================================================
"""

from __future__ import annotations

import io
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from landnet import (
    lard_raw_angles_to_R,
    get_runway_object_points,
    load_runway_db,
    intrinsics_from_fov,
    build_intrinsics,
    LARD_ANGLE_OFFSETS_DEG,
)

CORNER_ORDER = ["TR", "TL", "BL", "BR"]  # LARD CSV sutun sirasi; landnet.py'deki nokta sirasiyla BIREBIR eslesir


# =============================================================================
# 0) Yapilandirma
# =============================================================================

@dataclass
class DataConfig:
    table_path: str                        # .parquet ya da .csv
    runway_db_path: Optional[str] = None    # runways_db_V2_XPlane.json yolu (None -> hep fallback genislik/uzunluk)
    images_root: Optional[str] = None       # 'image' sutunu goreli dosya yolu ise, bu kok dizine gore cozulur

    img_size: int = 640
    hfov_deg: float = 60.0                  # kullanicinin belirttigi sabit varsayim

    val_fraction: float = 0.12
    split_seed: int = 42

    # Basit [0,1] -> normalize. ImageNet istatistikleri (pretrained backbone
    # kullanilmasa bile) makul, iyi test edilmis bir varsayilan.
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    photometric_aug: bool = True            # guvenli: geometriye DOKUNMAZ
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.15
    color_jitter_hue: float = 0.02

    # bkz. dosya basindaki UYARI -- varsayilan KAPALI
    geometric_aug: bool = False
    geometric_aug_scale_range: Tuple[float, float] = (0.85, 1.0)  # 1.0=orijinal boyut, <1.0=zoom-in
    geometric_aug_max_retries: int = 8       # koseler kirpma disinda kalirsa yeniden dener, basarisizsa crop uygulanmaz

    angle_offsets: Optional[Dict[str, float]] = None   # None -> landnet.LARD_ANGLE_OFFSETS_DEG kullanilir

    watermark_crop: bool = True             # watermark_height>0 ise alt banti kirp (bkz. _load_image)

    # --- Hugging Face otomatik indirme ---
    # True ise ve table_path'teki dosya MEVCUT DEGILSE, LARD V2 XPlane veri
    # setini Hugging Face'ten otomatik indirir ve table_path'e parquet olarak
    # kaydeder. Boylece ayri bir indirme adimi/scripti gerekmez.
    auto_download_from_hf: bool = True
    hf_dataset_name: str = "DEEL-AI/LARD_V2"   # HuggingFace dataset repo
    hf_config_name: str = "xplane"               # dataset config (xplane subset)


# =============================================================================
# 1) HuggingFace otomatik indirme + tablo yukleme + scenario-gruplu split
# =============================================================================

def _download_lard_from_hf(split: str, output_path: str,
                            hf_dataset: str = "DEEL-AI/LARD_V2",
                            hf_config: str = "xplane") -> str:
    """LARD V2 XPlane veri setinin belirtilen split'ini ('train' ya da 'test')
    Hugging Face'ten indirir ve output_path'e parquet olarak kaydeder.

    Gereksinimler: pip install datasets[vision] pyarrow
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Hugging Face 'datasets' kutuphanesi bulunamadi. "
            "Lutfen kurun: pip install datasets[vision] pyarrow"
        )

    print(f"{'=' * 70}")
    print(f"LARD V2 XPlane - '{split}' split indiriliyor...")
    print(f"Kaynak: huggingface.co/datasets/{hf_dataset} ({hf_config} config)")
    print(f"Hedef:  {output_path}")
    print(f"{'=' * 70}")

    ds = load_dataset(hf_dataset, name=hf_config, split=split)
    print(f"Indirme tamamlandi: {len(ds)} ornek, sutunlar: {ds.column_names}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    ds.to_parquet(output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Kaydedildi: {output_path} ({size_mb:.1f} MB)")
    print(f"{'=' * 70}\n")
    return output_path


def _ensure_table_exists(cfg: "DataConfig") -> str:
    """table_path dosyasi mevcutsa dogrudan dondurur; mevcut degilse ve
    auto_download_from_hf=True ise HuggingFace'ten indirir.

    table_path'ten split otomatik cikarilir:
      - 'train' iceren yol -> split='train'
      - 'test' iceren yol  -> split='test'
      - hicbiri yoksa       -> split='train' (varsayilan)
    """
    path = cfg.table_path
    if os.path.exists(path):
        return path

    if not cfg.auto_download_from_hf:
        raise FileNotFoundError(
            f"Tablo dosyasi bulunamadi: {path}\n"
            "auto_download_from_hf=False oldugu icin otomatik indirme devre disi."
        )

    # Split'i dosya adindan cikar
    basename = os.path.basename(path).lower()
    if "test" in basename:
        split = "test"
    else:
        split = "train"

    return _download_lard_from_hf(
        split=split,
        output_path=path,
        hf_dataset=cfg.hf_dataset_name,
        hf_config=cfg.hf_config_name,
    )


def _load_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Desteklenmeyen tablo formati (.parquet ya da .csv bekleniyor): {path}")


def scenario_grouped_split(df: pd.DataFrame, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """'scenario' sutununa gore GRUPLU train/val bolme.

    ONEMLI: Ayni scenario'nun frame'leri (ardisik/yuksek korelasyonlu
    yaklasma goruntuleri) ASLA farkli split'lere dagilmaz -- aksi halde
    val metrikleri YAPAY OLARAK IYI cikar (train'de gorulen sahneye COK
    benzer bir frame val'de degerlendirilir), gercek genelleme yetenegini
    YANLIS yansitir. Bu, kullanicinin onceki LandNet calismalarinda zaten
    tespit ettigi bir risktir (bkz. gecmis oturum notlari)."""
    scenarios = sorted(df["scenario"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(scenarios)
    n_val = max(1, round(len(scenarios) * val_fraction))
    val_scenarios = set(scenarios[:n_val])
    val_mask = df["scenario"].isin(val_scenarios).to_numpy()
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    return train_idx, val_idx


# =============================================================================
# 2) Kamera intrinsics -- kirpma/yeniden-boyutlandirma sonrasi DOGRU K
# =============================================================================

def compute_K_for_view(orig_w: float, orig_h: float, hfov_deg: float, target_size: int,
                        crop_box: Optional[Tuple[float, float, float, float]] = None) -> torch.Tensor:
    """Once orijinal goruntu icin TAM K'yi (hfov_deg varsayimiyla) hesaplar,
    sonra (varsa) crop_box=(x0,y0,cw,ch) kirpmasi + target_size'a yeniden
    boyutlandirmanin intrinsics uzerindeki DOGRU etkisini uygular:
        scale_x = target_size / cw,  scale_y = target_size / ch
        fx' = fx*scale_x,  fy' = fy*scale_y
        cx' = (cx-x0)*scale_x,  cy' = (cy-y0)*scale_y
    crop_box=None ise x0=y0=0, cw=orig_w, ch=orig_h (yani sadece resize).

    !!! Kirpilmis bolge ARTIK orijinal hfov_deg'i KAPSAMAZ -- bu yuzden
    K'yi crop SONRASI icin sifirdan (crop bolgesi=tam FOV varsayarak)
    hesaplamak YANLIS olur; once TAM goruntu K'si hesaplanip SONRA crop/resize
    donusumu uygulanmalidir (yukaridaki gibi) -- bu fonksiyon bunu dogru yapar.
    """
    K_full = intrinsics_from_fov(hfov_deg, orig_w, orig_h)
    fx, fy = K_full[0, 0].item(), K_full[1, 1].item()
    cx, cy = K_full[0, 2].item(), K_full[1, 2].item()

    if crop_box is None:
        x0, y0, cw, ch = 0.0, 0.0, float(orig_w), float(orig_h)
    else:
        x0, y0, cw, ch = crop_box

    scale_x = target_size / cw
    scale_y = target_size / ch
    fx2, fy2 = fx * scale_x, fy * scale_y
    cx2, cy2 = (cx - x0) * scale_x, (cy - y0) * scale_y
    return build_intrinsics(fx2, fy2, cx2, cy2)


# =============================================================================
# 3) Dataset
# =============================================================================

class LARDPoseDataset(Dataset):
    def __init__(self, cfg: DataConfig, split: str = "train"):
        assert split in ("train", "val")
        self.cfg = cfg
        self.split = split

        # Dosya yoksa ve auto_download_from_hf=True ise HuggingFace'ten indir
        table_path = _ensure_table_exists(cfg)
        df = _load_table(table_path)
        required_cols = {"image", "airport", "runway", "scenario", "yaw", "pitch", "roll"} | \
                         {f"{ax}_{c}" for ax in ("x", "y") for c in CORNER_ORDER}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Tabloda eksik sutunlar: {sorted(missing)}")

        train_idx, val_idx = scenario_grouped_split(df, cfg.val_fraction, cfg.split_seed)
        idx = train_idx if split == "train" else val_idx
        self.df = df.iloc[idx].reset_index(drop=True)

        self.runway_db = load_runway_db(cfg.runway_db_path) if cfg.runway_db_path else None
        self._object_points_cache: Dict[Tuple[str, str], Tuple[torch.Tensor, dict]] = {}

        if cfg.photometric_aug and split == "train":
            self.color_jitter = T.ColorJitter(
                brightness=cfg.color_jitter_brightness,
                contrast=cfg.color_jitter_contrast,
                saturation=cfg.color_jitter_saturation,
                hue=cfg.color_jitter_hue,
            )
        else:
            self.color_jitter = None

    def __len__(self) -> int:
        return len(self.df)

    # -------------------------------------------------------------------
    def _get_object_points(self, airport: str, runway: str) -> Tuple[torch.Tensor, dict]:
        key = (str(airport), str(runway))
        if key not in self._object_points_cache:
            pts, info = get_runway_object_points(airport, runway, db=self.runway_db)
            self._object_points_cache[key] = (pts, info)
        return self._object_points_cache[key]

    def _load_image(self, row) -> Tuple[Image.Image, int]:
        img_field = row["image"]
        if isinstance(img_field, dict) and "bytes" in img_field:
            img = Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
        elif isinstance(img_field, (bytes, bytearray)):
            img = Image.open(io.BytesIO(img_field)).convert("RGB")
        elif isinstance(img_field, str):
            path = img_field
            if self.cfg.images_root is not None and not os.path.isabs(path):
                path = os.path.join(self.cfg.images_root, path)
            img = Image.open(path).convert("RGB")
        else:
            raise TypeError(f"Bilinmeyen 'image' alan turu: {type(img_field)}")

        watermark_h = int(row.get("watermark_height", 0) or 0)
        if self.cfg.watermark_crop and watermark_h > 0:
            w, h = img.size
            img = img.crop((0, 0, w, h - watermark_h))
        return img, watermark_h

    def _sample_valid_crop(self, corners_orig: np.ndarray, orig_w: int, orig_h: int
                            ) -> Optional[Tuple[float, float, float, float]]:
        lo, hi = self.cfg.geometric_aug_scale_range
        for _ in range(self.cfg.geometric_aug_max_retries):
            scale = random.uniform(lo, hi)
            side = scale * min(orig_w, orig_h)
            max_x0 = orig_w - side
            max_y0 = orig_h - side
            x0 = random.uniform(0, max(max_x0, 0.0))
            y0 = random.uniform(0, max(max_y0, 0.0))
            inside = (
                np.all(corners_orig[:, 0] >= x0) and np.all(corners_orig[:, 0] <= x0 + side) and
                np.all(corners_orig[:, 1] >= y0) and np.all(corners_orig[:, 1] <= y0 + side)
            )
            if inside:
                return (x0, y0, side, side)
        return None  # guvenli fallback: crop uygulanmaz, orijinal (tam) goruntu kullanilir

    # -------------------------------------------------------------------
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[i]
        img, watermark_h = self._load_image(row)
        orig_w, orig_h = img.size

        corners_orig = np.array(
            [[row[f"x_{c}"], row[f"y_{c}"]] for c in CORNER_ORDER], dtype=np.float64
        )

        crop_box = None
        if self.cfg.geometric_aug and self.split == "train":
            crop_box = self._sample_valid_crop(corners_orig, orig_w, orig_h)

        if crop_box is not None:
            x0, y0, cw, ch = crop_box
            img = img.crop((int(round(x0)), int(round(y0)), int(round(x0 + cw)), int(round(y0 + ch))))
        img = img.resize((self.cfg.img_size, self.cfg.img_size), Image.BILINEAR)

        if self.color_jitter is not None:
            img = self.color_jitter(img)

        K = compute_K_for_view(orig_w, orig_h, self.cfg.hfov_deg, self.cfg.img_size, crop_box)

        if crop_box is not None:
            x0, y0, cw, ch = crop_box
        else:
            x0, y0, cw, ch = 0.0, 0.0, float(orig_w), float(orig_h)
        scale_x = self.cfg.img_size / cw
        scale_y = self.cfg.img_size / ch
        corners = (corners_orig - np.array([x0, y0])) * np.array([scale_x, scale_y])

        img_tensor = TF.to_tensor(img)
        img_tensor = TF.normalize(img_tensor, mean=list(self.cfg.normalize_mean), std=list(self.cfg.normalize_std))

        yaw_raw = torch.tensor([float(row["yaw"])])
        pitch_raw = torch.tensor([float(row["pitch"])])
        roll_raw = torch.tensor([float(row["roll"])])
        offsets = self.cfg.angle_offsets if self.cfg.angle_offsets is not None else LARD_ANGLE_OFFSETS_DEG
        R_gt = lard_raw_angles_to_R(yaw_raw, pitch_raw, roll_raw, offsets=offsets)[0]

        points_3d, pts_info = self._get_object_points(row["airport"], row["runway"])
        points_2d_gt = torch.tensor(corners, dtype=torch.float32)
        img_diag = torch.tensor(math.sqrt(2) * self.cfg.img_size, dtype=torch.float32)

        return {
            "image": img_tensor,
            "R_gt": R_gt,
            "points_3d": points_3d,
            "points_2d_gt": points_2d_gt,
            "K": K,
            "img_diag": img_diag,
            "runway_source": pts_info["source"],  # "runways_db_V2_XPlane" | "default_fallback"
        }


# =============================================================================
# 4) Tani/dogrulama araclari -- EGITIMDEN ONCE calistirilmasi siddetle onerilir
# =============================================================================

def sanity_check_angle_convention(table_path: str, n_sample: int = 500) -> Dict[str, float]:
    """pitch_raw'nin gercekten ~90 derece civarinda yogunlastigini (yani
    LARD_ANGLE_OFFSETS_DEG['pitch']=90 varsayiminin TEK bir ornege degil,
    genel veriye uydugunu) dogrular. Ayrica pitch_raw-90'in vertical_path_angle
    ile isaret/buyukluk olarak tutarli olup olmadigini kontrol eder.

    Cagirin ve CIKTIYI OKUYUN -- pitch_raw ortalamasi 90'dan uzaksa (ornegin
    <60 ya da >120), LARD_ANGLE_OFFSETS_DEG'i GUNCELLEMEDEN egitime
    BASLAMAYIN."""
    df = _load_table(table_path)
    n = min(n_sample, len(df))
    sample = df.sample(n=n, random_state=0) if len(df) > n else df

    pitch_raw = sample["pitch"].to_numpy(dtype=np.float64)
    yaw_raw = sample["yaw"].to_numpy(dtype=np.float64)
    roll_raw = sample["roll"].to_numpy(dtype=np.float64)

    report = {
        "n_sample": n,
        "pitch_raw_mean": float(pitch_raw.mean()),
        "pitch_raw_std": float(pitch_raw.std()),
        "pitch_raw_min": float(pitch_raw.min()),
        "pitch_raw_max": float(pitch_raw.max()),
        "yaw_raw_mean": float(yaw_raw.mean()),
        "yaw_raw_std": float(yaw_raw.std()),
        "roll_raw_mean": float(roll_raw.mean()),
        "roll_raw_std": float(roll_raw.std()),
    }

    if "vertical_path_angle" in sample.columns:
        vpa = sample["vertical_path_angle"].to_numpy(dtype=np.float64)
        pitch_corrected = pitch_raw - LARD_ANGLE_OFFSETS_DEG["pitch"]
        # beklenen iliski: pitch_corrected ~ -vertical_path_angle (isaret/buyukluk yakinligi, TAM esitlik degil)
        corr = np.corrcoef(pitch_corrected, -vpa)[0, 1] if len(sample) > 1 else float("nan")
        report["pitch_corrected_vs_neg_vpa_correlation"] = float(corr)
        report["pitch_corrected_mean"] = float(pitch_corrected.mean())
        report["vertical_path_angle_mean"] = float(vpa.mean())

    print("=" * 70)
    print("ACI KONVANSIYONU SANITY CHECK")
    print("=" * 70)
    print(f"Ornek boyutu: {report['n_sample']}")
    print(f"pitch_raw: mean={report['pitch_raw_mean']:.2f} std={report['pitch_raw_std']:.2f} "
          f"[{report['pitch_raw_min']:.2f}, {report['pitch_raw_max']:.2f}]")
    print(f"  -> Eger mean, 90'a YAKIN degilse (ornegin <60 ya da >120), "
          "LARD_ANGLE_OFFSETS_DEG['pitch']=90 varsayimi YANLIS olabilir!")
    print(f"yaw_raw:   mean={report['yaw_raw_mean']:.2f} std={report['yaw_raw_std']:.2f}")
    print(f"roll_raw:  mean={report['roll_raw_mean']:.2f} std={report['roll_raw_std']:.2f}")
    if "pitch_corrected_vs_neg_vpa_correlation" in report:
        c = report["pitch_corrected_vs_neg_vpa_correlation"]
        print(f"(pitch_raw-90) ile -vertical_path_angle korelasyonu: {c:.3f}")
        print(f"  -> Pozitif ve yuksek (ornegin >0.5) olmasi BEKLENIR (glideslope acisiyla "
              "tutarli pitch attitude). Dusuk/negatifse offset varsayimini sorgulayin.")
    print("=" * 70)
    return report


def report_runway_db_coverage(table_path: str, runway_db_path: str) -> Dict[str, float]:
    """Veri kumesindeki satirlarin YUZDE KACININ gercek runway_db'den (fallback
    DEGIL) geometriyle eslesdigini raporlar. Dusuk kapsama (ornegin <%90),
    ya normalize_runway_id()'nin yakalayamadigi bir format farkligi ya da
    DB'de eksik havaalanlari oldugunu gosterir -- bu, o satirlarin idealize
    (45m/3000m) geometriyle egitilecegi, hafif dogruluk kaybi anlamina gelir."""
    df = _load_table(table_path)
    db = load_runway_db(runway_db_path)
    from landnet import get_runway_object_points as _gop

    n_db, n_fallback = 0, 0
    missing_pairs = set()
    for _, row in df.iterrows():
        _, info = _gop(row["airport"], row["runway"], db=db)
        if info["source"] == "runways_db_V2_XPlane":
            n_db += 1
        else:
            n_fallback += 1
            missing_pairs.add((row["airport"], str(row["runway"])))

    total = n_db + n_fallback
    coverage = n_db / total if total else 0.0
    print("=" * 70)
    print("RUNWAY DB KAPSAMA RAPORU")
    print("=" * 70)
    print(f"Toplam satir: {total} | Gercek DB: {n_db} (%{100*coverage:.1f}) | Fallback: {n_fallback}")
    if missing_pairs:
        print(f"DB'de bulunamayan (airport, runway) ciftleri ({len(missing_pairs)} adet), ilk 20:")
        for p in list(sorted(missing_pairs))[:20]:
            print(f"  {p}")
    print("=" * 70)
    return {"coverage": coverage, "n_db": n_db, "n_fallback": n_fallback, "missing_pairs": missing_pairs}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LandNet veri kumesi tani araclari")
    parser.add_argument("--table_path", type=str, required=True)
    parser.add_argument("--runway_db_path", type=str, default=None)
    args = parser.parse_args()

    sanity_check_angle_convention(args.table_path)
    if args.runway_db_path:
        report_runway_db_coverage(args.table_path, args.runway_db_path)