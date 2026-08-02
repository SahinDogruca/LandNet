"""
train.py — LandNet egitim dongusu.

Hedef: pitch/yaw/roll icin cok yuksek dogruluk (hedef egitim RMSE ~0.001
derece, gercek dunya/val performansinda ~0.01 derece civarina ulasmak icin).
RTX 4070 Super (12GB VRAM) icin varsayilan degerlerle optimize edilmistir;
300 epoch gibi uzun kosular icin (checkpoint/resume, EMA, warmup+cosine LR,
gradient accumulation) tasarlanmistir.

============================ KULLANIMDAN ONCE ============================
1) EGITIME BASLAMADAN ONCE mutlaka calistirin:
       python landnet_data.py --table_path <tablonuz> --runway_db_path runways_db_V2_XPlane.json
   Bu, aci konvansiyonu varsayimini (pitch offset=90) ve runway DB kapsamasini
   raporlar. pitch_raw ortalamasi ~90'dan uzaksa DURUN ve
   landnet.LARD_ANGLE_OFFSETS_DEG'i (ya da TrainConfig.angle_offsets'i)
   duzeltmeden devam ETMEYIN -- yanlis konvansiyonla egitilen model, ne kadar
   uzun egitilirse egitilsin SISTEMATIK olarak yanlis ogrenir.

2) Ornek calistirma:
       python train.py --table_path data/train.parquet \\
                        --runway_db_path runways_db_V2_XPlane.json \\
                        --output_dir runs/landnet_v1 \\
                        --img_size 640 --batch_size 8 --grad_accum_steps 2 \\
                        --epochs 300

3) 12GB VRAM icin batch_size/img_size rehberi (yaklasik, GPU'ya gore degisebilir):
       img_size=640,  batch_size=8-12,  grad_accum_steps=1-2  (use_grad_checkpoint=False)
       img_size=1024, batch_size=2-4,   grad_accum_steps=4-8  (use_grad_checkpoint=True onerilir)
   Efektif batch_size = batch_size * grad_accum_steps; BatchNorm YERINE
   GroupNorm kullanildigi icin (bkz. landnet.py) kucuk fiziksel batch_size
   ISTIKRARI BOZMAZ -- ama optimizer gradyan gurultusu icin yine de efektif
   batch_size'i makul (>=16) tutmak onerilir.
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from landnet import (
    LandNetConfig, LandNet, LandNetLoss, PoseMetricLogger, refine_rotation,
)
from landnet_data import DataConfig, LARDPoseDataset, sanity_check_angle_convention, report_runway_db_coverage


# =============================================================================
# 0) Yapilandirma
# =============================================================================

@dataclass
class TrainConfig:
    table_path: str
    runway_db_path: str
    output_dir: str = "./landnet_runs/run1"

    img_size: int = 640
    hfov_deg: float = 60.0
    images_root: Optional[str] = None   # 'image' sutunu goreli yol ise kok dizin

    batch_size: int = 8
    grad_accum_steps: int = 2          # efektif batch = batch_size * grad_accum_steps
    num_workers: int = 8
    val_fraction: float = 0.12

    epochs: int = 300
    lr: float = 2e-4                    # kullanicinin onceki HPO bulgularina yakin (~1-4.5e-4)
    backbone_lr_mult: float = 1.0       # <1.0 verilirse CNN+Transformer govdesi icin ayri (dusuk) LR
    weight_decay: float = 1e-4          # minimal regularization (onceki bulgu: underfitting riski, overfitting degil)
    warmup_epochs: int = 5
    lr_floor_mult: float = 0.01         # cosine decay'in inecegi taban: lr * bu carpan

    # Reprojection loss WARM-UP: ilk N epoch boyunca SADECE rotasyon loss'u
    # kullanilir (reprojection agirligi 0). R henuz cok yanlisken reprojection
    # gradyani anlamsiz/gurultulu olabilir (bkz. landnet.py'deki clamp notu);
    # bu warm-up, egitimin erken kararliligini ekstra guvence altina alir.
    reproj_warmup_epochs: int = 10

    grad_clip_norm: float = 1.0
    use_amp: bool = True
    amp_dtype: str = "bf16"             # RTX 4070 Super (Ada) bf16'yi native destekler, fp16+GradScaler'dan daha kararli

    use_ema: bool = True
    ema_decay: float = 0.999

    use_grad_checkpoint: bool = False   # img_size=1024'te True onerilir
    norm_type: str = "group"            # bkz. landnet.py -- kucuk batch'te BatchNorm yerine GroupNorm

    seed: int = 42
    val_every: int = 1
    save_every: int = 5
    refine_eval_every: int = 5          # post-hoc refinement'i her val'de degil, N epoch'ta bir degerlendir (yavas)
    refine_eval_max_batches: int = 4     # refinement degerlendirmesi icin sadece ilk N val batch'i (hiz icin)

    resume_from: Optional[str] = None
    angle_offsets: Optional[Dict[str, float]] = None   # None -> landnet.LARD_ANGLE_OFFSETS_DEG

    allow_tf32: bool = True             # Ada/Ampere'de matmul hizi icin (son agirlik hassasiyetine etkisi ihmal edilebilir)


# =============================================================================
# 1) EMA -- NaN/Inf'e karsi korumali (bkz. kullanicinin gecmis LandNet
#    denemelerinde yasadigi "EMA weight contamination from NaN events" sorunu)
# =============================================================================

class SafeEMA:
    """Standart EMA, TEK farkla: her guncellemeden once ilgili tensorun
    SONLU (finite) oldugu kontrol edilir. Bozuk (NaN/Inf) bir agirlik ASLA
    EMA golgesine karismaz -- bozuk adim sessizce atlanir, EMA bir onceki
    saglikli durumunu korur."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.skipped_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        sd = model.state_dict()
        for k, v in sd.items():
            if not torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone()
                continue
            if not torch.isfinite(v).all():
                self.skipped_updates += 1
                continue
            self.shadow[k].mul_(self.decay).add_(v.detach().to(self.shadow[k].dtype), alpha=1 - self.decay)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}


# =============================================================================
# 2) LR schedule -- linear warmup + cosine decay
# =============================================================================

def lr_lambda_factory(total_steps: int, warmup_steps: int, floor_mult: float):
    def _fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return floor_mult + (1 - floor_mult) * cosine
    return _fn


# =============================================================================
# 3) Yardimci: batch'i cihaza tasi
# =============================================================================

def _to_device(batch: Dict, device, non_blocking: bool = True) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v
    return out


# =============================================================================
# 4) Validasyon dongusu (ham + istege bagli post-hoc refinement)
# =============================================================================

@torch.no_grad()
def validate(model: nn.Module, criterion: LandNetLoss, loader: DataLoader, device,
             amp_dtype: torch.dtype, run_refinement: bool = False,
             refine_max_batches: int = 4) -> Dict[str, float]:
    model.eval()
    raw_logger = PoseMetricLogger()
    refined_logger = PoseMetricLogger() if run_refinement else None
    total_loss, total_loss_rot, total_loss_reproj, n_batches = 0.0, 0.0, 0.0, 0

    pbar = tqdm(loader, desc="  [val]", leave=False, dynamic_ncols=True)
    for bi, batch in enumerate(pbar):
        batch = _to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            pred = model(batch["image"])
            out = criterion(pred, batch["R_gt"], K=batch["K"], points_3d=batch["points_3d"],
                             points_2d_gt=batch["points_2d_gt"], img_diag=batch["img_diag"])

        total_loss += out["loss"].item()
        total_loss_rot += out["loss_rotation"].item()
        total_loss_reproj += out.get("loss_reprojection", torch.tensor(0.0)).item()
        n_batches += 1

        raw_logger.update(pred["R"].float(), batch["R_gt"].float())

        if run_refinement and bi < refine_max_batches:
            # Not: refine_rotation icsel olarak autograd kullanir (Adam),
            # bu yuzden torch.no_grad() disina cikilir (enable_grad decorator
            # zaten fonksiyonun kendisinde var).
            refine_out = refine_rotation(
                pred["R"].float(), batch["K"].float(), batch["points_3d"].float(),
                batch["points_2d_gt"].float(), num_steps=100, lr=5e-3,
                img_diag=batch["img_diag"].float(),
            )
            refined_logger.update(refine_out["R_refined"], batch["R_gt"].float())

    metrics = {
        "val_loss": total_loss / max(1, n_batches),
        "val_loss_rotation": total_loss_rot / max(1, n_batches),
        "val_loss_reprojection": total_loss_reproj / max(1, n_batches),
    }
    metrics.update({f"raw_{k}": v for k, v in raw_logger.compute().items()})
    if run_refinement:
        metrics.update({f"refined_{k}": v for k, v in refined_logger.compute().items()})
    return metrics


# =============================================================================
# 5) Ana egitim dongusu
# =============================================================================

def train(cfg: TrainConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "train_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.allow_tf32
        torch.backends.cudnn.benchmark = True

    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16

    # --- Veri kumesi: dosya yoksa HuggingFace'ten otomatik indir ---
    data_cfg = DataConfig(
        table_path=cfg.table_path, runway_db_path=cfg.runway_db_path,
        images_root=cfg.images_root,
        img_size=cfg.img_size, hfov_deg=cfg.hfov_deg, val_fraction=cfg.val_fraction,
        split_seed=cfg.seed, angle_offsets=cfg.angle_offsets,
    )
    # Dosya yoksa HuggingFace'ten indir (DataConfig.auto_download_from_hf=True ise)
    from landnet_data import _ensure_table_exists
    resolved_table_path = _ensure_table_exists(data_cfg)

    # --- ACI KONVANSIYONU + RUNWAY DB KAPSAMA raporlanir (guvenlik) ---
    print("\n[1/5] Aci konvansiyonu ve runway DB kapsama kontrolleri calistiriliyor...")
    sanity_check_angle_convention(resolved_table_path)
    report_runway_db_coverage(resolved_table_path, cfg.runway_db_path)

    train_ds = LARDPoseDataset(data_cfg, split="train")
    val_ds = LARDPoseDataset(data_cfg, split="val")
    print(f"[2/5] Veri kumesi: train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                               persistent_workers=(cfg.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True,
                             persistent_workers=(cfg.num_workers > 0))

    model_cfg = LandNetConfig(img_size=cfg.img_size, norm_type=cfg.norm_type,
                               use_grad_checkpoint=cfg.use_grad_checkpoint)
    model = LandNet(model_cfg).to(device)
    criterion = LandNetLoss(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[3/5] LandNet parametre sayisi: {n_params / 1e6:.2f}M")

    # --- Discriminative LR (istege bagli): rotation_head her zaman tam LR, govde carpan ile ---
    if cfg.backbone_lr_mult != 1.0:
        head_params = list(model.rotation_head.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
        param_groups = [
            {"params": backbone_params, "lr": cfg.lr * cfg.backbone_lr_mult},
            {"params": head_params, "lr": cfg.lr},
            {"params": list(criterion.parameters()), "lr": cfg.lr},
        ]
    else:
        param_groups = [{"params": list(model.parameters()) + list(criterion.parameters()), "lr": cfg.lr}]

    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(total_steps, warmup_steps, cfg.lr_floor_mult))

    ema = SafeEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

    start_epoch = 0
    best_val_geodesic = float("inf")
    if cfg.resume_from is not None and os.path.exists(cfg.resume_from):
        print(f"[4/5] Checkpoint'ten devam ediliyor: {cfg.resume_from}")
        ckpt = torch.load(cfg.resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        criterion.load_state_dict(ckpt["criterion"])
        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"] + 1
        best_val_geodesic = ckpt.get("best_val_geodesic", float("inf"))
    else:
        print("[4/5] Sifirdan egitim baslatiliyor.")

    history_path = os.path.join(cfg.output_dir, "history.jsonl")
    print(f"[5/5] Egitim basliyor: {cfg.epochs} epoch, {steps_per_epoch} adim/epoch "
          f"(efektif batch={cfg.batch_size * cfg.grad_accum_steps})\n")

    global_step = start_epoch * steps_per_epoch
    epoch_pbar = tqdm(range(start_epoch, cfg.epochs), desc="Epoch", dynamic_ncols=True)
    for epoch in epoch_pbar:
        model.train()
        epoch_t0 = time.time()
        use_reproj_this_epoch = epoch >= cfg.reproj_warmup_epochs

        running_loss, running_rot, running_reproj = 0.0, 0.0, 0.0
        n_finite_steps, n_skipped_steps = 0, 0

        optimizer.zero_grad(set_to_none=True)
        batch_pbar = tqdm(train_loader, desc=f"  Epoch {epoch}", leave=False, dynamic_ncols=True)
        for step, batch in enumerate(batch_pbar):
            batch = _to_device(batch, device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                pred = model(batch["image"])
                reproj_kwargs = dict(K=batch["K"], points_3d=batch["points_3d"],
                                      points_2d_gt=batch["points_2d_gt"], img_diag=batch["img_diag"]) \
                    if use_reproj_this_epoch else dict(K=None, points_3d=None, points_2d_gt=None, img_diag=None)
                out = criterion(pred, batch["R_gt"], **reproj_kwargs)
                loss = out["loss"] / cfg.grad_accum_steps

            if not torch.isfinite(loss):
                # Bozuk (NaN/Inf) bir adim -- optimizer.step()'e ASLA izin verme,
                # gradyanlari temizle ve bu mini-batch'i atla (kullanicinin gecmiste
                # yasadigi NaN-kaynakli sorunlara karsi savunma).
                n_skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                grad_finite = all(
                    torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
                )
                if grad_finite:
                    optimizer.step()
                    if ema is not None:
                        ema.update(model)
                    n_finite_steps += 1
                else:
                    n_skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running_loss += out["loss"].item()
            running_rot += out["loss_rotation"].item()
            running_reproj += out.get("loss_reprojection", torch.tensor(0.0)).item()

            # tqdm postfix guncelle
            avg_loss = running_loss / (step + 1)
            avg_rot = running_rot / (step + 1)
            batch_pbar.set_postfix(loss=f"{avg_loss:.4f}", rot=f"{avg_rot:.4f}", skip=n_skipped_steps)

        n_batches = len(train_loader)
        train_metrics = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, n_batches),
            "train_loss_rotation": running_rot / max(1, n_batches),
            "train_loss_reprojection": running_reproj / max(1, n_batches),
            "reproj_active": use_reproj_this_epoch,
            "n_skipped_steps": n_skipped_steps,
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_sec": time.time() - epoch_t0,
        }
        epoch_pbar.set_postfix(
            loss=f"{train_metrics['train_loss']:.4f}",
            rot=f"{train_metrics['train_loss_rotation']:.4f}",
            lr=f"{train_metrics['lr']:.1e}",
            t=f"{train_metrics['epoch_time_sec']:.0f}s",
        )
        tqdm.write(f"[epoch {epoch:4d}/{cfg.epochs}] loss={train_metrics['train_loss']:.6f} "
              f"rot={train_metrics['train_loss_rotation']:.6f} "
              f"reproj={train_metrics['train_loss_reprojection']:.6f} "
              f"(aktif={use_reproj_this_epoch}) lr={train_metrics['lr']:.2e} "
              f"atlanan_adim={n_skipped_steps} sure={train_metrics['epoch_time_sec']:.1f}s")

        log_entry = dict(train_metrics)

        if (epoch + 1) % cfg.val_every == 0 or epoch == cfg.epochs - 1:
            run_refine = ((epoch + 1) % cfg.refine_eval_every == 0) or (epoch == cfg.epochs - 1)
            val_metrics = validate(model, criterion, val_loader, device, amp_dtype,
                                    run_refinement=run_refine, refine_max_batches=cfg.refine_eval_max_batches)
            tqdm.write(f"  [val] loss={val_metrics['val_loss']:.6f} "
                  f"geodesic_MAE={val_metrics['raw_geodesic_mae_deg']:.4f} deg "
                  f"geodesic_RMSE={val_metrics['raw_geodesic_rmse_deg']:.4f} deg")
            tqdm.write(f"        yaw_RMSE={val_metrics['raw_yaw_rmse_deg']:.4f} "
                  f"pitch_RMSE={val_metrics['raw_pitch_rmse_deg']:.4f} "
                  f"roll_RMSE={val_metrics['raw_roll_rmse_deg']:.4f} (derece)")
            if run_refine:
                tqdm.write(f"  [val+refine] geodesic_RMSE={val_metrics['refined_geodesic_rmse_deg']:.4f} deg "
                      f"(ham modelden fark: "
                      f"{val_metrics['raw_geodesic_rmse_deg'] - val_metrics['refined_geodesic_rmse_deg']:+.4f} deg)")
            log_entry.update(val_metrics)

            if val_metrics["raw_geodesic_rmse_deg"] < best_val_geodesic:
                best_val_geodesic = val_metrics["raw_geodesic_rmse_deg"]
                _save_checkpoint(cfg, model, optimizer, scheduler, criterion, ema, epoch,
                                  best_val_geodesic, os.path.join(cfg.output_dir, "best.pt"))
                tqdm.write(f"  -> YENI EN IYI model kaydedildi (geodesic_RMSE={best_val_geodesic:.4f} deg)")

        with open(history_path, "a") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.epochs - 1:
            _save_checkpoint(cfg, model, optimizer, scheduler, criterion, ema, epoch,
                              best_val_geodesic, os.path.join(cfg.output_dir, "last.pt"))

    print(f"\nEgitim tamamlandi. En iyi val geodesic RMSE: {best_val_geodesic:.4f} derece.")
    print(f"En iyi model: {os.path.join(cfg.output_dir, 'best.pt')}")


def _save_checkpoint(cfg, model, optimizer, scheduler, criterion, ema, epoch, best_val_geodesic, path):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "criterion": criterion.state_dict(),
        "epoch": epoch,
        "best_val_geodesic": best_val_geodesic,
        "config": asdict(cfg),
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    torch.save(ckpt, path)


# =============================================================================
# 6) CLI
# =============================================================================

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="LandNet egitim")
    p.add_argument("--table_path", type=str, required=True)
    p.add_argument("--runway_db_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./landnet_runs/run1")
    p.add_argument("--images_root", type=str, default=None)
    p.add_argument("--img_size", type=int, default=640)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--reproj_warmup_epochs", type=int, default=10)
    p.add_argument("--use_grad_checkpoint", action="store_true")
    p.add_argument("--norm_type", type=str, default="group", choices=["group", "batch"])
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    return TrainConfig(
        table_path=args.table_path, runway_db_path=args.runway_db_path, output_dir=args.output_dir,
        images_root=args.images_root,
        img_size=args.img_size, batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs, reproj_warmup_epochs=args.reproj_warmup_epochs,
        use_grad_checkpoint=args.use_grad_checkpoint, norm_type=args.norm_type,
        resume_from=args.resume_from, seed=args.seed,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)