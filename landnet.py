"""
LandNet (duzeltilmis + genisletilmis surum) — "Combine CNN and Transformer to
Learn Absolute Camera Pose for the Fixed-Wing Aircraft Approach and Landing"
(Shen, Yu, Zhang, Yan, Zhai; Remote Sens. 2025, 17(4), 653)

=======================================================================
BU SURUMDE YAPILAN DUZELTMELER (kullanicinin PDF'ten (sayfa 7-11)
paylastigi Methodology bolumune dayanarak):
=======================================================================

1) CNN Encoder (Table 1) -- ONCEKI VARSAYIM (ResNet-50 benzeri) YANLISTI.
   Gercek tablo:
     s1: mid=64,  out=256,  blok=4   (1 downsample-blok + 3 identity-blok)
     s2: mid=128, out=512,  blok=4   (tablo formatlamasi biraz belirsiz,
                                       en olasi okuma budur -- asagida
                                       config'ten degistirilebilir)
     s3: mid=256, out=1024, blok=3
     s4: mid=256, out=1024, blok=1   (!! s3 ile AYNI kanal sayisi, ResNet'in
                                       aksine s4'te kanal artmiyor -- tablo
                                       boyle acikca gosteriyor)
   NOT: s2 (blok=4) ve s4 (blok=1) degerleri PDF tablo hucrelerinin metne
   donusumunde ufak belirsizlik tasiyor (parantezli blok gorsel olarak iki
   kere goruntuleniyor); en tutarli/literal okuma yukaridaki gibi alindi.
   Bu sayilar LandNetConfig.cnn_stage_blocks / cnn_stage_channels uzerinden
   tek satirda degistirilebilir.

2) Feature Interactive Block (FIB) -- YON HATASI vardi.
   Metin acikca soyluyor:
     - CNN(local) -> Transformer(global) yonunde: downsample + CONCATENATION
       ("feature concatenation is performed to fuse the local and global
       features") -- ONCEKI SURUMDE YANLISLIKLA ADDITION yapiyordum, simdi
       CONCAT + linear-projection-back-to-E olarak duzeltildi.
     - Transformer(global) -> CNN(local) yonunde: upsample + 1x1 conv +
       ELEMENT-WISE ADDITION ("element-wise addition is employed to
       complete the feature enhancement for the CNN encoder") -- bu zaten
       dogruydu, degismedi.
   Ayrica Figure 5'e gore FIB, 4 CNN/4 Transformer blogu arasinda TOPLAM 3
   kere calisiyor (once 4 degil) -- bu da duzeltildi.

3) Attentional ConvTrans Fusion Block (ACFB) -- YANLIS "capraz-gate"
   varsayimim TAMAMEN degisti. Gercek Eq.(14)-(15) ve Figure 9'a gore:
     - f_s (uzamsal attention, CBAM-tarzi mean+max+conv+sigmoid) F_CNN'DEN
       hesaplanir ve F_CNN'in KENDİSİNİ gate'ler (self-attention, capraz
       degil!).
     - f_c (kanal attention, SENet-tarzi GAP+FC+ReLU+FC+sigmoid)
       F_Transformer'DAN hesaplanir ve F_Transformer'in KENDİSİNİ gate'ler.
     - F_out = Concat(F_CNN (x) f_s, F_Transformer (x) f_c)
   Bu artik Eq.(14)-(15) ile birebir orntusuyor.

4) Loss normu -- Eq.(16) acikca "Euclidean distance" (L2) diyor; varsayilan
   norm "l1"den "l2"ye cekildi.

=======================================================================
KULLANICI TALEBIYLE YAPILAN MIMARI DEGISIKLIKLER (makaleden sapma,
acikca boyle istendi):
=======================================================================

A) Pozisyon (translation) regresyonu TAMAMEN CIKARILDI. Model artik SADECE
   rotasyonu (pitch/yaw/roll) tahmin ediyor. Translation, geometrik
   refinement asamasinda (asagida C) kapali-form (closed-form) olarak,
   bilinen pist 3B koordinatlari + goruntudeki 2B kose noktalarindan
   COZULUYOR (network tarafindan OGRENILMIYOR) -- boylece network sadece
   goreceli olarak daha "kolay/az belirsiz" olan rotasyon problemine
   odaklaniyor.

B) Rotasyon temsili: QUATERNION yerine 6D continuous rotation
   representation (Zhou et al., CVPR 2019, "On the Continuity of Rotation
   Representations in Neural Networks") kullaniliyor. Gram-Schmidt tabanli,
   surekli (continuous) ve enjektif bir harita oldugu icin regresyon
   hatasi quaternion'a gore literatude sistematik olarak daha dusuk cikiyor
   (quaternion'daki q/-q double-cover sureksizligi yok). Loss, rotasyon
   MATRISI uzerinde (chordal/Frobenius) hesaplaniyor; pitch/yaw/roll SADECE
   loglama/metrik icin turetiliyor (egitimi etkilemiyor, gimbal-lock riski
   yok).

C) Geometrik Refinement: LandNet -> 6D rotasyon -> R -> (bilinen pist
   3B geometrisi + 2B kose noktalari ile) closed-form translation cozumu
   -> reprojection hatasi -> hem (i) egitim sirasinda differentiable bir
   ek loss terimi olarak, hem de (ii) inference sonrasi post-hoc iteratif
   optimizasyon (Adam ile birkac adim, R'yi hafifce duzeltiyor) olarak
   kullaniliyor. Ikisi de AYNI geometri modulunu paylasiyor.

D) Goruntu boyutu ESNEK: ICB ve CNN Encoder tamamen convolutional
   (fully-convolutional) oldugu icin dogal olarak her boyutu destekler.
   Transformer'in ogrenilen positional embedding'i ise standart ViT/DeiT
   teknigiyle (bicubic interpolation) farkli patch-grid boyutlarina
   otomatik uydurulur. Tek sart: img_size, 16'ya tam bolunebilmeli (stem
   /4 * patch-embed /4 = /16). 640/16=40, 1024/16=64 -- ikisi de calisir.

Yazar: Claude (Anthropic) — Sahin icin hazirlanmistir.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# =============================================================================
# 0) Yapilandirma
# =============================================================================

@dataclass
class LandNetConfig:
    # --- Girdi ---
    img_size: int = 640          # 640, 1024, vb. -- 16'ya tam bolunmeli
    in_channels: int = 3

    # --- ICB (Initial Convolutional Block), Bolum 3.2 ---
    stem_channels: int = 64

    # --- Normalizasyon katmani secimi ---
    # ONEMLI: 640-1024 cozunurlukte 12GB VRAM'de fiziksel batch_size kucuk
    # kalacagi icin (ozellikle 1024'te muhtemelen 2-4) BatchNorm'un batch
    # istatistikleri GURULTULU/guvenilmez olur -- bu da sub-0.1 derece hedefi
    # icin kabul edilemez bir instabilite kaynagidir. GroupNorm batch
    # boyutundan BAGIMSIZ oldugu icin varsayilan olarak seçildi (kucuk
    # batch'lerde production-grade dogruluk icin standart pratik).
    norm_type: str = "group"      # "group" (onerilen, kucuk batch'te kararli) | "batch"
    group_norm_groups: int = 32   # kanal sayisi bu sayiya (ya da en yakin bolene) gore gruplanir

    # --- CNN Encoder, Bolum 3.3 + Table 1 (DUZELTILDI, bkz. yukaridaki not) ---
    cnn_stage_channels: List[int] = field(default_factory=lambda: [256, 512, 1024, 1024])
    cnn_stage_blocks: List[int] = field(default_factory=lambda: [4, 4, 3, 1])
    cnn_stage_strides: List[int] = field(default_factory=lambda: [1, 2, 2, 2])
    bottleneck_reduction: int = 4  # Table 1'deki tum stage'lerde mid=out/4 orani tutarli

    # --- Transformer Encoder, Bolum 3.4 ---
    # ONEMLI (img_size esnekligi icin duzeltme): Makalede patch embedding
    # "4x4 conv stride 4" olarak sabit tanimli ve 224 girdi icin 14x14=196
    # token uretiyor. Eger bu stride'i SABIT tutup img_size'i 1024'e
    # cikarirsaniz, grid (1024/16)=64x64=4096 token olur ve self-attention
    # O(N^2) oldugu icin bellek/hesap ~450x artar (12GB VRAM'de OOM garanti
    # -- nitekim asagidaki duman testinde tam olarak bu yuzden crash oldu).
    # Bunun yerine patch grid'i SABIT bir boyuta (patch_grid_size) adaptive
    # pooling ile oturtuyoruz -- boylece token sayisi img_size'dan BAGIMSIZ
    # ve sabit kalir (bellek/hesap ongorulebilir), CNN kolu ise tamamen
    # convolutional oldugu icin dogal olarak img_size ile birlikte
    # olceklenmeye devam eder (yuksek cozunurlukten hala fayda saglanir,
    # sadece global Transformer kolu icin degil).
    patch_grid_size: int = 20     # 20x20=400 token (makaledeki 14x14'e yakin, guvenli varsayim)
    embed_dim: int = 384
    transformer_depth: int = 8    # 4 stage'e esit dagitilir (stage basina 2 blok)
    num_heads: int = 6
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0

    # --- ACFB / MLP head ---
    fusion_channels: int = 512
    head_hidden: int = 512
    head_dropout: float = 0.2

    # --- Rotasyon temsili ---
    rotation_repr: str = "6d"     # sadece "6d" destekleniyor (kullanici karari)

    # --- Bellek/hesap ayari ---
    # 1024 gibi buyuk cozunurluklerde (ozellikle 12GB VRAM'de daha buyuk
    # batch_size kullanmak icin) CNN stage'lerinde gradient checkpointing
    # aktif edilebilir: ileri gecis aktivasyonlarini SAKLAMAZ, geri yayilimda
    # yeniden hesaplar -- bellek ~%40-60 azalir, egitim suresi ~%20-30 artar.
    # Gercek zamanlilik gerekmedigi icin (kullanicinin belirttigi gibi) bu
    # takas genelde degerlidir.
    use_grad_checkpoint: bool = False

    # --- Loss ---
    rotation_loss_type: str = "chordal"   # "chordal" (Frobenius, onerilen) | "geodesic"
    use_reprojection_loss: bool = True
    reproj_norm_by_diag: bool = True      # piksel hatasini goruntu koseginine bolerek olcek-bagimsiz yap
    # ONEMLI: R henuz egitilmemis/yanlisken (ozellikle egitimin ilk adimlarinda,
    # gercek degerden ~180 derece uzak rotasyonlarda) duzlemsel pist noktalari
    # HICBIR t ile iyi aciklanamaz -- bazi noktalar kameranin "arkasina" dusebilir
    # ve perspektif bolme asiri buyuk piksel hatalarina yol acar. Ham L2 kullanmak
    # bu durumda optimizer'i bozacak devasa gradyanlar uretebilir. Bunun yerine
    # robust (Huber) loss varsayilan -- bu, bundle adjustment / PnP literaturunde
    # tam bu senaryo icin standart bir pratiktir.
    reprojection_loss_type: str = "huber"   # "huber" (onerilen) | "l2"
    reprojection_huber_delta: float = 0.05  # normalize edilmis (goruntu koseginine bolunmus) birimde
    # R cok yanlisken (ozellikle egitimin basinda) bazi 3B noktalar kameranin
    # ARKASINA duşebilir (z<=0), perspektif bolme SINIRSIZ buyuyebilir -- Huber'in
    # sekli (linear) bile SINIRSIZ bir girdide sonsuza gider. Bu yuzden loss'a
    # giren hatayi sabit bir ust sinirla clamp'liyoruz: R cok yanlisken reprojection
    # loss'un gradyani dogal olarak doygunlasir/~0'a yaklasir (rotasyon loss'u tek
    # basina yon verir), R makul hale geldikce reprojection devreye girer --
    # ORTUK bir warm-up mekanizmasi saglar, manuel epoch zamanlamasi gerekmez.
    reprojection_max_error: float = 1.0     # normalize edilmis birimde (1.0 = goruntu koseginin tamami)

    def __post_init__(self):
        assert self.transformer_depth % len(self.cnn_stage_blocks) == 0, \
            "transformer_depth, stage sayisina (4) tam bolunmeli"
        # patch_grid_size artik adaptive pooling ile elde edildigi icin img_size
        # sadece ICB'nin /4 downsample'ina bolunebilir olmali (cok gevsek bir kisit).
        assert self.img_size % 4 == 0, "img_size, ICB'nin /4 downsample'i nedeniyle 4'e bolunmeli"
        assert len(self.cnn_stage_channels) == len(self.cnn_stage_blocks) == len(self.cnn_stage_strides) == 4


# =============================================================================
# 1) Rotasyon yardimci fonksiyonlari
#    - Euler(ZYX, yaw-pitch-roll) <-> Rotasyon matrisi  (scipy 'ZYX' ile
#      dogrulanmis, bkz. sohbet: R = Rz(yaw) @ Ry(pitch) @ Rx(roll))
#    - 6D continuous representation <-> Rotasyon matrisi (Zhou et al. 2019)
# =============================================================================

def euler_zyx_to_matrix(yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    """(..., ) yaw, pitch, roll [radyan] -> (..., 3, 3) rotasyon matrisi.
    Konvansiyon: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  (scipy 'ZYX' extrinsic ile
    dogrulanmistir -- bkz. dosyanin basindaki gelistirme notlari)."""
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)

    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)

    Rz = torch.stack([
        torch.stack([cy, -sy, zeros], dim=-1),
        torch.stack([sy,  cy, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ], dim=-2)
    Ry = torch.stack([
        torch.stack([cp, zeros, sp], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-sp, zeros, cp], dim=-1),
    ], dim=-2)
    Rx = torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, cr, -sr], dim=-1),
        torch.stack([zeros, sr,  cr], dim=-1),
    ], dim=-2)

    return Rz @ Ry @ Rx


def matrix_to_euler_zyx(Rm: torch.Tensor, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(..., 3, 3) rotasyon matrisi -> (yaw, pitch, roll) [radyan].
    euler_zyx_to_matrix'in tam tersi (scipy ile dogrulanmis standart formul).
    Sadece LOGLAMA/METRIK icin kullanilir; egitim gradyani bu fonksiyondan
    GECMEZ (gimbal-lock durumunda turev kararsizligi egitimi etkilemesin diye).
    """
    yaw = torch.atan2(Rm[..., 1, 0], Rm[..., 0, 0])
    pitch = torch.atan2(-Rm[..., 2, 0],
                         torch.sqrt(Rm[..., 2, 1] ** 2 + Rm[..., 2, 2] ** 2).clamp_min(eps))
    roll = torch.atan2(Rm[..., 2, 1], Rm[..., 2, 2])
    return yaw, pitch, roll


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(..., 6) ham cikti -> (..., 3, 3) gecerli rotasyon matrisi.
    Zhou et al. (CVPR 2019) Gram-Schmidt tabanli surekli (continuous)
    parametrizasyon. Quaternion'daki q/-q double-cover sureksizligine sahip
    DEGILDIR; bu yuzden regresyon gorevlerinde ampirik olarak daha dusuk
    hataya ulasir."""
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    a2_proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2 - a2_proj, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # sutunlar b1,b2,b3 -> (...,3,3)


def matrix_to_rotation_6d(Rm: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6). Rm'nin ilk iki sutununu dogrudan alir (Zhou
    et al.'daki gibi); Gram-Schmidt zaten b1,b2'yi tekrar normalize edecegi
    icin bu, gecerli bir 6D temsildir. Test/karsilastirma amacli sunulur."""
    return torch.cat([Rm[..., :, 0], Rm[..., :, 1]], dim=-1)


# =============================================================================
# 1a-bis) ATTITUDE (NED/govde) <-> KAMERA-PROJEKSIYON (dunya-ENU) EKSEN DONUSUMU
# =============================================================================
#
# KRITIK BULGU (bu sohbette test edilerek duzeltildi): Basit bir transpoz
# YETERSIZDI. Iki AYRI eksen kurgusu var:
#   1) `euler_zyx_to_matrix`in urettigi R_attitude = Rz(yaw)@Ry(pitch)@Rx(roll),
#      standart havacilik 3-2-1 DCM'idir ve (X,Y,Z) = (ileri/forward,
#      sag/right, asagi/down) -- yani NED-benzeri bir govde/dunya ekseni
#      varsayar (bu, scipy 'ZYX' ile dogrulanan MATEMATIKSEL formuldur, ama
#      hangi fiziksel eksenin X/Y/Z oldugu ayri bir konu).
#   2) `default_runway_object_points`/`get_runway_object_points`in urettigi
#      pist kose noktalari ISE (X,Y,Z) = (sag/right, ileri/forward,
#      yukari/up) -- yani ENU-benzeri bir dunya ekseni kullanir.
#   3) Kamera/OpenCV projeksiyon konvansiyonu ise (X,Y,Z) =
#      (sag/right, asagi/down, ileri/forward) bekler (K'nin standart pinhole
#      formulu bunu varsayar).
#
# Bu ucu de birbirine baglamak icin IKI SABIT (yaw/pitch/roll'a BAGLI
# OLMAYAN) rotasyon matrisi tanimlanir:
#   M : NED_gorgu <- ENU_dunya  (v_NED = M @ v_ENU)
#   Q : Kamera_ekseni <- Govde_ekseni  (v_cam = Q @ v_body)
# ve R_attitude (govde->NED) ile birlikte, kamera-to-world(ENU) rotasyonu:
#   R_cw = M^T @ R_attitude @ Q^T
# world-to-camera (reprojection icin gereken) ise: R_wc = R_cw^T
#
# BU KOMPOZISYON, bu sohbette SAYISAL OLARAK DOGRULANMISTIR: yaw=pitch=roll=0
# ve kamera pistin arkasinda/yukarisinda konumlandirildiginda, UZAK esik
# koseleri (TR,TL) goruntude DAHA YUKARIDA (kucuk y) ve YAKIN esik koseleri
# (BL,BR) DAHA ASAGIDA (buyuk y) cikiyor -- gercek fotografla (runway_31.jpg)
# TUTARLI. Negatif pitch (burun asagi) verildiginde pist goruntude yukari
# kayiyor -- fizik olarak da beklenen davranis.
#
# !!! Yine de bu, TEK bir gercek etiketli (goruntu + bilinen R,t) ornek
# olmadan TAM DOGRULANAMAZ -- sadece FIZIKSEL TUTARLILIK testinden gecti.
# Elinizde gercek bir R,t etiketli ornek olursa MUTLAKA capraz kontrol edin.
_NED_FROM_ENU = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
_CAM_FROM_BODY = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def attitude_to_world_to_camera(R_attitude: torch.Tensor) -> torch.Tensor:
    """R_attitude (..., 3, 3) [govde-to-NED, euler_zyx_to_matrix ciktisi] ->
    R_wc (..., 3, 3) [dunya(ENU/pist-lokal)-to-kamera, reproject_points'in
    bekledigi konvansiyon]. bkz. yukaridaki KRITIK BULGU notu."""
    M = _NED_FROM_ENU.to(device=R_attitude.device, dtype=R_attitude.dtype)
    Q = _CAM_FROM_BODY.to(device=R_attitude.device, dtype=R_attitude.dtype)
    R_cw = M.transpose(-2, -1) @ R_attitude @ Q.transpose(-2, -1)
    return R_cw.transpose(-2, -1)


# =============================================================================
# 1b) LARD V2 XPlane HAM ACI KONVANSIYONU -- KRITIK, DOGRULANMASI GEREKEN VARSAYIM
# =============================================================================
#
# CSV'deki ham yaw/pitch/roll sutunlari, STANDART havacilik konvansiyonuyla
# (yatay duzlemden burun yukari/asagi = pitch, vb.) DOGRUDAN eslesmiyor
# olabilir. Somut kanit: ornek satirda pitch_raw=86.85 derece -- eger bu
# standart "yataydan burun acisi" olsaydi, kamera neredeyse DIKEY asagi
# bakiyor olurdu. Ama kullanicinin paylastigi gercek yaklasma goruntusunde
# (runway_31.jpg) kamera OLDUKCA LEVEL bir acidan, pistin ucuna dogru
# bakiyor -- dikeye yakin DEGIL. Bu, ham pitch'in muhtemelen "zeninden
# olcum" ya da benzer bir referansla (90 derece = level/yatay ucus, 0 derece
# = tam asagi) verildigini gosteriyor.
#
# Destekleyici ikinci kanit: pitch_raw - 90 = 86.85 - 90 = -3.15 derece --
# bu deger, AYNI SATIRDAKI vertical_path_angle=3.51 (tipik ~3 derecelik
# glideslope) ile ISARET VE BUYUKLUK OLARAK TUTARLI (burun hafifce asagi,
# glideslope'a yakin bir pitch attitude -- fizik olarak beklenen davranis).
#
# BU YUZDEN varsayilan donusum su sekilde uygulanir:
#       pitch_standart = pitch_raw - PITCH_OFFSET_DEG   (varsayilan offset=90)
#       yaw_standart   = yaw_raw   - YAW_OFFSET_DEG     (varsayilan offset=0)
#       roll_standart  = roll_raw  - ROLL_OFFSET_DEG    (varsayilan offset=0)
#
# !!! ONEMLI: Bu SADECE TEK BIR ORNEK SATIR + FIZIKSEL MUHAKEMEYE dayanan bir
# HIPOTEZDIR, kesinlik iddia edilmiyor. EGITIME BASLAMADAN ONCE MUTLAKA:
#   1) train.py'deki `sanity_check_angle_convention()` fonksiyonunu TUM
#      (ya da buyukce bir orneklem) veri uzerinde calistirin.
#   2) pitch_raw degerlerinin gercekten ~90 civarinda yogunlastigini (yani
#      offset'in TUM veri icin gecerli, tek bir satiya ozgu olmadigini)
#      dogrulayin.
#   3) Miktarindan eminseniz LARD_ANGLE_OFFSETS'i degistirin/onaylayin.
# Yanlis bir konvansiyonla egitilen model, mimari ne kadar iyi olursa olsun
# SISTEMATIK OLARAK yanlis ogrenir -- bu, dogruluk hedefine ulasmadaki EN
# BUYUK risktir, model kapasitesinden cok daha onemlidir.
LARD_ANGLE_OFFSETS_DEG = {"yaw": 0.0, "pitch": 90.0, "roll": 0.0}


def lard_raw_angles_to_R(yaw_raw_deg: torch.Tensor, pitch_raw_deg: torch.Tensor, roll_raw_deg: torch.Tensor,
                          offsets: Dict[str, float] = None) -> torch.Tensor:
    """LARD CSV'sindeki ham yaw/pitch/roll (derece) sutunlarini standart ZYX
    (Rz(yaw)@Ry(pitch)@Rx(roll)) rotasyon matrisine cevirir. bkz. yukaridaki
    KRITIK not -- offsets varsayilan olarak LARD_ANGLE_OFFSETS_DEG kullanir."""
    if offsets is None:
        offsets = LARD_ANGLE_OFFSETS_DEG
    yaw = torch.deg2rad(yaw_raw_deg - offsets["yaw"])
    pitch = torch.deg2rad(pitch_raw_deg - offsets["pitch"])
    roll = torch.deg2rad(roll_raw_deg - offsets["roll"])
    return euler_zyx_to_matrix(yaw, pitch, roll)


# =============================================================================
# 2) Initial Convolutional Block (ICB), Bolum 3.2
# =============================================================================

def make_norm(norm_type: str, channels: int, group_norm_groups: int = 32) -> nn.Module:
    """BatchNorm2d ya da GroupNorm dondurur. GroupNorm icin grup sayisi,
    group_norm_groups'u asmayacak sekilde kanal sayisinin bir boleni secilir
    (kucuk kanal sayilarinda -- ornegin bottleneck'in mid_channels=64 -- hata
    vermemesi icin)."""
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    elif norm_type == "group":
        g = min(group_norm_groups, channels)
        while channels % g != 0:
            g -= 1
        return nn.GroupNorm(num_groups=g, num_channels=channels)
    else:
        raise ValueError(f"Bilinmeyen norm_type: {norm_type}")


class InitialConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type: str = "group", group_norm_groups: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = make_norm(norm_type, out_channels, group_norm_groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = make_norm(norm_type, out_channels, group_norm_groups)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x  # (B, 64, H/4, W/4) -- her H,W icin gecerli (fully-conv)


# =============================================================================
# 3) CNN Encoder — Bottleneck residual bloklari (Table 1'e gore DUZELTILDI)
# =============================================================================

class Bottleneck(nn.Module):
    """1x1 down-proj -> BN -> ReLU -> 3x3 (stride=s) -> BN -> ReLU
    -> 1x1 up-proj -> BN -> (+ shortcut) -> ReLU
    Eq.(10)'daki C1 (identity shortcut) / C2 (1x1-projeksiyonlu shortcut)
    otomatik secilir."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, reduction: int = 4,
                 norm_type: str = "group", group_norm_groups: int = 32):
        super().__init__()
        mid_channels = max(out_channels // reduction, 1)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = make_norm(norm_type, mid_channels, group_norm_groups)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = make_norm(norm_type, mid_channels, group_norm_groups)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = make_norm(norm_type, out_channels, group_norm_groups)
        self.act = nn.ReLU(inplace=True)

        needs_projection = (stride != 1) or (in_channels != out_channels)
        if needs_projection:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                make_norm(norm_type, out_channels, group_norm_groups),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.act(out + identity)


class CNNStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, stride: int, reduction: int,
                 norm_type: str = "group", group_norm_groups: int = 32):
        super().__init__()
        blocks = [Bottleneck(in_channels, out_channels, stride=stride, reduction=reduction,
                              norm_type=norm_type, group_norm_groups=group_norm_groups)]
        for _ in range(num_blocks - 1):
            blocks.append(Bottleneck(out_channels, out_channels, stride=1, reduction=reduction,
                                      norm_type=norm_type, group_norm_groups=group_norm_groups))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# =============================================================================
# 4) Transformer Encoder — ViT tarzi, Bolum 3.4 / Figure 7
#    Pos-embed, farkli img_size'lar icin interpolate edilir (esneklik icin).
# =============================================================================

class PatchEmbed(nn.Module):
    """F_initial (herhangi bir H,W) -> 1x1 conv (kanal->embed_dim) ->
    ADAPTIVE AVERAGE POOLING ile SABIT (grid_size x grid_size) grid'e
    oturtma -> flatten. Bu sayede token sayisi img_size'dan bagimsiz,
    SABIT kalir (bkz. LandNetConfig.patch_grid_size aciklamasi -- OOM
    duzeltmesi icin gerekliydi)."""

    def __init__(self, in_channels: int, embed_dim: int, grid_size: int):
        super().__init__()
        self.grid_size = grid_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)
        x = F.adaptive_avg_pool2d(x, output_size=(self.grid_size, self.grid_size))
        x = x.flatten(2).transpose(1, 2)  # (B, grid_size*grid_size, E)
        return x, (self.grid_size, self.grid_size)


def interpolate_pos_embed(pos_embed: torch.Tensor, old_grid: Tuple[int, int], new_grid: Tuple[int, int]) -> torch.Tensor:
    """Ogrenilen positional embedding'i FARKLI bir patch_grid_size
    konfigurasyonuna (ornegin eski bir checkpoint'i yeni bir grid_size ile
    devam ettirmek icin) bicubic interpolation ile uydurur. NOT: patch_grid_size
    artik img_size'dan bagimsiz SABIT oldugu icin, ayni egitim/inference
    kosusu icinde bu fonksiyonun cagrilmasina GEREK YOKTUR -- sadece
    config degisikligiyle checkpoint tasirken elle kullanilir (bkz.
    LandNet.resize_pos_embed_for_new_grid)."""
    if old_grid == new_grid:
        return pos_embed
    B, N, E = pos_embed.shape
    assert N == old_grid[0] * old_grid[1]
    pe = pos_embed.reshape(1, old_grid[0], old_grid[1], E).permute(0, 3, 1, 2)
    pe = F.interpolate(pe, size=new_grid, mode="bicubic", align_corners=False)
    pe = pe.permute(0, 2, 3, 1).reshape(1, new_grid[0] * new_grid[1], E)
    return pe


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return out


class TransformerMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop: float, attn_drop: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = TransformerMLP(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# 5) Feature Interactive Block (FIB) — Bolum 3.5 / Figure 8 (YON DUZELTILDI)
# =============================================================================

class FIB_CNNtoTrans(nn.Module):
    """CNN(local) -> Transformer(global): downsample + 1x1 conv (kanal->E)
    + flatten + CONCATENATION (metinde acikca "concatenation" deniyor) +
    concat-sonrasi boyutu tekrar E'ye indiren linear projeksiyon (Figure 8'de
    acikca cizilmemis ama sabit boyut icin gerekli bir standart adim)."""

    def __init__(self, cnn_channels: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(cnn_channels, embed_dim, kernel_size=1)
        self.merge = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, cnn_feat: torch.Tensor, tokens: torch.Tensor, patch_hw: Tuple[int, int]) -> torch.Tensor:
        h_p, w_p = patch_hw
        aligned = F.adaptive_avg_pool2d(cnn_feat, output_size=(h_p, w_p))
        aligned = self.proj(aligned).flatten(2).transpose(1, 2)     # (B, N, E) -- local bilgi
        mixed = torch.cat([tokens, aligned], dim=-1)                 # concat -> (B, N, 2E) : "Mixed Global feature"
        return self.norm(self.merge(mixed))


class FIB_TransToCNN(nn.Module):
    """Transformer(global) -> CNN(local): upsample + 1x1 conv (E->C) +
    ELEMENT-WISE ADDITION (metinde bu yon icin acikca "addition" deniyor)."""

    def __init__(self, embed_dim: int, cnn_channels: int, norm_type: str = "group", group_norm_groups: int = 32):
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, cnn_channels, kernel_size=1)
        self.bn = make_norm(norm_type, cnn_channels, group_norm_groups)

    def forward(self, tokens: torch.Tensor, patch_hw: Tuple[int, int], cnn_feat: torch.Tensor) -> torch.Tensor:
        B, N, E = tokens.shape
        h_p, w_p = patch_hw
        grid = tokens.transpose(1, 2).reshape(B, E, h_p, w_p)
        grid = self.bn(self.proj(grid))
        grid = F.interpolate(grid, size=cnn_feat.shape[-2:], mode="bilinear", align_corners=False)
        return cnn_feat + grid                                        # element-wise addition -> "Mixed Local feature"


class FeatureInteractiveBlock(nn.Module):
    def __init__(self, cnn_channels: int, embed_dim: int, norm_type: str = "group", group_norm_groups: int = 32):
        super().__init__()
        self.c2t = FIB_CNNtoTrans(cnn_channels, embed_dim)
        self.t2c = FIB_TransToCNN(embed_dim, cnn_channels, norm_type=norm_type, group_norm_groups=group_norm_groups)

    def forward(self, cnn_feat: torch.Tensor, tokens: torch.Tensor, patch_hw: Tuple[int, int]):
        new_tokens = self.c2t(cnn_feat, tokens, patch_hw)
        new_cnn_feat = self.t2c(tokens, patch_hw, cnn_feat)
        return new_cnn_feat, new_tokens


# =============================================================================
# 6) Attentional ConvTrans Fusion Block (ACFB) — Eq.(14)-(15) / Figure 9
#    (SELF-GATING olarak DUZELTILDI -- capraz-gate DEGIL)
# =============================================================================

class ACFB(nn.Module):
    """
    f_s: CBAM-tarzi UZAMSAL attention, F_CNN'DEN hesaplanir, F_CNN'i gate'ler.
         Spatial = sigmoid(conv7x7(concat(mean_channel(F_CNN), max_channel(F_CNN))))   [Eq.15]
    f_c: SENet-tarzi KANAL attention, F_Transformer'DAN hesaplanir,
         F_Transformer'i gate'ler.
    F_out = Concat(F_CNN (x) f_s, F_Transformer (x) f_c)                              [Eq.14]
    """

    def __init__(self, cnn_channels: int, embed_dim: int, fusion_channels: int, se_reduction: int = 4):
        super().__init__()
        self.cnn_proj = nn.Conv2d(cnn_channels, fusion_channels, kernel_size=1)
        self.trans_proj = nn.Conv2d(embed_dim, fusion_channels, kernel_size=1)

        # f_s: CBAM spatial attention (mean+max over kanal ekseni -> concat -> conv7x7 -> sigmoid)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        # f_c: SENet channel attention
        se_hidden = max(fusion_channels // se_reduction, 1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fusion_channels, se_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_hidden, fusion_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, cnn_feat: torch.Tensor, tokens: torch.Tensor, patch_hw: Tuple[int, int]) -> torch.Tensor:
        B, N, E = tokens.shape
        h_p, w_p = patch_hw
        trans_grid = tokens.transpose(1, 2).reshape(B, E, h_p, w_p)

        f_cnn = self.cnn_proj(cnn_feat)                                             # (B, F, H, W)
        f_trans = self.trans_proj(trans_grid)                                        # (B, F, h_p, w_p)
        f_trans = F.interpolate(f_trans, size=f_cnn.shape[-2:], mode="bilinear", align_corners=False)

        # --- f_s: F_CNN'DEN hesaplanan uzamsal self-attention (Eq.15) ---
        mean_map = f_cnn.mean(dim=1, keepdim=True)
        max_map = f_cnn.max(dim=1, keepdim=True).values
        f_s = torch.sigmoid(self.spatial_conv(torch.cat([mean_map, max_map], dim=1)))  # (B,1,H,W)

        # --- f_c: F_Transformer'DAN hesaplanan kanal self-attention (SENet) ---
        f_c = self.channel_gate(f_trans)                                              # (B,F,1,1)

        gated_cnn = f_cnn * f_s          # F_CNN (x) f_s
        gated_trans = f_trans * f_c      # F_Transformer (x) f_c

        return torch.cat([gated_cnn, gated_trans], dim=1)  # Concat(...)  [Eq.14]


# =============================================================================
# 7) MLP Head — SADECE rotasyon (6D), pozisyon head'i KALDIRILDI
# =============================================================================

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# 8) LandNet — tum parcalari birlestiren ana model (rotasyon-only, esnek boyut)
# =============================================================================

class LandNet(nn.Module):
    """
    Ileri gecis:
        1) ICB          : img -> F_initial (64, H/4, W/4)
        2) Patch Embed   : F_initial -> tokens (B, N, E)  [pos-embed interpolate edilir]
        3) 4 stage boyunca CNN ilerler; ARDINDAN (4 stage arasinda TOPLAM 3 kere)
           FIB ile CNN<->Transformer bilgi alisverisi olur; Transformer ilerler.
        4) ACFB          : son F_cnn ve son tokens -> fused feature map
        5) GAP -> ortak gomme -> tek MLP head -> 6D rotasyon cikisi -> R (3x3)
    """

    def __init__(self, cfg: LandNetConfig):
        super().__init__()
        self.cfg = cfg

        self.icb = InitialConvBlock(cfg.in_channels, cfg.stem_channels,
                                     norm_type=cfg.norm_type, group_norm_groups=cfg.group_norm_groups)
        self.patch_embed = PatchEmbed(cfg.stem_channels, cfg.embed_dim, grid_size=cfg.patch_grid_size)

        g = cfg.patch_grid_size
        self._grid = (g, g)
        self.pos_embed = nn.Parameter(torch.zeros(1, g * g, cfg.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        cnn_in = cfg.stem_channels
        n_stages = len(cfg.cnn_stage_blocks)
        blocks_per_stage = cfg.transformer_depth // n_stages

        self.cnn_stages = nn.ModuleList()
        self.trans_stage_blocks = nn.ModuleList()
        self.fibs = nn.ModuleList()  # n_stages - 1 tane (stage'ler ARASINDA)

        for i, (out_ch, n_blocks, stride) in enumerate(
                zip(cfg.cnn_stage_channels, cfg.cnn_stage_blocks, cfg.cnn_stage_strides)):
            self.cnn_stages.append(CNNStage(cnn_in, out_ch, n_blocks, stride, cfg.bottleneck_reduction,
                                             norm_type=cfg.norm_type, group_norm_groups=cfg.group_norm_groups))
            self.trans_stage_blocks.append(nn.ModuleList([
                TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.drop_rate, cfg.attn_drop_rate)
                for _ in range(blocks_per_stage)
            ]))
            if i < n_stages - 1:
                self.fibs.append(FeatureInteractiveBlock(out_ch, cfg.embed_dim,
                                                           norm_type=cfg.norm_type, group_norm_groups=cfg.group_norm_groups))
            cnn_in = out_ch

        self.acfb = ACFB(cfg.cnn_stage_channels[-1], cfg.embed_dim, cfg.fusion_channels)

        fused_dim = cfg.fusion_channels * 2
        self.rotation_head = MLPHead(fused_dim, cfg.head_hidden, 6, cfg.head_dropout)  # 6D rotasyon

    def resize_pos_embed_for_new_grid(self, new_grid_size: int):
        """Farkli bir patch_grid_size ile devam etmek istediginizde (ornegin
        onceden 20x20 ile egitilmis bir checkpoint'i 32x32 ile fine-tune
        etmek icin) pos_embed'i bicubic interpolation ile yeniden boyutlandirir.
        Ayni kosu icinde CAGIRMANIZA GEREK YOKTUR (grid sabit kalir)."""
        new_pe = interpolate_pos_embed(self.pos_embed.data, self._grid, (new_grid_size, new_grid_size))
        self.pos_embed = nn.Parameter(new_pe)
        self._grid = (new_grid_size, new_grid_size)
        self.patch_embed.grid_size = new_grid_size

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        f_cnn = self.icb(img)
        tokens, patch_hw = self.patch_embed(f_cnn)
        tokens = tokens + self.pos_embed

        use_ckpt = self.cfg.use_grad_checkpoint and self.training

        n_stages = len(self.cnn_stages)
        for i in range(n_stages):
            if use_ckpt:
                f_cnn = torch.utils.checkpoint.checkpoint(self.cnn_stages[i], f_cnn, use_reentrant=False)
            else:
                f_cnn = self.cnn_stages[i](f_cnn)
            for blk in self.trans_stage_blocks[i]:
                if use_ckpt:
                    tokens = torch.utils.checkpoint.checkpoint(blk, tokens, use_reentrant=False)
                else:
                    tokens = blk(tokens)
            if i < n_stages - 1:  # FIB, sadece stage'ler ARASINDA (toplam n_stages-1 kere)
                f_cnn, tokens = self.fibs[i](f_cnn, tokens, patch_hw)

        fused = self.acfb(f_cnn, tokens, patch_hw)
        pooled = F.adaptive_avg_pool2d(fused, 1).flatten(1)

        rot6d = self.rotation_head(pooled)          # (B, 6)
        R = rotation_6d_to_matrix(rot6d)             # (B, 3, 3)

        return {"rot6d": rot6d, "R": R}


# =============================================================================
# 9) Kamera / Pist geometrisi -- refinement icin gerekli yardimcilar
# =============================================================================

def build_intrinsics(fx: float, fy: float, cx: float, cy: float, device=None, dtype=None) -> torch.Tensor:
    """(3,3) pinhole kamera intrinsic matrisi K."""
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
    return K


def intrinsics_from_fov(hfov_deg: float, width: float, height: float, device=None, dtype=None) -> torch.Tensor:
    """Yatay FOV (derece) + goruntu boyutundan simetrik pinhole K turetir
    (kalibrasyon dosyasi yoksa yaklasik olarak kullanilabilir)."""
    hfov = math.radians(hfov_deg)
    fx = (width / 2.0) / math.tan(hfov / 2.0)
    fy = fx  # kare piksel varsayimi
    cx, cy = width / 2.0, height / 2.0
    return build_intrinsics(fx, fy, cx, cy, device=device, dtype=dtype)


DEFAULT_RUNWAY_WIDTH_M = 45.0     # runways_db_V2_XPlane.json'daki 980 pist-ucu ortalamasi ~46.5m (bkz. gelistirme notu asagida)
DEFAULT_RUNWAY_LENGTH_M = 3000.0  # ayni istatistikte ortalama uzunluk ~2991m


def default_runway_object_points(width_m: float = DEFAULT_RUNWAY_WIDTH_M,
                                  length_m: float = DEFAULT_RUNWAY_LENGTH_M,
                                  device=None, dtype=None) -> torch.Tensor:
    """FALLBACK: gercek runways_db_V2_XPlane.json verisi bulunamadiginda
    (havaalani/pist eslesmedi ya da db hic verilmedi) kullanilan idealize
    dikdortgen pist kose noktalari. Varsayilan width/length degerleri, asagidaki
    get_runway_object_points() ile JSON'daki TUM 980 pist-ucu uzerinde
    dogrulanmis ortalama genislik (~46.5m) ve uzunluga (~2991m) dayanir.

    Nokta sirasi: TR, TL, BL, BR -- LARD CSV'sindeki x_TR,y_TR,x_TL,y_TL,
    x_BL,y_BL,x_BR,y_BR sutun sirasiyla BIREBIR eslesir (bkz.
    get_runway_object_points() dokumantasyonundaki T/B=uzak/yakin esik,
    L/R=sol/sag aciklamasi). Orijin = yakin esik orta noktasi, Y ekseni
    pist boyunca ileri (yakin->uzak), X ekseni saga, Z=0 (duzlemsel pist).
    """
    half_w = width_m / 2.0
    pts = torch.tensor([
        [half_w, length_m, 0.0],    # TR (uzak-sag)
        [-half_w, length_m, 0.0],   # TL (uzak-sol)
        [-half_w, 0.0, 0.0],        # BL (yakin-sol)
        [half_w, 0.0, 0.0],         # BR (yakin-sag)
    ], device=device, dtype=dtype)
    return pts  # (4, 3)


def load_runway_db(json_path: str) -> dict:
    """runways_db_V2_XPlane.json'u yukler. Yapisi: {AIRPORT: {RUNWAY: {'A':
    {...}, 'B':..., 'C':..., 'D':...}}}, her kose icin 'position' (ECEF
    x,y,z, metre) ve 'coordinate' (lat,lon,altitude, WGS84) var."""
    with open(json_path, "r") as f:
        return json.load(f)


def _corner_ecef(corner_entry: dict) -> torch.Tensor:
    p = corner_entry["position"]
    # ECEF koordinatlari ~1e6-1e7 metre mertebesinde; iki yakin nokta
    # (ör. A-B, ~30-70m ayrık) farkini alirken float32'nin ~7 basamak
    # hassasiyeti (5e6 civarinda ~0.5m mutlak hataya denk gelir) YETERSIZ
    # kalir. Bu yuzden ham ECEF hesaplari float64 ile yapilir; sadece son
    # (kucuk, yerel-cerceve) sonuc float32'ye dondurulur.
    return torch.tensor([p["x"], p["y"], p["z"]], dtype=torch.float64)


def runway_geometry_from_db(db: dict, airport: str, runway: str) -> Optional[Dict[str, object]]:
    """
    runways_db_V2_XPlane.json'dan (airport, runway) icin GERCEK 3B pist
    geometrisini cikarir.

    ================== DOGRULANMIS BULGULAR (bu sohbette test edildi) ==================
    980 pist-ucunun TAMAMI uzerinde (sifir anomaliyle) dogrulandi:
      - A, B = UZAK (kalkis yonundeki/far) esik cifti
      - C, D = YAKIN (bu 'runway' girdisinin esigi/near, dokunma noktasi) cifti
      Genislik (|A-B| = |C-D|): min 27.5m, max 72.6m, ORTALAMA 46.5m (gercekci).
      Uzunluk (|A-C| = |B-D|):  min 1087m, max 4882m, ORTALAMA 2991m (gercekci).

      Near/far atamasi, (C,D)->( A,B) yon vektorunun ENU pusula acisiyla,
      runway isminden turetilen beklenen heading (ör. "8L" -> ~080 derece)
      karsilastirilarak DOGRULANDI: KATL'da ortalama ~7 derece, FACT'ta
      ~25 derece fark -- bu tamamen MANYETIK SAPMA (declination) ile
      aciklanabilir buyuklukte (Cape Town/FACT icin tarihsel declination
      ~25 derece Bati, Atlanta/KATL icin ~5-6 derece Bati ile TUTARLI).
      Yani bu bir isaret/siralama hatasi DEGIL, beklenen fiziksel bir etki.

      Sol/sag (L/R) atamasi forward x up capraz carpimiyla (ENU-benzeri
      yerel cerceve, sag-el kurali) hesaplanir; test edilen orneklerde
      (FACT/01, KATL/8L, KATL/26L) TUTARLI cikti (C=SOL, D=SAG, A=SAG,
      B=SOL), ama YINE DE her cagride YENIDEN hesaplanir (kutuplara yakin
      ya da olagandisi bir pistte bu paternin bozulma ihtimaline karsi
      genel/saglam bir yontem, sabit kodlanmis bir kural degil).
    =====================================================================================

    Donus (bulunursa): {'width': float, 'length': float,
                         'points_TRTLBLBR': (4,3) float32 tensor}
    Bulunamazsa (airport/runway/kose eksikse): None.

    Nokta sirasi TR,TL,BL,BR -- LARD CSV sutun sirasiyla (x_TR,y_TR,...,
    x_BR,y_BR) BIREBIR eslesir. Yerel cerceve: orijin=yakin esik orta
    noktasi, X=sag, Y=ileri (pist boyunca, yakin->uzak), Z=yukari (duzlemsel
    pist varsayimiyla Z=0 alinir -- runway'in kendi hafif egimi/kavis payi
    bu olcekte ihmal edilebilir).
    """
    airport_data = db.get(airport)
    if airport_data is None:
        return None
    entry = airport_data.get(str(runway))
    if entry is None or not all(k in entry for k in "ABCD"):
        return None

    A = _corner_ecef(entry["A"])
    B = _corner_ecef(entry["B"])
    C = _corner_ecef(entry["C"])
    D = _corner_ecef(entry["D"])

    near_mid = (C + D) / 2.0
    far_mid = (A + B) / 2.0
    forward = far_mid - near_mid
    length = torch.linalg.norm(forward)
    forward = forward / length

    up = near_mid / torch.linalg.norm(near_mid)     # yerel radyal "yukari" yaklasimi (pist olceginde yeterli, bkz. not)
    right = torch.cross(forward, up, dim=-1)
    right = right / torch.linalg.norm(right)
    up = torch.cross(right, forward, dim=-1)          # yeniden ortogonalize (temiz sag-el cerceve icin)

    width = torch.linalg.norm(C - D)

    # NOT: local cercevenin +X ekseni zaten "right" olarak TANIMLANDI (cross
    # product ile), bu yuzden TR/BR'yi +half_w'de, TL/BL'yi -half_w'de
    # yerlestirmek YETERLI ve dogru -- C/D'nin veya A/B'nin ayri ayri hangi
    # fiziksel koseye karsilik geldigini tekrar hesaplamaya gerek yok (bu,
    # docstring'teki dogrulama sirasinda bir kez ayrica test edilmisti).
    half_w = width / 2.0
    zero = torch.zeros_like(length)
    TR = torch.stack([half_w, length, zero])
    TL = torch.stack([-half_w, length, zero])
    BL = torch.stack([-half_w, zero, zero])
    BR = torch.stack([half_w, zero, zero])
    points = torch.stack([TR, TL, BL, BR], dim=0).to(torch.float32)

    return {"width": float(width), "length": float(length), "points_TRTLBLBR": points}


def normalize_runway_id(runway: str) -> List[str]:
    """LARD CSV'sindeki 'runway' sutunu bazen SIFIR-DOLGUSUZ gelebiliyor
    (ornegin '1'), ama runways_db_V2_XPlane.json anahtarlari SIFIR-DOLGULU
    ('01'). Bu fonksiyon, DENENECEK aday anahtarlarin bir listesini dondurur
    (once en olasi, sonra alternatifler) -- boylece '1' -> '01' gibi
    eslesmeler SESSIZCE fallback'e dusmez.
    NOT: Bu, bu sohbette CSV orneginde 'runway'=1 iken JSON'da '01' anahtari
    goruldugu icin eklenen somut bir duzeltmedir (aksi halde HER FACT/1
    ornegi sessizce fallback genislik/uzunluk kullanirdi -- fark edilmesi
    zor, sessiz bir dogruluk kaybi kaynagi olurdu)."""
    runway = str(runway).strip().upper()
    candidates = [runway]
    # sayisal + opsiyonel L/R/C harfi ayikla, sayisal kismi 2 haneye tamamla
    digits = "".join(c for c in runway if c.isdigit())
    suffix = "".join(c for c in runway if c.isalpha())
    if digits:
        padded = digits.zfill(2) + suffix
        if padded not in candidates:
            candidates.append(padded)
        # bazi kayitlarda sondaki L/R/C olmayabilir ya da fazladan olabilir
        if digits.zfill(2) not in candidates:
            candidates.append(digits.zfill(2))
    return candidates


def get_runway_object_points(airport: str, runway: str, db: Optional[dict] = None,
                              device=None, dtype=None) -> Tuple[torch.Tensor, Dict[str, object]]:
    """
    Ana giris noktasi: once (db verilmisse) GERCEK runways_db_V2_XPlane.json
    geometrisini dener; airport/runway bulunamazsa ya da db=None ise
    DEFAULT_RUNWAY_WIDTH_M / DEFAULT_RUNWAY_LENGTH_M ile FALLBACK yapar.

    Kullanim (egitim dongusunde, ornek):
        db = load_runway_db("runways_db_V2_XPlane.json")   # bir kere yuklenir
        ...
        points_3d, info = get_runway_object_points(row.airport, row.runway, db=db,
                                                     device=device, dtype=torch.float32)
        if info["source"] == "default_fallback":
            logger.warning(f"{row.airport}/{row.runway} DB'de yok, fallback kullanildi")

    Donus: (points (4,3) tensor [TR,TL,BL,BR sirasiyla], info dict)
    info = {'width': float, 'length': float, 'source': 'runways_db_V2_XPlane' | 'default_fallback'}
    """
    if db is not None:
        for rwy_candidate in normalize_runway_id(runway):
            geo = runway_geometry_from_db(db, airport, rwy_candidate)
            if geo is not None:
                pts = geo["points_TRTLBLBR"].to(device=device, dtype=dtype)
                return pts, {"width": geo["width"], "length": geo["length"], "source": "runways_db_V2_XPlane"}

    pts = default_runway_object_points(DEFAULT_RUNWAY_WIDTH_M, DEFAULT_RUNWAY_LENGTH_M, device=device, dtype=dtype)
    return pts, {"width": DEFAULT_RUNWAY_WIDTH_M, "length": DEFAULT_RUNWAY_LENGTH_M, "source": "default_fallback"}


def solve_translation_given_rotation(R: torch.Tensor, K: torch.Tensor,
                                      points_3d: torch.Tensor, points_2d: torch.Tensor,
                                      ridge: float = 1e-3) -> torch.Tensor:
    """R sabitken (network'ten geliyor), bilinen 3B-2B eslesmelerinden
    translation t'yi KAPALI-FORM (linear least-squares, DLT tarzi) cozer.
    Boylece network t'yi hic ogrenmek zorunda kalmaz; t geometriden turetilir.

    AMP / bfloat16 / float16 altinda linalg.solve dtype uyumsuzlugunu ve
    sayisal kararsizligi onlemek icin tum girdiler float32'ye zorlanir.
    """
    orig_dtype = R.dtype
    device = R.device
    with torch.autocast(device_type=device.type, enabled=False):
        R_f = R.float()
        K_f = K.float()
        p3d_f = points_3d.float()
        p2d_f = points_2d.float()

        B = R_f.shape[0]

        if K_f.dim() == 2:
            K_f = K_f.unsqueeze(0).expand(B, -1, -1)
        if p3d_f.dim() == 2:
            p3d_f = p3d_f.unsqueeze(0).expand(B, -1, -1)

        M = torch.einsum("bij,bjk,bnk->bni", K_f, R_f, p3d_f)   # (B, N, 3) = K @ (R @ P_i)

        u = p2d_f[..., 0]  # (B, N)
        v = p2d_f[..., 1]

        K0 = K_f[:, 0:1, :]  # (B,1,3)
        K1 = K_f[:, 1:2, :]
        K2 = K_f[:, 2:3, :]

        A_u = K0 - u.unsqueeze(-1) * K2       # (B, N, 3)
        A_v = K1 - v.unsqueeze(-1) * K2       # (B, N, 3)
        A = torch.cat([A_u, A_v], dim=1)      # (B, 2N, 3)

        b_u = u * M[..., 2] - M[..., 0]       # (B, N)
        b_v = v * M[..., 2] - M[..., 1]       # (B, N)
        b = torch.cat([b_u, b_v], dim=1).unsqueeze(-1)   # (B, 2N, 1)

        AtA = A.transpose(-2, -1) @ A                                   # (B, 3, 3)
        Atb = A.transpose(-2, -1) @ b                                   # (B, 3, 1)
        eye = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0)

        diag_scale = AtA.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-8)  # (B,1)
        rel_ridge = (ridge * diag_scale).unsqueeze(-1)                  # (B,1,1)
        sol = torch.linalg.solve(AtA + rel_ridge * eye, Atb)            # (B, 3, 1)
        return sol.squeeze(-1).to(dtype=orig_dtype)


def reproject_points(R: torch.Tensor, t: torch.Tensor, K: torch.Tensor, points_3d: torch.Tensor) -> torch.Tensor:
    """R:(B,3,3) t:(B,3) K:(3,3)/(B,3,3) points_3d:(N,3)/(B,N,3) -> (B,N,2) piksel."""
    orig_dtype = R.dtype
    device = R.device
    with torch.autocast(device_type=device.type, enabled=False):
        R_f, t_f, K_f, p3d_f = R.float(), t.float(), K.float(), points_3d.float()
        B = R_f.shape[0]
        if K_f.dim() == 2:
            K_f = K_f.unsqueeze(0).expand(B, -1, -1)
        if p3d_f.dim() == 2:
            p3d_f = p3d_f.unsqueeze(0).expand(B, -1, -1)

        cam_pts = torch.einsum("bij,bnj->bni", R_f, p3d_f) + t_f.unsqueeze(1)   # (B,N,3)
        proj = torch.einsum("bij,bnj->bni", K_f, cam_pts)                          # (B,N,3)
        uv = proj[..., :2] / proj[..., 2:3].clamp_min(1e-6)
        return uv.to(dtype=orig_dtype)


def reprojection_error(R: torch.Tensor, K: torch.Tensor, points_3d: torch.Tensor,
                        points_2d_gt: torch.Tensor, img_diag: Optional[torch.Tensor] = None
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """R sabitken t'yi kapali-form cozer, reproject eder ve piksel hatasini dondurur."""
    orig_dtype = R.dtype
    device = R.device
    with torch.autocast(device_type=device.type, enabled=False):
        R_f, K_f, p3d_f, p2d_f = R.float(), K.float(), points_3d.float(), points_2d_gt.float()
        diag_f = img_diag.float() if img_diag is not None else None
        t_f = solve_translation_given_rotation(R_f, K_f, p3d_f, p2d_f)
        proj = reproject_points(R_f, t_f, K_f, p3d_f)
        err = (proj - p2d_f).norm(dim=-1)   # (B, N)
        if diag_f is not None:
            err = err / diag_f.unsqueeze(-1)
        return err.to(dtype=orig_dtype), t_f.to(dtype=orig_dtype)


# =============================================================================
# 10) Loss — rotasyon (chordal/geodesic) + (opsiyonel) reprojection,
#     ogrenilen belirsizlik agirliklandirmasi (Eq.17'nin ruhuna sadik,
#     pozisyon+quaternion yerine rotasyon+reprojection'a genisletilmis hali)
# =============================================================================

class LandNetLoss(nn.Module):
    def __init__(self, cfg: LandNetConfig, init_s_r: float = 0.0, init_s_p: float = 0.0):
        super().__init__()
        self.cfg = cfg
        self.s_r = nn.Parameter(torch.tensor(float(init_s_r)))
        self.s_p = nn.Parameter(torch.tensor(float(init_s_p)))

    @staticmethod
    def chordal_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        """Frobenius (chordal) rotasyon hatasi -- geodesic'e gore turevi
        sifira yakin bolgede kararli (sub-0.1 derece hedefi icin onemli;
        arccos tabanli geodesic loss'un turevi 1'e yaklastikca patlar)."""
        diff = R_pred - R_gt
        return diff.pow(2).sum(dim=(-2, -1)).mean()

    @staticmethod
    def geodesic_angle_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        """Sadece IZLEME/METRIK icin: iki rotasyon arasi jeodezik aci (derece).
        Gradyan icin KULLANILMAZ (bkz. chordal_loss)."""
        RtR = torch.einsum("bij,bik->bjk", R_pred, R_gt)  # R_pred^T @ R_gt
        trace = RtR[..., 0, 0] + RtR[..., 1, 1] + RtR[..., 2, 2]
        cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        return torch.rad2deg(torch.acos(cos_theta))

    def forward(self, pred: Dict[str, torch.Tensor], R_gt: torch.Tensor,
                K: Optional[torch.Tensor] = None,
                points_3d: Optional[torch.Tensor] = None,
                points_2d_gt: Optional[torch.Tensor] = None,
                img_diag: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        R_pred = pred["R"]

        if self.cfg.rotation_loss_type == "chordal":
            loss_rot = self.chordal_loss(R_pred, R_gt)
        elif self.cfg.rotation_loss_type == "geodesic":
            # Not: egitimde nadiren kullanilir (turev kararliligi nedeniyle);
            # burada radyan cinsinden, differentiable sekilde hesaplanir.
            RtR = torch.einsum("bij,bik->bjk", R_pred, R_gt)
            trace = RtR[..., 0, 0] + RtR[..., 1, 1] + RtR[..., 2, 2]
            cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            loss_rot = torch.acos(cos_theta).mean()
        else:
            raise ValueError(f"Bilinmeyen rotation_loss_type: {self.cfg.rotation_loss_type}")

        out = {
            "loss_rotation": loss_rot,
            "geodesic_deg": self.geodesic_angle_deg(R_pred, R_gt).mean().detach(),
        }

        use_reproj = (self.cfg.use_reprojection_loss and K is not None
                      and points_3d is not None and points_2d_gt is not None)
        if use_reproj:
            # R_pred, attitude (govde-to-NED) konvansiyonunda; reprojection icin
            # world(ENU)-to-camera gerekir (bkz. attitude_to_world_to_camera notu).
            R_pred_w2c = attitude_to_world_to_camera(R_pred)
            err, t_solved = reprojection_error(R_pred_w2c, K, points_3d, points_2d_gt,
                                                img_diag=img_diag if self.cfg.reproj_norm_by_diag else None)
            out["reprojection_error_raw_mean"] = err.mean().detach()  # clamp'siz, ham deger -- sadece izleme/teshis
            err_clamped = err.clamp(max=self.cfg.reprojection_max_error)
            if self.cfg.reprojection_loss_type == "huber":
                loss_reproj = F.huber_loss(err_clamped, torch.zeros_like(err_clamped),
                                            delta=self.cfg.reprojection_huber_delta)
            elif self.cfg.reprojection_loss_type == "l2":
                loss_reproj = err_clamped.mean()
            else:
                raise ValueError(f"Bilinmeyen reprojection_loss_type: {self.cfg.reprojection_loss_type}")
            out["loss_reprojection"] = loss_reproj
            out["reprojection_error_px_mean"] = err_clamped.mean().detach()  # clamp'li, egitimde GERCEKTEN kullanilan deger
            out["t_solved"] = t_solved.detach()
            total = (loss_rot * torch.exp(-self.s_r) + self.s_r
                     + loss_reproj * torch.exp(-self.s_p) + self.s_p)
        else:
            total = loss_rot * torch.exp(-self.s_r) + self.s_r

        out["loss"] = total
        out["s_r"] = self.s_r.detach()
        out["s_p"] = self.s_p.detach()
        return out


# =============================================================================
# 11) Post-hoc Geometrik Refinement (inference sonrasi)
#     LandNet -> 6D -> R -> (K, 3B pist noktalari, GOZLENEN 2B kose noktalari)
#     -> reprojection hatasi -> R'yi birkac adimda iyilestir -> Refined R
# =============================================================================

@torch.enable_grad()
def refine_rotation(R_init: torch.Tensor, K: torch.Tensor, points_3d: torch.Tensor,
                     points_2d_obs: torch.Tensor, num_steps: int = 100, lr: float = 1e-3,
                     img_diag: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Egitilmis network'un urettigi R_init'i (attitude/body-to-world
    konvansiyonunda), GOZLENEN 2B kose noktalariyla (train/eval sirasinda
    CSV'den, gercek dunyada bir kose-tespit modulunden gelebilir)
    reprojection hatasini minimize edecek sekilde iyilestirir.

    R, 6D temsil uzerinden optimize edilir (surekli/gimbal-lock'suz oldugu
    icin optimizasyon acisindan da quaternion/Euler'e gore daha kararli).
    Network agirliklari SABIT tutulur; sadece poz (R) degisir.

    ONEMLI: R_init/R_refined hep ATTITUDE (body-to-world) konvansiyonunda
    tutulur/dondurulur (LandNet'in geri kalaniyla tutarli olsun diye); ancak
    reproject_points/reprojection_error world-to-camera bekledigi icin,
    ic hesaplamada TRANSPOZU kullanilir (bkz. reproject_points docstring).

    Not: real-time olmasi gerekmiyor (kullanicinin belirttigi gibi); Adam ile
    birkac-yuz adim CPU/GPU'da saniyenin cok altinda surer (kucuk 3x3 problem).

    BILINEN SINIRLAMA (bu sohbette gozlemlendi): R_init cok kotu (ornegin
    tamamen egitilmemis/rastgele agirlikli bir modelden, gercek degerden
    ~90 derece+ uzak) oldugunda, gradyan tabanli bu yerel optimizasyon
    ARA SIRA kotu bir yerel minimuma takilabilir (reprojection hatasi
    beklenenden yuksek kalir). Bu, PnP-refinement literaturunde bilinen,
    yerel (non-convex) optimizasyonun dogasindan kaynaklanan bir durumdur --
    kod hatasi degildir. PRATIKTE bu risk dusuktur, cunku refinement GENELDE
    egitilmis (zaten birkac derece dogruluga yakin) bir modelin ciktisina
    uygulanir, tamamen rastgele agirliklara degil. Yakinsama sorunu
    gozlemlerseniz num_steps'i artirmayi (ornegin 300-500) ya da farkli bir
    lr denemeyi dusunun.
    """
    d6 = matrix_to_rotation_6d(R_init).clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([d6], lr=lr)
    history = []
    for step in range(num_steps):
        optimizer.zero_grad()
        R = rotation_6d_to_matrix(d6)
        R_w2c = attitude_to_world_to_camera(R)
        err, t = reprojection_error(R_w2c, K, points_3d, points_2d_obs, img_diag=img_diag)
        loss = err.mean()
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    with torch.no_grad():
        R_refined = rotation_6d_to_matrix(d6)
        err_final, t_final = reprojection_error(attitude_to_world_to_camera(R_refined), K, points_3d, points_2d_obs, img_diag=img_diag)

    return {
        "R_refined": R_refined.detach(),
        "t_refined": t_final.detach(),
        "reprojection_error_px": err_final.detach(),
        "loss_history": history,
    }


# =============================================================================
# 12) Metrikler — pitch/yaw/roll icin MAE/RMSE (derece), guzel loglama
# =============================================================================

def _angle_diff_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Iki aci (radyan) arasindaki farki, +-180 derece sarmalanma sorunu
    olmadan hesaplar."""
    d = a - b
    d = (d + math.pi) % (2 * math.pi) - math.pi
    return torch.rad2deg(d)


class PoseMetricLogger:
    """pitch/yaw/roll icin MAE/RMSE (derece) biriktirir ve guzel bir string
    olarak raporlar. Egitim/dogrulama dongusunde her epoch sonunda kullanin:

        logger = PoseMetricLogger()
        for batch in loader:
            ...
            logger.update(R_pred, R_gt)
        print(logger.report())
        logger.reset()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._abs_errors = {"yaw": [], "pitch": [], "roll": []}
        self._geodesic = []

    @torch.no_grad()
    def update(self, R_pred: torch.Tensor, R_gt: torch.Tensor):
        yaw_p, pitch_p, roll_p = matrix_to_euler_zyx(R_pred)
        yaw_g, pitch_g, roll_g = matrix_to_euler_zyx(R_gt)

        self._abs_errors["yaw"].append(_angle_diff_deg(yaw_p, yaw_g).abs())
        self._abs_errors["pitch"].append(_angle_diff_deg(pitch_p, pitch_g).abs())
        self._abs_errors["roll"].append(_angle_diff_deg(roll_p, roll_g).abs())
        self._geodesic.append(LandNetLoss.geodesic_angle_deg(R_pred, R_gt))

    def compute(self) -> Dict[str, float]:
        out = {}
        for axis in ("yaw", "pitch", "roll"):
            errs = torch.cat(self._abs_errors[axis])
            out[f"{axis}_mae_deg"] = errs.mean().item()
            out[f"{axis}_rmse_deg"] = errs.pow(2).mean().sqrt().item()
        geo = torch.cat(self._geodesic)
        out["geodesic_mae_deg"] = geo.mean().item()
        out["geodesic_rmse_deg"] = geo.pow(2).mean().sqrt().item()
        return out

    def report(self) -> str:
        m = self.compute()
        return (
            f"  yaw   -> MAE: {m['yaw_mae_deg']:.4f} deg | RMSE: {m['yaw_rmse_deg']:.4f} deg\n"
            f"  pitch -> MAE: {m['pitch_mae_deg']:.4f} deg | RMSE: {m['pitch_rmse_deg']:.4f} deg\n"
            f"  roll  -> MAE: {m['roll_mae_deg']:.4f} deg | RMSE: {m['roll_rmse_deg']:.4f} deg\n"
            f"  geodesic (toplam) -> MAE: {m['geodesic_mae_deg']:.4f} deg | RMSE: {m['geodesic_rmse_deg']:.4f} deg"
        )


# =============================================================================
# 13) Duman testi (smoke test)
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device}")

    # --- Gercek pist veritabani (varsa) yuklenir; smoke test bunu FACT/01
    # ile gosterir, ayni zamanda bilinmeyen bir havaalani/pist icin
    # FALLBACK mekanizmasini da test eder. ---
    runway_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runways_db_V2_XPlane.json")
    try:
        runway_db = load_runway_db(runway_db_path)
        print(f"Pist veritabani yuklendi: {len(runway_db)} havaalani.")
    except FileNotFoundError:
        runway_db = None
        print("Pist veritabani bulunamadi, tum testler FALLBACK genislik/uzunlukle calisacak.")

    pts_real, info_real = get_runway_object_points("FACT", "01", db=runway_db)
    print(f"FACT/01 gercek geometri: genislik={info_real['width']:.2f}m, "
          f"uzunluk={info_real['length']:.2f}m, kaynak={info_real['source']}")
    pts_fallback, info_fallback = get_runway_object_points("XXXX", "99", db=runway_db)
    print(f"Bilinmeyen havaalani (XXXX/99) fallback: genislik={info_fallback['width']:.2f}m, "
          f"uzunluk={info_fallback['length']:.2f}m, kaynak={info_fallback['source']}")
    assert info_real["source"] == "runways_db_V2_XPlane"
    assert info_fallback["source"] == "default_fallback"
    print("Runway DB + fallback mekanizmasi dogru calisiyor.\n")

    for test_img_size in (640, 1024):
        print(f"\n===== img_size={test_img_size} testi =====")
        # Not: use_grad_checkpoint=True sadece bu duman testinin sinirli
        # bellekli ortamda rahat calismasi icin; gercek egitiminizde 12GB
        # VRAM'de genelde kapali (False) daha hizli olur, batch_size'a gore ayarlayin.
        cfg = LandNetConfig(img_size=test_img_size, use_grad_checkpoint=(test_img_size >= 1024))
        model = LandNet(cfg).to(device)
        criterion = LandNetLoss(cfg).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"LandNet parametre sayisi: {n_params / 1e6:.2f}M")

        batch_size = 1 if test_img_size >= 1024 else 2
        img = torch.randn(batch_size, cfg.in_channels, test_img_size, test_img_size, device=device)

        # --- Sahte ground-truth (gercekte CSV'den: yaw,pitch,roll); batch_size'a gore olceklenir ---
        yaw_gt = torch.deg2rad(torch.tensor([5.0, -3.0][:batch_size], device=device))
        pitch_gt = torch.deg2rad(torch.tensor([-2.5, 1.2][:batch_size], device=device))
        roll_gt = torch.deg2rad(torch.tensor([0.5, -0.8][:batch_size], device=device))
        R_gt = euler_zyx_to_matrix(yaw_gt, pitch_gt, roll_gt)

        # FOV=60 derece varsayimi (kullanicinin belirttigi gibi) ve GERCEK FACT/01 pist geometrisi
        K = intrinsics_from_fov(hfov_deg=60.0, width=test_img_size, height=test_img_size, device=device)
        points_3d = pts_real.to(device=device)

        # Ground-truth pozu kullanarak sahte 2B kose gozlemleri uretelim (gercekte CSV'nin
        # x_TR,y_TR,x_TL,y_TL,x_BL,y_BL,x_BR,y_BR sutunlarindan gelir)
        t_gt_full = torch.tensor([[2.0, -50.0, 600.0], [-5.0, -40.0, 550.0]], device=device)
        t_gt = t_gt_full[:batch_size]
        points_2d_gt = reproject_points(attitude_to_world_to_camera(R_gt), t_gt, K, points_3d)
        img_diag = torch.full((batch_size,), math.sqrt(2) * test_img_size, device=device)

        # --- Forward ---
        pred = model(img)
        print("rot6d shape:", pred["rot6d"].shape, "| R shape:", pred["R"].shape)

        out = criterion(pred, R_gt, K=K, points_3d=points_3d, points_2d_gt=points_2d_gt, img_diag=img_diag)
        print("loss:", out["loss"].item())
        print("loss_rotation (chordal):", out["loss_rotation"].item())
        print("loss_reprojection (huber, egitimde kullanilan, clamp'li):", out["loss_reprojection"].item())
        print("reprojection_error_raw_mean (ham, clamp'siz, sadece izleme):", out["reprojection_error_raw_mean"].item())
        print("geodesic_deg (izleme):", out["geodesic_deg"].item())

        out["loss"].backward()
        print("Backward tamamlandi -- gradyanlar akti.")

        # --- Metrik loglama ---
        logger = PoseMetricLogger()
        logger.update(pred["R"].detach(), R_gt)
        print("Pose metrikleri (egitilmemis rastgele agirliklarla, sadece format ornegi):")
        print(logger.report())

        # --- Post-hoc refinement testi ---
        print("Post-hoc refinement calisiyor...")
        refine_out = refine_rotation(pred["R"].detach(), K, points_3d, points_2d_gt,
                                      num_steps=200, lr=5e-3, img_diag=img_diag)
        err_before = out["reprojection_error_raw_mean"].item()
        err_after = refine_out["reprojection_error_px"].mean().item()
        print(f"  reprojection error (once): {err_before:.4f} (norm. piksel)")
        print(f"  reprojection error (sonra): {err_after:.4f} (norm. piksel)")
        assert err_after <= err_before + 1e-4, "Refinement reprojection hatasini azaltmadi!"
        print("  Refinement reprojection hatasini basariyla azalti.")

    # --- Rotasyon yardimci fonksiyon dogrulugu ---
    print("\n===== Rotasyon donusum testleri =====")
    yaw = torch.tensor([0.3])
    pitch = torch.tensor([0.15])
    roll = torch.tensor([-0.2])
    Rm = euler_zyx_to_matrix(yaw, pitch, roll)
    yaw_b, pitch_b, roll_b = matrix_to_euler_zyx(Rm)
    print("Euler round-trip fark (yaw,pitch,roll):",
          (yaw - yaw_b).abs().item(), (pitch - pitch_b).abs().item(), (roll - roll_b).abs().item())

    d6 = matrix_to_rotation_6d(Rm)
    Rm2 = rotation_6d_to_matrix(d6)
    print("6D round-trip fark (Frobenius):", (Rm - Rm2).abs().max().item())

    print("\nTum testler basariyla tamamlandi.")