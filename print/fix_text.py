"""reply-card_source.png（元画像）の文字修正をして reply-card_original.png を作る。

文字（周りのうっすらした光彩ごと）を背景との差分として切り出し、
位置を動かしてから背景に戻す。元の位置は周囲の紙の色と質感で埋める。
"""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).parent
img = np.asarray(Image.open(HERE / 'reply-card_source.png').convert('RGB')).astype(float)
rng = np.random.default_rng(0)


def edit(y0, y1, x0, x1, pieces, drop=None):
    """領域内の文字を pieces=[(xa, xb, dx), ...] の列範囲ごとに dx だけ動かす。
    drop(mask_fn) で消したいピクセルの差分を 0 にできる。"""
    band = img[y0:y1, x0:x1].copy()
    lum = band.mean(2)
    # 背景推定: 文字＋光彩を除いた紙の色をなめらかに補間
    rough = ndimage.median_filter(lum, size=15)
    glyph = ndimage.binary_dilation((lum < rough - 12) | (lum > rough + 6), iterations=3)
    w = (~glyph).astype(float)
    den = np.maximum(ndimage.gaussian_filter(w, 5), 1e-6)
    smooth = np.stack([ndimage.gaussian_filter(band[..., c] * w, 5) for c in range(3)], -1) / den[..., None]
    noise_sd = (band[~glyph] - smooth[~glyph]).std(0) * 0.8
    tex = ndimage.gaussian_filter(rng.normal(size=lum.shape), 0.7)
    tex = tex / tex.std()
    bg = np.where(glyph[..., None], smooth + tex[..., None] * noise_sd, band)
    diff = np.where(glyph[..., None], band - smooth, 0.0)
    if drop is not None:
        diff[drop(lum, x0, y0)] = 0
    new = np.zeros_like(diff)
    for xa, xb, dx in pieces:
        seg = diff[:, xa - x0:xb - x0]
        dst = new[:, xa - x0 + dx:xb - x0 + dx]
        take = np.abs(seg).sum(2) > np.abs(dst).sum(2)
        dst[take] = seg[take]
    img[y0:y1, x0:x1] = bg + new


# 1. Northen -> Northern: 同じ行の "Nor" の r を写して挿入し、行を中央に保つ
edit(331, 380, 530, 1015, [(540, 707, -10), (621, 643, 709 - 623 - 10), (707, 1000, 11)])


# 2. "A days" -> "A day": s を消して中央に寄せ直す
def drop_s(lum, x0, y0):
    lab, _ = ndimage.label(lum < 225, structure=np.ones((3, 3)))
    objs = ndimage.find_objects(lab)
    pick = lambda lo, hi: [i + 1 for i, s in enumerate(objs)
                           if lo <= s[1].start + x0 < hi and (lab[s] == i + 1).sum() > 20][0]
    s_mask = ndimage.binary_dilation(lab == pick(1377, 1400), iterations=4)
    y_mask = ndimage.binary_dilation(lab == pick(1360, 1370), iterations=2)
    return s_mask & ~y_mask


edit(104, 147, 1290, 1410, [(1295, 1400, 5)], drop=drop_s)

# 3. "Deard Buena Vista  18:30 Start" が右に約38px寄っていたので中央へ
edit(382, 432, 430, 1120, [(480, 1110, -38)])

Image.fromarray(np.clip(img, 0, 255).round().astype(np.uint8)).save(HERE / 'reply-card_original.png')
