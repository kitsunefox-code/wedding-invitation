"""reply-card_source.png（元画像）の文字修正をして reply-card_original.png を作る。

文字（周りのうっすらした光彩ごと）を背景との差分として切り出し、
位置を動かしてから背景に戻す。元の位置は周囲の紙の色と質感で埋める。
"""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from PIL import ImageDraw, ImageFont

HERE = Path(__file__).parent
img = np.asarray(Image.open(HERE / 'reply-card_source.png').convert('RGB')).astype(float)
rng = np.random.default_rng(0)


def separate(y0, y1, x0, x1):
    """領域を「紙の背景」と「文字（光彩込み）の差分」に分ける。"""
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
    return bg, diff, lum


def paste(new, seg, at):
    """差分 seg を new の列 at から重ねる（濃い方を採用）。"""
    dst = new[:, at:at + seg.shape[1]]
    take = np.abs(seg).sum(2) > np.abs(dst).sum(2)
    dst[take] = seg[take]


def edit(y0, y1, x0, x1, pieces, drop=None):
    """領域内の文字を pieces=[(xa, xb, dx), ...] の列範囲ごとに dx だけ動かす。
    drop(mask_fn) で消したいピクセルの差分を 0 にできる。"""
    bg, diff, lum = separate(y0, y1, x0, x1)
    if drop is not None:
        diff[drop(lum, x0, y0)] = 0
    new = np.zeros_like(diff)
    for xa, xb, dx in pieces:
        paste(new, diff[:, xa - x0:xb - x0], xa - x0 + dx)
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

# 4. 返信はがきの敬語: 出席/欠席 -> ご出席/ご欠席、お名前 -> ご芳名
#    「ご」は同じはがきの「ご出欠」から写す。「芳」は同じ書体（Noto Serif JP Medium）で描く。
RA = (470, 514)          # お名前 の行
RB = (542, 586)          # ご出欠 の行（RA の 72px 下）
bgA, dA, _ = separate(*RA, 510, 640)
bgB, dB, _ = separate(*RB, 510, 1120)
go = dB[:, 521 - 510:553 - 510]           # 「ご出欠」の「ご」

newB = np.zeros_like(dB)
for xa, xb, dx in ((510, 620, 0), (686, 735, 0), (744, 822, 30), (943, 992, 0), (1000, 1080, 30)):
    paste(newB, dB[:, xa - 510:xb - 510], xa - 510 + dx)   # ラベル・四角はそのまま、出席/欠席は1文字分右へ
paste(newB, go, 747 - 510)                 # ご出席（「ご」の字面が元の「出」の位置 752 に来るように）
paste(newB, go, 1003 - 510)                # ご欠席
img[RB[0]:RB[1], 510:1120] = bgB + newB


def glyph_diff(ch, x0, y0, w, h, origin, size=33, ss=4):
    """文字 ch を背景との差分として描く（4倍で描いて縮小、元の文字と同じ淡い光彩も付ける）。"""
    font = ImageFont.truetype(str(HERE / 'fonts' / 'NotoSerifJP-Medium-subset.ttf'), size * ss)
    c = Image.new('L', (w * ss, h * ss), 0)
    ImageDraw.Draw(c).text(origin, ch, font=font, fill=255)
    a = np.clip(np.asarray(c.resize((w, h), Image.BOX)) / 255.0 * 1.1, 0, 1)  # 元の文字の太さに合わせる
    paper = bgA.mean((0, 1))
    ink = np.array([13.4, 27.0, 27.2])
    halo = np.clip(ndimage.gaussian_filter(ndimage.binary_dilation(a > 0.3, iterations=2).astype(float), 1.2) - a, 0, 1)
    return (ink - paper) * a[..., None] + 4.0 * halo[..., None]


newA = np.zeros_like(dA)
paste(newA, go, 521 - 510)                                   # ご（お の位置）
paste(newA, dA[:, 551 - 510:585 - 510], 551 - 510 + 34)      # 名 を3文字目へ
# 「芳」を 名 があった2文字目に。原点は既存の「名」に合わせて求めた値（字間を「ご出欠」とそろえて +2px）
paste(newA, glyph_diff('芳', 548, 470, 42, 44, (22, -12)), 548 - 510)
img[RA[0]:RA[1], 510:640] = bgA + newA

Image.fromarray(np.clip(img, 0, 255).round().astype(np.uint8)).save(HERE / 'reply-card_original.png')
