"""reply-card_original.png から印刷用 PNG/PDF を一括生成する。"""
from pathlib import Path
from PIL import Image
import img2pdf

HERE = Path(__file__).parent
src = Image.open(HERE / 'reply-card_original.png').convert('RGB')
W, H = src.size


def render(name, pw, ph, cw, dpi, shift=0):
    """用紙 pw x ph mm の中央に幅 cw mm で絵を置く（shift mm だけ右へずらす）。"""
    ch = cw * H / W
    px = lambda mm: round(mm / 25.4 * dpi)
    canvas = Image.new('RGB', (px(pw), px(ph)), 'white')
    art = src.resize((px(cw), px(ch)), Image.LANCZOS)
    canvas.paste(art, ((px(pw) - px(cw)) // 2 + px(shift), (px(ph) - px(ch)) // 2))
    png = HERE / f'{name}.png'
    canvas.save(png, dpi=(dpi, dpi))
    layout = img2pdf.get_layout_fun((img2pdf.mm_to_pt(pw), img2pdf.mm_to_pt(ph)))
    png.with_suffix('.pdf').write_bytes(img2pdf.convert(str(png), layout_fun=layout))


def fit(pw, ph, margin):
    """上下左右 margin mm 以内に収まる絵の幅。"""
    return min((pw - 2 * margin) / W, (ph - 2 * margin) / H) * W


# 3cm 余白（用紙サイズ別）
for paper, (pw, ph) in {'A4': (297, 210), 'B5': (257, 182), 'A5': (210, 148), 'hagaki': (148, 100)}.items():
    render(f'reply-card_{paper}_margin3cm', pw, ph, fit(pw, ph, 30), 300)

# はがき
HW, HH = 148, 100
tb = lambda m: (HH - 2 * m) * W / H  # 上下余白 m mm になる絵の幅
render('reply-card_hagaki_margin5mm', HW, HH, fit(HW, HH, 5), 350)
render('reply-card_hagaki_margin3mm', HW, HH, fit(HW, HH, 3), 350)
render('reply-card_hagaki_side10mm', HW, HH, HW - 20, 350)
render('reply-card_hagaki_6mm', HW, HH, tb(6), 350)
render('reply-card_hagaki_4mm', HW, HH, tb(4), 350)

# 会社の複合機の手差しで左に 2mm ずれる分を右へ補正
render('reply-card_hagaki_4mm_shift-right2mm', HW, HH, tb(4), 350, shift=2)
render('reply-card_hagaki_6mm_shift-right2mm', HW, HH, tb(6), 350, shift=2)
render('reply-card_hagaki_margin3mm_shift-right2mm', HW, HH, fit(HW, HH, 3), 350, shift=2)
