"""Draw the train/val accuracy curve (and the gap) from train_vs_val_curve.csv."""
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"C:\Users\Ed\Desktop\Recent Project\LGL-Lab-AIC")
rows = list(csv.DictReader((ROOT / "train_vs_val_curve.csv").open(encoding="utf-8")))
ep = [int(r["epoch"]) for r in rows]
tr = [float(r["train_acc"]) for r in rows]
va = [float(r["val_acc"]) for r in rows]
vam = [float(r["val_macro"]) for r in rows]
gap = [float(r["gap"]) for r in rows]

# 74.2132 (416+SE) single-view val_acc, HANDOFF §53.3
REF = {1: .6550, 2: .6966, 4: .7219, 5: .7240, 8: .7443, 10: .7593,
       15: .7767, 17: .7839, 18: .7857, 19: .7876, 20: .7877}

W, H, S = 1200, 720, 2
img = Image.new("RGB", (W * S, H * S), "white")
d = ImageDraw.Draw(img)


def font(sz):
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "consola.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, sz * S)
        except OSError:
            continue
    return ImageFont.load_default()


f_ti, f_su, f_lb, f_x, f_tk = font(17), font(11), font(13), font(11), font(10)


def T(*a, **k):
    pass


L, R = 78 * S, (W - 250) * S
TOP, BOT = 116 * S, (H - 78) * S
X0, X1, Y0, Y1 = 2, 20, 0.64, 0.90


def px(e):
    return L + (e - X0) / (X1 - X0) * (R - L)


def py(v):
    return BOT - (v - Y0) / (Y1 - Y0) * (BOT - TOP)


d.text((60 * S, 18 * S), "训练/验证准确率曲线 —— 500 类半决赛，416px，rank64+qkv+SE+local-head+local-attn",
       font=f_ti, fill=(22, 26, 32))
d.text((60 * S, 44 * S), "gate_init=0.02, augment=clip_color_erasing, bs72, 20 轮（ep19/20 因实例关机未完成）",
       font=f_su, fill=(110, 115, 125))
d.text((60 * S, 62 * S), "train_acc = 训练集分层抽样 5,997 张、与验证集同一 eval transform 实测；val_* = 训练日志",
       font=f_su, fill=(110, 115, 125))

for g in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
    d.line([(L, py(g)), (R, py(g))], fill=(232, 234, 238), width=S)
    s = f"{g:.2f}"
    d.text((L - 14 * S - d.textlength(s, font=f_x), py(g) - 7 * S), s, font=f_x, fill=(95, 100, 110))
for e in range(2, 21, 2):
    d.line([(px(e), BOT), (px(e), BOT + 5 * S)], fill=(130, 135, 145), width=S)
    s = str(e)
    d.text((px(e) - d.textlength(s, font=f_x) / 2, BOT + 9 * S), s, font=f_x, fill=(95, 100, 110))
d.line([(L, TOP - 8 * S), (L, BOT)], fill=(130, 135, 145), width=S)
d.line([(L, BOT), (R, BOT)], fill=(130, 135, 145), width=S)
lab = "epoch"
d.text(((L + R) / 2 - d.textlength(lab, font=f_lb) / 2, BOT + 30 * S), lab, font=f_lb, fill=(70, 75, 85))

# incumbent reference, dashed
pts = [(px(e), py(v)) for e, v in sorted(REF.items()) if X0 <= e <= X1]
for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
    for k in range(0, 11, 2):
        d.line([(x0 + (x1 - x0) * k / 11, y0 + (y1 - y0) * k / 11),
                (x0 + (x1 - x0) * (k + 1) / 11, y0 + (y1 - y0) * (k + 1) / 11)],
               fill=(150, 155, 165), width=S)


def poly(vals, color, w):
    d.line([(px(e), py(v)) for e, v in zip(ep, vals)], fill=color, width=w * S, joint="curve")


for e, t, v in zip(ep, tr, va):                      # the gap, drawn as thin verticals
    d.line([(px(e), py(t)), (px(e), py(v))], fill=(206, 211, 220), width=S)
poly(tr, (32, 106, 220), 3)
poly(va, (228, 122, 18), 3)
poly(vam, (198, 58, 138), 2)
for e, v in zip(ep, tr):
    d.ellipse([px(e) - 3.5 * S, py(v) - 3.5 * S, px(e) + 3.5 * S, py(v) + 3.5 * S], fill=(32, 106, 220))
for e, v in zip(ep, va):
    d.ellipse([px(e) - 3.5 * S, py(v) - 3.5 * S, px(e) + 3.5 * S, py(v) + 3.5 * S], fill=(228, 122, 18))

d.text((px(18) - 96 * S, py(tr[-1]) - 22 * S), f"train {tr[-1]:.4f}", font=f_lb, fill=(32, 106, 220))
d.text((px(18) - 78 * S, py(va[-1]) + 9 * S), f"val {va[-1]:.4f}", font=f_lb, fill=(228, 122, 18))

# legend
lx, ly = L + 18 * S, TOP + 6 * S
d.rectangle([lx - 8 * S, ly - 8 * S, lx + 168 * S, ly + 62 * S], fill=(255, 255, 255),
            outline=(226, 229, 234), width=S)
for i, (color, txt, w) in enumerate(((  (32, 106, 220), "train_acc (实测)", 3),
                                     ((228, 122, 18), "val_acc (日志)", 3),
                                     ((198, 58, 138), "val_macro (日志)", 2),
                                     ((150, 155, 165), "74.2132 参考 val_acc", 1))):
    yy = ly + i * 15 * S
    d.line([(lx, yy), (lx + 26 * S, yy)], fill=color, width=w * S)
    d.text((lx + 34 * S, yy - 7 * S), txt, font=f_tk, fill=(60, 65, 75))

# gap table on the right
gx = R + 26 * S
d.text((gx, TOP - 4 * S), "gap = train − val", font=f_lb, fill=(30, 34, 40))
d.line([(gx, TOP + 18 * S), (gx + 150 * S, TOP + 18 * S)], fill=(226, 229, 234), width=S)
for i, (e, t, v, g) in enumerate(zip(ep, tr, va, gap)):
    yy = TOP + (26 + i * 15) * S
    d.text((gx, yy), f"ep{e:>2}", font=f_tk, fill=(95, 100, 110))
    d.text((gx + 40 * S, yy), f"{g * 100:+.2f} pp", font=f_tk,
           fill=(170, 60, 40) if g > 0.095 else (60, 65, 75))
d.text((gx, TOP + (26 + len(ep) * 15 + 14) * S), "ep6 起稳定在 8.3~8.8 pp（ep4 为 7.5）",
       font=f_tk, fill=(60, 65, 75))
d.text((gx, TOP + (26 + len(ep) * 15 + 28) * S), "→ 未过拟合（过拟合时 gap 会持续变宽）",
       font=f_tk, fill=(60, 65, 75))

out = ROOT / "train_vs_val_curve.png"
img.resize((W, H), Image.LANCZOS).save(out)
print("wrote", out)
