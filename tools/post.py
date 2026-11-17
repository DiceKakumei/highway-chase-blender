"""EXR(リニア) -> 仕上げ(にじみ・トーンマップ・周辺減光・粒子・タイトル・暗転) -> PNG。
usage: python tools/post.py <exr_dir> <png_dir> [--params JSON] [--sheet sheet.jpg] [--notitle]
コマ番号 f はファイル名(0024.exr)から取り、t = f / 24。"""
import os, sys, json, glob
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

G = {
    'expo': 1.0,
    'bloom': 0.16, 'bloom_thr': 1.1,
    'contrast': 1.06, 'sat': 1.05,
    'vignette': 0.22, 'grain': 0.012,
    'lift': [0.0, 0.0, 0.004],
}
FONTS = ['/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
         'C:/Windows/Fonts/YuGothB.ttc', 'C:/Windows/Fonts/meiryob.ttc']


def sm(x):
    x = min(1.0, max(0.0, x)); return x * x * (3 - 2 * x)


def aces(x):
    x = x * 0.6
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def srgb(x):
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.maximum(x, 1e-8), 1 / 2.4) - 0.055)


def grade(lin, f):
    h, w = lin.shape[:2]; s = h / 720.0
    img = np.nan_to_num(lin, nan=0.0, posinf=50.0, neginf=0.0) * G['expo']
    img = np.clip(img, 0, 400)
    # bloom: bright parts spread at several radii (computed at half size)
    lum = img.max(axis=2, keepdims=True)
    bright = img * np.clip((lum - G['bloom_thr']) / np.maximum(lum, 1e-6), 0, 1)
    bright = np.minimum(bright, 60.0)
    small = cv2.resize(bright, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    acc = np.zeros_like(small)
    for sig, wt in ((2.0, 0.45), (6.0, 0.30), (16.0, 0.17), (40.0, 0.08)):
        acc += wt * cv2.GaussianBlur(small, (0, 0), sig * s)
    img = img + G['bloom'] * cv2.resize(acc, (w, h), interpolation=cv2.INTER_LINEAR)
    out = srgb(aces(img))
    # contrast around mid grey, saturation, slight cool lift in the shadows
    out = np.clip((out - 0.5) * G['contrast'] + 0.5, 0, 1)
    gray = out.mean(axis=2, keepdims=True)
    out = np.clip(gray + (out - gray) * G['sat'], 0, 1)
    out = out + np.array(G['lift'], np.float32) * (1 - out)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    out = out * (1 - G['vignette'] * np.clip(r2 / 2, 0, 1))[..., None]
    rs = np.random.RandomState(f)
    out = out + rs.normal(0, G['grain'], (h, w, 1)).astype(np.float32) * (1 - out) * 1.0
    return np.clip(out, 0, 1)


_title = {}


def title_layer(w, h):
    """左下の題字「追跡」(白、うすい影)。RGBA の PIL 画像。"""
    if (w, h) in _title:
        return _title[(w, h)]
    k = h / 1080.0
    font = None
    for p in FONTS:
        if os.path.exists(p):
            font = ImageFont.truetype(p, int(200 * k), index=0); small = ImageFont.truetype(p, int(40 * k), index=0); break
    lay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    if font is None:
        print('WARNING: no CJK font found; title skipped'); _title[(w, h)] = lay; return lay
    sh = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(sh)
    x0, y0 = int(110 * k), int(h - 150 * k - 290 * k)
    def text(draw, fill):
        x = x0
        for ch in '追跡':
            draw.text((x, y0), ch, font=font, fill=fill); x += int(draw.textlength(ch, font=font) + 14 * k)
        x = x0 + int(8 * k)
        for ch in 'HIGHWAY CHASE':
            draw.text((x, y0 + int(232 * k)), ch, font=small, fill=fill); x += int(draw.textlength(ch, font=small) + 20 * k)
    text(d, (0, 0, 0, 200))
    sh = sh.filter(ImageFilter.GaussianBlur(10 * k))
    lay = Image.alpha_composite(lay, sh)
    text(ImageDraw.Draw(lay), (255, 255, 255, 255))
    _title[(w, h)] = lay
    return lay


def finish(out, f, notitle=False):
    t = f / 24.0
    h, w = out.shape[:2]
    a = min((t - 0.5) / 0.5, (4.2 - t) / 0.7)
    a = min(1.0, max(0.0, a))
    if a > 0 and not notitle:
        lay = np.asarray(title_layer(w, h)).astype(np.float32) / 255.0
        al = lay[..., 3:4] * a
        out = out * (1 - al) + lay[..., :3] * al
    return out * (1 - sm((t - 26.0) / 3.8))


def main():
    args = sys.argv[1:]
    src, dst = args[0], args[1]
    sheet = None; notitle = False
    i = 2
    while i < len(args):
        if args[i] == '--params':
            G.update(json.loads(args[i + 1])); i += 2
        elif args[i] == '--sheet':
            sheet = args[i + 1]; i += 2
        elif args[i] == '--notitle':
            notitle = True; i += 1
        else:
            i += 1
    os.makedirs(dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, '[0-9][0-9][0-9][0-9].exr')))
    thumbs = []
    for p in files:
        f = int(os.path.basename(p)[:4])
        lin = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if lin is None:
            print('cannot read', p); continue
        lin = lin[..., :3][..., ::-1].astype(np.float32)
        out = finish(grade(lin, f), f, notitle)
        bgr = (np.clip(out[..., ::-1], 0, 1) * 255 + 0.5).astype(np.uint8)
        cv2.imwrite(os.path.join(dst, '%04d.png' % f), bgr)
        if sheet:
            th = cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.putText(th, 'f%d t=%.2f' % (f, f / 24.0), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            thumbs.append(th)
        if len(files) <= 40 or f % 48 == 0:
            print('post', f, 'mean', round(float(out.mean()), 3), flush=True)
    if sheet and thumbs:
        cols = 3
        while len(thumbs) % cols:
            thumbs.append(np.zeros_like(thumbs[0]))
        rows = [np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)]
        cv2.imwrite(sheet, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print('done', len(files), 'frames')


if __name__ == '__main__':
    main()
