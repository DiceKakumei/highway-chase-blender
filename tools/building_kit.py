"""building_kit.py — 現実のビルに準拠した「ビルを作るキット」(Blender Python)。他の動画から import して使える。
依存: bpy・mathutils(Blender内のみ)・numpy・標準ライブラリ。

使い方
    import sys; sys.path.insert(0, '<このファイルのフォルダ>')
    import building_kit as bk
    b = bk.make_building('office', width=33, depth=26.4, floors=16, seed=3, detail='near', night=True, hour=20)
    b.location = (x, y, 0)

    kind   : 'office'(ガラスのカーテンウォール) / 'zakkyo'(小中オフィス・雑居ビル、タイル貼り) / 'mansion'(中層マンション、バルコニー)
    detail : 'near'(実立体: 窓の奥まり・サッシ・柱型・梁型・手すり) / 'mid'(柱型・腰壁・窓の奥まりは立体、サッシは模様)
             / 'far'(箱 + 手続き材質)
    night/hour : 夜の窓の点灯率と色温度(hour = 19〜21時で最大、23時以降は少なめ)
    戻り値 : 親Empty(子 = ビル本体のメッシュ)。原点 = 敷地の中心・地面(z=0)。+Z=上、正面(バルコニー側・メイン)= -Y 向き。
    幅・奥行きはモジュール(office 1.65m / zakkyo 3.9m / mansion 3.3m)の整数倍に丸める。実寸は b['kit_info'] (JSON文字列)。

仕様の根拠: リサーチ/2026-11-19_現実のビルの構造と汚れ_外観.md (★=出典あり、推=推定)。コード中の「★」「推」は同ファイルの区分。
ジオメトリ生成(build_geometry)は bpy 無しで動く(寸法・面数の数値確認用)。材質・オブジェクト化だけが bpy を使う。
"""
import math, random, json, time
import numpy as np

try:
    import bpy
except Exception:          # bpy 無しでも build_geometry は使える
    bpy = None

# ------------------------------------------------------------------------------------------------
# 仕様(調査の値)。単位 m
# ------------------------------------------------------------------------------------------------
SPECS = {
    # A. 中高層オフィス(カーテンウォール)
    'office': dict(label='オフィス(ガラスのカーテンウォール)', P=1.65, Hf=4.0, G=5.0, minb=(8, 6),     # P=横ピッチ(推1.5〜1.8)、Hf=階高(推約4.0)、G=1階
                   SILL=0.9, VIS=2.5,            # ビジョンガラス高(推2.4〜2.7)=2.5、腰=スパンドレル下0.9+上0.6=1.5(推1.3〜1.6)
                   MULL_W=0.10, MULL_D=0.20,     # マリオン見付け(推60〜120mm)=100mm、出(推150〜250mm)=200mm
                   REC=0.045,                    # ガラスはマリオン外面より30〜60mm奥(推)=45mm
                   FASC_OUT=0.06,                # 梁型(階ごとの水平帯)はマリオン外面より60mm出る(推)
                   COL_W=0.45, COL_OUT=0.15,     # 柱型(カバー)幅450mm、出150mm(推150〜250)。4スパン(6.6m)ごと
                   age=(0.10, 0.50)),
    # B. 小〜中規模オフィス・雑居ビル(タイル貼りRC)
    'zakkyo': dict(label='小中オフィス・雑居ビル(タイル貼り)', P=3.9, Hf=3.6, G=4.2, minb=(2, 2),    # 階高 推3.5〜3.8
                   PIL_W=0.70, PIL_OUT=0.20,     # 柱型 推600〜800mm角相当、外に150〜250mm
                   BEAM_H=0.70, BEAM_OUT=0.20,   # 梁型 推500〜800mmの帯
                   SILL=0.90, WIN_H=1.5,         # 腰壁高さ約0.9(推)。窓高: 推1.1は住宅腰窓の例。階高3.6に合わせ1.5を採用(ずれ)
                   JAMB=0.5,                     # 窓幅 = P - 2*JAMB = 2.9m(2枚引違い)
                   REC=0.12,                     # RC壁の奥まり80〜150mm(推) = 120mm
                   SASH_W=0.05, SASH_D=0.06,
                   age=(0.30, 0.90)),
    # C. 中層マンション(RC、バルコニー)
    'mansion': dict(label='中層マンション(バルコニー付き)', P=3.3, Hf=3.0, G=3.6, minb=(6, 4),        # 階高約3.0m★
                    SLAB=0.20, BAL=1.4,          # 床スラブ約200mm★、バルコニー出 推1.2〜1.5 = 1.4
                    RAIL_H=1.1,                  # 手すり高さ1.1m以上★
                    WIN_W=1.65, WIN_H=2.0, WIN_Z=0.05,   # 掃き出し窓 推1.65×2.0
                    REC=0.10,                    # RC壁の奥まり80〜150mm(推)
                    CORR=1.5,                    # 外廊下(背面)の出
                    age=(0.30, 0.80)),
}
KIND_ALIAS = {'オフィス': 'office', 'ガラスオフィス': 'office', '雑居ビル': 'zakkyo', '雑居': 'zakkyo', 'タイル': 'zakkyo',
              'マンション': 'mansion', 'office': 'office', 'zakkyo': 'zakkyo', 'mansion': 'mansion'}
RED_LIGHT_MIN_H = 60.0       # 高さ60m以上は赤い航空障害灯★(航空法51条)


def room_dims(kind):
    """窓1枚の(幅, 高さ, 枠幅, 中桟の数) = mid の「サッシ模様」用"""
    S = SPECS[kind]
    if kind == 'office':
        return (S['P'], S['VIS'], S['MULL_W'] / 2, 0)
    if kind == 'zakkyo':
        return (S['P'] - 2 * S['JAMB'], S['WIN_H'], S['SASH_W'], 1)
    return (S['WIN_W'], S['WIN_H'], 0.05, 1)


# ------------------------------------------------------------------------------------------------
# メッシュのため込み(bpy不要)
# ------------------------------------------------------------------------------------------------
class MD:
    """面をためる。頂点は共有しない。role=材質の役割名(あとで材質に対応づける)。"""
    def __init__(self):
        self.V = []; self.F = []; self.UV = []; self.WIN = []; self.MI = []; self.SM = []
        self.mats = []; self._m = {}

    def mi(self, role):
        if role not in self._m:
            self._m[role] = len(self.mats); self.mats.append(role)
        return self._m[role]

    def face(self, pts, role, uvs, win=(0.0, 0.0, 0.0, 0.0), smooth=False):
        b = len(self.V); n = len(pts)
        self.V.extend(pts); self.UV.extend(uvs)
        self.WIN.extend([win] * n)
        self.F.append(tuple(range(b, b + n))); self.MI.append(self.mi(role)); self.SM.append(bool(smooth))


class Fr:
    """ファサードの座標系。 P(a, z, d) = 壁面に沿って a、高さ z、外向きに d。 (a, z, d) は右手系(a×z = n)。"""
    def __init__(self, p0, t, n, L, uo=0.0):
        self.p0 = p0; self.t = t; self.n = n; self.L = L; self.uo = uo

    def P(self, a, z, d):
        return (self.p0[0] + self.t[0] * a + self.n[0] * d, self.p0[1] + self.t[1] * a + self.n[1] * d, z)


RF = Fr((0.0, 0.0), (1.0, 0.0), (0.0, -1.0), 1e6, 0.0)       # 世界座標の箱用: a=x, d=-y


def fquad(md, fr, loc, role, uvs=None, win=(0, 0, 0, 0)):
    """loc = [(a,z,d)...](外側から見て反時計回り)。 uvs 省略時は (a+uo, z) = メートルのUV(材質の汚れ・タイルがこれを使う)"""
    pts = [fr.P(*q) for q in loc]
    if uvs is None:
        uvs = [(q[0] + fr.uo, q[1]) for q in loc]
    md.face(pts, role, uvs, win)


def fbox(md, fr, a0, a1, z0, z1, d0, d1, role, skip=(), win=(0, 0, 0, 0)):
    """箱。skip = 出さない面('d+' 外向き,'d-' 壁側,'a+','a-','z+' 上,'z-' 下)。UV: 前面 (a,z)、側面 (d,z)、上下 (a,d)"""
    uo = fr.uo

    def add(code, loc, k):
        if code in skip:
            return
        pts = [fr.P(*q) for q in loc]
        if k == 'f':
            uvs = [(q[0] + uo, q[1]) for q in loc]
        elif k == 's':
            uvs = [(q[2] + uo, q[1]) for q in loc]
        else:
            uvs = [(q[0] + uo, q[2]) for q in loc]
        md.face(pts, role, uvs, win)
    add('d+', [(a0, z0, d1), (a1, z0, d1), (a1, z1, d1), (a0, z1, d1)], 'f')
    add('d-', [(a1, z0, d0), (a0, z0, d0), (a0, z1, d0), (a1, z1, d0)], 'f')
    add('a+', [(a1, z0, d1), (a1, z0, d0), (a1, z1, d0), (a1, z1, d1)], 's')
    add('a-', [(a0, z0, d0), (a0, z0, d1), (a0, z1, d1), (a0, z1, d0)], 's')
    add('z+', [(a0, z1, d1), (a1, z1, d1), (a1, z1, d0), (a0, z1, d0)], 't')
    add('z-', [(a0, z0, d0), (a1, z0, d0), (a1, z0, d1), (a0, z0, d1)], 't')


def wbox(md, x0, x1, y0, y1, z0, z1, role, skip=(), win=(0, 0, 0, 0)):
    """世界座標(軸平行)の箱。 skip の 'd+' = y小側(-Y面)、'd-' = y大側(+Y面)"""
    fbox(md, RF, x0, x1, z0, z1, -y1, -y0, role, skip, win)


# ------------------------------------------------------------------------------------------------
# 夜の窓の点灯
# ------------------------------------------------------------------------------------------------
def lit_rate(kind, hour, night=True):
    """時刻から点灯率。調査(推): オフィス 19〜21時=40〜60%、23時以降=10〜25%。マンション 19〜22時=30〜50%、深夜10〜20%。"""
    if not night:
        return 0.0
    h = hour % 24.0
    if kind == 'mansion':
        pts = [(0, 0.12), (5, 0.10), (6, 0.2), (17, 0.3), (19, 0.42), (22, 0.40), (23, 0.22), (24, 0.15)]
    elif kind == 'zakkyo':
        pts = [(0, 0.14), (5, 0.08), (7, 0.2), (17, 0.4), (19, 0.46), (21, 0.42), (23, 0.18), (24, 0.14)]
    else:
        pts = [(0, 0.16), (5, 0.10), (7, 0.2), (17, 0.4), (19, 0.50), (21, 0.50), (23, 0.20), (24, 0.16)]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return float(np.interp(h, xs, ys))


def make_lit(kind, ncell, floors, seed, hour, night):
    """窓ごとの点灯。 arr[f, cell] = (明るさ, 色温度/10000, ブラインド/カーテンの量, 色指標or1階のシャッター)。 明るさ0=消灯。
    cell = 周囲を正面から時計まわりに数えた通し番号(1セル=1モジュール幅)。"""
    rng = random.Random(int(seed) * 1013 + 7)
    r = lit_rate(kind, hour, night)
    arr = np.zeros((floors, ncell, 4), np.float32)
    if not night:                                   # 昼・夕方(night=False): 室内は見せない(窓はガラスの反射だけ)
        for c in range(ncell):
            arr[0, c, 3] = 1.0 if (kind == 'zakkyo' and rng.random() < 0.18) else 0.0
        return arr
    # 1階(ロビー・店舗): 夜はよく点く
    for c in range(ncell):
        if kind == 'office':
            on = rng.random() < (0.93 if night else 0.35)
            arr[0, c] = (1.5 * (0.9 + 0.2 * rng.random()), (4200 + 400 * rng.random()) / 1e4, 0.0, 0.0) if on else (0, 0, 0, 0)
        elif kind == 'zakkyo':
            shut = rng.random() < 0.18
            on = (not shut) and rng.random() < (0.78 if night else 0.3)
            warm = rng.random() < 0.3
            arr[0, c] = ((1.2 + 0.4 * rng.random()), ((2800 + 400 * rng.random()) if warm else (4000 + 1000 * rng.random())) / 1e4, 0.0, 1.0 if shut else 0.0) if on else (0, 0, 0, 1.0 if shut else 0.0)
        else:
            on = rng.random() < (0.85 if night else 0.3)
            arr[0, c] = (1.2 + 0.3 * rng.random(), (3000 + 1500 * rng.random()) / 1e4, 0.0, 0.0) if on else (0, 0, 0, 0)
    for f in range(1, floors):
        if kind == 'office':                       # フロア単位のまだら(推)
            pf = min(1.0, r / 0.855)
            floor_on = rng.random() < pf
            baseT = 4000 + 1000 * rng.random(); baseI = 0.85 + 0.35 * rng.random()
            c = 0
            while c < ncell:
                zl = rng.randint(4, 8)
                zone_on = floor_on and rng.random() < 0.9
                for k in range(c, min(ncell, c + zl)):
                    if zone_on and rng.random() < 0.95:
                        bl = rng.random() * 0.7 if rng.random() < 0.55 else 0.0
                        arr[f, k] = (baseI * (0.9 + 0.2 * rng.random()), (baseT + (rng.random() - 0.5) * 300) / 1e4, bl, 0.0)
                c += zl
        elif kind == 'zakkyo':                     # テナントごと(1〜3ベイ)
            c = 0
            while c < ncell:
                ln = rng.randint(1, 3)
                on = rng.random() < r / 0.95
                warm = rng.random() < 0.25
                T = (2700 + 500 * rng.random()) if warm else (3900 + 1200 * rng.random())
                I = 0.7 + 0.5 * rng.random(); bl = rng.random() * 0.8 if rng.random() < 0.5 else 0.0
                for k in range(c, min(ncell, c + ln)):
                    if on:
                        arr[f, k] = (I * (0.9 + 0.2 * rng.random()), T / 1e4, bl, 0.0)
                c += ln
        else:                                      # マンション: 住戸(2セル)ごとにランダム
            for u in range(0, ncell, 2):
                on = rng.random() < r
                if not on:
                    continue
                q = rng.random()
                T = (2700 + 300 * rng.random()) if q < 0.82 else ((3500 + 1500 * rng.random()) if q < 0.94 else 6200)
                I = (0.5 + 0.6 * rng.random()) * (0.55 if T > 6000 else 1.0)
                cur = 0.12 + 0.45 * rng.random(); hue = rng.random()
                for k in range(u, min(ncell, u + 2)):
                    if rng.random() < 0.88:
                        arr[f, k] = (I * (0.85 + 0.3 * rng.random()), T / 1e4, cur, hue)
    return arr


# ------------------------------------------------------------------------------------------------
# 窓まわりの部品
# ------------------------------------------------------------------------------------------------
class X:
    pass


def glass_role(ctx, lit, plain=False):
    """glass 系の役割名。 mid のときだけ、サッシ枠を模様(材質)で描く(窓の寸法を役割名に入れる)。"""
    base = 'glass_lit' if lit else 'glass'
    if ctx.detail == 'mid' and not plain:
        ww, wh, fw, nm = ctx.frame
        return '%s|%.3f|%.3f|%.3f|%d' % (base, ww, wh, fw, nm)
    return base


def wall_around(ctx, fr, a0, a1, z0, z1, wa0, wa1, wz0, wz1, d, role):
    """区画(a0..a1, z0..z1)から開口(wa0..wa1, wz0..wz1)を抜いた壁(最大4枚)"""
    md = ctx.md
    if wz0 > z0 + 1e-6:
        fquad(md, fr, [(a0, z0, d), (a1, z0, d), (a1, wz0, d), (a0, wz0, d)], role)
    if wz1 < z1 - 1e-6:
        fquad(md, fr, [(a0, wz1, d), (a1, wz1, d), (a1, z1, d), (a0, z1, d)], role)
    if wa0 > a0 + 1e-6:
        fquad(md, fr, [(a0, wz0, d), (wa0, wz0, d), (wa0, wz1, d), (a0, wz1, d)], role)
    if wa1 < a1 - 1e-6:
        fquad(md, fr, [(wa1, wz0, d), (a1, wz0, d), (a1, wz1, d), (wa1, wz1, d)], role)


def reveal(ctx, fr, a0, a1, z0, z1, rec, role='wall'):
    """窓の奥まり(抱き)。壁面(d=0)からガラス面(d=-rec)までの4面"""
    md = ctx.md
    uo = fr.uo
    fquad(md, fr, [(a0, z0, -rec), (a0, z1, -rec), (a0, z1, 0.0), (a0, z0, 0.0)], role, uvs=[(-rec, z0), (-rec, z1), (0, z1), (0, z0)])                    # 左(+a向き)
    fquad(md, fr, [(a1, z0, 0.0), (a1, z1, 0.0), (a1, z1, -rec), (a1, z0, -rec)], role, uvs=[(0, z0), (0, z1), (-rec, z1), (-rec, z0)])                    # 右(-a向き)
    fquad(md, fr, [(a0, z0, -rec), (a0, z0, 0.0), (a1, z0, 0.0), (a1, z0, -rec)], role, uvs=[(a0 + uo, rec), (a0 + uo, 0), (a1 + uo, 0), (a1 + uo, rec)])      # 下(窓台、+z向き)
    fquad(md, fr, [(a0, z1, 0.0), (a0, z1, -rec), (a1, z1, -rec), (a1, z1, 0.0)], role, uvs=[(a0 + uo, 0), (a0 + uo, rec), (a1 + uo, rec), (a1 + uo, 0)])      # 上(-z向き)


def glass_pane(ctx, fr, a0, a1, z0, z1, dg, li, plain=False):
    """ガラス1枚。 li = (明るさ,..) 点灯なら glass_lit。UV=窓内のメートル座標。"""
    lit = li is not None and li[0] > 0
    role = glass_role(ctx, lit, plain)
    fquad(ctx.md, fr, [(a0, z0, dg), (a1, z0, dg), (a1, z1, dg), (a0, z1, dg)], role,
          uvs=[(0, 0), (a1 - a0, 0), (a1 - a0, z1 - z0), (0, z1 - z0)])


def sash_frame(ctx, fr, a0, a1, z0, z1, dg, fw, dep, nmull, role='sash'):
    """窓枠(サッシ)。ガラス面 dg から手前へ dep 出る。 nmull = 縦の中桟の数"""
    md = ctx.md; d0 = dg; d1 = dg + dep
    fbox(md, fr, a0, a1, z0, z0 + fw, d0, d1, role, skip=('d-', 'z-'))
    fbox(md, fr, a0, a1, z1 - fw, z1, d0, d1, role, skip=('d-', 'z+'))
    fbox(md, fr, a0, a0 + fw, z0 + fw, z1 - fw, d0, d1, role, skip=('d-', 'a-'))
    fbox(md, fr, a1 - fw, a1, z0 + fw, z1 - fw, d0, d1, role, skip=('d-', 'a+'))
    for k in range(nmull):
        am = a0 + (a1 - a0) * (k + 1) / (nmull + 1)
        fbox(md, fr, am - fw * 0.6, am + fw * 0.6, z0 + fw, z1 - fw, d0, d1 - 0.01, role, skip=('d-',))


def room_for(ctx, fr, a0, a1, zfloor, zceil, zw0, zw1, dg, li, rkind, depth, rng):
    """点灯する窓の奥の部屋(奥行きのある板): 奥壁・天井(照明の並び)・床・左右の壁 + ガラスの内側のブラインド/カーテンの面。
    li = (明るさ, 色温度/1e4, ブラインド量, 色指標)。窓の外から見える範囲だけ意味がある。"""
    md = ctx.md
    win = (float(li[0]), float(li[1]), float(li[2]), float(li[3]))
    db = dg - depth; w = a1 - a0; h = zceil - zfloor
    fquad(md, fr, [(a0, zfloor, db), (a1, zfloor, db), (a1, zceil, db), (a0, zceil, db)], 'room_wall', uvs=[(0, 0), (w, 0), (w, h), (0, h)], win=win)
    fquad(md, fr, [(a0, zceil, dg), (a1, zceil, dg), (a1, zceil, db), (a0, zceil, db)], 'room_ceil_' + rkind, uvs=[(0, 0), (w, 0), (w, depth), (0, depth)], win=win)
    fquad(md, fr, [(a0, zfloor, db), (a1, zfloor, db), (a1, zfloor, dg), (a0, zfloor, dg)], 'room_floor', uvs=[(0, 0), (w, 0), (w, depth), (0, depth)], win=win)
    fquad(md, fr, [(a0, zfloor, dg), (a0, zfloor, db), (a0, zceil, db), (a0, zceil, dg)], 'room_wall', uvs=[(0, 0), (depth, 0), (depth, h), (0, h)], win=win)
    fquad(md, fr, [(a1, zfloor, db), (a1, zfloor, dg), (a1, zceil, dg), (a1, zceil, db)], 'room_wall', uvs=[(0, 0), (depth, 0), (depth, h), (0, h)], win=win)
    # ブラインド/カーテン面(ガラスの内側10cm)
    dgi = dg - 0.10; ww = a1 - a0; wh = zw1 - zw0
    fquad(md, fr, [(a0, zw0, dgi), (a1, zw0, dgi), (a1, zw1, dgi), (a0, zw1, dgi)], 'room_blind_' + rkind, uvs=[(0, wh), (ww, wh), (ww, 0), (0, 0)], win=win)   # v = 上端からの距離


# ------------------------------------------------------------------------------------------------
# 屋上
# ------------------------------------------------------------------------------------------------
def roof_items(ctx):
    md = ctx.md; rng = ctx.rng; W = ctx.W; D = ctx.D; H = ctx.H; kind = ctx.kind; det = ctx.detail
    hp = {'office': 1.0, 'zakkyo': 0.6, 'mansion': 0.5}[kind]       # パラペット高
    t = 0.25
    # 屋上面
    md.face([(-W / 2, -D / 2, H), (W / 2, -D / 2, H), (W / 2, D / 2, H), (-W / 2, D / 2, H)], 'roof',
            [(-W / 2, -D / 2), (W / 2, -D / 2), (W / 2, D / 2), (-W / 2, D / 2)])
    # パラペット(外面はファサードの材質=汚れが上端に出る)
    prole = 'beam' if kind == 'office' else 'wall'
    for i, fr in enumerate(ctx.frames):
        a0, a1 = (0.0, fr.L) if i in (0, 2) else (t, fr.L - t)
        fbox(md, fr, a0, a1, H, H + hp, -t, 0.0, prole, skip=('a-', 'a+', 'z-'))
    top = H                      # 設備は屋上面(H)に置く。パラペットの天端は H+hp
    rects = []

    def place(w, d, margin=1.3):
        if W - 2 * margin < w or D - 2 * margin < d:
            return None
        for _ in range(60):
            x = rng.uniform(-W / 2 + margin + w / 2, W / 2 - margin - w / 2); y = rng.uniform(-D / 2 + margin + d / 2, D / 2 - margin - d / 2)
            r = (x - w / 2 - 0.6, x + w / 2 + 0.6, y - d / 2 - 0.6, y + d / 2 + 0.6)
            if all(r[1] < q[0] or r[0] > q[1] or r[3] < q[2] or r[2] > q[3] for q in rects):
                rects.append(r); return x, y
        return None

    def box(x, y, w, d, h, role, z0=None, skip=()):
        zz = top if z0 is None else z0
        wbox(md, x - w / 2, x + w / 2, y - d / 2, y + d / 2, zz, zz + h, role, skip=skip)

    info = {'items': []}
    # 塔屋(階段室・EV機械室)
    pw, pd, ph = {'office': (min(W * 0.34, 14.0), min(D * 0.28, 9.0), 3.6), 'zakkyo': (min(W * 0.4, 3.4), min(D * 0.3, 2.8), 2.7),
                  'mansion': (min(W * 0.2, 6.0), min(D * 0.3, 3.8), 2.8)}[kind]
    p = place(pw, pd)
    pent_top = None
    if p:
        box(p[0], p[1], pw, pd, ph, 'pent', skip=('z-',))
        # 屋上のキャップ(わずかな庇)と扉
        wbox(md, p[0] - pw / 2 - 0.1, p[0] + pw / 2 + 0.1, p[1] - pd / 2 - 0.1, p[1] + pd / 2 + 0.1, top + ph, top + ph + 0.12, 'roof', skip=('z-',))
        if det != 'far':
            wbox(md, p[0] - 0.5, p[0] + 0.5, p[1] - pd / 2 - 0.02, p[1] - pd / 2, top, top + 2.0, 'door', skip=('d-', 'a-', 'a+', 'z-', 'z+'))
            if pw / 2 - 0.3 - 0.75 > 0.5:                                           # ガラリ(換気口)。扉と重ならない位置
                wbox(md, p[0] + 0.75, p[0] + pw / 2 - 0.3, p[1] - pd / 2 - 0.02, p[1] - pd / 2, top + ph - 1.2, top + ph - 0.4, 'dark', skip=('d-', 'a-', 'a+', 'z-', 'z+'))
        pent_top = (p[0], p[1], top + ph + 0.12)
        info['items'].append('pent')
    # キュービクル(受変電)
    cw, cd, ch = (4.0, 1.6, 2.4) if kind == 'office' else (2.0, 1.1, 1.9)
    p = place(cw, cd)
    if p and det != 'far':
        box(p[0], p[1], cw, cd, ch, 'equip', skip=('z-',))
        wbox(md, p[0] - cw / 2 + 0.2, p[0] + cw / 2 - 0.2, p[1] - cd / 2 - 0.02, p[1] - cd / 2, top + 0.5, top + ch - 0.5, 'dark', skip=('d-', 'a-', 'a+', 'z-', 'z+'))
        info['items'].append('cubicle')
    if kind == 'office':
        if det != 'far':
            for k in range(2):                            # 冷却塔
                p = place(2.6, 2.6)
                if p:
                    box(p[0], p[1], 2.6, 2.6, 3.0, 'equip2', skip=('z-',))
                    wbox(md, p[0] - 0.9, p[0] + 0.9, p[1] - 0.9, p[1] + 0.9, top + 3.0, top + 3.01, 'dark', skip=('z-',))
                    info['items'].append('cooling')
            p = place(5.0, 2.4)                           # チラー・空調機器の並び
            if p:
                for k in range(4):
                    box(p[0] - 1.8 + k * 1.2, p[1], 1.0, 1.0, 1.4, 'ac', skip=('z-',))
                info['items'].append('chiller')
    else:
        if det != 'far':
            tw = 2.2 if kind == 'zakkyo' else 3.0
            p = place(tw, tw)
            if p:                                          # 高架水槽(脚つき)
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        wbox(md, p[0] + sx * (tw / 2 - 0.15) - 0.07, p[0] + sx * (tw / 2 - 0.15) + 0.07, p[1] + sy * (tw / 2 - 0.15) - 0.07, p[1] + sy * (tw / 2 - 0.15) + 0.07, top, top + 1.1, 'equip2', skip=('z-',))
                box(p[0], p[1], tw, tw, 2.0, 'tank', z0=top + 1.1)
                info['items'].append('tank')
            if kind == 'zakkyo' or kind == 'mansion':
                p = place(4.6, 1.2)                        # 室外機の列
                if p:
                    for k in range(5):
                        box(p[0] - 1.8 + k * 0.9, p[1], 0.8, 0.35, 0.7, 'ac', skip=('z-',))
                        if det == 'near':
                            wbox(ctx.md, p[0] - 1.8 + k * 0.9 - 0.25, p[0] - 1.8 + k * 0.9 + 0.25, p[1] - 0.18, p[1] - 0.175, top + 0.12, top + 0.62, 'dark', skip=('d-', 'a-', 'a+', 'z-', 'z+'))
                    info['items'].append('ac_row')
    # アンテナ・避雷針
    nant = {'office': 1, 'zakkyo': 2, 'mansion': 2}[kind]
    if det != 'far':
        for k in range(nant):
            p = place(0.6, 0.6, margin=0.8)
            if not p:
                continue
            ph2 = {'office': 9.0, 'zakkyo': 3.5, 'mansion': 4.5}[kind]
            wbox(md, p[0] - 0.04, p[0] + 0.04, p[1] - 0.04, p[1] + 0.04, top, top + ph2, 'equip2', skip=('z-',))
            if kind != 'office':
                for j in range(3):
                    zz = top + ph2 * (0.55 + 0.15 * j); ln = 0.9 - 0.18 * j
                    wbox(md, p[0] - ln / 2, p[0] + ln / 2, p[1] - 0.015, p[1] + 0.015, zz, zz + 0.03, 'equip2', skip=())
            info['items'].append('antenna')
        if kind == 'office' and pent_top:
            wbox(md, pent_top[0] + pw / 2 - 0.9 - 0.03, pent_top[0] + pw / 2 - 0.9 + 0.03, pent_top[1] - 0.03, pent_top[1] + 0.03, pent_top[2], pent_top[2] + 7.0, 'equip2', skip=('z-',))
            info['items'].append('lightning_rod')
    # 航空障害灯(高さ60m以上は赤)★
    info['red_lights'] = 0
    if H + hp >= RED_LIGHT_MIN_H:
        ls = 0.45
        zl = (pent_top[2] + 0.0) if pent_top else top
        for (cx, cy) in [(-W / 2 + 0.6, -D / 2 + 0.6), (W / 2 - 0.6, -D / 2 + 0.6), (W / 2 - 0.6, D / 2 - 0.6), (-W / 2 + 0.6, D / 2 - 0.6)]:
            wbox(md, cx - ls / 2, cx + ls / 2, cy - ls / 2, cy + ls / 2, H + hp, H + hp + ls, 'lamp_red', skip=('z-',))
            info['red_lights'] += 1
        if pent_top:
            wbox(md, pent_top[0] - ls / 2, pent_top[0] + ls / 2, pent_top[1] - ls / 2, pent_top[1] + ls / 2, pent_top[2], pent_top[2] + ls, 'lamp_red', skip=('z-',))
            info['red_lights'] += 1
    return info


# ------------------------------------------------------------------------------------------------
# 種類ごとのファサード
# ------------------------------------------------------------------------------------------------
def hr(ctx, f, c, salt=0):
    """窓ごとの決定的な乱数 0..1(LODが変わっても同じ窓は同じ結果)"""
    x = (int(f) * 73856093) ^ (int(c) * 19349663) ^ (int(salt) * 83492791) ^ ((int(ctx.seed) * 2654435761) & 0xffffffff)
    x &= 0xffffffff
    x ^= x >> 15; x = (x * 2246822519) & 0xffffffff; x ^= x >> 13; x = (x * 3266489917) & 0xffffffff; x ^= x >> 16
    return (x & 0xffffff) / float(0x1000000)


def li_at(ctx, f, c):
    a = ctx.lit[f, c % ctx.ncell]
    return tuple(float(v) for v in a)


def room_depth(ctx, fi, f, c, lo, hi):
    lim = (ctx.D if fi in (0, 2) else ctx.W) / 2 - 0.4
    return max(1.6, min(lo + (hi - lo) * hr(ctx, f, c, 11), lim))


def fac_office(ctx, fr, fi):
    S = ctx.S; P = S['P']; Hf = S['Hf']; G = S['G']; md = ctx.md; near = ctx.detail == 'near'
    n = int(round(fr.L / P)); c0 = int(round(fr.uo / P))
    REC = S['REC']; MDp = S['MULL_D']; MW = S['MULL_W']; SILL = S['SILL']; VIS = S['VIS']
    ctx.frame = room_dims('office')
    for f in range(ctx.floors):
        z0 = ctx.zf(f)
        if f == 0:
            fbox(md, fr, -0.2, fr.L, 0.0, 0.6, -0.05, 0.04, 'base', skip=('d-',))                 # 腰の石の基壇
            gz0, gz1 = 0.6, 4.2
            ctx.frame = (P, gz1 - gz0, MW / 2, 0)
            for i in range(n):
                a0 = i * P; a1 = a0 + P; c = c0 + i
                fquad(md, fr, [(a0, gz1, -REC), (a1, gz1, -REC), (a1, G, -REC), (a0, G, -REC)], 'spandrel')
                li = li_at(ctx, 0, c)
                glass_pane(ctx, fr, a0, a1, gz0, gz1, -REC, li)
                if li[0] > 0:
                    room_for(ctx, fr, a0, a1, 0.0, gz1, gz0, gz1, -REC, li, 'office', room_depth(ctx, fi, 0, c, 4.0, 7.0), None)
            if near:
                for i in range(n + 1):
                    a = i * P
                    fbox(md, fr, a - MW / 2, a + MW / 2, gz0, gz1, -MDp, 0.0, 'sash', skip=('d-', 'z-', 'z+'))
                fbox(md, fr, 0.0, fr.L, gz1 - 0.035, gz1 + 0.035, -0.16, -0.01, 'sash', skip=('d-',))
            if fi == 0:                                                                            # 玄関の庇(ひさし)
                cw = min(6.6, fr.L * 0.5)
                fbox(md, fr, fr.L / 2 - cw / 2, fr.L / 2 + cw / 2, 4.0, 4.2, 0.0, 2.8, 'canopy', skip=('d-',))
            ctx.frame = room_dims('office')
            continue
        for i in range(n):
            a0 = i * P; a1 = a0 + P; c = c0 + i
            zs0 = z0 + SILL; zs1 = zs0 + VIS
            fquad(md, fr, [(a0, z0, -REC), (a1, z0, -REC), (a1, zs0, -REC), (a0, zs0, -REC)], 'spandrel')       # 腰(スパンドレル)下
            fquad(md, fr, [(a0, zs1, -REC), (a1, zs1, -REC), (a1, z0 + Hf, -REC), (a0, z0 + Hf, -REC)], 'spandrel')   # 上
            li = li_at(ctx, f, c)
            glass_pane(ctx, fr, a0, a1, zs0, zs1, -REC, li)
            if li[0] > 0:
                room_for(ctx, fr, a0, a1, z0, zs1, zs0, zs1, -REC, li, 'office', room_depth(ctx, fi, f, c, 3.6, 6.5), None)
            if near:                                                                                 # 横桟(トランサム)
                fbox(md, fr, a0, a1, zs0 - 0.035, zs0 + 0.035, -0.16, -0.01, 'sash', skip=('d-',))
                fbox(md, fr, a0, a1, zs1 - 0.035, zs1 + 0.035, -0.16, -0.01, 'sash', skip=('d-',))
        if near:                                                                                      # 縦桟(マリオン)
            for i in range(n + 1):
                a = i * P
                fbox(md, fr, a - MW / 2, a + MW / 2, z0, z0 + Hf, -MDp, 0.0, 'sash', skip=('d-', 'z-', 'z+'))
        # 梁型: 床スラブ小口の水平帯(階ごと)
        fbox(md, fr, -0.2, fr.L, z0 - 0.10, z0 + 0.25, -MDp, S['FASC_OUT'], 'beam', skip=('d-',))
    # 柱型(カバー): 4スパンごと + 隅。壁面(マリオン外面)から150mm出る
    for i in range(n + 1):
        if i % 4 == 0 or i == n:
            a = i * P
            lo, hi = (-S['COL_OUT'], S['COL_W']) if i == 0 else ((fr.L - S['COL_W'], fr.L) if i == n else (a - S['COL_W'] / 2, a + S['COL_W'] / 2))
            fbox(md, fr, lo, hi, 0.0, ctx.H, -MDp, S['COL_OUT'], 'pillar', skip=('d-', 'z-'))


def fac_zakkyo(ctx, fr, fi):
    S = ctx.S; P = S['P']; Hf = S['Hf']; G = S['G']; md = ctx.md; near = ctx.detail == 'near'
    n = int(round(fr.L / P)); c0 = int(round(fr.uo / P)); front = fi in (0, 2)
    PW = S['PIL_W']; PO = S['PIL_OUT']; BH = S['BEAM_H']; BO = S['BEAM_OUT']; REC = S['REC']; J = S['JAMB']
    SILL = S['SILL']; WH = S['WIN_H']
    ctx.frame = room_dims('zakkyo')
    for f in range(ctx.floors):
        z0 = ctx.zf(f); h = G if f == 0 else Hf
        # 柱型
        for i in range(n + 1):
            a = i * P
            lo, hi = (-PO, PW) if i == 0 else ((fr.L - PW, fr.L) if i == n else (a - PW / 2, a + PW / 2))
            fbox(md, fr, lo, hi, z0, z0 + h - BH, 0.0, PO, 'pillar', skip=('d-', 'z-', 'z+'))      # 梁型の帯と重ならないよう、柱は梁の下まで(同一平面の重なりは黒く抜ける)
        # 梁型(各階の上端、外に出る帯)
        fbox(md, fr, -PO, fr.L, z0 + h - BH, z0 + h, 0.0, BO, 'beam', skip=('d-',))
        for i in range(n):
            a0 = i * P; a1 = a0 + P; c = c0 + i
            if f == 0:
                if front:
                    wz0, wz1 = 0.15, 3.0
                    wall_around(ctx, fr, a0, a1, 0.0, G, a0 + J, a1 - J, wz0, wz1, 0.0, 'wall')
                    li = li_at(ctx, 0, c); shut = li[3] > 0.5
                    if ctx.detail != 'far':
                        reveal(ctx, fr, a0 + J, a1 - J, wz0, wz1, REC)
                        if shut:
                            fquad(md, fr, [(a0 + J, wz0, -0.04), (a1 - J, wz0, -0.04), (a1 - J, wz1, -0.04), (a0 + J, wz1, -0.04)], 'shutter',
                                  uvs=[(0, 0), (2.9, 0), (2.9, wz1 - wz0), (0, wz1 - wz0)])
                        else:
                            glass_pane(ctx, fr, a0 + J, a1 - J, wz0, wz1, -REC, li, plain=True)
                            if li[0] > 0:
                                room_for(ctx, fr, a0 + J, a1 - J, 0.0, 3.2, wz0, wz1, -REC, li, 'office', room_depth(ctx, fi, 0, c, 3.5, 6.0), None)
                            if near:
                                sash_frame(ctx, fr, a0 + J, a1 - J, wz0, wz1, -REC, S['SASH_W'], S['SASH_D'], 1)
                    if fi == 0 or hr(ctx, 0, c, 5) < 0.4:                                            # 看板(文字なし・色面だけ)
                        k = int(hr(ctx, 0, c, 6) * 4) % 4
                        fbox(md, fr, a0 + 0.45, a1 - 0.45, 3.05, 3.7, 0.0, 0.28, 'sign%d' % k, skip=('d-',))
                else:
                    fquad(md, fr, [(a0, 0, 0), (a1, 0, 0), (a1, G, 0), (a0, G, 0)], 'wall')
                continue
            if front:
                wa0, wa1 = a0 + J, a1 - J; wz0, wz1 = z0 + SILL, z0 + SILL + WH
                wall_around(ctx, fr, a0, a1, z0, z0 + Hf, wa0, wa1, wz0, wz1, 0.0, 'wall')
            elif i == 0:                                                                              # 側面: 小窓1列
                cx = a0 + P / 2; wa0, wa1 = cx - 0.45, cx + 0.45; wz0, wz1 = z0 + 1.3, z0 + 2.2
                wall_around(ctx, fr, a0, a1, z0, z0 + Hf, wa0, wa1, wz0, wz1, 0.0, 'wall')
            else:
                fquad(md, fr, [(a0, z0, 0), (a1, z0, 0), (a1, z0 + Hf, 0), (a0, z0 + Hf, 0)], 'wall')
                continue
            if ctx.detail == 'far':
                continue
            li = li_at(ctx, f, c)
            reveal(ctx, fr, wa0, wa1, wz0, wz1, REC)
            glass_pane(ctx, fr, wa0, wa1, wz0, wz1, -REC, li, plain=not front)
            if li[0] > 0:
                room_for(ctx, fr, wa0, wa1, z0, z0 + 2.7, wz0, wz1, -REC, li, 'office', room_depth(ctx, fi, f, c, 3.0, 5.5), None)
            if near:
                sash_frame(ctx, fr, wa0, wa1, wz0, wz1, -REC, S['SASH_W'], S['SASH_D'], 1 if front else 0)
                fbox(md, fr, wa0 - 0.05, wa1 + 0.05, wz0 - 0.06, wz0, 0.0, 0.07, 'beam', skip=('d-',))        # 窓台(壁面より前に出る部分。雨だれの起点。開口の底面とは重ねない)
                if front and hr(ctx, f, c, 3) < 0.25:                                                       # 室外機
                    cx = (wa0 + wa1) / 2 + (hr(ctx, f, c, 4) - 0.5) * 1.0
                    fbox(md, fr, cx - 0.4, cx + 0.4, z0 + 0.12, z0 + 0.78, 0.0, 0.36, 'ac', skip=('d-',))
    if fi == 1:
        escape_stair(ctx, fr)


def escape_stair(ctx, fr):
    """外付けの非常階段(鉄骨・黒〜錆色)。側面の中央。推"""
    if ctx.detail == 'far' or fr.L < 7.8:
        return
    md = ctx.md; S = ctx.S; Hf = S['Hf']; G = S['G']
    sc = fr.L * 0.5; a0 = sc - 2.6; a1 = sc + 2.6; W_ = 1.0; BO_ = S['BEAM_OUT']          # 梁型の出より外側から踊り場を張る
    ztop = ctx.zf(ctx.floors - 1)
    zs = [0.0, G / 2]
    z = G
    while z <= ztop + 1e-6:
        zs.append(z)
        if z + Hf / 2 <= ztop + 1e-6:
            zs.append(z + Hf / 2)
        z += Hf
    pts = []
    for k, zk in enumerate(zs):
        pts.append((a0 + 1.2 if k % 2 == 0 else a1 - 1.2, zk))
    for k, (ak, zk) in enumerate(pts):
        if k > 0:                                                                                      # 踊り場
            if k % 2 == 0:
                fbox(md, fr, a0, a0 + 1.2, zk - 0.08, zk, BO_, W_ + 0.1, 'stair', skip=('d-',))
            else:
                fbox(md, fr, a1 - 1.2, a1, zk - 0.08, zk, BO_, W_ + 0.1, 'stair', skip=('d-',))
    for k in range(len(pts) - 1):
        (aA, zA), (aB, zB) = pts[k], pts[k + 1]
        fquad(md, fr, [(aA, zA, 0.1), (aB, zB, 0.1), (aB, zB, W_), (aA, zA, W_)], 'stair', uvs=[(0, 0), (1, 0), (1, 1), (0, 1)])
        fquad(md, fr, [(aA, zA, W_), (aB, zB, W_), (aB, zB, 0.1), (aA, zA, 0.1)], 'stair', uvs=[(0, 0), (1, 0), (1, 1), (0, 1)])
        ln = math.hypot(aB - aA, zB - zA)
        fquad(md, fr, [(aA, zA, W_ + 0.1), (aB, zB, W_ + 0.1), (aB, zB + 0.95, W_ + 0.1), (aA, zA + 0.95, W_ + 0.1)], 'rail_bars', uvs=[(0, 0), (ln, 0), (ln, 0.95), (0, 0.95)])
    for k, (ak, zk) in enumerate(pts):
        if k > 0:
            aa0, aa1 = (a0, a0 + 1.2) if k % 2 == 0 else (a1 - 1.2, a1)
            fquad(md, fr, [(aa0, zk, W_ + 0.1), (aa1, zk, W_ + 0.1), (aa1, zk + 0.95, W_ + 0.1), (aa0, zk + 0.95, W_ + 0.1)], 'rail_bars', uvs=[(0, 0), (1.2, 0), (1.2, 0.95), (0, 0.95)])
    for aa in (a0 + 0.04, a1 - 0.04):
        fbox(md, fr, aa - 0.05, aa + 0.05, 0.0, ztop, W_ + 0.05, W_ + 0.15, 'stair', skip=('d-', 'z-', 'z+'))


def fac_mansion(ctx, fr, fi):
    S = ctx.S; P = S['P']; Hf = S['Hf']; G = S['G']; md = ctx.md; near = ctx.detail == 'near'
    n = int(round(fr.L / P)); c0 = int(round(fr.uo / P)); U = 2 * P
    SLAB = S['SLAB']; BAL = S['BAL']; RH = S['RAIL_H']; REC = S['REC']; WW = S['WIN_W']; WH = S['WIN_H']; WZ = S['WIN_Z']
    ctx.frame = room_dims('mansion')
    front = fi == 0; back = fi == 2
    for f in range(ctx.floors):
        z0 = ctx.zf(f); topf = f == ctx.floors - 1
        z1 = z0 + Hf if topf else z0 + Hf - SLAB
        if f == 0:
            z1 = G - SLAB
        if f > 0 and front:
            fbox(md, fr, 0.0, fr.L, z0 - SLAB, z0, 0.0, BAL, 'slab', skip=('d-',))                       # バルコニーの床スラブ(小口が水平線に見える)
            if ctx.rail_solid:
                fbox(md, fr, 0.0, fr.L, z0, z0 + RH, BAL - 0.10, BAL, 'wall', skip=('z-',))
                if near:
                    fbox(md, fr, -0.02, fr.L + 0.02, z0 + RH, z0 + RH + 0.04, BAL - 0.13, BAL + 0.03, 'slab', skip=('z-',))
            else:
                fbox(md, fr, 0.0, fr.L, z0, z0 + 0.15, BAL - 0.06, BAL, 'wall', skip=('z-',))
                fbox(md, fr, 0.0, fr.L, z0 + RH - 0.04, z0 + RH, BAL - 0.07, BAL + 0.01, 'rail_metal', skip=('z-',))
                if near:
                    m_ = int(fr.L / 0.12)
                    for k in range(m_ + 1):
                        x = 0.06 + k * (fr.L - 0.12) / m_
                        fbox(md, fr, x - 0.01, x + 0.01, z0 + 0.15, z0 + RH - 0.04, BAL - 0.05, BAL - 0.03, 'rail_metal', skip=('d-', 'z+', 'z-'))
                else:
                    fquad(md, fr, [(0, z0 + 0.15, BAL - 0.04), (fr.L, z0 + 0.15, BAL - 0.04), (fr.L, z0 + RH - 0.04, BAL - 0.04), (0, z0 + RH - 0.04, BAL - 0.04)], 'rail_bars',
                          uvs=[(fr.uo, 0), (fr.uo + fr.L, 0), (fr.uo + fr.L, RH - 0.19), (fr.uo, RH - 0.19)])
            if ctx.detail != 'far':
                for k in range(n // 2 + 1):                                                            # 隔て板(住戸の境)
                    a = k * U
                    fbox(md, fr, max(a - 0.02, 0.0), min(a + 0.02, fr.L), z0, z0 + 2.0, 0.0, BAL - 0.1, 'wall', skip=('d-', 'z-'))
                if near:
                    for k in range(n // 2):
                        if hr(ctx, f, c0 + 2 * k, 3) < 0.3:
                            x = k * U + 0.7 + hr(ctx, f, c0 + 2 * k, 4) * 4.0
                            fbox(md, fr, x, x + 0.8, z0, z0 + 0.7, 0.2, 0.5, 'ac', skip=('d-', 'z-'))     # バルコニーの室外機
        if back and f > 0:
            CO = S['CORR']
            fbox(md, fr, 0.0, fr.L, z0 - SLAB, z0, 0.0, CO, 'slab', skip=('d-',))                       # 外廊下
            fbox(md, fr, 0.0, fr.L, z0, z0 + RH, CO - 0.12, CO, 'wall', skip=('z-',))
        for i in range(n):
            a0 = i * P; a1 = a0 + P; c = c0 + i
            if f == 0:
                if front:
                    wa0, wa1, wz0, wz1 = a0 + 0.3, a1 - 0.3, 0.1, 2.9
                    wall_around(ctx, fr, a0, a1, 0.0, G - SLAB, wa0, wa1, wz0, wz1, 0.0, 'wall')
                    if ctx.detail != 'far':
                        reveal(ctx, fr, wa0, wa1, wz0, wz1, REC)
                        li = li_at(ctx, 0, c)
                        glass_pane(ctx, fr, wa0, wa1, wz0, wz1, -REC, li, plain=True)
                        if li[0] > 0:
                            room_for(ctx, fr, wa0, wa1, 0.0, 3.2, wz0, wz1, -REC, li, 'office', room_depth(ctx, fi, 0, c, 3.5, 6.0), None)
                        if near:
                            sash_frame(ctx, fr, wa0, wa1, wz0, wz1, -REC, 0.05, 0.06, 2)
                else:
                    fquad(md, fr, [(a0, 0, 0), (a1, 0, 0), (a1, G - SLAB, 0), (a0, G - SLAB, 0)], 'wall')
                continue
            if front:
                wa0 = a0 + (P - WW) / 2; wa1 = wa0 + WW; wz0 = z0 + WZ; wz1 = wz0 + WH
                wall_around(ctx, fr, a0, a1, z0, z1, wa0, wa1, wz0, wz1, 0.0, 'wall')
            elif back:
                if i % 2 == 0:                                                                        # 玄関ドア
                    da = a0 + 1.2
                    fquad(md, fr, [(a0, z0, 0), (da, z0, 0), (da, z1, 0), (a0, z1, 0)], 'wall')
                    fquad(md, fr, [(da + 0.95, z0, 0), (a1, z0, 0), (a1, z1, 0), (da + 0.95, z1, 0)], 'wall')
                    fquad(md, fr, [(da, z0 + 2.05, 0), (da + 0.95, z0 + 2.05, 0), (da + 0.95, z1, 0), (da, z1, 0)], 'wall')
                    if ctx.detail != 'far':
                        fbox(md, fr, da, da + 0.95, z0, z0 + 2.05, -0.03, 0.02, 'door', skip=('d-', 'z-'))
                        fbox(md, fr, a0 + 0.3, a0 + 0.6, z0 + 2.3, z0 + 2.4, 0.0, 0.1, 'lamp_corr', skip=('d-',), win=(1.0 if ctx.night else 0.0, 0.30, 0.0, 0.0))
                    continue
                wa0 = a0 + 1.4; wa1 = wa0 + 0.6; wz0 = z0 + 1.1; wz1 = z0 + 2.0
                wall_around(ctx, fr, a0, a1, z0, z1, wa0, wa1, wz0, wz1, 0.0, 'wall')
            else:
                fquad(md, fr, [(a0, z0, 0), (a1, z0, 0), (a1, z1, 0), (a0, z1, 0)], 'wall')
                if f > 0 and ctx.detail != 'far':
                    fbox(md, fr, a0, a1, z0 - SLAB, z0, 0.0, 0.05, 'slab', skip=('d-',))
                continue
            if ctx.detail == 'far':
                continue
            li = li_at(ctx, f, c)
            reveal(ctx, fr, wa0, wa1, wz0, wz1, REC)
            glass_pane(ctx, fr, wa0, wa1, wz0, wz1, -REC, li, plain=not front)
            if li[0] > 0:
                room_for(ctx, fr, wa0, wa1, z0, z0 + 2.4, wz0, wz1, -REC, li, 'apt', room_depth(ctx, fi, f, c, 3.0, 5.0), None)
            if near:
                sash_frame(ctx, fr, wa0, wa1, wz0, wz1, -REC, 0.05, 0.06, 1 if front else 0)


FACADE = {'office': fac_office, 'zakkyo': fac_zakkyo, 'mansion': fac_mansion}


# ------------------------------------------------------------------------------------------------
# 組み立て(bpy不要)
# ------------------------------------------------------------------------------------------------
def build_geometry(kind, width, depth, floors, seed=0, detail='near', night=True, hour=20.0, age=None):
    """ビルの形(面・UV・窓の属性・材質の役割名)を作る。 -> (MD, info辞書, ctx)"""
    kind = KIND_ALIAS.get(kind, kind)
    S = SPECS[kind]; P = S['P']
    ctx = X(); ctx.kind = kind; ctx.S = S; ctx.detail = detail; ctx.night = night; ctx.seed = int(seed)
    nW = max(S['minb'][0], int(round(width / P))); nD = max(S['minb'][1], int(round(depth / P)))
    if kind == 'mansion' and nW % 2:
        nW += 1
    W = nW * P; D = nD * P
    floors = max(2, int(floors)); G = S['G']; Hf = S['Hf']
    H = G + (floors - 1) * Hf
    rng = random.Random(int(seed) * 1000003 + 11)
    ctx.W = W; ctx.D = D; ctx.H = H; ctx.floors = floors; ctx.nW = nW; ctx.nD = nD
    ctx.zf = lambda f: 0.0 if f == 0 else G + (f - 1) * Hf
    ctx.rng = random.Random(int(seed) * 31 + 5)
    ctx.rail_solid = rng.random() < 0.5
    ctx.age = float(age) if age is not None else S['age'][0] + (S['age'][1] - S['age'][0]) * rng.random()
    ctx.hour = hour
    ctx.ncell = 2 * (nW + nD)
    ctx.lit = make_lit(kind, ctx.ncell, floors, seed, hour, night)
    ctx.frames = [Fr((-W / 2, -D / 2), (1.0, 0.0), (0.0, -1.0), W, 0.0), Fr((W / 2, -D / 2), (0.0, 1.0), (1.0, 0.0), D, W),
                  Fr((W / 2, D / 2), (-1.0, 0.0), (0.0, 1.0), W, W + D), Fr((-W / 2, D / 2), (0.0, -1.0), (-1.0, 0.0), D, 2 * W + D)]
    ctx.frame = room_dims(kind)
    ctx.md = MD()
    if detail == 'far':
        for fr in ctx.frames:
            fquad(ctx.md, fr, [(0, 0, 0), (fr.L, 0, 0), (fr.L, H, 0), (0, H, 0)], 'far')
    else:
        for i, fr in enumerate(ctx.frames):
            FACADE[kind](ctx, fr, i)
    ritems = roof_items(ctx)
    md = ctx.md
    V = np.array(md.V, np.float64)
    info = dict(kind=kind, detail=detail, seed=int(seed), night=bool(night), hour=float(hour), module_m=P, floors=floors,
                floor_height_m=Hf, ground_floor_m=G, width_m=W, depth_m=D, height_m=H,
                bbox=[float(x) for x in np.r_[V.min(0), V.max(0)]], faces=len(md.F), tris=int(sum(len(f) - 2 for f in md.F)),
                lit_rate_target=lit_rate(kind, hour, night), age=ctx.age, roof=ritems, rail_solid=ctx.rail_solid,
                lit_windows=int((ctx.lit[1:, :, 0] > 0).sum()), windows_total=int(ctx.lit[1:].shape[0] * ctx.lit.shape[1]))
    return md, info, ctx


# ------------------------------------------------------------------------------------------------
# 材質(手続き的なシェーダ。 bpy が要る)
# ------------------------------------------------------------------------------------------------
def srgb2lin(c):
    return tuple((x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4) for x in c)


class NB:
    """ノード作りの補助。数値でもソケットでも渡せる。"""
    def __init__(self, nt):
        self.nt = nt

    def node(self, t, **kw):
        n = self.nt.nodes.new(t)
        for k, v in kw.items():
            setattr(n, k, v)
        return n

    def put(self, n, idx, v):
        if v is None:
            return
        if isinstance(v, (int, float)):
            n.inputs[idx].default_value = float(v)
        else:
            self.nt.links.new(v, n.inputs[idx])

    def m(self, op, a=None, b=None, c=None, clamp=False):
        n = self.node('ShaderNodeMath', operation=op); n.use_clamp = clamp
        self.put(n, 0, a); self.put(n, 1, b); self.put(n, 2, c)
        return n.outputs[0]

    def sstep(self, v, a, b):
        """smoothstep(a, b, v)。 a > b なら 1 → 0 の向き。 a, b は数値でもソケットでもよい"""
        t = self.m('DIVIDE', self.m('SUBTRACT', v, a), self.m('SUBTRACT', b, a), clamp=True)
        return self.m('MULTIPLY', self.m('MULTIPLY', t, t), self.m('SUBTRACT', 3.0, self.m('MULTIPLY', t, 2.0)))

    def vec(self, x, y, z):
        n = self.node('ShaderNodeCombineXYZ')
        self.put(n, 0, x); self.put(n, 1, y); self.put(n, 2, z)
        return n.outputs[0]

    def sep(self, v):
        n = self.node('ShaderNodeSeparateXYZ'); self.nt.links.new(v, n.inputs[0])
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def noise(self, vec, scale=1.0, detail=3.0, rough=0.55):
        n = self.node('ShaderNodeTexNoise')
        n.inputs['Scale'].default_value = scale; n.inputs['Detail'].default_value = detail; n.inputs['Roughness'].default_value = rough
        self.nt.links.new(vec, n.inputs['Vector'])
        return n.outputs['Fac']

    def white(self, a, b, c):
        n = self.node('ShaderNodeTexWhiteNoise', noise_dimensions='3D')
        self.nt.links.new(self.vec(a, b, c), n.inputs['Vector'])
        return n.outputs['Value']

    def rgb(self, c):
        n = self.node('ShaderNodeRGB'); n.outputs[0].default_value = (c[0], c[1], c[2], 1.0)
        return n.outputs[0]

    def vscale(self, a, k):
        n = self.node('ShaderNodeVectorMath', operation='SCALE')
        self.nt.links.new(a, n.inputs[0]); self.put(n, 3, k)
        return n.outputs[0]

    def vop(self, op, a, b):
        n = self.node('ShaderNodeVectorMath', operation=op)
        self.nt.links.new(a, n.inputs[0]); self.nt.links.new(b, n.inputs[1])
        return n.outputs[0]

    def vmix(self, a, b, f):
        return self.vop('ADD', a, self.vscale(self.vop('SUBTRACT', b, a), f))

    def uv(self):
        tc = self.node('ShaderNodeTexCoord')
        x, y, _ = self.sep(tc.outputs['UV'])
        return x, y

    def attr(self, name='win'):
        a = self.node('ShaderNodeAttribute'); a.attribute_name = name
        sc = self.node('ShaderNodeSeparateColor'); self.nt.links.new(a.outputs['Color'], sc.inputs[0])
        return sc.outputs[0], sc.outputs[1], sc.outputs[2], a.outputs['Alpha']

    def blackbody(self, tk):
        n = self.node('ShaderNodeBlackbody'); self.put(n, 0, tk)
        return n.outputs[0]

    def rng_between(self, v, a, b):
        return self.m('ADD', a, self.m('MULTIPLY', v, b - a))


def new_material(name):
    mat = bpy.data.materials.new(name)
    try:
        mat.use_nodes = True
    except Exception:
        pass
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    return mat, nt, out


def principled(nt, **kw):
    b = nt.nodes.new('ShaderNodeBsdfPrincipled')
    for k, v in kw.items():
        try:
            b.inputs[k].default_value = v
        except Exception as e:
            print('principled', k, e)
    return b


def finish(mat, nt, out, shader, simple=None):
    nt.links.new(shader, out.inputs['Surface'])
    if simple is not None:
        mat['kit_simple'] = json.dumps(simple)
    return mat


def dirt_socket(nb, u, v, C, streak=True):
    """汚れ 0..1(手続き的)。 u,v = ファサードの座標(m。 u は面ごとに P の整数倍ずれる)。調査1-4:
    窓台・庇の真下の縦の雨だれ / 上端(パラペット下)と出隅の黒ずみ / 1階の下端の暗い帯 / 大きな縦のムラ"""
    S = C['S']; P = S['P']; Hf = S['Hf']; G = S['G']; sd = C['seed'] * 1.713 + 0.37
    W, D = C['W'], C['D']

    def nz(ax, ay, det=3.0):
        return nb.noise(nb.vec(nb.m('MULTIPLY', u, ax), nb.m('MULTIPLY', v, ay), sd), 1.0, det)
    terms = []
    if streak:
        v1 = nb.m('SUBTRACT', v, G)
        cellf = nb.m('FLOOR', nb.m('DIVIDE', u, P))
        um = nb.m('MULTIPLY', nb.m('FRACT', nb.m('DIVIDE', u, P)), P)
        fl = nb.m('FLOOR', nb.m('DIVIDE', v1, Hf))
        hv = nb.white(cellf, fl, sd); hv2 = nb.white(cellf, fl, sd + 11.0)
        on = nb.m('GREATER_THAN', hv, 0.30)                                       # 筋が出ない窓もある
        wmax = nb.m('ADD', 0.08, nb.m('MULTIPLY', nz(1.7, 0.5, 2.0), 0.28))       # 窓の隅の筋の太さ
        line = 0.0
        for e in C['edges']:
            d = nb.m('ABSOLUTE', nb.m('SUBTRACT', um, e))
            line = nb.m('MAXIMUM', line, nb.sstep(d, wmax, 0.0))
        dsill = nb.m('MULTIPLY', nb.m('FRACT', nb.m('DIVIDE', nb.m('SUBTRACT', C['sill'], v1), Hf)), Hf)   # 窓台から下への距離
        ls = nb.m('ADD', 1.4, nb.m('MULTIPLY', hv2, 1.8))                         # 筋の長さ 1.4〜3.2m(推1〜3m)
        decay = nb.sstep(dsill, ls, 0.0)
        tex = nb.m('ADD', 0.35, nb.m('MULTIPLY', nz(6.0, 0.22, 4.0), 1.1), clamp=True)
        terms.append(nb.m('MULTIPLY', nb.m('MULTIPLY', nb.m('MULTIPLY', nb.m('MULTIPLY', line, decay), on), tex), 2.6))
        gen = nb.m('MULTIPLY', nb.m('MULTIPLY', nb.sstep(nz(9.0, 0.12, 4.0), 0.42, 0.62), decay), 0.9)      # 窓台の幅いっぱいに流れる細い筋
        terms.append(nb.m('MULTIPLY', gen, nb.m('MULTIPLY', nb.m('GREATER_THAN', hv2, 0.25), 1.0)))
    broad = nb.m('MULTIPLY', nb.sstep(nz(0.55, 0.12, 3.0), 0.45, 0.80), 0.30)        # 大きな縦のムラ
    terms.append(broad)
    top = nb.m('MULTIPLY', nb.sstep(nb.m('SUBTRACT', C['H'], v), 2.4, 0.0), nb.m('ADD', 0.45, nb.m('MULTIPLY', nz(2.0, 0.6), 0.9)))
    terms.append(nb.m('MULTIPLY', top, 0.85))                                       # 上端
    m_ = nb.m('MULTIPLY', nb.m('FRACT', nb.m('DIVIDE', u, W + D)), W + D)           # 半周の中の位置
    inW = nb.m('LESS_THAN', m_, W)
    s_ = nb.m('SUBTRACT', m_, nb.m('MULTIPLY', nb.m('SUBTRACT', 1.0, inW), W))
    L_ = nb.m('ADD', nb.m('MULTIPLY', inW, W), nb.m('MULTIPLY', nb.m('SUBTRACT', 1.0, inW), D))
    cd = nb.m('MINIMUM', s_, nb.m('SUBTRACT', L_, s_))                              # 出隅までの距離
    terms.append(nb.m('MULTIPLY', nb.sstep(cd, 1.4, 0.0), nb.m('MULTIPLY', nb.m('ADD', 0.4, nb.m('MULTIPLY', nz(3.0, 0.4), 0.9)), 0.6)))
    bot = nb.m('MULTIPLY', nb.sstep(v, 3.2, 0.0), nb.m('ADD', 0.55, nb.m('MULTIPLY', nz(2.5, 1.2), 0.8)))
    terms.append(nb.m('MULTIPLY', bot, 0.9))                                       # 1階の下端(跳ね返り)
    d = terms[0]
    for t in terms[1:]:
        d = nb.m('ADD', d, t)
    return nb.m('MINIMUM', d, 1.0)


def mat_wall(C, name, base, tile=False, streak=True, dirt_k=1.0, rough=0.88):
    """外壁(コンクリート塗装/タイル)。汚れ・タイル目地・粗さ・凹凸。 base = 新品のアルベド(リニア)"""
    mat, nt, out = new_material(name); nb = NB(nt)
    b = principled(nt)
    u, v = nb.uv()
    far = C['detail'] == 'far'
    dirt = nb.m('MULTIPLY', dirt_socket(nb, u, v, C, streak), dirt_k * (1.3 if tile else 1.4), clamp=True)       # タイルは汚れにくい★(塗装・吹付けは強く)
    A = C['A']
    sd = C['seed'] * 1.713 + 0.37
    if tile and not far:
        pu, pv, jw = 0.236, 0.069, 0.009                                           # 二丁掛 60×227mm + 目地9mm★
        row = nb.m('FLOOR', nb.m('DIVIDE', v, pv)); off = nb.m('FRACT', nb.m('MULTIPLY', row, 0.5))
        cu = nb.m('ADD', nb.m('DIVIDE', u, pu), off)
        col = nb.m('FLOOR', cu)
        fu = nb.m('MULTIPLY', nb.m('FRACT', cu), pu); fv = nb.m('MULTIPLY', nb.m('FRACT', nb.m('DIVIDE', v, pv)), pv)
        joint = nb.m('MAXIMUM', nb.m('LESS_THAN', fu, jw), nb.m('LESS_THAN', fv, jw))
        tone = nb.m('ADD', 0.90, nb.m('MULTIPLY', nb.white(col, row, sd), 0.2))
        tile_c = nb.vscale(nb.rgb(base), tone)
        colr = nb.vmix(tile_c, nb.rgb((0.14, 0.14, 0.135)), joint)
        rgh = nb.m('ADD', nb.m('ADD', 0.28, nb.m('MULTIPLY', dirt, 0.2)), nb.m('MULTIPLY', joint, 0.6))      # 釉薬タイル 0.25〜0.4★(推)・目地0.9
        bump = nb.node('ShaderNodeBump'); bump.inputs['Strength'].default_value = 0.6; bump.inputs['Distance'].default_value = 0.006
        nt.links.new(nb.m('SUBTRACT', 1.0, joint), bump.inputs['Height']); nt.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    else:
        mott = nb.m('ADD', 0.90, nb.m('MULTIPLY', nb.noise(nb.vec(nb.m('MULTIPLY', u, 0.7), nb.m('MULTIPLY', v, 0.7), sd + 3.0), 1.0, 4.0), 0.2))
        colr = nb.vscale(nb.rgb(base), mott)
        rgh = nb.m('ADD', 0.80 if not tile else 0.4, nb.m('MULTIPLY', dirt, 0.1))
        if not far:
            bump = nb.node('ShaderNodeBump'); bump.inputs['Strength'].default_value = 0.25; bump.inputs['Distance'].default_value = 0.01
            nt.links.new(nb.noise(nb.vec(nb.m('MULTIPLY', u, 14.0), nb.m('MULTIPLY', v, 14.0), sd), 1.0, 4.0), bump.inputs['Height'])
            nt.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    # 汚れ: 築年数による全体の明度低下(20〜40%) + 汚れの場所の暗さ(最大20〜40%)
    dark = nb.m('MINIMUM', nb.m('ADD', A * 0.35, nb.m('MULTIPLY', dirt, A * 0.65)), 0.75)        # 汚れの最も濃い所で A(築年数に応じた20〜60%)だけ暗い
    tint = nb.vmix(nb.rgb((1, 1, 1)), nb.rgb((0.9, 0.86, 0.8)), nb.m('MULTIPLY', dirt, 0.7))
    final = nb.vop('MULTIPLY', nb.vscale(colr, nb.m('SUBTRACT', 1.0, dark)), tint)
    nt.links.new(final, b.inputs['Base Color'])
    nt.links.new(rgh, b.inputs['Roughness'])
    for k, vv in (('Specular IOR Level', 0.5), ('IOR', 1.5)):
        try:
            b.inputs[k].default_value = vv
        except Exception:
            pass
    avg = tuple(min(1.0, c * (1 - A * 0.55)) for c in base)
    return finish(mat, nt, out, b.outputs['BSDF'], dict(c=list(avg), r=0.5 if tile else 0.88, m=0.0))


def mat_plain(C, name, color, rough=0.5, metal=0.0, emit=None, emit_strength=0.0):
    mat, nt, out = new_material(name)
    b = principled(nt, **{'Base Color': (*color, 1.0), 'Roughness': rough, 'Metallic': metal})
    simple = dict(c=list(color), r=rough, m=metal)
    if emit is not None:
        b.inputs['Emission Color'].default_value = (*emit, 1.0); b.inputs['Emission Strength'].default_value = emit_strength
        simple.update(e=list(emit), es=emit_strength)
    return finish(mat, nt, out, b.outputs['BSDF'], simple)


def frame_mask(nb, u, v, ww, wh, fw, nm):
    msk = nb.m('MAXIMUM', nb.m('MAXIMUM', nb.m('LESS_THAN', u, fw), nb.m('GREATER_THAN', u, ww - fw)),
               nb.m('MAXIMUM', nb.m('LESS_THAN', v, fw), nb.m('GREATER_THAN', v, wh - fw)))
    for k in range(1, nm + 1):
        am = ww * k / (nm + 1)
        msk = nb.m('MAXIMUM', msk, nb.m('LESS_THAN', nb.m('ABSOLUTE', nb.m('SUBTRACT', u, am)), fw * 0.6))
    return msk


def mat_glass(C, name, lit, frame=None):
    """ガラス。 IOR1.5(法線入射で反射4%★)。消灯 = 暗くて空を映す。 点灯 = 透明(奥の部屋が見える)+ フレネル反射。
    frame=(幅,高さ,枠幅,中桟数) なら、mid のためのサッシ枠を模様で描く。"""
    mat, nt, out = new_material(name); nb = NB(nt)
    tint = C['glass_tint']
    if lit:
        tr = nb.node('ShaderNodeBsdfTransparent')
        gl = nb.node('ShaderNodeBsdfGlossy'); gl.inputs['Color'].default_value = (1, 1, 1, 1); gl.inputs['Roughness'].default_value = 0.02
        fr = nb.node('ShaderNodeFresnel'); fr.inputs['IOR'].default_value = 1.5
        fac = nb.m('MAXIMUM', fr.outputs['Fac'], C['glass_refl_min'])
        mx = nb.node('ShaderNodeMixShader'); nt.links.new(fac, mx.inputs[0]); nt.links.new(tr.outputs[0], mx.inputs[1]); nt.links.new(gl.outputs[0], mx.inputs[2])
        sh = mx.outputs[0]
    else:
        b = principled(nt, **{'Base Color': (*tint, 1.0), 'Metallic': C['glass_metal'], 'Roughness': 0.03, 'IOR': 1.5})
        sh = b.outputs['BSDF']
    if frame is not None:
        ww, wh, fw, nm = frame
        u, v = nb.uv()
        msk = frame_mask(nb, u, v, ww, wh, fw, nm)
        sb = principled(nt, **{'Base Color': (*C['sash_col'], 1.0), 'Metallic': C['sash_metal'], 'Roughness': 0.42})
        mx2 = nb.node('ShaderNodeMixShader'); nt.links.new(msk, mx2.inputs[0]); nt.links.new(sh, mx2.inputs[1]); nt.links.new(sb.outputs[0], mx2.inputs[2])
        sh = mx2.outputs[0]
    simple = dict(c=list(tint), r=0.05, m=0.85) if not lit else dict(c=[0.05, 0.06, 0.07], r=0.05, m=0.5, a=0.25)
    return finish(mat, nt, out, sh, simple)


def emit_shader(nb, T, I, k, tint=None):
    """部屋の発光: 色 = 色温度 T(K) の黒体放射 × 明るさ I × k"""
    col = nb.blackbody(T)
    if tint is not None:
        col = nb.vop('MULTIPLY', col, tint)
    em = nb.node('ShaderNodeEmission'); nb.nt.links.new(col, em.inputs['Color'])
    nb.put(em, 1, k if not isinstance(k, (int, float)) else float(k))
    return em.outputs[0]


def mat_room(C, name, part, rkind):
    """室内の板。 part: wall / floor / ceil / blind。 属性 win = (明るさ, 色温度/1e4, ブラインド・カーテンの量, 色指標)"""
    mat, nt, out = new_material(name); nb = NB(nt)
    I, Tn, B, Aa = nb.attr('win')
    T = nb.m('MULTIPLY', Tn, 10000.0)
    E = C['E']
    u, v = nb.uv()
    if part == 'wall':
        k = nb.m('MULTIPLY', nb.m('MULTIPLY', I, E), nb.m('ADD', 0.22, nb.m('MULTIPLY', nb.sstep(v, 0.6, 1.5), 0.55)))        # 下は机・棚で暗く、上は天井の照明で明るい
        sh = emit_shader(nb, T, I, k)
    elif part == 'floor':
        sh = emit_shader(nb, T, I, nb.m('MULTIPLY', nb.m('MULTIPLY', I, E), 0.22))
    elif part == 'ceil':
        if rkind == 'office':                                                        # 天井の照明の並び(1.8m格子に 1.2×0.3m の器具)
            ux = nb.m('FRACT', nb.m('DIVIDE', u, 1.8)); vy = nb.m('FRACT', nb.m('DIVIDE', v, 1.8))
            pan = nb.m('MULTIPLY', nb.m('LESS_THAN', nb.m('MULTIPLY', nb.m('ABSOLUTE', nb.m('SUBTRACT', ux, 0.5)), 1.8), 0.6),
                       nb.m('LESS_THAN', nb.m('MULTIPLY', nb.m('ABSOLUTE', nb.m('SUBTRACT', vy, 0.5)), 1.8), 0.15))
            kk = nb.m('ADD', 0.28, nb.m('MULTIPLY', pan, 2.4))
        else:                                                                        # 住戸: 中央の丸いシーリング
            dd = nb.m('ADD', nb.m('POWER', nb.m('SUBTRACT', u, 0.82), 2.0), nb.m('POWER', nb.m('SUBTRACT', v, 1.6), 2.0))
            kk = nb.m('ADD', 0.22, nb.m('MULTIPLY', nb.m('LESS_THAN', dd, 0.09), 2.2))
        sh = emit_shader(nb, T, I, nb.m('MULTIPLY', nb.m('MULTIPLY', I, E), kk))
    else:                                                                            # blind
        tr = nb.node('ShaderNodeBsdfTransparent')
        if rkind == 'office':                                                        # 横型ブラインド(上から B*1.8m 下りる)。 uv の v = 上端からの距離
            cov = nb.m('LESS_THAN', v, nb.m('MULTIPLY', B, 1.8))
            slat = nb.m('ADD', 0.55, nb.m('MULTIPLY', nb.m('LESS_THAN', nb.m('FRACT', nb.m('DIVIDE', v, 0.028)), 0.72), 0.45))
            mask = nb.m('MULTIPLY', cov, nb.m('GREATER_THAN', slat, 0.0))
            sh_e = emit_shader(nb, T, I, nb.m('MULTIPLY', nb.m('MULTIPLY', I, E), nb.m('MULTIPLY', slat, 0.5)))
        else:                                                                        # カーテン(左右の布。色は住戸ごと)
            cw = nb.m('MULTIPLY', B, 0.82)
            cov = nb.m('MAXIMUM', nb.m('LESS_THAN', u, cw), nb.m('GREATER_THAN', u, nb.m('SUBTRACT', 1.65, cw)))
            fold = nb.m('ADD', 0.78, nb.m('MULTIPLY', nb.m('SINE', nb.m('MULTIPLY', u, 38.0)), 0.18))
            cc = nb.vmix(nb.rgb((0.95, 0.80, 0.55)), nb.rgb((0.55, 0.68, 0.85)), nb.m('MULTIPLY', Aa, 1.0))
            sh_e = emit_shader(nb, T, I, nb.m('MULTIPLY', nb.m('MULTIPLY', I, E), nb.m('MULTIPLY', fold, 0.85)), tint=cc)
            mask = cov
        mx = nb.node('ShaderNodeMixShader'); nt.links.new(mask, mx.inputs[0]); nt.links.new(tr.outputs[0], mx.inputs[1]); nt.links.new(sh_e, mx.inputs[2])
        sh = mx.outputs[0]
    return finish(mat, nt, out, sh, dict(c=[0.04, 0.04, 0.04], r=0.9, m=0.0, e=[1.0, 0.82, 0.55], es=1.2))


def mat_bars(C, name):
    """手すり・柵: 縦の細い棒(透け)。 uv の u = 沿った距離(m)"""
    mat, nt, out = new_material(name); nb = NB(nt)
    u, v = nb.uv()
    bar = nb.m('MAXIMUM', nb.m('LESS_THAN', nb.m('FRACT', nb.m('DIVIDE', u, 0.12)), 0.16), nb.m('LESS_THAN', v, 0.05))
    tr = nb.node('ShaderNodeBsdfTransparent'); b = principled(nt, **{'Base Color': (0.05, 0.05, 0.055, 1.0), 'Metallic': 0.8, 'Roughness': 0.45})
    mx = nb.node('ShaderNodeMixShader'); nt.links.new(bar, mx.inputs[0]); nt.links.new(tr.outputs[0], mx.inputs[1]); nt.links.new(b.outputs[0], mx.inputs[2])
    return finish(mat, nt, out, mx.outputs[0], dict(c=[0.05, 0.05, 0.055], r=0.5, m=0.8, a=0.5))


def mat_shutter(C, name):
    mat, nt, out = new_material(name); nb = NB(nt)
    u, v = nb.uv()
    slat = nb.m('ADD', 0.75, nb.m('MULTIPLY', nb.m('LESS_THAN', nb.m('FRACT', nb.m('DIVIDE', v, 0.075)), 0.85), 0.25))
    b = principled(nt, **{'Metallic': 0.7, 'Roughness': 0.42})
    nt.links.new(nb.vscale(nb.rgb((0.30, 0.31, 0.30)), slat), b.inputs['Base Color'])
    return finish(mat, nt, out, b.outputs[0], dict(c=[0.3, 0.31, 0.3], r=0.45, m=0.7))


def mat_lamp_corr(C, name):
    mat, nt, out = new_material(name); nb = NB(nt)
    I, Tn, B, Aa = nb.attr('win')
    b = principled(nt, **{'Base Color': (0.5, 0.5, 0.48, 1.0), 'Roughness': 0.4})
    nt.links.new(nb.blackbody(nb.m('MULTIPLY', Tn, 10000.0)), b.inputs['Emission Color'])
    nt.links.new(nb.m('MULTIPLY', I, C['E'] * 1.4), b.inputs['Emission Strength'])
    return finish(mat, nt, out, b.outputs[0], dict(c=[0.5, 0.5, 0.48], r=0.4, m=0.0, e=[1.0, 0.8, 0.5], es=1.0))


def mat_far(C, name, litimg):
    """遠景: 箱の4面にこの材質1枚で、窓・柱・梁・腰壁・点灯・汚れを描く。点灯は litmap(窓ごとの表)を引く。"""
    mat, nt, out = new_material(name); nb = NB(nt)
    S = C['S']; P = S['P']; Hf = S['Hf']; G = S['G']; kind = C['kind']
    u, v = nb.uv()
    up = nb.m('GREATER_THAN', v, G - 1e-4)
    vg = nb.m('SUBTRACT', v, G)
    fidx = nb.m('MULTIPLY', up, nb.m('ADD', nb.m('FLOOR', nb.m('DIVIDE', vg, Hf)), 1.0))
    vf = nb.m('ADD', nb.m('MULTIPLY', up, nb.m('MULTIPLY', nb.m('FRACT', nb.m('DIVIDE', vg, Hf)), Hf)), nb.m('MULTIPLY', nb.m('SUBTRACT', 1.0, up), v))
    um = nb.m('MULTIPLY', nb.m('FRACT', nb.m('DIVIDE', u, P)), P)
    cellu = nb.m('FLOOR', nb.m('DIVIDE', u, P))

    def inside(r):
        return nb.m('MULTIPLY', nb.m('MULTIPLY', nb.m('GREATER_THAN', um, r[0]), nb.m('LESS_THAN', um, r[1])),
                    nb.m('MULTIPLY', nb.m('GREATER_THAN', vf, r[2]), nb.m('LESS_THAN', vf, r[3])))
    if kind == 'office':
        up_r = (0.05, P - 0.05, S['SILL'], S['SILL'] + S['VIS']); gr_r = (0.05, P - 0.05, 0.6, 4.2)
    elif kind == 'zakkyo':
        up_r = (S['JAMB'], P - S['JAMB'], S['SILL'], S['SILL'] + S['WIN_H']); gr_r = (S['JAMB'], P - S['JAMB'], 0.15, 3.0)
    else:
        z0w = S['RAIL_H'] if C['rail_solid'] else S['WIN_Z']
        up_r = ((P - S['WIN_W']) / 2, (P + S['WIN_W']) / 2, z0w, S['WIN_Z'] + S['WIN_H']); gr_r = (0.3, P - 0.3, 0.1, 2.9)
    M = nb.m('ADD', nb.m('MULTIPLY', up, inside(up_r)), nb.m('MULTIPLY', nb.m('SUBTRACT', 1.0, up), inside(gr_r)))
    # 壁側
    dirt = nb.m('MULTIPLY', dirt_socket(nb, u, v, C, True), 1.0 if kind != 'zakkyo' else 0.6)
    A = C['A']
    wcol = nb.rgb(C['wall_base'])
    if kind == 'office':
        mull = nb.m('MAXIMUM', nb.m('LESS_THAN', um, 0.05), nb.m('GREATER_THAN', um, P - 0.05))
        wcol = nb.vmix(nb.rgb(C['spandrel_col']), nb.rgb(C['sash_col']), mull)
    elif kind == 'zakkyo':
        pil = nb.m('MAXIMUM', nb.m('LESS_THAN', um, S['PIL_W'] / 2), nb.m('GREATER_THAN', um, P - S['PIL_W'] / 2))
        beam = nb.m('GREATER_THAN', vf, Hf - S['BEAM_H'])
        wcol = nb.vmix(nb.vmix(nb.rgb(C['wall_base']), nb.rgb(C['pillar_base']), pil), nb.rgb(C['beam_base']), beam)
    else:
        slab = nb.m('GREATER_THAN', vf, Hf - S['SLAB'])
        rail = nb.m('LESS_THAN', vf, S['RAIL_H'])
        wcol = nb.vmix(nb.vmix(nb.rgb(C['wall_base']), nb.rgb(C['rail_base']), nb.m('MULTIPLY', rail, up)), nb.rgb(C['slab_base']), nb.m('MULTIPLY', slab, up))
    dark = nb.m('MINIMUM', nb.m('ADD', A * 0.35, nb.m('MULTIPLY', dirt, A * 0.65)), 0.75)
    wall = principled(nt, **{'Roughness': 0.8})
    nt.links.new(nb.vscale(wcol, nb.m('SUBTRACT', 1.0, dark)), wall.inputs['Base Color'])
    # 窓側
    li = nb.node('ShaderNodeTexImage'); li.image = litimg; li.interpolation = 'Closest'; li.extension = 'EXTEND'
    nt.links.new(nb.vec(nb.m('DIVIDE', nb.m('ADD', cellu, 0.5), float(C['ncell'])),
                        nb.m('DIVIDE', nb.m('ADD', fidx, 0.5), float(C['floors'])), 0.0), li.inputs['Vector'])
    sc = nb.node('ShaderNodeSeparateColor'); nt.links.new(li.outputs['Color'], sc.inputs[0])
    Il, Tl = sc.outputs[0], sc.outputs[1]
    gl = principled(nt, **{'Base Color': (*C['glass_tint'], 1.0), 'Metallic': C['glass_metal'], 'Roughness': 0.03, 'IOR': 1.5})
    nt.links.new(nb.blackbody(nb.m('MULTIPLY', Tl, 10000.0)), gl.inputs['Emission Color'])
    nt.links.new(nb.m('MULTIPLY', Il, C['E'] * 0.55), gl.inputs['Emission Strength'])
    mx = nb.node('ShaderNodeMixShader'); nt.links.new(M, mx.inputs[0]); nt.links.new(wall.outputs[0], mx.inputs[1]); nt.links.new(gl.outputs[0], mx.inputs[2])
    return finish(mat, nt, out, mx.outputs[0], dict(c=list(C['wall_base']), r=0.6, m=0.0))


# ------------------------------------------------------------------------------------------------
# 色の組み合わせ・役割 → 材質
# ------------------------------------------------------------------------------------------------
TILE_SETS = [   # (壁, 柱型, 梁型) 新品のアルベド(リニア)。 白・ベージュ・茶・灰★(allabout)
    ((0.52, 0.43, 0.31), (0.60, 0.60, 0.58), (0.60, 0.60, 0.58)),
    ((0.60, 0.60, 0.58), (0.22, 0.13, 0.09), (0.60, 0.60, 0.58)),
    ((0.30, 0.17, 0.11), (0.52, 0.43, 0.31), (0.52, 0.43, 0.31)),
    ((0.38, 0.38, 0.39), (0.60, 0.60, 0.58), (0.38, 0.38, 0.39)),
    ((0.58, 0.50, 0.34), (0.26, 0.15, 0.10), (0.58, 0.50, 0.34)),
]
PAINT_SETS = [(0.55, 0.47, 0.36), (0.62, 0.61, 0.58), (0.50, 0.50, 0.50), (0.45, 0.34, 0.25), (0.58, 0.55, 0.46)]
GLASS_SETS = [((0.020, 0.045, 0.050), 0.30), ((0.040, 0.045, 0.050), 0.20), ((0.050, 0.040, 0.035), 0.25), ((0.015, 0.030, 0.060), 0.35)]
SASH_SETS = [((0.50, 0.51, 0.53), 0.9), ((0.04, 0.04, 0.045), 0.5)]        # シルバー(陽極酸化)、黒(推: base0.03〜0.05, metal0.5)


def make_context(ctx, emit):
    kind = ctx.kind; S = ctx.S; r = random.Random(ctx.seed * 977 + 3)
    gt, gm = r.choice(GLASS_SETS)
    sc, sm = r.choice(SASH_SETS)
    C = dict(kind=kind, S=S, W=ctx.W, D=ctx.D, H=ctx.H, seed=ctx.seed % 997, detail=ctx.detail, A=0.1 + 0.5 * ctx.age, E=emit, night=ctx.night,
             glass_tint=gt, glass_metal=gm if kind == 'office' else 0.05, glass_refl_min=0.10 if kind == 'office' else 0.04,
             sash_col=sc if kind == 'office' else (0.45, 0.46, 0.48), sash_metal=sm if kind == 'office' else 0.85,
             spandrel_col=(0.05, 0.065, 0.075), rail_solid=ctx.rail_solid, ncell=ctx.ncell, floors=ctx.floors)
    P = S['P']
    if kind == 'office':
        C.update(edges=[0.05, P - 0.05], sill=S['SILL'], wall_base=(0.42, 0.42, 0.42), pillar_base=r.choice([(0.55, 0.55, 0.55), (0.35, 0.35, 0.37)]),
                 beam_base=(0.40, 0.41, 0.43))
    elif kind == 'zakkyo':
        w, p_, b_ = r.choice(TILE_SETS)
        C.update(edges=[S['JAMB'], P - S['JAMB']], sill=S['SILL'], wall_base=w, pillar_base=p_, beam_base=b_)
    else:
        wb = r.choice(PAINT_SETS)
        C.update(edges=[(P - S['WIN_W']) / 2, (P + S['WIN_W']) / 2], sill=S['WIN_Z'], wall_base=wb, pillar_base=tuple(c * 0.55 for c in wb),
                 beam_base=wb, slab_base=(0.58, 0.58, 0.56), rail_base=wb if ctx.rail_solid else (0.07, 0.07, 0.08))
    C.setdefault('slab_base', (0.58, 0.58, 0.56))
    C.setdefault('rail_base', C['wall_base'])
    return C


def build_material(role, C, name, litimg):
    kind = C['kind']; base = role.split('|')[0]
    if base in ('glass', 'glass_lit'):
        fr = None
        if '|' in role:
            p = role.split('|')[1:]; fr = (float(p[0]), float(p[1]), float(p[2]), int(p[3]))
        return mat_glass(C, name, base == 'glass_lit', fr)
    if role == 'far':
        return mat_far(C, name, litimg)
    if role == 'wall':
        if kind == 'zakkyo':
            return mat_wall(C, name, C['wall_base'], tile=True)
        return mat_wall(C, name, C['wall_base'], tile=False, streak=(kind != 'office'))
    if role == 'pillar':
        if kind == 'zakkyo':
            return mat_wall(C, name, C['pillar_base'], tile=True, streak=False, dirt_k=0.8)
        if kind == 'office':
            return mat_wall(C, name, C['pillar_base'], tile=False, streak=False, dirt_k=0.5, rough=0.5)
        return mat_wall(C, name, C['pillar_base'], tile=False, streak=False)
    if role == 'beam':
        if kind == 'zakkyo':
            return mat_wall(C, name, C['beam_base'], tile=True, streak=False, dirt_k=1.0)
        return mat_wall(C, name, C['beam_base'], tile=False, streak=False, dirt_k=0.6, rough=0.5)
    if role == 'slab':
        return mat_wall(C, name, C['slab_base'], tile=False, streak=False, dirt_k=0.9)
    if role == 'pent':
        return mat_wall(C, name, (0.40, 0.40, 0.38), tile=False, streak=False, dirt_k=0.8)
    if role == 'roof':
        return mat_plain(C, name, (0.22, 0.22, 0.21), 0.9)
    if role == 'spandrel':
        return mat_plain(C, name, C['spandrel_col'], 0.08, 0.45)
    if role == 'sash':
        return mat_plain(C, name, C['sash_col'], 0.42, C['sash_metal'])
    if role == 'base':
        return mat_plain(C, name, (0.035, 0.035, 0.04), 0.35, 0.1)
    if role == 'canopy':
        return mat_plain(C, name, (0.03, 0.03, 0.035), 0.4, 0.6)
    if role.startswith('sign'):
        cols = [(0.62, 0.62, 0.60), (0.55, 0.40, 0.10), (0.04, 0.17, 0.15), (0.45, 0.06, 0.05)]
        c = cols[int(role[4:]) % 4]
        return mat_plain(C, name, c, 0.5, 0.0, emit=c if C['night'] else None, emit_strength=1.6)
    if role == 'shutter':
        return mat_shutter(C, name)
    if role == 'door':
        return mat_plain(C, name, (0.07, 0.08, 0.09), 0.4, 0.5)
    if role == 'dark':
        return mat_plain(C, name, (0.008, 0.008, 0.01), 0.8)
    if role == 'ac':
        return mat_plain(C, name, (0.60, 0.61, 0.60), 0.5)
    if role == 'equip':
        return mat_plain(C, name, (0.32, 0.40, 0.36), 0.55, 0.3)
    if role == 'equip2':
        return mat_plain(C, name, (0.45, 0.46, 0.47), 0.5, 0.8)
    if role == 'tank':
        return mat_plain(C, name, (0.55, 0.58, 0.60), 0.45)
    if role == 'stair':
        return mat_plain(C, name, (0.07, 0.05, 0.045), 0.6, 0.6)
    if role == 'rail_metal':
        return mat_plain(C, name, (0.05, 0.05, 0.055), 0.45, 0.8)
    if role == 'rail_bars':
        return mat_bars(C, name)
    if role == 'lamp_red':
        return mat_plain(C, name, (0.2, 0.01, 0.01), 0.5, 0.0, emit=(1.0, 0.03, 0.02), emit_strength=8.0)
    if role == 'lamp_corr':
        return mat_lamp_corr(C, name)
    if role == 'room_wall':
        return mat_room(C, name, 'wall', 'office')
    if role == 'room_floor':
        return mat_room(C, name, 'floor', 'office')
    if role.startswith('room_ceil_'):
        return mat_room(C, name, 'ceil', role[len('room_ceil_'):])
    if role.startswith('room_blind_'):
        return mat_room(C, name, 'blind', role[len('room_blind_'):])
    raise KeyError('unknown role: ' + role)


# ------------------------------------------------------------------------------------------------
# Blender オブジェクト化
# ------------------------------------------------------------------------------------------------
def _litmap_image(name, lit):
    f, c = lit.shape[0], lit.shape[1]
    img = bpy.data.images.new(name, width=c, height=f, alpha=True, float_buffer=True)
    px = np.zeros((f, c, 4), np.float32); px[:, :, 0:3] = lit[:, :, 0:3]; px[:, :, 3] = 1.0
    img.pixels.foreach_set(px.ravel())
    try:
        img.colorspace_settings.name = 'Non-Color'
    except Exception:
        pass
    img.pack()
    return img


def make_building(kind, width, depth, floors, seed=0, detail='near', night=True, hour=20.0, age=None, name=None, emit=0.9, collection=None):
    """ビル1棟を作る。 -> 親Empty(子 = メッシュ)。 kind: 'office' | 'zakkyo' | 'mansion'(日本語名も可)。
    width/depth はモジュールの整数倍に丸める。 hour = 夜の時刻(窓の点灯率)。 age = 0(新築)〜1(築40年超)、省略時は種類と seed から。
    emit = 窓の明るさ(発光の強さ)。 collection = 入れる Collection(省略時は現在のシーン)。"""
    t0 = time.time()
    md, info, ctx = build_geometry(kind, width, depth, floors, seed, detail, night, hour, age)
    kind = ctx.kind
    nm = name or '%s_s%d_%s' % (kind, int(seed), detail)
    C = make_context(ctx, emit)
    litimg = _litmap_image(nm + '_litmap', ctx.lit) if detail == 'far' else None
    mats = [build_material(role, C, '%s_%s' % (nm, role.replace('|', '_')), litimg) for role in md.mats]
    me = bpy.data.meshes.new(nm + '_mesh')
    me.from_pydata([tuple(v) for v in md.V], [], [tuple(f) for f in md.F])
    uvl = me.uv_layers.new(name='UVMap')
    uvl.data.foreach_set('uv', np.asarray(md.UV, np.float32).ravel())
    ca = me.color_attributes.new('win', 'FLOAT_COLOR', 'CORNER')
    ca.data.foreach_set('color', np.asarray(md.WIN, np.float32).ravel())
    me.polygons.foreach_set('material_index', np.asarray(md.MI, np.int32))
    for m in mats:
        me.materials.append(m)
    me.update()
    body = bpy.data.objects.new(nm + '_body', me)
    root = bpy.data.objects.new(nm, None); root.empty_display_size = 2.0
    coll = collection or bpy.context.scene.collection
    coll.objects.link(root); coll.objects.link(body)
    body.parent = root
    info['gen_seconds'] = round(time.time() - t0, 3)
    info['materials'] = len(mats)
    root['kit_info'] = json.dumps(info, ensure_ascii=False)
    return root


def building_info(root):
    return json.loads(root['kit_info'])


def simplify_materials(root):
    """書き出し(glTF)用に、手続き材質を単色の Principled に置き換える(汚れは平均色に焼き込み、点灯は発光色)。"""
    cache = {}
    for ob in [root] + list(root.children_recursive):
        if ob.type != 'MESH':
            continue
        for slot in ob.material_slots:
            m = slot.material
            if m is None or 'kit_simple' not in m:
                continue
            key = m['kit_simple']
            if key not in cache:
                s = json.loads(key)
                nm, nt_, out_ = new_material('S_' + m.name)
                b = principled(nt_)
                nt_.links.new(b.outputs['BSDF'], out_.inputs['Surface'])
                b.inputs['Base Color'].default_value = (*s['c'], 1.0)
                b.inputs['Roughness'].default_value = s.get('r', 0.5); b.inputs['Metallic'].default_value = s.get('m', 0.0)
                if 'e' in s:
                    b.inputs['Emission Color'].default_value = (*s['e'], 1.0); b.inputs['Emission Strength'].default_value = s.get('es', 1.0)
                if 'a' in s:
                    b.inputs['Alpha'].default_value = s['a']
                    for attr_, val in (('surface_render_method', 'BLENDED'), ('blend_method', 'BLEND')):
                        try:
                            setattr(nm, attr_, val)
                        except Exception:
                            pass
                cache[key] = nm
            slot.material = cache[key]


def export_glb(root, path):
    """root(親Empty+子)だけを .glb に書き出す。 材質は simplify_materials 済みであること。 glTFは +Y 上(Blenderの既定変換)。"""
    for o in bpy.data.objects:
        o.select_set(False)
    for o in [root] + list(root.children_recursive):
        o.select_set(True)
    bpy.context.view_layer.objects.active = root
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=True, export_apply=False)


# ------------------------------------------------------------------------------------------------
# 調査の仕様との照合(bpy不要。作った形から測る)
# ------------------------------------------------------------------------------------------------
def _bbs(md, idxs):
    out = []
    for i in idxs:
        V = np.array([md.V[j] for j in md.F[i]])
        out.append((V.min(0), V.max(0)))
    return out


def _sel(md, pred):
    return [i for i, r in enumerate(md.MI) if pred(md.mats[r])]


def spec_report(seed=3):
    """作った形から寸法を測り、調査の仕様(リサーチ/2026-11-19_現実のビルの構造と汚れ_外観.md)と並べる。 -> [dict(item, spec, got, ok, src)]"""
    rows = []

    def add(item, spec, got, ok, src):
        rows.append(dict(item=item, spec=spec, got=got, ok=bool(ok), src=src))
    # ---- オフィス
    md, info, ctx = build_geometry('office', 33, 26.4, 16, seed, 'near', True, 20)
    y0 = -ctx.D / 2; S = ctx.S
    gl = [i for i in _sel(md, lambda r: r == 'glass' or r == 'glass_lit')]
    up = [(lo, hi) for lo, hi in _bbs(md, gl) if abs(hi[1] - lo[1]) < 1e-6 and abs(lo[1] - (y0 + S['REC'])) < 1e-6 and lo[2] > 6 and hi[0] - lo[0] > 1.0]
    vis = up[0][1][2] - up[0][0][2]
    add('オフィス 階高', '約4.0m(推)', '%.2fm' % ctx.S['Hf'], abs(ctx.S['Hf'] - 4.0) < 1e-6, '推')
    add('オフィス ビジョンガラス高', '2.4〜2.7m(推)', '%.2fm' % vis, 2.4 <= vis <= 2.7, '推')
    z0f = ctx.zf(2)                                    # 2階の1スパン(下の腰 + 上の帯)
    sp = [(lo, hi) for lo, hi in _bbs(md, _sel(md, lambda r: r == 'spandrel')) if lo[2] >= z0f - 1e-6 and hi[2] <= z0f + S['Hf'] + 1e-6
          and abs(lo[1] - (y0 + S['REC'])) < 1e-6 and hi[0] - lo[0] > 1.0 and abs(lo[0] + ctx.W / 2) < 1e-6]
    spz = [round(hi[2] - lo[2], 3) for lo, hi in sp]
    sp_total = sum(spz)
    add('オフィス スパンドレル帯(下+上)', '1.3〜1.6m(推)', '%.2fm (%s)' % (sp_total, '+'.join('%.2f' % z for z in spz)), 1.3 <= sp_total <= 1.6, '推')
    add('オフィス 横ピッチ(1スパン)', '1.5〜1.8m(推)', '%.2fm' % S['P'], 1.5 <= S['P'] <= 1.8, '推')
    mull = [(lo, hi) for lo, hi in _bbs(md, _sel(md, lambda r: r == 'sash')) if hi[2] - lo[2] > 3.9 and lo[1] >= y0 - 1e-6 and hi[1] <= y0 + 0.3 and abs(lo[0]) < ctx.W / 2 - 1]
    fw = [hi[0] - lo[0] for lo, hi in mull if hi[0] - lo[0] > 1e-6 and hi[1] - lo[1] < 1e-6]
    dp = [hi[1] - lo[1] for lo, hi in mull if hi[0] - lo[0] < 1e-6]
    add('オフィス マリオンの見付け(正面幅)', '60〜120mm(推)', '%.0fmm' % (min(fw) * 1000), 0.06 <= min(fw) <= 0.12, '推')
    add('オフィス マリオンの出(奥行)', '150〜250mm(推)', '%.0fmm' % (min(dp) * 1000), 0.15 <= min(dp) <= 0.25, '推')
    add('オフィス ガラスの奥まり(マリオン外面から)', '30〜60mm(推)', '%.0fmm' % (S['REC'] * 1000), 0.03 <= S['REC'] <= 0.06, '推')
    pil = [(lo, hi) for lo, hi in _bbs(md, _sel(md, lambda r: r == 'pillar')) if hi[1] - lo[1] < 1e-6 and abs(hi[1] - (y0 + S['COL_OUT'] * -1)) < 1e-3 or False]
    pf = [(lo, hi) for lo, hi in _bbs(md, _sel(md, lambda r: r == 'pillar')) if hi[2] - lo[2] > 60 and hi[1] - lo[1] < 1e-6 and lo[1] < y0]
    pout = -(pf[0][0][1] - y0) if pf else float('nan')
    add('オフィス 柱型の出(壁面から)', '150〜250mm(推)', '%.0fmm' % (pout * 1000), 0.15 - 1e-6 <= pout <= 0.25, '推')
    fa = [(lo, hi) for lo, hi in _bbs(md, _sel(md, lambda r: r == 'beam')) if hi[0] - lo[0] > 20 and hi[1] - lo[1] < 1e-6 and lo[2] > 6]
    add('オフィス 梁型(各階の水平帯)', '各階にあり(推)', '高さ%.2fm・マリオン外面から%.0fmm出る' % (fa[0][1][2] - fa[0][0][2], (y0 - fa[0][0][1]) * 1000) if fa else 'なし', bool(fa), '推')
    ritems = info['roof']
    add('オフィス 屋上設備', '塔屋・冷却塔・キュービクル・室外機・アンテナ(推)', ', '.join(sorted(set(ritems['items']))), {'pent', 'cooling', 'cubicle'} <= set(ritems['items']), '推')
    add('オフィス 赤い障害灯(60m以上)', '高さ60m以上で必須(★)', '高さ%.1fm → %d個' % (info['height_m'], ritems['red_lights']), ritems['red_lights'] > 0 and info['height_m'] >= 60, '★')
    # ---- 雑居ビル
    md, info, ctx = build_geometry('zakkyo', 11.7, 15.6, 8, seed, 'near', True, 20)
    y0 = -ctx.D / 2; S = ctx.S
    add('雑居 階高', '3.5〜3.8m(推)', '%.2fm' % S['Hf'], 3.5 <= S['Hf'] <= 3.8, '推')
    pf = [(lo, hi) for lo, hi in _bbs(md, _sel(md, lambda r: r == 'pillar')) if hi[1] - lo[1] < 1e-6 and lo[1] < y0 - 0.1 and hi[2] - lo[2] < 4]
    pw = sorted(set(round(hi[0] - lo[0], 3) for lo, hi in pf))
    pw_mid = [w for w in pw if 0.5 < w < 0.9]
    pout = -(pf[0][0][1] - y0)
    add('雑居 柱型の幅', '600〜800mm(推)', '%.0fmm' % (pw_mid[0] * 1000), 0.6 <= pw_mid[0] <= 0.8, '推')
    add('雑居 柱型の出', '150〜250mm(推)', '%.0fmm' % (pout * 1000), 0.15 - 1e-6 <= pout <= 0.25, '推')
    bm = [(lo, hi) for lo, hi in _bbs(md, _sel(md, lambda r: r == 'beam')) if hi[0] - lo[0] > 10 and hi[1] - lo[1] < 1e-6 and lo[2] > 5 and hi[2] - lo[2] > 0.3]
    add('雑居 梁型の帯の高さ', '500〜800mm(推)', '%.0fmm' % ((bm[0][1][2] - bm[0][0][2]) * 1000), 0.5 <= bm[0][1][2] - bm[0][0][2] <= 0.8, '推')
    gl = [(lo, hi) for lo, hi in _bbs(md, _sel(md, lambda r: r == 'glass' or r == 'glass_lit')) if abs(hi[1] - lo[1]) < 1e-6 and abs(lo[1] - (y0 + S['REC'])) < 1e-6 and lo[2] > 5 and hi[0] - lo[0] > 2]
    wz0 = gl[0][0][2]; flo = [ctx.zf(f) for f in range(ctx.floors)]
    fl_ = max(z for z in flo if z <= wz0 + 1e-6)
    add('雑居 窓(腰窓)の寸法', '幅1.65m×高1.1m(推。住宅腰窓の例)', '幅%.2fm(2枚)×高%.2fm' % (gl[0][1][0] - gl[0][0][0], gl[0][1][2] - gl[0][0][2]), False,
        '推(ずれ: 階高3.6mに合わせ高さ1.5mを採用。幅は柱間2.9m)')
    add('雑居 腰壁の高さ', '約0.9m(推)', '%.2fm' % (wz0 - fl_), abs(wz0 - fl_ - 0.9) < 0.05, '推')
    add('雑居 窓の奥まり', '80〜150mm(推)', '%.0fmm' % (S['REC'] * 1000), 0.08 <= S['REC'] <= 0.15, '推')
    sh = [(lo, hi) for lo, hi in _bbs(md, _sel(md, lambda r: r == 'sash')) if hi[2] - lo[2] < 0.06 and hi[0] - lo[0] > 1]
    add('雑居 サッシ枠の見付け', '(数値出典なし)', '%.0fmm' % (S['SASH_W'] * 1000), True, '推')
    add('雑居 外壁タイル', '二丁掛60×227mm、目地約8〜10mm、白・ベージュ・茶・灰(★/推)', 'タイル60×227mm+目地9mm(材質のシェーダ定数)、色は%d組から選択' % len(TILE_SETS), True, '★')
    add('雑居 非常階段・室外機・看板', '側面に外付け鉄骨階段、窓下に室外機、看板は文字なし(推)', '階段=%s、室外機=あり(near)、看板=文字なし色面' % ('あり' if any(r == 'stair' for r in md.mats) else 'なし'), 'stair' in md.mats, '推')
    # ---- マンション
    md, info, ctx = build_geometry('mansion', 26.4, 13.2, 10, seed, 'near', True, 20)
    y0 = -ctx.D / 2; S = ctx.S
    add('マンション 階高', '約3.0m(★)', '%.2fm' % S['Hf'], abs(S['Hf'] - 3.0) < 1e-6, '★')
    sl = [(lo, hi) for lo, hi in _bbs(md, _sel(md, lambda r: r == 'slab')) if hi[0] - lo[0] > 20 and lo[1] < y0 - 1.0 and hi[2] - lo[2] < 0.3]
    add('マンション バルコニーの出', '1.2〜1.5m(推)', '%.2fm' % (y0 - sl[0][0][1]), 1.2 <= y0 - sl[0][0][1] <= 1.5, '推')
    add('マンション 床スラブ小口の厚み', '約200mm(★)', '%.0fmm' % ((sl[0][1][2] - sl[0][0][2]) * 1000), abs(sl[0][1][2] - sl[0][0][2] - 0.2) < 0.01, '★')
    zr = ctx.zf(3)
    if ctx.rail_solid:
        wl = [(lo, hi) for lo, hi in _bbs(md, _sel(md, lambda r: r == 'wall')) if hi[0] - lo[0] > 20 and abs(lo[1] - (y0 - 1.4 + 0.1 - 0.1 + 0.0)) < 0.2 and abs(lo[2] - zr) < 1e-6]
        rh = (wl[0][1][2] - zr) if wl else float('nan'); kind_r = 'solid'
    else:
        wl = [(lo, hi) for lo, hi in _bbs(md, _sel(md, lambda r: r == 'rail_metal')) if hi[0] - lo[0] > 20 and abs(hi[2] - zr - 1.1) < 0.2]
        rh = (max(h[2] for l, h in wl) - zr) if wl else float('nan'); kind_r = 'bars'
    add('マンション 手すりの高さ(%s型)' % kind_r, '1.1m以上(★)', '%.2fm' % rh, rh >= 1.1 - 1e-6, '★')
    gl = [(lo, hi) for lo, hi in _bbs(md, _sel(md, lambda r: r == 'glass' or r == 'glass_lit')) if abs(hi[1] - lo[1]) < 1e-6 and abs(lo[1] - (y0 + S['REC'])) < 1e-6 and lo[2] > 4 and 1.0 < hi[0] - lo[0] < 2.5]
    add('マンション 掃き出し窓', '幅1.65×高2.0m(推)', '幅%.2fm×高%.2fm' % (gl[0][1][0] - gl[0][0][0], gl[0][1][2] - gl[0][0][2]), abs(gl[0][1][0] - gl[0][0][0] - 1.65) < 0.01 and abs(gl[0][1][2] - gl[0][0][2] - 2.0) < 0.01, '推')
    add('マンション 窓の奥まり', '80〜150mm(推)', '%.0fmm' % (S['REC'] * 1000), 0.08 <= S['REC'] <= 0.15, '推')
    add('マンション 室外機・外廊下・屋上', '室外機・外廊下(背面)・塔屋/高架水槽/アンテナ(推)', ', '.join(sorted(set(info['roof']['items']))), {'pent', 'tank', 'antenna'} <= set(info['roof']['items']), '推')
    # ---- 夜の窓(点灯率・色温度)
    for kind, hours, rng_ in (('office', (20, 23.5), ((0.40, 0.60), (0.10, 0.25))), ('mansion', (20, 23.5), ((0.30, 0.50), (0.10, 0.20))), ('zakkyo', (20, 23.5), ((0.30, 0.55), (0.10, 0.25)))):
        for h, (lo_, hi_) in zip(hours, rng_):
            vals = []
            for s_ in range(40):
                lit = make_lit(kind, 80 if kind != 'mansion' else 80, 14, s_, h, True)
                vals.append(float((lit[1:, :, 0] > 0).mean()))
            m_ = float(np.mean(vals))
            add('夜の窓 点灯率 %s %.0f時' % (kind, h), '%d〜%d%%(推)' % (lo_ * 100, hi_ * 100), '平均%.0f%%' % (m_ * 100), lo_ - 0.03 <= m_ <= hi_ + 0.03, '推')
    lit = make_lit('office', 80, 14, 1, 20, True)
    T = lit[1:, :, 1][lit[1:, :, 0] > 0] * 1e4
    add('夜の窓 オフィスの色温度', '4000〜5000K(白、推)', '%.0f〜%.0fK(平均%.0fK)' % (T.min(), T.max(), T.mean()), T.min() >= 3800 and T.max() <= 5200, '推')
    lit = make_lit('mansion', 80, 14, 1, 20, True)
    T = lit[1:, :, 1][lit[1:, :, 0] > 0] * 1e4
    add('夜の窓 マンションの色温度', '2700〜3000K主体(暖色、推)', '中央値%.0fK(2700〜3000K=%d%%、混ぜ込み: 3500〜5000K・TV 6200K)' % (np.median(T), 100 * np.mean((T >= 2690) & (T <= 3010))), 2690 <= np.median(T) <= 3010, '推')
    return rows
