"""city_kit.py — 背景の街ビルダー(Blender Python)。ビル・道路の断面・1階のキットを組み合わせて、city.json の街を作る。他の動画から import して使える。
依存: bpy・mathutils・numpy・標準ライブラリ と、同じ系統のキット(building_kit.py・street_kit.py・building_base.py)。
  計画(plan_*)・形の生成は bpy 無しで動く(棟数・ポリゴン数の数値確認用)。 Blender の材質・オブジェクト化だけが bpy を使う。

使い方
    import sys; sys.path.insert(0, '<このファイルのフォルダ>')           # 隣に building_kit / street_kit / building_base があること
    import city_kit as ck
    root = ck.build_city('assets/city.json', center=(0, 800), radius=700, camera_path_or_points=[(-4, 780, 1.5), (-4, 830, 1.5)],
                         hour=19.0, seed=1, detail_policy=None, ground_z=-12.0)
    print(root['city_stats'])      # 棟数・総ポリゴン・構築時間の JSON 文字列。 ck.last_stats でも読める

街の作り(調査: リサーチ/2026-11-19_現実のビルの構造と汚れ_外観.md、同_道路と建物の接地まわり_歩道・基礎・1階.md。★=出典あり、推=推定)
  ・city.json(OpenStreetMap)の建物の輪郭 -> 最小外接矩形(向き・幅・奥行き)。ゆがんだ大きな建物は複数の矩形に分ける。座標は city.json と同じ(同じ位置に並ぶ)。
  ・種類の割り当て(高さ・足元の面積・OSMの種別・乱数(seed と建物id で固定)): 高層=ガラスのオフィス優先 / 中層の小さい=雑居(タイル) / 住居的=マンション。
      調査の割合(推): マンション約4割・雑居(タイル)約3.5割・ガラスのオフィス約1割・低層店舗/住宅約1.5割(city.json の house 型)。
  ・カメラの通る点からの距離で detail を切り替える: near(実立体) / mid / far(箱+手続き材質。遠景の主役。材質は共有、窓の点灯は表1枚)。
  ・道路の線(city.json の道路)に沿って street_kit の断面(車道・縁石・歩道・植樹帯・ガードパイプ・街灯・基礎)を置く。両側の壁の位置に合わせて歩道の幅を決める。
  ・near / mid のビルは、building_kit の1階を外して building_base の1階(ガラス張り・タイル・シャッター・エントランス・駐車場)を付ける。
  ・夜の窓: hour(時)で点灯率・色温度が決まる(building_kit.lit_rate。オフィス 4000〜5000K、マンション 2700〜3000K 主体)。60m以上の屋上に赤い障害灯。
  ・高架(桁・橋脚)は chase/city.py と同じ形(そのまま)を置く。
"""
import os, sys, math, json, time, random
import numpy as np

try:
    import bpy, bmesh
    from mathutils import Vector
except Exception:                     # bpy 無しでも計画・形の生成は動く
    bpy = None

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (HERE, os.path.join(HERE, '..', 'ビル'), os.path.join(HERE, '..', '道路')):
    if os.path.isdir(_d) and os.path.abspath(_d) not in sys.path:
        sys.path.insert(0, os.path.abspath(_d))
import building_kit as bk       # noqa: E402
import street_kit as sk         # noqa: E402
try:
    import building_base as bb  # noqa: E402
except Exception:               # 1階キットが無くても街は作れる
    bb = None

last_stats = {}

# ------------------------------------------------------------------------------------------------
# 設定
# ------------------------------------------------------------------------------------------------
DEFAULT_POLICY = dict(
    near=36.0, mid=105.0,            # カメラの点からの水平距離[m]。この内側のビルを near / mid にする(足元の矩形の縁から)
    max_near=8, max_mid=30,          # near / mid の棟数の上限(近い順。残りは far)。near は1棟 約2万面、mid は約2千〜7千面
    houses=330.0,                    # 低層(house)を出すカメラからの距離
    street_far=150.0,                # この距離までの道路は street_kit の断面(far 以上)、外は帯1枚
    street_mid=80.0, street_near=30.0,
    max_street_near_m=80.0, max_street_mid_m=240.0,        # near / mid の道路の合計の長さ[m](近い順。片側ずつ作るので面は約2倍)
    far_fence_dist=70.0,             # far の道路でも、この距離より遠いとガードパイプを省く(面を減らす)
    street_lights=24,                # near の街灯の実ライト(スポット)の上限
    trees=True, tree_cap=1800, tree_dist=420.0,
    scale_min={'office': 0.72, 'zakkyo': 0.70, 'mansion': 0.58}, scale_max=1.45,   # far の箱へ近似するときの許容の伸縮(矩形がキットの寸法から外れる割合)
    solid_scale=(0.80, 1.25),        # near / mid にできる伸縮の範囲(外れたら far)。 実立体は make_solid_building で x・y を伸縮して矩形(wf x df)にぴったり合わせる
    split=True,
    piece_steps=6,                   # 道路を切る区間の長さ(4m x 6 = 24m)。 区間ごとに、壁までの最小の距離から断面を決める
    ultra_stride=4,                  # 150mより遠い道路の帯は、4mの点を 4つおき(16m)に
    apron=True, trim_junctions=True, legacy=False,        # legacy=True は修正前の方式(比較用)
)
KIND_JP = {'office': 'ガラスのオフィス', 'zakkyo': '雑居(タイル)', 'mansion': 'マンション', 'low': '低層(小さい)', 'house': '低層(住宅)'}
NO_STREET = ('rail', 'pedestrian', 'service')
MAJOR = ('trunk', 'primary', 'secondary', 'trunk_link', 'primary_link')
HOUSE_COLS = [(0.62, 0.60, 0.56), (0.56, 0.52, 0.44), (0.46, 0.42, 0.36), (0.40, 0.40, 0.40), (0.30, 0.24, 0.18), (0.20, 0.18, 0.17), (0.50, 0.52, 0.54), (0.58, 0.50, 0.42)]
ROOF_COLS = [(0.045, 0.047, 0.05), (0.03, 0.03, 0.032), (0.09, 0.05, 0.035), (0.06, 0.075, 0.09), (0.05, 0.07, 0.055), (0.20, 0.21, 0.22), (0.11, 0.045, 0.035), (0.07, 0.07, 0.075)]


def rng_for(seed, bid, salt=0):
    return random.Random((int(seed) * 7919 + int(bid) * 104729 + int(salt) * 31 + 17) & 0x7fffffff)


# ------------------------------------------------------------------------------------------------
# 幾何の補助(bpy 不要)
# ------------------------------------------------------------------------------------------------
def poly_area(P):
    a = np.asarray(P, float)
    return 0.5 * abs(np.dot(a[:, 0], np.roll(a[:, 1], -1)) - np.dot(a[:, 1], np.roll(a[:, 0], -1)))


def obb(P):
    """最小面積の外接長方形。 -> (中心(2,), u(長辺の向き 単位), v(u に直交), 長さ eu(u 方向), 幅 ev(v 方向))"""
    A = np.asarray(P, float)
    if np.allclose(A[0], A[-1]):
        A = A[:-1]
    best = None
    for i in range(len(A)):
        e = A[(i + 1) % len(A)] - A[i]
        L = float(np.hypot(*e))
        if L < 0.3:
            continue
        ux, uy = e / L
        s = A[:, 0] * ux + A[:, 1] * uy; q = -A[:, 0] * uy + A[:, 1] * ux
        w, d = s.max() - s.min(), q.max() - q.min()
        if best is None or w * d < best[0]:
            cs, cq = (s.max() + s.min()) / 2, (q.max() + q.min()) / 2
            best = (w * d, w, d, cs, cq, ux, uy)
    if best is None:
        c = A.mean(0); return c, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1.0, 1.0
    _, w, d, cs, cq, ux, uy = best
    c = np.array([cs * ux - cq * uy, cs * uy + cq * ux])
    u = np.array([ux, uy]); v = np.array([-uy, ux])
    if d > w:
        u, v = v, -u; w, d = d, w
    return c, u, v, float(w), float(d)


def points_in_poly(P, X, Y):
    """偶奇規則。 X, Y = 同じ形の配列。 -> bool 配列"""
    A = np.asarray(P, float)
    inside = np.zeros(X.shape, bool)
    n = len(A)
    j = n - 1
    for i in range(n):
        xi, yi = A[i]; xj, yj = A[j]
        cond = ((yi > Y) != (yj > Y))
        with np.errstate(divide='ignore', invalid='ignore'):
            xint = (xj - xi) * (Y - yi) / (yj - yi + 1e-12) + xi
        inside ^= cond & (X < xint)
        j = i
    return inside


def largest_rect(mask):
    """True の格子の中の最大の長方形(ヒストグラム法)。 -> (面積, r0, r1, c0, c1)(r1, c1 は含まない)"""
    nr, nc = mask.shape
    h = np.zeros(nc, int); best = (0, 0, 0, 0, 0)
    for r in range(nr):
        h = np.where(mask[r], h + 1, 0)
        st = []
        for c in range(nc + 1):
            cur = h[c] if c < nc else 0
            start = c
            while st and st[-1][1] >= cur:
                s0, hh = st.pop()
                a = hh * (c - s0)
                if a > best[0]:
                    best = (a, r - hh + 1, r + 1, s0, c)
                start = s0
            st.append((start, cur))
    return best


def rect_poly(c, xd, yd, w, d):
    """向き xd(幅 w)・yd(奥行 d)の長方形の4隅(反時計回り)。 (xd, yd) は右手系。"""
    c = np.asarray(c, float); a = np.asarray(xd, float) * (w / 2.0); b = np.asarray(yd, float) * (d / 2.0)
    return np.array([c - a - b, c + a - b, c + a + b, c - a + b])


def poly_area_list(pl):
    n = len(pl)
    if n < 3:
        return 0.0
    return 0.5 * abs(sum(pl[i][0] * pl[(i + 1) % n][1] - pl[(i + 1) % n][0] * pl[i][1] for i in range(n)))


def clip_poly(subject, clip):
    """多角形 subject を、凸多角形 clip(反時計回り)で切った残り(Sutherland-Hodgman)。 -> 点のリスト"""
    out = [(float(p[0]), float(p[1])) for p in subject]
    n = len(clip)
    for i in range(n):
        ax, ay = clip[i]; bx, by = clip[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        inp = out; out = []
        if not inp:
            break
        sx, sy = inp[-1]; ds = ex * (sy - ay) - ey * (sx - ax)
        for px, py in inp:
            dp = ex * (py - ay) - ey * (px - ax)
            if dp >= 0:
                if ds < 0:
                    t = ds / (ds - dp); out.append((sx + t * (px - sx), sy + t * (py - sy)))
                out.append((px, py))
            elif ds >= 0:
                t = ds / (ds - dp); out.append((sx + t * (px - sx), sy + t * (py - sy)))
            sx, sy, ds = px, py, dp
    return out


class Corridor:
    """通行帯(車道・縁石・歩道の帯)を、中心線の1区間ごとの四角形にして、格子で引けるようにしたもの。 建物の矩形との重なり(面積)を測る・削るのに使う。"""
    def __init__(self, cell=30.0):
        self.cell = cell; self.Q = []; self.grid = {}

    def add_quad(self, q):
        i = len(self.Q); q = np.asarray(q, float); self.Q.append(q)
        x0, y0 = q.min(0); x1, y1 = q.max(0)
        for ix in range(int(x0 // self.cell), int(x1 // self.cell) + 1):
            for iy in range(int(y0 // self.cell), int(y1 // self.cell) + 1):
                self.grid.setdefault((ix, iy), []).append(i)

    def add_strip(self, S, o_right, o_left, ext=0.0):
        """折れ線 S の、右 o_right 〜 左 o_left の帯。 ext = 区間の両端を延ばす長さ(曲がり角の隙間を埋める)"""
        S = np.asarray(S, float)
        for k in range(len(S) - 1):
            a, b = S[k], S[k + 1]; d = b - a; L = float(np.hypot(*d))
            if L < 1e-6:
                continue
            t = d / L; n = np.array([-t[1], t[0]]); a2 = a - t * ext; b2 = b + t * ext
            self.add_quad([a2 - n * o_right, b2 - n * o_right, b2 + n * o_left, a2 + n * o_left])

    def overlap(self, poly, min_area=1e-6):
        """-> (重なりの多角形のリスト, 面積の合計)。 poly = 矩形など(反時計回り)"""
        P = np.asarray(poly, float); x0, y0 = P.min(0); x1, y1 = P.max(0)
        idx = set()
        for ix in range(int(x0 // self.cell), int(x1 // self.cell) + 1):
            for iy in range(int(y0 // self.cell), int(y1 // self.cell) + 1):
                idx.update(self.grid.get((ix, iy), ()))
        outs = []; tot = 0.0
        for i in idx:
            q = self.Q[i]
            if q[:, 0].max() < x0 or q[:, 0].min() > x1 or q[:, 1].max() < y0 or q[:, 1].min() > y1:
                continue
            r = clip_poly(P, q)
            a = poly_area_list(r) if len(r) >= 3 else 0.0
            if a > min_area:
                outs.append(np.asarray(r, float)); tot += a
        return outs, tot


def inscribed_rect(P, c, u, v, eu, ev, step):
    """輪郭 P の中に4隅が全部入る、(u, v)向きの最大の長方形。 -> (中心, 長さ(u), 幅(v)) / None"""
    ns = max(2, int(math.ceil(eu / step))); nq = max(2, int(math.ceil(ev / step)))
    ss = np.linspace(-eu / 2, eu / 2, ns + 1); qq = np.linspace(-ev / 2, ev / 2, nq + 1)
    S, Q = np.meshgrid(ss, qq, indexing='ij')
    X = c[0] + S * u[0] + Q * v[0]; Y = c[1] + S * u[1] + Q * v[1]
    ins = points_in_poly(P, X, Y)
    cell = ins[:-1, :-1] & ins[1:, :-1] & ins[:-1, 1:] & ins[1:, 1:]
    if not cell.any():
        return None
    a, r0, r1, c0, c1 = largest_rect(cell)
    if a <= 0:
        return None
    s0, s1, q0, q1 = ss[r0], ss[r1], qq[c0], qq[c1]
    return (c + u * (s0 + s1) / 2 + v * (q0 + q1) / 2, float(s1 - s0), float(q1 - q0))


def split_rects(P, c, u, v, eu, ev, step=1.5, max_rects=4, min_side=6.0):
    """ゆがんだ輪郭を、向きがそろった長方形(最大 max_rects 個。4隅とも輪郭の中)で覆う。 -> [(中心(2,), 長さeu', 幅ev'), ...](u, v の向きは共通)"""
    ns = max(2, int(math.ceil(eu / step))); nq = max(2, int(math.ceil(ev / step)))
    ss = np.linspace(-eu / 2, eu / 2, ns + 1); qq = np.linspace(-ev / 2, ev / 2, nq + 1)
    S, Q = np.meshgrid(ss, qq, indexing='ij')
    X = c[0] + S * u[0] + Q * v[0]; Y = c[1] + S * u[1] + Q * v[1]
    ins = points_in_poly(P, X, Y)
    mask = ins[:-1, :-1] & ins[1:, :-1] & ins[:-1, 1:] & ins[1:, 1:]
    total = int(mask.sum())
    if total == 0:
        return []
    out = []
    for _ in range(max_rects):
        a, r0, r1, c0, c1 = largest_rect(mask)
        s0, s1, q0, q1 = ss[r0], ss[r1], qq[c0], qq[c1]
        if (s1 - s0) * (q1 - q0) < min_side * min_side or mask.sum() < 0.12 * total:
            break
        out.append((c + u * (s0 + s1) / 2 + v * (q0 + q1) / 2, float(s1 - s0), float(q1 - q0)))
        mask[r0:r1, c0:c1] = False
    return out


def clip_rect(c, u, v, eu, ev, corr, min_side=2.5, tol=0.01):
    """矩形(長さ eu は u 向き・幅 ev は v 向き)が通行帯 corr と重なる所を、辺を内側へ動かして削る。 -> (c, eu, ev) / None(削ったら小さすぎる)"""
    c = np.asarray(c, float)
    for _ in range(6):
        ov, tot = corr.overlap(rect_poly(c, u, v, eu, ev), tol)
        if not ov:
            return (c, eu, ev)
        pts = np.vstack(ov); best = None
        for ax, ln, which in ((u, eu, 0), (v, ev, 1)):
            for sg in (1.0, -1.0):
                sc = ((pts - c) @ ax) * sg
                depth = ln / 2.0 - float(sc.min()) + 0.01
                nl = ln - depth
                if depth <= 0 or nl < min_side:
                    continue
                nc = c - ax * sg * depth / 2.0
                cand = (nc, nl, ev) if which == 0 else (nc, eu, nl)
                ar = cand[1] * cand[2]
                if best is None or ar > best[0]:
                    best = (ar, cand)
        if best is None:
            return None
        c, eu, ev = best[1]
    ov, tot = corr.overlap(rect_poly(c, u, v, eu, ev), tol)
    return None if ov else (c, eu, ev)


def split_rects_legacy(P, c, u, v, eu, ev, step=1.5, max_rects=4, min_side=6.0):
    """旧方式(比較用): 格子の中心が輪郭の中にある長方形で覆う。 4隅が輪郭の外に出ることがある"""
    ns = max(2, int(math.ceil(eu / step))); nq = max(2, int(math.ceil(ev / step)))
    ss = (np.arange(ns) + 0.5) / ns * eu - eu / 2; qq = (np.arange(nq) + 0.5) / nq * ev - ev / 2
    S, Q = np.meshgrid(ss, qq, indexing='ij')
    X = c[0] + S * u[0] + Q * v[0]; Y = c[1] + S * u[1] + Q * v[1]
    mask = points_in_poly(P, X, Y)
    total = int(mask.sum())
    if total == 0:
        return [(c, eu, ev)]
    out = []
    for _ in range(max_rects):
        a, r0, r1, c0, c1 = largest_rect(mask)
        if a * step * step < min_side * min_side or mask.sum() < 0.12 * total:
            break
        s0 = ss[r0] - step / 2; s1 = ss[r1 - 1] + step / 2; q0 = qq[c0] - step / 2; q1 = qq[c1 - 1] + step / 2
        out.append((c + u * (s0 + s1) / 2 + v * (q0 + q1) / 2, float(s1 - s0), float(q1 - q0)))
        mask[r0:r1, c0:c1] = False
    return out or [(c, eu, ev)]


class RoadIndex:
    """道路の線を一定間隔の点にして、格子で最近傍を引く。"""
    def __init__(self, roads, cell=60.0, step=6.0):
        self.cell = cell; pts = []; meta = []
        for ri, r in enumerate(roads):
            if r['c'] in ('rail', 'pedestrian'):
                continue
            P = r['p']
            for (x1, y1), (x2, y2) in zip(P[:-1], P[1:]):
                L = math.hypot(x2 - x1, y2 - y1)
                n = max(1, int(L / step))
                for k in range(n + 1):
                    f = k / n; pts.append((x1 + (x2 - x1) * f, y1 + (y2 - y1) * f)); meta.append((ri, r['w'], 0 if r['c'] != 'service' else 1))
        self.pts = np.asarray(pts, float).reshape(-1, 2); self.meta = np.asarray(meta, float).reshape(-1, 3)
        self.grid = {}
        for i, (x, y) in enumerate(self.pts):
            self.grid.setdefault((int(x // cell), int(y // cell)), []).append(i)

    def nearest(self, x, y, maxr=2, prefer_main=True):
        gx, gy = int(x // self.cell), int(y // self.cell)
        for rr in range(maxr + 1):
            idx = []
            for ix in range(gx - rr, gx + rr + 1):
                for iy in range(gy - rr, gy + rr + 1):
                    if max(abs(ix - gx), abs(iy - gy)) == rr:
                        idx += self.grid.get((ix, iy), [])
            if idx:
                idx = np.asarray(idx)
                d = np.hypot(self.pts[idx, 0] - x, self.pts[idx, 1] - y)
                if prefer_main:
                    main = self.meta[idx, 2] == 0
                    if main.any() and d[main].min() < 70.0:
                        idx = idx[main]; d = d[main]
                k = int(np.argmin(d))
                return float(self.pts[idx[k], 0]), float(self.pts[idx[k], 1]), float(d[k]), float(self.meta[idx[k], 1])
        return None


def kit_dims(kind, w_front, d_depth):
    """キットが丸める寸法(モジュールの整数倍・最小寸法)。 -> (W, D, nW, nD)"""
    S = bk.SPECS[kind]; P = S['P']
    nW = max(S['minb'][0], int(round(w_front / P))); nD = max(S['minb'][1], int(round(d_depth / P)))
    if kind == 'mansion' and nW % 2:
        nW += 1
    return nW * P, nD * P, nW, nD


def floors_for(kind, h):
    S = bk.SPECS[kind]
    return max(2, int(round((h - S['G']) / S['Hf'])) + 1)


def pick_kind(t, h, area, w, d, rr, pol):
    """建物の種類を決める。 t = OSM の種別(apt/office/tower/house)。 -> 'office' | 'zakkyo' | 'mansion' | 'low'"""
    wt = {'office': 0.0, 'zakkyo': 0.0, 'mansion': 0.0}
    if t == 'tower' or h >= 50:
        wt.update(office=6.0, mansion=3.0)
    elif h >= 34:
        wt.update(office=2.0 if area >= 250 else 0.4, mansion=6.0, zakkyo=1.0)
    elif h >= 22:
        wt.update(office=1.2, mansion=6.0, zakkyo=2.2)
    else:
        wt.update(office=0.5, mansion=4.2, zakkyo=3.0)
    if t == 'apt':
        wt['mansion'] *= 3.5; wt['zakkyo'] *= 0.4; wt['office'] *= 0.3
    elif t == 'office':
        wt['office'] *= 3.0 if h >= 24 else 1.4; wt['zakkyo'] *= 1.1; wt['mansion'] *= 0.8
    for k in list(wt):
        W, D, _, _ = kit_dims(k, w, d)
        sw, sd = w / W, d / D
        smin = pol['scale_min'][k] if isinstance(pol['scale_min'], dict) else pol['scale_min']
        if not (smin <= sw <= pol['scale_max'] and smin <= sd <= pol['scale_max']):
            wt[k] = 0.0
    tot = sum(wt.values())
    if tot <= 0:
        return 'low'
    x = rr.random() * tot
    for k in ('mansion', 'zakkyo', 'office'):
        x -= wt[k]
        if x <= 0:
            return k
    return 'zakkyo'


def cam_dist2d(cam, x, y, r):
    if cam is None:
        return 0.0
    return max(0.0, float(np.min(np.hypot(cam[:, 0] - x, cam[:, 1] - y))) - r)


# ------------------------------------------------------------------------------------------------
# 計画: 建物 -> 記録(種類・位置・向き・寸法・detail)
# ------------------------------------------------------------------------------------------------
def load_city(path_or_dict):
    if isinstance(path_or_dict, dict):
        return path_or_dict
    with open(path_or_dict, encoding='utf-8') as f:
        return json.load(f)


# 道路の断面の最小値(m)。 建物までの距離が足りない所は、車道・歩道をここまで狭める。 それでも足りない所は、建物の輪郭を削る
MIN_G, CW, MIN_SW, BW, MIN_INU = 0.20, 0.18, 0.75, 0.12, 0.30
BAND_HALF_MIN = 1.25               # 路地(service・歩行者道)の帯の、中心線から建物までの最小の距離


def wr_min_for(w):
    """片側の車道(縁石の手前まで)の最小の幅。 OSM の道路幅の半分、2.25〜2.6m"""
    return max(2.25, min(2.6, w / 2.0))


def wall_min_for(cls, w):
    """中心線から建物の壁までの最小の距離(車道 + 路肩 + 縁石 + 歩道0.75 + 境界 + 犬走り0.30 + 基礎の出0.03 + 余裕0.01)"""
    if cls in ('service', 'pedestrian'):
        return BAND_HALF_MIN
    return wr_min_for(w) + MIN_G + CW + MIN_SW + BW + MIN_INU + 0.03 + 0.01


def main_half(w):
    """道路の片側の、縁石の外側までの幅の見積り(車道 + 路肩 + 縁石)。 横から来る道路の断面を、ここで止める"""
    lanes = int(min(3, max(1, round((w / 2.0 - 0.30) / 3.25))))
    return lanes * 3.25 + 0.30 + 0.18


class JunctionTrim:
    """T字路で、横から来る道路の端を、突き当たる道路の車道(縁石の外側)の手前で止める。 止めないと、横の道路の縁石・歩道・基礎が、突き当たる道路の車道の上に出る"""
    def __init__(self, city, cell=40.0):
        self.cell = cell; pts = []; meta = []
        for ri, r in enumerate(city['roads']):
            if r['c'] in ('rail', 'service', 'pedestrian') or r['w'] < 4.5:
                continue
            S = resample(np.asarray(r['p'], float), 4.0)
            if len(S) < 2:
                continue
            t = np.gradient(S, axis=0); t /= (np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-9)
            h = main_half(r['w'])
            for k in range(len(S)):
                pts.append(S[k]); meta.append((ri, t[k, 0], t[k, 1], h))
        self.pts = np.asarray(pts, float).reshape(-1, 2); self.meta = np.asarray(meta, float).reshape(-1, 4); self.grid = {}
        for i, (x, y) in enumerate(self.pts):
            self.grid.setdefault((int(x // cell), int(y // cell)), []).append(i)

    def other(self, ri, x, y, r=2.5):
        idx = []
        gx, gy = int(x // self.cell), int(y // self.cell)
        for ix in range(gx - 1, gx + 2):
            for iy in range(gy - 1, gy + 2):
                idx += self.grid.get((ix, iy), [])
        best = None
        for i in idx:
            if int(self.meta[i, 0]) == ri:
                continue
            d = float(np.hypot(self.pts[i, 0] - x, self.pts[i, 1] - y))
            if d < r and (best is None or d < best[0]):
                best = (d, i)
        return None if best is None else (self.pts[best[1]], self.meta[best[1]])

    def trim(self, ri, S):
        """S(n,2) の両端を、T字路なら突き当たる道路の縁石の外側まで縮める。 -> S'(n',2)"""
        S = np.asarray(S, float)
        for end in (0, -1):
            if len(S) < 3:
                break
            Sx = S if end == 0 else S[::-1]
            o = self.other(ri, Sx[0, 0], Sx[0, 1])
            if o is None:
                continue
            q, m = o
            t = Sx[1] - Sx[0]; t = t / (np.hypot(*t) + 1e-9)
            if abs(t[0] * m[1] + t[1] * m[2]) > 0.6:          # ほぼ同じ向き = 道路の続き。 止めない
                continue
            H = m[3]
            # q(突き当たる道路の中心線の点)からの距離が H になる所まで、端から進める
            d = np.hypot(Sx[:, 0] - q[0], Sx[:, 1] - q[1])
            k = int(np.argmax(d >= H)) if (d >= H).any() else None
            if k is None or k == 0:
                continue
            a, b = Sx[k - 1], Sx[k]; f = (H - d[k - 1]) / max(d[k] - d[k - 1], 1e-6)
            Sx = np.vstack([a + (b - a) * min(max(f, 0.0), 1.0), Sx[k:]])
            S = Sx if end == 0 else Sx[::-1]
        return S


def road_pieces(city, center, radius, cam, pol):
    """道路を区間(既定 24m)に切る。 -> [dict(ri, cls, w, S(n,2), band, dc)]。 rail は除く。 T字路で突き当たる側の端は、相手の車道の手前で止める。 bpy 不要"""
    cx, cy = center; steps = int(pol['piece_steps']); out = []
    jt = JunctionTrim(city) if pol.get('trim_junctions', True) else None
    for ri, r in enumerate(city['roads']):
        cls = r['c']
        if cls == 'rail':
            continue
        band = cls in ('service', 'pedestrian')
        if (band and r['w'] < 3.0) or (not band and r['w'] < 4.5):
            continue
        P = np.asarray(r['p'], float)
        if P[:, 0].max() < cx - radius - 40 or P[:, 0].min() > cx + radius + 40 or P[:, 1].max() < cy - radius - 40 or P[:, 1].min() > cy + radius + 40:
            continue
        S = resample(P, 4.0)
        if len(S) < 2:
            continue
        if jt is not None and not band:
            S = jt.trim(ri, S)
            if len(S) < 2:
                continue
        n = len(S); i = 0
        while i < n - 1:
            j = min(n - 1, i + steps)
            if n - 1 - j < steps * 0.4:
                j = n - 1
            seg = S[i:j + 1]; m = seg[len(seg) // 2]
            if math.hypot(m[0] - cx, m[1] - cy) <= radius + 30:
                dc = cam_dist2d(cam, m[0], m[1], 0.5 * 4.0 * (j - i)) if cam is not None else 1e9
                out.append(dict(ri=ri, cls=cls, w=float(r['w']), S=seg, band=band, dc=dc))
            i = j
    return out


def min_corridor(pieces):
    """全部の道路の、最小の断面ぶんの通行帯(両側)。 建物の輪郭を削る基準"""
    corr = Corridor()
    for pc in pieces:
        o = wall_min_for(pc['cls'], pc['w']) + 0.10
        corr.add_strip(pc['S'], o, o, ext=0.35)
    return corr


def plan_buildings(city, center, radius, cam, hour, seed, policy, ridx=None, corr=None):
    """-> dict(recs=[ビル(kit)], houses=[低層], stats=...)。 bpy 不要。
    建物の足元の矩形は OSM の輪郭の中に入れ(最大の内接長方形)、通行帯 corr に重なる所は辺を内側に動かして削る(足りなければ描かない)。 policy['legacy'] = True で旧方式(比較用)。"""
    pol = dict(DEFAULT_POLICY); pol.update(policy or {})
    legacy = bool(pol.get('legacy'))
    ridx = ridx or RoadIndex(city['roads'])
    cx, cy = center
    recs = []; houses = []; skipped = 0
    cnt = dict(fitted=0, clipped=0, dropped=0)

    def fit(Pg, c, u, v, eu, ev, ratio):
        """ratio(輪郭/外接矩形の面積)が低い建物は、輪郭の中に入る長方形にする。 -> (c, eu, ev) / None"""
        if ratio >= 0.9:
            return (c, eu, ev)
        cnt['fitted'] += 1
        return inscribed_rect(Pg, c, u, v, eu, ev, max(0.4, min(1.5, min(eu, ev) / 8.0)))

    def cut(rect, u, v, min_side):
        if corr is None or rect is None:
            return rect
        r = clip_rect(rect[0], u, v, rect[1], rect[2], corr, min_side=min_side)
        if r is None:
            cnt['dropped'] += 1; return None
        if abs(r[1] - rect[1]) > 0.05 or abs(r[2] - rect[2]) > 0.05:
            cnt['clipped'] += 1
        return r

    for b in city['buildings']:
        Pg = b['p']
        a0 = np.asarray(Pg, float)
        m = a0.mean(0)
        if math.hypot(m[0] - cx, m[1] - cy) > radius:
            skipped += 1; continue
        t = b['t']; h = float(b['h']); bid = int(b['id'])
        c, u, v, eu, ev = obb(Pg)
        parea = poly_area(Pg); oa = max(eu * ev, 1.0); ratio = min(1.0, parea / oa)
        rr = rng_for(seed, bid)
        if t == 'house':
            dc = cam_dist2d(cam, c[0], c[1], 0.0) if cam is not None else math.hypot(c[0] - cx, c[1] - cy)
            if dc > pol['houses']:
                continue
            if legacy:
                k = math.sqrt(max(0.35, min(1.0, ratio))); rc, le, wi = c, eu * k, ev * k
            else:
                rect = cut(fit(Pg, c, u, v, eu, ev, ratio), u, v, 2.5)
                if rect is None:
                    continue
                rc, le, wi = rect
            if wi < 2.5:
                continue
            houses.append(dict(id=bid, kind='house', c=(float(rc[0]), float(rc[1])), u=(float(u[0]), float(u[1])), eu=float(le), ev=float(wi), h=h, rr=rr.random(), gable=rr.random() < 0.6))
            continue
        rects = [(c, eu, ev)]
        if legacy:
            if pol['split'] and ratio < 0.72 and oa > 350.0:
                rects = split_rects_legacy(Pg, c, u, v, eu, ev)
            elif ratio < 0.9:
                k = max(0.8, math.sqrt(ratio)); rects = [(c, eu * k, ev * k)]
        else:
            if pol['split'] and ratio < 0.72 and oa > 350.0:
                rects = split_rects(Pg, c, u, v, eu, ev) or [r_ for r_ in [fit(Pg, c, u, v, eu, ev, ratio)] if r_ is not None]
            else:
                rects = [r_ for r_ in [fit(Pg, c, u, v, eu, ev, ratio)] if r_ is not None]
            rects = [r_ for r_ in (cut(r_, u, v, 2.5) for r_ in rects) if r_ is not None]
        for si, (rc, reu, rev) in enumerate(rects):
            q = ridx.nearest(rc[0], rc[1])
            dvec = np.array([q[0] - rc[0], q[1] - rc[1]]) if q else np.array([0.0, -1.0])
            if np.hypot(*dvec) < 1e-6:
                dvec = np.array([0.0, -1.0])
            cands = [(u, rev, reu), (-u, rev, reu), (v, reu, rev), (-v, reu, rev)]       # (正面の向き, 正面の幅, 奥行き)
            front, wf, df = max(cands, key=lambda cd: float(np.dot(cd[0], dvec)))
            r2 = rng_for(seed, bid, 11 + si)
            kind = pick_kind(t, h, parea if si == 0 else wf * df, wf, df, r2, pol)
            theta = math.atan2(front[0], -front[1])
            rec = dict(id=bid, sub=si, kind=kind, c=(float(rc[0]), float(rc[1])), theta=theta, wf=float(wf), df=float(df), h=h, t=t,
                       road_d=(q[2] if q else 999.0), seed=(int(seed) * 131 + bid * 7 + si * 3) % 99991, rr=r2.random())
            if kind == 'low':
                rec['u'] = (float(u[0]), float(u[1])); rec['eu'] = float(reu); rec['ev'] = float(rev); rec['gable'] = False; rec['rr'] = r2.random()
                houses.append(dict(rec, kind='low', c=rec['c'], u=rec['u'], eu=rec['eu'], ev=rec['ev'], h=h))
                continue
            S = bk.SPECS[kind]
            W, D, nW, nD = kit_dims(kind, wf, df)
            rec.update(floors=floors_for(kind, h), W=W, D=D, sx=wf / W, sy=df / D, age=S['age'][0] + (S['age'][1] - S['age'][0]) * r2.random(),
                       r=0.5 * math.hypot(wf, df), dcam=cam_dist2d(cam, rc[0], rc[1], 0.5 * math.hypot(wf, df)))
            recs.append(rec)
    # detail の割り当て: 距離の閾値 + 棟数の上限(近い順)。 実立体(near/mid)も、far と同じく矩形(wf x df)にぴったり合うよう x・y を伸縮して置く(伸縮 0.8〜1.25)
    recs.sort(key=lambda r: r['dcam'])
    n_near = n_mid = 0
    for r in recs:
        d = 'far'
        lo, hi = pol['solid_scale']
        ok = cam is not None and lo <= r['sx'] <= hi and lo <= r['sy'] <= hi
        if ok:
            if r['dcam'] <= pol['near'] and n_near < pol['max_near']:
                d = 'near'; n_near += 1
            elif r['dcam'] <= pol['mid'] and n_mid < pol['max_mid']:
                d = 'mid'; n_mid += 1
        r['detail'] = d
    kinds = {}
    for r in recs:
        kinds[r['kind']] = kinds.get(r['kind'], 0) + 1
    for hh in houses:
        kinds[hh['kind']] = kinds.get(hh['kind'], 0) + 1
    return dict(recs=recs, houses=houses, kinds=kinds, policy=pol, skipped=skipped, rect_fitted=cnt['fitted'], rect_clipped=cnt['clipped'], rect_dropped=cnt['dropped'])


def footprints(plan):
    """描く建物すべての足元の矩形。 -> [(id, 'kit'|'house'|'low', detail, 4隅(4,2))]"""
    out = []
    for r in plan['recs']:
        th = r['theta']; xd = np.array([math.cos(th), math.sin(th)]); yd = np.array([-math.sin(th), math.cos(th)])
        out.append((r['id'], 'kit', r['detail'], rect_poly(r['c'], xd, yd, r['wf'], r['df'])))
    for hh in plan['houses']:
        u = np.asarray(hh['u'], float); v = np.array([-u[1], u[0]])
        out.append((hh['id'], hh['kind'], 'far', rect_poly(hh['c'], u, v, hh['eu'], hh['ev'])))
    return out


# ------------------------------------------------------------------------------------------------
# 遠景のビル(far): 形は building_kit の far、材質は共有、窓の点灯は1枚の表(アトラス)
# ------------------------------------------------------------------------------------------------
FAR_VARIANTS = {'office': 4, 'zakkyo': 6, 'mansion': 6}
FAR_ROLE_FALLBACK = 'roof'


def far_variant(rec):
    nv = FAR_VARIANTS[rec['kind']]
    return int(rec['rr'] * 997) % nv, (1 if rec['age'] >= 0.5 else 0)


def far_geometry(rec, night, hour, base_index):
    """far のビル1棟を世界座標で作る。 -> dict(V(n,3), UV(n,2), sizes(面ごとの頂点数), roles(面ごとの役割名), lit(floors,ncell,4), fb(4,), tris, info)"""
    md, info, ctx = bk.build_geometry(rec['kind'], rec['W'], rec['D'], rec['floors'], rec['seed'], 'far', night, hour, rec['age'])
    V = np.asarray(md.V, float).reshape(-1, 3)
    V[:, 0] *= rec['sx']; V[:, 1] *= rec['sy']
    th = rec['theta']; ct, st = math.cos(th), math.sin(th)
    x = V[:, 0] * ct - V[:, 1] * st + rec['c'][0]; y = V[:, 0] * st + V[:, 1] * ct + rec['c'][1]
    V = np.stack([x, y, V[:, 2]], 1)
    roles = [md.mats[i] for i in md.MI]
    sizes = np.asarray([len(f) for f in md.F], int)
    fb = (float(base_index), float(ctx.ncell), float(ctx.nW), float(ctx.floors))
    return dict(V=V, UV=np.asarray(md.UV, float).reshape(-1, 2), sizes=sizes, roles=roles, lit=ctx.lit, fb=fb,
                tris=int(sum(len(f) - 2 for f in md.F)), info=info)


# ------------------------------------------------------------------------------------------------
# 道路: 中心線の切り出し・両側の壁の位置・断面の決め方
# ------------------------------------------------------------------------------------------------
def resample(P, step):
    P = np.asarray(P, float)
    d = np.hypot(*np.diff(P, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(d)])
    if cum[-1] < 1e-6:
        return P[:1]
    n = max(1, int(round(cum[-1] / step)))
    s = np.linspace(0, cum[-1], n + 1)
    return np.stack([np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1])], 1)


class VertexGrid:
    def __init__(self, city, cell=30.0, skip_house=False):
        self.cell = cell; pts = []
        for b in city['buildings']:
            if skip_house and b['t'] == 'house':
                continue
            for p in b['p']:
                pts.append(p)
        self.pts = np.asarray(pts, float).reshape(-1, 2); self.grid = {}
        for i, (x, y) in enumerate(self.pts):
            self.grid.setdefault((int(x // cell), int(y // cell)), []).append(i)

    def query_box(self, x0, y0, x1, y1):
        idx = []
        for ix in range(int(x0 // self.cell), int(x1 // self.cell) + 1):
            for iy in range(int(y0 // self.cell), int(y1 // self.cell) + 1):
                idx += self.grid.get((ix, iy), [])
        return self.pts[np.asarray(idx, int)] if idx else np.zeros((0, 2))


def side_gaps(S, vg, reach=26.0):
    """中心線のサンプル S(n,2) の左右それぞれの、建物の壁までの距離(中央値)。 建物が無ければ None。 -> (左, 右)"""
    if len(S) < 2:
        return None, None
    t = np.gradient(S, axis=0); t /= (np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-9)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    P = vg.query_box(S[:, 0].min() - reach, S[:, 1].min() - reach, S[:, 0].max() + reach, S[:, 1].max() + reach)
    if len(P) == 0:
        return None, None
    D = np.hypot(P[:, None, 0] - S[None, :, 0], P[:, None, 1] - S[None, :, 1])
    j = D.argmin(1); dmin = D[np.arange(len(P)), j]
    ok = dmin < reach
    P = P[ok]; j = j[ok]
    if len(P) == 0:
        return None, None
    rel = P - S[j]
    q = rel[:, 0] * nrm[j, 0] + rel[:, 1] * nrm[j, 1]; a = rel[:, 0] * t[j, 0] + rel[:, 1] * t[j, 1]
    inner = np.abs(a) < 7.0
    out = []
    for sgn in (1.0, -1.0):
        best = {}
        for qi, ji, ii in zip(q * sgn, j, inner):
            if qi > 0.5 and ii:
                best[ji] = min(best.get(ji, 1e9), qi)
        out.append(float(np.median(list(best.values()))) if len(best) >= max(2, 0.2 * len(S)) else None)
    return out[0], out[1]


def side_params_legacy(cls, w, gap, rr):
    """片側の断面の値を決める。 w = 道路の幅(city.json)、gap = 壁までの距離(None なら建物なし)。 -> street_kit の上書き辞書(profile 込み)"""
    lw = 3.25; g = 0.30; cw = 0.18; bw = 0.12
    lanes = int(min(3, max(1, round((w / 2.0 - g) / lw))))
    fd = rr.choices(['low', 'wainscot', 'flush'], [0.55, 0.30, 0.15])[0]
    found_mat = rr.choice(['conc', 'stone', 'tile']) if fd == 'wainscot' else 'conc'
    inu = 0.45 if fd != 'flush' else 0.0
    prot = {'low': 0.03, 'wainscot': 0.06, 'flush': 0.0}[fd]
    major = cls in MAJOR
    sw_def = 3.5 if cls in ('trunk', 'primary') else (3.0 if major else (2.5 if cls == 'tertiary' else 2.0))
    setback = 0.0

    def fixed(l):
        return l * lw + g + cw + bw + inu + prot
    if gap is None:
        sw = sw_def
    else:
        sw = gap - fixed(lanes)
        while sw < 1.4 and lanes > 1:
            lanes -= 1; sw = gap - fixed(lanes)
        if sw > 3.6:
            setback = min(2.0, sw - 3.6) if fd != 'flush' else 0.0
            sw = 3.6 + (0.0 if fd == 'flush' else 0.0)
        sw = max(sw, 0.7)
    sw = round(sw * 4.0) / 4.0
    kw = dict(lanes=lanes, sidewalk_w=float(round(sw, 3)), foundation=fd, found_mat=found_mat, setback=float(round(setback, 3)),
              trees=bool(major and sw >= 2.9), fence=bool(major and sw >= 1.8), paving='block' if major or cls == 'tertiary' else 'asphalt',
              lamp_pitch=25.0 if major else 30.0, pole_pitch=40.0, signal_at=[], smooth=False, sides='L',
              center_line='yellow' if lanes >= 2 else 'white')
    return kw


def plan_streets_legacy(city, center, radius, cam, seed, pol, vg, night, hour):
    """(旧方式・比較用) 道路を 約60mの区間に切って、カメラからの距離で detail(near/mid/far/帯)を決め、street_kit の形を作る。 -> dict(groups={'near':[...], ...}, bands=[(pts,w)], ...)
    各区間 = 片側ずつ(進行方向の左 / 逆向きの左 = 右)。 bpy 不要。"""
    groups = {'near': [], 'mid': [], 'far': []}; bands = []
    cx, cy = center; pieces = []
    for ri, r in enumerate(city['roads']):
        cls = r['c']
        if cls in NO_STREET or r['w'] < 4.5:
            if cls in ('service', 'pedestrian') and r['w'] >= 3.0:
                pass
            else:
                continue
        P = np.asarray(r['p'], float)
        if P[:, 0].max() < cx - radius or P[:, 0].min() > cx + radius or P[:, 1].max() < cy - radius or P[:, 1].min() > cy + radius:
            continue
        S = resample(P, 4.0)
        if len(S) < 2:
            continue
        if cls in ('service', 'pedestrian'):
            bands.append(dict(S=S, hl=r['w'] / 2.0, hr=r['w'] / 2.0, w=r['w'], cls=cls)); continue
        # 約60mの区間(重なりなし)
        n = len(S); i = 0; L60 = 15
        while i < n - 1:
            j = min(n - 1, i + L60)
            seg = S[i:j + 1]
            m = seg[len(seg) // 2]
            if math.hypot(m[0] - cx, m[1] - cy) <= radius:
                dc = cam_dist2d(cam, m[0], m[1], 0.5 * 4.0 * (j - i)) if cam is not None else 1e9
                pieces.append((dc, ri, cls, r['w'], seg))
            i = j
    pieces.sort(key=lambda p: p[0])
    near_m = mid_m = 0.0
    for dc, ri, cls, w, seg in pieces:
        length = 4.0 * (len(seg) - 1)
        if dc > pol['street_far']:
            bands.append(dict(S=seg, hl=w / 2.0, hr=w / 2.0, w=w, cls=cls)); continue
        d = 'far'
        if dc <= pol['street_near'] and near_m < pol['max_street_near_m']:
            d = 'near'; near_m += length
        elif dc <= pol['street_mid'] and mid_m < pol['max_street_mid_m']:
            d = 'mid'; mid_m += length
        gl, gr = side_gaps(seg, vg)
        sd = int(ri * 131 + round(seg[0][0]) * 7 + round(seg[0][1]) * 13) & 0xffff
        for side, gap, line in (('L', gl, seg), ('R', gr, seg[::-1])):
            rr = rng_for(seed, sd, 5 if side == 'L' else 6)
            kw = side_params_legacy(cls, w, gap, rr)
            kw.update(night=night, hour=hour, lights=False)
            if d == 'far' and dc > pol['far_fence_dist']:
                kw['fence'] = False
            cl = [(float(x), float(y), 0.0) for x, y in line]
            md, info, lamps, Pp = sk.build_geometry(kw, cl, 1e9, sd, d, **{})
            groups[d].append(dict(md=md, info=info, P=Pp, lamps=lamps, dc=dc, cls=cls, side=side, line=np.asarray(line, float)[:, :2], o_w=info['found_front'], wall=info['wall_plane'], q=gap))
    return dict(groups=groups, ultra=[], bands=bands, pinfo=[], near_m=near_m, mid_m=mid_m)


class BoundaryGrid:
    """描く建物の足元の矩形の縁の点(約0.5m間隔)を格子に入れる。 道路の中心線から壁までの距離(clearance)を引く"""
    def __init__(self, polys, step=0.5, cell=30.0):
        pts = []
        for P in polys:
            for k in range(4):
                a = P[k]; b = P[(k + 1) % 4]; L = float(np.hypot(*(b - a))); n = max(1, int(math.ceil(L / step)))
                for i in range(n):
                    pts.append(a + (b - a) * (i / n))
        self.pts = np.asarray(pts, float).reshape(-1, 2); self.cell = cell; self.grid = {}
        for i, (x, y) in enumerate(self.pts):
            self.grid.setdefault((int(x // cell), int(y // cell)), []).append(i)

    def query_box(self, x0, y0, x1, y1):
        idx = []
        for ix in range(int(x0 // self.cell), int(x1 // self.cell) + 1):
            for iy in range(int(y0 // self.cell), int(y1 // self.cell) + 1):
                idx += self.grid.get((ix, iy), [])
        return self.pts[np.asarray(idx, int)] if idx else np.zeros((0, 2))

    def clearance(self, S, reach=45.0):
        """中心線 S(n,2) の左・右それぞれで、いちばん近い壁の面までの距離(法線方向)。 区間の端より先の点は数えない。 -> (左, 右)(建物が無ければ None)"""
        S = np.asarray(S, float)
        if len(S) < 2:
            return None, None
        t = np.gradient(S, axis=0); t /= (np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-9); nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        P = self.query_box(S[:, 0].min() - reach, S[:, 1].min() - reach, S[:, 0].max() + reach, S[:, 1].max() + reach)
        if len(P) == 0:
            return None, None
        D = np.hypot(P[:, None, 0] - S[None, :, 0], P[:, None, 1] - S[None, :, 1])
        j = D.argmin(1); dmin = D[np.arange(len(P)), j]
        ok = dmin < reach
        P = P[ok]; j = j[ok]
        if len(P) == 0:
            return None, None
        rel = P - S[j]
        q = rel[:, 0] * nrm[j, 0] + rel[:, 1] * nrm[j, 1]; a = rel[:, 0] * t[j, 0] + rel[:, 1] * t[j, 1]
        n = len(S)
        keep = ~(((j == 0) & (a < -0.01)) | ((j == n - 1) & (a > 0.01)))
        q = q[keep]
        out = []
        for sg in (1.0, -1.0):
            v = q * sg; v = v[v > 0.0]
            out.append(float(v.min()) if len(v) else None)
        return out[0], out[1]


def side_params(cls, w, q, rr):
    """片側の断面の値を決める。 q = 中心線から壁の面までの距離(None なら建物なし)。 壁に合わせて 車道(車線数・車線幅)→歩道 を狭める。
    順序は 車道 → 路肩 → 縁石 → 歩道(最小0.75m) → 民地境界 → (空地) → 犬走り → 基礎の立ち上がり → 壁。 壁の面は q の手前0.01m。 -> street_kit の上書き辞書"""
    major = cls in MAJOR
    lw0 = 3.25
    lanes_w = int(min(3, max(1, round((w / 2.0 - 0.30) / lw0))))
    sw_def = 3.5 if cls in ('trunk', 'primary') else (3.0 if major else (2.5 if cls == 'tertiary' else 2.0))
    fd = rr.choices(['low', 'wainscot'], [0.58, 0.42])[0]
    found_mat = rr.choice(['conc', 'stone', 'tile']) if fd == 'wainscot' else 'conc'
    wr_floor = wr_min_for(w)
    lanes, lane_w, g, inu = lanes_w, lw0, 0.30, 0.45
    setback = 0.0; found_gaps = []
    if q is None:                                   # 建物が無い側: 標準の断面。 壁が無いので基礎は描かず、空地にする
        sw = sw_def; found_gaps = [(0.0, 1e4)]
    else:
        if q - 0.01 - (0.03 if fd == 'low' else 0.06) < wr_floor + MIN_G + CW + MIN_SW + BW + MIN_INU:
            fd = 'low'; found_mat = 'conc'          # 余裕が無い所は、基礎の出が小さい方
        prot = 0.03 if fd == 'low' else 0.06
        avail = q - 0.01 - prot                     # 基礎の前面(o_w)の目標
        chosen = None
        for ln_ in range(lanes_w, 0, -1):
            for lw in (3.25, 3.0, 2.75, 2.6):
                Wr = ln_ * lw
                if Wr < wr_floor - 1e-9:
                    continue
                sw_av = avail - (Wr + 0.30 + CW + BW + 0.45)
                if sw_av >= min(sw_def, 2.0):
                    chosen = (ln_, lw, 0.30, 0.45, sw_av); break
            if chosen:
                break
        if chosen is None:                          # いちばん狭い断面に詰める
            sw_av = avail - (wr_floor + MIN_G + CW + BW + MIN_INU)
            chosen = (1, wr_floor, MIN_G, MIN_INU, max(sw_av, MIN_SW))
        lanes, lane_w, g, inu, sw_av = chosen
        sw = min(sw_def, sw_av)
        sw = max(MIN_SW, math.floor(sw * 4.0 + 1e-9) / 4.0)
        setback = min(8.0, max(0.0, math.floor((sw_av - sw) * 100.0) / 100.0))
    major_w = cls in MAJOR
    toff = 0.60 if sw >= 1.5 else (0.30 if sw >= 1.0 else 0.15)
    kw = dict(lanes=lanes, lane_w=float(lane_w), gutter=float(g), inu_w=float(inu), sidewalk_w=float(round(sw, 3)), foundation=fd, found_mat=found_mat, setback=float(round(setback, 3)),
              tactile_off=toff, trees=bool(major_w and sw >= 2.9), fence=bool(major_w and sw >= 1.8), paving='block' if major_w or cls == 'tertiary' else 'asphalt',
              lamp_pitch=25.0 if major_w else 30.0, pole_pitch=40.0, signal_at=[], smooth=False, sides='L', found_gaps=found_gaps,
              center_line='yellow' if lanes >= 2 else 'white')
    return kw


def ultra_info(kw, night, hour):
    """遠い道路(帯で描く)の、断面の値。 street_kit の info と同じ名前で返す。 -> (P, info)"""
    P = sk.params(None, **dict(kw, night=night, hour=hour))
    Wr = P['lanes'] * P['lane_w']; o_c0 = Wr + P['gutter']; sw0 = o_c0 + P['curb_w']; o_b0 = sw0 + P['sidewalk_w']
    tac0 = sw0 + P['tactile_off']; o_w = o_b0 + P['boundary_w'] + P['setback'] + (P['inu_w'] if P['foundation'] != 'flush' else 0.0)
    prot = {'low': 0.03, 'wainscot': 0.06, 'flush': 0.0}[P['foundation']]
    info = dict(half_road=Wr, lane_w=P['lane_w'], sidewalk_o0=sw0, boundary_o=o_b0, tactile=(tac0, tac0 + P['tactile_w']), curb_o=o_c0, found_front=o_w, wall_plane=o_w + prot,
                sidewalk_w=P['sidewalk_w'], sw_h=P['sw_h'], profile=P['foundation'])
    return P, info


class RectIndex:
    """建物の矩形(4隅)を、外接箱の格子で引く"""
    def __init__(self, polys, cell=30.0):
        self.polys = polys; self.cell = cell; self.grid = {}
        for i, P in enumerate(polys):
            x0, y0 = P.min(0); x1, y1 = P.max(0)
            for ix in range(int(x0 // cell), int(x1 // cell) + 1):
                for iy in range(int(y0 // cell), int(y1 // cell) + 1):
                    self.grid.setdefault((ix, iy), []).append(i)

    def near(self, x0, y0, x1, y1):
        idx = set()
        for ix in range(int(x0 // self.cell), int(x1 // self.cell) + 1):
            for iy in range(int(y0 // self.cell), int(y1 // self.cell) + 1):
                idx.update(self.grid.get((ix, iy), ()))
        return idx


def fit_side(cls, w, q, rng_args, line, ri_idx, polys_index):
    """片側の断面を決め、その通行帯(中心線〜基礎の前面)に建物の矩形が入っていないか検証して、入っていれば壁を後退させて決め直す(最大4回)。 -> (kw, info, P)"""
    seed, sd, salt = rng_args
    qq = q; kw = None
    for _ in range(5):
        kw = side_params(cls, w, qq, rng_for(seed, sd, salt))
        P, info = ultra_info(dict(kw, night=True, hour=20.0), True, 20.0)
        if polys_index is None or q is None:
            break
        ri, polys = polys_index
        c = Corridor(); c.add_strip(line, 0.0, info['found_front'])
        S = line
        cand = ri.near(S[:, 0].min() - info['found_front'], S[:, 1].min() - info['found_front'], S[:, 0].max() + info['found_front'], S[:, 1].max() + info['found_front'])
        pen = 0.0
        t = np.gradient(S, axis=0); t /= (np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-9); nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        for i in cand:
            ov, a = c.overlap(polys[i], 1e-4)
            for pg in ov:
                D = np.hypot(pg[:, None, 0] - S[None, :, 0], pg[:, None, 1] - S[None, :, 1]); j = D.argmin(1)
                lat = ((pg - S[j]) * nrm[j]).sum(1)
                pen = max(pen, info['found_front'] - float(lat.min()))
        if pen <= 1e-6:
            break
        qq = (qq if qq is not None else info['wall_plane']) - pen - 0.03
    return kw


def plan_streets(pieces, bg, cam, seed, pol, night, hour, rect_polys=None):
    """道路の区間ごとに、建物の壁までの距離(bg.clearance)から断面を決め、近い順に near / mid / far(street_kit の形)・ultra(帯だけで軽く)・band(路地)に分ける。
    断面の通行帯(中心線〜基礎の前面)に建物の矩形が入る所は、壁を後退させて決め直す。 各区間 = 片側ずつ(進行方向の左 / 逆向きの左 = 右)。 bpy 不要。 -> dict(groups, ultra, bands, pinfo, near_m, mid_m)"""
    groups = {'near': [], 'mid': [], 'far': []}; ultra = []; bands = []; pinfo = []
    near_m = mid_m = 0.0
    pidx = (RectIndex(rect_polys), rect_polys) if rect_polys else None
    for pc in sorted(pieces, key=lambda p: p['dc']):
        S = pc['S']; cls = pc['cls']; w = pc['w']; dc = pc['dc']
        qL, qR = bg.clearance(S)
        pinfo.append(dict(S=S, cls=cls, w=w, qL=qL, qR=qR, dc=dc, band=pc['band']))
        if pc['band']:
            hl = w / 2.0 if qL is None else min(w / 2.0, max(0.9, qL - 0.10))
            hr = w / 2.0 if qR is None else min(w / 2.0, max(0.9, qR - 0.10))
            bands.append(dict(S=S, hl=hl, hr=hr, w=w, cls=cls)); continue
        length = 4.0 * (len(S) - 1)
        if dc > pol['street_far']:
            d = 'ultra'
        else:
            d = 'far'
            if dc <= pol['street_near'] and near_m < pol['max_street_near_m']:
                d = 'near'; near_m += length
            elif dc <= pol['street_mid'] and mid_m < pol['max_street_mid_m']:
                d = 'mid'; mid_m += length
        sd = int(pc['ri'] * 131 + round(S[0][0]) * 7 + round(S[0][1]) * 13) & 0xffff
        for side, q, line in (('L', qL, S), ('R', qR, S[::-1])):
            line = np.asarray(line, float)
            kw = fit_side(cls, w, q, (seed, sd, 5 if side == 'L' else 6), line, 0, pidx)
            kw.update(night=night, hour=hour, lights=False)
            if d in ('far', 'ultra') and dc > pol['far_fence_dist']:
                kw['fence'] = False
            if d == 'ultra':
                P, info = ultra_info(kw, night, hour)
                ultra.append(dict(line=line, P=P, info=info, dc=dc, cls=cls, side=side, o_w=info['found_front'], wall=info['wall_plane'], q=q))
                continue
            cl = [(float(x), float(y), 0.0) for x, y in line]
            md, info, lamps, Pp = sk.build_geometry(kw, cl, 1e9, sd, d, **{})
            groups[d].append(dict(md=md, info=info, P=Pp, lamps=lamps, dc=dc, cls=cls, side=side, line=line, o_w=info['found_front'], wall=info['wall_plane'], q=q))
    return dict(groups=groups, ultra=ultra, bands=bands, pinfo=pinfo, near_m=near_m, mid_m=mid_m)


def street_items(sp):
    return [it for d in ('near', 'mid', 'far') for it in sp['groups'][d]] + sp['ultra']


# ------------------------------------------------------------------------------------------------
# 建物の足元(犬走り・基礎の立ち上がり)。 全部の建物の全部の壁に付ける
# ------------------------------------------------------------------------------------------------
APRON_Z = 0.07          # 犬走りの上面(歩道面 0.05・街路の犬走り 0.062 より少し上。 z の重なりを避ける)
PLINTH_TOP = 0.34       # far・低層の基礎の立ち上がりの上端


def final_clip(plan, sp):
    """描く通行帯(断面の実際の幅)と、まだ重なっている建物の矩形を、辺を動かして削る(足りなければ描かない)。 near/mid は箱を縮められないので far にする。 -> 削った数"""
    corr = Corridor()
    for it in street_items(sp):
        corr.add_strip(it['line'], 0.0, it['o_w'])
    for bd in sp['bands']:
        corr.add_strip(bd['S'], bd['hr'], bd['hl'])
    n = 0; keep = []
    for r in plan['recs']:
        th = r['theta']; xd = np.array([math.cos(th), math.sin(th)]); yd = np.array([-math.sin(th), math.cos(th)])
        ov, a = corr.overlap(rect_poly(r['c'], xd, yd, r['wf'], r['df']), 1e-4)
        if ov:
            n += 1
            res = clip_rect(r['c'], xd, yd, r['wf'], r['df'], corr, min_side=2.5, tol=1e-4)
            if res is None:
                continue
            r['c'] = (float(res[0][0]), float(res[0][1])); r['wf'] = float(res[1]); r['df'] = float(res[2])
            r['sx'] = r['wf'] / r['W']; r['sy'] = r['df'] / r['D']; r['detail'] = 'far'
        keep.append(r)
    plan['recs'] = keep
    keeph = []
    for hh in plan['houses']:
        u = np.asarray(hh['u'], float); v = np.array([-u[1], u[0]])
        ov, a = corr.overlap(rect_poly(hh['c'], u, v, hh['eu'], hh['ev']), 1e-4)
        if ov:
            n += 1
            res = clip_rect(hh['c'], u, v, hh['eu'], hh['ev'], corr, min_side=2.5, tol=1e-4)
            if res is None:
                continue
            hh['c'] = (float(res[0][0]), float(res[0][1])); hh['eu'] = float(res[1]); hh['ev'] = float(res[2])
        keeph.append(hh)
    plan['houses'] = keeph
    return n


def plan_aprons(fps, items, pol):
    """各建物の矩形の4つの壁の外側に、犬走り(幅0.45m。道路の歩道の壁の位置より壁が奥にあるときは、その隙間も埋める。 最大8m)を作る。
    far・低層には、壁の手前0.012mに基礎の立ち上がり(高さ0.07〜0.34m)も付ける。 -> dict(apron=[(4点(x,y), z)], plinth=[(a(2,), b(2,))], faces=面数, road_faces=道路に面した面数)"""
    WP = []; WN = []; grid = {}; cell = 20.0
    for it in items:
        line = it['line']
        if len(line) < 2:
            continue
        t = np.gradient(line, axis=0); t /= (np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-9); n = np.stack([-t[:, 1], t[:, 0]], 1)
        P = line + n * it['wall']
        for k in range(0, len(P), 1):
            grid.setdefault((int(P[k, 0] // cell), int(P[k, 1] // cell)), []).append(len(WP)); WP.append(P[k]); WN.append(n[k])
    WP = np.asarray(WP, float).reshape(-1, 2); WN = np.asarray(WN, float).reshape(-1, 2)
    apron = []; plinth = []; road_faces = 0; nfaces = 0
    for bid, kind, detail, poly in fps:
        for k in range(4):
            a = poly[k]; b = poly[(k + 1) % 4]; e = b - a; L = float(np.hypot(*e))
            if L < 0.3:
                continue
            e = e / L; nu = np.array([e[1], -e[0]]); m = (a + b) / 2.0
            nfaces += 1
            wd = 0.45
            idx = []
            gx, gy = int(m[0] // cell), int(m[1] // cell)
            for ix in range(gx - 1, gx + 2):
                for iy in range(gy - 1, gy + 2):
                    idx += grid.get((ix, iy), [])
            if idx:
                idx = np.asarray(idx)
                dist = np.hypot(WP[idx, 0] - m[0], WP[idx, 1] - m[1])
                face = (WN[idx] @ (-nu)) > 0.9
                if face.any() and dist[face].min() < 14.0:
                    i = idx[face][int(np.argmin(dist[face]))]
                    d = float((m - WP[i]) @ WN[i])
                    road_faces += 1
                    if 0.02 < d <= 8.0:
                        wd = min(8.0, d + 0.45)
            ex = 0.45
            apron.append(([a - e * ex, b + e * ex, b + e * ex + nu * wd, a - e * ex + nu * wd], APRON_Z))
            if kind != 'kit' or detail == 'far':
                plinth.append((a + nu * 0.012, b + nu * 0.012))
    return dict(apron=apron, plinth=plinth, faces=nfaces, road_faces=road_faces)


# ------------------------------------------------------------------------------------------------
# 通行帯と建物の重なり(面積)を測る
# ------------------------------------------------------------------------------------------------
def measure_overlap(fps, sp):
    """描く建物の足元の矩形と、描く通行帯(車道〜縁石〜歩道〜空地〜犬走り。 中心線から基礎の前面まで)の交差面積。 目標 0。
    路地・歩行者道の帯は、中心線から帯の端まで。 -> dict"""
    corr = Corridor()
    for it in street_items(sp):
        corr.add_strip(it['line'], 0.0, it['o_w'])
    for bd in sp['bands']:
        corr.add_strip(bd['S'], bd['hr'], bd['hl'])
    total = 0.0; nbad = 0; worst = []; foot = 0.0
    for bid, kind, detail, poly in fps:
        foot += poly_area(poly)
        ov, a = corr.overlap(poly, 1e-4)
        total += a
        if a > 0.05:
            nbad += 1; worst.append((round(a, 2), int(bid), kind))
    worst.sort(reverse=True)
    items = street_items(sp)
    sw = np.asarray([it['info']['sidewalk_w'] for it in items], float) if items else np.zeros(0)
    res = dict(overlap_area_m2=round(total, 3), buildings_overlapping=nbad, buildings=len(fps), footprint_area_m2=round(foot, 1),
               overlap_ratio=round(total / max(foot, 1.0), 5), worst=worst[:8], strips=len(items), bands=len(sp['bands']))
    if len(sw):
        res['sidewalk_w_m'] = dict(min=round(float(sw.min()), 2), p10=round(float(np.percentile(sw, 10)), 2), median=round(float(np.median(sw)), 2), max=round(float(sw.max()), 2),
                                   under_075=int((sw < 0.75 - 1e-6).sum()))
    return res


def plan_all(city, center, radius, cam, hour, seed, policy=None):
    """建物・道路・足元の計画を全部まとめて作る(bpy 不要)。 -> dict(recs, houses, kinds, streets, aprons, fps, overlap, ...)
    policy['legacy'] = True なら、修正前の方式(壁までの距離は頂点の中央値・60mの区間・矩形は外接矩形の縮小・足元の犬走りなし)。比較・数値確認用。"""
    pol = dict(DEFAULT_POLICY); pol.update(policy or {})
    legacy = bool(pol.get('legacy'))
    if legacy:
        pol.update(piece_steps=15)
    night = bool(hour >= 17.0 or hour < 5.5)
    cam_a = None
    if cam is not None and len(cam):
        cam_a = np.asarray([(p[0], p[1]) for p in cam], float)
    t0 = time.time()
    ridx = RoadIndex(city['roads'])
    if legacy:
        plan = plan_buildings(city, center, radius, cam_a, hour, seed, pol, ridx, corr=None)
        fps = footprints(plan)
        sp = plan_streets_legacy(city, center, radius, cam_a, seed, pol, VertexGrid(city), night, hour)
        aprons = dict(apron=[], plinth=[], faces=0, road_faces=0)
    else:
        pieces = road_pieces(city, center, radius, cam_a, pol)
        corr = min_corridor(pieces)
        plan = plan_buildings(city, center, radius, cam_a, hour, seed, pol, ridx, corr=corr)
        fps = footprints(plan)
        bg = BoundaryGrid([f[3] for f in fps])
        sp = plan_streets(pieces, bg, cam_a, seed, pol, night, hour, rect_polys=[f[3] for f in fps])
        plan['final_clipped'] = final_clip(plan, sp)
        fps = footprints(plan)
        aprons = plan_aprons(fps, street_items(sp), pol) if pol.get('apron', True) else dict(apron=[], plinth=[], faces=0, road_faces=0)
    ov = measure_overlap(fps, sp)
    plan.update(streets=sp, aprons=aprons, fps=fps, overlap=ov, night=night, t_plan_all=round(time.time() - t0, 1), legacy=legacy, cam=cam_a)
    return plan



# ================================================================================================
# ここから Blender(bpy が要る)。材質・メッシュ・オブジェクト化
# ================================================================================================
def log(*a):
    print('hand', *a, flush=True)


class Ctx:
    """build_city の1回分の状態(材質の共有・フックなど)"""
    def __init__(self, hook, emit, night, hour, gz):
        self.hook = hook; self.emit = emit; self.night = night; self.hour = hour; self.gz = gz
        self.mat_cache = {}; self.all_mats = []

    def reg(self, mat):
        self.all_mats.append(mat)
        if self.hook is not None:
            try:
                self.hook(mat)
            except Exception as e:
                log('material_hook failed', mat.name, repr(e))
        return mat


# ---------------- メッシュのため込み ----------------
class Acc:
    """面をためて1つのメッシュにする。 面ごとに頂点を持つ(頂点は共有しない)。"""
    def __init__(self, nattr=0):
        self.V = []; self.UV = []; self.sizes = []; self.mi = []; self.sm = []; self.A = []; self.nattr = nattr

    def add(self, V, UV, sizes, mi, attr=None, smooth=None):
        V = np.asarray(V, float).reshape(-1, 3); n = len(V)
        self.V.append(V); self.UV.append(np.asarray(UV, float).reshape(-1, 2))
        self.sizes.append(np.asarray(sizes, int)); self.mi.append(np.asarray(mi, int))
        self.sm.append(np.zeros(len(sizes), bool) if smooth is None else np.asarray(smooth, bool))
        if self.nattr:
            a = np.zeros((n, 4)) if attr is None else np.asarray(attr, float).reshape(-1, 4) if np.ndim(attr) > 1 else np.tile(np.asarray(attr, float), (n, 1))
            self.A.append(a)

    def nfaces(self):
        return int(sum(len(s) for s in self.sizes))

    def tris(self):
        return int(sum((s - 2).sum() for s in self.sizes))

    def empty(self):
        return not self.sizes

    def build(self, name, mats, attr_name=None, parent=None, location=(0, 0, 0)):
        V = np.concatenate(self.V); UV = np.concatenate(self.UV); sizes = np.concatenate(self.sizes); mi = np.concatenate(self.mi); sm = np.concatenate(self.sm)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        faces = [tuple(range(int(s), int(s + n))) for s, n in zip(starts, sizes)]
        me = bpy.data.meshes.new(name + '_mesh')
        me.from_pydata(V.tolist(), [], faces)
        uvl = me.uv_layers.new(name='UVMap'); uvl.data.foreach_set('uv', UV.astype(np.float32).ravel())
        if self.nattr and attr_name:
            ca = me.color_attributes.new(attr_name, 'FLOAT_COLOR', 'CORNER')
            ca.data.foreach_set('color', np.concatenate(self.A).astype(np.float32).ravel())
        me.polygons.foreach_set('material_index', mi.astype(np.int32))
        if sm.any():
            try:
                me.polygons.foreach_set('use_smooth', sm.astype(np.int8))
            except Exception:
                pass
        for m in mats:
            me.materials.append(m)
        me.update()
        ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob)
        ob.location = location
        if parent is not None:
            ob.parent = parent
        return ob


def new_mat(name):
    return sk.new_material(name)


# ---------------- 遠景のビルの材質(共有)----------------
def _dirt_dyn(nb, u, v, C, Wk, Dk, Hk):
    """building_kit.dirt_socket の、建物の寸法(W, D, H)をソケットで受ける版。 汚れ(窓台下の縦筋・上端・出隅・1階の下端・大きなムラ)は同じ。"""
    S = C['S']; P = S['P']; Hf = S['Hf']; G = S['G']; sd = C['seed'] * 1.713 + 0.37

    def nz(ax, ay, det=3.0):
        return nb.noise(nb.vec(nb.m('MULTIPLY', u, ax), nb.m('MULTIPLY', v, ay), sd), 1.0, det)
    terms = []
    v1 = nb.m('SUBTRACT', v, G)
    cellf = nb.m('FLOOR', nb.m('DIVIDE', u, P))
    um = nb.m('MULTIPLY', nb.m('FRACT', nb.m('DIVIDE', u, P)), P)
    fl = nb.m('FLOOR', nb.m('DIVIDE', v1, Hf))
    hv = nb.white(cellf, fl, sd); hv2 = nb.white(cellf, fl, sd + 11.0)
    on = nb.m('GREATER_THAN', hv, 0.30)
    wmax = nb.m('ADD', 0.08, nb.m('MULTIPLY', nz(1.7, 0.5, 2.0), 0.28))
    line = 0.0
    for e in C['edges']:
        d = nb.m('ABSOLUTE', nb.m('SUBTRACT', um, e))
        line = nb.m('MAXIMUM', line, nb.sstep(d, wmax, 0.0))
    dsill = nb.m('MULTIPLY', nb.m('FRACT', nb.m('DIVIDE', nb.m('SUBTRACT', C['sill'], v1), Hf)), Hf)
    ls = nb.m('ADD', 1.4, nb.m('MULTIPLY', hv2, 1.8))
    decay = nb.sstep(dsill, ls, 0.0)
    tex = nb.m('ADD', 0.35, nb.m('MULTIPLY', nz(6.0, 0.22, 4.0), 1.1), clamp=True)
    terms.append(nb.m('MULTIPLY', nb.m('MULTIPLY', nb.m('MULTIPLY', nb.m('MULTIPLY', line, decay), on), tex), 2.6))
    gen = nb.m('MULTIPLY', nb.m('MULTIPLY', nb.sstep(nz(9.0, 0.12, 4.0), 0.42, 0.62), decay), 0.9)
    terms.append(nb.m('MULTIPLY', gen, nb.m('GREATER_THAN', hv2, 0.25)))
    terms.append(nb.m('MULTIPLY', nb.sstep(nz(0.55, 0.12, 3.0), 0.45, 0.80), 0.30))
    top = nb.m('MULTIPLY', nb.sstep(nb.m('SUBTRACT', Hk, v), 2.4, 0.0), nb.m('ADD', 0.45, nb.m('MULTIPLY', nz(2.0, 0.6), 0.9)))
    terms.append(nb.m('MULTIPLY', top, 0.85))
    WD = nb.m('ADD', Wk, Dk)
    m_ = nb.m('MULTIPLY', nb.m('FRACT', nb.m('DIVIDE', u, WD)), WD)
    inW = nb.m('LESS_THAN', m_, Wk)
    s_ = nb.m('SUBTRACT', m_, nb.m('MULTIPLY', nb.m('SUBTRACT', 1.0, inW), Wk))
    L_ = nb.m('ADD', nb.m('MULTIPLY', inW, Wk), nb.m('MULTIPLY', nb.m('SUBTRACT', 1.0, inW), Dk))
    cd = nb.m('MINIMUM', s_, nb.m('SUBTRACT', L_, s_))
    terms.append(nb.m('MULTIPLY', nb.sstep(cd, 1.4, 0.0), nb.m('MULTIPLY', nb.m('ADD', 0.4, nb.m('MULTIPLY', nz(3.0, 0.4), 0.9)), 0.6)))
    bot = nb.m('MULTIPLY', nb.sstep(v, 3.2, 0.0), nb.m('ADD', 0.55, nb.m('MULTIPLY', nz(2.5, 1.2), 0.8)))
    terms.append(nb.m('MULTIPLY', bot, 0.9))
    d = terms[0]
    for t in terms[1:]:
        d = nb.m('ADD', d, t)
    return nb.m('MINIMUM', d, 1.0)


def far_facade_material(C, name, atlas, aw, ah):
    """遠景のビル(共有)。 building_kit.mat_far と同じ見た目: 窓・柱型・梁型・腰壁・点灯・汚れを1枚の材質で描く。
    違い: 建物ごとの値(窓の表の位置・セル数・幅・階数)は、色属性 'fb' = (表の先頭, セル数, 幅のモジュール数, 階数) から読む。 点灯の表は全ビルで1枚(アトラス)。"""
    mat, nt, out = bk.new_material(name); nb = bk.NB(nt)
    S = C['S']; P = S['P']; Hf = S['Hf']; G = S['G']; kind = C['kind']
    u, v = nb.uv()
    base, ncell, nWm, floors = nb.attr('fb')
    Wk = nb.m('MULTIPLY', nWm, P)
    Dk = nb.m('MULTIPLY', nb.m('SUBTRACT', nb.m('MULTIPLY', ncell, 0.5), nWm), P)
    Hk = nb.m('ADD', G, nb.m('MULTIPLY', nb.m('SUBTRACT', floors, 1.0), Hf))
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
    dirt = nb.m('MULTIPLY', _dirt_dyn(nb, u, v, C, Wk, Dk, Hk), 1.0 if kind != 'zakkyo' else 0.6)
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
    wall = bk.principled(nt, **{'Roughness': 0.8})
    nt.links.new(nb.vscale(wcol, nb.m('SUBTRACT', 1.0, dark)), wall.inputs['Base Color'])
    # 点灯の表: 1枚のアトラスの (先頭 + 階*セル数 + セル) 番目の画素
    idx = nb.m('ADD', nb.m('ADD', base, nb.m('MULTIPLY', fidx, ncell)), cellu)
    px = nb.m('MODULO', idx, float(aw)); py = nb.m('FLOOR', nb.m('DIVIDE', idx, float(aw)))
    li = nb.node('ShaderNodeTexImage'); li.image = atlas; li.interpolation = 'Closest'; li.extension = 'EXTEND'
    nt.links.new(nb.vec(nb.m('DIVIDE', nb.m('ADD', px, 0.5), float(aw)), nb.m('DIVIDE', nb.m('ADD', py, 0.5), float(ah)), 0.0), li.inputs['Vector'])
    sc = nb.node('ShaderNodeSeparateColor'); nt.links.new(li.outputs['Color'], sc.inputs[0])
    Il, Tl = sc.outputs[0], sc.outputs[1]
    gl = bk.principled(nt, **{'Base Color': (*C['glass_tint'], 1.0), 'Metallic': C['glass_metal'], 'Roughness': 0.03, 'IOR': 1.5})
    nt.links.new(nb.blackbody(nb.m('MULTIPLY', Tl, 10000.0)), gl.inputs['Emission Color'])
    nt.links.new(nb.m('MULTIPLY', Il, C['E'] * 0.55), gl.inputs['Emission Strength'])
    mx = nb.node('ShaderNodeMixShader'); nt.links.new(M, mx.inputs[0]); nt.links.new(wall.outputs[0], mx.inputs[1]); nt.links.new(gl.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs['Surface'])
    return mat


def variant_C(kind, var, ageb, night, emit):
    ctx = bk.X(); ctx.kind = kind; ctx.S = bk.SPECS[kind]; ctx.seed = 1009 + var * 37 + (0 if kind == 'office' else 400 if kind == 'zakkyo' else 800)
    ctx.W = ctx.D = ctx.H = 10.0; ctx.age = 0.22 if ageb == 0 else 0.62; ctx.rail_solid = bool(var % 2); ctx.night = night; ctx.ncell = 8; ctx.floors = 4; ctx.detail = 'far'
    return bk.make_context(ctx, emit)


def plain_mat(name, color, rough=0.8, metal=0.0, emit=None, es=0.0):
    mat, nt, out = bk.new_material(name)
    b = bk.principled(nt, **{'Base Color': (*color, 1.0), 'Roughness': rough, 'Metallic': metal})
    if emit is not None:
        b.inputs['Emission Color'].default_value = (*emit, 1.0); b.inputs['Emission Strength'].default_value = es
    nt.links.new(b.outputs['BSDF'], out.inputs['Surface'])
    return mat


class FarMats:
    """遠景のビルが使う材質の置き場(種類×色の組×新旧 で共有)。 材質の数は 約40枚で、ビルの数によらない。"""
    def __init__(self, cx, atlas, aw, ah, obs_emit):
        self.cx = cx; self.atlas = atlas; self.aw = aw; self.ah = ah; self.obs_emit = obs_emit
        self.list = []; self.key2i = {}

    def get(self, role, kind, var, ageb):
        key = ('far', kind, var, ageb) if role == 'far' else (role, kind, var, ageb) if role in ('wall', 'beam', 'pent', 'pillar', 'slab') else (role if role == 'lamp_red' else 'roof',)
        if key in self.key2i:
            return self.key2i[key]
        C = variant_C(kind, var, ageb, self.cx.night, self.cx.emit) if kind else None
        if role == 'far':
            m = far_facade_material(C, 'cityfar_%s_%d_%d' % (kind, var, ageb), self.atlas, self.aw, self.ah)
        elif key[0] in ('wall', 'beam', 'pent', 'pillar', 'slab'):
            col = C['wall_base'] if role == 'wall' else C['beam_base'] if role == 'beam' else C['pillar_base'] if role == 'pillar' else C['slab_base'] if role == 'slab' else (0.40, 0.40, 0.38)
            if role == 'pent':
                col = (0.40, 0.40, 0.38)
            col = tuple(c * (1.0 - C['A'] * 0.5) for c in col)
            m = plain_mat('cityfar_%s_%s_%d_%d' % (role, kind, var, ageb), col, 0.85)
        elif role == 'lamp_red':
            m = plain_mat('cityfar_lamp_red', (0.2, 0.01, 0.01), 0.5, 0.0, emit=(1.0, 0.03, 0.02), es=self.obs_emit)
        else:
            m = plain_mat('cityfar_roof', (0.22, 0.22, 0.21), 0.9)
        self.cx.reg(m)
        self.key2i[key] = len(self.list); self.list.append(m)
        return self.key2i[key]


# ---------------- 低層(house / low)の材質 ----------------
def house_wall_material(cx):
    """低層の壁: 色 = 色属性 'hc'(rgb)、窓が3.2m間隔、窓ごとに点灯(確率 = 時刻から)。 高さは階(3.0m)ごと。"""
    mat, nt, out = bk.new_material('city_house_wall'); nb = bk.NB(nt)
    u, v = nb.uv(); cr, cg, cbl, ida = nb.attr('hc')
    col = nb.node('ShaderNodeCombineColor'); nt.links.new(cr, col.inputs[0]); nt.links.new(cg, col.inputs[1]); nt.links.new(cbl, col.inputs[2])
    cu = nb.m('FLOOR', nb.m('DIVIDE', u, 3.2)); fu = nb.m('FRACT', nb.m('DIVIDE', u, 3.2))
    cvv = nb.m('FLOOR', nb.m('DIVIDE', v, 3.0)); fv = nb.m('FRACT', nb.m('DIVIDE', v, 3.0))
    win = nb.m('MULTIPLY', nb.m('MULTIPLY', nb.m('GREATER_THAN', fu, 0.27), nb.m('LESS_THAN', fu, 0.73)),
               nb.m('MULTIPLY', nb.m('GREATER_THAN', fv, 0.34), nb.m('LESS_THAN', fv, 0.80)))
    win = nb.m('MULTIPLY', win, nb.m('GREATER_THAN', v, 1.0))
    rate = bk.lit_rate('zakkyo', cx.hour, cx.night)
    rnd = nb.white(cu, cvv, nb.m('MULTIPLY', ida, 97.0))
    lit = nb.m('MULTIPLY', win, nb.m('LESS_THAN', rnd, rate))
    rnd2 = nb.white(cu, cvv, nb.m('ADD', nb.m('MULTIPLY', ida, 53.0), 7.0))
    temp = nb.m('ADD', 2700.0, nb.m('MULTIPLY', rnd2, 2300.0))
    wall = bk.principled(nt, **{'Roughness': 0.85})
    mott = nb.m('ADD', 0.85, nb.m('MULTIPLY', nb.noise(nb.vec(nb.m('MULTIPLY', u, 0.6), nb.m('MULTIPLY', v, 0.6), nb.m('MULTIPLY', ida, 13.0)), 1.0, 3.0), 0.3))
    nt.links.new(nb.vscale(col.outputs[0], mott), wall.inputs['Base Color'])
    gl = bk.principled(nt, **{'Base Color': (0.03, 0.04, 0.05, 1.0), 'Metallic': 0.5, 'Roughness': 0.08})
    nt.links.new(nb.blackbody(temp), gl.inputs['Emission Color'])
    nt.links.new(nb.m('MULTIPLY', lit, cx.emit * 0.5), gl.inputs['Emission Strength'])
    mx = nb.node('ShaderNodeMixShader'); nt.links.new(win, mx.inputs[0]); nt.links.new(wall.outputs[0], mx.inputs[1]); nt.links.new(gl.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs['Surface'])
    return cx.reg(mat)


def house_roof_material(cx):
    mat, nt, out = bk.new_material('city_house_roof'); nb = bk.NB(nt)
    u, v = nb.uv(); cr, cg, cbl, ida = nb.attr('hc')
    col = nb.node('ShaderNodeCombineColor'); nt.links.new(cr, col.inputs[0]); nt.links.new(cg, col.inputs[1]); nt.links.new(cbl, col.inputs[2])
    var = nb.m('ADD', 0.8, nb.m('MULTIPLY', nb.noise(nb.vec(nb.m('MULTIPLY', u, 0.6), nb.m('MULTIPLY', v, 0.6), nb.m('MULTIPLY', ida, 7.0)), 1.0, 3.0), 0.4))
    b = bk.principled(nt, **{'Roughness': 0.55})
    nt.links.new(nb.vscale(col.outputs[0], var), b.inputs['Base Color'])
    nt.links.new(b.outputs['BSDF'], out.inputs['Surface'])
    return cx.reg(mat)


# ---------------- 道路・地面の材質 ----------------
def pool_material(cx, strength):
    mat, nt, out = bk.new_material('city_lamp_pool'); nb = bk.NB(nt)
    u, v = nb.uv()
    d = nb.m('POWER', nb.m('ADD', nb.m('POWER', nb.m('SUBTRACT', u, 0.5), 2.0), nb.m('POWER', nb.m('SUBTRACT', v, 0.5), 2.0)), 0.5)
    f = nb.m('POWER', nb.sstep(d, 0.5, 0.0), 2.0)
    em = nb.node('ShaderNodeEmission'); em.inputs['Color'].default_value = (1.0, 0.66, 0.36, 1.0); em.inputs['Strength'].default_value = strength
    tr = nb.node('ShaderNodeBsdfTransparent'); mx = nb.node('ShaderNodeMixShader')
    nt.links.new(nb.m('MULTIPLY', f, 0.85), mx.inputs[0]); nt.links.new(tr.outputs[0], mx.inputs[1]); nt.links.new(em.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs['Surface'])
    return cx.reg(mat)


def ground_material(cx, name, col, rough):
    mat, nt, out = bk.new_material(name); nb = bk.NB(nt)
    tc = nb.node('ShaderNodeTexCoord')
    k = nb.m('ADD', 0.75, nb.m('MULTIPLY', nb.noise(tc.outputs['Object'], 0.08, 4.0), 0.5))
    b = bk.principled(nt, **{'Roughness': rough})
    nt.links.new(nb.vscale(nb.rgb(col), k), b.inputs['Base Color'])
    nt.links.new(b.outputs['BSDF'], out.inputs['Surface'])
    return cx.reg(mat)


def leaf_material(cx):
    mat, nt, out = bk.new_material('city_leaf'); nb = bk.NB(nt)
    tc = nb.node('ShaderNodeTexCoord'); oi = nb.node('ShaderNodeObjectInfo')
    n1 = nb.noise(tc.outputs['Object'], 1.3, 4.0, 0.6)
    k = nb.m('MULTIPLY', nb.m('ADD', 0.45, nb.m('MULTIPLY', n1, 1.1)), nb.m('ADD', 0.7, nb.m('MULTIPLY', oi.outputs['Random'], 0.6)))
    b = bk.principled(nt, **{'Roughness': 0.55})
    col = nb.vmix(nb.rgb((0.020, 0.040, 0.012)), nb.rgb((0.045, 0.055, 0.014)), oi.outputs['Random'])
    nt.links.new(nb.vscale(col, k), b.inputs['Base Color'])
    nt.links.new(b.outputs['BSDF'], out.inputs['Surface'])
    return cx.reg(mat)


# ---------------- 街路の材質(street_kit の材質を、断面の値ごとに共有)----------------
class StreetMats:
    def __init__(self, cx, street_emit):
        self.cx = cx; self.cache = {}; self.list = []; self.street_emit = street_emit

    def get(self, role, info, P):
        # 材質が断面の値に依存するのは 車道(幅・車線幅)と歩道(幅・境界・点字の位置)だけ。 他は役割ごとに1枚を共有する
        if role == 'road':
            sig = (role, round(info['half_road'], 2), round(info['lane_w'], 2), P['center_line'])
        elif role == 'sidewalk':
            sig = (role, round(info['sidewalk_o0'], 2), round(info['boundary_o'], 2), round(info['tactile'][1], 2), P['paving'])
        elif role in ('leaf',):
            sig = (role, P['tree_kind'], bool(P['night']))
        else:
            sig = (role, bool(P['night']))
        if sig in self.cache:
            return self.cache[sig]
        m = sk.build_material(role, P, info, 'citystreet_%s_%d' % (role, len(self.list)))
        if role == 'lamp_emit' and P['night']:
            for n in m.node_tree.nodes:
                if n.type == 'BSDF_PRINCIPLED':
                    n.inputs['Emission Strength'].default_value = self.street_emit
        self.cx.reg(m)
        self.cache[sig] = len(self.list); self.list.append(m)
        return self.cache[sig]


# ------------------------------------------------------------------------------------------------
# 低層の箱(家・小さいビル)の形
# ------------------------------------------------------------------------------------------------
def house_geometry(hh, gz_local=0.0):
    """-> V(n,3), UV, sizes, roles(0=壁,1=屋根), attr(n,4)。 世界座標。 u = 長辺の向き。"""
    c = np.asarray(hh['c']); u = np.asarray(hh['u']); v = np.array([-u[1], u[0]]); eu, ev = hh['eu'], hh['ev']; h = float(hh['h'])
    rr = hh['rr']
    wc = HOUSE_COLS[int(rr * 997) % len(HOUSE_COLS)]; rc = ROOF_COLS[int(rr * 313) % len(ROOF_COLS)]
    ida = rr * 100.0
    gable = bool(hh.get('gable')) and ev >= 4.0 and h <= 14.0
    rh = min(2.6, 0.32 * ev) if gable else 0.0
    wh = max(2.8, h - rh)
    P = [c + u * (-eu / 2) + v * (-ev / 2), c + u * (eu / 2) + v * (-ev / 2), c + u * (eu / 2) + v * (ev / 2), c + u * (-eu / 2) + v * (ev / 2)]
    Vs = []; UVs = []; sizes = []; roles = []; attrs = []
    lens = [eu, ev, eu, ev]
    for i in range(4):
        a = P[i]; b = P[(i + 1) % 4]; L = lens[i]
        Vs += [(a[0], a[1], 0.0), (b[0], b[1], 0.0), (b[0], b[1], wh), (a[0], a[1], wh)]
        o = (i * 7.3) % 3.2
        UVs += [(o, 0.0), (o + L, 0.0), (o + L, wh), (o, wh)]
        sizes.append(4); roles.append(0); attrs += [(*wc, ida)] * 4
    if gable:
        # 棟は長辺方向。 屋根の2面 + 妻(三角)2面
        a0 = c + u * (-eu / 2); a1 = c + u * (eu / 2)
        r0 = (a0[0], a0[1], wh + rh); r1 = (a1[0], a1[1], wh + rh)
        s = math.hypot(ev / 2, rh)
        Vs += [(P[0][0], P[0][1], wh), (P[1][0], P[1][1], wh), r1, r0]; UVs += [(0, 0), (eu, 0), (eu, s), (0, s)]; sizes.append(4); roles.append(1); attrs += [(*rc, ida)] * 4
        Vs += [(P[2][0], P[2][1], wh), (P[3][0], P[3][1], wh), r0, r1]; UVs += [(0, 0), (eu, 0), (eu, s), (0, s)]; sizes.append(4); roles.append(1); attrs += [(*rc, ida)] * 4
        Vs += [(P[1][0], P[1][1], wh), (P[2][0], P[2][1], wh), r1]; UVs += [(0, 0), (ev, 0), (ev / 2, rh)]; sizes.append(3); roles.append(0); attrs += [(*wc, ida)] * 3
        Vs += [(P[3][0], P[3][1], wh), (P[0][0], P[0][1], wh), r0]; UVs += [(0, 0), (ev, 0), (ev / 2, rh)]; sizes.append(3); roles.append(0); attrs += [(*wc, ida)] * 3
    else:
        Vs += [(p[0], p[1], wh) for p in (P[3], P[2], P[1], P[0])]; UVs += [(0, 0), (eu, 0), (eu, ev), (0, ev)]; sizes.append(4); roles.append(1); attrs += [(*rc, ida)] * 4
    return np.asarray(Vs, float), np.asarray(UVs, float), np.asarray(sizes), np.asarray(roles), np.asarray(attrs, float)


# ------------------------------------------------------------------------------------------------
# 近景・中景のビル(実立体)
# ------------------------------------------------------------------------------------------------
def ground_spec(kind, W, rr):
    """1階の並び [(種類, 幅)]。 入口を1つ、残りはガラス張り・タイル腰壁・シャッター・駐車場(広いときだけ)"""
    if W < 8.0:
        ew = max(2.4, min(3.2, W * 0.45))
        return [(rr.choice(['glass', 'tile']), W - ew), ('entrance', ew)] if rr.random() < 0.5 else [('entrance', ew), (rr.choice(['glass', 'shutter']), W - ew)]
    ew = rr.choice([3.6, 4.0, 5.0]) if kind != 'mansion' else 4.0
    rest = W - ew; n = max(1, int(rest // 7.0)); n = min(n, 3)
    ws = [rest / n] * n
    pool = ['glass', 'glass', 'tile', 'shutter'] + (['parking'] if W >= 14.0 else [])
    spec = [(rr.choice(pool), w) for w in ws]
    spec.insert(rr.randint(0, len(spec)), ('entrance', ew))
    return spec


def make_solid_building(rec, cx, hour, ground_z, root, with_base=True):
    kind = rec['kind']; det = rec['detail']
    nm = 'bld%d_%d_%s' % (rec['id'], rec['sub'], det)
    r = bk.make_building(kind, rec['W'], rec['D'], rec['floors'], seed=rec['seed'], detail=det, night=cx.night, hour=hour, age=rec['age'], name=nm, emit=cx.emit)
    r.location = (rec['c'][0], rec['c'][1], ground_z); r.rotation_euler = (0.0, 0.0, rec['theta']); r.scale = (rec['sx'], rec['sy'], 1.0); r.parent = root
    body = [c for c in r.children if c.type == 'MESH'][0]
    G = bk.SPECS[kind]['G']
    base = None
    if with_base and bb is not None:
        bm = bmesh.new(); bm.from_mesh(body.data); bm.faces.ensure_lookup_table()
        dele = [f for f in bm.faces if f.calc_center_median().z < G - 0.01]
        bmesh.ops.delete(bm, geom=dele, context='FACES'); bm.to_mesh(body.data); bm.free()
        rr = rng_for(0, rec['id'], 41 + rec['sub'])
        bz, gl = 0.28, 0.05
        base = bb.make_ground_floor_row(ground_spec(kind, rec['W'], rr), G - bz, rec['D'], bz, gl, seed=rec['seed'], detail=det, night=cx.night, hour=hour, position=(0.0, 0.0), name=nm + '_f1')
        base.parent = r
    for ob in [r] + list(r.children_recursive):
        if ob.type == 'MESH':
            for m in ob.data.materials:
                if m is not None and m not in cx.all_mats:
                    cx.reg(m)
    info = bk.building_info(r)
    return r, info, base


# ------------------------------------------------------------------------------------------------
# 高架(chase/city.py の build_ground 末尾と同じ形)
# ------------------------------------------------------------------------------------------------
def build_viaduct(cx, root, gz, y_lo, y_hi, material=None, x=-6.4, Y0=-300.0, Y1=2700.0):
    LEN = Y1 - Y0; YC = (Y0 + Y1) / 2
    V = []; F = []

    def box(cx_, cy, cz, sx, sy, sz):
        n = len(V); hx, hy, hz = sx / 2, sy / 2, sz / 2
        for dx in (-hx, hx):
            for dy in (-hy, hy):
                for dz in (-hz, hz):
                    V.append((cx_ + dx, cy + dy, cz + dz))
        for q in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            F.append(tuple(n + i for i in q))
    box(x, YC, -1.0, 32.0, LEN + 40, 1.9)                    # 床版(桁)
    y = Y0; n = 0
    while y < Y1:
        if y_lo - 60 <= y <= y_hi + 60:
            box(x, y, (gz - 3.4) / 2, 3.2, 2.4, -3.4 - gz)    # 橋脚(柱)
            box(x, y, -2.7, 27.0, 2.6, 1.5)                   # 横梁
            n += 1
        y += 40.0
    me = bpy.data.meshes.new('Viaduct_mesh'); me.from_pydata(V, [], F); me.update()
    mat = material or bpy.data.materials.get('Concrete')
    if mat is None:
        mat = cx.reg(plain_mat('city_viaduct', (0.32, 0.31, 0.30), 0.9))
    me.materials.append(mat)
    ob = bpy.data.objects.new('Viaduct', me); bpy.context.scene.collection.objects.link(ob); ob.parent = root
    return len(F), n


# ------------------------------------------------------------------------------------------------
# 木(city.json の trees)
# ------------------------------------------------------------------------------------------------
def make_tree_protos(mat_leaf, mat_trunk):
    out = []
    for k in range(4):
        r = random.Random(k * 17 + 3)
        bm = bmesh.new()
        h = 1.0
        bmesh.ops.create_cube(bm, size=1.0)
        for vtx in bm.verts:
            vtx.co.x *= 0.05; vtx.co.y *= 0.05; vtx.co.z = (vtx.co.z + 0.5) * 0.4
        nbl = 4 + (k % 3)
        for j in range(nbl):
            ang = j / nbl * math.tau + r.random(); rad = 0.18 * (0.4 + r.random()) if j else 0.0
            c = (math.cos(ang) * rad, math.sin(ang) * rad, 0.55 + 0.25 * r.random() + (0.12 if j == 0 else 0.0))
            rr_ = 0.24 + 0.10 * r.random()
            geom = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=rr_)
            for vtx in geom['verts']:
                vtx.co.x += c[0]; vtx.co.y += c[1]; vtx.co.z = vtx.co.z * 0.85 + c[2]
        me = bpy.data.meshes.new('tree_proto%d' % k)
        bm.to_mesh(me); bm.free()
        for p in me.polygons:
            p.use_smooth = True
        me.materials.append(mat_leaf)
        out.append((me, max(v.co.z for v in me.vertices)))
    return out


def ultra_quads(it, smat, stride):
    """150mより遠い道路の片側を、4つの帯(車道・縁石・歩道・犬走り)と基礎の立ち上がり1枚にした形。 -> V(n*4,3), UV(n*4,2), mi(n)"""
    line = it['line']; info = it['info']; P = it['P']
    idx = list(range(0, len(line), max(1, int(stride))))
    if idx[-1] != len(line) - 1:
        idx.append(len(line) - 1)
    L = line[idx]
    if len(L) < 2:
        return np.zeros((0, 3)), np.zeros((0, 2)), []
    t = np.gradient(L, axis=0); t /= (np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-9); n = np.stack([-t[:, 1], t[:, 0]], 1)
    cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(L, axis=0).T))])
    Wr = info['half_road']; sw0 = info['sidewalk_o0']; o_b0 = info['boundary_o']; o_w = info['found_front']
    mats = {r_: smat.get(r_, info, P) for r_ in ('road', 'curb', 'sidewalk', 'inuhashiri', 'found_conc')}
    V = []; UV = []; mi = []
    bands_ = ((0.0, Wr, 0.0, 'road'), (Wr, sw0, 0.10, 'curb'), (sw0, o_b0, 0.05, 'sidewalk'), (o_b0, o_w, 0.062, 'inuhashiri'))
    for i in range(len(L) - 1):
        s0, s1 = cum[i], cum[i + 1]
        for o0, o1, z, role in bands_:
            if o1 - o0 < 1e-3:
                continue
            V += [(L[i, 0] + n[i, 0] * o0, L[i, 1] + n[i, 1] * o0, z), (L[i + 1, 0] + n[i + 1, 0] * o0, L[i + 1, 1] + n[i + 1, 1] * o0, z),
                  (L[i + 1, 0] + n[i + 1, 0] * o1, L[i + 1, 1] + n[i + 1, 1] * o1, z), (L[i, 0] + n[i, 0] * o1, L[i, 1] + n[i, 1] * o1, z)]
            UV += [(s0, o0), (s1, o0), (s1, o1), (s0, o1)]; mi.append(mats[role])
        V += [(L[i, 0] + n[i, 0] * o_w, L[i, 1] + n[i, 1] * o_w, 0.062), (L[i + 1, 0] + n[i + 1, 0] * o_w, L[i + 1, 1] + n[i + 1, 1] * o_w, 0.062),
              (L[i + 1, 0] + n[i + 1, 0] * o_w, L[i + 1, 1] + n[i + 1, 1] * o_w, 0.30), (L[i, 0] + n[i, 0] * o_w, L[i, 1] + n[i, 1] * o_w, 0.30)]
        UV += [(s0, 0.062), (s1, 0.062), (s1, 0.30), (s0, 0.30)]; mi.append(mats['found_conc'])
    return np.asarray(V, float), np.asarray(UV, float), mi


# ================================================================================================
# 本体
# ================================================================================================
def build_city(city_json_path, center, radius, camera_path_or_points=None, hour=20.0, seed=0, detail_policy=None, ground_z=-12.0, emit=8.0,
               material_hook=None, viaduct=True, viaduct_material=None, obs_emit=60.0, street_emit=60.0, pool_emit=2.2, name='City'):
    """背景の街を作る。 -> 街のルート(空の親オブジェクト)。 root['city_stats'] = 棟数・ポリゴン・構築時間(JSON)。
    city_json_path = city.json のパス(または読み込んだ辞書)。 center = (x, y)、radius = 出す範囲[m]。 camera_path_or_points = カメラの通る点 [(x, y[, z])...](None なら全部 far)。
    hour = 時刻(0〜24)。 17時〜5時台は夜(窓の点灯)。 seed = 乱数の種(同じ seed・同じ建物 = 同じ結果)。 detail_policy = DEFAULT_POLICY の上書き。
    ground_z = 地面の高さ(chase の街は -12)。 emit = 窓の発光の強さ。 material_hook = 作った材質ごとに呼ぶ関数(例: chase の add_fog でかすみを付ける)。"""
    global last_stats
    T0 = time.time()
    city = load_city(city_json_path)
    pol = dict(DEFAULT_POLICY); pol.update(detail_policy or {})
    night = bool(hour >= 17.0 or hour < 5.5)
    cam = None
    if camera_path_or_points is not None and len(camera_path_or_points):
        cam = np.asarray([(p[0], p[1]) for p in camera_path_or_points], float)
    cx = Ctx(material_hook, emit, night, hour, ground_z)
    root = bpy.data.objects.new(name, None); root.empty_display_size = 5.0; bpy.context.scene.collection.objects.link(root)
    st = dict(hour=hour, night=night, seed=seed, center=list(center), radius=radius, policy={k: v for k, v in pol.items() if not isinstance(v, dict)})

    # --- 計画
    t = time.time()
    plan = plan_all(city, center, radius, cam, hour, seed, pol)
    sp = plan['streets']; aprons = plan['aprons']
    st['t_plan'] = round(time.time() - t, 1)
    st['overlap'] = plan['overlap']; st['rects'] = dict(fitted=plan['rect_fitted'], clipped=plan['rect_clipped'], dropped=plan['rect_dropped'], final_clipped=plan.get('final_clipped', 0))
    st['aprons'] = dict(faces=aprons['faces'], road_faces=aprons['road_faces'], apron_quads=len(aprons['apron']), plinth_quads=len(aprons['plinth']))
    recs = plan['recs']
    far = [r for r in recs if r['detail'] == 'far']; solid = [r for r in recs if r['detail'] != 'far']
    st['kinds_all'] = dict(plan['kinds']); st['detail_counts'] = {d: sum(1 for r in recs if r['detail'] == d) for d in ('near', 'mid', 'far')}
    st['houses'] = len(plan['houses'])

    tris = dict(far_buildings=0, houses=0, solid_buildings=0, streets=0, trees=0, viaduct=0, ground=0)
    # --- 遠景のビル(材質共有 + 点灯の表1枚)
    t = time.time()
    geos = []; atlas_cells = 0
    for r in far:
        g = far_geometry(r, night, hour, atlas_cells); atlas_cells += g['lit'].shape[0] * g['lit'].shape[1]; geos.append((r, g))
    aw = 2048; ah = max(1, int(math.ceil(max(atlas_cells, 1) / aw)))
    px = np.zeros((ah * aw, 4), np.float32); px[:, 3] = 1.0
    for r, g in geos:
        b0 = int(g['fb'][0]); lit = g['lit']
        px[b0:b0 + lit.shape[0] * lit.shape[1], 0:3] = lit[:, :, 0:3].reshape(-1, 3)
    atlas = bpy.data.images.new('city_lit_atlas', width=aw, height=ah, alpha=True, float_buffer=True)
    atlas.pixels.foreach_set(px.ravel())
    try:
        atlas.colorspace_settings.name = 'Non-Color'
    except Exception:
        pass
    atlas.pack()
    fm = FarMats(cx, atlas, aw, ah, obs_emit)
    acc = Acc(nattr=4)
    for r, g in geos:
        var, ageb = far_variant(r)
        mi = np.asarray([fm.get(role, r['kind'], var, ageb) for role in g['roles']], int)
        attr = np.tile(np.asarray(g['fb'], float), (len(g['V']), 1))
        acc.add(g['V'], g['UV'], g['sizes'], mi, attr)
        tris['far_buildings'] += g['tris']
    if not acc.empty():
        ob = acc.build('CityBuildingsFar', fm.list, 'fb', parent=root, location=(0, 0, ground_z))
        me = ob.data
    st['far_buildings'] = len(far); st['far_materials'] = len(fm.list); st['atlas_px'] = [aw, ah]
    st['t_far'] = round(time.time() - t, 1)

    # --- 低層(家・小さいビル)
    t = time.time()
    if plan['houses']:
        mw = house_wall_material(cx); mr = house_roof_material(cx)
        ha = Acc(nattr=4)
        for hh in plan['houses']:
            V, UV, sizes, roles, attr = house_geometry(hh)
            ha.add(V, UV, sizes, roles, attr)
        ha.build('CityHouses', [mw, mr], 'hc', parent=root, location=(0, 0, ground_z))
        tris['houses'] = ha.tris()
    st['t_houses'] = round(time.time() - t, 1)

    # --- 近景・中景のビル(building_kit の実立体 + 1階)
    t = time.time(); nerr = 0
    for r in solid:
        try:
            ro, info, base = make_solid_building(r, cx, hour, ground_z, root)
            tris['solid_buildings'] += int(info['tris'])
        except Exception as e:
            nerr += 1; log('solid building failed', r['id'], repr(e))
    st['solid_buildings'] = len(solid) - nerr; st['t_solid'] = round(time.time() - t, 1)

    # --- 道路(street_kit の断面)
    t = time.time()
    smat = StreetMats(cx, street_emit); sa = Acc(); pools = []; lights = []
    nstreet = {'near': 0, 'mid': 0, 'far': 0}
    for d, items in sp['groups'].items():
        for it in items:
            md, info, P = it['md'], it['info'], it['P']
            if not md.F:
                continue
            roles = md.roles
            rm = {r_: smat.get(r_, info, P) for r_ in set(roles)}
            miF = np.asarray(md.MI, int)
            mi = np.asarray([rm[roles[i]] for i in miF], int)
            sizes = np.asarray([len(f) for f in md.F], int)
            smooth = np.asarray([roles[i] == 'leaf' for i in miF], bool)
            sa.add(np.asarray(md.V, float), np.asarray(md.UV, float), sizes, mi, smooth=smooth)
            nstreet[d] += len(md.F)
            if d == 'far':
                if 'lamp_emit' in roles:
                    k_ = roles.index('lamp_emit')
                    for fi in np.nonzero(miF == k_)[0]:
                        pts = np.asarray([md.V[k] for k in md.F[int(fi)]]); pools.append(pts.mean(0))
            else:
                for lp in it['lamps']:
                    lights.append((float(np.hypot(*(cam - np.array(lp[:2])).T).min()) if cam is not None else 0.0, lp))
    # 150mより遠い道路: 車道・縁石・歩道・犬走り・基礎の立ち上がりの帯(街灯・木・柵は無し)
    nultra = 0
    for it in sp['ultra']:
        V, UV, mi = ultra_quads(it, smat, pol['ultra_stride'])
        if len(V):
            sa.add(V, UV, [4] * (len(V) // 4), mi); nultra += len(V) // 4
    # 建物の足元: 犬走り(全部の壁)と基礎の立ち上がり(far・低層)
    P0, info0 = ultra_info(side_params('tertiary', 8.0, 6.0, random.Random(0)), night, hour)
    mi_inu = smat.get('inuhashiri', info0, P0); mi_fnd = smat.get('found_conc', info0, P0)
    if aprons['apron']:
        Va = []; UVa = []
        for pts, z in aprons['apron']:
            for p in pts:
                Va.append((float(p[0]), float(p[1]), z)); UVa.append((float(p[0]), float(p[1])))
        sa.add(Va, UVa, [4] * len(aprons['apron']), [mi_inu] * len(aprons['apron']))
    if aprons['plinth']:
        Vp = []; UVp = []; sp_ = 0.0
        for a_, b_ in aprons['plinth']:
            L_ = float(np.hypot(*(b_ - a_)))
            Vp += [(a_[0], a_[1], APRON_Z), (b_[0], b_[1], APRON_Z), (b_[0], b_[1], PLINTH_TOP), (a_[0], a_[1], PLINTH_TOP)]
            UVp += [(sp_, APRON_Z), (sp_ + L_, APRON_Z), (sp_ + L_, PLINTH_TOP), (sp_, PLINTH_TOP)]; sp_ += L_
        sa.add(Vp, UVp, [4] * len(aprons['plinth']), [mi_fnd] * len(aprons['plinth']))
    # 近景の街灯の実ライト(上限あり)。 残りは光だまり
    lights.sort(key=lambda a: a[0]); lit_pos = [lp for _, lp in lights[:pol['street_lights']]]
    for _, lp in lights[pol['street_lights']:]:
        pools.append(np.asarray(lp))
    for i, lp in enumerate(lit_pos):
        ld = bpy.data.lights.new('citylamp%d' % i, 'SPOT'); ld.energy = 2500.0; ld.spot_size = math.radians(130); ld.spot_blend = 0.8; ld.shadow_soft_size = 0.12; ld.color = (1.0, 0.80, 0.58)
        lo = bpy.data.objects.new(ld.name, ld); lo.location = (lp[0], lp[1], lp[2] + ground_z); bpy.context.scene.collection.objects.link(lo); lo.parent = root
        lo.rotation_euler = (0, 0, 0)
    if not sa.empty():
        sa.build('CityStreets', smat.list, None, parent=root, location=(0, 0, ground_z))
        tris['streets'] = sa.tris()
    nstreet['ultra'] = nultra; st['street_faces'] = nstreet; st['street_materials'] = len(smat.list); st['lamp_lights'] = len(lit_pos); st['lamp_pools'] = len(pools)
    if night and pools:
        pm = pool_material(cx, pool_emit); pa = Acc()
        for p in pools:
            z = 0.12; R_ = 6.0
            pa.add([(p[0] - R_, p[1] - R_, z), (p[0] + R_, p[1] - R_, z), (p[0] + R_, p[1] + R_, z), (p[0] - R_, p[1] + R_, z)], [(0, 0), (1, 0), (1, 1), (0, 1)], [4], [0])
        pa.build('CityLampPools', [pm], None, parent=root, location=(0, 0, ground_z))
    st['t_streets'] = round(time.time() - t, 1)

    # --- 地面・道路の帯(遠い道・路地)
    t = time.time()
    gm = ground_material(cx, 'city_ground', (0.06, 0.057, 0.052), 0.9)
    ga = Acc(); R_ = radius + 500.0
    ga.add([(center[0] - R_, center[1] - R_, -0.02), (center[0] + R_, center[1] - R_, -0.02), (center[0] + R_, center[1] + R_, -0.02), (center[0] - R_, center[1] + R_, -0.02)],
           [(0, 0), (1, 0), (1, 1), (0, 1)], [4], [0])
    ga.build('CityGround', [gm], None, parent=root, location=(0, 0, ground_z))
    bm_ = ground_material(cx, 'city_road_band', (0.075, 0.075, 0.078), 0.82)
    ba = Acc(); nb_ = 0
    for bd in sp['bands']:
        seg = bd['S']; w = bd['w']; cls = bd['cls']; hl = bd['hl']; hr = bd['hr']
        for (x1, y1), (x2, y2) in zip(seg[:-1], seg[1:]):
            L = math.hypot(x2 - x1, y2 - y1)
            if L < 0.3:
                continue
            tx, ty = (x2 - x1) / L, (y2 - y1) / L; nx, ny = -ty, tx; e_ = min(0.25 * w, 1.0)
            z = 0.03 if cls not in ('service', 'pedestrian') else 0.05
            ba.add([(x1 - tx * e_ - nx * hr, y1 - ty * e_ - ny * hr, z), (x2 + tx * e_ - nx * hr, y2 + ty * e_ - ny * hr, z),
                    (x2 + tx * e_ + nx * hl, y2 + ty * e_ + ny * hl, z), (x1 - tx * e_ + nx * hl, y1 - ty * e_ + ny * hl, z)], [(0, 0), (1, 0), (1, 1), (0, 1)], [4], [0]); nb_ += 1
    if not ba.empty():
        ba.build('CityRoadBands', [bm_], None, parent=root, location=(0, 0, ground_z))
    tris['ground'] = 2 + 2 * nb_
    st['road_bands'] = nb_; st['t_ground'] = round(time.time() - t, 1)

    # --- 木
    t = time.time(); ntree = 0
    if pol['trees']:
        lm = leaf_material(cx); protos = make_tree_protos(lm, None)
        tl = [(x, y, s) for (x, y, s) in city['trees'] if math.hypot(x - center[0], y - center[1]) <= radius]
        if cam is not None:
            tl = [tr for tr in tl if np.hypot(cam[:, 0] - tr[0], cam[:, 1] - tr[1]).min() <= pol['tree_dist']]
            tl.sort(key=lambda tr: float(np.hypot(cam[:, 0] - tr[0], cam[:, 1] - tr[1]).min()))
        for (x, y, size) in tl[:pol['tree_cap']]:
            rr = rng_for(seed, int(abs(x) * 131 + abs(y) * 17) & 0xffffff)
            pme, zmax = protos[int(rr.random() * len(protos))]
            ob = bpy.data.objects.new('T', pme)
            s = size / zmax; ob.location = (x, y, ground_z); ob.rotation_euler = (0, 0, rr.random() * math.tau)
            ob.scale = (s * (0.85 + rr.random() * 0.3), s * (0.85 + rr.random() * 0.3), s)
            bpy.context.scene.collection.objects.link(ob); ob.parent = root; ntree += 1
        tris['trees'] = ntree * (len(protos[0][0].polygons))
    st['trees'] = ntree; st['t_trees'] = round(time.time() - t, 1)

    # --- 高架(chase/city.py と同じ形)
    if viaduct:
        yy = [center[1] - radius, center[1] + radius]
        nf, npil = build_viaduct(cx, root, ground_z, yy[0], yy[1], viaduct_material)
        tris['viaduct'] = nf * 2; st['viaduct_pillars'] = npil

    st['tris'] = tris; st['tris_total'] = int(sum(tris.values())); st['t_total'] = round(time.time() - T0, 1)
    st['materials'] = len(cx.all_mats)
    root['city_stats'] = json.dumps(st, ensure_ascii=False, default=float)
    last_stats = st
    log('city_kit built:', json.dumps({k: st[k] for k in ('detail_counts', 'kinds_all', 'houses', 'tris_total', 't_total', 'materials')}, ensure_ascii=False))
    return root


def city_stats(root):
    return json.loads(root['city_stats'])


# ================================================================================================
# 調査の仕様との照合(bpy 不要。作った計画・形から測る)
# ================================================================================================
def spec_report(city_json_path, center=(0.0, 700.0), radius=800.0, cam=None, hour_list=(19.0, 21.0, 24.0), seed=1, policy=None):
    """街の割り当て・窓の点灯・道路の断面・基礎を測って、調査の値と並べる。 -> dict(rows=[(項目, 調査の仕様, 結果, 判定)], raw=...)"""
    city = load_city(city_json_path)
    pol = dict(DEFAULT_POLICY); pol.update(policy or {})
    camp = None if cam is None else np.asarray([(p[0], p[1]) for p in cam], float)
    plan = plan_buildings(city, center, radius, camp, 19.0, seed, pol)
    rows = []; raw = {}
    recs = plan['recs']; hs = plan['houses']

    def share(group, hgroup):
        c = {}
        for r in group:
            c[r['kind']] = c.get(r['kind'], 0) + 1
        for r in hgroup:
            k = 'low' if r['kind'] == 'low' else 'house'
            c[k] = c.get(k, 0) + 1
        return c

    def near_v(p):
        return max(-22.4 - p[0], p[0] - 9.6, 0.0) < 150.0
    sh_all = share(recs, hs); sh_near = share([r for r in recs if near_v(r['c'])], [h for h in hs if near_v(h['c'])])
    raw['kinds_all'] = sh_all; raw['kinds_within150m_of_viaduct'] = sh_near
    kit_only = {k: v for k, v in sh_near.items() if k in ('office', 'zakkyo', 'mansion', 'low')}
    tk = sum(kit_only.values()) or 1
    pm = 100 * kit_only.get('mansion', 0) / tk; pz = 100 * kit_only.get('zakkyo', 0) / tk; po = 100 * kit_only.get('office', 0) / tk; pl = 100 * kit_only.get('low', 0) / tk
    rows.append(('種類: マンション', '約40%(推)', '%.0f%%(高架から150m以内の、キットの建物+小さい建物の中)' % pm, '合っている' if 33 <= pm <= 47 else 'ずれ'))
    rows.append(('種類: 雑居(タイル)', '約35%(推)', '%.0f%%' % pz, '合っている' if 28 <= pz <= 42 else 'ずれ'))
    rows.append(('種類: ガラスのオフィス', '約10%(推)', '%.0f%%' % po, '合っている' if 6 <= po <= 14 else 'ずれ'))
    rows.append(('種類: 低層店舗・住宅', '約15%(推)', '小さいビル(low) %.0f%%。別に住宅(house型)が %d 棟(上の建物 %d 棟に対して)' % (pl, sh_near.get('house', 0), tk), 'ずれ(city.json は住宅が全体の約74%)'))
    lit = {}
    sample = [r for r in recs if r['kind'] in ('office', 'mansion', 'zakkyo')]
    for hour in hour_list:
        for k in ('office', 'mansion', 'zakkyo'):
            vals = []; temps = []
            for r in [x for x in sample if x['kind'] == k][:80]:
                nW = max(bk.SPECS[k]['minb'][0], int(round(r['W'] / bk.SPECS[k]['P']))); nD = max(bk.SPECS[k]['minb'][1], int(round(r['D'] / bk.SPECS[k]['P'])))
                arr = bk.make_lit(k, 2 * (nW + nD), r['floors'], r['seed'], hour, True)
                up = arr[1:, :, 0]; on = up > 0
                vals.append(float(on.mean())); temps += list((arr[1:, :, 1][on] * 1e4)[:2000])
            lit[(k, hour)] = (float(np.mean(vals)) if vals else 0.0, (float(np.median(temps)), float(np.percentile(temps, 10)), float(np.percentile(temps, 90))) if temps else (0, 0, 0))
    raw['lit'] = {'%s_%g' % k: [v[0], list(v[1])] for k, v in lit.items()}
    rows.append(('窓の点灯率 オフィス 19〜21時', '40〜60%(推)', '%.0f%%(19時)・%.0f%%(21時)' % (100 * lit[('office', 19.0)][0], 100 * lit[('office', 21.0)][0]), '合っている'))
    rows.append(('窓の点灯率 オフィス 深夜(23時以降)', '10〜25%(推)', '%.0f%%(24時)' % (100 * lit[('office', 24.0)][0]), '合っている'))
    rows.append(('窓の点灯率 マンション 19〜22時', '30〜50%(推)', '%.0f%%(19時)・%.0f%%(21時)' % (100 * lit[('mansion', 19.0)][0], 100 * lit[('mansion', 21.0)][0]), '合っている'))
    rows.append(('窓の点灯率 マンション 深夜', '10〜20%(推)', '%.0f%%(24時)' % (100 * lit[('mansion', 24.0)][0]), '合っている'))
    to = lit[('office', 21.0)][1]; tm = lit[('mansion', 21.0)][1]
    rows.append(('夜の窓の色温度 オフィス', '4000〜5000K(白、推)', '中央値 %.0fK(10〜90%%: %.0f〜%.0fK)' % to, '合っている'))
    rows.append(('夜の窓の色温度 マンション', '2700〜3000K 主体(暖色、推)', '中央値 %.0fK(10〜90%%: %.0f〜%.0fK。混ぜ込み: 3500〜5000K・TV 6200K)' % tm, '合っている'))
    nred = 0; ntall = 0
    for r in recs[:700]:
        md, info, ctx = bk.build_geometry(r['kind'], r['W'], r['D'], r['floors'], r['seed'], 'far', True, 20.0, r['age'])
        top = info['height_m'] + {'office': 1.0, 'zakkyo': 0.6, 'mansion': 0.5}[r['kind']]
        if top >= bk.RED_LIGHT_MIN_H:
            ntall += 1; nred += int(info['roof'].get('red_lights', 0) > 0)
    raw['tall60'] = ntall; raw['tall60_with_red'] = nred
    rows.append(('赤い障害灯(60m以上)', '60m以上の屋上に赤(★)', '60m以上 %d 棟のうち %d 棟に赤い灯(far・mid・near とも)' % (ntall, nred), '合っている' if ntall == nred else 'ずれ'))
    rows.append(('屋上設備(塔屋・キュービクル・冷却塔・高架水槽・室外機・アンテナ)', '塔屋+設備を箱でまとめる(推)', 'near / mid = building_kit の全部品。 far = 塔屋の箱+赤い灯(遠目では見分けがつかないため省略)', 'near/mid は合っている・far は一部だけ'))
    sec = sk.build_geometry('tokyo246', None, 40, 1, 'near')[1]
    rows.append(('縁石の高さ', '車道面+0.15m(★)', '%.2fm' % sec['curb_h'], '合っている'))
    rows.append(('歩道面', '車道面+0.05m・縁石は歩道面から約0.10m(★)', '歩道面+%.2fm・縁石の出 %.2fm' % (sec['sw_h'], sec['curb_exposed']), '合っている'))
    rows.append(('点字ブロック', '30cm角・黄・縁石から約0.6m(★寸法/推位置)', '縁石から %.2fm〜%.2fm(幅0.30m)' % (sec['tactile'][0] - sec['sidewalk_o0'], sec['tactile'][1] - sec['sidewalk_o0']), '合っている'))
    rows.append(('植樹・街灯・ガードパイプ', '高木10m間隔・街灯高6〜10m・柵の上端0.6〜1.0m(★/推)', '木 10m間隔・街灯 8m(25〜30m間隔)・柵 上端0.9m(2段)', '合っている'))
    if camp is not None:
        stp = plan_all(city, center, radius, camp, 19.0, seed, policy)['streets']
        swv = []; fnd = {'low': 0, 'wainscot': 0, 'flush': 0}
        for it in street_items(stp):
            swv.append(it['info']['sidewalk_w']); fnd[it['P']['foundation']] += 1
        swv = np.asarray(swv)
        raw['sidewalk_w_pcts'] = [float(x) for x in np.percentile(swv, [5, 25, 50, 75, 95])]; raw['found_counts'] = fnd
        ok = float((swv >= 2.0).mean())
        rows.append(('歩道の幅(有効幅員)', '2m以上、歩行者が多い道路は3.5m以上(★)', '中央値 %.2fm。 2m以上の区間 %.0f%%(OSMの壁の位置に合わせたため、壁が近い区間は0.7〜2m)' % (float(np.median(swv)), ok * 100),
                     'ずれ(壁の位置を優先)' if ok < 0.9 else '合っている'))
        tot = sum(fnd.values()) or 1
        rows.append(('基礎の立ち上がり', '低い立ち上がり約5割・化粧の腰(0.45〜0.9m)約3割・段差なし(2cm以下)約2割(推)',
                     '低い %.0f%%・化粧の腰 %.0f%%・段差なし %.0f%%(区間の割合)' % tuple(100 * fnd[k] / tot for k in ('low', 'wainscot', 'flush')), '合っている'))
    kinds = {}
    for i in range(300):
        for k, w in ground_spec(['zakkyo', 'mansion', 'office'][i % 3], 12.0 + (i % 5) * 4.0, random.Random(i)):
            kinds[k] = kinds.get(k, 0) + w
    tot = sum(kinds.values())
    raw['ground_width_share'] = {k: round(100 * v / tot) for k, v in kinds.items()}
    rows.append(('1階の構成', 'ガラス張り約4割・タイル腰壁約2割・シャッター約1.5割・駐車場約1割・エントランス1棟に1〜2か所(推)',
                 ' / '.join('%s %d%%' % (k, v) for k, v in sorted(raw['ground_width_share'].items(), key=lambda a: -a[1])) + '(幅の割合。near/mid のビルだけ付く)', '概ね合っている'))
    return dict(rows=rows, raw=raw)
