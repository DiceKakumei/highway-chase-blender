"""street_kit.py — 現実の道路の横断面(車道→縁石→歩道→植樹帯→境界→基礎)を作るキット(Blender Python)。他の動画から import して使える。
依存: bpy・mathutils(Blender内のみ)・numpy・標準ライブラリ。 build_geometry は bpy 無しで動く(寸法・面数の数値確認用)。

使い方
    import sys; sys.path.insert(0, '<このファイルのフォルダ>')
    import street_kit as sk
    st = sk.make_street_section('tokyo246', None, 120, seed=1, detail='near', night=True)      # 直線 120m(+X 方向)
    st = sk.make_street_section({'foundation': 'wainscot', 'sidewalk_w': 3.5}, [(0,0),(40,5),(80,25)], None, 2, 'mid')   # 曲線(Catmull-Rom)
    info = sk.street_info(st)    # info['wall_plane'] = 建物の壁面(外面)の道路中心からの距離、info['base_z'] = 1階の下端の高さ

座標: 道路の中心線 = +X 方向(曲線ならその接線)、中心線の左(+Y)側が建物(sides='L')。 sides='LR' で両側。 z=0 = 車道の路面。
    建物は正面(道路側)が -Y 向き(building_kit と同じ)。壁面は y = info['wall_plane'](直線のとき)。
    ビルのキットを置く: 建物の中心 y = wall_plane + 奥行/2。 1階は building_base.make_ground_floor(..., base_z=info['base_z']) で足す。
detail: 'near'(縁石の面取り・点字の突起・ガードパイプ・マンホール・グレーチング・信号・電柱・電線・木の細部) / 'mid'(面取り・点字は色帯・木は少ない塊) / 'far'(帯だけ + 街灯の灯り + 木は点)

仕様の根拠: リサーチ/2026-11-19_道路と建物の接地まわり_歩道・基礎・1階.md (★=出典あり、推=推定)。コード中の「★」「推」は同ファイルの区分。
"""
import math, json, time
import numpy as np

try:
    import bpy
except Exception:
    bpy = None

# ------------------------------------------------------------------------------------------------
# 仕様(調査の値)。単位 m
# ------------------------------------------------------------------------------------------------
BASE = dict(
    lanes=2, lane_w=3.25,            # 1車線3.0〜3.25m(推)。片側の車線数
    gutter=0.30,                     # 路肩・側溝 0.25〜0.5(推)
    curb_w=0.18, curb_h=0.15,        # 縁石 厚0.15〜0.2(推)・高さ標準15cm★
    bevel=0.015,                     # 縁石の角の面取り 0.01〜0.02(推)
    sw_h=0.05,                       # 歩道面 = 車道面+5cm(セミフラット)★
    sidewalk_w=3.0,                  # 歩道幅員 2m以上、多い所3.5m以上★(縁石の歩道側から民地境界まで)
    paving='block',                  # 'block' = 透水性インターロッキング(厚60mm・100x200)★ / 'asphalt'
    tactile_off=0.60, tactile_w=0.30,  # 点字ブロック: 縁石から約0.6m(推)、30cm角★
    trees=True, tree_pitch=10.0,     # 高木10m程度★
    tree_off=1.10, pit_w=0.90, tree_h=8.0, tree_kind='zelkova',   # 桝の幅: 植樹幅0.76★+縁0.07x2
    fence=True, fence_h=0.90, fence_off=0.12,   # ガードパイプ: 上端 0.6〜1.0m★(車両用)
    fence_color='dark',              # 景観色(ダークグレー)★ / 'galv' = 亜鉛めっき
    lamp_pitch=25.0, lamp_h=8.0, lamp_off=0.45, lamp_arm=1.8,   # 街灯 6〜10m・縁石から0.3〜0.5m内(推)
    pole_pitch=40.0, pole_h=12.0,    # 電柱 約12m(推)
    boundary_w=0.12, boundary_up=0.03,   # 民地境界の見切り・境界ブロック 厚0.1〜0.15、歩道面+0〜0.1(推)
    setback=0.0,                     # セットバックの空地 0〜2m(推)
    foundation='low',                # 'low'(GL+0.15〜0.3) / 'wainscot'(GL+0.45〜0.9) / 'flush'(歩道と同じ高さ・段差2cm以下)
    found_mat='conc',                # 'conc' | 'stone' | 'tile'  (wainscot で石・タイルを選ぶ)
    found_h=None,                    # None = low:0.20 / wainscot:0.70 / flush:0.02
    inu_w=0.45,                      # 犬走り 0.3〜0.6(推)
    flashing_h=0.03,                 # 壁下端の金属見切り 幅20〜40mm(推)
    found_gaps=[],                   # 入口の前などで基礎・犬走り・見切りを切る [(s0, s1), ...]
    driveways=[],                    # 車路の切り下げ [(s0, s1), ...] s = 中心線に沿った距離(平坦部の範囲)。すりつけ 0.75m★
    signal_at=None,                  # 信号柱を立てる s の一覧。None = 長さの 0.8 の位置に1本(near/mid)
    center_line='yellow',            # 中心線 'yellow' | 'white' | None
    night=True, hour=20.0, lights=True, lamp_power=2500.0,
    sides='L',                       # 'L' = 左(+Y)だけ / 'LR' = 両側
    smooth=True,
)
PRESETS = {
    'tokyo246': dict(lanes=3, sidewalk_w=3.5, foundation='low', fence=True, trees=True),                # 国道246(片側3車線)
    'shopping': dict(lanes=1, sidewalk_w=2.5, fence=False, trees=False, paving='block', foundation='wainscot', found_mat='tile', lamp_pitch=20.0),
    'narrow': dict(lanes=1, sidewalk_w=2.0, fence=False, trees=False, paving='asphalt', foundation='low', lamp_pitch=30.0, pole_pitch=30.0),
    'setback': dict(lanes=2, sidewalk_w=3.0, setback=1.5, foundation='flush', trees=True),
}
FOUND_ALIAS = {'A': 'low', 'B': 'wainscot', 'C': 'flush', 'low': 'low', 'wainscot': 'wainscot', 'flush': 'flush'}

# 役割ごとの単純材質(glb 用 / 手続き材質の平均色)。 c = リニア色、r = 粗さ、m = 金属度
SIMPLE = {
    'road': dict(c=(0.12, 0.12, 0.125), r=0.85), 'gutter': dict(c=(0.12, 0.12, 0.12), r=0.9), 'curb': dict(c=(0.36, 0.36, 0.35), r=0.9),
    'sidewalk': dict(c=(0.30, 0.29, 0.27), r=0.88), 'tactile': dict(c=(0.62, 0.48, 0.05), r=0.7), 'tactile_bar': dict(c=(0.62, 0.48, 0.05), r=0.7),
    'boundary': dict(c=(0.30, 0.30, 0.29), r=0.9), 'setback': dict(c=(0.27, 0.26, 0.25), r=0.88), 'inuhashiri': dict(c=(0.28, 0.28, 0.27), r=0.92),
    'found_conc': dict(c=(0.27, 0.27, 0.26), r=0.9), 'found_stone': dict(c=(0.12, 0.12, 0.125), r=0.35), 'found_tile': dict(c=(0.16, 0.10, 0.075), r=0.35),
    'flashing': dict(c=(0.6, 0.6, 0.6), r=0.35, m=1.0), 'soil': dict(c=(0.04, 0.03, 0.02), r=1.0), 'trunk': dict(c=(0.05, 0.04, 0.035), r=0.95),
    'leaf': dict(c=(0.035, 0.085, 0.02), r=0.8), 'fence_dark': dict(c=(0.10, 0.10, 0.11), r=0.5, m=0.7), 'fence_galv': dict(c=(0.55, 0.57, 0.58), r=0.5, m=1.0),
    'pole_paint': dict(c=(0.30, 0.31, 0.32), r=0.5, m=0.6), 'pole_conc': dict(c=(0.38, 0.38, 0.37), r=0.9), 'lamp_emit': dict(c=(1, 0.8, 0.5), r=0.5, e=(1.0, 0.75, 0.45), es=20.0),
    'sig_housing': dict(c=(0.02, 0.02, 0.02), r=0.5), 'sig_g': dict(c=(0.0, 0.3, 0.15), r=0.4, e=(0.1, 1.0, 0.5), es=8.0), 'sig_r': dict(c=(0.3, 0.02, 0.02), r=0.4),
    'sig_y': dict(c=(0.3, 0.2, 0.02), r=0.4), 'manhole': dict(c=(0.07, 0.05, 0.04), r=0.6, m=0.5), 'grating': dict(c=(0.05, 0.05, 0.05), r=0.5, m=0.8),
    'wire': dict(c=(0.01, 0.01, 0.01), r=0.6),
}
LEAF_COL = {'zelkova': (0.035, 0.085, 0.02), 'ginkgo': (0.20, 0.17, 0.02)}


def srgb2lin(c):
    return tuple((x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4) for x in c)


def params(profile=None, **kw):
    P = dict(BASE)
    if isinstance(profile, str):
        P.update(PRESETS[profile])
    elif isinstance(profile, dict):
        if 'preset' in profile:
            P.update(PRESETS[profile['preset']])
        P.update({k: v for k, v in profile.items() if k != 'preset'})
    P.update(kw)
    P['foundation'] = FOUND_ALIAS[P['foundation']]
    if P['found_h'] is None:
        P['found_h'] = {'low': 0.20, 'wainscot': 0.70, 'flush': 0.02}[P['foundation']]
    return P


# ------------------------------------------------------------------------------------------------
# メッシュのため込み(面ごとに頂点を持つ。bpy 不要)
# ------------------------------------------------------------------------------------------------
class MD:
    def __init__(self):
        self.V = []; self.F = []; self.UV = []; self.MI = []; self.roles = []; self.rid = {}

    def role(self, r):
        if r not in self.rid:
            self.rid[r] = len(self.roles); self.roles.append(r)
        return self.rid[r]

    def face(self, pts, role, uvs=None):
        base = len(self.V)
        for p in pts:
            self.V.append((float(p[0]), float(p[1]), float(p[2])))
        self.F.append(tuple(range(base, base + len(pts))))
        if uvs is None:
            a = np.asarray(pts, float); n = np.cross(a[1] - a[0], a[2] - a[0]); ax = int(np.argmax(np.abs(n)))
            i, j = [(1, 2), (0, 2), (0, 1)][ax]
            uvs = [(p[i], p[j]) for p in a]
        self.UV.extend((float(u), float(v)) for u, v in uvs)
        self.MI.append(self.role(role))

    def nfaces(self):
        return len(self.F)

    def bounds(self):
        a = np.asarray(self.V); return a.min(0), a.max(0)


class Fr:
    """右手系の局所座標(a=進行方向, b=左, z=上)。 pt(a, b, z) = 世界座標"""
    def __init__(self, o, t, n):
        self.o = np.asarray(o, float); self.t = np.asarray(t, float); self.n = np.asarray(n, float)

    def pt(self, a, b, z):
        return self.o + self.t * a + self.n * b + np.array([0, 0, z])


def box(md, fr, a0, a1, b0, b1, z0, z1, role, skip=()):
    """直方体(外向きの法線)。 skip = 作らない面('-z','+z','-a','+a','-b','+b')"""
    P = lambda a, b, z: fr.pt(a, b, z)
    faces = {
        '-z': [P(a0, b0, z0), P(a0, b1, z0), P(a1, b1, z0), P(a1, b0, z0)],
        '+z': [P(a0, b0, z1), P(a1, b0, z1), P(a1, b1, z1), P(a0, b1, z1)],
        '-a': [P(a0, b0, z0), P(a0, b0, z1), P(a0, b1, z1), P(a0, b1, z0)],
        '+a': [P(a1, b0, z0), P(a1, b1, z0), P(a1, b1, z1), P(a1, b0, z1)],
        '-b': [P(a0, b0, z0), P(a1, b0, z0), P(a1, b0, z1), P(a0, b0, z1)],
        '+b': [P(a0, b1, z0), P(a0, b1, z1), P(a1, b1, z1), P(a1, b1, z0)],
    }
    for k, pts in faces.items():
        if k not in skip:
            md.face(pts, role)


def ring_basis(axis):
    axis = np.asarray(axis, float); axis = axis / (np.linalg.norm(axis) + 1e-12)
    h = np.array([0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(h, axis); u /= np.linalg.norm(u); v = np.cross(axis, u)
    return u, v, axis


def tube(md, p0, p1, r0, r1, sides, role, caps=True, uvscale=1.0):
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    u, v, ax = ring_basis(p1 - p0); L = float(np.linalg.norm(p1 - p0))
    ang = [2 * math.pi * i / sides for i in range(sides)]
    R0 = [p0 + r0 * (math.cos(a) * u + math.sin(a) * v) for a in ang]
    R1 = [p1 + r1 * (math.cos(a) * u + math.sin(a) * v) for a in ang]
    for i in range(sides):
        j = (i + 1) % sides
        md.face([R0[i], R0[j], R1[j], R1[i]], role, [(i / sides * uvscale, 0), ((i + 1) / sides * uvscale, 0), ((i + 1) / sides * uvscale, L), (i / sides * uvscale, L)])
    if caps:
        md.face(R1, role); md.face(R0[::-1], role)


def disc(md, c, r, z, n, role, uv_r=True):
    pts = [(c[0] + r * math.cos(2 * math.pi * i / n), c[1] + r * math.sin(2 * math.pi * i / n), z) for i in range(n)]
    md.face(pts, role, [(r + (p[0] - c[0]), r + (p[1] - c[1])) for p in pts])


_ICO = None


def icosphere(sub):
    global _ICO
    t = (1 + 5 ** 0.5) / 2
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6),
         (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    V = [np.array(p, float) / np.linalg.norm(p) for p in V]
    for _ in range(sub):
        cache = {}; NF = []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = (V[a] + V[b]) / 2; V.append(m / np.linalg.norm(m)); cache[k] = len(V) - 1
            return cache[k]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            NF += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = NF
    V = np.array(V)
    fixed = []
    for f in F:   # 外向きに揃える
        a, b, c = V[f[0]], V[f[1]], V[f[2]]
        fixed.append(f if np.dot(np.cross(b - a, c - a), a + b + c) > 0 else (f[0], f[2], f[1]))
    return V, fixed


def blob(md, c, r, sub, role, rng, squash=1.0, bump=0.18):
    V, F = icosphere(sub)
    k = 1.0 + bump * (np.asarray([rng.random() for _ in range(len(V))]) - 0.5) * 2
    P = V * k[:, None] * r
    P[:, 2] *= squash
    P = P + np.asarray(c, float)
    for f in F:
        md.face([P[i] for i in f], role)


# ------------------------------------------------------------------------------------------------
# 中心線
# ------------------------------------------------------------------------------------------------
def _curve_points(centerline, smooth):
    if hasattr(centerline, 'type') and getattr(centerline, 'type', None) == 'CURVE' and bpy is not None:
        dg = bpy.context.evaluated_depsgraph_get(); ev = centerline.evaluated_get(dg); me = ev.to_mesh()
        pts = [tuple(centerline.matrix_world @ v.co) for v in me.vertices]; ev.to_mesh_clear(); return np.asarray(pts, float)
    pts = np.asarray([(p[0], p[1], p[2] if len(p) > 2 else 0.0) for p in centerline], float)
    if smooth and len(pts) >= 3:
        P = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]]); out = []
        for k in range(1, len(P) - 2):
            p0, p1, p2, p3 = P[k - 1], P[k], P[k + 1], P[k + 2]
            for t in np.linspace(0, 1, 24, endpoint=False):
                out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
        out.append(pts[-1]); pts = np.asarray(out)
    return pts


class Line:
    def __init__(self, pts):
        self.pts = np.asarray(pts, float)
        d = np.linalg.norm(np.diff(self.pts[:, :2], axis=0), axis=1)
        self.cum = np.concatenate([[0], np.cumsum(d)]); self.L = float(self.cum[-1])

    def pos(self, s):
        s = np.clip(s, 0, self.L)
        return np.array([np.interp(s, self.cum, self.pts[:, k]) for k in range(3)])

    def frame(self, s):
        e = 0.3
        a = self.pos(max(s - e, 0)); b = self.pos(min(s + e, self.L)); t = b - a; t[2] = 0
        t = t / (np.linalg.norm(t) + 1e-12); n = np.array([-t[1], t[0], 0.0])
        return Fr(self.pos(s), t, n)


# ------------------------------------------------------------------------------------------------
# ジオメトリ
# ------------------------------------------------------------------------------------------------
def tree_geometry(md, fr, b, z0, h, kind, role_t, role_l, sub, nblob, rng):
    """街路樹1本。 fr 上の (a=0, b) の位置。 トランク + 葉の塊(でこぼこの球)"""
    base = fr.pt(0, b, z0)
    th = h * 0.34
    tube(md, base, base + np.array([0, 0, th]), 0.17, 0.10, 8 if sub >= 2 else 6, role_t)
    crown_r = h * (0.30 if kind == 'zelkova' else 0.20)
    cz = z0 + th + (h - th) * 0.5
    ch = (h - th) * 0.5
    for k in range(nblob):
        if nblob == 1:
            c = base + np.array([0, 0, cz - z0]); rr = crown_r * 1.15; sq = ch / crown_r
        else:
            ang = 2 * math.pi * (k / max(nblob - 1, 1)) + rng.random() * 0.5
            rad = crown_r * (0.55 if k else 0.0) * (0.7 + 0.5 * rng.random())
            up = (rng.random() - 0.35) * ch * 0.9
            c = base + np.array([math.cos(ang) * rad, math.sin(ang) * rad, cz - z0 + up]); rr = crown_r * (0.62 + 0.25 * rng.random()); sq = 0.85
        blob(md, c, rr, sub, role_l, rng, squash=sq)


def build_geometry(profile=None, centerline=None, length=100.0, seed=0, detail='near', **kw):
    """-> (MD, info, lamps)。 bpy 不要。"""
    P = params(profile, **kw)
    rng = __import__('random').Random(int(seed) * 7919 + 13)
    pts = _curve_points(centerline, P['smooth']) if centerline is not None else np.array([[0, 0, 0], [float(length or 100.0), 0, 0]])
    ln = Line(pts); L = ln.L
    if centerline is not None and length:
        L = min(L, float(length))
    near, mid = detail == 'near', detail in ('near', 'mid')
    Wr = P['lanes'] * P['lane_w']; g = P['gutter']; cw = P['curb_w']; ch = P['curb_h']; swh = P['sw_h']
    o_c0 = Wr + g; sw0 = o_c0 + cw; o_b0 = sw0 + P['sidewalk_w']
    bev = P['bevel'] if mid else 0.0
    tac0 = sw0 + P['tactile_off']; tac1 = tac0 + P['tactile_w']
    pit0 = sw0 + P['tree_off']; pit1 = pit0 + P['pit_w']
    drives = [(float(a), float(b)) for a, b in P['driveways']]
    ramp = 0.75
    # 木・街灯などの s
    trees_s = []
    if P['trees']:
        s = 5.0
        while s < L - 2.0:
            if not any(a - 1.5 <= s <= b + 1.5 for a, b in drives):
                trees_s.append(s)
            s += P['tree_pitch']
    pit_half = 0.6 + 0.07
    # 駅(station)
    ds = {'near': 1.0, 'mid': 2.0, 'far': 5.0}[detail]
    if centerline is not None and len(pts) > 3:
        ds = min(ds, 1.0 if near else 2.0)
    S = set(np.arange(0, L, ds).tolist()) | {L, 0.9, max(L - 0.9, 0)}
    for a, b in drives:
        S |= {a - ramp, a, b, b + ramp}
    for a, b in P['found_gaps']:
        S |= {a, b}
    for s in trees_s:
        S |= {s - pit_half, s + pit_half}
    S = np.array(sorted(x for x in S if -1e-9 <= x <= L + 1e-9)); S = S[np.concatenate([[True], np.diff(S) > 1e-4])]
    FR = [ln.frame(float(s)) for s in S]
    md = MD(); info = {}

    def curb_top(s):
        h = ch
        for a, b in drives:
            if a <= s <= b:
                h = 0.02
            elif a - ramp < s < a:
                h = 0.02 + (ch - 0.02) * (a - s) / ramp
            elif b < s < b + ramp:
                h = 0.02 + (ch - 0.02) * (s - b) / ramp
        return h

    def sw_a(s):
        return min(swh, curb_top(s))

    def zsw(o, s):
        a = sw_a(s)
        return a + (swh - a) * min(max((o - sw0) / 0.9, 0.0), 1.0)

    def in_pit(sm):
        return any(abs(sm - s) < pit_half - 1e-6 for s in trees_s)

    def in_drive(sm):
        return any(a - ramp <= sm <= b + ramp for a, b in drives)

    fd = P['foundation']; fh = P['found_h']
    prot = {'low': 0.03, 'wainscot': 0.06, 'flush': 0.0}[fd]
    sb0 = o_b0 + P['boundary_w']; sb1 = sb0 + P['setback']
    o_w = sb1 + (P['inu_w'] if fd != 'flush' else 0.0)           # 基礎の前面
    wall_plane = o_w + prot
    gl = swh                                                      # 民地側の地盤面(歩道と同じ高さ)
    ftop = gl + fh
    base_z = ftop + (P['flashing_h'] if fd != 'flush' else 0.0)
    found_role = {'conc': 'found_conc', 'stone': 'found_stone', 'tile': 'found_tile'}[P['found_mat'] if fd == 'wainscot' else 'conc']
    sw_role = 'sidewalk'
    sides = [+1] if P['sides'] == 'L' else [-1, +1]

    for sign in sides:
        def PT(i, o, z):
            return FR[i].o + FR[i].n * (sign * o) + np.array([0, 0, z])

        def Q(i, oa, za, ob, zb, role, uvz=False):
            ev = lambda zf, k: zf(S[k]) if callable(zf) else zf
            z0a, z1a, z0b, z1b = ev(za, i), ev(za, i + 1), ev(zb, i), ev(zb, i + 1)
            if abs(oa - ob) < 1e-9 and abs(z0a - z0b) < 1e-6 and abs(z1a - z1b) < 1e-6:
                return
            pp = [PT(i, oa, z0a), PT(i + 1, oa, z1a), PT(i + 1, ob, z1b), PT(i, ob, z0b)]
            if abs(oa - ob) < 1e-9:
                uv = [(S[i], z0a), (S[i + 1], z1a), (S[i + 1], z1b), (S[i], z0b)]
            else:
                uv = [(S[i], abs(oa)), (S[i + 1], abs(oa)), (S[i + 1], abs(ob)), (S[i], abs(ob))]
            if sign < 0:
                pp = pp[::-1]; uv = uv[::-1]
            md.face(pp, role, uv)

        Z = lambda v: (lambda s: v)
        for i in range(len(S) - 1):
            sm = 0.5 * (S[i] + S[i + 1]); drv = in_drive(sm)
            # 1. 車道
            Q(i, 0.0, 0.0, Wr, 0.0, 'road')
            # 2. 路肩・側溝
            Q(i, Wr, 0.0, o_c0, 0.0, 'gutter')
            # 3. 縁石(面取り付き)
            top = lambda s: curb_top(s)
            bv = lambda s: min(bev, curb_top(s) * 0.45)
            Q(i, o_c0, 0.0, o_c0, lambda s: top(s) - bv(s), 'curb')
            if bev > 0:
                Q(i, o_c0, lambda s: top(s) - bv(s), o_c0 + bev, top, 'curb')
                Q(i, o_c0 + bev, top, sw0 - bev, top, 'curb')
                Q(i, sw0 - bev, top, sw0, lambda s: top(s) - bv(s), 'curb')
            else:
                Q(i, o_c0, top, sw0, top, 'curb')
            Q(i, sw0, lambda s: top(s) - (bv(s) if bev > 0 else 0), sw0, sw_a, 'curb')
            # 4. 歩道
            zs = lambda o: (lambda s: zsw(o, s))
            sw_pav = 'sidewalk'
            if in_pit(sm):
                Q(i, sw0, zs(sw0), tac0, zs(tac0), sw_pav); Q(i, tac0, zs(tac0), tac1, zs(tac1), 'tactile')
                Q(i, tac1, zs(tac1), pit0, zs(pit0), sw_pav)
                e = 0.07; zz = lambda s: zsw(pit0, s)
                Q(i, pit0, zs(pit0), pit0 + e, zs(pit0 + e), 'curb')
                Q(i, pit0 + e, zs(pit0 + e), pit0 + e, lambda s: zsw(pit0, s) - 0.03, 'curb')
                Q(i, pit0 + e, lambda s: zsw(pit0, s) - 0.03, pit1 - e, lambda s: zsw(pit0, s) - 0.03, 'soil')
                Q(i, pit1 - e, lambda s: zsw(pit0, s) - 0.03, pit1 - e, zs(pit1 - e), 'curb')
                Q(i, pit1 - e, zs(pit1 - e), pit1, zs(pit1), 'curb')
                Q(i, pit1, zs(pit1), o_b0, zs(o_b0), sw_pav)
            else:
                Q(i, sw0, zs(sw0), tac0, zs(tac0), sw_pav); Q(i, tac0, zs(tac0), tac1, zs(tac1), 'tactile')
                Q(i, tac1, zs(tac1), o_b0, zs(o_b0), sw_pav)
            # 点字ブロックの突起(線状4本・JIS T 9251: 線の中心間隔75mm、突起高5mm)
            if near and 0.9 <= sm <= L - 0.9:
                for k in range(4):
                    oc = tac0 + 0.0375 + 0.075 * k; w2 = 0.0085; zb = lambda o: (lambda s: zsw(o, s)); zt = lambda o: (lambda s: zsw(o, s) + 0.005)
                    Q(i, oc - w2, zb(oc - w2), oc - w2, zt(oc - w2), 'tactile_bar'); Q(i, oc - w2, zt(oc - w2), oc + w2, zt(oc + w2), 'tactile_bar')
                    Q(i, oc + w2, zt(oc + w2), oc + w2, zb(oc + w2), 'tactile_bar')
            # 6. 民地境界(見切りブロック)
            bt = lambda s: zsw(o_b0, s) + P['boundary_up']
            Q(i, o_b0, zs(o_b0), o_b0, bt, 'boundary'); Q(i, o_b0, bt, sb0, bt, 'boundary'); Q(i, sb0, bt, sb0, Z(gl), 'boundary')
            # セットバック(公開空地)
            if P['setback'] > 0:
                Q(i, sb0, Z(gl), sb1, Z(gl), 'setback')
            gap = any(a <= sm <= b for a, b in P['found_gaps'])
            if gap:
                Q(i, sb1, Z(gl), wall_plane, Z(gl), 'setback'); continue
            # 犬走り
            if fd != 'flush':
                Q(i, sb1, Z(gl), o_w, Z(gl + 0.012), 'inuhashiri')
            # 基礎の化粧立ち上がり
            zin = gl + (0.012 if fd != 'flush' else 0.0)
            if fd == 'flush':
                Q(i, o_w, Z(gl), o_w, Z(ftop), found_role)
                Q(i, o_w, Z(ftop), o_w + 0.0, Z(ftop), found_role)
            else:
                Q(i, o_w, Z(zin), o_w, Z(ftop), found_role)
                if prot > 0:
                    Q(i, o_w, Z(ftop), o_w + prot, Z(ftop), found_role)
                    Q(i, o_w + prot, Z(ftop), o_w + prot, Z(ftop), found_role)
                # 壁の下端の金属見切り(細い帯)
                fl = P['flashing_h']
                Q(i, wall_plane - 0.012, Z(ftop), wall_plane - 0.012, Z(ftop + fl), 'flashing')
                Q(i, wall_plane - 0.012, Z(ftop + fl), wall_plane, Z(ftop + fl), 'flashing')
                Q(i, wall_plane - 0.012, Z(ftop), wall_plane, Z(ftop), 'flashing')
        # 木の桝の端の壁
        for s in trees_s:
            for sx, nrm in ((s - pit_half, +1), (s + pit_half, -1)):
                fr = ln.frame(sx); za = zsw(pit0, sx); zb_ = za - 0.03
                p = [fr.pt(0, sign * pit0, za), fr.pt(0, sign * pit1, za), fr.pt(0, sign * pit1, zb_), fr.pt(0, sign * pit0, zb_)]
                if (nrm > 0) == (sign > 0):
                    p = p[::-1]
                md.face(p, 'curb')
        # --- 点状ブロック(警告)・点字の端(始端と終端の0.9m = 3枚)
        if near:
            for (sa, sbn) in ((0.0, 0.9), (L - 0.9, L)):
                for ia in range(3):
                    for jb in range(5):
                        for kb in range(5):
                            s_ = sa + 0.05 + ia * 0.3 + (jb * 0.055 + 0.0 if False else 0) + 0.02 + jb * 0.055
                            fr = ln.frame(min(s_, L)); ob = tac0 + 0.0 + 0.0375 + kb * 0.055
                            base = fr.pt(0, sign * ob, zsw(ob, s_)); cc = base
                            tube(md, cc, cc + np.array([0, 0, 0.005]), 0.0060, 0.0048, 6, 'tactile_bar', caps=True)
        # --- 小物(props)
        zfloor = swh
        # ガードパイプ(歩車道境界の柵)
        if P['fence'] and mid:
            fh_ = P['fence_h']; fo = sw0 + P['fence_off']; role_f = 'fence_' + ('dark' if P['fence_color'] == 'dark' else 'galv')
            s = 0.0; posts = []
            while s <= L + 1e-6:
                posts.append(min(s, L)); s += 2.0
            for a_ in posts:
                if in_drive(a_) or any(abs(a_ - t) < 0.3 for t in trees_s):
                    pass
                if in_drive(a_):
                    continue
                fr = ln.frame(a_); base = fr.pt(0, sign * fo, 0); base[2] = zsw(fo, a_)
                tube(md, base, base + np.array([0, 0, fh_]), 0.04, 0.04, 8 if near else 4, role_f, caps=near)
            for k in range(len(posts) - 1):
                a_, b_ = posts[k], posts[k + 1]
                if in_drive(a_) or in_drive(b_):
                    continue
                f1 = ln.frame(a_); f2 = ln.frame(b_)
                for hz in (fh_ - 0.03, fh_ * 0.5):
                    p1 = f1.pt(0, sign * fo, zsw(fo, a_) + hz); p2 = f2.pt(0, sign * fo, zsw(fo, b_) + hz)
                    tube(md, p1, p2, 0.03, 0.03, 8 if near else 4, role_f, caps=False)
        # 街灯(柱 + アーム + 灯具)
        lamps = []
        s = 8.0
        while s < L - 1.0:
            fr = ln.frame(s); pb = fr.pt(0, sign * (sw0 + P['lamp_off']), swh); H = P['lamp_h']
            toward = -fr.n * sign   # 車道側
            if detail == 'far':
                tube(md, pb, pb + np.array([0, 0, H]), 0.07, 0.05, 4, 'pole_paint', caps=False)
                head = pb + np.array([0, 0, H]) + toward * P['lamp_arm']
                box(md, Fr(head, fr.t, fr.n), -0.5, 0.5, -0.3, 0.3, -0.15, 0.1, 'lamp_emit')
            else:
                tube(md, pb, pb + np.array([0, 0, H]), 0.09, 0.055, 10 if near else 6, 'pole_paint')
                p1 = pb + np.array([0, 0, H]); p2 = p1 + toward * P['lamp_arm'] + np.array([0, 0, 0.45]); p3 = p2 + toward * 0.9
                tube(md, p1 - np.array([0, 0, 0.1]), p2, 0.045, 0.035, 8 if near else 4, 'pole_paint')
                tube(md, p2, p3, 0.035, 0.03, 8 if near else 4, 'pole_paint')
                hc = p3 + toward * 0.35 - np.array([0, 0, 0.12]); fh_ = Fr(hc, fr.t, fr.n)
                box(md, fh_, -0.55, 0.55, -0.16, 0.16, -0.07, 0.07, 'pole_paint', skip=('-z',))
                box(md, fh_, -0.5, 0.5, -0.13, 0.13, -0.075, -0.07, 'lamp_emit', skip=('+z',))
                lamps.append(tuple(hc - np.array([0, 0, 0.1])))
            s += P['lamp_pitch']
        # 電柱(民地境界の少し内側)と電線
        if mid:
            poles = []; s = 17.0
            while s < L - 1.0:
                fr = ln.frame(s); pb = fr.pt(0, sign * (o_b0 - 0.30), zsw(o_b0, s)); H = P['pole_h']
                tube(md, pb, pb + np.array([0, 0, H]), 0.17, 0.12, 10 if near else 6, 'pole_conc')
                fr2 = Fr(pb, fr.t, fr.n)
                for zc, wd in ((H - 0.5, 1.2), (H - 1.4, 1.0)):
                    box(md, fr2, -0.04, 0.04, -wd, wd, zc, zc + 0.12, 'pole_paint')
                if near and (int(s / P['pole_pitch']) % 2 == 0):
                    b_ = sorted((-0.70 * sign, -0.25 * sign)); box(md, fr2, -0.25, 0.25, b_[0], b_[1], H - 3.3, H - 2.4, 'pole_paint')
                poles.append(s); s += P['pole_pitch']
            if near:
                for k in range(len(poles) - 1):
                    f1 = ln.frame(poles[k]); f2 = ln.frame(poles[k + 1])
                    for zc, bb in ((P['pole_h'] - 0.4, -1.0), (P['pole_h'] - 0.4, 0.0), (P['pole_h'] - 0.4, 1.0), (P['pole_h'] - 1.3, 0.8)):
                        pa = f1.pt(0, sign * (o_b0 - 0.30) + bb, zsw(o_b0, 0) + zc); pb2 = f2.pt(0, sign * (o_b0 - 0.30) + bb, zsw(o_b0, 0) + zc)
                        prev = pa
                        for q in range(1, 7):
                            tq = q / 6; cur = pa + (pb2 - pa) * tq; cur = cur - np.array([0, 0, 0.5 * math.sin(math.pi * tq)])
                            tube(md, prev, cur, 0.008, 0.008, 4, 'wire', caps=False); prev = cur
        # 信号柱(近景・中景)
        sig = P['signal_at'] if P['signal_at'] is not None else [L * 0.8]
        if mid and sign > 0:
            for s in sig:
                fr = ln.frame(s); pb = fr.pt(0, sign * (sw0 + 0.35), swh)
                tube(md, pb, pb + np.array([0, 0, 5.2]), 0.10, 0.07, 10 if near else 6, 'pole_paint')
                toward = -fr.n * sign; p1 = pb + np.array([0, 0, 5.0]); p2 = p1 + toward * 4.5 + np.array([0, 0, 0.3])
                tube(md, p1, p2, 0.06, 0.04, 8 if near else 4, 'pole_paint')
                hc = p1 + toward * 3.6 + np.array([0, 0, 0.1]); fs = Fr(hc, fr.t, fr.n)
                box(md, fs, -0.45, 0.45, -0.12, 0.12, -0.2, 0.2, 'sig_housing')
                for k, rr in enumerate(('sig_r', 'sig_y', 'sig_g')):
                    cx = -0.28 + 0.28 * k
                    box(md, fs, cx - 0.1, cx + 0.1, -0.13, -0.12, -0.1, 0.1, rr)
        # マンホール・グレーチング(近景)
        if near:
            s = 12.0
            while s < L - 2.0:
                fr = ln.frame(s); c = fr.pt(0, sign * (Wr - 1.3), 0.0)
                disc(md, (c[0], c[1]), 0.30, 0.004, 20, 'manhole'); disc(md, (c[0], c[1]), 0.23, 0.0045, 20, 'grating')
                s += 40.0
            s = 6.0
            while s < L - 2.0:
                fr = ln.frame(s); fg = Fr(fr.pt(0, sign * (Wr + g * 0.5), 0.0), fr.t, fr.n)
                box(md, fg, -0.30, 0.30, -0.13, 0.13, 0.0, 0.006, 'grating', skip=('-z',))
                s += 20.0
        # 街路樹
        for s in trees_s:
            fr = ln.frame(s); bc = sign * 0.5 * (pit0 + pit1); zb0 = swh - 0.03
            sub = {'near': 2, 'mid': 1, 'far': 1}[detail]; nb_ = {'near': 7, 'mid': 4, 'far': 1}[detail]
            hh = P['tree_h'] * (0.9 + 0.2 * rng.random())
            tree_geometry(md, fr, bc, zb0, hh, P['tree_kind'], 'trunk', 'leaf', sub, nb_, rng) if detail != 'far' else \
                blob(md, fr.pt(0, bc, zb0 + hh * 0.62), hh * 0.30, 1, 'leaf', rng, squash=0.85, bump=0.1)

    info.update(dict(
        profile=P['foundation'], half_road=Wr, curb_o=o_c0, sidewalk_o0=sw0, boundary_o=o_b0, wall_plane=wall_plane, found_front=o_w, base_z=base_z, found_top=ftop, gl=gl,
        curb_h=ch, curb_exposed=ch - swh, sw_h=swh, sidewalk_w=P['sidewalk_w'], tactile=(tac0, tac1), pit=(pit0, pit1), length=L, setback=P['setback'], found_prot=prot,
        found_role=found_role, trees=len(trees_s), lamps=len(lamps), faces=md.nfaces(), detail=detail, sides=P['sides'], seed=seed, driveways=drives, lane_w=P['lane_w'],
        lanes=P['lanes']))
    return md, info, lamps, P


# ------------------------------------------------------------------------------------------------
# 材質(手続き的なシェーダ。 bpy が要る)
# ------------------------------------------------------------------------------------------------
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

    def voronoi(self, vec, scale):
        n = self.node('ShaderNodeTexVoronoi'); n.voronoi_dimensions = '3D'; n.feature = 'F1'
        n.inputs['Scale'].default_value = scale; self.nt.links.new(vec, n.inputs['Vector'])
        return n.outputs['Distance'], n.outputs['Color']

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

    def coords(self):
        tc = self.node('ShaderNodeTexCoord')
        x, y, _ = self.sep(tc.outputs['UV'])
        ox, oy, oz = self.sep(tc.outputs['Object'])
        return x, y, oz

    def bump(self, h, strength=0.3, dist=0.02):
        n = self.node('ShaderNodeBump'); n.inputs['Strength'].default_value = strength; n.inputs['Distance'].default_value = dist
        self.nt.links.new(h, n.inputs['Height'])
        return n.outputs['Normal']


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


def finish(mat, nt, out, shader, role=None):
    nt.links.new(shader, out.inputs['Surface'])
    if role in SIMPLE:
        mat['kit_simple'] = json.dumps(SIMPLE[role])
    return mat


def flat_material(name, role, night=True, leaf=None):
    s = dict(SIMPLE[role])
    mat, nt, out = new_material(name)
    kw = dict(roughness=s.get('r', 0.5), metallic=s.get('m', 0.0))
    b = principled(nt, **{'Base Color': (*(leaf or s['c']), 1.0), 'Roughness': s.get('r', 0.5), 'Metallic': s.get('m', 0.0)})
    if 'e' in s:
        b.inputs['Emission Color'].default_value = (*s['e'], 1.0); b.inputs['Emission Strength'].default_value = s.get('es', 1.0) if night else 0.0
    finish(mat, nt, out, b.outputs[0], role)
    return mat, b


def build_material(role, P, info, name):
    """役割ごとの手続き材質。 u,v = (中心線に沿った距離, 中心線からの距離)[m]、立面は (s, 高さ)。 z = オブジェクト座標の高さ。"""
    if role in ('soil',) and False:
        pass
    simple_only = ('flashing', 'trunk', 'leaf', 'fence_dark', 'fence_galv', 'pole_paint', 'pole_conc', 'lamp_emit', 'sig_housing', 'sig_g', 'sig_r', 'sig_y', 'manhole', 'grating', 'wire', 'soil')
    if role in simple_only:
        mat, b = flat_material(name, role, P['night'], LEAF_COL.get(P['tree_kind']) if role == 'leaf' else None)
        if role in ('leaf', 'trunk', 'soil', 'pole_conc'):
            nt = mat.node_tree; nb = NB(nt)
            u, v, z = nb.coords(); base = SIMPLE[role]['c'] if role != 'leaf' else LEAF_COL.get(P['tree_kind'])
            tc = nb.node('ShaderNodeTexCoord'); ox, oy, oz = nb.sep(tc.outputs['Object'])
            n1 = nb.noise(tc.outputs['Object'], 6.0 if role == 'leaf' else 12.0)
            k = nb.m('ADD', 0.6, nb.m('MULTIPLY', n1, 0.8))
            col = nb.vscale(nb.rgb(base), k)
            nt.links.new(col, b.inputs['Base Color'])
        return mat
    mat, nt, out = new_material(name); nb = NB(nt)
    u, v, z = nb.coords()
    S = SIMPLE[role]; base = S['c']; rough = S['r']
    uvw = nb.vec(u, v, 0.0)
    n_lo = nb.noise(uvw, 0.25, 3.0); n_mid = nb.noise(uvw, 3.0, 4.0); n_hi = nb.noise(uvw, 60.0, 2.0); n_fine = nb.noise(uvw, 220.0, 1.0)
    var = nb.m('MULTIPLY', nb.m('ADD', 0.80, nb.m('MULTIPLY', n_lo, 0.4)), nb.m('ADD', 0.90, nb.m('MULTIPLY', n_mid, 0.2)))
    col = nb.vscale(nb.rgb(base), nb.m('MULTIPLY', var, nb.m('ADD', 0.86, nb.m('MULTIPLY', n_fine, 0.28))))
    sw0 = info['sidewalk_o0']; bnd = info['boundary_o']; Wr = info['half_road']; lw = info['lane_w']
    rough_v = rough
    bump_h = n_hi; bump_s = 0.15
    if role == 'road':
        # 轍(車線の中の2本の帯)
        wp = nb.m('MODULO', v, lw)
        d1 = nb.m('ABSOLUTE', nb.m('SUBTRACT', wp, lw * 0.5 - 0.85)); d2 = nb.m('ABSOLUTE', nb.m('SUBTRACT', wp, lw * 0.5 + 0.85))
        track = nb.m('SUBTRACT', 1.0, nb.sstep(nb.m('MINIMUM', d1, d2), 0.0, 0.38))
        col = nb.vscale(col, nb.m('SUBTRACT', 1.0, nb.m('MULTIPLY', track, 0.20)))
        # 油じみ
        oil = nb.sstep(nb.noise(nb.vec(u, v, 3.7), 0.12, 2.0), 0.56, 0.64)
        oil = nb.m('MULTIPLY', oil, track)
        col = nb.vscale(col, nb.m('SUBTRACT', 1.0, nb.m('MULTIPLY', oil, 0.45)))
        # 路肩の砂ぼこり
        dust = nb.sstep(v, Wr - 0.9, Wr)
        col = nb.vmix(col, nb.rgb((0.22, 0.20, 0.17)), nb.m('MULTIPLY', dust, nb.m('MULTIPLY', n_mid, 0.5)))
        # 白線(破線: 6m線・9m空き(推)、外側線: 実線)、中心線(黄)
        wear = nb.sstep(nb.m('ADD', n_hi, nb.m('MULTIPLY', n_mid, 0.5)), 0.55, 0.85)
        dl = nb.m('MINIMUM', wp, nb.m('SUBTRACT', lw, wp))
        lane_line = nb.m('SUBTRACT', 1.0, nb.sstep(dl, 0.06, 0.08))
        inrange = nb.m('MULTIPLY', nb.m('GREATER_THAN', v, 0.5), nb.m('LESS_THAN', v, Wr - 0.5))
        dash = nb.m('LESS_THAN', nb.m('MODULO', u, 15.0), 6.0)
        m_dash = nb.m('MULTIPLY', nb.m('MULTIPLY', lane_line, inrange), dash)
        edge = nb.m('SUBTRACT', 1.0, nb.sstep(nb.m('ABSOLUTE', nb.m('SUBTRACT', v, Wr - 0.25)), 0.065, 0.085))
        white = nb.m('MULTIPLY', nb.m('MAXIMUM', m_dash, edge), nb.m('SUBTRACT', 1.0, nb.m('MULTIPLY', wear, 0.55)))
        col = nb.vmix(col, nb.rgb((0.42, 0.42, 0.40)), white)
        if P['center_line']:
            cl = nb.m('SUBTRACT', 1.0, nb.sstep(v, 0.07, 0.10))
            cc = (0.42, 0.30, 0.02) if P['center_line'] == 'yellow' else (0.42, 0.42, 0.40)
            col = nb.vmix(col, nb.rgb(cc), nb.m('MULTIPLY', cl, nb.m('SUBTRACT', 1.0, nb.m('MULTIPLY', wear, 0.5))))
        bump_s = 0.35
    elif role == 'gutter':
        slit = nb.sstep(nb.m('ABSOLUTE', nb.m('SINE', nb.m('MULTIPLY', u, 25.0))), 0.0, 0.5)
        col = nb.vscale(col, nb.m('ADD', 0.55, nb.m('MULTIPLY', slit, 0.45)))
        col = nb.vmix(col, nb.rgb((0.13, 0.12, 0.10)), nb.m('MULTIPLY', nb.sstep(n_mid, 0.4, 0.8), 0.6))
    elif role in ('curb', 'boundary', 'inuhashiri', 'setback', 'found_conc'):
        # 黒ずみ(車道側・下端)・砂・欠け
        low = nb.sstep(z, 0.12 if role != 'found_conc' else 0.30, 0.0)
        col = nb.vscale(col, nb.m('SUBTRACT', 1.0, nb.m('MULTIPLY', low, nb.m('ADD', 0.25, nb.m('MULTIPLY', n_mid, 0.30)))))
        if role == 'curb':
            chip = nb.sstep(nb.noise(uvw, 14.0, 2.0), 0.66, 0.74)
            col = nb.vscale(col, nb.m('SUBTRACT', 1.0, nb.m('MULTIPLY', chip, 0.35)))
        if role == 'found_conc':   # 泥はね: 下端 0.3m は暗い帯(推: 明度を20〜35%下げ、上端をぼかす)
            splash = nb.m('MULTIPLY', nb.sstep(z, 0.40, 0.05), nb.m('ADD', 0.7, nb.m('MULTIPLY', n_mid, 0.6)))
            col = nb.vmix(col, nb.rgb((0.07, 0.06, 0.05)), nb.m('MULTIPLY', splash, 0.55))
    elif role == 'found_stone':
        vd, vc = nb.voronoi(nb.vec(u, v, 0.0), 90.0)
        cr, cg, cb = nb.sep(vc)
        col = nb.vscale(col, nb.m('ADD', 0.7, nb.m('MULTIPLY', cr, 0.7)))
        splash = nb.m('MULTIPLY', nb.sstep(z, 0.40, 0.05), nb.m('ADD', 0.7, nb.m('MULTIPLY', n_mid, 0.6)))
        col = nb.vmix(col, nb.rgb((0.05, 0.045, 0.04)), nb.m('MULTIPLY', splash, 0.5))
    elif role == 'found_tile':
        br = nb.node('ShaderNodeTexBrick'); br.offset = 0.5
        nt.links.new(nb.vec(nb.m('MULTIPLY', u, 1.0 / 0.227), nb.m('MULTIPLY', z, 1.0 / 0.06), 0.0), br.inputs['Vector'])
        br.inputs['Scale'].default_value = 1.0; br.inputs['Mortar Size'].default_value = 0.03; br.inputs['Brick Width'].default_value = 1.0; br.inputs['Row Height'].default_value = 1.0
        br.inputs['Color1'].default_value = (0.19, 0.115, 0.08, 1); br.inputs['Color2'].default_value = (0.15, 0.09, 0.065, 1); br.inputs['Mortar'].default_value = (0.2, 0.19, 0.17, 1)
        col = nb.vscale(br.outputs['Color'], nb.m('ADD', 0.85, nb.m('MULTIPLY', n_mid, 0.3)))
        splash = nb.m('MULTIPLY', nb.sstep(z, 0.45, 0.05), nb.m('ADD', 0.7, nb.m('MULTIPLY', n_mid, 0.6)))
        col = nb.vmix(col, nb.rgb((0.04, 0.035, 0.03)), nb.m('MULTIPLY', splash, 0.5))
        bump_h = br.outputs['Fac']; bump_s = 0.4
    elif role in ('sidewalk',):
        paver = P['paving'] == 'block'
        if paver:
            br = nb.node('ShaderNodeTexBrick'); br.offset = 0.5
            nt.links.new(nb.vec(nb.m('MULTIPLY', u, 5.0), nb.m('MULTIPLY', v, 10.0), 0.0), br.inputs['Vector'])   # 200 x 100 mm のブロック
            br.inputs['Scale'].default_value = 1.0; br.inputs['Mortar Size'].default_value = 0.025; br.inputs['Brick Width'].default_value = 1.0; br.inputs['Row Height'].default_value = 1.0
            br.inputs['Color1'].default_value = (0.36, 0.35, 0.33, 1); br.inputs['Color2'].default_value = (0.27, 0.26, 0.25, 1); br.inputs['Mortar'].default_value = (0.08, 0.075, 0.07, 1)
            col = nb.vscale(br.outputs['Color'], nb.m('ADD', 0.82, nb.m('MULTIPLY', n_lo, 0.35)))
            bump_h = br.outputs['Fac']; bump_s = 0.5
        else:
            col = nb.vscale(nb.rgb((0.10, 0.10, 0.105)), nb.m('ADD', 0.7, nb.m('MULTIPLY', n_mid, 0.6)))
        # 人通り: 歩行帯(中央)が明るく、端が暗い
        mid_ = 0.5 * (info['tactile'][1] + bnd)
        walk = nb.m('SUBTRACT', 1.0, nb.sstep(nb.m('ABSOLUTE', nb.m('SUBTRACT', v, mid_)), 0.2, 1.4))
        col = nb.vscale(col, nb.m('ADD', 0.88, nb.m('MULTIPLY', walk, 0.22)))
        # ガム(黒い点。人通りが多い所ほど密)
        vd, vc = nb.voronoi(nb.vec(u, v, 0.0), 5.0); cr, cg, cb = nb.sep(vc)
        gum = nb.m('MULTIPLY', nb.m('MULTIPLY', nb.m('GREATER_THAN', cr, 0.6), nb.m('SUBTRACT', 1.0, nb.sstep(vd, 0.03, 0.065))), nb.m('ADD', 0.35, walk))
        col = nb.vmix(col, nb.rgb((0.025, 0.024, 0.022)), nb.m('MULTIPLY', gum, 0.9))
        # 油染み
        oil = nb.sstep(nb.noise(nb.vec(u, v, 9.1), 0.4, 2.0), 0.60, 0.68)
        col = nb.vscale(col, nb.m('SUBTRACT', 1.0, nb.m('MULTIPLY', oil, 0.45)))
        # 縁石際の砂・落ち葉、水の跳ね跡(車道側が暗い)
        sand = nb.sstep(v, sw0 + 0.5, sw0)
        col = nb.vmix(col, nb.rgb((0.30, 0.26, 0.20)), nb.m('MULTIPLY', sand, nb.m('MULTIPLY', nb.sstep(n_mid, 0.3, 0.7), 0.6)))
        vd2, vc2 = nb.voronoi(nb.vec(u, v, 3.0), 3.2); r2, g2, b2 = nb.sep(vc2)
        leaf = nb.m('MULTIPLY', nb.m('MULTIPLY', nb.m('GREATER_THAN', r2, 0.55), nb.m('SUBTRACT', 1.0, nb.sstep(vd2, 0.10, 0.22))), nb.m('ADD', nb.sstep(v, sw0 + 0.8, sw0), nb.sstep(v, bnd - 0.8, bnd)))
        col = nb.vmix(col, nb.rgb((0.20, 0.12, 0.03)), nb.m('MULTIPLY', leaf, 0.75))
        rough = 0.85
    elif role in ('tactile', 'tactile_bar'):
        wear = nb.sstep(nb.noise(uvw, 8.0, 2.0), 0.4, 0.8)
        col = nb.vscale(nb.rgb((0.62, 0.46, 0.04)), nb.m('SUBTRACT', 1.15, nb.m('MULTIPLY', wear, 0.35)))
        joint = nb.m('ADD', nb.m('SUBTRACT', 1.0, nb.sstep(nb.m('MINIMUM', nb.m('MODULO', u, 0.3), nb.m('SUBTRACT', 0.3, nb.m('MODULO', u, 0.3))), 0.002, 0.005)), 0.0)
        col = nb.vmix(col, nb.rgb((0.05, 0.04, 0.02)), nb.m('MULTIPLY', joint, 0.7))
        # 踏まれて磨かれ、中心が褪せる
        col = nb.vmix(col, nb.rgb((0.46, 0.40, 0.2)), nb.m('MULTIPLY', nb.sstep(n_mid, 0.5, 0.8), 0.3))
    b = principled(nt, **{'Roughness': rough_v if role != 'sidewalk' else rough})
    if role in ('found_stone', 'found_tile'):
        b.inputs['Roughness'].default_value = 0.35
    nt.links.new(col, b.inputs['Base Color'])
    nt.links.new(nb.bump(bump_h, bump_s), b.inputs['Normal'])
    if S.get('m'):
        b.inputs['Metallic'].default_value = S['m']
    return finish(mat, nt, out, b.outputs[0], role)


# ------------------------------------------------------------------------------------------------
# オブジェクト化
# ------------------------------------------------------------------------------------------------
def make_street_section(profile=None, centerline=None, length_or_curve=100.0, seed=0, detail='near', name=None, collection=None, **kw):
    """道路の断面を中心線に沿って押し出す。 -> 親Empty(子 = 道路メッシュ + 街灯のスポットライト)。
    profile: プリセット名('tokyo246' | 'shopping' | 'narrow' | 'setback')か、BASE の上書き辞書。 centerline: None(+X 方向の直線)か点の列か Curve オブジェクト。
    length_or_curve: 長さ(m)。 kw で個別の値を上書き可(例 foundation='wainscot', driveways=[(30,36)])。"""
    t0 = time.time()
    length = length_or_curve if isinstance(length_or_curve, (int, float)) else None
    md, info, lamps, P = build_geometry(profile, centerline, length or 100.0, seed, detail, **kw)
    nm = name or 'street_%s_s%d_%s' % (info['profile'], int(seed), detail)
    mats = [build_material(role, P, info, '%s_%s' % (nm, role)) for role in md.roles]
    me = bpy.data.meshes.new(nm + '_mesh')
    me.from_pydata(md.V, [], md.F)
    uvl = me.uv_layers.new(name='UVMap')
    uvl.data.foreach_set('uv', np.asarray(md.UV, np.float32).ravel())
    me.polygons.foreach_set('material_index', np.asarray(md.MI, np.int32))
    sm = [1 if md.roles[i] in ('leaf',) else 0 for i in md.MI]      # 木の葉の塊だけなめらかな陰影
    try:
        me.polygons.foreach_set('use_smooth', sm)
    except Exception:
        pass
    for m in mats:
        me.materials.append(m)
    me.update()
    body = bpy.data.objects.new(nm + '_body', me)
    root = bpy.data.objects.new(nm, None); root.empty_display_size = 2.0
    coll = collection or bpy.context.scene.collection
    coll.objects.link(root); coll.objects.link(body); body.parent = root
    if P['lights'] and P['night'] and lamps:
        for i, p in enumerate(lamps):
            ld = bpy.data.lights.new('%s_lamp%d' % (nm, i), 'SPOT'); ld.energy = P['lamp_power']; ld.spot_size = math.radians(130); ld.spot_blend = 0.8
            ld.shadow_soft_size = 0.12; ld.color = (1.0, 0.80, 0.58)
            lo = bpy.data.objects.new(ld.name, ld); lo.location = p; coll.objects.link(lo); lo.parent = root
    info['gen_seconds'] = round(time.time() - t0, 3); info['materials'] = len(mats)
    root['kit_info'] = json.dumps(info, ensure_ascii=False, default=float)
    return root


def street_info(root):
    return json.loads(root['kit_info'])


def simplify_materials(root):
    """書き出し(glTF)用に、手続き材質を単色の Principled に置き換える。"""
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
                nm = bpy.data.materials.new('S_' + m.name)
                try:
                    nm.use_nodes = True
                except Exception:
                    pass
                b = next(n for n in nm.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
                b.inputs['Base Color'].default_value = (*s['c'], 1.0)
                b.inputs['Roughness'].default_value = s.get('r', 0.5); b.inputs['Metallic'].default_value = s.get('m', 0.0)
                if 'e' in s:
                    b.inputs['Emission Color'].default_value = (*s['e'], 1.0); b.inputs['Emission Strength'].default_value = s.get('es', 1.0)
                cache[key] = nm
            slot.material = cache[key]


def export_glb(root, path):
    for o in bpy.data.objects:
        o.select_set(False)
    for o in [root] + list(root.children_recursive):
        o.select_set(True)
    bpy.context.view_layer.objects.active = root
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=True, export_apply=False)
