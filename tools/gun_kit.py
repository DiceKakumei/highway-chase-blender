"""動く拳銃キット(使い回し用)。 Meshy の1枚メッシュ gun.glb を部品に切り分け、発砲サイクルの動きと薬莢を時刻の関数で出す。
座標(銃のローカル): 原点=グリップ中心、+Y=銃口、+Z=上、+X=右(排莢口の側)。 単位 m。 chase/rider.py の import_gun と同じ。
構成:
  (A) 切り分け(numpy のみ。bpy なし): load_gun_glb / build_parts  -> {部品名: Mesh}
  (B) 動き(純関数): gun_pose(t_since_fire, fired_count, total_rounds, trigger, stretch)  -> {部品名: (dx,dy,dz,rx)}
  (C) 薬莢(純関数): casing_launch / casing_state / casing_mesh
  (D) Blender に適用(bpy が要る): build_blend(...) / apply_pose(...) / export_glb(...)
調査(出典と★/推): リサーチ/2026-11-19_拳銃の機構と発砲サイクル_動く部品.md
"""
import math, os, sys, json, struct
import numpy as np

# ============================================================ 寸法・時刻(★=出典あり、推=推定。調査ファイルの番号は 1-x)
GUN_SCALE_NOTE = 'gun.glb は全長約0.2m(実銃 G17 の約202mm と同程度)'
Z_SLIDE = 0.0579            # スライドとフレームの境(メッシュ上の合わせ目 z=0.0578〜0.0580)
Y_SLIDE_FRONT = 0.1620      # スライド前面(メッシュの小さな銃口の突起は、ここで切って手続きのバレルに替える)
Y_MUZZLE = 0.1655           # 銃口の先(メッシュ値)
BARREL_Z = 0.0665           # バレル中心の高さ(メッシュの銃口から測定)
BARREL_R = 0.0066           # バレル外径 約13mm(推。9mm拳銃のバレル外径)
BARREL_Y0 = 0.0200          # バレル後端(薬室の後ろ)。ボア長 114mm(★ G17)+薬室のかたまり約31mm=全長約146mm(推)
HOOD_LEN = 0.0500           # 薬室のかたまりの長さ(排莢口が後ろまで下がったとき、薬室が見えるように)
PORT_Y = (0.0500, 0.0815)   # 排莢口の前後(長さ約30mm、推 1-7)
PORT_Z0 = 0.0670            # 排莢口の下端(高さ約15mm、推 1-7)
INNER_X = -0.0072           # スライド内側(バレル溝)の左の壁

SLIDE_BACK = 0.034          # スライド後退量(実測例 34mm=.22 の S&W Model 41 ★。9mm は推 30〜38mm)
BARREL_BACK = 0.0045        # バレルの後退(推 3〜6mm)
BARREL_REAR_DROP = 0.0035   # バレル後端が下がる量(推 3〜4mm)
BARREL_PIVOT_Y = 0.140      # バレルが傾く支点(銃口側のブッシング付近。後端が3.5mm下がるなら、傾きは約2.3度。調査の「5〜8度」とは食い違う→ 3.5mm を優先)
TRIGGER_PULL = 0.0097       # 引き金の後退(Gen5 試験値 0.38in=9.7mm ★弱)
TRIGGER_PIVOT = (0.0556, 0.0466)    # 引き金ピン(メッシュの穴の位置 y,z)
TRIGGER_ROT = math.radians(-3.0)    # 引き金のわずかな回転(推)
STOP_LIFT = 0.0035          # スライドストップが上がる量(推 3〜4mm。後端が上がる回転で表す)
STOP_PIVOT = (0.0485, 0.0525)       # スライドストップの支点(前端)
STOP_REAR_Y = 0.0245

# 発砲サイクル(0ms = 撃針が雷管を打つ)。 (時刻ms, スライド後退m) の通過点。 ★ 後退の最大 9〜12ms、全周期 45〜57ms(.45ACP と .22 の実測。9mm は間の推定)
SLIDE_KEYS = [(0.0, 0.0), (1.5, 0.0045), (3.0, 0.0125), (6.0, 0.0245), (9.0, 0.0325), (10.5, 0.0340),
              (12.0, 0.0322), (13.5, 0.0340),          # 後端ではね返る(Hackaday ★「1〜2回はね返る」)
              (20.0, 0.0335), (26.0, 0.0300), (33.0, 0.0215), (40.0, 0.0105), (46.0, 0.0030), (50.0, 0.0000),
              (51.5, 0.0009), (53.0, 0.0000)]            # 閉鎖で小さくはね返る
T_CYCLE_MS = 53.0
EJECT_MS = 6.5              # 薬莢が排莢口から出る時刻(推 5〜8ms)
TRIGGER_RELEASE_MS = (2.0, 55.0)   # 引き金が戻る(推 50〜60ms)

CASING_DRAG = 2.6           # chase/fx.py の casing_pos と同じ式(空気抵抗 k=2.6、重力 9.81、路面ではね返る)
GRAVITY = 9.81
CASING_LEN = 0.01915        # 9x19mm パラベラム: 全長 19.15mm、底径 約9.9mm、口 約9.65mm
CASING_R_BASE = 0.00495
CASING_R_MOUTH = 0.00483
CASING_FLOOR_Z = 0.0035

PART_NAMES = ['frame', 'slide', 'barrel', 'trigger', 'slide_stop', 'magazine_base']
PARENT = {'frame': None, 'slide': 'frame', 'barrel': 'frame', 'trigger': 'frame', 'slide_stop': 'frame', 'magazine_base': 'frame'}
# 各部品の原点(回転の支点)。 メッシュは原点を引いて持つ。 動きは (dx,dy,dz,rx)= 静止位置からの移動(銃ローカル)と X 軸回りの回転(rad)
PIVOT = {'frame': (0, 0, 0), 'slide': (0, 0, 0), 'barrel': (0, BARREL_PIVOT_Y, BARREL_Z), 'trigger': (0, TRIGGER_PIVOT[0], TRIGGER_PIVOT[1]),
         'slide_stop': (0, STOP_PIVOT[0], STOP_PIVOT[1]), 'magazine_base': (0, 0, 0)}
COLORS = {'frame': (0.20, 0.45, 0.95), 'slide': (0.95, 0.25, 0.20), 'barrel': (0.95, 0.85, 0.15), 'trigger': (0.2, 0.85, 0.35),
          'slide_stop': (0.9, 0.35, 0.9), 'magazine_base': (0.15, 0.85, 0.85)}


# ============================================================ (A) メッシュ(numpy)
class Mesh:
    """P(n,3) 位置、N(n,3) 法線、F(m,3) 三角形、mat(m,) 材質番号(0=表面、1=切り口・内側、2=バレルなど鋼、3=薬莢)"""
    def __init__(self, P, N, F, mat):
        self.P = np.asarray(P, float).reshape(-1, 3); self.N = np.asarray(N, float).reshape(-1, 3)
        self.F = np.asarray(F, np.int64).reshape(-1, 3); self.mat = np.asarray(mat, np.int16).reshape(-1)

    def copy(self):
        return Mesh(self.P.copy(), self.N.copy(), self.F.copy(), self.mat.copy())


def concat(ms):
    ms = [m for m in ms if m is not None and len(m.F)]
    if not ms:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros(0, np.int16))
    off = np.cumsum([0] + [len(m.P) for m in ms])[:-1]
    return Mesh(np.vstack([m.P for m in ms]), np.vstack([m.N for m in ms]), np.vstack([m.F + o for m, o in zip(ms, off)]), np.concatenate([m.mat for m in ms]))


def take(m, mask):
    F = m.F[mask]; used, inv = np.unique(F.ravel(), return_inverse=True)
    return Mesh(m.P[used], m.N[used], inv.reshape(-1, 3), m.mat[mask])


def flip(m):
    return Mesh(m.P, -m.N, m.F[:, [0, 2, 1]], m.mat)


def tri_normals(m):
    T = m.P[m.F]; n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); return n / (np.linalg.norm(n, axis=1)[:, None] + 1e-30)


def centroids(m):
    return m.P[m.F].mean(1)


def boundary_edges(m, tol=1e-6):
    """位置で溶接したうえで、1回しか使われない辺の数(穴の目安)と、3回以上使われる辺の数"""
    key = np.round(m.P / tol).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True); W = inv.ravel()[m.F]
    e = np.sort(np.concatenate([W[:, [0, 1]], W[:, [1, 2]], W[:, [2, 0]]]), axis=0 if False else 1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return int((cnt == 1).sum()), int((cnt > 2).sum())


def load_gun_glb(path):
    """gun.glb を読み、銃ローカル座標(原点=グリップ中心、銃口 +Y、上 +Z)の Mesh にする(chase/rider.py import_gun と同じ変換)"""
    b = open(path, 'rb').read()
    l = struct.unpack('<I', b[12:16])[0]; j = json.loads(b[20:20 + l]); off = 20 + l + 8
    bv = j['bufferViews']

    def acc(i, dt, n):
        a = j['accessors'][i]; v = bv[a['bufferView']]
        return np.frombuffer(b, dt, a['count'] * n, off + v.get('byteOffset', 0) + a.get('byteOffset', 0)).reshape(-1, n)
    P = acc(0, '<f4', 3).astype(float); N = acc(1, '<f4', 3).astype(float); F = acc(3, '<u2', 1).reshape(-1, 3).astype(np.int64)
    Pg = np.c_[-P[:, 2], 0.065 - P[:, 0], P[:, 1] - 0.045]
    Ng = np.c_[-N[:, 2], -N[:, 0], N[:, 1]]
    return Mesh(Pg, Ng, F, np.zeros(len(F)))


# ---- 平面で切る(切り口を塞ぐ)
def _tess(rings2d):
    """[外周, 穴, 穴...] の2D点列 -> 三角形(連結した点列への添字)"""
    pts = np.vstack(rings2d)
    try:
        import mapbox_earcut as ec
        ends = np.cumsum([len(r) for r in rings2d]).astype(np.uint32)
        return np.asarray(ec.triangulate_float64(pts.astype(np.float64), ends), np.int64).reshape(-1, 3)
    except ImportError:
        pass
    from mathutils import Vector                    # Blender 内(earcut が無いとき)
    from mathutils.geometry import tessellate_polygon
    tri = tessellate_polygon([[Vector((x, y, 0.0)) for x, y in r] for r in rings2d])
    return np.asarray(tri, np.int64).reshape(-1, 3)


def _pip(pt, poly):
    x, y = pt; xs, ys = poly[:, 0], poly[:, 1]; xs2, ys2 = np.roll(xs, -1), np.roll(ys, -1)
    c = ((ys > y) != (ys2 > y)) & (x < (xs2 - xs) * (y - ys) / (ys2 - ys + 1e-30) + xs)
    return bool(c.sum() % 2)


def _cap(loops3, n):
    """切り口の閉じた線(3D)の集まり -> 平面 n 上の三角形 (点(k,3), 面(m,3))。 入れ子の穴に対応"""
    n = np.asarray(n, float); n /= np.linalg.norm(n)
    u = np.cross(n, [0, 0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0, 0]); u /= np.linalg.norm(u); v = np.cross(n, u)
    L2 = [np.c_[l @ u, l @ v] for l in loops3]
    contain = [[j for j, q in enumerate(L2) if j != i and abs(_area(q)) > abs(_area(L2[i])) and _pip(L2[i][0], q)] for i in range(len(L2))]
    depth = [len(c) for c in contain]
    parent = [max(c, key=lambda j: depth[j]) if c else -1 for c in contain]
    PT = []; FA = []; base = 0
    for i, p in enumerate(L2):
        if depth[i] % 2:
            continue
        holes = [j for j in range(len(L2)) if depth[j] % 2 and parent[j] == i]
        rings = [p] + [L2[j] for j in holes]; ring3 = [loops3[i]] + [loops3[j] for j in holes]
        try:
            tri = _tess(rings)
        except Exception as e:
            print('hand gun: cap tess failed', e); continue
        PT.append(np.vstack(ring3)); FA.append(tri + base); base += sum(len(r) for r in ring3)
    if not PT:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return np.vstack(PT), np.vstack(FA)


def _area(p):
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def _loops(segs, tol=1e-7):
    """線分 [(a,b)] (3D) -> 閉じた線の一覧。 閉じないものは数だけ返す"""
    key = lambda p: tuple(np.round(np.asarray(p) / tol).astype(np.int64))
    adj = {}; pos = {}
    for a, b in segs:
        ka, kb = key(a), key(b)
        if ka == kb:
            continue
        pos[ka] = a; pos[kb] = b
        adj.setdefault(ka, []).append(kb); adj.setdefault(kb, []).append(ka)
    used = set(); loops = []; opened = 0
    for s in list(adj):
        for t in list(adj[s]):
            e = frozenset((s, t))
            if e in used:
                continue
            used.add(e); path = [s, t]; cur = t; closed = False
            while True:
                nxt = None
                for w in adj[cur]:
                    if frozenset((cur, w)) not in used:
                        nxt = w; break
                if nxt is None:
                    break
                used.add(frozenset((cur, nxt)))
                if nxt == path[0]:
                    closed = True; break
                path.append(nxt); cur = nxt
            if closed and len(path) >= 3:
                loops.append(np.array([pos[k] for k in path]))
            else:
                opened += 1
    return loops, opened


def split_plane(m, n, d, cap=True, capmat=1):
    """平面 n·x=d で切る。 戻り値 (pos側 n·x>d, neg側, 開いた線の数)。 cap=True で切り口に蓋(法線は pos側 -n、neg側 +n)"""
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    s = m.P @ n - d; s = np.where(np.abs(s) < 1e-9, 1e-9, s)
    pos = (s > 0)[m.F]; npos = pos.sum(1)
    allp = npos == 3; alln = npos == 0; st = ~(allp | alln)
    P = [m.P]; N = [m.N]; nv = len(m.P)
    pf = [m.F[allp]]; pm = [m.mat[allp]]; nf = [m.F[alln]]; nm = [m.mat[alln]]
    NP = []; NN = []; segs = []
    for ti in np.nonzero(st)[0]:
        f = m.F[ti]; sg = pos[ti]
        k = next(i for i in range(3) if sg[i] != sg[(i + 1) % 3] and sg[i] != sg[(i + 2) % 3])
        v0, v1, v2 = f[k], f[(k + 1) % 3], f[(k + 2) % 3]
        pts = []
        for w in (v1, v2):
            t = s[v0] / (s[v0] - s[w]); p = m.P[v0] + t * (m.P[w] - m.P[v0])
            q = m.N[v0] + t * (m.N[w] - m.N[v0]); q = q / (np.linalg.norm(q) + 1e-30)
            pts.append((p, q))
        ia, ib = nv + len(NP), nv + len(NP) + 1
        NP += [pts[0][0], pts[1][0]]; NN += [pts[0][1], pts[1][1]]
        lone = [[v0, ia, ib]]; quad = [[ia, v1, v2], [ia, v2, ib]]; mt = m.mat[ti]
        if sg[k]:
            pf.append(np.array(lone)); pm.append([mt]); nf.append(np.array(quad)); nm.append([mt, mt])
        else:
            nf.append(np.array(lone)); nm.append([mt]); pf.append(np.array(quad)); pm.append([mt, mt])
        segs.append((pts[0][0], pts[1][0]))
    if NP:
        P.append(np.array(NP)); N.append(np.array(NN))
    PA = np.vstack(P); NA = np.vstack(N)
    A = Mesh(PA, NA, np.vstack(pf) if pf else np.zeros((0, 3), np.int64), np.concatenate([np.asarray(x, np.int16).ravel() for x in pm]) if pm else np.zeros(0, np.int16))
    B = Mesh(PA, NA, np.vstack(nf), np.concatenate([np.asarray(x, np.int16).ravel() for x in nm]))
    A = take(A, np.ones(len(A.F), bool)); B = take(B, np.ones(len(B.F), bool))
    opened = 0
    if cap and segs:
        loops, opened = _loops(segs)
        CP, CF = _cap(loops, n)
        if len(CF):
            Tn = np.cross(CP[CF[:, 1]] - CP[CF[:, 0]], CP[CF[:, 2]] - CP[CF[:, 0]])
            for target, tgt_n in ((A, -n), (B, n)):
                ok = (Tn @ tgt_n) > 0
                F2 = np.where(ok[:, None], CF, CF[:, [0, 2, 1]])
                cm = Mesh(CP, np.tile(tgt_n, (len(CP), 1)), F2, np.full(len(F2), capmat, np.int16))
                if target is A:
                    A = concat([A, cm])
                else:
                    B = concat([B, cm])
    return A, B, opened


def clip_keep(m, planes, capmat=1):
    """各半空間 n·x<=d の共通部分だけ残す(切り口に蓋)。 planes=[(n,d)]。 戻り値 (Mesh, 開いた線の数)"""
    cur = m; op = 0
    for n, d in planes:
        _, cur, o = split_plane(cur, n, d, True, capmat); op += o
    return cur, op


def carve(m, planes):
    """凸領域(半空間 n·x<=d の共通部分)を切り出す。 戻り値 (中身(切り口つき), 残り(切り口の内側に蓋), 開いた線の数)。 残りの蓋の材質は 1"""
    inside, op = clip_keep(m, planes, capmat=100)
    cur = m
    for n, d in planes:
        a, b, _ = split_plane(cur, n, d, False); cur = concat([a, b])
    c = centroids(cur)
    inmask = np.ones(len(cur.F), bool)
    for n, d in planes:
        inmask &= (c @ np.asarray(n, float) / np.linalg.norm(n) - d / np.linalg.norm(n)) <= 0
    capf = inside.mat == 100
    rest = concat([take(cur, ~inmask), flip(take(inside, capf))])
    rest.mat = np.where(rest.mat == 100, 1, rest.mat).astype(np.int16)
    inside.mat = np.where(inside.mat == 100, 1, inside.mat).astype(np.int16)
    return inside, rest, op


def box_planes(x=(None, None), y=(None, None), z=(None, None)):
    pl = []
    for ax, (lo, hi) in enumerate((x, y, z)):
        e = np.zeros(3); e[ax] = 1.0
        if hi is not None:
            pl.append((e.copy(), hi))
        if lo is not None:
            pl.append((-e, -lo))
    return pl


def poly_planes(poly, x=(None, None)):
    """y-z 平面の凸多角形(反時計回り)を、X 方向にのばした柱 -> 半空間"""
    pl = []
    k = len(poly)
    for i in range(k):
        (y0, z0), (y1, z1) = poly[i], poly[(i + 1) % k]
        ey, ez = y1 - y0, z1 - z0
        nrm = np.array([0.0, ez, -ey]); nrm /= np.linalg.norm(nrm)     # 反時計回りなら右向き法線 = 外向き
        pl.append((nrm, float(nrm @ np.array([0, y0, z0]))))
    pl += box_planes(x=x)
    return pl


# ---- 手続きのメッシュ
def revolve(profile, seg=32, axis='y', center=(0, 0, 0), mat=2, closed_cap=False):
    """(位置t, 半径r) の輪郭を軸まわりに回転した Mesh。 axis='y' なら +Y 方向に t"""
    pr = np.asarray(profile, float); k = len(pr); th = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    P = []; N = []
    for i, (t, r) in enumerate(pr):
        for a in th:
            P.append((r * math.cos(a), t, r * math.sin(a))); N.append((math.cos(a), 0, math.sin(a)))
    P = np.array(P) + np.array(center); N = np.array(N)
    # 輪郭の傾きで法線を補正
    for i in range(k):
        i0, i1 = max(i - 1, 0), min(i + 1, k - 1); dt = pr[i1, 0] - pr[i0, 0]; dr = pr[i1, 1] - pr[i0, 1]
        L = math.hypot(dt, dr) + 1e-30
        for s in range(seg):
            c = N[i * seg + s]; N[i * seg + s] = (c[0] * dt / L, -dr / L, c[2] * dt / L)
    F = []
    for i in range(k - 1):
        for s in range(seg):
            a, b = i * seg + s, i * seg + (s + 1) % seg; c, d = (i + 1) * seg + s, (i + 1) * seg + (s + 1) % seg
            F += [(a, c, b), (b, c, d)]      # t が増える向きに、外向きの巻き
    return Mesh(P, N, F, np.full(len(F), mat))


def box_mesh(x0, x1, y0, y1, z0, z1, mat=1, faces='all'):
    """直方体 Mesh。 faces: 使う面の集合(例 'all' か {'-x','+z'})。 外向き法線"""
    c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    quads = {'-z': ((0, 3, 2, 1), (0, 0, -1)), '+z': ((4, 5, 6, 7), (0, 0, 1)), '-y': ((0, 1, 5, 4), (0, -1, 0)), '+y': ((3, 7, 6, 2), (0, 1, 0)),
             '-x': ((0, 4, 7, 3), (-1, 0, 0)), '+x': ((1, 2, 6, 5), (1, 0, 0))}
    P = []; N = []; F = []
    for k, (q, nn) in quads.items():
        if faces != 'all' and k not in faces:
            continue
        b = len(P); P += [c[i] for i in q]; N += [nn] * 4; F += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    return Mesh(P, N, F, np.full(len(F), mat))


def make_barrel():
    """手続きのバレル(鋼)。 銃口 y=Y_MUZZLE、後端 y=BARREL_Y0。 後ろ 50mm は角ばった薬室のかたまり(排莢口から見える)、前は円筒。 銃口は内側が黒い穴。 薬室の上面に黒い凹み(チャンバー)"""
    r = BARREL_R; zc = BARREL_Z; yh = BARREL_Y0 + HOOD_LEN
    tube = revolve([(yh - 0.002, r), (Y_MUZZLE - 0.0006, r), (Y_MUZZLE, r * 0.93), (Y_MUZZLE, r * 0.55)], seg=40, center=(0, 0, zc), mat=2)
    hole = revolve([(Y_MUZZLE, r * 0.55), (Y_MUZZLE - 0.012, r * 0.52), (Y_MUZZLE - 0.012, 0.0)], seg=24, center=(0, 0, zc), mat=1)
    hole.F = hole.F[:, [0, 2, 1]]; hole.N = -hole.N                      # 穴は内向き
    hood = box_mesh(-0.0070, 0.0070, BARREL_Y0, yh, zc - 0.0055, zc + 0.0108, mat=2)
    ang = np.linspace(0, 2 * math.pi, 28, endpoint=False); cy = BARREL_Y0 + 0.0165; zt = zc + 0.0108
    ring = np.c_[0.0042 * np.cos(ang), cy + 0.0080 * np.sin(ang), np.full(28, zt)]
    inner = np.c_[0.0036 * np.cos(ang), cy + 0.0068 * np.sin(ang), np.full(28, zt - 0.0030)]
    P = np.vstack([ring, inner, [[0, cy, zt - 0.0030]]]); Nn = np.zeros_like(P); Nn[:, 2] = 1
    Fh = []
    for i in range(28):
        j = (i + 1) % 28; Fh += [(i, 28 + i, j), (j, 28 + i, 28 + j), (28 + i, 56, 28 + j)]
    Fh = np.array(Fh); trn = np.cross(P[Fh[:, 1]] - P[Fh[:, 0]], P[Fh[:, 2]] - P[Fh[:, 0]])
    Fh = np.where((trn[:, 2] < 0)[:, None], Fh[:, [0, 2, 1]], Fh)            # 向きをそろえる
    pocket = Mesh(P, Nn, Fh, np.full(len(Fh), 1))
    return concat([tube, hole, hood, pocket])


def make_casing(seg=20):
    """9mm の空の薬莢(真鍮)。 軸は +Z(長さ方向)、底(リム)が z=0、口が z=CASING_LEN。 口は内側が黒い穴、底にリムと雷管の凹み"""
    R, Rm, L = CASING_R_BASE, CASING_R_MOUTH, CASING_LEN
    prof = [(0.0, R * 0.55), (0.00015, R * 0.78), (0.0003, R * 1.005), (0.0011, R * 1.005), (0.0012, R * 0.98), (0.0016, R * 0.995), (L - 0.0012, Rm), (L - 0.0004, Rm * 0.99), (L, Rm * 0.93), (L, Rm * 0.80)]
    body = revolve(prof, seg=seg, mat=3)
    # y 軸方向の回転体を z 軸に向け直す
    Rm_ = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]]); body.P = body.P @ Rm_.T; body.N = body.N @ Rm_.T
    ins = revolve([(L, Rm * 0.80), (L - 0.012, Rm * 0.74), (L - 0.012, 0.0)], seg=seg, mat=1)
    ins.P = ins.P @ Rm_.T; ins.N = ins.N @ Rm_.T
    ins.N = -ins.N; ins.F = ins.F[:, [0, 2, 1]]     # 口の内側は内向き
    return concat([body, ins])


# ---- 切り分け本体
def build_parts(glb_path, verbose=True):
    """gun.glb -> {部品名: Mesh}(銃ローカル座標。原点は後で PIVOT で引く)。 report に各部品の穴の数などを入れる"""
    m = load_gun_glb(glb_path); rep = {}
    # (1) スライド / フレーム: 合わせ目の平面 z=Z_SLIDE
    slide, frame, o1 = split_plane(m, (0, 0, 1.0), Z_SLIDE, True)
    # (2) スライド前面: 小さな銃口の突起(Meshy)は捨て、手続きのバレルに替える
    stub, slide, o2 = split_plane(slide, (0, 1.0, 0), Y_SLIDE_FRONT, True)
    # (3) 排莢口: スライドの右上を直方体で切り出す(内側=バレルの溝の壁・床・前後の壁が、蓋として残る)
    _, slide, o3 = carve(slide, box_planes(x=(INNER_X, 0.05), y=PORT_Y, z=(PORT_Z0, 0.2)))
    # (4) 引き金(凸の柱で切り出す)
    trig_poly = [(0.0462, 0.0368), (0.0575, 0.0160), (0.0650, 0.0165), (0.0660, 0.0213), (0.0670, 0.0300), (0.0668, 0.0395), (0.0600, 0.0395)]     # y-z の反時計回り
    trig, frame, o4 = carve(frame, poly_planes(trig_poly, x=(-0.0065, 0.0065)))
    # (5) スライドストップ(左側のレバー)
    stop, frame, o5 = carve(frame, box_planes(x=(-0.0200, -0.0127), y=(0.0215, 0.0490), z=(0.0435, Z_SLIDE)))
    # (6) 弾倉の底板(グリップの下)
    frame, magb, o6 = split_plane(frame, (0, 0, 1.0), -0.0385, True)
    parts = {'frame': frame, 'slide': slide, 'barrel': make_barrel(), 'trigger': trig, 'slide_stop': stop, 'magazine_base': magb}
    rep['open_loops'] = dict(slide_frame=o1, stub=o2, port=o3, trigger=o4, slide_stop=o5, mag=o6)
    for k, v in parts.items():
        rep[k] = dict(tris=int(len(v.F)), boundary=boundary_edges(v)[0], bbox=([np.round(v.P.min(0), 4).tolist(), np.round(v.P.max(0), 4).tolist()] if len(v.P) else None))
    if verbose:
        print('hand gun: parts', json.dumps(rep))
    return parts, rep


# ============================================================ (B) 動き(純関数)
def _pchip(keys, t):
    xs = np.array([k[0] for k in keys], float); ys = np.array([k[1] for k in keys], float)
    if t <= xs[0]:
        return float(ys[0])
    if t >= xs[-1]:
        return float(ys[-1])
    h = np.diff(xs); dl = np.diff(ys) / h; dd = np.zeros(len(xs))
    for i in range(1, len(xs) - 1):
        if dl[i - 1] * dl[i] > 0:
            w1 = 2 * h[i] + h[i - 1]; w2 = h[i] + 2 * h[i - 1]; dd[i] = (w1 + w2) / (w1 / dl[i - 1] + w2 / dl[i])
    dd[0] = dd[-1] = 0.0                                  # 両端は速度 0(撃針が打った瞬間・閉鎖で止まる)
    i = int(np.searchsorted(xs, t) - 1); u = (t - xs[i]) / h[i]
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1; h10 = u ** 3 - 2 * u ** 2 + u; h01 = -2 * u ** 3 + 3 * u ** 2; h11 = u ** 3 - u ** 2
    return float(h00 * ys[i] + h10 * h[i] * dd[i] + h01 * ys[i + 1] + h11 * h[i] * dd[i + 1])


def _smooth(a, b, t):
    u = min(max((t - a) / (b - a), 0.0), 1.0); return u * u * (3 - 2 * u)


def slide_back(t_ms, locked=False):
    """スライドの後退量 m(0=閉鎖、SLIDE_BACK=最後尾)。 locked=True: 最後の1発のあと、後退位置で止まる(スライドストップ)"""
    if t_ms is None or t_ms < 0:
        return 0.0
    if locked:                                          # 後端まで行き、ストップの切り欠きに入って止まる(はね返りは吸収)
        s = _pchip(SLIDE_KEYS[:6], t_ms) if t_ms <= 10.5 else SLIDE_BACK - SLIDE_BACK * 0.015 * _smooth(10.5, 13.5, t_ms)
        return s
    return _pchip(SLIDE_KEYS, t_ms)


def gun_pose(t_since_fire=None, fired_count=0, total_rounds=6, trigger=0.0, stretch=3.5):
    """部品ごとの姿勢 {部品名: (dx,dy,dz,rx)}。 静止位置からの移動(銃ローカル m)と、X 軸まわりの回転(rad。部品の PIVOT のまわり)。
    t_since_fire: 発砲(撃針が雷管を打つ)からの経過秒(動画上の時間)。 None=発砲していない。
    fired_count: この発砲を含めた、これまでに撃った数。 total_rounds: 弾の数。 fired_count>=total_rounds の発砲が最後の1発(スライドは後退位置で止まる)
    trigger: 発砲前の引き金の引き(0〜1)。 発砲後は自動で戻る(約50〜60ms)。 発砲していない状態では trigger そのまま
    stretch: 時間の引き伸ばし倍率(実時間の約50msのサイクルを、動画で見えるように stretch 倍に)。 既定 3.5 = 約0.19秒。 1.0 で実時間
    スライドが止まったあと(最後の1発のあと)は、t_since_fire を大きくしても止まったまま。 fired_count>=total_rounds かつ t_since_fire=None のときも止まった状態(撃ち切って構えている)"""
    last = fired_count >= total_rounds and fired_count > 0
    if t_since_fire is None:
        t_ms = None
    else:
        t_ms = t_since_fire / max(stretch, 1e-6) * 1000.0
    pose = {k: (0.0, 0.0, 0.0, 0.0) for k in PART_NAMES}
    # スライドと、スライドストップ
    if t_ms is None:
        s = SLIDE_BACK * 0.985 if last else 0.0
    else:
        s = slide_back(t_ms, locked=last)
    pose['slide'] = (0.0, -s, 0.0, 0.0)
    lift = 0.0
    if last:
        # 弾倉の底板(フォロワー)が最後の弾のあとレバーを押し上げる。 スライドが切り欠きの位置(ほぼ後端)に来てから
        lift = 1.0 if t_ms is None else _smooth(8.0, 11.0, t_ms)
    ang = -math.atan2(STOP_LIFT * lift, STOP_PIVOT[0] - STOP_REAR_Y)
    pose['slide_stop'] = (0.0, 0.0, 0.0, ang)
    # バレル: 後退の前半でスライドと一緒に後ろへ(最大 BARREL_BACK)、後端が下がり、外れたところで止まる。 戻りでスライドに押されて前へ
    ret = (14.0, 22.0) if last else (37.0, 47.0)       # 最後の1発: スライドは後ろで止まるが、バレルはバネで前へ戻る
    if t_ms is None:
        bb = 0.0; tilt = 0.0
    else:
        back = 1.0 - _smooth(ret[0], ret[1], t_ms)
        tilt = _smooth(0.0, 4.0, t_ms) * back; bb = BARREL_BACK * _smooth(0.0, 4.5, t_ms) * back
    L = BARREL_PIVOT_Y - BARREL_Y0
    pose['barrel'] = (0.0, -bb, 0.0, math.atan2(BARREL_REAR_DROP * tilt, L))     # X軸まわりの正回転 = 後端が下がる
    # 引き金: 引き(後ろへ・わずかに回る)。 発砲のあと自動で戻る
    pull = trigger if t_ms is None else trigger * (1.0 - _smooth(TRIGGER_RELEASE_MS[0], TRIGGER_RELEASE_MS[1], t_ms))
    pose['trigger'] = (0.0, -TRIGGER_PULL * pull, 0.0, TRIGGER_ROT * pull)
    return pose


def casing_launch(seed=0, speed=None):
    """排莢の初期状態(銃ローカル)。 -> dict(t_ms, p0, v0, spin(rad/s ×3), axis0)。 初速 3〜5m/s、向きは右0.8:後0.4:上0.45(推 1-4)、回転 10〜30回転/秒(推)"""
    r = np.random.RandomState(seed)
    sp = r.uniform(3.0, 5.0) if speed is None else speed
    d = np.array([0.8, -0.4, 0.45]) + r.uniform(-0.12, 0.12, 3); d /= np.linalg.norm(d)
    s_ms = EJECT_MS
    y_port = 0.5 * (PORT_Y[0] + PORT_Y[1]) - slide_back(s_ms)
    p0 = np.array([0.012, y_port, 0.0725])
    spin = r.uniform(10, 30) * 2 * math.pi * np.array([r.choice([-1, 1]) * r.uniform(0.3, 1.0), r.uniform(-0.4, 0.4), r.uniform(0.3, 1.0)])
    return dict(t_ms=s_ms, p0=p0, v0=sp * d, spin=spin, axis0=np.array([0.2, 0.0, 1.0]))


def casing_state(c, t_since_fire, stretch=3.5, floor_z=-0.40, k=CASING_DRAG, base_vel=(0, 0, 0)):
    """薬莢の位置と向き(銃ローカル。静止した銃として)。 -> (位置, 回転行列) か None(まだ出ていない)。
    式は chase/fx.py の casing_pos と同じ(空気抵抗 k=2.6、重力、路面 floor_z ではね返る)。 時間は stretch で引き伸ばし。
    走行中の銃に使うときは、base_vel(銃のワールド速度)を v0 に足し、floor_z を路面の高さに、p0 を銃のワールド位置に直す"""
    age = t_since_fire / max(stretch, 1e-6) - c['t_ms'] / 1000.0
    if age < 0:
        return None
    n = max(1, int(age / 0.002)); dt = age / n
    p = np.array(c['p0'], float); v = np.array(c['v0'], float) + np.array(base_vel, float)
    for _ in range(n):
        v = v * (1 - k * dt) + np.array([0, 0, -GRAVITY * dt]); p = p + v * dt
        if p[2] < floor_z:
            p[2] = floor_z; v = np.array([v[0] * 0.5, v[1] * 0.5, -v[2] * 0.35])
    w = c['spin'] * age * (0.25 if p[2] <= floor_z + 1e-9 else 1.0)
    return p, _rot_xyz(*w)


def _rot_xyz(a, b, c):
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]]); Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]]); Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def part_transform(nm, pe):
    """部品 nm の姿勢 pe=(dx,dy,dz,rx) -> 銃ルート座標での 4x4 行列(原点の支点 PIVOT まわりに X 軸回転してから移動)。 numpy 確認用"""
    dx, dy, dz, rx = pe; pv = np.array(PIVOT[nm], float); c, s = math.cos(rx), math.sin(rx)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]); T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pv + np.array([dx, dy, dz]) - R @ pv
    return T


def posed_parts(parts, pose):
    out = {}
    for nm, m in parts.items():
        T = part_transform(nm, pose[nm]); out[nm] = Mesh(m.P @ T[:3, :3].T + T[:3, 3], m.N @ T[:3, :3].T, m.F, m.mat)
    return out


# ============================================================ (D) Blender(bpy が要る)
MAT_SPEC = {0: ('gun_body', (0.04, 0.042, 0.048), 0.6, 0.42), 1: ('gun_inner', (0.022, 0.023, 0.026), 0.85, 0.5),
            2: ('gun_steel', (0.20, 0.205, 0.215), 0.95, 0.28), 3: ('casing_brass', (0.80, 0.55, 0.18), 1.0, 0.28)}


def _material(bpy, mid, color=None):
    name, col, met, rough = MAT_SPEC[mid]
    if color is not None:
        if mid in (1, 2):
            color = tuple(c * 0.45 for c in color)       # 色分け表示では、切り口・内側を濃くする
        met, rough = 0.0, 0.55
        name = '%s_c%02d%02d%02d' % (name, *[int(c * 99) for c in color]); col = color
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value = (*col, 1.0)
    b.inputs['Metallic'].default_value = met; b.inputs['Roughness'].default_value = rough
    return m


def mesh_to_object(bpy, name, mesh, origin=(0, 0, 0), color=None, collection=None):
    """Mesh -> Blender オブジェクト。 origin を引いて持ち、オブジェクトの位置を origin にする(回転の支点)"""
    P = mesh.P - np.asarray(origin, float)
    me = bpy.data.meshes.new(name); me.from_pydata(P.tolist(), [], mesh.F.tolist()); me.update()
    mids = sorted(set(mesh.mat.tolist()))
    for mid in mids:
        me.materials.append(_material(bpy, mid, color))
    idx = {mid: i for i, mid in enumerate(mids)}
    me.polygons.foreach_set('material_index', [idx[x] for x in mesh.mat.tolist()])
    me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
    try:
        me.normals_split_custom_set_from_vertices([tuple(x) for x in mesh.N])
    except Exception as e:
        print('hand gun: custom normals', e)
    ob = bpy.data.objects.new(name, me); ob.location = tuple(origin)
    (collection or bpy.context.scene.collection).objects.link(ob)
    return ob


def build_blend(bpy, glb_path, parts=None, color_by_part=False):
    """部品ごとのオブジェクトと親子を作る。 戻り値 {名前: オブジェクト}(+ 'gun_root' の Empty)"""
    if parts is None:
        parts, _ = build_parts(glb_path)
    root = bpy.data.objects.new('gun_root', None); bpy.context.scene.collection.objects.link(root)
    obs = {'gun_root': root}
    for nm in PART_NAMES:
        ob = mesh_to_object(bpy, nm, parts[nm], PIVOT[nm], COLORS[nm] if color_by_part else None)
        obs[nm] = ob
    for nm in PART_NAMES:
        par = PARENT[nm]; ob = obs[nm]
        ob.parent = obs[par] if par else root
        if par:
            ob.location = tuple(PIVOT[nm]); ob.matrix_parent_inverse.identity()
            if PIVOT[par] != (0, 0, 0):
                ob.location = tuple(np.array(PIVOT[nm]) - np.array(PIVOT[par]))
        ob.rotation_mode = 'XYZ'
    return obs


def apply_pose(obs, pose):
    for nm, (dx, dy, dz, rx) in pose.items():
        ob = obs[nm]; base = np.array(PIVOT[nm]) - np.array(PIVOT[PARENT[nm]] if PARENT[nm] else (0, 0, 0))
        ob.location = tuple(base + np.array([dx, dy, dz])); ob.rotation_euler = (rx, 0.0, 0.0)


def add_casing_object(bpy, name='casing'):
    ob = mesh_to_object(bpy, name, make_casing(), (0, 0, CASING_LEN * 0.5))
    return ob


def make_actions(bpy, obs, fps=240, stretch=3.5, total_rounds=6):
    """アクション「発砲」「発砲(弾切れ)」を作る(1/fps 秒刻み。fps=240)。 NLA に並べる"""
    sc = bpy.context.scene; sc.render.fps = fps
    try:
        bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
    except Exception:
        pass
    dur = (T_CYCLE_MS + 8) / 1000.0 * stretch
    nfr = int(math.ceil(dur * fps)) + 1
    made = {}
    for clip, fired in (('発砲', 1), ('発砲(弾切れ)', total_rounds)):
        for nm in PART_NAMES:
            if nm == 'frame':
                continue
            ob = obs[nm]
            ob.animation_data_create()
            act = bpy.data.actions.new('%s_%s' % (nm, clip)); ob.animation_data.action = act
            for f in range(nfr):
                t = f / fps
                pose = gun_pose(t, fired, total_rounds, 1.0, stretch)
                apply_pose({nm: ob}, {nm: pose[nm]})
                ob.keyframe_insert('location', frame=f + 1); ob.keyframe_insert('rotation_euler', index=0, frame=f + 1)
            tr = ob.animation_data.nla_tracks.new(); tr.name = clip
            tr.strips.new(clip, 1, act)
            ob.animation_data.action = None
            made[(nm, clip)] = act
    sc.frame_start = 1; sc.frame_end = nfr
    return made


def export_glb(bpy, path, animation=True):
    kw = dict(filepath=path, export_format='GLB', use_selection=False, export_apply=False)
    tries = []
    if animation:
        tries.append(dict(kw, export_animations=True, export_animation_mode='NLA_TRACKS'))
        tries.append(dict(kw, export_animations=True))
    tries.append(dict(kw, export_animations=False))
    for k in tries:
        try:
            bpy.ops.export_scene.gltf(**k); print('hand gun: glb exported', os.path.basename(path), 'anim=', k.get('export_animations'), k.get('export_animation_mode')); return k
        except Exception as e:
            print('hand gun: glb export failed', str(e)[:200])
    return None
