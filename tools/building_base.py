"""building_base.py — ビルの足元(1階)を作るキット(Blender Python)。他の動画から import して使える。street_kit.py と同じフォルダに置く。
依存: bpy・mathutils(Blender内のみ)・numpy・標準ライブラリ。 build_geometry は bpy 無しで動く(寸法・面数の数値確認用)。

使い方
    import sys; sys.path.insert(0, '<このファイルのフォルダ>')
    import street_kit as sk, building_base as bb
    st = sk.make_street_section('tokyo246', None, 120, 1, 'near'); I = sk.street_info(st)
    g = bb.make_ground_floor('glass', width=12.0, height=4.2, seed=3, detail='near', depth=10.0, base_z=I['base_z'], gl=I['gl'], position=(40, I['wall_plane'] + 5.0))
    # 上の階(building_kit)は z = base_z + height の上から。 gaps = bb.floor_info(g)['gaps_x'] を street の found_gaps に渡すと、入口の前の基礎を切れる。

kind   : 'glass'(ガラス張り+金属レール) / 'tile'(タイル・石の腰壁) / 'shutter'(シャッター) / 'entrance'(エントランス) / 'parking'(駐車場の出入口)(日本語名も可)
座標   : 原点 = 建物の底面(幅 x 奥行)の中心。+Z=上、正面(道路側) = -Y 向き(building_kit・street_kit と同じ)。壁面 = y=-depth/2。 z は絶対値(base_z が1階の床)。
         position=(x, y) は底面の中心の位置、rotation = Z 回りの角度[rad]。
看板   : 帯(色面)だけ。文字・ロゴは作らない。
仕様の根拠: リサーチ/2026-11-19_道路と建物の接地まわり_歩道・基礎・1階.md の 1-3(1階の構成の種類)。★=出典あり、推=推定。
"""
import math, json, time, random
import numpy as np

try:
    import bpy
except Exception:
    bpy = None
import street_kit as sk
from street_kit import MD, Fr, box, tube, NB, new_material, principled


def finish(mat, nt, out, shader, role=None):
    nt.links.new(shader, out.inputs['Surface'])
    if role in SIMPLE:
        mat['kit_simple'] = json.dumps(SIMPLE[role])
    return mat


KINDS = {'glass': 'glass', 'ガラス張り': 'glass', 'ガラス': 'glass', 'tile': 'tile', '腰壁': 'tile', 'タイル': 'tile', 'shutter': 'shutter', 'シャッター': 'shutter',
         'entrance': 'entrance', 'エントランス': 'entrance', '玄関': 'entrance', 'parking': 'parking', '駐車場': 'parking'}
FASCIA = [(0.80, 0.80, 0.78), (0.10, 0.10, 0.11), (0.03, 0.06, 0.16), (0.30, 0.07, 0.05), (0.04, 0.17, 0.09), (0.45, 0.30, 0.10)]   # リニア色。帯だけ(文字なし)
WALLS = [(0.50, 0.48, 0.44), (0.62, 0.60, 0.56), (0.40, 0.38, 0.35), (0.55, 0.50, 0.42)]
RAILS = {'silver': dict(c=(0.62, 0.63, 0.65), r=0.35), 'bronze': dict(c=(0.10, 0.065, 0.04), r=0.4), 'black': dict(c=(0.02, 0.02, 0.022), r=0.45)}
ROOMLIGHT = [(1.0, 0.80, 0.55), (1.0, 0.92, 0.80), (0.86, 0.93, 1.0)]

SIMPLE = {
    'wall': dict(c=(0.5, 0.48, 0.44), r=0.9), 'wall_tile': dict(c=(0.30, 0.20, 0.15), r=0.35), 'stone': dict(c=(0.10, 0.10, 0.105), r=0.3), 'stone_cap': dict(c=(0.15, 0.15, 0.15), r=0.4),
    'stainless': dict(c=(0.62, 0.62, 0.64), r=0.35, m=1.0), 'rail_metal': dict(c=(0.62, 0.63, 0.65), r=0.35, m=1.0), 'kick': dict(c=(0.05, 0.05, 0.055), r=0.45, m=0.8),
    'glass': dict(c=(0.10, 0.12, 0.13), r=0.05, m=0.5, a=0.35), 'room_wall': dict(c=(0.5, 0.45, 0.38), r=0.9, e=(1.0, 0.82, 0.58), es=2.5), 'room_light': dict(c=(1, 1, 1), r=0.5, e=(1.0, 0.9, 0.75), es=12.0),
    'room_ceiling': dict(c=(0.6, 0.6, 0.58), r=0.9), 'room_floor': dict(c=(0.45, 0.43, 0.40), r=0.4), 'furniture': dict(c=(0.35, 0.28, 0.20), r=0.6),
    'fascia': dict(c=(0.1, 0.1, 0.11), r=0.5), 'shutter': dict(c=(0.42, 0.43, 0.44), r=0.45, m=0.8), 'shutter_box': dict(c=(0.35, 0.36, 0.37), r=0.5, m=0.7),
    'door_frame': dict(c=(0.10, 0.065, 0.04), r=0.4, m=1.0), 'canopy': dict(c=(0.55, 0.56, 0.58), r=0.45, m=0.9), 'downlight': dict(c=(1, 1, 1), r=0.5, e=(1.0, 0.9, 0.75), es=60.0),
    'entry_floor': dict(c=(0.12, 0.12, 0.125), r=0.3), 'step': dict(c=(0.3, 0.3, 0.29), r=0.9), 'handrail': dict(c=(0.62, 0.62, 0.64), r=0.3, m=1.0),
    'dark_interior': dict(c=(0.03, 0.03, 0.03), r=0.9), 'ceiling_pipe': dict(c=(0.35, 0.36, 0.37), r=0.6, m=0.5), 'parking_floor': dict(c=(0.07, 0.07, 0.07), r=0.85),
    'sign_dark': dict(c=(0.02, 0.02, 0.02), r=0.9),
}


def params(**kw):
    return kw


# ------------------------------------------------------------------------------------------------
# ジオメトリ
# ------------------------------------------------------------------------------------------------
class G:
    def __init__(self, W, D, H, bz, gl, rng, night, detail):
        self.md = MD(); self.W = W; self.D = D; self.H = H; self.bz = bz; self.gl = gl; self.rng = rng; self.night = night; self.detail = detail
        self.yf = -D / 2.0; self.fr = Fr((0, 0, 0), (1, 0, 0), (0, 1, 0)); self.gaps = []; self.dims = {}

    def X(self, x0, x1, d0, d1, z0, z1, role, skip=()):
        if x1 - x0 < 1e-6 or z1 - z0 < 1e-6 or d1 - d0 < 1e-6:
            return
        box(self.md, self.fr, x0, x1, self.yf + d0, self.yf + d1, z0, z1, role, skip)

    def front(self, x0, x1, z0, z1, d, role):          # 法線 -Y
        if x1 - x0 < 1e-6 or z1 - z0 < 1e-6:
            return
        y = self.yf + d
        self.md.face([(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)], role, [(x0, z0), (x1, z0), (x1, z1), (x0, z1)])

    def side(self, x, d0, d1, z0, z1, role, face):     # face=+1: 法線 +X(内向き左壁など) / -1: 法線 -X
        y0, y1 = self.yf + d0, self.yf + d1
        p = [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]
        uv = [(d0, z0), (d1, z0), (d1, z1), (d0, z1)]
        if face < 0:
            p = p[::-1]; uv = uv[::-1]
        self.md.face(p, role, uv)

    def flat(self, x0, x1, d0, d1, z, role, up=True):
        y0, y1 = self.yf + d0, self.yf + d1
        p = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]; uv = [(x0, d0), (x1, d0), (x1, d1), (x0, d1)]
        if not up:
            p = p[::-1]; uv = uv[::-1]
        self.md.face(p, role, uv)

    def wall(self, x0, x1, z0, z1, holes, role, d=0.0):
        """正面の壁(穴あき)。 holes = [(hx0,hx1,hz0,hz1)]。 穴の内側(見込み)は rec で別に作る"""
        xs = sorted({x0, x1} | {min(max(h[0], x0), x1) for h in holes} | {min(max(h[1], x0), x1) for h in holes})
        for a, b in zip(xs[:-1], xs[1:]):
            m = 0.5 * (a + b); cuts = sorted((max(h[2], z0), min(h[3], z1)) for h in holes if h[0] <= m <= h[1])
            z = z0
            for c0, c1 in cuts:
                if c0 > z:
                    self.front(a, b, z, c0, d, role)
                z = max(z, c1)
            if z < z1:
                self.front(a, b, z, z1, d, role)

    def reveal(self, hx0, hx1, hz0, hz1, rec, role, d=0.0):
        self.side(hx0, d, d + rec, hz0, hz1, role, +1); self.side(hx1, d, d + rec, hz0, hz1, role, -1)
        self.flat(hx0, hx1, d, d + rec, hz0, role, True); self.flat(hx0, hx1, d, d + rec, hz1, role, False)


def room(g, x0, x1, z0, z1, rd, opt, d0=0.0, cut=None, fl=None):
    """店内(窓の奥の箱)。 奥の壁・側壁・床・天井・天井灯の面。 d=d0〜rd。 cut=(cx0,cx1,r): その x 範囲は d<r を床・天井から除く(入口の凹みの段差を見せるため)"""
    far = g.detail == 'far'
    g.front(x0, x1, z0, z1, rd, 'room_wall')
    if far:
        return
    g.side(x0, d0, rd, z0, z1, 'room_wall', +1); g.side(x1, d0, rd, z0, z1, 'room_wall', -1)
    parts = [(x0, x1, d0)] if cut is None else [(x0, cut[0], d0), (cut[1], x1, d0), (cut[0], cut[1], max(d0, cut[2]))]
    for a, b, dd in parts:
        g.flat(a, b, dd, rd, z0, fl or 'room_floor', True); g.flat(a, b, dd, rd, z1, 'room_ceiling', False)
    nx = max(1, int((x1 - x0) / 1.8)); nd = max(1, int((rd - d0) / 1.8)); px = (x1 - x0) / nx; pd = (rd - d0) / nd
    for i in range(nx):
        for j in range(nd):
            cx = x0 + px * (i + 0.5); cd = d0 + pd * (j + 0.5)
            if cut is not None and cut[0] < cx < cut[1] and cd < cut[2] + 0.6:
                continue
            g.flat(cx - 0.3, cx + 0.3, cd - 0.6, cd + 0.6, z1 - 0.003, 'room_light', False)
    if g.detail == 'near':      # 棚・カウンター(箱)
        for k in range(4):
            wx = g.rng.uniform(0.8, 2.2); xc = g.rng.uniform(x0 + wx / 2 + 0.3, x1 - wx / 2 - 0.3); dd = g.rng.uniform(max(2.0, d0 + 1.0), rd - 1.0)
            g.X(xc - wx / 2, xc + wx / 2, dd, dd + 0.6, z0, z0 + g.rng.uniform(0.9, 1.6), 'furniture')


def steps(g, x0, x1, r, hd, handrail=True):
    """入口の凹み(深さ r)の中の段差とスロープなしの階段。 外の地面 gl から 床 bz まで。 hd = bz - gl"""
    bz = g.bz; gl = g.gl
    if hd <= 0.025:
        g.flat(x0, x1, 0.0, r, bz, 'entry_floor', True); g.dims['steps'] = 0; return
    n = int(math.ceil(hd / 0.16)); rs = hd / n; L0 = r - 0.30 * (n - 1)
    for k in range(1, n):
        d0 = 0.30 * (k - 1); g.X(x0, x1, d0, d0 + 0.30, gl, gl + k * rs, 'step', skip=('-z',))
    g.X(x0, x1, 0.30 * (n - 1), r, gl, bz, 'step', skip=('-z',)); g.flat(x0, x1, 0.30 * (n - 1), r, bz + 0.001, 'entry_floor', True)
    g.dims['steps'] = n; g.dims['riser'] = round(rs, 3); g.dims['landing'] = round(L0, 2)
    if hd > 0.16 and handrail and g.detail != 'far':          # 高さ16cm超の傾斜には手すり★(群馬県)。 0.85m・ステンレス丸パイプ 径34〜42mm
        for xx in (x0 + 0.12, x1 - 0.12):
            a = np.array([xx, g.yf + 0.0, gl + rs + 0.85]); b = np.array([xx, g.yf + 0.30 * (n - 1), bz + 0.85]); c = np.array([xx, g.yf + r - 0.1, bz + 0.85])
            tube(g.md, a, b, 0.019, 0.019, 8, 'handrail', caps=True); tube(g.md, b, c, 0.019, 0.019, 8, 'handrail', caps=True)
            tube(g.md, b, np.array([xx, b[1], bz]), 0.012, 0.012, 6, 'handrail', caps=False)
        g.dims['handrail_h'] = 0.85


def fascia(g, fas, out=0.10):
    z1 = g.bz + g.H; col = g.rng.randrange(len(FASCIA)); g.dims['fascia_color'] = col
    g.X(-g.W / 2, g.W / 2, -out, 0.12, z1 - fas, z1, 'fascia')


def k_glass(g, o):
    W, H, bz, rng = g.W, g.H, g.bz, g.rng
    fas = o.get('fascia', rng.uniform(0.5, 0.8)); stall = o.get('stall', rng.choice([0.45, 0.6, 0.75, 0.9]))
    rw = o.get('rail_w', round(rng.uniform(0.06, 0.12), 3)); nb = max(1, int(round(W / o.get('mull_pitch', rng.uniform(1.2, 1.7))))); pitch = W / nb
    mw = o.get('mull_w', 0.075); dh = 2.4; dw = min(1.8, W - 0.8)
    door = o.get('door', True) and W >= 3.0
    gt = bz + H - fas; g.dims.update(fascia=round(fas, 2), stall=stall, rail_w=rw, mull_pitch=round(pitch, 2), mull_w=mw, glass_h=round(H - fas - stall - rw, 2), glass_ratio=round((H - fas - stall) / H, 2), stall_ratio=round(stall / H, 2), fascia_ratio=round(fas / H, 2))
    xd0 = xd1 = 0.0
    if door:
        k = max(1, nb // 2) if nb > 1 else 0; xd = -W / 2 + pitch * k if nb > 1 else 0.0
        xd = min(max(xd, -W / 2 + dw / 2 + 0.2), W / 2 - dw / 2 - 0.2); xd0, xd1 = xd - dw / 2, xd + dw / 2
        g.gaps.append((xd0, xd1)); g.dims.update(door_w=round(dw, 2), door_h=dh)
    segs = [(-W / 2, W / 2)] if not door else [(-W / 2, xd0), (xd1, W / 2)]
    stall_role = rng.choice(['stainless', 'stainless', 'wall_tile', 'stone']); rail_kind = rng.choice(list(RAILS)); g.dims.update(stall_mat=stall_role, rail_kind=rail_kind)
    for a, b in segs:
        g.X(a, b, -0.01, 0.0, bz + 0.05, bz + stall, stall_role)           # 腰の化粧板(ステンレス・タイル・石)
        g.X(a, b, -0.012, 0.0, bz, bz + 0.05, 'kick')                      # 床際の金属の見切り・巾木
        g.X(a, b, -0.03, 0.10, bz + stall, bz + stall + rw, 'rail_metal')  # 下枠(金属のレール) 見付け60〜120mm
        g.front(a, b, bz + stall + rw, gt - rw, 0.03, 'glass')
    g.X(-W / 2, W / 2, -0.03, 0.10, gt - rw, gt, 'rail_metal')            # 上枠
    for i in range(nb + 1):
        x = -W / 2 + pitch * i
        if door and xd0 - 1e-6 < x < xd1 + 1e-6:
            continue
        a = max(x - mw / 2, -W / 2); b = min(x + mw / 2, W / 2)
        g.X(a, b, -0.02, 0.10, bz + stall + rw, gt - rw, 'rail_metal')     # 縦桟(マリオン)
    if door:
        r = 0.9; hd = bz - g.gl
        g.front(xd0, xd1, bz + dh + 0.15, gt - rw, 0.03, 'glass')                                           # 欄間
        g.X(xd0, xd1, -0.02, 0.10, bz + dh, bz + dh + 0.15, 'door_frame')
        g.side(xd0, 0.0, r, bz, bz + dh + 0.15, 'stainless', +1); g.side(xd1, 0.0, r, bz, bz + dh + 0.15, 'stainless', -1)
        g.flat(xd0, xd1, 0.0, r, bz + dh + 0.15, 'stainless', False)
        steps(g, xd0, xd1, r, hd)
        g.X(xd0, xd0 + 0.13, r - 0.05, r + 0.06, bz, bz + dh, 'door_frame'); g.X(xd1 - 0.13, xd1, r - 0.05, r + 0.06, bz, bz + dh, 'door_frame')
        g.X(xd0, xd1, r - 0.05, r + 0.06, bz + dh - 0.12, bz + dh, 'door_frame'); g.X(-0.05 + (xd0 + xd1) / 2, 0.05 + (xd0 + xd1) / 2, r - 0.05, r + 0.06, bz, bz + dh, 'door_frame')
        g.front(xd0 + 0.13, xd1 - 0.13, bz + 0.05, bz + dh - 0.12, r, 'glass')
    rd = min(o.get('room_depth', 6.0), g.D - 0.4)
    room(g, -W / 2, W / 2, bz, gt - 0.05, rd, o, cut=((xd0, xd1, 0.9) if door else None))
    fascia(g, fas)


def k_tile(g, o):
    W, H, bz, rng = g.W, g.H, g.bz, g.rng
    fas = o.get('fascia', rng.uniform(0.5, 0.8)); wh = o.get('wain_h', rng.choice([0.6, 0.9, 1.2])); mat = o.get('wain_mat', rng.choice(['wall_tile', 'stone', 'wall_tile']))
    g.dims.update(wain_h=wh, wain_mat=mat, fascia=round(fas, 2))
    g.front(-W / 2, W / 2, bz, bz + wh, 0.0, mat)
    g.X(-W / 2, W / 2, -0.03, 0.02, bz + wh, bz + wh + 0.04, 'stone_cap')       # 笠木(上端の水平の見切り)
    nw = max(1, int(W // 3.2)); ww = 1.6; holes = []
    for i in range(nw):
        cx = -W / 2 + W * (i + 0.5) / nw; holes.append((cx - ww / 2, cx + ww / 2, bz + 1.5, min(bz + 3.0, bz + H - fas - 0.3)))
    g.wall(-W / 2, W / 2, bz + wh + 0.04, bz + H - fas, holes, 'wall')
    for hx0, hx1, hz0, hz1 in holes:
        g.reveal(hx0, hx1, hz0, hz1, 0.12, 'wall'); g.front(hx0, hx1, hz0, hz1, 0.14, 'glass')
        g.X(hx0 - 0.03, hx1 + 0.03, -0.05, 0.02, hz0 - 0.04, hz0, 'stone_cap')
        g.X(hx0, hx0 + 0.05, 0.10, 0.17, hz0, hz1, 'rail_metal'); g.X(hx1 - 0.05, hx1, 0.10, 0.17, hz0, hz1, 'rail_metal'); g.X((hx0 + hx1) / 2 - 0.025, (hx0 + hx1) / 2 + 0.025, 0.10, 0.17, hz0, hz1, 'rail_metal')
        room(g, hx0, hx1, hz0, hz1, 0.9 if g.detail != 'far' else 0.4, o) if False else g.front(hx0, hx1, hz0, hz1, 0.45, 'room_wall')
    g.dims.update(windows=len(holes), win_w=ww)
    fascia(g, fas)


def k_shutter(g, o):
    W, H, bz, rng = g.W, g.H, g.bz, g.rng
    fas = o.get('fascia', rng.uniform(0.5, 0.8)); sw = o.get('shutter_w', rng.choice([1.8, 2.4, 3.0])); sh = o.get('shutter_h', rng.choice([2.0, 2.3, 2.5]))
    n = max(1, int(W // (sw + 0.8))); holes = []; open_i = o.get('open_index', n // 2 if o.get('open', True) else -1)
    g.dims.update(shutter_w=sw, shutter_h=sh, n_shutters=n, fascia=round(fas, 2), box_h=0.4)
    for i in range(n):
        cx = -W / 2 + W * (i + 0.5) / n; holes.append((cx - sw / 2, cx + sw / 2, bz, bz + sh))
    g.wall(-W / 2, W / 2, bz + 0.45, bz + H - fas, [(a, b, max(c, bz + 0.45), d) for a, b, c, d in holes], 'wall')
    g.wall(-W / 2, W / 2, bz, bz + 0.45, holes, 'wall_tile')
    for i, (hx0, hx1, hz0, hz1) in enumerate(holes):
        g.X(hx0 - 0.07, hx0, -0.01, 0.10, hz0, hz1 + 0.4, 'shutter_box'); g.X(hx1, hx1 + 0.07, -0.01, 0.10, hz0, hz1 + 0.4, 'shutter_box')   # ガイドレール
        g.X(hx0 - 0.07, hx1 + 0.07, -0.12, 0.10, hz1, hz1 + 0.4, 'shutter_box')                                              # 格納箱(0.3〜0.5m)
        top = hz1; low = hz0
        if i == open_i:
            low = hz1 - 1.0                      # 夜は一部だけ開いて照明が出る
            room(g, hx0, hx1, hz0, hz1, min(5.0, g.D - 0.4), o)
        g.front(hx0, hx1, low, top, 0.06, 'shutter'); g.X(hx0, hx1, 0.0, 0.10, low, low + 0.08, 'rail_metal')
    fascia(g, fas)


def k_entrance(g, o):
    W, H, bz, rng = g.W, g.H, g.bz, g.rng
    fas = o.get('fascia', rng.uniform(0.5, 0.8)); ew = o.get('ent_w', rng.uniform(2.4, 4.0)); ew = min(ew, W - 0.6); eh = min(o.get('ent_h', rng.uniform(2.5, 3.5)), H - fas - 0.4); co = o.get('canopy_out', rng.uniform(0.9, 1.5))
    r = o.get('recess', 1.5); cx = o.get('ent_x', 0.0); x0, x1 = cx - ew / 2, cx + ew / 2; hd = bz - g.gl
    g.dims.update(ent_w=round(ew, 2), ent_h=round(eh, 2), canopy_out=round(co, 2), recess=r, fascia=round(fas, 2)); g.gaps.append((x0, x1))
    g.wall(-W / 2, W / 2, bz, bz + H - fas, [(x0, x1, bz, bz + eh)], 'stone')
    g.reveal(x0, x1, bz, bz + eh, r, 'stone')
    steps(g, x0, x1, r, hd)
    # 風除室(二重ドア): 奥の自動ドア、手前に庇
    for dd in ((r - 0.9) if r > 1.2 else None, r):
        if dd is None:
            continue
        g.X(x0, x0 + 0.13, dd - 0.05, dd + 0.06, bz, bz + eh - 0.2, 'door_frame'); g.X(x1 - 0.13, x1, dd - 0.05, dd + 0.06, bz, bz + eh - 0.2, 'door_frame')
        g.X(x0, x1, dd - 0.05, dd + 0.06, bz + eh - 0.2, bz + eh - 0.05, 'door_frame'); g.front(x0 + 0.13, x1 - 0.13, bz + 0.03, bz + eh - 0.2, dd, 'glass')
        g.X(cx - 0.05, cx + 0.05, dd - 0.05, dd + 0.06, bz, bz + eh - 0.2, 'door_frame')
    room(g, x0 + 0.01, x1 - 0.01, bz, bz + eh - 0.05, min(r + 4.5, g.D - 0.3), o, d0=r, fl='entry_floor') if g.detail != 'far' else g.front(x0, x1, bz, bz + eh, r + 0.5, 'room_wall')
    g.X(x0 - 0.5, x1 + 0.5, -co, 0.05, bz + eh + 0.05, bz + eh + 0.25, 'canopy')                 # 庇(薄い金属の板、出0.9〜1.5m)
    if g.detail != 'far':
        nl = 4
        for i in range(nl):
            for j in range(2):
                px = x0 + (x1 - x0) * (i + 0.5) / nl; dd = -co * (0.35 + 0.35 * j)
                g.flat(px - 0.09, px + 0.09, dd - 0.09 , dd + 0.09, bz + eh + 0.049, 'downlight', False)
    g.X(-W / 2, W / 2, -0.03, 0.02, bz + 0.0, bz + 0.0 + 0.0001, 'stone') if False else None
    fascia(g, fas)


def k_parking(g, o):
    W, H, bz, rng = g.W, g.H, g.bz, g.rng
    fas = o.get('fascia', rng.uniform(0.5, 0.8)); pw = min(o.get('park_w', rng.uniform(2.5, 6.0)), W - 0.6); ph = o.get('park_h', rng.uniform(2.2, 2.8)); cx = o.get('park_x', 0.0)
    x0, x1 = cx - pw / 2, cx + pw / 2; g.gaps.append((x0, x1)); g.dims.update(park_w=round(pw, 2), park_h=round(ph, 2), fascia=round(fas, 2))
    g.wall(-W / 2, W / 2, bz, bz + H - fas, [(x0, x1, g.gl, g.gl + ph)], 'wall')
    zb = g.gl; zt = g.gl + ph; rd = min(6.5, g.D - 0.3)
    g.reveal(x0, x1, zb, zt, 0.25, 'wall')
    g.front(x0, x1, zb, zt, rd, 'dark_interior')
    if g.detail != 'far':
        g.side(x0, 0.25, rd, zb, zt, 'dark_interior', +1); g.side(x1, 0.25, rd, zb, zt, 'dark_interior', -1)
        g.flat(x0, x1, 0.0, rd, zb + 0.0005, 'parking_floor', True); g.flat(x0, x1, 0.25, rd, zt, 'dark_interior', False)
        for k in range(3):
            dd = 1.0 + 2.0 * k
            g.X(x0 + 0.2, x1 - 0.2, dd, dd + 0.18, zt - 0.22, zt - 0.003, 'dark_interior', skip=('+z',))
            g.X(x0 + 0.5, x0 + 1.1, dd + 0.3, dd + 1.1, zt - 0.05, zt - 0.003, 'downlight', skip=('+z',))
        for xx in (x0 + 0.35, x1 - 0.35):
            tube(g.md, np.array([xx, g.yf + 0.3, zt - 0.45]), np.array([xx, g.yf + rd, zt - 0.45]), 0.05, 0.05, 6, 'ceiling_pipe', caps=False)
    g.X(x0 - 0.05, x1 + 0.05, -0.03, 0.04, zt, zt + 0.12, 'rail_metal')
    fascia(g, fas)


BUILD = {'glass': k_glass, 'tile': k_tile, 'shutter': k_shutter, 'entrance': k_entrance, 'parking': k_parking}


def build_geometry(kind, width, height, seed=0, detail='near', depth=10.0, base_z=0.28, gl=0.05, night=True, **opts):
    kind = KINDS[kind]
    rng = random.Random(int(seed) * 104729 + 7)
    g = G(float(width), float(depth), float(height), float(base_z), float(gl), rng, night, detail)
    g.dims['kind'] = kind
    BUILD[kind](g, opts)
    wcol = rng.randrange(len(WALLS)); lcol = rng.randrange(len(ROOMLIGHT))
    info = dict(kind=kind, width=g.W, depth=g.D, height=g.H, base_z=g.bz, top_z=g.bz + g.H, gl=g.gl, front_y=g.yf, gaps_x=[(round(a, 3), round(b, 3)) for a, b in g.gaps],
                faces=g.md.nfaces(), detail=detail, seed=seed, dims=g.dims, wall_col=wcol, light_col=lcol, rail_kind=g.dims.get('rail_kind', 'silver'), fascia_col=g.dims.get('fascia_color', 0))
    return g.md, info


# ------------------------------------------------------------------------------------------------
# 材質
# ------------------------------------------------------------------------------------------------
def build_material(role, info, night, name):
    d = info['dims']
    S = SIMPLE[role]; mat, nt, out = new_material(name); nb = NB(nt); bz = info['base_z']
    tc = nb.node('ShaderNodeTexCoord'); ox, oy, oz = nb.sep(tc.outputs['Object']); ux, uy = None, None
    uvx, uvy, _ = nb.sep(tc.outputs['UV'])
    obj = tc.outputs['Object']
    zrel = nb.m('SUBTRACT', oz, bz)
    base = S['c']
    if role == 'wall':
        base = WALLS[info['wall_col']]
    elif role == 'fascia':
        base = FASCIA[info['fascia_col']]
    elif role in ('rail_metal', 'door_frame') and role == 'rail_metal':
        base = RAILS[info['rail_kind']]['c']
    # 夜の室内灯
    if role in ('room_wall', 'room_light', 'downlight'):
        lc = ROOMLIGHT[info['light_col']]
        es = {'room_wall': (2.6 if night else 0.7), 'room_light': (14.0 if night else 4.0), 'downlight': (60.0 if night else 6.0)}[role]
        b = principled(nt, **{'Roughness': 0.9})
        b.inputs['Emission Color'].default_value = (*lc, 1.0)
        if role == 'room_wall':
            band = nb.m('ADD', 0.55, nb.m('MULTIPLY', nb.sstep(nb.m('ABSOLUTE', nb.m('SINE', nb.m('MULTIPLY', ox, 2.2))), 0.0, 0.6), 0.45))
            nz = nb.noise(obj, 3.0)
            vscale = nb.m('MULTIPLY', band, nb.m('ADD', 0.6, nb.m('MULTIPLY', nz, 0.8)))
            col = nb.vscale(nb.rgb(lc), vscale); nt.links.new(col, b.inputs['Emission Color'])
            b.inputs['Base Color'].default_value = (0.4, 0.38, 0.33, 1)
        else:
            b.inputs['Base Color'].default_value = (0.9, 0.9, 0.9, 1)
        b.inputs['Emission Strength'].default_value = es
        return finish(mat, nt, out, b.outputs[0], role)
    if role == 'glass':
        # 夜も昼も、奥の店内が見える薄いガラス: 影を作る光線(Is Shadow Ray)は素通し。見た目は浅い反射。
        lp = nb.node('ShaderNodeLightPath'); lw = nb.node('ShaderNodeLayerWeight'); lw.inputs['Blend'].default_value = 0.3
        fr = nb.m('MULTIPLY', nb.m('MULTIPLY', lw.outputs['Fresnel'], 1.0), nb.m('SUBTRACT', 1.0, lp.outputs['Is Shadow Ray']))
        fr = nb.m('MAXIMUM', fr, 0.04)
        tr = nt.nodes.new('ShaderNodeBsdfTransparent'); gl_ = nt.nodes.new('ShaderNodeBsdfGlossy'); gl_.inputs['Roughness'].default_value = 0.03; gl_.inputs['Color'].default_value = (0.7, 0.8, 0.85, 1)
        mx = nt.nodes.new('ShaderNodeMixShader'); nt.links.new(fr, mx.inputs['Fac']); nt.links.new(tr.outputs[0], mx.inputs[1]); nt.links.new(gl_.outputs[0], mx.inputs[2])
        return finish(mat, nt, out, mx.outputs[0], 'glass')
    if role in ('stainless', 'handrail', 'rail_metal', 'kick', 'door_frame', 'canopy', 'shutter_box', 'ceiling_pipe'):
        c = RAILS[info['rail_kind']]['c'] if role == 'rail_metal' else base
        hair = nb.noise(nb.vec(nb.m('MULTIPLY', ox, 0.6), nb.m('MULTIPLY', oy, 0.6), nb.m('MULTIPLY', oz, 120.0)), 8.0, 2.0)  # ヘアライン
        col = nb.vscale(nb.rgb(c), nb.m('ADD', 0.85, nb.m('MULTIPLY', hair, 0.3)))
        # 足元(下 0.5m)の泥はね・汚れ
        dirt = nb.m('MULTIPLY', nb.sstep(zrel, 0.5, 0.0), 0.35)
        col = nb.vmix(col, nb.rgb((0.05, 0.045, 0.04)), dirt)
        b = principled(nt, **{'Metallic': S.get('m', 1.0), 'Roughness': RAILS[info['rail_kind']]['r'] if role == 'rail_metal' else S['r']})
        nt.links.new(col, b.inputs['Base Color']); nt.links.new(nb.bump(hair, 0.15, 0.01), b.inputs['Normal'])
        return finish(mat, nt, out, b.outputs[0], role)
    # 壁・タイル・石・シャッター ほか(拡散)
    n1 = nb.noise(obj, 1.2, 3.0); n2 = nb.noise(obj, 30.0, 2.0)
    col = nb.vscale(nb.rgb(base), nb.m('MULTIPLY', nb.m('ADD', 0.85, nb.m('MULTIPLY', n1, 0.3)), nb.m('ADD', 0.92, nb.m('MULTIPLY', n2, 0.16))))
    bump_h = n2; bump_s = 0.2; rough = S['r']
    if role == 'wall_tile':
        br = nb.node('ShaderNodeTexBrick'); br.offset = 0.5
        nt.links.new(nb.vec(nb.m('MULTIPLY', ox, 1.0 / 0.227), nb.m('MULTIPLY', oz, 1.0 / 0.06), 0.0), br.inputs['Vector'])
        br.inputs['Scale'].default_value = 1.0; br.inputs['Mortar Size'].default_value = 0.03; br.inputs['Brick Width'].default_value = 1.0; br.inputs['Row Height'].default_value = 1.0   # 二丁掛 60x227
        br.inputs['Color1'].default_value = (0.20, 0.13, 0.095, 1); br.inputs['Color2'].default_value = (0.16, 0.10, 0.075, 1); br.inputs['Mortar'].default_value = (0.22, 0.21, 0.19, 1)
        col = nb.vscale(br.outputs['Color'], nb.m('ADD', 0.85, nb.m('MULTIPLY', n1, 0.3))); bump_h = br.outputs['Fac']; bump_s = 0.5
    elif role in ('stone', 'stone_cap'):
        vd, vc = nb.voronoi(obj, 80.0); cr, _, _ = nb.sep(vc)
        col = nb.vscale(col, nb.m('ADD', 0.75, nb.m('MULTIPLY', cr, 0.6)))
        gx = nb.m('MINIMUM', nb.m('MODULO', ox, 0.6), nb.m('SUBTRACT', 0.6, nb.m('MODULO', ox, 0.6))); gz = nb.m('MINIMUM', nb.m('MODULO', oz, 1.2), nb.m('SUBTRACT', 1.2, nb.m('MODULO', oz, 1.2)))
        joint = nb.m('MAXIMUM', nb.m('SUBTRACT', 1.0, nb.sstep(gx, 0.002, 0.005)), nb.m('SUBTRACT', 1.0, nb.sstep(gz, 0.002, 0.005)))
        col = nb.vscale(col, nb.m('SUBTRACT', 1.0, nb.m('MULTIPLY', joint, 0.7)))
        rough = 0.3
    elif role == 'shutter':
        slat = nb.m('ADD', 0.72, nb.m('MULTIPLY', nb.sstep(nb.m('ABSOLUTE', nb.m('SINE', nb.m('MULTIPLY', oz, math.pi / 0.09))), 0.0, 0.5), 0.28))   # スラット幅 約90mm
        col = nb.vscale(col, slat); bump_h = nb.m('ABSOLUTE', nb.m('SINE', nb.m('MULTIPLY', oz, math.pi / 0.09))); bump_s = 0.6
    elif role == 'fascia':
        col = nb.vscale(nb.rgb(base), nb.m('ADD', 0.92, nb.m('MULTIPLY', n1, 0.1)))
    elif role in ('entry_floor', 'room_floor', 'parking_floor', 'step', 'furniture', 'room_ceiling', 'dark_interior'):
        pass
    if role in ('wall', 'wall_tile', 'stone', 'step', 'wall'):
        # 足元の暗い帯(雨の跳ね返り・泥): 地面から0.3〜0.6m、明度-20〜35%、上端はぼかす(推)
        low = nb.m('MULTIPLY', nb.sstep(zrel, 0.55, 0.0), nb.m('ADD', 0.6, nb.m('MULTIPLY', n1, 0.7)))
        col = nb.vscale(col, nb.m('SUBTRACT', 1.0, nb.m('MULTIPLY', low, 0.35)))
        # 庇・窓台から下がる縦の雨だれ
        st = nb.noise(nb.vec(nb.m('MULTIPLY', ox, 7.0), 0.0, nb.m('MULTIPLY', oz, 0.25)), 3.0, 2.0)
        col = nb.vscale(col, nb.m('SUBTRACT', 1.0, nb.m('MULTIPLY', nb.sstep(st, 0.55, 0.75), 0.18)))
    b = principled(nt, **{'Roughness': rough, 'Metallic': S.get('m', 0.0)})
    nt.links.new(col, b.inputs['Base Color']); nt.links.new(nb.bump(bump_h, bump_s, 0.01), b.inputs['Normal'])
    return finish(mat, nt, out, b.outputs[0], role)


# ------------------------------------------------------------------------------------------------
# オブジェクト化
# ------------------------------------------------------------------------------------------------
def make_ground_floor(kind, width, height, seed=0, detail='near', depth=10.0, base_z=0.28, gl=0.05, night=True, hour=20.0, position=(0.0, 0.0), rotation=0.0,
                      name=None, collection=None, **opts):
    """ビルの1階を作る。 -> 親Empty(子 = メッシュ)。 width x depth = ビルの底面、position = その中心(x, y)。 base_z = 1階の床(street_kit の info['base_z'])、gl = 歩道面の高さ。
    height = 1階の高さ(床から2階の床の下まで。推 3.5〜4.5)。 opts: fascia(看板帯の高さ)・stall(腰の高さ)・rail_w・mull_pitch・door・wain_h・shutter_w/h・ent_w/h・canopy_out・park_w/h など。"""
    t0 = time.time()
    md, info = build_geometry(kind, width, height, seed, detail, depth, base_z, gl, night, **opts)
    nm = name or 'floor1_%s_s%d_%s' % (info['kind'], int(seed), detail)
    mats = [build_material(r, info, night, '%s_%s' % (nm, r)) for r in md.roles]
    me = bpy.data.meshes.new(nm + '_mesh'); me.from_pydata(md.V, [], md.F)
    uvl = me.uv_layers.new(name='UVMap'); uvl.data.foreach_set('uv', np.asarray(md.UV, np.float32).ravel())
    me.polygons.foreach_set('material_index', np.asarray(md.MI, np.int32))
    for m in mats:
        me.materials.append(m)
    me.update()
    body = bpy.data.objects.new(nm + '_body', me); root = bpy.data.objects.new(nm, None); root.empty_display_size = 1.5
    coll = collection or bpy.context.scene.collection
    coll.objects.link(root); coll.objects.link(body); body.parent = root
    root.location = (position[0], position[1], 0.0); root.rotation_euler = (0, 0, rotation)
    info['gen_seconds'] = round(time.time() - t0, 3)
    root['kit_info'] = json.dumps(info, ensure_ascii=False, default=float)
    return root


def make_ground_floor_row(spec, height, depth, base_z, gl, seed=0, detail='near', night=True, hour=20.0, position=(0.0, 0.0), name='floor1_row', collection=None):
    """1階を種類ごとに横に並べる。 spec = [('glass', 12.0), ('entrance', 5.0), ('shutter', 6.0)]。 position = 全体の中心(x, y)。 -> 親Empty。 info['gaps_x'] は全体の中心基準の x"""
    coll = collection or bpy.context.scene.collection
    root = bpy.data.objects.new(name, None); coll.objects.link(root); root.location = (position[0], position[1], 0.0)
    total = sum(w for _, w in spec); x = -total / 2; gaps = []; gk = []; faces = 0
    for i, (k, w) in enumerate(spec):
        c = make_ground_floor(k, w, height, seed + i * 17, detail, depth, base_z, gl, night, hour, position=(x + w / 2, 0.0), name='%s_%d' % (name, i), collection=coll)
        c.parent = root; inf = floor_info(c); faces += inf['faces']
        gaps += [(round(a + x + w / 2, 3), round(b + x + w / 2, 3)) for a, b in inf['gaps_x']]; gk += [inf['kind']] * len(inf['gaps_x'])
        x += w
    root['kit_info'] = json.dumps(dict(width=total, depth=depth, base_z=base_z, top_z=base_z + height, gaps_x=gaps, gaps_kind=gk, faces=faces, kinds=[k for k, _ in spec]), default=float)
    return root


def floor_info(root):
    return json.loads(root['kit_info'])


def simplify_materials(root):
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
                b.inputs['Base Color'].default_value = (*s['c'], 1.0); b.inputs['Roughness'].default_value = s.get('r', 0.5); b.inputs['Metallic'].default_value = s.get('m', 0.0)
                if 'e' in s:
                    b.inputs['Emission Color'].default_value = (*s['e'], 1.0); b.inputs['Emission Strength'].default_value = s.get('es', 1.0)
                if s.get('a') is not None:
                    b.inputs['Alpha'].default_value = s['a']
                    for attr_, val in (('surface_render_method', 'BLENDED'), ('blend_method', 'BLEND')):
                        try:
                            setattr(nm, attr_, val)
                        except Exception:
                            pass
                cache[key] = nm
            slot.material = cache[key]


export_glb = sk.export_glb
