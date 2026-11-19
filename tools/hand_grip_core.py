"""Meshy製グローブの手(hand_open.glb)に拳銃を握らせる計算部分。numpyだけ(bpyなし)なので、ローカルでも試せる。
調査: リサーチ/2026-11-19_拳銃の握り方_現実の挙動.md (★=出典あり、推=推定)。仕様を探索の評価に落としてある。
座標: メッシュ(glTF) 指=+Y、親指=-X側、手のひら=-Z、手の甲=+Z。 手の座標(MHで変換) 手のひら=-X、親指側=+Z、指=+Y。
銃の座標: x=右、y=銃口、z=上、原点=グリップ中心。 銃メッシュ(gun.glb)の寸法(m):
  バックストラップ: z=0.035〜0.045 で y=-0.013〜-0.021 (高い位置)、ビーバーテールの先は z=0.05〜0.055, y=-0.030
  グリップ前面: z=0.005 で y=0.033、z=0.01 で y=0.036。 引き金ガードの底(付け根)は z=0.014、y=0.045〜0.088
  引き金: 板は x=±0.004、前面は y≈0.058〜0.065、z=0.02〜0.042。 ガードの板は x=±0.007。 フレーム側面 x=±0.015、スライド下縁 z≈0.055
"""
import math, json
import numpy as np

C0 = np.array([0.0145, 0.0, 0.0])
FINGERS = {   # mcp(付け根の関節), end(先端の中心線の終点), 半径
    'Index': dict(mcp=(-0.025, 0.172, 0.016), end=(-0.025, 0.258, 0.002), r=0.0125),
    'Middle': dict(mcp=(0.0145, 0.177, 0.018), end=(0.0145, 0.269, 0.002), r=0.0130),
    'Ring': dict(mcp=(0.050, 0.172, 0.014), end=(0.050, 0.258, -0.004), r=0.0125),
    'Pinky': dict(mcp=(0.070, 0.165, 0.010), end=(0.078, 0.229, -0.012), r=0.0110),
}
RATIO = (0.43, 0.30, 0.27)
THUMB_J = [(-0.030, 0.085, 0.000), (-0.052, 0.125, -0.012), (-0.068, 0.160, -0.022), (-0.080, 0.180, -0.027)]
THUMB_R = 0.016
HAND_HEAD = (0.0145, 0.03, 0.005); HAND_TAIL = (0.0170, 0.17, 0.012)
MH = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1.0]])      # メッシュ座標 -> 手の座標
WEB_M = (-0.036, 0.135, -0.027)        # 親指の股(親指と人さし指の間の谷)の手のひら側の皮膚。メッシュ座標(ずらす前)。 outline の分析で特定
CHAINS = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
BONES = ['Hand'] + [c + str(i) for c in CHAINS for i in (1, 2, 3)]
PADR = 0.0015      # 手袋の厚みぶん、指の半径に足す(メッシュの実測の貫通に合わせた)
TAPER = np.repeat([1.0, 0.95, 0.88], 4)     # サンプル12点ごとの半径の倍率(基節・中節・末節)

# ---- 仕様(銃の座標)
GUARD_BASE_Z = 0.014          # 引き金ガードの付け根(底)
TRIG_FRONT = np.array([0.0, 0.064, 0.028])   # 発砲中、人さし指の腹が当たる引き金の前面
WEB_TARGET_Z = 0.045          # 親指の股を当てる高さ(バックストラップの高い位置、ビーバーテールの下)
THUMB_SIDE_X = -0.031         # 親指の中心線(フレーム左側面 x=-0.015 の外)
THUMB_T2 = np.array([THUMB_SIDE_X, -0.012, 0.046]); THUMB_T3 = np.array([THUMB_SIDE_X, 0.014, 0.046])
RANGE = {'MCP': (45, 70), 'PIP': (60, 90), 'DIP': (20, 40)}      # 推定。範囲として使い、貫通しない範囲で外れてよい
INDEX_FIRE_NOMINAL = (20, 45, 20)


def sh(p):
    return np.asarray(p, float) - C0


def chain_joints(f):
    a = sh(f['mcp']); b = sh(f['end']); d = b - a
    pts = [a]; acc = 0.0
    for r in RATIO:
        acc += r; pts.append(a + d * acc)
    return pts


JOINTS = {n: chain_joints(f) for n, f in FINGERS.items()}
JOINTS['Thumb'] = [sh(p) for p in THUMB_J]
RAD = {n: f['r'] for n, f in FINGERS.items()}; RAD['Thumb'] = THUMB_R
WEB = sh(WEB_M)


# ---------------------------------------------------------------- 回転
def rotvec(v):
    v = np.asarray(v, float); a = np.linalg.norm(v)
    if a < 1e-9:
        return np.eye(3)
    k = v / a; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def swing3(a, b):
    a = np.asarray(a, float) / np.linalg.norm(a); b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b); sn = np.linalg.norm(v)
    if sn < 1e-9:
        return np.eye(3)
    return rotvec(v / sn * math.atan2(sn, c))


def Rx(a):
    return rotvec((a, 0, 0))


def hand_matrix(pos, yaw, pitch, roll):
    """メッシュ(ずらし後) -> 銃の座標。 手首: yaw=銃の z まわり、pitch=銃の x まわり、roll=手の指の軸まわり(度)"""
    def R4(R3):
        M = np.eye(4); M[:3, :3] = R3; return M
    T = np.eye(4); T[:3, 3] = pos
    Rz_ = R4(rotvec((0, 0, math.radians(yaw)))); Rx_ = R4(rotvec((math.radians(pitch), 0, 0))); Ry_ = R4(rotvec((0, math.radians(roll), 0)))
    return T @ Rz_ @ Rx_ @ Ry_ @ MH


# ---------------------------------------------------------------- 重み
def smooth(t):
    t = min(max(t, 0.0), 1.0); return t * t * (3 - 2 * t)


def seg_dist(P, a, b):
    ab = b - a; L = np.linalg.norm(ab); u = ab / L
    t = np.clip((P - a) @ u, 0, L)
    return np.linalg.norm(P - (a + np.outer(t, u)), axis=1), t, L


WEIGHT_VARIANTS = {
    # thumb_bw=関節まわりのつなぎ幅の半分、thumb_extra=親指の影響半径に足す値、base_bw=親指の付け根(手の骨→Thumb1)のつなぎ幅の半分、s_cut=付け根の手前のどこまで親指に含めるか
    'w0': dict(thumb_bw=0.020, thumb_extra=0.018, base_bw=0.020, s_cut=-0.012),
    'w1': dict(thumb_bw=0.030, thumb_extra=0.024, base_bw=0.040, s_cut=-0.030),      # 広げると、袖口にとげ状のゆがみが出た(h12で却下)
    'w2': dict(thumb_bw=0.035, thumb_extra=0.030, base_bw=0.060, s_cut=-0.050),      # 同上(さらに悪い)
    'w3': dict(thumb_bw=0.014, thumb_extra=0.014, base_bw=0.012, s_cut=0.000),       # 狭く・鋭く
    'w4': dict(thumb_bw=0.025, thumb_extra=0.018, base_bw=0.030, s_cut=-0.012),      # 付け根だけやや広く(範囲は同じ)
}


def compute_weights(P, wp=None):
    """P = 頂点(ずらし後) -> W (N, 骨数), owner, best。 いちばん近い指の中心線に属する頂点は、その指の骨(関節の近くでなめらかにつなぐ)。 ほかは Hand"""
    wp = wp or WEIGHT_VARIANTS['w0']
    N = len(P); W = np.zeros((N, len(BONES))); bi = {b: i for i, b in enumerate(BONES)}
    best = np.full(N, 1e9); owner = np.full(N, -1); sarr = np.zeros(N)
    for ci, c in enumerate(CHAINS):
        J = JOINTS[c]; cum = 0.0; dmin = np.full(N, 1e9); smin = np.zeros(N)
        for k in range(3):
            d, t, L = seg_dist(P, J[k], J[k + 1])
            m = d < dmin; dmin = np.where(m, d, dmin); smin = np.where(m, cum + t, smin); cum += L
        m = dmin < best; best = np.where(m, dmin, best); owner = np.where(m, ci, owner); sarr = np.where(m, smin, sarr)
    own_final = np.full(N, -1)
    for i in range(N):
        W[i, 0] = 1.0
        ci = owner[i]; c = CHAINS[ci]; th = (c == 'Thumb')
        R = RAD[c] + (wp['thumb_extra'] if th else 0.013)
        s = sarr[i]
        if best[i] > R or s < (wp['s_cut'] if th else -0.012):
            continue
        own_final[i] = ci
        J = JOINTS[c]; l1 = np.linalg.norm(J[1] - J[0]); l2 = np.linalg.norm(J[2] - J[1])
        bw = wp['thumb_bw'] if th else 0.008
        b0 = wp['base_bw'] if th else bw
        t0 = smooth((s - 0.0) / (2 * b0) + 0.5); t1 = smooth((s - l1) / (2 * bw) + 0.5); t2 = smooth((s - l1 - l2) / (2 * bw) + 0.5)
        W[i, 0] = 1 - t0
        W[i, bi[c + '1']] = t0 * (1 - t1); W[i, bi[c + '2']] = t1 * (1 - t2); W[i, bi[c + '3']] = t2
    return W, own_final, best


# ---------------------------------------------------------------- 銃までの距離(格子)
class SDF:
    """銃メッシュまでの符号つき距離(外が正)の格子。 trilinear補間。 範囲外は 0.04"""
    def __init__(self, org, step, val):
        self.org = np.asarray(org, float); self.step = float(step); self.val = val.astype(np.float32); self.n = np.array(val.shape)

    @staticmethod
    def build(org, step, shape, dist_fn):
        org = np.asarray(org, float)
        ii = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing='ij'), -1).reshape(-1, 3)
        pts = org + ii * step
        v = np.asarray(dist_fn(pts), float).reshape(shape)
        return SDF(org, step, v)

    def __call__(self, P):
        P = np.asarray(P, float); sh_ = P.shape[:-1]; f = P.reshape(-1, 3)
        g = (f - self.org) / self.step; i0 = np.floor(g).astype(int); fr = g - i0
        ok = np.all((i0 >= 0) & (i0 <= self.n - 2), axis=1)
        i0 = np.clip(i0, 0, self.n - 2); V = self.val
        x, y, z = i0[:, 0], i0[:, 1], i0[:, 2]; fx, fy, fz = fr[:, 0], fr[:, 1], fr[:, 2]
        c00 = V[x, y, z] * (1 - fx) + V[x + 1, y, z] * fx; c10 = V[x, y + 1, z] * (1 - fx) + V[x + 1, y + 1, z] * fx
        c01 = V[x, y, z + 1] * (1 - fx) + V[x + 1, y, z + 1] * fx; c11 = V[x, y + 1, z + 1] * (1 - fx) + V[x + 1, y + 1, z + 1] * fx
        c0 = c00 * (1 - fy) + c10 * fy; c1 = c01 * (1 - fy) + c11 * fy
        out = c0 * (1 - fz) + c1 * fz
        return np.where(ok, out, 0.04).reshape(sh_)


SDF_ORG = (-0.07, -0.06, -0.07); SDF_STEP = 0.0025; SDF_SHAPE = (57, 77, 69)


# ---------------------------------------------------------------- 指の順運動学(まとめて)
def finger_batch(c, angs, M):
    """angs (K,3) 度(MCP,PIP,DIP) -> 中心線の点 (K,12,3)(銃の座標)、 末節の腹の点 (K,3)、 先端の向き(K,3)"""
    angs = np.asarray(angs, float); K = len(angs); J = JOINTS[c]
    cum = np.zeros(K); p = np.tile(J[0], (K, 1)); out = []; pad = None; dlast = None
    for k in range(3):
        d0 = J[k + 1] - J[k]; L = np.linalg.norm(d0); d0 = d0 / L
        cum = cum + np.radians(angs[:, k]); a = -cum; ca, sa = np.cos(a), np.sin(a)
        d = np.stack([np.full(K, d0[0]), d0[1] * ca - d0[2] * sa, d0[1] * sa + d0[2] * ca], 1)
        for q in (0.25, 0.5, 0.75, 1.0):
            out.append(p + d * L * q)
        if k == 2:                                     # 腹 = 末節の中点 + 手のひら側の法線 * 半径
            n = np.stack([np.zeros(K), -np.sin(cum), -np.cos(cum)], 1)
            pad = p + d * L * 0.5 + n * RAD[c] * 0.85; dlast = d
        p = p + d * L
    pts = np.stack(out, 1)
    R3 = M[:3, :3]; t = M[:3, 3]
    return pts @ R3.T + t, pad @ R3.T + t, dlast @ R3.T


def make_grid(r1, r2, r3, step):
    a = np.arange(r1[0], r1[1] + 1e-9, step); b = np.arange(r2[0], r2[1] + 1e-9, step); c = np.arange(r3[0], r3[1] + 1e-9, step)
    return np.stack(np.meshgrid(a, b, c, indexing='ij'), -1).reshape(-1, 3)


def range_pen(angs):
    pen = 0.0
    for k, nm in enumerate(('MCP', 'PIP', 'DIP')):
        lo, hi = RANGE[nm]; pen = pen + np.maximum(lo - angs[:, k], 0) + np.maximum(angs[:, k] - hi, 0)
    return pen


def pen_term(g, w=2.5):
    return (np.maximum(-g - 0.0008, 0) * 1000) ** 2 * w


GRID_C = make_grid((20, 110), (20, 110), (0, 80), 10)      # 探索中(粗い)
GRID_F = make_grid((20, 120), (20, 120), (0, 80), 5)       # 最終(細かい)
GRID_IA = make_grid((0, 20), (0, 20), (0, 20), 10)         # 人さし指、構え
GRID_IB = make_grid((0, 50), (10, 90), (0, 50), 10)        # 人さし指、発砲中
GRID_IB_F = make_grid((0, 50), (10, 90), (0, 50), 5)
GRID_IA_F = make_grid((0, 20), (0, 20), (0, 20), 5)


def cost_wrap_finger(sdf, M, c, grid):
    """中指・薬指・小指: グリップ前面に巻く。 -> (コスト配列, 点)"""
    pts, pad, dl = finger_batch(c, grid, M)
    g = sdf(pts) - (RAD[c] + PADR) * TAPER[None, :]
    cost = pen_term(g).sum(1)
    gseg = np.maximum(g.reshape(len(grid), 3, 4).min(2), 0)
    wseg = np.array([1.0, 1.0, 0.7]) if c != 'Pinky' else np.array([0.0, 0.0, 0.0])      # 小指は浅く余ってよい(接触は求めない、貫通だけ避ける)
    cost += (gseg * wseg).sum(1) * 1000 * 0.35
    cost += range_pen(grid) * (0.04 if c != 'Pinky' else 0.08)
    if c == 'Pinky':                                   # 小指がグリップの下にはみ出す場合は、ふつうに曲げるだけ(浅く余ってよい)
        near = np.maximum(0.03 - np.minimum(g.min(1), 0.03), 0)
        return cost * 1.0
    tip = pts[:, -1, 0]
    cost += np.where(tip > -0.004, 0.8 + (tip + 0.004) * 20, 0.0)
    return cost


def cost_index_fire(sdf, M, grid):
    pts, pad, dl = finger_batch('Index', grid, M)
    g = sdf(pts) - (RAD['Index'] + PADR) * TAPER[None, :] * np.r_[np.ones(8), np.full(4, 0.85)][None, :]
    cost = pen_term(g).sum(1)
    dz = np.maximum(np.abs(pad[:, 2] - TRIG_FRONT[2]) - 0.009, 0)             # 引き金の板は z=0.02〜0.042 の幅がある
    d = np.sqrt((pad[:, 0] - TRIG_FRONT[0]) ** 2 + (pad[:, 1] - TRIG_FRONT[1]) ** 2 + dz ** 2) * 1000
    cost += (d / 2.5) ** 2
    cost += 0.02 * np.abs(grid - np.array(INDEX_FIRE_NOMINAL)[None, :]).sum(1)
    return cost, d


def cost_index_ready(sdf, M, grid):
    pts, pad, dl = finger_batch('Index', grid, M)
    g = sdf(pts) - (RAD['Index'] + PADR) * TAPER[None, :]
    cost = pen_term(g).sum(1)
    cost += 0.06 * grid.sum(1)                                              # 伸ばす
    gm = np.maximum(g[:, :8].min(1) - 0.003, 0) * 1000
    cost += gm * 0.25                                                       # フレームに沿わせる(浮かさない)
    cost += np.where(pts[:, -1, 1] < 0.085, (0.085 - pts[:, -1, 1]) * 1000 * 0.1, 0.0)   # 指先はガード前方まで
    return cost


WEB_TARGET = np.array([0.006, -0.022, WEB_TARGET_Z])   # バックストラップの高い位置(ビーバーテールの下)の面の、すぐ後ろ


def web_terms(sdf, M):
    w = (M @ np.r_[WEB, 1.0])[:3]; g = float(sdf(w[None, :])[0])
    d = float(np.linalg.norm(w - WEB_TARGET))
    c = (max(d - 0.008, 0) * 1000) ** 2 * 0.3 + (min(max(g, -0.002) - 0.003, 0) * 1000) ** 2 * 0.1 + (max(g - 0.008, 0) * 1000) ** 2 * 0.1
    return c, w, g


def hand_cost(sdf, x, ctx, detail=False):
    pos = x[:3]; yaw, pitch, roll = x[3:6]
    M = hand_matrix(pos, yaw, pitch, roll)
    q = ctx['palm'] @ M[:3, :3].T + M[:3, 3]; g = sdf(q)
    pen = (np.maximum(0.0005 - g, 0) * 1000) ** 1.5
    gs = np.sort(g); k = max(3, len(gs) // 8)
    cost = pen.sum() * 3 + float(np.mean(np.maximum(gs[:k] - 0.002, 0))) * 3000
    parts = {'palm': cost}
    cw, wpos, wg = web_terms(sdf, M); cost += cw; parts['web'] = cw
    mid = (M @ np.r_[JOINTS['Middle'][0], 1.0])[:3]
    cm = (max(abs(mid[2] - (GUARD_BASE_Z - RAD['Middle'] - 0.0025)) - 0.003, 0) * 1000) ** 2 * 1.0; cost += cm; parts['midz'] = cm
    best = {}
    for c in ('Middle', 'Ring', 'Pinky'):
        cs = cost_wrap_finger(sdf, M, c, ctx['grid']); i = int(np.argmin(cs)); cost += cs[i]; best[c] = ctx['grid'][i]; parts[c] = float(cs[i])
    cb, dd = cost_index_fire(sdf, M, ctx['gib']); i = int(np.argmin(cb)); cost += cb[i] * 1.0; best['IndexFire'] = ctx['gib'][i]; parts['IdxFire'] = float(cb[i]); parts['IdxFireDist_mm'] = float(dd[i])
    ca = cost_index_ready(sdf, M, ctx['gia']); i = int(np.argmin(ca)); cost += ca[i] * 0.5; best['IndexReady'] = ctx['gia'][i]; parts['IdxReady'] = float(ca[i])
    if detail:
        return cost, M, best, dict(web=wpos, web_gap=wg, mid_z=mid[2], parts=parts)
    return cost


BOUNDS = np.array([[0.005, 0.055], [-0.215, -0.09], [-0.045, 0.01], [-12, 12], [-10, 10], [-30, 30]])


def search_hand(sdf, P, owner, W, rng, n0=700, n_start=6, log=print):
    idx = np.nonzero((W[:, 0] > 0.9) & (P[:, 2] < -0.01))[0][::6]       # 手の骨にほぼ固定の、手のひら側の頂点
    ctx = dict(palm=P[idx], grid=GRID_C, gib=GRID_IB, gia=GRID_IA)
    X = BOUNDS[:, 0] + rng.random((n0, 6)) * (BOUNDS[:, 1] - BOUNDS[:, 0])
    cs = np.array([hand_cost(sdf, x, ctx) for x in X]); order = np.argsort(cs)[:n_start]
    log('hand search: random best %.2f' % cs[order[0]])
    results = []
    steps0 = np.array([0.004, 0.006, 0.004, 4, 3, 8], float)
    for si in order:
        x = X[si].copy(); c = cs[si]; steps = steps0.copy()
        for rnd in range(4):
            improved = True; it = 0
            while improved and it < 6:
                improved = False; it += 1
                for d in range(6):
                    for sg in (1, -1):
                        y = x.copy(); y[d] = np.clip(y[d] + sg * steps[d], BOUNDS[d, 0], BOUNDS[d, 1])
                        cy = hand_cost(sdf, y, ctx)
                        if cy < c - 1e-6:
                            x, c = y, cy; improved = True
            steps = steps * 0.5
        log('  start -> cost %.2f pos %s ang %s' % (c, np.round(x[:3], 4), np.round(x[3:], 1)))
        results.append((c, x))
    results.sort(key=lambda t: t[0])
    c, x = results[0]
    # 最終: 細かい格子
    ctx['grid'] = GRID_F; ctx['gib'] = GRID_IB_F; ctx['gia'] = GRID_IA_F
    cost, M, best, info = hand_cost(sdf, x, ctx, detail=True)
    return x, M, best, info, cost


# ---------------------------------------------------------------- 親指
def thumb_points(M, dirs):
    """dirs (K,3,3) 銃の座標の向き -> 中心線の関節 P(K,4,3) とサンプル(K,12,3)"""
    J = JOINTS['Thumb']; L = [np.linalg.norm(J[i + 1] - J[i]) for i in range(3)]
    K = len(dirs); p = np.tile((M @ np.r_[J[0], 1.0])[:3], (K, 1)); P = [p]; S = []
    for i in range(3):
        q = p + dirs[:, i, :] * L[i]
        for t in (0.25, 0.5, 0.75, 1.0):
            S.append(p + (q - p) * t)
        p = q; P.append(p)
    return np.stack(P, 1), np.stack(S, 1)


def thumb_cost(sdf, M, dirs, rest_d):
    P, S = thumb_points(M, dirs)
    rad = THUMB_R * np.repeat([0.95, 0.85, 0.75], 4)
    g = sdf(S) - rad[None, :]
    cost = pen_term(g, 3.0).sum(1)
    cost += np.linalg.norm(P[:, 2] - THUMB_T2[None, :], axis=1) * 1000 * 0.12 + np.linalg.norm(P[:, 3] - THUMB_T3[None, :], axis=1) * 1000 * 0.25
    gl = np.maximum(g[:, 4:].min(1) - 0.004, 0) * 1000; cost += gl * 0.1
    for i in range(2):
        ang = np.arccos(np.clip((dirs[:, i] * dirs[:, i + 1]).sum(1), -1, 1)); cost += np.maximum(ang - 0.5, 0) * 6       # 先端が丸まらない
    for i in range(3):
        ang = np.arccos(np.clip((dirs[:, i] * rest_d[i][None, :]).sum(1), -1, 1)); cost += np.maximum(ang - 1.9, 0) * 4 + ang * 0.04
    cost += (1 - dirs[:, 2, 1]) * 2.0                                                                               # 先端は前向き
    return cost


def fit_thumb(sdf, M, rng, log=print, n=60000):
    J = JOINTS['Thumb']; L = [np.linalg.norm(J[i + 1] - J[i]) for i in range(3)]
    rest_d = [(M[:3, :3] @ (J[i + 1] - J[i])) / L[i] for i in range(3)]

    def rnd(base, spread, K):
        v = base[:, :, :] + rng.normal(0, spread, base.shape); return v / np.linalg.norm(v, axis=2, keepdims=True)
    base = np.tile(np.array(rest_d)[None], (n, 1, 1)); D = rnd(base, 0.9, n)
    cs = thumb_cost(sdf, M, D, rest_d); order = np.argsort(cs)[:30]
    best = (cs[order[0]], D[order[0]].copy())
    for s0 in order[:6]:
        cur = D[s0].copy(); cc = cs[s0]; sp = 0.25
        for it in range(40):
            cand = rnd(np.tile(cur[None], (1500, 1, 1)), sp, 1500); cc_ = thumb_cost(sdf, M, cand, rest_d); j = int(np.argmin(cc_))
            if cc_[j] < cc:
                cc, cur = cc_[j], cand[j].copy()
            else:
                sp *= 0.8
        if cc < best[0]:
            best = (cc, cur)
    cost, dirs = best
    P, S = thumb_points(M, dirs[None])
    log('thumb cost %.2f tip %s (target %s)' % (cost, np.round(P[0, 3], 3), np.round(THUMB_T3, 3)))
    return P[0, 1:], dirs, cost


def thumb_D(M, wps):
    Mi = np.linalg.inv(M); J = JOINTS['Thumb']
    pts = [(Mi @ np.r_[np.asarray(w, float), 1.0])[:3] for w in wps]
    D = {}; Qp = np.eye(3); head = J[0].copy()
    for i in range(3):
        d0 = J[i + 1] - J[i]; L = np.linalg.norm(d0); d0 = d0 / L
        Q = swing3(d0, pts[i] - head)
        D['Thumb%d' % (i + 1)] = Qp.T @ Q; head = head + Q @ d0 * L; Qp = Q
    return D


def flex_D(curl):
    D = {}
    for c in ('Index', 'Middle', 'Ring', 'Pinky'):
        for i, a in enumerate(curl.get(c, (0, 0, 0))):
            D[c + str(i + 1)] = Rx(-math.radians(a))
    return D


# ---------------------------------------------------------------- メッシュ(スキニング後)の頂点を numpy で求める(貫通の確認・微調整用)
def bone_mats(D):
    """D = {骨: 3x3 (関節まわりの回転、メッシュ座標)} -> 骨ごとの 4x4 (静止 -> ポーズ)。 Hand は動かさない"""
    A = {'Hand': np.eye(4)}
    for c in CHAINS:
        Ap = np.eye(4)
        for i in range(3):
            nm = c + str(i + 1); h = JOINTS[c][i]; Dm = D.get(nm, np.eye(3))
            T = np.eye(4); T[:3, :3] = Dm; T[:3, 3] = h - Dm @ h
            Ap = Ap @ T; A[nm] = Ap
    return A


def skin(P, W, A, M):
    out = np.zeros_like(P)
    for bi, b in enumerate(BONES):
        w = W[:, bi]; m = w > 1e-6
        if m.any():
            out[m] += w[m, None] * (P[m] @ A[b][:3, :3].T + A[b][:3, 3])
    return out @ M[:3, :3].T + M[:3, 3]


def mesh_pen(sdf, P, W, owner, D, M):
    """-> {グループ: (貫通した頂点数, 最大の深さ mm)}、 全頂点の gap"""
    Q = skin(P, W, bone_mats(D), M); g = sdf(Q); res = {}
    for k in range(-1, 5):
        m = owner == k; gk = g[m]; n = int((gk < 0).sum())
        res['palm' if k < 0 else CHAINS[k]] = (n, float(-gk.min() * 1000) if n else 0.0)
    return res, g, Q


# ---------------------------------------------------------------- 実際のメッシュの貫通を使った微調整
class MeshEval:
    """銃の近くの頂点だけを取り出し、 ポーズ D ごとの貫通(0.4mm超の部分の二乗和)を求める"""
    def __init__(self, sdf, P, W, M, D0, cap=60000):
        Q = skin(P, W, bone_mats(D0), M); g = sdf(Q)
        keep = (g < 0.025) | (W[:, 0] < 0.999)          # 手の骨に固定で、銃から遠い頂点は動かない(除く)。 指・親指の頂点は全部入れる
        idx = np.nonzero(keep)[0]
        self.sdf = sdf; self.M = M; self.P = P[idx]; self.W = W[idx]

    def pen(self, D):
        Q = skin(self.P, self.W, bone_mats(D), self.M); g = self.sdf(Q); d = np.maximum(-g * 1000 - 0.4, 0)
        return float((d ** 2).sum() * 0.04), float(d.max() + 0.4 if d.max() > 0 else 0.0)


def _orig_cost(sdf, M, c, a, mode):
    g = np.array([a], float)
    if mode == 'wrap':
        return float(cost_wrap_finger(sdf, M, c, g)[0])
    if mode == 'fire':
        return float(cost_index_fire(sdf, M, g)[0][0])
    return float(cost_index_ready(sdf, M, g)[0])


def refine_finger(sdf, ev, M, curl, Dth, c, mode, log=print):
    a = np.array(curl[c], float); lo = np.array([0, 0, 0.0]); hi = np.array([120, 120, 90.0])

    def obj(v):
        cl = dict(curl); cl[c] = tuple(v); D = flex_D(cl); D.update(Dth)
        return _orig_cost(sdf, M, c, v, mode) + ev.pen(D)[0]
    f = obj(a)
    for step in (5.0, 2.5, 1.25):
        for sweep in range(6):
            imp = False
            for k in range(3):
                for sg in (step, -step):
                    b = a.copy(); b[k] = np.clip(b[k] + sg, lo[k], hi[k]); fb = obj(b)
                    if fb < f - 1e-6:
                        a, f = b, fb; imp = True
            if not imp:
                break
    return tuple(float(t) for t in a), f


def refine_thumb(sdf, ev_fn, M, curl, dirs, rest_d, rng, iters=22, ncand=120):
    def to_D(dr):
        P, S = thumb_points(M, dr[None]); return thumb_D(M, P[0, 1:]), P[0, 1:]

    def obj(dr):
        D, _ = to_D(dr); D2 = dict(flex_D(curl)); D2.update(D)
        return float(thumb_cost(sdf, M, dr[None], rest_d)[0]) + ev_fn.pen(D2)[0]
    cur = dirs.copy(); f = obj(cur); sp = 0.10
    for it in range(iters):
        imp = False
        for _ in range(ncand):
            c = cur + rng.normal(0, sp, cur.shape); c = c / np.linalg.norm(c, axis=1, keepdims=True); fc = obj(c)
            if fc < f - 1e-6:
                cur, f = c, fc; imp = True
        if not imp:
            sp *= 0.7
    D, wps = to_D(cur)
    return cur, wps, D, f


def solve(sdf, P, owner, W, rng, log=print):
    x, M, best, info, cost = search_hand(sdf, P, owner, W, rng, log=log)
    log('BEST cost %.2f pos %s yaw/pitch/roll %s' % (cost, np.round(x[:3], 4), np.round(x[3:], 2)))
    log('parts %s' % {k: round(float(v), 2) for k, v in info['parts'].items()})
    wps, tdirs, tcost = fit_thumb(sdf, M, rng, log=log)
    J = JOINTS['Thumb']; rest_d = [(M[:3, :3] @ (J[i + 1] - J[i])) / np.linalg.norm(J[i + 1] - J[i]) for i in range(3)]
    Dth = thumb_D(M, wps)
    curl = {c: tuple(float(t) for t in best[c]) for c in ('Middle', 'Ring', 'Pinky')}; curl['Index'] = tuple(float(t) for t in best['IndexFire'])
    ev = MeshEval(sdf, P, W, M, dict(flex_D(curl), **Dth))
    log('mesh pen before refine: %s' % (ev.pen(dict(flex_D(curl), **Dth)),))
    for sweep in range(2):
        for c in ('Middle', 'Ring', 'Pinky'):
            curl[c], f = refine_finger(sdf, ev, M, curl, Dth, c, 'wrap')
        tdirs, wps, Dth, f = refine_thumb(sdf, ev, M, curl, tdirs, rest_d, rng)
        log('refine sweep %d: curl %s thumb cost %.2f pen %s' % (sweep, {k: tuple(round(t) for t in v) for k, v in curl.items()}, f, ev.pen(dict(flex_D(curl), **Dth))))
    curl_f = dict(curl); curl_f['Index'], f = refine_finger(sdf, ev, M, curl, Dth, 'Index', 'fire')
    curl_r = dict(curl); curl_r['Index'] = tuple(float(t) for t in best['IndexReady'])
    ev_r = MeshEval(sdf, P, W, M, dict(flex_D(curl_r), **Dth))
    curl_r['Index'], f = refine_finger(sdf, ev_r, M, curl_r, Dth, 'Index', 'ready')
    return dict(x=x, M=M, wps=wps, Dth=Dth, curl={'fire': curl_f, 'ready': curl_r}, info=info, cost=cost, tdirs=tdirs)
