"""高速チェイスの動き。three.js版 scene.js の移植(純関数。bpyは使わない)。
座標は three.js のまま: x=右, y=上, z=手前(前進は -z)。Blender へは to_blender() で (x, -z, y)。"""
import math
import numpy as np

V = 46.0
FPS = 24
DUR = 30.0
TAU = math.pi * 2
CUTS = [2.6, 4.4, 6.0, 8.0, 9.6, 11.6, 13.0, 14.4, 16.0, 17.5, 19.0, 21.0, 23.4, 25.0]   # 15 shots


def sm(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def clamp(x, a, b):
    return min(b, max(a, x))


def mix(a, b, k):
    return a + (b - a) * k


def rng(seed):
    """mulberry32 (scene.js と同じ乱数列)"""
    s = [seed & 0xFFFFFFFF]

    def imul(a, b):
        return (a * b) & 0xFFFFFFFF

    def f():
        s[0] = (s[0] + 0x6D2B79F5) & 0xFFFFFFFF
        a = s[0]
        t = imul(a ^ (a >> 15), 1 | a)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return f


def keyed_sm(keys, t):
    if t <= keys[0][0]:
        return keys[0][1]
    for i in range(1, len(keys)):
        if t <= keys[i][0]:
            t0, v0 = keys[i - 1]; t1, v1 = keys[i]
            return v0 + (v1 - v0) * sm((t - t0) / (t1 - t0))
    return keys[-1][1]


def keyed_lin(keys, t):
    if t <= keys[0][0]:
        return keys[0][1]
    for i in range(1, len(keys)):
        if t <= keys[i][0]:
            t0, v0 = keys[i - 1]; t1, v1 = keys[i]
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return keys[-1][1]


# ---- the two bikes ---------------------------------------------------------------------------
LANE_A = [[0, 1.8], [2.2, 1.8], [3.4, -1.8], [5.5, -1.8], [6.7, -5.4], [8.2, -5.4], [9.6, -1.8], [11, -1.8], [12.2, 1.8], [13.6, 1.8],
          [14.8, 5.4], [17.4, 5.4], [18.6, 1.8], [23.0, 1.8], [24.0, 0.2], [24.8, 1.8], [30, 1.8]]
GAP = [[0, 38], [4.5, 31], [9, 23], [13, 18], [19.5, 9], [22, 2.2], [24, 0.4], [25, 0.4], [26.5, 9], [28, 15], [30, 20]]


def xA(t):
    return keyed_sm(LANE_A, t)


def zA(t):
    return -V * t - 30


def gap_at(t):
    return keyed_lin(GAP, t)


def near_miss(t):
    return math.exp(-(((t - 23.6) / 0.45) ** 2))


def xB(t):
    g = gap_at(t)
    d = xA(t - g / V)
    return mix(d, xA(t) - (3.6 - 2.0 * near_miss(t)), sm((t - 18) / 3))


def zB(t):
    return zA(t) + gap_at(t)


def vx(f, t):
    return (f(t + 0.05) - f(t - 0.05)) / 0.1


# ---- riding dynamics (2026-11-18 v4) -----------------------------------------------------------------
# 経路(車線変更)から「ハンドル(逆操舵) → 車体が傾く(リーン) → 向きが変わる(ヨー)」の順を作る。
#  ・リーン = 横加速度 ax から atan(ax/g) × LEAN_K。速度ではなく加速度に比例(車線変更の真ん中=直進で傾き0)。
#  ・リーンは経路より TAU 秒先に出す(傾けてから曲がる)。2次遅れでなめらかにし、傾く速さ(ロール角速度)に比例してハンドルを逆に切る。
#  ・サスペンション: 車輪は常に路面(z=r)。車体だけが相対的に上下する(ジャンプしない)。
#  ・ライダー: 胴体は車体より遅れて傾き(慣性)、頭は水平を保ち、進行方向を見る。
G = 9.81
RIDE_DT = 1.0 / 240
RIDE_T0, RIDE_T1 = -1.0, 31.0
WB = 1.4                # ホイールベース(m)
LEAN_K, LEAN_MAX, TAU = 0.50, 0.55, 0.38
W_LEAN = 7.0            # リーンの応答(rad/s)
K_COUNTER, K_STEADY = 0.12, 0.05     # 逆操舵(ロール角速度1 rad/s あたり何rad)、定常の切り足し
RHO_TIRE = 0.085        # タイヤの断面半径(傾くと接地点がずれて車輪の中心が下がる分)
W_SUS = 2 * math.pi * 2.0   # サスの固有振動(rad/s)
BUMP_AMP = 0.030        # 大きなバンプ(段差・補修跡)のとき、車体が車輪に対して動く大きさ(m)
NZ = {'A': {'ph': [0.3, 1.7, 4.1, 2.2, 5.0, 0.9], 'bumps': [3.7, 7.4, 12.9, 18.3, 22.1, 27.0]},
      'B': {'ph': [2.1, 0.4, 3.3, 5.5, 1.2, 4.4], 'bumps': [4.9, 9.2, 14.1, 19.4, 23.0, 26.2]}}
_RIDE = {}


def _sos(u, w, z, dt=RIDE_DT):
    """2次遅れ x'' + 2 z w x' + w^2 x = w^2 u  -> (x, dx/dt)"""
    x = np.empty_like(u); v = np.empty_like(u)
    xx = float(u[0]); vv = 0.0
    for i in range(len(u)):
        vv += (w * w * (u[i] - xx) - 2 * z * w * vv) * dt
        xx += vv * dt
        x[i] = xx; v[i] = vv
    return x, v


def _shift(a, sec):
    """a(t + sec) (端は端の値)"""
    n = int(round(sec / RIDE_DT))
    if n == 0:
        return a
    out = np.empty_like(a)
    if n > 0:
        out[:-n] = a[n:]; out[-n:] = a[-1]
    else:
        out[-n:] = a[:n]; out[:-n] = a[0]
    return out


def _ride_tables(bid):
    dt = RIDE_DT
    ts = np.arange(RIDE_T0, RIDE_T1, dt)
    fx = xA if bid == 'A' else xB
    ph = NZ[bid]['ph']
    drift = 0.030 * np.sin(TAU * 0.35 * ts + ph[0]) + 0.018 * np.sin(TAU * 0.61 * ts + ph[1])
    xp = np.array([fx(t) for t in ts]) + drift
    h = 5
    ax = (np.roll(xp, -h) - 2 * xp + np.roll(xp, h)) / (h * dt) ** 2
    ax[:h] = 0; ax[-h:] = 0
    u = np.clip(LEAN_K * np.arctan(ax / G), -LEAN_MAX, LEAN_MAX)
    lean, lean_rate = _sos(_shift(u, TAU), W_LEAN, 1.0)
    wob = 0.010 * np.sin(TAU * 0.9 * ts + ph[2]) + 0.006 * np.sin(TAU * 1.9 * ts + ph[3])
    wob_rate = TAU * 0.9 * 0.010 * np.cos(TAU * 0.9 * ts + ph[2]) + TAU * 1.9 * 0.006 * np.cos(TAU * 1.9 * ts + ph[3])
    phi = lean + wob                                  # +は右に傾く(ブレンダーの Y 軸まわり)
    phid = lean_rate + wob_rate
    steer = K_COUNTER * phid - K_STEADY * phi         # +は左に切る(逆操舵: 右に傾けたいとき先に左へ)
    vxp = (np.roll(xp, -h) - np.roll(xp, h)) / (2 * h * dt)
    heading = -np.arctan(vxp / V)                     # +は左向き(ブレンダーの Z 軸まわり)
    heading[:h] = 0; heading[-h:] = 0
    # 胴体: 車体より遅れて傾く(慣性)。絶対の傾きは車体より少し内側へ
    tor_abs, _ = _sos(1.10 * phi, 11.0, 0.75)
    rho = 0.14 * np.tanh((tor_abs - phi) / 0.14)      # 胴体が車体に対して動く量は最大 約8度まで
    head_abs, _ = _sos(0.30 * tor_abs, 10.0, 0.7)
    head_roll = 0.12 * np.tanh((head_abs - tor_abs) / 0.12)
    gaze, _ = _sos(_shift(heading, 0.45), 9.0, 0.8)
    head_yaw = 0.75 * (gaze - heading)
    # サスペンション: 路面の段差の列(30mごとの継手 + 大きなバンプ)に対する、車体と車輪の相対位置
    t_ev = [bt for bt in NZ[bid]['bumps']]
    uf = np.zeros_like(ts)
    for n in range(-2, 60):
        te = (30.0 * n + 7.0 * ph[4]) / V
        uf += (0.0045 + 0.0020 * math.sin(n * 2.3 + ph[0])) * np.exp(-(((ts - te) / 0.035) ** 2))
    for bt in t_ev:
        uf += BUMP_AMP * np.exp(-(((ts - bt) / 0.05) ** 2))
    ur = _shift(uf, -WB / V)
    out = {'ts': ts, 'x': xp, 'drift': drift, 'phi': phi, 'steer': steer, 'rho': rho, 'head_roll': head_roll, 'head_yaw': head_yaw}
    for k, uu in (('f', uf), ('r', ur)):
        fz, _ = _sos(uu, W_SUS, 0.36)
        out['rel_' + k] = fz - uu * 0.55           # 車体の上下(車輪に対して): 低域の応答 - 入力の一部(縮む)
    heave = 0.5 * (out['rel_f'] + out['rel_r'])
    out['heave'] = heave
    out['fpitch'] = (out['rel_f'] - out['rel_r']) / WB
    tz, _ = _sos(heave, 2 * math.pi * 1.7, 0.35)
    out['tor_dz'] = tz - heave
    tp, _ = _sos(out['fpitch'], 2 * math.pi * 1.7, 0.4)
    out['tor_pitch'] = tp - out['fpitch']
    return out


def ride(bid, t):
    tb = _RIDE.get(bid)
    if tb is None:
        tb = _RIDE[bid] = _ride_tables(bid)
    ts = tb['ts']
    r = {k: float(np.interp(t, ts, tb[k])) for k in ('x', 'drift', 'phi', 'steer', 'rho', 'head_roll', 'head_yaw', 'heave', 'fpitch', 'rel_f', 'rel_r', 'tor_dz', 'tor_pitch')}
    return r


def bike_noise(bid, t):
    """カメラが車体に付くカット用: 横ずれ、横速度、上下(=車体のばね上の動き)、ピッチ"""
    r = ride(bid, t)
    r2 = ride(bid, t + 0.04); r1 = ride(bid, t - 0.04)
    return {'dx': r['drift'], 'vlat': (r2['drift'] - r1['drift']) / 0.08, 'dy': r['heave'], 'pitch': r['fpitch']}


def bike_pose(bid, t):
    """-> dict: x, y, z (three.js), yaw (about +y), pitch (about +x), roll (about +z, three.js sign), dist (m travelled forward),
    steer (+=左), body_dz / body_pitch (車体のばね上の上下/ピッチ), torso_*, head_*"""
    fz = zA if bid == 'A' else zB
    z = fz(t)
    r = ride(bid, t)
    h = 0.05
    vxm = (ride(bid, t + h)['x'] - ride(bid, t - h)['x']) / (2 * h)
    phi = r['phi']
    return {'x': r['x'], 'y': RHO_TIRE * (1 - math.cos(phi)), 'z': z, 'yaw': -math.atan(vxm / V), 'pitch': 0.0, 'roll': -phi, 'dist': -z,
            'steer': r['steer'], 'body_dz': r['heave'], 'body_pitch': r['fpitch'], 'torso_roll': r['rho'], 'torso_dz': r['tor_dz'],
            'torso_pitch': r['tor_pitch'], 'head_roll': r['head_roll'], 'head_yaw': r['head_yaw']}


# ---- cameras: 15 shots, hard cuts ----------------------------------------------------------------
def shot_index(t):
    for i, c in enumerate(CUTS):
        if t < c:
            return 1 if i == 2 else i      # 4.4〜6.0秒の「Bの前輪アップ」はカット(ユーザー指示)。2カット目(Aの背後を低く)を6.0秒まで延ばす
    return len(CUTS)


def shot_at(t, k=None):
    """-> (p, l, fov_deg, roll). k を渡すと、そのカットの式で t を評価する(モーションブラー用に、カット境界をまたがない)。"""
    az, bz, ax, bx = zA(t), zB(t), xA(t), xB(t)
    mz, mx = (az + bz) / 2, (ax + bx) / 2
    vA, vB = vx(xA, t), vx(xB, t)
    if k is None:
        k = shot_index(t)
    roll = 0.0
    if k == 0:      # 1 fly-by: A flashes past, B right behind (whip pan)
        cz = zA(1.05); w = sm((t - 1.7) / 0.5)
        p = [-4.0, 0.5, cz]; l = [mix(ax, bx, w), 0.9, mix(az, bz, w)]; fov = 40
    elif k == 1:    # 2 low behind A, tilted
        p = [ax * 0.5, 0.85, az + 3.4]; l = [ax, 0.95, az - 6]; fov = 62; roll = clamp(-vA / V * 5, -0.2, 0.2) + 0.05
    elif k == 2:    # 3 B's front wheel, extreme close-up
        p = [bx + 1.1, 0.45, bz - 0.7]; l = [bx, 0.4, bz - 0.9]; fov = 55; roll = 0.1
    elif k == 3:    # 4 from A's tail, looking back at B   (Blender版: カメラを車体の外=尾部の後ろに出した)
        nA = bike_noise('A', t)
        p = [ax + nA['dx'], 1.05 + nA['dy'], az + 1.35]; l = [bx, 1.0, bz]; fov = 38
        roll = clamp(-(vA + nA['vlat']) / V * 3.4, -0.5, 0.5) * 0.8 + nA['pitch'] * 2
    elif k == 4:    # 5 close side tracking on A
        p = [ax + 4.5, 0.75, az + 0.5]; l = [ax, 1.0, az - 0.3]; fov = 36
    elif k == 5:    # 6 crane descent over the traffic
        p = [mx * 0.7, 20 - 10 * sm((t - CUTS[4]) / 2.0), mz + 12]; l = [mx, 0, mz - 4]; fov = 55
    elif k == 6:    # 7 head-on dolly-in, Dutch tilt
        d = 16 - 10 * sm((t - CUTS[5]) / 1.4)
        p = [ax * 0.5, 0.7, az - d]; l = [ax, 1.0, az + 2]; fov = 46; roll = 0.14
    elif k == 7:    # 8 chaser POV (ahead of B's screen)
        nB = bike_noise('B', t)
        p = [bx + nB['dx'], 1.55 + nB['dy'], bz - 1.05]; l = [ax, 1.0, az]; fov = 66
        roll = clamp(-(vB + nB['vlat']) / V * 3.4, -0.5, 0.5) * 0.8 + nB['pitch'] * 2
    elif k == 8:    # 9 A squeezes past the truck
        p = [ax + 5.2, 2.6, az + 6]; l = [ax, 0.9, az - 3]; fov = 46
    elif k == 9:    # 10 low behind B, A far ahead
        p = [bx * 0.8, 0.6, bz + 3.0]; l = [bx, 1.0, bz - 10]; fov = 58; roll = -0.08
    elif k == 10:   # 11 A's helmet, extreme close-up   (Blender版: ヘルメットの位置に合わせた)
        p = [ax + 1.5, 1.22, az - 1.3]; l = [ax, 1.2, az - 0.30]; fov = 30
    elif k == 11:   # 12 second fly-by, both bikes
        cz = zA(19.7)
        p = [-3.4, 0.5, cz]; l = [mx, 0.9, mz]; fov = 42
    elif k == 12:   # 13 side by side
        p = [mx + 11, 2.2, mz + 0.2]; l = [mx, 0.9, mz]; fov = 36
    elif k == 13:   # 14 head-on, the swerve (near miss)
        p = [mx, 0.6, mz - 8]; l = [mx, 0.9, mz + 1]; fov = 52; roll = -0.12
    else:           # 15 pull away
        mz25 = (zA(CUTS[13]) + zB(CUTS[13])) / 2
        cz = mz25 + 16 - 30 * (t - CUTS[13])
        p = [mx + 1.5, 3.3, cz]; l = [mx, 1.2, mz - 2]; fov = 26
    return p, l, fov, roll


# ---- traffic: every lane runs at ONE speed with spacing, so cars never pass through each other ----
DIM = {'car': [2.0, 4.4], 'truck': [2.6, 11.0], 'bike': [0.9, 2.4]}     # [width, length] used for the overlap checks (scene.js と同じ)
LANE_V = {-5.4: 25, -1.8: 31, 1.8: 27, 5.4: 33}
HERO = {'lane': 1.8, 'v': 27, 'tMeet': 15.9}
HERO['z0'] = zA(HERO['tMeet']) + HERO['v'] * HERO['tMeet']


def pos_of(c, t):
    return c['z0'] - c['v'] * t if c['dir'] < 0 else c['z0'] + c['v'] * t


def _hits_bike(c):
    wc, lc = DIM[c['kind']]
    t = 0.0
    while t <= 30:
        z = pos_of(c, t)
        for bx_, bz_ in ((xA(t), zA(t)), (xB(t), zB(t))):
            if abs(z - bz_) < (lc + DIM['bike'][1]) / 2 + 2.5 and abs(c['x'] - bx_) < (wc + DIM['bike'][0]) / 2 + 0.7:
                return True
        t += 0.05
    return False


def _hits_camera(c):
    wc, lc = DIM[c['kind']]
    t = 0.0
    while t <= 30:
        sp = shot_at(t)[0]
        if sp[1] < 3.5 and abs(pos_of(c, t) - sp[2]) < lc / 2 + 3 and abs(c['x'] - sp[0]) < wc / 2 + 1.5:
            return True
        t += 0.05
    return False


_FULL = {}


def _full_timeline(step=0.05, tmax=45.0):
    """追跡(0〜30秒)+続編(30〜45秒)の、主役2台(A・B)の位置とカメラ(位置・見る先)の時刻表。 周囲の車は、これを先に決めたうえで、後から重ならないように作る"""
    if 'tl' in _FULL:
        return _FULL['tl']
    import sequel as _SQ
    ts = np.arange(0.0, tmax + 1e-9, step)
    ax, az, bx, bz, cp, cl = [], [], [], [], [], []
    for t in ts:
        if t < 30.0:
            ax.append(xA(t)); az.append(zA(t)); bx.append(xB(t)); bz.append(zB(t)); p, l, _, _ = shot_at(t)
        else:
            ax.append(_SQ.xA(t)); az.append(_SQ.zA(t)); bx.append(_SQ.xB(t)); bz.append(_SQ.zB(t)); p, l, _, _ = _SQ.shot_at(t)
        cp.append(p); cl.append(l)
    tl = dict(ts=ts, A=(np.asarray(ax), np.asarray(az)), B=(np.asarray(bx), np.asarray(bz)), cp=np.asarray(cp, float), cl=np.asarray(cl, float))
    _FULL['tl'] = tl
    return tl


def _seg_hits_box(p, q, cx, cz, hx, hz):
    """線分 p→q(x,z 平面。配列)が、箱(中心 cx,cz・半幅 hx,hz。配列)に当たるか。 スラブ法"""
    d = q - p
    t0 = np.zeros(len(cx)); t1 = np.ones(len(cx))
    ok = np.ones(len(cx), bool)
    for k, (c, h) in enumerate(((cx, hx), (cz, hz))):
        lo = c - h - p[:, k]; hi = c + h - p[:, k]
        dk = d[:, k]
        small = np.abs(dk) < 1e-9
        with np.errstate(divide='ignore', invalid='ignore'):
            ta = np.where(small, -np.inf, lo / dk); tb = np.where(small, np.inf, hi / dk)
        tmin = np.minimum(ta, tb); tmax = np.maximum(ta, tb)
        ok &= ~(small & ((lo > 0) | (hi < 0)))
        t0 = np.maximum(t0, np.where(small, -np.inf, tmin)); t1 = np.minimum(t1, np.where(small, np.inf, tmax))
    return ok & (t0 <= t1)


def conflicts(c, why=False):
    """追跡+続編の全時間(0〜45秒)で、車 c が主役の邪魔になるか。 邪魔 = ①バイクA・Bに近すぎる ②カメラの箱に入る ③低いカメラと見る先の間に入って視界をさえぎる。 邪魔な車は、最初から作らない(途中で消さない)"""
    tl = _full_timeline()
    ts = tl['ts']
    wc, lc = DIM[c['kind']]
    z = c['z0'] - c['v'] * ts if c['dir'] < 0 else c['z0'] + c['v'] * ts
    x = np.full(len(ts), float(c['x']))
    for (bx, bz) in (tl['A'], tl['B']):
        if np.any((np.abs(z - bz) < (lc + 2.4) / 2 + 3.0) & (np.abs(x - bx) < (wc + 2.4) / 2 + 0.8)):
            return 'bike' if why else True
    cp, cl = tl['cp'], tl['cl']                      # three.js 座標 (x, y=高さ, z)
    low = cp[:, 1] < 4.5
    if np.any(low & (np.abs(z - cp[:, 2]) < lc / 2 + 4.0) & (np.abs(x - cp[:, 0]) < wc / 2 + 2.2)):
        return 'camera' if why else True
    seglen = np.hypot(cl[:, 0] - cp[:, 0], cl[:, 2] - cp[:, 2])
    sel = low & (seglen > 1.5)
    if sel.any():
        P = np.stack([cp[sel, 0], cp[sel, 2]], 1); Q = np.stack([cl[sel, 0], cl[sel, 2]], 1)
        if np.any(_seg_hits_box(P, Q, x[sel], z[sel], wc / 2 + 0.5, lc / 2 + 0.5)):
            return 'view' if why else True
    return None if why else False


def traffic():
    """-> list of dict(x, z0, v, dir, kind, hero, pal)  pal = 0..7 (色の番号)"""
    cR = rng(2024)
    cars = [{'x': HERO['lane'], 'z0': HERO['z0'], 'v': HERO['v'], 'dir': -1, 'kind': 'truck', 'hero': True, 'pal': 7}]
    for lane in (-5.4, -1.8, 1.8, 5.4):
        z = 170.0
        while z > -1250:
            kind = 'truck' if cR() < 0.2 else 'car'
            ln = DIM[kind][1]
            z -= ln / 2
            cand = {'x': lane, 'z0': z, 'v': LANE_V[lane], 'dir': -1, 'kind': kind, 'hero': False}
            near_hero = lane == HERO['lane'] and abs(z - HERO['z0']) < 22
            if not near_hero and not conflicts(cand):
                cand['pal'] = int(cR() * 8)
                cars.append(cand)
            z -= ln / 2 + 12 + cR() * 55
    for lane in (-14.5, -18.5):
        z = -3900.0                                            # 45秒ぶん走っても(v=32 → 1440m)、道の全域に対向車が居る
        while z < 140:
            kind = 'truck' if cR() < 0.15 else 'car'
            ln = DIM[kind][1]
            z += ln / 2
            pal = int(cR() * 8)
            cand = {'x': lane, 'z0': z, 'v': 32, 'dir': 1, 'kind': kind, 'hero': False, 'pal': pal}
            if not conflicts(cand):
                cars.append(cand)
            z += ln / 2 + 20 + cR() * 90
    return cars


def overlaps(cars):
    """自己検査: 車両どうし(車・トラック・バイク)が重なる組の数。0 が正しい。"""
    ents = [{'n': c['kind'], 'x': (lambda t, c=c: c['x']), 'z': (lambda t, c=c: pos_of(c, t)), 'd': DIM[c['kind']]} for c in cars]
    ents.append({'n': 'A', 'x': xA, 'z': zA, 'd': DIM['bike']})
    ents.append({'n': 'B', 'x': xB, 'z': zB, 'd': DIM['bike']})
    out = []
    ts = [i * 0.05 for i in range(601)]
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            a, b = ents[i], ents[j]
            if a['n'] not in 'AB' and b['n'] not in 'AB' and abs(a['x'](0) - b['x'](0)) > 3:
                continue
            for t in ts:
                dx = abs(a['x'](t) - b['x'](t)) - (a['d'][0] + b['d'][0]) / 2
                dz = abs(a['z'](t) - b['z'](t)) - (a['d'][1] + b['d'][1]) / 2
                if dx < 0 and dz < 0:
                    out.append((a['n'], b['n'], round(t, 2)))
                    break
    return out


def to_blender(p):
    return (p[0], -p[2], p[1])


if __name__ == '__main__':
    cs = traffic()
    print('vehicles', len(cs), 'same-direction', sum(1 for c in cs if c['dir'] < 0), 'trucks', sum(1 for c in cs if c['kind'] == 'truck'))
    print('overlapping pairs', len(overlaps(cs)))
    print('hero z0', HERO['z0'])
    for t in (0, 1.0, 5, 10, 15.9, 23.6, 29.9):
        p, l, fov, roll = shot_at(t)
        print('t=%.1f shot %d  A(%.2f, %.1f)  B(%.2f, %.1f) gap %.1f  cam %s fov %s roll %.3f' % (t, shot_index(t) + 1, xA(t), zA(t), xB(t), zB(t), gap_at(t), [round(v, 2) for v in p], fov, roll))
