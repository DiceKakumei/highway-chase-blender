"""続編「打ち合い」(15秒)の動き。純関数(bpy を使わない)。motion.py と同じ座標(three.js: x=右, y=上, z=手前。前進は -z)。
時刻: 世界の時刻 t = TOFF + τ。τ=0 は前作の暗転の後ろ(t=30 s)で、AとBの位置・車線は前作の最後から続ける。
  A(赤・追われる側)・B(水色・追う側)。Bが銃を抜き、連射し、Aは避ける。Bが追い越し、後輪ロックで横滑りして止まる。Aも手前で止まる。
render.py は --scene sequel のとき、motion の代わりにこのモジュール(同じ関数名)を M として使う。"""
import math
import numpy as np
import motion as _M
from motion import (FPS, sm, clamp, mix, keyed_sm, keyed_lin, rng, DIM, HERO, LANE_V, pos_of, to_blender, G, WB, LEAN_K, LEAN_MAX,
                    W_LEAN, K_COUNTER, K_STEADY, RHO_TIRE, W_SUS, BUMP_AMP, NZ, _sos, _shift)

TOFF = 30.0
DUR = 15.0
TWO_PI = 2 * math.pi
DT = 1.0 / 240
CUTS = [2.0, 3.5, 5.0, 6.5, 8.0, 9.5, 10.8, 11.6, 12.3, 14.0]          # 11 カット(τ)
Z0A = -46.0 * TOFF - 30.0                                              # 前作の t=30 でのAの位置(three.js z)
SLIDE_T0 = 12.2                                                        # Bが後輪をロックして滑り始める
SLIDE_DUR = 2.7                                                        # 滑って止まるまで
YAW_SLIDE = math.radians(86.0)
ROLL_SLIDE = -0.30                                                     # 横滑り中の傾き(進行方向と逆へ、上が倒れる)。ブレンダーのロールは + = 右へ
A_BRAKE_T0 = 12.3

# ---- 速度(m/s)と車線 ---------------------------------------------------------------------------------
VA_KEYS = [[0, 46.0], [6.5, 46.0], [10.0, 28.0], [20, 28.0]]
CLOSE_KEYS = [[0, 2.25], [4.0, 2.25], [4.6, 0.0], [9.6, 0.0], [10.4, 7.0], [20, 7.0]]       # B の速度 - A の速度
LANE_A_S = [[0, 1.8], [4.7, 1.8], [5.6, 4.9], [6.9, 1.0], [8.2, 4.7], [9.4, 1.8], [20, 1.8]]
LANE_B_S = [[0, -1.8], [11.0, -1.8], [12.3, 1.0], [20, 1.0]]

# ---- 銃の動き(τ) ----------------------------------------------------------------------------------
T_GRIP_LEAVE, T_CHEST, T_DRAWN = 1.0, 1.9, 2.6            # 右手: ハンドル → 上着の内側 → 銃を抜いて構える
T_HOLSTER0, T_HOLSTER1, T_REGRIP = 10.8, 11.4, 11.9      # 構え → しまう → ハンドルへ戻す
BURSTS = [(5.2, 5.9), (6.4, 7.1), (7.7, 8.4), (8.8, 9.6), (9.9, 10.5)]
FIRE_EVERY = 3.0 / FPS                                    # 8発/秒


def fire_times():
    out = []
    for a, b in BURSTS:
        t = a
        while t < b:
            out.append(round(t * FPS) / FPS); t += FIRE_EVERY
    return out


FIRES = fire_times()


def gun_state(tau):
    """-> dict(chest, aim, vis, recoil)  chest/aim = 右手の位置の重み(0..1)。grip → chest → aim。vis = 銃が見えるか。recoil = 発砲の反動(0..1)"""
    a = sm((tau - T_GRIP_LEAVE) / (T_CHEST - T_GRIP_LEAVE))
    b = sm((tau - T_CHEST) / (T_DRAWN - T_CHEST))
    h = sm((tau - T_HOLSTER0) / (T_HOLSTER1 - T_HOLSTER0))
    g = sm((tau - T_HOLSTER1) / (T_REGRIP - T_HOLSTER1))
    chest = a * (1 - b) + h * (1 - g) * 0 + 0.0
    aim = b * (1 - h)
    chest = (a - b * 1.0) * (1 - h) + h * (1 - g)           # 構えに入るまで chest、しまう途中でまた chest
    chest = clamp(chest, 0.0, 1.0)
    vis = 1.0 if (T_CHEST - 0.25) < tau < (T_HOLSTER1 - 0.05) else 0.0
    rec = 0.0
    for ft in FIRES:
        d = tau - ft
        if 0 <= d < 0.12:
            rec = max(rec, math.exp(-d / 0.035))
    return {'chest': chest, 'aim': clamp(aim, 0.0, 1.0), 'vis': vis, 'recoil': rec, 'grip': clamp(1 - chest - aim, 0.0, 1.0)}


# ---- 経路の表(τ = -1 .. 16.5, 240 Hz) ---------------------------------------------------------------------
_TB = {}
TS = np.arange(-1.0, 16.5, DT)


def _speed_tables():
    vA = np.array([keyed_sm(VA_KEYS, t) for t in TS])
    ab = 0.95 * G * np.array([sm((t - A_BRAKE_T0) / 0.2) for t in TS])        # Aの急ブレーキ
    vA = vA.copy()
    v = 0.0
    out = np.empty_like(vA); cur = None
    for i, t in enumerate(TS):
        if t < A_BRAKE_T0:
            cur = vA[i]; out[i] = cur
        else:
            cur = max(0.0, cur - ab[i] * DT); out[i] = cur
    vA = out
    close = np.array([keyed_sm(CLOSE_KEYS, t) for t in TS])
    vB = vA + close
    # B: 横滑り(一定の減速)
    i0 = int(np.searchsorted(TS, SLIDE_T0)); v0 = vB[i0]
    a0 = v0 / (SLIDE_DUR - 0.1)
    cur = v0
    # Aのブレーキより前の速度を使う(滑り始めの瞬間のAは、まだ減速していない)
    for i in range(i0, len(TS)):
        t = TS[i]
        cur = max(0.0, cur - a0 * sm((t - SLIDE_T0) / 0.2) * DT)
        vB[i] = cur
    return vA, vB


def tables(bid):
    if bid in _TB:
        return _TB[bid]
    vA, vB = _speed_tables()
    v = vA if bid == 'A' else vB
    z = Z0A + (20.0 if bid == 'B' else 0.0) - np.concatenate([[0.0], np.cumsum(v[:-1] * DT)]) - 0.0
    # τ=0 で z が Z0A(+20) になるように
    i0 = int(np.searchsorted(TS, 0.0))
    z = z - (z[i0] - (Z0A + (20.0 if bid == 'B' else 0.0)))
    keys = LANE_A_S if bid == 'A' else LANE_B_S
    ph = NZ[bid]['ph']
    drift = 0.030 * np.sin(TWO_PI * 0.35 * TS + ph[0]) + 0.018 * np.sin(TWO_PI * 0.61 * TS + ph[1])
    xp = np.array([keyed_sm(keys, t) for t in TS]) + drift
    h = 5
    ax = (np.roll(xp, -h) - 2 * xp + np.roll(xp, h)) / (h * DT) ** 2
    ax[:h] = 0; ax[-h:] = 0
    u = np.clip(LEAN_K * np.arctan(ax / G), -LEAN_MAX, LEAN_MAX)
    lean, lean_rate = _sos(_shift(u, 0.38), W_LEAN, 1.0)
    wob = 0.010 * np.sin(TWO_PI * 0.9 * TS + ph[2]) + 0.006 * np.sin(TWO_PI * 1.9 * TS + ph[3])
    wob_rate = TWO_PI * 0.9 * 0.010 * np.cos(TWO_PI * 0.9 * TS + ph[2]) + TWO_PI * 1.9 * 0.006 * np.cos(TWO_PI * 1.9 * TS + ph[3])
    phi = lean + wob
    phid = lean_rate + wob_rate
    steer = K_COUNTER * phid - K_STEADY * phi
    vxp = (np.roll(xp, -h) - np.roll(xp, h)) / (2 * h * DT)
    heading = -np.arctan(vxp / np.maximum(v, 8.0))
    heading[:h] = 0; heading[-h:] = 0
    slide = np.zeros_like(TS); roll_s = np.zeros_like(TS); dist_r = np.cumsum(v * DT) - v[0] * DT
    if bid == 'B':
        k = np.array([sm((t - SLIDE_T0) / 0.9) for t in TS])
        slide = YAW_SLIDE * k
        k2 = np.array([sm((t - SLIDE_T0) / 0.5) for t in TS])
        roll_s = ROLL_SLIDE * k2
        lock = int(np.searchsorted(TS, SLIDE_T0))
        dist_r = dist_r.copy(); dist_r[lock:] = dist_r[lock]        # 後輪はロック(回らない)
        # 横滑りで前輪のハンドルが切れる(カウンターではなく、車体の向きへ)
        steer = steer + 0.35 * np.array([sm((t - SLIDE_T0) / 0.6) * (1 - sm((t - SLIDE_T0 - 1.6) / 0.8)) for t in TS])
    # 胴体・頭(ライダーが車体に対して動く量)
    tor_abs, _ = _sos(1.10 * phi, 11.0, 0.75)
    rho = 0.14 * np.tanh((tor_abs - phi) / 0.14)
    head_abs, _ = _sos(0.30 * tor_abs, 10.0, 0.7)
    head_roll = 0.12 * np.tanh((head_abs - tor_abs) / 0.12)
    gaze, _ = _sos(_shift(heading, 0.45), 9.0, 0.8)
    head_yaw = 0.75 * (gaze - heading)
    # 前後の加減速 → 車体の前下がり(ブレーキのときの沈み込み)
    acc = np.gradient(v, DT)
    dive = np.clip(-acc / G, 0.0, 1.5) * 0.045
    dive, _ = _sos(dive, 14.0, 0.6)
    # サスペンション(段差の列 + 大きなバンプ)
    uf = np.zeros_like(TS)
    for n in range(-2, 80):
        te = (30.0 * n + 7.0 * ph[4]) / 40.0
        uf += (0.0045 + 0.0020 * math.sin(n * 2.3 + ph[0])) * np.exp(-(((TS - te) / 0.035) ** 2))
    for bt in (3.1, 7.3, 11.0, 14.1):
        uf += BUMP_AMP * np.exp(-(((TS - (bt + (0.4 if bid == 'B' else 0.0))) / 0.05) ** 2))
    ur = _shift(uf, -WB / 40.0)
    out = {'ts': TS, 'x': xp, 'z': z, 'v': v, 'phi': phi, 'steer': steer, 'rho': rho, 'head_roll': head_roll, 'head_yaw': head_yaw,
           'slide': slide, 'roll_s': roll_s, 'dist': -z, 'dist_r': dist_r + (-z[0]), 'heading': heading, 'dive': dive}
    for kk, uu in (('f', uf), ('r', ur)):
        fz, _ = _sos(uu, W_SUS, 0.36)
        out['rel_' + kk] = fz - uu * 0.55
    out['heave'] = 0.5 * (out['rel_f'] + out['rel_r'])
    out['fpitch'] = (out['rel_f'] - out['rel_r']) / WB + dive
    tz, _ = _sos(out['heave'], TWO_PI * 1.7, 0.35)
    out['tor_dz'] = tz - out['heave']
    tp, _ = _sos(out['fpitch'], TWO_PI * 1.7, 0.4)
    out['tor_pitch'] = tp - out['fpitch']
    _TB[bid] = out
    return out


def _at(bid, key, tau):
    tb = tables(bid)
    return float(np.interp(tau, tb['ts'], tb[key]))


def xA(t): return _at('A', 'x', t - TOFF)
def zA(t): return _at('A', 'z', t - TOFF)
def xB(t): return _at('B', 'x', t - TOFF)
def zB(t): return _at('B', 'z', t - TOFF)


def bike_pose(bid, t):
    """motion.bike_pose と同じ形(+ dist_r = 後輪の走行距離)。t = 世界の時刻"""
    tau = t - TOFF
    tb = tables(bid)
    g = lambda k: float(np.interp(tau, tb['ts'], tb[k]))
    z = g('z'); x = g('x')
    h = 0.05
    vxm = (float(np.interp(tau + h, tb['ts'], tb['x'])) - float(np.interp(tau - h, tb['ts'], tb['x']))) / (2 * h)
    v = max(g('v'), 8.0)
    phi = g('phi') + g('roll_s')
    head_yaw = g('head_yaw')
    if bid == 'A':                                        # Aは、撃たれる前後に後ろ(B)を振り返る
        head_yaw += 1.45 * sm((tau - 3.4) / 0.5) * (1 - sm((tau - 4.9) / 0.5))
    return {'x': x, 'y': RHO_TIRE * (1 - math.cos(phi)), 'z': z, 'yaw': -math.atan(vxm / v) + g('slide'), 'pitch': 0.0, 'roll': -phi,
            'dist': g('dist'), 'dist_r': g('dist_r'), 'steer': g('steer'), 'body_dz': g('heave'), 'body_pitch': g('fpitch'),
            'torso_roll': g('rho'), 'torso_dz': g('tor_dz'), 'torso_pitch': g('tor_pitch') + (0.0 if bid == 'B' else 1.2 * g('dive')),
            'head_roll': g('head_roll'), 'head_yaw': head_yaw, 'v': g('v'), 'slide': g('slide')}


# ---- カメラ(11カット)。 three.js 座標 ------------------------------------------------------------------
def shot_index(t):
    tau = t - TOFF
    for i, c in enumerate(CUTS):
        if tau < c:
            return i
    return len(CUTS)


def stop_point():
    return xB(TOFF + SLIDE_T0 + SLIDE_DUR), zB(TOFF + SLIDE_T0 + SLIDE_DUR)


def shot_at(t, k=None):
    """-> (p, l, fov_deg, roll)"""
    tau = t - TOFF
    if k is None:
        k = shot_index(t)
    ax, az, bx, bz = xA(t), zA(t), xB(t), zB(t)
    mx, mz = (ax + bx) / 2, (az + bz) / 2
    sx, sz = stop_point()
    roll = 0.0
    if k == 0:       # 1 Bの後ろ上から引き(AとB)
        p = [bx + 1.2, 2.9, bz + 7.5]; l = [ax - 0.4, 1.0, az]; fov = 50
    elif k == 1:     # 2 Bのライダーを斜め前から
        p = [bx + 1.7, 1.55, bz - 2.3]; l = [bx - 0.02, 1.28, bz - 0.15]; fov = 36
    elif k == 2:     # 3 Aの肩越し、奥にB
        p = [ax + 0.5, 1.42, az + 1.55]; l = [bx, 1.15, bz]; fov = 34
    elif k == 3:     # 4 銃口を横から
        p = [bx + 1.9, 1.4, bz - 0.65]; l = [bx + 0.3, 1.32, bz - 0.95]; fov = 28
    elif k == 4:     # 5 路面すれすれ、後ろから(Aが避ける)
        p = [ax - 0.8, 0.32, az + 6.5]; l = [ax + 0.2, 0.45, az - 3]; fov = 46
    elif k == 5:     # 6 斜め前上から2台
        p = [mx + 3.6, 3.6, mz - 11]; l = [mx, 0.9, mz]; fov = 40; roll = 0.04
    elif k == 6:     # 7 俯瞰
        p = [mx, 17.0, mz + 4]; l = [mx, 0.0, mz - 1.5]; fov = 50
    elif k == 7:     # 8 銃(弾切れ → しまう)
        p = [bx + 1.1, 1.3, bz - 0.55]; l = [bx + 0.18, 1.18, bz - 0.3]; fov = 30
    elif k == 8:     # 9 路肩側から低く、追い越し
        p = [mx + 7.5, 0.6, mz]; l = [mx, 0.95, mz]; fov = 40
    elif k == 9:     # 10 路面すれすれ(世界に固定)。Bが近づいて滑る
        zc = zB(TOFF + SLIDE_T0 + 1.1)
        p = [sx + 5.8, 0.42, zc]; l = [bx, 0.75, bz]; fov = 38
    else:            # 11 引き(世界に固定)
        p = [sx + 11.0, 1.5, sz + 7.0]; l = [sx - 0.2, 0.8, sz + 1.5]; fov = 34
    return p, l, fov, roll


# ---- 交通 ----------------------------------------------------------------------------------------------
traffic = _M.traffic


def hidden_cars(cars):
    """続編の2台・カメラと重なる車の番号(描かない)。前作の交通をそのまま使い、邪魔な車だけ消す。"""
    out = set()
    camp = []
    for ti in np.arange(0.0, DUR, 0.1):
        t = TOFF + ti
        camp.append(shot_at(t))
    for i, c in enumerate(cars):
        wc, lc = DIM[c['kind']]
        for ti in np.arange(0.0, DUR, 0.1):
            t = TOFF + ti
            z = pos_of(c, t)
            for (bx_, bz_, wide) in ((xA(t), zA(t), 1.4), (xB(t), zB(t), 2.4)):
                if abs(z - bz_) < (lc + 2.4) / 2 + 3.0 and abs(c['x'] - bx_) < (wc + wide) / 2 + 0.8:
                    out.add(i)
            p = camp[int(round(ti / 0.1))][0]
            if p[1] < 4 and abs(z - p[2]) < lc / 2 + 4.0 and abs(c['x'] - p[0]) < wc / 2 + 2.2:
                out.add(i)
        # 滑って止まるまでのBの通り道(終点より後ろ側)も空ける
        sx, sz = stop_point()
        if abs(c['x'] - sx) < 3.2 and sz - 80 < pos_of(c, TOFF + DUR) < sz + 25:
            out.add(i)
    return out


if __name__ == '__main__':
    for tau in (0, 2, 4, 5.5, 7, 9, 10, 11, 12, 12.2, 12.6, 13, 13.5, 14, 14.9, 15):
        t = TOFF + tau
        pa, pb = bike_pose('A', t), bike_pose('B', t)
        print('τ=%5.2f shot %2d  A x%6.2f z%8.1f v%5.1f | B x%6.2f z%8.1f v%5.1f slide%.2f | gap(B-A, +=Bが後ろ) %6.1f  gun %s' % (
            tau, shot_index(t) + 1, pa['x'], pa['z'], pa['v'], pb['x'], pb['z'], pb['v'], pb['slide'], pb['z'] - pa['z'],
            {k: round(v, 2) for k, v in gun_state(tau).items()}))
    print('stop point', stop_point(), 'fires', len(FIRES))
    cars = _M.traffic()
    print('hidden cars', len(hidden_cars(cars)), 'of', len(cars))
