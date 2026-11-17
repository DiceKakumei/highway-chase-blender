"""高速チェイスの動き。three.js版 scene.js の移植(純関数。bpyは使わない)。
座標は three.js のまま: x=右, y=上, z=手前(前進は -z)。Blender へは to_blender() で (x, -z, y)。"""
import math

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


NZ = {'A': {'ph': [0.3, 1.7, 4.1, 2.2, 5.0, 0.9], 'bumps': [3.7, 7.4, 12.9, 18.3, 22.1, 27.0]},
      'B': {'ph': [2.1, 0.4, 3.3, 5.5, 1.2, 4.4], 'bumps': [4.9, 9.2, 14.1, 19.4, 23.0, 26.2]}}


def bike_noise(bid, t):
    ph = NZ[bid]['ph']

    def w(tt):
        return 0.055 * math.sin(TAU * 1.25 * tt + ph[0]) + 0.035 * math.sin(TAU * 0.77 * tt + ph[1]) + 0.012 * math.sin(TAU * 2.3 * tt + ph[2])
    dx = w(t)
    vlat = (w(t + 0.04) - w(t - 0.04)) / 0.08
    dy = 0.02 + 0.010 * math.sin(TAU * 2.05 * t + ph[3]) + 0.007 * math.sin(TAU * 2.64 * t + ph[4]) + 0.003 * math.sin(TAU * 13.1 * t + ph[5]) + 0.002 * math.sin(TAU * 17.7 * t + ph[0])
    pitch = 0.006 * math.sin(TAU * 2.64 * t + ph[4] + 1.0)
    for bt in NZ[bid]['bumps']:
        e = math.exp(-(((t - bt) / 0.06) ** 2))
        dy += 0.03 * e
        pitch += 0.015 * e
    return {'dx': dx, 'vlat': vlat, 'dy': dy, 'pitch': pitch}


def bike_pose(bid, t):
    """-> dict: x, y, z (three.js), yaw (about +y), pitch (about +x), roll (about +z, three.js sign), dist (m travelled forward)"""
    fx, fz = (xA, zA) if bid == 'A' else (xB, zB)
    x, z, vxm = fx(t), fz(t), vx(fx, t)
    n = bike_noise(bid, t)
    lat = (vxm + n['vlat']) / V
    return {'x': x + n['dx'], 'y': n['dy'], 'z': z, 'yaw': -math.atan(lat) * 1.1, 'pitch': n['pitch'], 'roll': -clamp(lat * 3.4, -0.5, 0.5), 'dist': -z}


# ---- cameras: 15 shots, hard cuts ----------------------------------------------------------------
def shot_index(t):
    for i, c in enumerate(CUTS):
        if t < c:
            return i
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
            if not near_hero and not _hits_bike(cand) and not _hits_camera(cand):
                cand['pal'] = int(cR() * 8)
                cars.append(cand)
            z -= ln / 2 + 12 + cR() * 55
    for lane in (-14.5, -18.5):
        z = -2500.0
        while z < 140:
            kind = 'truck' if cR() < 0.15 else 'car'
            ln = DIM[kind][1]
            z += ln / 2
            pal = int(cR() * 8)
            cars.append({'x': lane, 'z0': z, 'v': 32, 'dir': 1, 'kind': kind, 'hero': False, 'pal': pal})
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
