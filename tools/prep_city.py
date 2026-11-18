"""実在の街の地図データ(OpenStreetMap, ODbL)を、動画の道路に沿った座標に直して assets/city.json を作る。
  参考にした場所: 首都高速3号渋谷線(下り)の 渋谷 → 池尻大橋 → 三軒茶屋 → 駒沢。
  道路は動画では真っすぐなので、「首都高の中心線に沿った距離 = Y(前)」「中心線から右への距離 = X」に置き直す。
usage: python tools/prep_city.py <center.json> <raw_dir> <out city.json>
  raw_dir: fetch で取った bld_*.json / roads.json / green.json(Overpass の out tags geom)"""
import glob, json, math, os, sys
import numpy as np

S_OFF = 3400.0          # 中心線の s=3400 m(渋谷駅の西)を Y=0 にする
DECK_CX = -6.4          # 高架の中心の X
SHIFT = 8.0             # 高架が実物より広いぶん、両側の街を外へずらす
Y_MIN, Y_MAX = -1700.0, 5200.0
D_NEAR = 420.0          # これより外は、高い建物だけ残す
D_MAX = 1500.0
STATIONS = [('shibuya', 35.6580, 139.7016, 450, 2.0), ('sangenjaya', 35.6437, 139.6703, 220, 1.4), ('ikejiri', 35.6506, 139.6846, 150, 1.25),
            ('komazawa', 35.6334, 139.6617, 150, 1.2), ('nakameguro', 35.6443, 139.6990, 200, 1.3), ('daikanyama', 35.6481, 139.7033, 150, 1.15),
            ('ebisu', 35.6467, 139.7101, 300, 1.5), ('shimokita', 35.6614, 139.6680, 180, 1.2)]

C = json.load(open(sys.argv[1]))
RAW = sys.argv[2]
LAT0, LON0 = C['LAT0'], C['LON0']
KX = 111320 * math.cos(math.radians(LAT0)); KY = 110540
CH = np.asarray(C['chain'], dtype=np.float64); SS = np.asarray(C['S'], dtype=np.float64)
A = CH[:-1]; B = CH[1:]; AB = B - A; LEN = np.linalg.norm(AB, axis=1); T = AB / LEN[:, None]


def to_xy(lonlat):
    """(N,2) lon,lat → 動画の座標 (x, y), と中心線からの符号つき距離 d"""
    p = np.empty_like(lonlat, dtype=np.float64)
    p[:, 0] = (lonlat[:, 0] - LON0) * KX; p[:, 1] = (lonlat[:, 1] - LAT0) * KY
    out = np.empty((len(p), 3))
    for i0 in range(0, len(p), 20000):
        q = p[i0:i0 + 20000]
        rel = q[:, None, :] - A[None, :, :]
        t = np.clip((rel * T[None]).sum(-1), 0, LEN[None])
        near = A[None] + T[None] * t[..., None]
        dist = np.linalg.norm(q[:, None, :] - near, axis=-1)
        k = dist.argmin(1); idx = np.arange(len(q))
        tt = (rel[idx, k] * T[k]).sum(-1)                                  # 端より外は、端の向きのまま延長する
        tt = np.where((k == 0) & (tt < 0) | (k == len(A) - 1) & (tt > LEN[k]), tt, np.clip(tt, 0, LEN[k]))
        d = rel[idx, k, 0] * T[k, 1] - rel[idx, k, 1] * T[k, 0]            # 進行方向の右が正
        s = SS[k] + tt
        out[i0:i0 + 20000, 0] = d + np.sign(d) * SHIFT + DECK_CX
        out[i0:i0 + 20000, 1] = s - S_OFF
        out[i0:i0 + 20000, 2] = d
    return out


def station_xy():
    ll = np.asarray([[lo, la] for _, la, lo, _, _ in STATIONS])
    return to_xy(ll)[:, :2]


ST = station_xy()


def rng(seed):
    a = seed & 0xffffffff

    def r():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xffffffff
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xffffffff
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xffffffff)) & 0xffffffff
        return ((t ^ (t >> 14)) & 0xffffffff) / 4294967296
    return r


def area_centroid(P):
    x, y = P[:, 0], P[:, 1]; x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y; a = cr.sum() / 2
    if abs(a) < 1e-6:
        return 0.0, P.mean(0)
    return a, np.array([((x + x1) * cr).sum() / (6 * a), ((y + y1) * cr).sum() / (6 * a)])


def simplify(P, tol):
    """閉じた多角形の、ほぼ一直線上の点を間引く。"""
    P = [tuple(p) for p in P]
    changed = True
    while changed and len(P) > 4:
        changed = False
        for i in range(len(P)):
            a, b, c = P[i - 1], P[i], P[(i + 1) % len(P)]
            L = math.dist(a, c)
            dd = math.dist(a, b) if L < 1e-6 else abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / L
            if dd < tol:
                del P[i]; changed = True; break
    return np.asarray(P)


def load(pattern):
    els = []
    for f in sorted(glob.glob(os.path.join(RAW, pattern))):
        try:
            els += json.load(open(f, encoding='utf-8'))['elements']
        except Exception as e:
            print('skip', f, e)
    return els


def geom(e):
    if e['type'] == 'way':
        return [np.asarray([[p['lon'], p['lat']] for p in e['geometry']])]
    if e['type'] == 'relation':
        return [np.asarray([[p['lon'], p['lat']] for p in m['geometry']]) for m in e.get('members', []) if m.get('role') == 'outer' and m.get('geometry')]
    return []


# ---------------------------------------------------------------- roads
ROAD_W = {'trunk': 15.0, 'primary': 14.0, 'secondary': 11.0, 'tertiary': 8.5, 'residential': 5.0, 'unclassified': 5.5, 'living_street': 4.0,
          'service': 3.5, 'pedestrian': 5.0, 'trunk_link': 6.0, 'primary_link': 6.0, 'secondary_link': 5.5, 'tertiary_link': 5.0, 'rail': 7.0}
roads = []; arterial = []
for e in load('roads.json'):
    t = e.get('tags', {})
    cls = t.get('highway') or ('rail' if t.get('railway') else None)
    if cls not in ROAD_W or t.get('tunnel') in ('yes', 'building_passage') or t.get('covered') == 'yes' or t.get('area') == 'yes':
        continue
    for g in geom(e):
        q = to_xy(g)
        if q[:, 1].max() < Y_MIN or q[:, 1].min() > Y_MAX or np.abs(q[:, 2]).min() > D_NEAR + 80:
            continue
        w = ROAD_W[cls]
        if t.get('lanes', '').isdigit() and cls != 'rail':
            w = max(w, int(t['lanes']) * 3.3 + 1.0) if t.get('oneway') == 'yes' else max(w, int(t['lanes']) * 3.2)
        # 高架の真下をまたぐ線は、左右で分ける(左右を別々に外へずらしているため)
        parts = []; cur = [q[0]]
        for a_, b_ in zip(q[:-1], q[1:]):
            if a_[2] * b_[2] < 0:
                parts.append(cur); cur = []
            cur.append(b_)
        parts.append(cur)
        for part in parts:
            if len(part) >= 2:
                pp = np.asarray(part)
                roads.append({'c': cls, 'w': round(w, 1), 'p': [[round(float(v[0]), 1), round(float(v[1]), 1)] for v in pp]})
                if cls in ('trunk', 'primary', 'secondary'):
                    arterial.append(pp[:, :2])
print('roads', len(roads), 'arterial polylines', len(arterial))

# arterial の線分(建物の高さの見積もりに使う)
if arterial:
    AA = np.concatenate([a[:-1] for a in arterial]); AB2 = np.concatenate([a[1:] for a in arterial]) - AA
    AL = np.maximum(np.linalg.norm(AB2, axis=1), 1e-6); AT = AB2 / AL[:, None]


def dist_arterial(pts):
    if not arterial:
        return np.full(len(pts), 999.0)
    out = np.empty(len(pts))
    for i0 in range(0, len(pts), 4000):
        q = pts[i0:i0 + 4000]
        rel = q[:, None, :] - AA[None]
        t = np.clip((rel * AT[None]).sum(-1), 0, AL[None])
        out[i0:i0 + 4000] = np.linalg.norm(rel - AT[None] * t[..., None], axis=-1).min(1)
    return out


# ---------------------------------------------------------------- buildings
raw_b = []
for e in load('bld_*.json'):
    t = e.get('tags', {})
    if t.get('building') in ('roof', 'no') or t.get('location') == 'underground' or t.get('layer', '0').startswith('-'):
        continue
    for g in geom(e):
        if len(g) >= 4:
            raw_b.append((e['id'], t, g))
print('raw buildings', len(raw_b))
seen = set(); B_ = []
allpts = to_xy(np.concatenate([g for _, _, g in raw_b]))
off = 0
for (bid, t, g) in raw_b:
    q = allpts[off:off + len(g)]; off += len(g)
    if bid in seen:
        continue
    seen.add(bid)
    P = q[:-1, :2] if np.allclose(q[0, :2], q[-1, :2]) else q[:, :2]
    dmin = np.abs(q[:, 2]).min()
    if len(P) < 3 or P[:, 1].max() < Y_MIN or P[:, 1].min() > Y_MAX or dmin > D_MAX:
        continue
    if q[:, 2].min() * q[:, 2].max() < 0 or dmin < 7.0:              # 高架の真下にあるもの
        continue
    if P[:, 0].min() < 12.6 and P[:, 0].max() > -25.4:               # 高架にかかるもの
        continue
    a, c = area_centroid(P)
    if abs(a) < 14.0:
        continue
    if a < 0:
        P = P[::-1]
    B_.append((bid, t, simplify(P, 0.35), abs(a), c, dmin))
cent = np.asarray([b[4] for b in B_])
da = dist_arterial(cent)
out_b = []; stat = {}
for (bid, t, P, a, c, dmin), dart in zip(B_, da):
    r = rng(bid)
    kind = t.get('building', 'yes')
    sf = 1.0; st = None
    for (nm, _, _, rad, f), sxy in zip(STATIONS, ST):
        dd = math.dist(c, sxy)
        if dd < rad:
            k = 1.0 + (f - 1.0) * (1.0 - dd / rad) ** 0.7
            if k > sf:
                sf, st = k, nm
    near_art = dart < 16.0 + math.sqrt(a) * 0.6
    tagged = False
    if 'height' in t:
        try:
            h = float(str(t['height']).replace('m', '').strip()); tagged = True
        except ValueError:
            h = None
    else:
        h = None
    lv = None
    if h is None and t.get('building:levels', '').replace('.', '').isdigit():
        lv = max(1, int(float(t['building:levels']))); tagged = True
    if h is None and lv is None:
        u = r()
        if kind in ('house', 'detached') or (a < 130 and not near_art and sf < 1.3):
            lv = 2 if u < 0.72 else 3
            if a < 30:
                lv = 1 if u < 0.6 else 2
        elif kind in ('school', 'university', 'kindergarten', 'college'):
            lv = 3 + int(u * 3)
        elif kind in ('garage', 'garages', 'shed', 'hut'):
            lv = 1
        else:
            if a < 130:
                lv = 3 + int(u * 5) if near_art else 3 + int(u * 2)
            elif a < 300:
                lv = 5 + int(u * 7) if near_art else 3 + int(u * 3)
            elif a < 800:
                lv = 8 + int(u * 7) if near_art else 3 + int(u * 5)
            elif a < 2500:
                lv = 9 + int(u * 8) if near_art else 4 + int(u * 6)
            else:
                lv = 10 + int(u * 10) if near_art else 4 + int(u * 6)
            if kind in ('apartments', 'residential') and not near_art:
                lv = min(lv, 3 + int(u * 8))
            if sf > 1.0 and a > 110:
                lv = int(lv * sf + 0.5)
                if st == 'shibuya' and a > 1400 and r() < 0.55:
                    lv = 18 + int(r() * 22)
    if h is None:
        fh = 3.0 if lv <= 3 else (3.3 if kind in ('apartments', 'residential') else 3.6)
        h = lv * fh + (0.6 if lv <= 3 else 1.2)
    lv = lv or max(1, int(h / 3.5))
    if dmin > D_NEAR and h < 45:                                       # 遠くは高い建物だけ
        continue
    if kind in ('house', 'detached') or (lv <= 3 and a < 170):
        ty = 'house'
    elif h >= 60:
        ty = 'tower'
    elif kind in ('apartments', 'residential', 'dormitory') or (not near_art and sf < 1.25 and lv <= 15 and r() < 0.7) or (near_art and r() < 0.4):
        ty = 'apt'
    else:
        ty = 'office'
    stat[ty] = stat.get(ty, 0) + 1
    out_b.append({'id': int(bid % 1000000007), 't': ty, 'h': round(float(h), 1), 'lv': int(lv), 'a': round(a), 'g': 1 if tagged else 0,
                  'p': [[round(float(v[0]), 2), round(float(v[1]), 2)] for v in P]})
print('buildings kept', len(out_b), stat, 'tagged', sum(b['g'] for b in out_b))
hs = np.asarray([b['h'] for b in out_b]); print('height pct 50/90/99/max', np.percentile(hs, [50, 90, 99]).round(1), hs.max())

# ---------------------------------------------------------------- green / water / trees
green = []; water = []; trees = []
for e in load('green.json'):
    t = e.get('tags', {})
    if e['type'] == 'node':
        q = to_xy(np.asarray([[e['lon'], e['lat']]]))[0]
        if Y_MIN < q[1] < Y_MAX and 7 < abs(q[2]) < D_NEAR:
            trees.append([round(float(q[0]), 1), round(float(q[1]), 1), 1])
        continue
    kind = None
    if t.get('natural') == 'water' or t.get('waterway') in ('riverbank', 'dock'):
        kind = 'water'
    elif t.get('waterway'):
        kind = 'stream'
    elif t.get('natural') == 'tree_row':
        kind = 'row'
    elif t.get('natural') in ('wood', 'scrub') or t.get('landuse') == 'forest':
        kind = 'wood'
    elif t.get('leisure') in ('park', 'garden', 'playground') or t.get('landuse') in ('grass', 'recreation_ground', 'meadow'):
        kind = 'park'
    elif t.get('landuse') in ('cemetery', 'religious') or t.get('amenity') == 'place_of_worship':
        kind = 'shrine'
    elif t.get('leisure') == 'pitch' or t.get('amenity') in ('school', 'university'):
        kind = 'yard'
    if kind is None:
        continue
    for g in geom(e):
        q = to_xy(g)
        if q[:, 1].max() < Y_MIN or q[:, 1].min() > Y_MAX or np.abs(q[:, 2]).min() > D_NEAR or q[:, 2].min() * q[:, 2].max() < 0:
            continue
        pts = [[round(float(v[0]), 1), round(float(v[1]), 1)] for v in q[:, :2]]
        if kind == 'row':
            for a_, b_ in zip(q[:-1], q[1:]):
                n = max(1, int(math.dist(a_[:2], b_[:2]) / 7.0))
                for k in range(n):
                    f = (k + 0.5) / n
                    trees.append([round(float(a_[0] + (b_[0] - a_[0]) * f), 1), round(float(a_[1] + (b_[1] - a_[1]) * f), 1), 1])
        elif kind in ('water', 'stream'):
            water.append({'k': kind, 'p': pts})
        elif len(pts) >= 4:
            green.append({'k': kind, 'p': pts})
print('green', len(green), 'water', len(water), 'mapped trees', len(trees))


# ---------------------------------------------------------------- trees(街路樹・公園・庭木)。建物の中には置かない
CELL = 12.0
grid = {}
polys = [np.asarray(b['p']) for b in out_b]
for i, P in enumerate(polys):
    for gx_ in range(int(P[:, 0].min() // CELL), int(P[:, 0].max() // CELL) + 1):
        for gy_ in range(int(P[:, 1].min() // CELL), int(P[:, 1].max() // CELL) + 1):
            grid.setdefault((gx_, gy_), []).append(i)


def pip(P, x, y):
    inside = False
    n = len(P)
    for i in range(n):
        x1, y1 = P[i]; x2, y2 = P[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def free(x, y, margin=1.2):
    if -27.0 < x < 14.0 or abs(x) > 400 or not (-500 < y < 3300):
        return False
    for i in grid.get((int(x // CELL), int(y // CELL)), []):
        P = polys[i]
        if P[:, 0].min() - margin < x < P[:, 0].max() + margin and P[:, 1].min() - margin < y < P[:, 1].max() + margin:
            if pip(P, x, y):
                return False
            # ふちに近すぎるものも外す
            for k in range(len(P)):
                ax_, ay_ = P[k]; bx_, by_ = P[(k + 1) % len(P)]
                L = math.hypot(bx_ - ax_, by_ - ay_) or 1e-6
                tt = max(0.0, min(1.0, ((x - ax_) * (bx_ - ax_) + (y - ay_) * (by_ - ay_)) / (L * L)))
                if math.hypot(x - ax_ - (bx_ - ax_) * tt, y - ay_ - (by_ - ay_) * tt) < margin:
                    return False
    return True


T2 = []
rt = rng(20261118)
for (x, y, _) in trees:
    if free(x, y, 0.8):
        T2.append([x, y, round(8 + rt() * 4, 1)])
nm = len(T2)
for rd in roads:                                                   # 街路樹
    c = rd['c']
    if c not in ('trunk', 'primary', 'secondary', 'tertiary'):
        continue
    step = 9.0 if c != 'tertiary' else 13.0
    off_ = rd['w'] / 2 + 1.6
    acc_ = rt() * step
    pts = rd['p']
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        L = math.hypot(x2 - x1, y2 - y1)
        if L < 0.5:
            continue
        nx_, ny_ = (y2 - y1) / L, -(x2 - x1) / L
        while acc_ < L:
            f = acc_ / L
            for sgn in ((1, -1) if c != 'tertiary' else ((1,) if int(acc_ / step) % 2 else (-1,))):
                x = x1 + (x2 - x1) * f + nx_ * off_ * sgn; y = y1 + (y2 - y1) * f + ny_ * off_ * sgn
                if rt() < 0.8 and free(x, y, 1.0):
                    T2.append([round(x, 1), round(y, 1), round((9.0 if c != 'tertiary' else 6.5) * (0.8 + rt() * 0.45), 1)])
            acc_ += step
        acc_ -= L
ns = len(T2) - nm
for g in green:                                                    # 公園・社寺・校庭
    P = np.asarray(g['p']); a, _ = area_centroid(P[:-1] if np.allclose(P[0], P[-1]) else P); a = abs(a)
    if a < 40:
        continue
    k = g['k']
    n = int(a / {'park': 75.0, 'wood': 40.0, 'shrine': 60.0, 'yard': 500.0}[k]) + 1
    for _ in range(min(n, 400)):
        x = P[:, 0].min() + rt() * (P[:, 0].max() - P[:, 0].min()); y = P[:, 1].min() + rt() * (P[:, 1].max() - P[:, 1].min())
        if pip(P, x, y) and free(x, y, 1.5):
            T2.append([round(x, 1), round(y, 1), round((8 if k != 'wood' else 11) * (0.75 + rt() * 0.7), 1)])
ng = len(T2) - nm - ns
for b, P in zip(out_b, polys):                                     # 庭木
    if b['t'] != 'house' or rt() > 0.33:
        continue
    k = int(rt() * len(P)); ax_, ay_ = P[k]; bx_, by_ = P[(k + 1) % len(P)]
    L = math.hypot(bx_ - ax_, by_ - ay_) or 1e-6
    x = (ax_ + bx_) / 2 + (by_ - ay_) / L * 1.6; y = (ay_ + by_) / 2 - (bx_ - ax_) / L * 1.6
    if free(x, y, 1.0):
        T2.append([round(x, 1), round(y, 1), round(3.5 + rt() * 3.0, 1)])
print('trees: mapped', nm, 'street', ns, 'green', ng, 'garden', len(T2) - nm - ns - ng, 'total', len(T2))
trees = T2

json.dump({'source': 'OpenStreetMap contributors (ODbL)', 'buildings': out_b, 'roads': roads, 'green': green, 'water': water, 'trees': trees},
          open(sys.argv[3], 'w'), separators=(',', ':'))
print('wrote', sys.argv[3], os.path.getsize(sys.argv[3]) // 1024, 'KB')
