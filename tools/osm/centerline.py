"""首都高3号渋谷線(下り=西行き)の中心線をつなぎ、いちばん真っすぐな3.4 kmの区間を選ぶ。
usage: python centerline.py mw.json center.json"""
import json, math, sys

LAT0, LON0 = 35.645, 139.66
KX = 111320 * math.cos(math.radians(LAT0)); KY = 110540


def xy(p):
    return ((p['lon'] - LON0) * KX, (p['lat'] - LAT0) * KY)


d = json.load(open(sys.argv[1], encoding='utf-8'))
ways = [e for e in d['elements'] if e.get('tags', {}).get('ref') == '3']
segs = [[xy(p) for p in w['geometry']] for w in ways]
# 西行き(x が減る向き)の way だけ
west = [s for s in segs if s[-1][0] < s[0][0]]
print('ways', len(segs), 'westbound', len(west))
# 端点でつなぐ
west.sort(key=lambda s: -s[0][0])
chain = list(west[0]); used = {0}
while True:
    best = None
    for i, s in enumerate(west):
        if i in used:
            continue
        dd = math.dist(chain[-1], s[0])
        if dd < 30 and (best is None or dd < best[0]):
            best = (dd, i)
    if best is None:
        break
    used.add(best[1]); chain += west[best[1]][1:]
print('chain points', len(chain), 'used ways', len(used))
# 弧長
S = [0.0]
for a, b in zip(chain[:-1], chain[1:]):
    S.append(S[-1] + math.dist(a, b))
print('length', round(S[-1]), 'start', [round(v) for v in chain[0]], 'end', [round(v) for v in chain[-1]])


def at(s):
    for i in range(len(S) - 1):
        if S[i + 1] >= s:
            f = (s - S[i]) / max(1e-9, S[i + 1] - S[i])
            return (chain[i][0] + (chain[i + 1][0] - chain[i][0]) * f, chain[i][1] + (chain[i + 1][1] - chain[i][1]) * f)
    return chain[-1]


# 3.4 km の窓ごとに、直線からの最大のずれを測る
WIN = 3400
out = []
s0 = 0
while s0 + WIN <= S[-1]:
    a, b = at(s0), at(s0 + WIN)
    L = math.dist(a, b); ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    dev = 0
    for k in range(0, WIN + 1, 50):
        p = at(s0 + k)
        dev = max(dev, abs((p[0] - a[0]) * -uy + (p[1] - a[1]) * ux))
    out.append((dev, s0, a, b))
    s0 += 100
out.sort()
for dev, s0, a, b in out[:8]:
    lat = lambda p: (LAT0 + p[1] / KY, LON0 + p[0] / KX)
    print('dev %.0f m  s0 %d  from %.5f,%.5f to %.5f,%.5f' % (dev, s0, *lat(a), *lat(b)))
json.dump({'chain': chain, 'S': S, 'LAT0': LAT0, 'LON0': LON0, 'windows': [(dv, s) for dv, s, _, _ in out]}, open(sys.argv[2], 'w'))
