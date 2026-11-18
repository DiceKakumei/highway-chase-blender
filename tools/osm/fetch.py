"""渋谷線沿い(渋谷〜三軒茶屋〜駒沢)の建物・道・木・緑地・水面を Overpass から取る。
usage: python fetch.py <out_dir>"""
import json, os, subprocess, sys, time

HOST = 'https://overpass.private.coffee/api/interpreter'
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
LAT = (35.628, 35.674); LON = (139.650, 139.728)
NX = 4


def run(name, body, bbox):
    out = os.path.join(OUT, name + '.json')
    if os.path.exists(out) and os.path.getsize(out) > 2000:
        return
    q = '[out:json][timeout:170];(%s);out tags geom;' % body.replace('BB', '(%f,%f,%f,%f)' % bbox)
    for attempt in range(5):
        subprocess.run(['curl', '-sS', '-A', 'city-layout-research/1.0', '--max-time', '240', '-o', out, '--data-urlencode', 'data=' + q, HOST])
        ok = os.path.exists(out) and open(out, 'rb').read(1) == b'{'
        print(name, os.path.getsize(out) if os.path.exists(out) else 0, 'ok' if ok else 'retry', flush=True)
        if ok:
            return
        time.sleep(10 * (attempt + 1))


for i in range(NX):
    lo = LON[0] + (LON[1] - LON[0]) * i / NX; hi = LON[0] + (LON[1] - LON[0]) * (i + 1) / NX
    bb = (LAT[0], lo, LAT[1], hi)
    run('bld_%d' % i, 'way["building"]BB;relation["building"]BB;', bb)
    time.sleep(3)
bb = (LAT[0], LON[0], LAT[1], LON[1])
run('roads', 'way["highway"]BB;way["railway"~"rail|light_rail"]BB;', bb)
time.sleep(3)
run('green', 'node["natural"="tree"]BB;way["natural"~"tree_row|wood|water|scrub"]BB;way["leisure"~"park|garden|pitch|playground"]BB;'
    'way["landuse"~"grass|forest|cemetery|religious|recreation_ground|meadow"]BB;way["waterway"]BB;way["amenity"~"place_of_worship|school|university"]BB;'
    'relation["natural"="water"]BB;relation["leisure"="park"]BB;', bb)
