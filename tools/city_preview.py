"""city_preview.py — 背景の街(tools/city_kit.build_city)の、修正前(legacy)と修正後(new)を、同じ視点・同じ時刻で撮る。 chase/ は編集せず、import して実際に動かす。
  blender --background --factory-startup --python tools/city_preview.py -- OUT [params.json]
  params.json: {"samples": 32, "res": [800, 450], "views": ["g_night", "foot_a", ...], "which": ["legacy", "new"], "emit": 8.0, "exposure": 0.0,
                "small_samples": 16, "small_res": [640, 360], "random_n": 10}
  views: g_night(地上の目線・夜の右側の壁。 指摘の場所) / foot_a / foot_b(足元の寄り) / v1_night / v1_dusk(高架の目線) / r01〜r10(道路沿いのランダムな地点。 new だけ)
         overlap(数値だけ。 通行帯と建物の交差面積を legacy と new で測る。 いつも先に測る)
出力: OUT/<view>_<legacy|new|old>.png、OUT/perf.json(交差面積・棟数・ポリゴン・構築時間・1枚の描画時間)、OUT/overlap.json
hand_view.yml で動かす: gh workflow run hand_view.yml -f script=tools/city_preview.py -f params='{"views":["g_night"]}'"""
import bpy, os, sys, json, math, time, io, re, contextlib, random, ast
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, 'chase'), HERE]
ARGV = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else ['out']
OUT = ARGV[0]; os.makedirs(OUT, exist_ok=True)
PRM = json.load(open(ARGV[1])) if len(ARGV) > 1 else {}
SAMPLES = int(PRM.get('samples', 32)); RES = PRM.get('res', [800, 450]); WHICH = PRM.get('which', ['legacy', 'new']); EMIT = float(PRM.get('emit', 8.0))
SSAMPLES = int(PRM.get('small_samples', 16)); SRES = PRM.get('small_res', [640, 360]); NRAND = int(PRM.get('random_n', 10))
EXPO = float(PRM.get('exposure', 0.0)); WORLD_NIGHT = float(PRM.get('world_night', 0.04))
ONLY = PRM.get('views') or ['overlap', 'g_night', 'foot_a', 'foot_b', 'v1_night']


def log(*a):
    print('hand', *a, flush=True)


import render as R        # noqa: E402  (chase/render.py: 編集しない。材質の補助と P を使う)
import city as OLD        # noqa: E402  (chase/city.py: いまの街)
import city_kit as ck     # noqa: E402

CITY = os.path.join(ROOT, 'assets', 'city.json')
GZ = R.P['ground_z']
CENTER = (40.0, 720.0); RADIUS = 900.0

# ------------------------------------------------------------------ 視点(世界座標 = city.json と同じ。 z = 高さ)
V1 = dict(pos=(-3.0, 548.0, 2.6), tgt=(75.0, 640.0, -8.0), fov=52.0)            # 高架の目線(路面+2.6m = 地上14.6m)。 斜め下の右前の街
V2 = dict(pos=(-2.0, 588.0, 1.3), tgt=(4.0, 770.0, 7.0), fov=38.0)              # 追跡カメラに近い位置(路面+1.3m)。 前方の遠景の街並み


def find_ground_view(city):
    """地上の目線(1ブロック): 両側に建物が並ぶ、高架から少し離れた道の区間を探す。 -> dict(pos, tgt, fov)。 修正前の比較画像と同じ場所"""
    vg = ck.VertexGrid(city); vgk = ck.VertexGrid(city, skip_house=True)
    best = None
    for r in city['roads']:
        if r['c'] not in ('tertiary', 'residential', 'unclassified', 'secondary') or r['w'] < 5.5:
            continue
        S = ck.resample(r['p'], 4.0)
        for i in range(0, max(1, len(S) - 12), 4):
            seg = S[i:i + 13]
            if len(seg) < 13:
                continue
            m = seg[6]
            if not (25 < m[0] < 170 and 420 < m[1] < 760):
                continue
            gl, gr = ck.side_gaps(seg, vg)
            if gl is None or gr is None:
                continue
            P = vgk.query_box(m[0] - 45, m[1] - 45, m[0] + 45, m[1] + 45)
            score = len(P) - 4.0 * abs(gl - gr)
            if best is None or score > best[0]:
                best = (score, seg, gl, gr, r['w'], r['c'])
    if best is None:
        return dict(pos=(80.0, 560.0, GZ + 1.7), tgt=(80.0, 620.0, GZ + 6.0), fov=55.0)
    _, seg, gl, gr, w, c = best
    d = seg[-1] - seg[0]; d /= np.hypot(*d)
    p0 = seg[0] - d * 8.0; t0 = seg[-1] + d * 40.0
    log('ground view road', c, w, 'gaps', round(gl, 1), round(gr, 1), 'at', [round(x) for x in seg[6]])
    return dict(pos=(float(p0[0]), float(p0[1]), GZ + 1.7), tgt=(float(t0[0]), float(t0[1]), GZ + 7.0), fov=58.0)


def piece_frame(pc, k):
    """区間 pc の k 番目の点での 位置・進行方向・左の法線"""
    S = pc['S']; i = max(0, min(len(S) - 1, k))
    a = S[max(0, i - 1)]; b = S[min(len(S) - 1, i + 1)]; t = b - a; t = t / (np.hypot(*t) + 1e-9)
    return S[i], t, np.array([-t[1], t[0]])


def foot_view(pc, side, q, back=False):
    """壁の足元の寄り: 歩道の上(壁の手前 約2.4m)から、先の壁の足元を見る"""
    k = len(pc['S']) // 2
    p, t, n = piece_frame(pc, k); sg = 1.0 if side == 'L' else -1.0
    if back:
        t = -t
    off = max(1.3, q - 2.4)
    pos = p + n * sg * off
    tgt = p + t * 5.0 + n * sg * q
    return dict(pos=(float(pos[0]), float(pos[1]), GZ + 1.6), tgt=(float(tgt[0]), float(tgt[1]), GZ + 0.25), fov=62.0)


def pick_views(plan_new, gv):
    """修正後の計画から、足元の寄り2つと、道路沿いのランダムな10地点を選ぶ。 -> dict(view名 -> 視点dict)"""
    pinfo = [p for p in plan_new['streets']['pinfo'] if not p['band']]
    out = {}
    gp = np.asarray(gv['pos'][:2], float)
    near = sorted(pinfo, key=lambda p: float(np.hypot(p['S'][:, 0] - gp[0], p['S'][:, 1] - gp[1]).min()))
    near = [p for p in near if (p['qL'] is not None or p['qR'] is not None)][:3]
    if near:
        pc = near[0]
        sides = [(s, q) for s, q in (('R', pc['qR']), ('L', pc['qL'])) if q is not None and q < 14.0]
        sides.sort(key=lambda a: a[1])
        if sides:
            out['foot_a'] = foot_view(pc, sides[0][0], sides[0][1])
            out['foot_b'] = foot_view(pc, sides[1][0], sides[1][1]) if len(sides) > 1 else foot_view(pc, sides[0][0], sides[0][1], back=True)
    rnd = random.Random(20261119)
    cands = [p for p in pinfo if p['cls'] in ('primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'trunk') and p['qL'] is not None and p['qR'] is not None
             and p['qL'] < 22 and p['qR'] < 22 and math.hypot(p['S'][len(p['S']) // 2][0] - CENTER[0], p['S'][len(p['S']) // 2][1] - CENTER[1]) < 600]
    rnd.shuffle(cands)
    chosen = []
    for p in cands:
        m = p['S'][len(p['S']) // 2]
        if all(math.hypot(m[0] - c[0], m[1] - c[1]) > 110 for c in chosen):
            chosen.append(m); pc = p
            k = 1; pp, t, n = piece_frame(pc, k)
            pos = pp - t * 2.0; tgt = pp + t * 40.0
            out['r%02d' % len(chosen)] = dict(pos=(float(pos[0]), float(pos[1]), GZ + 1.7), tgt=(float(tgt[0]), float(tgt[1]), GZ + 4.0), fov=62.0,
                                              road=pc['cls'], qL=round(pc['qL'], 1), qR=round(pc['qR'], 1))
        if len(chosen) >= NRAND:
            break
    return out


def set_cam(v):
    camd = bpy.data.cameras.new('Cam'); camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24.0
    camd.lens = 12.0 / math.tan(math.radians(v['fov']) / 2.0); camd.clip_start = 0.08; camd.clip_end = 9000
    co = bpy.data.objects.new('Cam', camd); bpy.context.scene.collection.objects.link(co); bpy.context.scene.camera = co
    co.location = v['pos']; co.rotation_euler = (Vector(v['tgt']) - Vector(v['pos'])).to_track_quat('-Z', 'Y').to_euler()
    return co


def move_cam(co, v):
    co.data.lens = 12.0 / math.tan(math.radians(v['fov']) / 2.0)
    co.location = v['pos']; co.rotation_euler = (Vector(v['tgt']) - Vector(v['pos'])).to_track_quat('-Z', 'Y').to_euler()


def mesh_stats(before):
    faces = tris = objs = 0; seen = {}
    for ob in bpy.data.objects:
        if ob.name in before or ob.type != 'MESH':
            continue
        objs += 1
        me = ob.data
        if me.name not in seen:
            seen[me.name] = (len(me.polygons), len(me.loop_triangles))
        f, t = seen[me.name]; faces += f; tris += t
    return dict(objects=objs, faces=faces, tris=tris)


def fresh(mode, samples, w, h):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    R._fog_group = None
    R.P['world'] = 0.6 if mode == 'dusk' else WORLD_NIGHT
    R.P['motion_blur'] = 0
    R.setup_render(samples, w, h)
    sc = bpy.context.scene
    im = sc.render.image_settings; im.file_format = 'PNG'; im.color_mode = 'RGB'; im.color_depth = '8'
    try:
        sc.view_settings.view_transform = 'AgX'; sc.view_settings.exposure = EXPO
    except Exception as e:
        log('view_transform', e)
    R.build_world(); R.build_road()
    return sc


def build_old():
    before = {o.name for o in bpy.data.objects}
    t0 = time.time(); buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        OLD.build(R)
    dt = time.time() - t0
    out = buf.getvalue(); sys.stdout.write(out)
    cnt = {}
    m = re.search(r"city buildings: (\{.*?\})", out)
    if m:
        try:
            cnt = ast.literal_eval(m.group(1))
        except Exception:
            pass
    st = mesh_stats(before); st.update(build_s=round(dt, 1), buildings=cnt)
    return st


def build_new(cam_pts, hour, center, radius, legacy):
    before = {o.name for o in bpy.data.objects}
    t0 = time.time()
    root = ck.build_city(CITY, center, radius, cam_pts, hour=hour, seed=1, ground_z=GZ, emit=EMIT, material_hook=R.add_fog, detail_policy=dict(legacy=bool(legacy)),
                         obs_emit=R.P['obs_emit'], street_emit=R.P['street_emit'], pool_emit=R.P['pool_emit'])
    dt = time.time() - t0
    bpy.context.view_layer.update()
    st = mesh_stats(before); st.update(build_s=round(dt, 1), city=json.loads(root['city_stats']))
    return st


def main():
    city = ck.load_city(CITY)
    perf = dict(samples=SAMPLES, res=RES, emit=EMIT, views={}, builds={}, renders={}, overlap={})
    gv = find_ground_view(city)
    cams = [V1['pos'], V2['pos'], gv['pos']]
    # --- 数値: 通行帯と建物の足元の矩形の交差面積(修正前 legacy・修正後 new)。 同じ範囲・同じカメラ
    plan_new = None
    for which in ('legacy', 'new'):
        t0 = time.time()
        p = ck.plan_all(city, CENTER, RADIUS, cams, 21.0, 1, dict(legacy=(which == 'legacy')))
        ov = dict(p['overlap']); ov.update(plan_s=round(time.time() - t0, 1), recs=len(p['recs']), houses=len(p['houses']), rect_fitted=p['rect_fitted'], rect_clipped=p['rect_clipped'],
                  rect_dropped=p['rect_dropped'], final_clipped=p.get('final_clipped', 0), aprons=dict(faces=p['aprons']['faces'], road_faces=p['aprons']['road_faces']))
        perf['overlap'][which] = ov
        log('OVERLAP', which, json.dumps(ov, ensure_ascii=False))
        if which == 'new':
            plan_new = p
    json.dump(perf['overlap'], open(os.path.join(OUT, 'overlap.json'), 'w'), indent=1, ensure_ascii=False, default=float)
    # --- 視点
    picked = pick_views(plan_new, gv)
    VIEWS = {'g_night': ('night', gv, 'g'), 'v1_night': ('night', V1, 'v1n'), 'v1_dusk': ('dusk', V1, 'v1d')}
    for k in ('foot_a', 'foot_b'):
        if k in picked:
            VIEWS[k] = ('night', picked[k], 'g')
    for k, v in picked.items():
        if k.startswith('r'):
            VIEWS[k] = ('night', v, k)
    views = {k: v for k, v in VIEWS.items() if k in ONLY}
    perf['views'] = {k: dict(mode=m, pos=v['pos'], tgt=v['tgt'], fov=v['fov'], **({'road': v['road'], 'qL': v['qL'], 'qR': v['qR']} if 'road' in v else {})) for k, (m, v, b) in views.items()}
    json.dump(perf, open(os.path.join(OUT, 'perf.json'), 'w'), indent=1, ensure_ascii=False, default=float)
    groups = {}
    for k, (m, v, b) in views.items():
        groups.setdefault(b, []).append(k)
    for which in WHICH:
        for b, ks in groups.items():
            small = b.startswith('r') or b == 'g' and False
            if b.startswith('r') and which != 'new' and not PRM.get('random_legacy'):
                continue
            mode = views[ks[0]][0]
            vs = {k: views[k][1] for k in ks}
            w, h, sm = (SRES[0], SRES[1], SSAMPLES) if b.startswith('r') else (RES[0], RES[1], SAMPLES)
            key = '%s_%s' % (which, b)
            try:
                sc = fresh(mode, sm, w, h)
                hour = 18.5 if mode == 'dusk' else 21.0
                if b.startswith('r'):
                    v0 = vs[ks[0]]; center = (v0['pos'][0], v0['pos'][1]); radius = 480.0; cp = [v0['pos']]
                else:
                    center = CENTER; radius = RADIUS; cp = [v['pos'] for v in vs.values()] + ([V1['pos'], V2['pos']] if b == 'g' and False else [])
                perf['builds'][key] = build_new(cp, hour, center, radius, which == 'legacy') if which in ('legacy', 'new') else build_old()
                log('BUILD', key, json.dumps({k: v for k, v in perf['builds'][key].items() if k != 'city'}, ensure_ascii=False))
            except Exception as e:
                import traceback; traceback.print_exc()
                log('ERR build', key, repr(e)); continue
            co = set_cam(next(iter(vs.values())))
            for vk, v in vs.items():
                move_cam(co, v)
                sc.render.filepath = os.path.join(os.path.abspath(OUT), '%s_%s.png' % (vk, which))
                t0 = time.time()
                try:
                    bpy.ops.render.render(write_still=True)
                except Exception as e:
                    log('ERR render', vk, which, repr(e)); continue
                perf['renders']['%s_%s' % (vk, which)] = round(time.time() - t0, 1)
                log('RENDER', vk, which, '%.1fs' % (time.time() - t0))
            json.dump(perf, open(os.path.join(OUT, 'perf.json'), 'w'), indent=1, ensure_ascii=False, default=float)
    json.dump(perf, open(os.path.join(OUT, 'perf.json'), 'w'), indent=1, ensure_ascii=False, default=float)


main()
