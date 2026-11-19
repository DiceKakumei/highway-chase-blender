"""city_preview.py — 「いまの街(chase/city.py)」と「新しい街(tools/city_kit.build_city)」を、同じ視点・同じ時刻で並べて撮る。 chase/ は編集せず、import して実際に動かす。
  blender --background --factory-startup --python tools/city_preview.py -- OUT [params.json]
  params.json: {"samples": 32, "res": [800, 450], "views": ["v1_dusk", ...], "which": ["old", "new"], "emit": 3.0, "exposure": 0.0}
出力: OUT/<view>_<old|new>.png、OUT/perf.json(棟数・ポリゴン・構築時間・1枚の描画時間)
hand_view.yml で動かす: gh workflow run hand_view.yml -f script=tools/city_preview.py"""
import bpy, os, sys, json, math, time, io, re, contextlib, random, ast
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, 'chase'), HERE]
ARGV = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else ['out']
OUT = ARGV[0]; os.makedirs(OUT, exist_ok=True)
PRM = json.load(open(ARGV[1])) if len(ARGV) > 1 else {}
SAMPLES = int(PRM.get('samples', 32)); RES = PRM.get('res', [800, 450]); WHICH = PRM.get('which', ['old', 'new']); EMIT = float(PRM.get('emit', 3.0))
EXPO = float(PRM.get('exposure', 0.0)); WORLD_NIGHT = float(PRM.get('world_night', 0.04))


def log(*a):
    print('hand', *a, flush=True)


import render as R        # noqa: E402  (chase/render.py: 編集しない。材質の補助と P を使う)
import city as OLD        # noqa: E402  (chase/city.py: いまの街)
import city_kit as ck     # noqa: E402

CITY = os.path.join(ROOT, 'assets', 'city.json')
GZ = R.P['ground_z']

# ------------------------------------------------------------------ 視点(世界座標 = city.json と同じ。 z = 高さ)
V1 = dict(pos=(-3.0, 548.0, 2.6), tgt=(75.0, 640.0, -8.0), fov=52.0)            # 高架の目線(路面+2.6m = 地上14.6m)。 斜め下の右前の街
V2 = dict(pos=(-2.0, 588.0, 1.3), tgt=(4.0, 770.0, 7.0), fov=38.0)              # 追跡カメラに近い位置(路面+1.3m)。 前方の遠景の街並み
VIEWS = {'v1_dusk': ('dusk', V1), 'v1_night': ('night', V1), 'v2_dusk': ('dusk', V2), 'v2_night': ('night', V2), 'v3_night': ('night', None)}
ONLY = PRM.get('views')


def find_ground_view(city):
    """地上の目線(1ブロック): 両側に建物が並ぶ、高架から少し離れた道の区間を探す。 -> dict(pos, tgt, fov)"""
    vg = ck.VertexGrid(city)
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
            P = vg.query_box(m[0] - 40, m[1] - 40, m[0] + 40, m[1] + 40)
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


def build_new(cam_pts, hour):
    before = {o.name for o in bpy.data.objects}
    t0 = time.time()
    root = ck.build_city(CITY, (40.0, 720.0), 900.0, cam_pts, hour=hour, seed=1, ground_z=GZ, emit=EMIT, material_hook=R.add_fog,
                         obs_emit=R.P['obs_emit'], street_emit=R.P['street_emit'], pool_emit=R.P['pool_emit'])
    dt = time.time() - t0
    bpy.context.view_layer.update()
    st = mesh_stats(before); st.update(build_s=round(dt, 1), city=json.loads(root['city_stats']))
    return st


def main():
    city = ck.load_city(CITY)
    gv = find_ground_view(city)
    views = {k: (m, (v if v is not None else gv)) for k, (m, v) in VIEWS.items()}
    if ONLY:
        views = {k: v for k, v in views.items() if k in ONLY}
    cam_pts = [V1['pos'], V2['pos'], gv['pos']]
    perf = dict(samples=SAMPLES, res=RES, emit=EMIT, views={k: dict(mode=m, pos=v['pos'], tgt=v['tgt'], fov=v['fov']) for k, (m, v) in views.items()}, builds={}, renders={})
    for which in WHICH:
        for mode in ('dusk', 'night'):
            vs = {k: v for k, (m, v) in views.items() if m == mode}
            if not vs:
                continue
            key = '%s_%s' % (which, mode)
            try:
                sc = fresh(mode, SAMPLES, RES[0], RES[1])
                hour = 18.5 if mode == 'dusk' else 21.0
                perf['builds'][key] = build_old() if which == 'old' else build_new(cam_pts, hour)
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
