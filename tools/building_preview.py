"""building_preview.py — ビルのキット(tools/building_kit.py)の見本画像・書き出し・仕様照合。
  blender --background --factory-startup --python tools/building_preview.py -- OUT [params.json]
  params.json: {"part": "all"|"check"|"catalog"|"preview", "samples": 32, "res": [960, 640], "shots": ["01", ...], "emit": 5.0}
出力 OUT/NN_*.png(昼=夕方の光 / 夜。近景・遠景・LOD比較・屋上)、OUT/bld_*.blend, bld_*.glb、OUT/spec_check.json、OUT/catalog.json
hand_view.yml で動かす(チェイス本編とは無関係)。"""
import bpy, math, os, sys, json, time, random
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import building_kit as bk

ARGV = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else ['out']
OUT = ARGV[0]; os.makedirs(OUT, exist_ok=True)
PRM = json.load(open(ARGV[1])) if len(ARGV) > 1 else {}
PART = PRM.get('part', 'all'); SAMPLES = int(PRM.get('samples', 32)); RES = PRM.get('res', [960, 640]); EMIT = float(PRM.get('emit', 0.9))
ONLY = PRM.get('shots')


def log(*a):
    print('hand', *a, flush=True)


# ------------------------------------------------------------------ シーン
def fresh():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = SAMPLES
    sc.cycles.use_adaptive_sampling = True; sc.cycles.adaptive_threshold = 0.03
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = 'OPENIMAGEDENOISE'
    except Exception:
        pass
    sc.cycles.max_bounces = 8; sc.cycles.glossy_bounces = 6; sc.cycles.transmission_bounces = 8; sc.cycles.transparent_max_bounces = 24
    sc.render.resolution_x, sc.render.resolution_y = RES; sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    try:
        sc.view_settings.view_transform = 'AgX'
    except Exception:
        pass
    return sc


def make_world(sc, mode, sun_dir):
    w = bpy.data.worlds.new('W')
    try:
        w.use_nodes = True
    except Exception:
        pass
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputWorld'); bg = nt.nodes.new('ShaderNodeBackground')
    nb = bk.NB(nt)
    tc = nt.nodes.new('ShaderNodeTexCoord')
    nv = nt.nodes.new('ShaderNodeVectorMath'); nv.operation = 'NORMALIZE'; nt.links.new(tc.outputs['Generated'], nv.inputs[0])
    x, y, z = nb.sep(nv.outputs[0])
    if mode == 'day':       # 夕方の光(夕焼けの地平線)
        hor, zen, gro, glow = (1.0, 0.52, 0.26), (0.07, 0.15, 0.40), (0.05, 0.04, 0.04), (1.0, 0.42, 0.12)
    else:                   # 夜(薄明の名残)
        hor, zen, gro, glow = (0.030, 0.040, 0.085), (0.003, 0.006, 0.020), (0.004, 0.004, 0.005), (0.0, 0.0, 0.0)
    col = nb.vmix(nb.rgb(hor), nb.rgb(zen), nb.sstep(z, 0.0, 0.55))
    dot = nt.nodes.new('ShaderNodeVectorMath'); dot.operation = 'DOT_PRODUCT'; nt.links.new(nv.outputs[0], dot.inputs[0]); dot.inputs[1].default_value = tuple(sun_dir)
    g = nb.m('POWER', nb.m('MAXIMUM', dot.outputs['Value'], 0.0), 6.0)
    col = nb.vop('ADD', col, nb.vscale(nb.rgb(glow), nb.m('MULTIPLY', g, 1.6)))
    col = nb.vmix(col, nb.rgb(gro), nb.sstep(z, 0.0, -0.08))
    nt.links.new(col, bg.inputs['Color']); bg.inputs['Strength'].default_value = 1.0
    nt.links.new(bg.outputs[0], out.inputs['Surface'])
    sc.world = w


def add_sun(sc, direction_to_sun, strength, color, angle=0.03):
    ld = bpy.data.lights.new('sun', 'SUN'); ld.energy = strength; ld.color = color; ld.angle = angle
    lo = bpy.data.objects.new('sun', ld); sc.collection.objects.link(lo)
    lo.rotation_euler = (-Vector(direction_to_sun)).to_track_quat('-Z', 'Y').to_euler()


def add_ground(sc, size=900.0, color=(0.06, 0.06, 0.065)):
    me = bpy.data.meshes.new('ground')
    s = size / 2
    me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    o = bpy.data.objects.new('ground', me); sc.collection.objects.link(o)
    mat, nt, out_ = bk.new_material('asphalt')
    b = bk.principled(nt, **{'Base Color': (*color, 1.0), 'Roughness': 0.85})
    nt.links.new(b.outputs[0], out_.inputs['Surface'])
    me.materials.append(mat)


def add_street_lights(sc, locs, power=3000.0, color=(1.0, 0.72, 0.42)):
    for loc in locs:
        ld = bpy.data.lights.new('lamp', 'POINT'); ld.energy = power; ld.color = color; ld.shadow_soft_size = 0.3
        lo = bpy.data.objects.new('lamp', ld); lo.location = loc; sc.collection.objects.link(lo)


def set_cam(sc, pos, target, lens):
    cd = bpy.data.cameras.new('cam'); cd.lens = lens; cd.clip_end = 3000.0; cd.sensor_width = 36
    co = bpy.data.objects.new('cam', cd); sc.collection.objects.link(co); sc.camera = co
    co.location = pos
    co.rotation_euler = (Vector(target) - Vector(pos)).to_track_quat('-Z', 'Y').to_euler()


def build(kind, w, d, fl, seed, detail, night, hour=20.0, loc=(0, 0, 0), rotz=0.0, name=None, age=None):
    r = bk.make_building(kind, w, d, fl, seed=seed, detail=detail, night=night, hour=hour, name=name, emit=EMIT, age=age)
    r.location = loc; r.rotation_euler = (0, 0, rotz)
    return r


def render(sc, name):
    sc.render.filepath = os.path.join(OUT, name + '.png')
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    log('rendered', name, '%.0fs' % (time.time() - t0))


# ------------------------------------------------------------------ カット
SUN_DUSK = Vector((0.55, -0.80, 0.20)).normalized()      # 正面(-Y)側の低い夕日
SUN_NIGHT = Vector((-0.4, -0.5, 0.75)).normalized()


def scene_basic(mode):
    sc = fresh()
    sd = SUN_DUSK if mode == 'day' else SUN_NIGHT
    make_world(sc, mode, sd)
    if mode == 'day':
        add_sun(sc, sd, 4.0, (1.0, 0.70, 0.42))
    else:
        add_sun(sc, sd, 0.06, (0.65, 0.75, 1.0), angle=0.06)
    add_ground(sc)
    return sc


def shot_single(name, mode, spec, cam, extra_lights=None):
    sc = scene_basic(mode)
    kind, w, d, fl, seed, age = spec
    night = mode == 'night'
    build(kind, w, d, fl, seed, 'near', night, age=age)
    if mode == 'night':
        add_street_lights(sc, extra_lights or [(-7, -(d / 2) - 9, 6.0), (9, -(d / 2) - 10, 6.0)])
    set_cam(sc, *cam)
    render(sc, name)


def shot_lod(name, detail, cam, spec=('zakkyo', 11.7, 15.6, 8, 5, 0.75)):
    sc = scene_basic('day')
    kind, w, d, fl, seed, age = spec
    build(kind, w, d, fl, seed, detail, False, age=age)
    set_cam(sc, *cam)
    render(sc, name)


def street_buildings(mode, seed=5):
    """高架(x=0, 高さ9.5m)から見た街並み: 道の両側にビルを並べる。 近い順に near / mid / far"""
    rng = random.Random(seed)
    res = []
    for side in (+1, -1):
        y = 24.0 + (0 if side > 0 else 9.0)
        while y < 420.0:
            kind = rng.choices(['mansion', 'zakkyo', 'office'], [0.40, 0.33, 0.27])[0]
            if kind == 'mansion':
                w = 6.6 * rng.choice([3, 4, 5]); d = 13.2; fl = rng.randint(6, 14)
            elif kind == 'zakkyo':
                w = 3.9 * rng.choice([2, 3, 4]); d = 3.9 * rng.choice([3, 4]); fl = rng.randint(5, 9)
            else:
                w = 1.65 * rng.choice([10, 14, 18]); d = 1.65 * rng.choice([10, 14]); fl = rng.randint(9, 20)
            yc = y + w / 2
            dist = math.hypot(15.0, yc)
            detail = 'near' if dist < 70 else ('mid' if dist < 170 else 'far')
            res.append((kind, w, d, fl, rng.randint(1, 999), detail, side, yc))
            y += w + rng.uniform(1.0, 5.0)
    return res


def shot_street(name, mode):
    sc = scene_basic(mode)
    night = mode == 'night'
    n_near = 0
    for (kind, w, d, fl, seed, detail, side, yc) in street_buildings(mode):
        # 正面(-Y)が道(x=0)を向く: 右側は -90°、左側は +90°
        S = bk.SPECS[kind]
        W_ = max(S['minb'][0], round(w / S['P'])) * S['P']; D_ = max(S['minb'][1], round(d / S['P'])) * S['P']
        x = side * (15.0 + 1.6 + D_ / 2)
        build(kind, w, d, fl, seed, detail, night, 20.5, loc=(x, yc, 0), rotz=(-math.pi / 2 if side > 0 else math.pi / 2))
    set_cam(sc, (0.0, 0.0, 9.5), (0.0, 190.0, 17.0), 30)
    # 道路灯の代わりに、路面の高さに暖色の灯りを少し(夜)
    if night:
        add_street_lights(sc, [(0, 14, 7), (0, 38, 7), (0, 62, 7)], power=2500)
    render(sc, name)


def shot_roof(name, mode):
    sc = scene_basic(mode)
    night = mode == 'night'
    build('office', 33, 26.4, 16, 3, 'near', night, age=0.35)
    if mode == 'day':
        set_cam(sc, (15.0, -14.0, 73.0), (-1.0, 0.0, 66.5), 35)
    else:
        set_cam(sc, (46.0, -85.0, 58.0), (0.0, 0.0, 62.0), 55)
    render(sc, name)


SHOTS = [
    ('01_day_office_near', lambda n: shot_single(n, 'day', ('office', 33, 26.4, 16, 3, 0.35), ((13.0, -13.2 - 14.0, 9.0), (14.0, -13.2, 20.0), 30))),
    ('02_day_zakkyo_near', lambda n: shot_single(n, 'day', ('zakkyo', 11.7, 15.6, 8, 5, 0.75), ((5.5, -7.8 - 9.0, 9.5), (1.5, -7.8, 11.5), 35))),
    ('03_day_mansion_near', lambda n: shot_single(n, 'day', ('mansion', 26.4, 13.2, 10, 7, 0.6), ((9.0, -8.0 - 8.0, 12.0), (7.0, -8.0, 12.5), 35))),
    ('04_night_office_near', lambda n: shot_single(n, 'night', ('office', 33, 26.4, 16, 3, 0.35), ((13.0, -13.2 - 14.0, 9.0), (14.0, -13.2, 20.0), 30))),
    ('05_night_zakkyo_near', lambda n: shot_single(n, 'night', ('zakkyo', 11.7, 15.6, 8, 5, 0.75), ((5.5, -7.8 - 9.0, 9.5), (1.5, -7.8, 11.5), 35))),
    ('06_night_mansion_near', lambda n: shot_single(n, 'night', ('mansion', 26.4, 13.2, 10, 7, 0.6), ((9.0, -8.0 - 8.0, 12.0), (7.0, -8.0, 12.5), 35))),
    ('07_day_street', lambda n: shot_street(n, 'day')),
    ('08_night_street', lambda n: shot_street(n, 'night')),
    ('09_lod_near', lambda n: shot_lod(n, 'near', ((4.0, -7.8 - 14.0, 10.0), (0.0, -7.8, 11.5), 35))),
    ('10_lod_mid', lambda n: shot_lod(n, 'mid', ((4.0, -7.8 - 14.0, 10.0), (0.0, -7.8, 11.5), 35))),
    ('11_lod_far', lambda n: shot_lod(n, 'far', ((4.0, -7.8 - 14.0, 10.0), (0.0, -7.8, 11.5), 35))),
    ('12_day_roof_office', lambda n: shot_roof(n, 'day')),
    ('13_night_tower_top', lambda n: shot_roof(n, 'night')),
    ('14_day_zakkyo_side_stair', lambda n: shot_single(n, 'day', ('zakkyo', 11.7, 15.6, 8, 5, 0.75), ((5.85 + 13.0, -3.0, 12.0), (5.85, 0.0, 12.0), 28))),
    ('15_night_mansion_corridor', lambda n: shot_single(n, 'night', ('mansion', 26.4, 13.2, 10, 7, 0.6), ((-6.0, 6.6 + 1.5 + 9.0, 12.0), (-4.0, 6.6 + 1.5, 12.0), 35),
                                                         extra_lights=[(-8, 18, 6), (6, 18, 6)])),
]


# ------------------------------------------------------------------ カタログ(.blend / .glb)
CATALOG = [   # 名前, kind, 幅, 奥行き, 階数, seed, 築年数の指標(0=新築 … 1=築40年超)
    ('office_A', 'office', 33.0, 26.4, 16, 3, 0.35), ('office_B', 'office', 19.8, 16.5, 10, 11, 0.70),
    ('zakkyo_A', 'zakkyo', 11.7, 15.6, 8, 5, 0.75), ('zakkyo_B', 'zakkyo', 15.6, 11.7, 6, 21, 0.40),
    ('mansion_A', 'mansion', 26.4, 13.2, 10, 7, 0.60), ('mansion_B', 'mansion', 19.8, 13.2, 14, 12, 0.30),
]


def do_catalog():
    cdir = OUT                                           # hand_view.yml は out/*.blend out/*.glb を Release に上げる
    infos = {}
    for (nm, kind, w, d, fl, seed, age) in CATALOG:
        sc = fresh()
        roots = {}
        x = 0.0
        for det in ('near', 'mid', 'far'):
            r = build(kind, w, d, fl, seed, det, True, 20.0, loc=(x, 0, 0), name='%s_%s' % (nm, det), age=age)
            roots[det] = r; infos['%s_%s' % (nm, det)] = bk.building_info(r)
            x += bk.building_info(r)['width_m'] + 14.0
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(cdir, 'bld_' + nm + '.blend'), compress=True)
        for det in ('near', 'far'):                      # glb は材質を単純化して(元の .blend は上で保存済み)
            r = roots[det]
            r.location = (0, 0, 0)
            bk.simplify_materials(r)
            try:
                bk.export_glb(r, os.path.join(cdir, 'bld_%s_%s.glb' % (nm, det)))
            except Exception as e:
                log('ERR glb', nm, det, repr(e))
        log('catalog', nm, {k: (v['faces'], v['tris'], v['gen_seconds']) for k, v in infos.items() if k.startswith(nm)})
    json.dump(infos, open(os.path.join(OUT, 'catalog.json'), 'w'), ensure_ascii=False, indent=1)
    return infos


def do_check():
    rows = bk.spec_report()
    json.dump(rows, open(os.path.join(OUT, 'spec_check.json'), 'w'), ensure_ascii=False, indent=1)
    for r in rows:
        log('SPEC', 'OK' if r['ok'] else 'NG', r['item'], '|', r['spec'], '|', r['got'])


def main():
    t0 = time.time()
    if PART in ('all', 'check'):
        do_check()
    if PART in ('all', 'catalog'):
        do_catalog()
    if PART in ('all', 'preview'):
        for name, fn in SHOTS:
            if ONLY and not any(name.startswith(s) for s in ONLY):
                continue
            try:
                fn(name)
            except Exception as e:
                import traceback
                log('ERR', name, repr(e)); traceback.print_exc()
    log('total %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
