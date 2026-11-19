"""アセット庫(車・トラック・バイク・銃・手・ライダー・街)のプレビュー画像を作る。チェイス本編とは無関係。
  blender --background --factory-startup --python tools/asset_preview.py -- OUT [params.json]
  params.json: {"part": "assets" | "city" | "all", "samples": 24}
出力 OUT/*.png (600x600)。 1つの素材につき4枚(斜め前・横・上・後ろ)。ファイル名 NN_名前_a_front / b_side / c_top / d_back。
  OUT/asset_info.json = 寸法(Blender座標、m)・頂点数・三角数。
座標の約束(Blender): X=右、Y=前、Z=上。 路面 z=0。
assemble_vehicle(name) は chase/render.py の build_proto と同じ合体の手順(vehicles.json の値)を、材質の加工と灯りの追加なしでやる。
別のシーンへ持っていくときの参考実装として、アセット庫の コード/ にもコピーしてある。"""
import bpy, bmesh, json, math, os, sys, time
import numpy as np
from mathutils import Matrix, Vector

ARGV = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else ['out']
OUT = ARGV[0]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRM = json.load(open(ARGV[1])) if len(ARGV) > 1 else {}
PART = PRM.get('part', 'all')
SAMPLES = int(PRM.get('samples', 24))
CFG = json.load(open(os.path.join(ROOT, 'vehicles.json'), encoding='utf-8'))
INFO = {}
RES = 600


# ------------------------------------------------------------------ 読み込みと合体(build_proto と同じ式)
def co_array(me):
    a = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', a)
    return a.reshape(-1, 3)


def import_glb(path):
    """GLBを読み込み、親Empty(軸変換・スケール)を焼き込んで、一番大きいメッシュ1つを返す(ほかは捨てる)"""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path, import_shading='SMOOTH')
    new = [o for o in bpy.data.objects if o not in before]
    bpy.context.view_layer.update()
    meshes = [o for o in new if o.type == 'MESH']
    for o in meshes:
        mw = o.matrix_world.copy(); o.parent = None; o.data.transform(mw); o.matrix_world = Matrix.Identity(4)
    for o in new:
        if o.type != 'MESH':
            bpy.data.objects.remove(o)
    meshes.sort(key=lambda o: -len(o.data.vertices))
    for o in meshes[1:]:
        bpy.data.objects.remove(o)
    return meshes[0]


def make_opaque(mat):
    for n in mat.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            for l in list(n.inputs['Alpha'].links):
                mat.node_tree.links.remove(l)
            n.inputs['Alpha'].default_value = 1.0


def assemble_vehicle(name):
    """vehicles.json の name(veh_sedan など)を、車体+車輪に合体して返す -> [Object] (z=0 が路面、前=+Y、原点=車体の水平中心)"""
    c = CFG[name]
    rot = math.radians(c['rot_deg']) if 'rot_deg' in c else {'bike': math.pi, 'car': -math.pi / 2}[c['axes']]
    d = os.path.join(ROOT, 'assets', name)
    ob = import_glb(os.path.join(d, c.get('body_file', '2_nowheel.glb')))
    me = ob.data
    me.transform(Matrix.Rotation(rot, 4, 'Z'))
    co = co_array(me); lo, hi = co.min(0), co.max(0)
    s = c['L'] / (hi[1] - lo[1])                                   # 向きを直したあとの Y 方向の長さが L になる倍率
    me.transform(Matrix.Translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])))
    me.transform(Matrix.Diagonal((s, s, s, 1.0)))
    me.transform(Matrix.Translation((0, 0, c['body_z'])))
    me.update()
    objs = [ob]
    ow = import_glb(os.path.join(ROOT, 'assets', c.get('wheels_dir', name), '3_wheels.glb'))
    wme = ow.data
    wme.transform(Matrix.Rotation(math.pi / 2, 4, 'Z'))            # 車軸 -> X
    cache = {}
    for i, w in enumerate(c['wheels']):
        key = (w['island'], bool(w.get('flip')), w['r'], w.get('wscale', 1.0))
        if key not in cache:
            y0, y1 = c['wheel_islands'][w['island']]
            m2 = wme.copy()
            bm = bmesh.new(); bm.from_mesh(m2)
            dele = [f for f in bm.faces if not (y0 <= f.calc_center_median().y < y1)]
            bmesh.ops.delete(bm, geom=dele, context='FACES')
            bm.to_mesh(m2); bm.free()
            a = co_array(m2); wlo, whi = a.min(0), a.max(0)
            r0 = max(whi[1] - wlo[1], whi[2] - wlo[2]) / 2
            k = w['r'] / r0
            m2.transform(Matrix.Translation(tuple(-(wlo + whi) / 2)))
            m2.transform(Matrix.Diagonal((k * w.get('wscale', 1.0), k, k, 1.0)))
            if w.get('flip'):
                m2.transform(Matrix.Rotation(math.pi, 4, 'Z'))
            m2.update(); cache[key] = m2
        wo = bpy.data.objects.new('%s_w%d' % (name, i), cache[key]); bpy.context.scene.collection.objects.link(wo)
        wo.location = (w['x'], w['y'], w['r']); objs.append(wo)
    bpy.data.objects.remove(ow)
    return objs


# ------------------------------------------------------------------ 撮影
def bbox(objs):
    dg = bpy.context.evaluated_depsgraph_get(); lo = Vector((1e9,) * 3); hi = Vector((-1e9,) * 3)
    for o in objs:
        e = o.evaluated_get(dg)
        me = e.to_mesh()
        for v in me.vertices:
            p = e.matrix_world @ v.co
            lo = Vector(map(min, lo, p)); hi = Vector(map(max, hi, p))
        e.to_mesh_clear()
    return lo, hi


def tris(objs):
    dg = bpy.context.evaluated_depsgraph_get(); n = 0; v = 0
    for o in objs:
        e = o.evaluated_get(dg); me = e.to_mesh(); n += sum(len(p.vertices) - 2 for p in me.polygons); v += len(me.vertices); e.to_mesh_clear()
    return n, v


def setup_scene(std=True):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene; sc.render.engine = 'CYCLES'
    sc.cycles.samples = SAMPLES; sc.cycles.device = 'CPU'; sc.cycles.use_denoising = True
    sc.render.resolution_x = RES; sc.render.resolution_y = RES; sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGB'
    if std:
        sc.view_settings.view_transform = 'Standard'
    cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam
    return sc, cam


def reset_stage(sc):
    for o in list(bpy.data.objects):
        if o.name not in ('c',):
            bpy.data.objects.remove(o)
    for m in list(bpy.data.meshes):
        if m.users == 0:
            bpy.data.meshes.remove(m)
    w = bpy.data.worlds.new('w'); w.use_nodes = True
    w.node_tree.nodes['Background'].inputs[1].default_value = 0.35
    w.node_tree.nodes['Background'].inputs[0].default_value = (0.5, 0.52, 0.58, 1)
    sc.world = w


def stage(sc, objs, frame=None):
    """台と照明を、物の大きさに合わせて置く -> (中心, 大きさ)"""
    lo, hi = bbox(objs); c = (lo + hi) / 2; S = max(hi - lo)
    if frame:
        c, S = Vector(frame[0]), frame[1]
    rad = max(hi.x - lo.x, hi.y - lo.y) * 0.62 + S * 0.05
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=rad, depth=S * 0.01, location=(c.x, c.y, lo.z - S * 0.005))
    base = bpy.context.object; mat = bpy.data.materials.new('base'); mat.use_nodes = True
    mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.06, 0.065, 0.075, 1); mat.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.8
    base.data.materials.append(mat)
    for loc, e in (((-1.2, -1.6, 1.4), 450), ((1.6, -0.8, 1.0), 300), ((0.2, 1.8, 1.2), 200)):
        ld = bpy.data.lights.new('l', 'AREA'); ld.energy = e * S * S * 0.35; ld.size = S
        lo_ = bpy.data.objects.new('l', ld); lo_.location = c + Vector(loc) * S * 1.4; sc.collection.objects.link(lo_)
        lo_.rotation_euler = (c - lo_.location).to_track_quat('-Z', 'Y').to_euler()
    return c, S


VIEWS = (('a_front', (0.75, 1.0, 0.55)), ('b_side', (1.0, 0.0, 0.12)), ('c_top', (0.0, -0.03, 1.0)), ('d_back', (-0.75, -1.0, 0.55)))


def shoot_asset(sc, cam, tag, objs, views=VIEWS, dist_mul=2.3, extra=None, frame=None):
    c, S = stage(sc, objs, frame)
    cam.data.lens = 50; cam.data.clip_start = S * 0.01; cam.data.clip_end = S * 50
    for vn, d in views:
        d = Vector(d).normalized(); cam.location = c + d * S * dist_mul
        cam.rotation_euler = (c - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = os.path.join(OUT, '%s_%s.png' % (tag, vn)); bpy.ops.render.render(write_still=True)
    lo, hi = bbox(objs); n, v = tris(objs)
    INFO[tag] = {'bbox_min': [round(x, 4) for x in lo], 'bbox_max': [round(x, 4) for x in hi], 'size_xyz': [round(x, 4) for x in (hi - lo)], 'tris': n, 'verts': v}
    if extra:
        INFO[tag].update(extra)
    print('asset', tag, INFO[tag], flush=True)


def part_assets():
    sc, cam = setup_scene()
    n = 0
    for name in ('veh_sedan', 'veh_suv', 'veh_hatch', 'veh_truck', 'bike_A', 'bike_B', 'bike_B2'):
        n += 1; t0 = time.time()
        try:
            reset_stage(sc)
            objs = assemble_vehicle(name)
            shoot_asset(sc, cam, '%02d_%s' % (n, name), objs, dist_mul=1.75 if name != 'veh_truck' else 1.6)
        except Exception as e:
            import traceback; traceback.print_exc(); print('ERR', name, e, flush=True)
        print('time', name, round(time.time() - t0, 1), flush=True)
    # 銃・手・ライダーは、chase/rider.py の関数で読み込む(本編と同じ向きにそろう)
    sys.path.insert(0, os.path.join(ROOT, 'chase')); sys.path.insert(0, os.path.join(ROOT, 'tools'))
    d = os.path.join(ROOT, 'assets', 'bike_B2')
    R = RD = None
    # 8: 銃(本編の import_gun と同じ補正: 原点=グリップ中心、銃口=+Y、上=+Z)
    try:
        import render as R
        R.vehicle_material = lambda *a, **k: None           # 本編の色替え・かすみの加工は付けない
        import rider as RD
        reset_stage(sc)
        gun = RD.import_gun(R, d)
        gun.rotation_mode = 'QUATERNION'
        shoot_asset(sc, cam, '08_gun_normalized', [gun], dist_mul=1.9, extra={'note': 'rider.import_gun で補正後: 原点=グリップ中心, 銃口=+Y, 上=+Z'})
    except Exception as e:
        import traceback; traceback.print_exc(); print('ERR gun', e, flush=True); R = None
    # 9: 手(開いた静止姿勢)。 GLBを素のまま読み込んだ向き
    try:
        reset_stage(sc)
        h = import_glb(os.path.join(d, 'hand_open.glb')); [make_opaque(m) for m in h.data.materials]
        shoot_asset(sc, cam, '09_hand_open_raw', [h], dist_mul=1.9, extra={'note': 'GLBを読み込んだままの向き(Blender上で指=+Z)'})
    except Exception as e:
        import traceback; traceback.print_exc(); print('ERR hand_open', e, flush=True)
    # 10: 手に指ボーンを付け、銃を握らせた状態(ready と fire)
    try:
        if R is None:
            raise RuntimeError('render module unavailable')
        reset_stage(sc)
        hd = RD.build_hand(R, d); gun = RD.import_gun(R, d)
        hd['ob'].hide_render = False
        ao = hd['ao']; ao.rotation_mode = 'QUATERNION'
        T = Matrix(hd['T'].tolist()); ao.matrix_world = T                # 手のメッシュ→銃の座標。 銃は原点のまま
        views = (('a_front', (0.9, 1.0, 0.7)), ('b_side', (1.0, 0.0, 0.1)), ('c_top', (0.0, -0.03, 1.0)))
        shoot_asset(sc, cam, '10_hand_grip_ready', [hd['ob'], gun], views=views, dist_mul=1.9,
                    extra={'note': 'hand_grip_poses.json の ready 状態。銃=原点、手=hand_to_gun'})
        for bn in ('Index1', 'Index2', 'Index3'):
            pb = ao.pose.bones[bn]; pb.rotation_quaternion = tuple(hd['q']['fire'][bn])
        bpy.context.view_layer.update()
        lo_, hi_ = bbox([hd['ob'], gun]); c = (lo_ + hi_) / 2; S = max(hi_ - lo_)
        d_ = Vector((0.9, 1.0, 0.7)).normalized(); cam.location = c + d_ * S * 1.9
        cam.rotation_euler = (c - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = os.path.join(OUT, '10_hand_grip_ready_d_fire.png'); bpy.ops.render.render(write_still=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print('ERR hand grip', e, flush=True)
    # 11: ライダー(Meshy自動リグ)。 静止姿勢のまま
    try:
        reset_stage(sc)
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=os.path.join(d, 'rider_rigged.glb'), import_shading='SMOOTH')
        new = [o for o in bpy.data.objects if o not in before]
        meshes = [o for o in new if o.type == 'MESH']
        for o in meshes:
            [make_opaque(m) for m in o.data.materials]
        for o in new:
            if o.type not in ('ARMATURE', 'MESH'):
                bpy.data.objects.remove(o)
        bpy.context.view_layer.update()
        ao = next((o for o in new if o.type == 'ARMATURE' and o.name in bpy.data.objects), None)
        ex = {'armature_scale': [round(x, 4) for x in ao.scale] if ao else None, 'bones': len(ao.data.bones) if ao else 0}
        pts = [ao.matrix_world @ b.head_local for b in ao.data.bones] if ao else []
        if pts:
            blo = Vector(map(min, *pts)); bhi = Vector(map(max, *pts)); ex['bones_bbox_size'] = [round(x, 3) for x in (bhi - blo)]
            for o in meshes:
                ex.setdefault('mesh_names', []).append(o.name)
            fr = ((blo + bhi) / 2, max(bhi - blo) * 1.15)
        else:
            fr = None
        shoot_asset(sc, cam, '11_rider_rigged', meshes, dist_mul=1.9, extra=ex, frame=fr)
    except Exception as e:
        import traceback; traceback.print_exc(); print('ERR rider', e, flush=True)


def part_city():
    """chase/city.py で街を組み立て、上から・斜めから・高架の目線から撮る(本編と同じ空と路面)"""
    sys.path.insert(0, os.path.join(ROOT, 'chase')); sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import render as R
    R.P['motion_blur'] = 0
    sc, cam = setup_scene(std=False)
    R.setup_render(SAMPLES, RES, RES)
    sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGB'; sc.render.image_settings.color_depth = '8'
    sc.cycles.film_exposure = 1.0
    try:
        sc.view_settings.view_transform = 'AgX'
    except Exception:
        pass
    t0 = time.time()
    R.build_world(); R.build_road()
    import city
    city.build(R)
    print('city built %.1fs, objects %d' % (time.time() - t0, len(bpy.data.objects)), flush=True)
    cj = json.load(open(os.path.join(ROOT, 'assets', 'city.json'), encoding='utf-8'))
    ys = np.array([b['p'][0][1] for b in cj['buildings'] if abs(b['p'][0][0]) < 200 and b['t'] in ('office', 'tower', 'apt')])
    best = max(range(0, 4000, 50), key=lambda y0: int(((ys >= y0) & (ys < y0 + 200)).sum()))
    yc = best + 100.0
    print('city window y=%d..%d (offices/towers/apts: %d)' % (best, best + 200, int(((ys >= best) & (ys < best + 200)).sum())), flush=True)
    cam.data.clip_start = 0.1; cam.data.clip_end = 9000
    shots = (
        ('c1_top', (-6, yc, 420), (-6, yc + 0.5, 0), 40.0),                     # 真上から
        ('c2_aerial', (-6, yc - 190, 120), (-6, yc + 60, 8), 40.0),             # 斜め上から高架と両側のビル
        ('c3_deck_eye', (-3, yc - 30, 2.6), (-8, yc + 90, 14), 40.0),           # 高架の上(車の目線)から前方
        ('c4_facade', (-5, yc, 9), (-60, yc + 30, 22), 36.0),                    # 高架から左のビルの壁を近くで
    )
    for nm, p, look, fov in shots:
        cam.location = Vector(p); cam.rotation_euler = (Vector(look) - Vector(p)).to_track_quat('-Z', 'Y').to_euler()
        cam.data.lens = 18.0 / math.tan(math.radians(fov) / 2)
        sc.render.filepath = os.path.join(OUT, '20_city_%s.png' % nm); t1 = time.time(); bpy.ops.render.render(write_still=True)
        print('shot', nm, round(time.time() - t1, 1), flush=True)
    INFO['city'] = {'buildings': len(cj['buildings']), 'roads': len(cj['roads']), 'green': len(cj['green']), 'water': len(cj['water']), 'trees': len(cj['trees']),
                    'window_y': [best, best + 200], 'build_seconds': round(time.time() - t0, 1)}


if __name__ == '__main__':          # 別のスクリプトから import して assemble_vehicle だけ使えるようにする
    os.makedirs(OUT, exist_ok=True)
    if PART in ('assets', 'all'):
        part_assets()
    if PART in ('city', 'all'):
        try:
            part_city()
        except Exception as e:
            import traceback; traceback.print_exc(); print('ERR city', e, flush=True)
    json.dump(INFO, open(os.path.join(OUT, 'asset_info_%s.json' % PART), 'w'), ensure_ascii=False, indent=1)
    print('done', PART, flush=True)
