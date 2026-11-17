"""高速チェイス(写実版)の組み立てと描画。 blender --background --factory-startup --python chase/render.py -- [args]
  --frames "24,84,100-110"   描画するコマ(0始まり、t = f/24)
  --out DIR                  EXR の出力先
  --samples N  --res WxH
  --params JSON              P の上書き
Blender の座標: X=右, Y=前, Z=上。動きは motion.py (three.js 座標) を to_blender() で変換する。"""
import bpy, bmesh, json, math, os, sys, time
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import motion as M   # noqa: E402

CFG = json.load(open(os.path.join(ROOT, 'vehicles.json'), encoding='utf-8'))

P = {
    'world': 0.6,            # HDRI の強さ
    'sky_rot': 0.911,        # 夕焼けを進行方向(+Y)に向ける回転
    'film_exposure': 0.5,
    'lamp_power': 26000.0,   # 街灯(スポット)
    'lamp_color': [1.0, 0.60, 0.27],
    'lamp_emit': 500.0,
    'head_power': 14000.0,   # 車のヘッドライト(1台1灯のスポット)
    'bike_head_power': 12000.0,
    'head_emit': 500.0,
    'tail_emit': 40.0,
    'tail_point': 90.0,
    'tail_mask_emit': 60.0,
    'win_emit': 3.0,
    'city_emit': 30.0,
    'fog_dist': 1100.0,
    'fog_cool': [0.50, 0.42, 0.60],
    'fog_warm': [1.2, 0.72, 0.72],
    'coat': 0.7,
    'body_rough_mul': 0.75,
    'motion_blur': 1,
    'shutter': 0.5,
    'wheel_spin_range': 160.0,
    'clamp_indirect': 8.0,
    'adaptive': 0.03,
    'ground_z': -12.0,
}

Y0, Y1 = -300.0, 2700.0          # road extent (three.js Z0=300 .. Z1=-2700)
LEN = Y1 - Y0
YC = (Y0 + Y1) / 2


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def link(obj, parent=None):
    bpy.context.scene.collection.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    return obj


def empty(name, parent=None):
    o = bpy.data.objects.new(name, None)
    o.empty_display_size = 0.2
    return link(o, parent)


def clear_nodes(tree):
    for n in list(tree.nodes):
        tree.nodes.remove(n)


def new_material(name):
    mat = bpy.data.materials.new(name)
    try:
        mat.use_nodes = True
    except Exception:
        pass
    nt = mat.node_tree
    clear_nodes(nt)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    return mat, nt, out


def principled(nt, out, **kw):
    b = nt.nodes.new('ShaderNodeBsdfPrincipled')
    for k, v in kw.items():
        b.inputs[k].default_value = v
    nt.links.new(b.outputs['BSDF'], out.inputs['Surface'])
    return b


def math_node(nt, op, a=None, b=None, clamp=False):
    n = nt.nodes.new('ShaderNodeMath'); n.operation = op; n.use_clamp = clamp
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[0]


def vmath(nt, op, a=None, b=None, scale=None):
    n = nt.nodes.new('ShaderNodeVectorMath'); n.operation = op
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (tuple, list)):
            n.inputs[i].default_value = v[:3]
        else:
            nt.links.new(v, n.inputs[i])
    if scale is not None:
        if isinstance(scale, (int, float)):
            n.inputs[3].default_value = scale
        else:
            nt.links.new(scale, n.inputs[3])
    return n.outputs[0]


def vlerp(nt, a, b, f):
    """a + (b - a) * f  (a, b: colour sockets or tuples; f: float socket)"""
    d = vmath(nt, 'SUBTRACT', b, a)
    return vmath(nt, 'ADD', a, vmath(nt, 'SCALE', d, scale=f))


_fog_group = None


def fog_group():
    """距離のかすみ: カメラからの距離で、空の色(進行方向は夕焼け色)へ混ぜる。"""
    global _fog_group
    if _fog_group is not None:
        return _fog_group
    g = bpy.data.node_groups.new('Fog', 'ShaderNodeTree')
    g.interface.new_socket(name='Shader', in_out='INPUT', socket_type='NodeSocketShader')
    g.interface.new_socket(name='Shader', in_out='OUTPUT', socket_type='NodeSocketShader')
    gi = g.nodes.new('NodeGroupInput'); go = g.nodes.new('NodeGroupOutput')
    cam = g.nodes.new('ShaderNodeCameraData')
    d = math_node(g, 'DIVIDE', cam.outputs['View Distance'], -P['fog_dist'])
    fac = math_node(g, 'SUBTRACT', 1.0, math_node(g, 'EXPONENT', d), clamp=True)
    geo = g.nodes.new('ShaderNodeNewGeometry')
    sep = g.nodes.new('ShaderNodeSeparateXYZ'); g.links.new(geo.outputs['Incoming'], sep.inputs[0])
    ahead = math_node(g, 'POWER', math_node(g, 'MULTIPLY', sep.outputs['Y'], -1.0, clamp=True), 3.0)
    col = vlerp(g, tuple(c * P['world'] for c in P['fog_cool']), tuple(c * P['world'] for c in P['fog_warm']), ahead)
    em = g.nodes.new('ShaderNodeEmission'); g.links.new(col, em.inputs['Color']); em.inputs['Strength'].default_value = 1.0
    mix = g.nodes.new('ShaderNodeMixShader')
    g.links.new(fac, mix.inputs[0]); g.links.new(gi.outputs[0], mix.inputs[1]); g.links.new(em.outputs[0], mix.inputs[2])
    g.links.new(mix.outputs[0], go.inputs[0])
    _fog_group = g
    return g


def add_fog(mat):
    nt = mat.node_tree
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
    if not out.inputs['Surface'].links:
        return
    src = out.inputs['Surface'].links[0].from_socket
    gn = nt.nodes.new('ShaderNodeGroup'); gn.node_tree = fog_group()
    nt.links.new(src, gn.inputs[0]); nt.links.new(gn.outputs[0], out.inputs['Surface'])


def emission_material(name, color, strength, one_sided=False):
    mat, nt, out = new_material(name)
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Color'].default_value = (*color, 1.0); em.inputs['Strength'].default_value = strength
    if one_sided:                                  # 裏側からは光って見えないようにする
        geo = nt.nodes.new('ShaderNodeNewGeometry')
        nt.links.new(math_node(nt, 'MULTIPLY', math_node(nt, 'SUBTRACT', 1.0, geo.outputs['Backfacing']), strength), em.inputs['Strength'])
    nt.links.new(em.outputs[0], out.inputs['Surface'])
    add_fog(mat)
    return mat


def simple_material(name, color, rough=0.6, metallic=0.0, fog=True):
    mat, nt, out = new_material(name)
    principled(nt, out, **{'Base Color': (*color, 1.0), 'Roughness': rough, 'Metallic': metallic})
    if fog:
        add_fog(mat)
    return mat


def mesh_object(name, verts, faces, mats=(), uvs=None, face_mat=None, parent=None, smooth=False):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    if uvs is not None:
        uvl = me.uv_layers.new(name='UVMap')
        uvl.data.foreach_set('uv', np.asarray(uvs, dtype=np.float32).ravel())
    for m in mats:
        me.materials.append(m)
    if face_mat is not None:
        me.polygons.foreach_set('material_index', np.asarray(face_mat, dtype=np.int32))
    if smooth:
        me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
    me.update()
    return link(bpy.data.objects.new(name, me), parent)


class Geo:
    """箱・四角をためて、1つのメッシュにする。"""
    def __init__(self):
        self.v = []; self.f = []

    def box(self, cx, cy, cz, sx, sy, sz):
        n = len(self.v)
        hx, hy, hz = sx / 2, sy / 2, sz / 2
        for dx in (-hx, hx):
            for dy in (-hy, hy):
                for dz in (-hz, hz):
                    self.v.append((cx + dx, cy + dy, cz + dz))
        for q in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            self.f.append(tuple(n + i for i in q))

    def quad(self, pts):
        n = len(self.v)
        self.v.extend(pts); self.f.append((n, n + 1, n + 2, n + 3))

    def hquad(self, x0, x1, y0, y1, z):
        self.quad([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)])

    def obj(self, name, mat, parent=None):
        return mesh_object(name, self.v, self.f, mats=[mat], parent=parent)


# ------------------------------------------------------------------------------------------------
# world, road, city
# ------------------------------------------------------------------------------------------------
def build_world():
    w = bpy.data.worlds.new('World')
    try:
        w.use_nodes = True
    except Exception:
        pass
    bpy.context.scene.world = w
    nt = w.node_tree; clear_nodes(nt)
    out = nt.nodes.new('ShaderNodeOutputWorld')
    bg = nt.nodes.new('ShaderNodeBackground'); bg.inputs['Strength'].default_value = P['world']
    env = nt.nodes.new('ShaderNodeTexEnvironment'); env.image = bpy.data.images.load(os.path.join(ROOT, 'assets', 'tex', 'sky.hdr'))
    tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Rotation'].default_value = (0, 0, P['sky_rot'])
    nt.links.new(tc.outputs['Generated'], mp.inputs['Vector']); nt.links.new(mp.outputs['Vector'], env.inputs['Vector'])
    nt.links.new(env.outputs['Color'], bg.inputs['Color']); nt.links.new(bg.outputs[0], out.inputs['Surface'])


def asphalt_material():
    mat, nt, out = new_material('Asphalt')
    b = principled(nt, out)
    tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = (0.5, 0.5, 0.5)
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])

    def tex(fn, noncolor):
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = bpy.data.images.load(os.path.join(ROOT, 'assets', 'tex', fn))
        if noncolor:
            t.image.colorspace_settings.name = 'Non-Color'
        nt.links.new(mp.outputs['Vector'], t.inputs['Vector'])
        return t
    d = tex('asphalt_diff.jpg', False); r = tex('asphalt_rough.jpg', True); n = tex('asphalt_nor.jpg', True)
    # large-scale variation so that the 2 m tile does not read as a pattern (tyre-polished lanes are a little darker and smoother)
    noise = nt.nodes.new('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value = 0.07; noise.inputs['Detail'].default_value = 3.0
    nt.links.new(tc.outputs['Object'], noise.inputs['Vector'])
    var = math_node(nt, 'ADD', math_node(nt, 'MULTIPLY', noise.outputs[0], 0.5), 0.75)
    nt.links.new(vmath(nt, 'SCALE', d.outputs['Color'], scale=var), b.inputs['Base Color'])
    nt.links.new(math_node(nt, 'MULTIPLY', r.outputs['Color'], 0.85), b.inputs['Roughness'])
    nm = nt.nodes.new('ShaderNodeNormalMap'); nm.inputs['Strength'].default_value = 1.0
    nt.links.new(n.outputs['Color'], nm.inputs['Color']); nt.links.new(nm.outputs['Normal'], b.inputs['Normal'])
    add_fog(mat)
    return mat


def build_road():
    asphalt = asphalt_material()
    g = Geo(); g.hquad(-22.4, 9.6, Y0 - 20, Y1 + 20, 0.0); g.obj('Deck', asphalt)

    paint = simple_material('Paint', (0.78, 0.78, 0.76), rough=0.55)
    g = Geo()
    for lx in (-3.6, 0.0, 3.6, -16.5):
        y = -300.0
        while y < 2700:
            g.hquad(lx - 0.09, lx + 0.09, y - 1.8, y + 1.8, 0.006)
            y += 9.0
    for ex in (-7.6, 7.6, -21.1, -11.9):
        g.hquad(ex - 0.1, ex + 0.1, Y0, Y1, 0.006)
    g.obj('LaneMarks', paint)

    concrete = simple_material('Concrete', (0.42, 0.42, 0.41), rough=0.85)
    g = Geo()
    g.box(9.35, YC, -0.5, 0.3, LEN + 40, 3.0)            # right parapet (top at z = 1.0)
    g.box(-22.25, YC, -0.5, 0.3, LEN + 40, 3.0)          # far parapet
    g.box(-12.7, YC, 0.5, 0.5, LEN + 40, 1.0)            # median barrier
    g.obj('Barriers', concrete)

    steel = simple_material('Galvanized', (0.62, 0.64, 0.66), rough=0.42, metallic=1.0)
    g = Geo()
    prof = [(0.04, 0.44), (0.0, 0.48), (0.0, 0.54), (0.05, 0.60), (0.05, 0.64), (0.0, 0.70), (0.0, 0.76), (0.04, 0.80)]
    for rx, sgn in ((8.6, 1.0), (-9.4, -1.0)):                       # corrugated face towards the road
        for (d0, z0), (d1, z1) in zip(prof[:-1], prof[1:]):
            g.quad([(rx + sgn * d0, Y0, z0), (rx + sgn * d0, Y1, z0), (rx + sgn * d1, Y1, z1), (rx + sgn * d1, Y0, z1)])
        y = Y0
        while y < Y1:
            g.box(rx + sgn * 0.11, y, 0.39, 0.10, 0.12, 0.78)
            y += 4.0
    g.obj('GuardRails', steel)

    # street lamps: right side + median (double arm), every 45 m
    pole = simple_material('Pole', (0.30, 0.31, 0.33), rough=0.5, metallic=0.9)
    lamp_emit = emission_material('LampHead', P['lamp_color'], P['lamp_emit'])
    gp = Geo(); ge = Geo(); spots = []
    y = -280.0
    while y < Y1:
        gp.box(9.0, y, 5.0, 0.20, 0.20, 10.0); gp.box(8.05, y, 10.0, 2.1, 0.12, 0.12); gp.box(7.2, y, 9.97, 0.9, 0.42, 0.14)
        ge.hquad(6.82, 7.58, y - 0.17, y + 0.17, 9.895); spots.append((7.2, y, 9.85))
        ym = y - 22.0
        gp.box(-12.7, ym, 5.5, 0.22, 0.22, 9.0); gp.box(-12.7, ym, 10.0, 6.6, 0.12, 0.12)
        for hx in (-16.0, -9.4):
            gp.box(hx, ym, 9.97, 0.9, 0.42, 0.14); ge.hquad(hx - 0.38, hx + 0.38, ym - 0.17, ym + 0.17, 9.895); spots.append((hx, ym, 9.85))
        y += 45.0
    gp.obj('LampPoles', pole); ge.obj('LampHeads', lamp_emit)
    ld = bpy.data.lights.new('StreetLamp', 'SPOT')
    ld.energy = P['lamp_power']; ld.color = P['lamp_color']; ld.spot_size = math.radians(150); ld.spot_blend = 0.6; ld.shadow_soft_size = 0.25
    for i, p in enumerate(spots):
        o = bpy.data.objects.new('Lamp%03d' % i, ld); o.location = p; link(o)
    print('street lamps:', len(spots))


def facade_material():
    """ビルの外壁: 1.8 m ごとの窓、5.4 m ごとの部屋単位で点灯。棟ごとに点灯率を変える。"""
    mat, nt, out = new_material('Facade')
    b = principled(nt, out)
    uv = nt.nodes.new('ShaderNodeTexCoord')
    sep = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(uv.outputs['UV'], sep.inputs[0])
    u, v = sep.outputs['X'], sep.outputs['Y']

    def noise2(a, b_):
        comb = nt.nodes.new('ShaderNodeCombineXYZ'); nt.links.new(a, comb.inputs[0])
        if isinstance(b_, (int, float)):
            comb.inputs[1].default_value = b_
        else:
            nt.links.new(b_, comb.inputs[1])
        wn = nt.nodes.new('ShaderNodeTexWhiteNoise'); wn.noise_dimensions = '2D'; nt.links.new(comb.outputs[0], wn.inputs['Vector'])
        sc = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(wn.outputs['Color'], sc.inputs[0])
        return wn.outputs['Value'], sc
    cv = math_node(nt, 'DIVIDE', v, 3.3); iv = math_node(nt, 'FLOOR', cv); fv = math_node(nt, 'FRACT', cv)
    cu = math_node(nt, 'DIVIDE', u, 1.8); iu = math_node(nt, 'FLOOR', cu); fu = math_node(nt, 'FRACT', cu)
    room_u = math_node(nt, 'FLOOR', math_node(nt, 'DIVIDE', u, 5.4))
    bid = math_node(nt, 'FLOOR', math_node(nt, 'DIVIDE', u, 1000.0))
    r_room, sc_room = noise2(room_u, iv)
    r_win, _ = noise2(iu, math_node(nt, 'ADD', iv, 500.0))
    r_bld, _ = noise2(bid, 7.0)
    win = math_node(nt, 'MULTIPLY', math_node(nt, 'MULTIPLY', math_node(nt, 'GREATER_THAN', fu, 0.10), math_node(nt, 'LESS_THAN', fu, 0.90)),
                    math_node(nt, 'MULTIPLY', math_node(nt, 'GREATER_THAN', fv, 0.30), math_node(nt, 'LESS_THAN', fv, 0.80)))
    thr = math_node(nt, 'ADD', 0.52, math_node(nt, 'MULTIPLY', r_bld, 0.44))
    lit = math_node(nt, 'MULTIPLY', math_node(nt, 'GREATER_THAN', r_room, thr), math_node(nt, 'GREATER_THAN', r_win, 0.22))
    e = math_node(nt, 'MULTIPLY', win, lit)
    col = vlerp(nt, (1.0, 0.70, 0.40), (0.75, 0.86, 1.0), math_node(nt, 'GREATER_THAN', sc_room.outputs[1], 0.70))
    bright = math_node(nt, 'ADD', math_node(nt, 'POWER', sc_room.outputs[2], 2.0), 0.12)
    strength = math_node(nt, 'MULTIPLY', e, math_node(nt, 'MULTIPLY', bright, P['win_emit']))
    nt.links.new(col, b.inputs['Emission Color']); nt.links.new(strength, b.inputs['Emission Strength'])
    wall = vlerp(nt, (0.055, 0.055, 0.06), (0.12, 0.115, 0.11), r_bld)
    nt.links.new(vlerp(nt, wall, (0.012, 0.014, 0.02), win), b.inputs['Base Color'])
    nt.links.new(math_node(nt, 'SUBTRACT', 0.75, math_node(nt, 'MULTIPLY', win, 0.65)), b.inputs['Roughness'])
    add_fog(mat)
    return mat


def build_city():
    gz = P['ground_z']
    facade = facade_material(); roof = simple_material('Roof', (0.05, 0.05, 0.055), rough=0.9)
    V, F, UV, FM = [], [], [], []

    def building(i, bx, by, bw, bh, bd):
        x0, x1, y0, y1, z0, z1 = bx - bw / 2, bx + bw / 2, by - bd / 2, by + bd / 2, gz, bh
        n = len(V)
        V.extend([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)])
        u = 1000.0 * i; h = z1 - z0
        for (a, b_), wlen in (((0, 1), bw), ((1, 2), bd), ((2, 3), bw), ((3, 0), bd)):
            F.append((n + a, n + b_, n + b_ + 4, n + a + 4)); FM.append(0)
            UV.extend([(u, 0), (u + wlen, 0), (u + wlen, h), (u, h)]); u += wlen + 50
        F.append((n + 4, n + 5, n + 6, n + 7)); FM.append(1); UV.extend([(0, 0)] * 4)

    r = M.rng(77); i = 0
    for _ in range(150):                                   # scene.js と同じ並び
        side = -1 if r() < 0.5 else 1
        bx = side * (45 + r() * 220); bz = 300 - r() * (3000 + 200); bw = 10 + r() * 26; bh = 22 + r() * 110 * (0.4 + r()); bd = 10 + r() * 24
        building(i, bx, -bz, bw, bh, bd); i += 1
    for _ in range(90):                                    # far skyline against the afterglow
        bx = (r() - 0.5) * 900; bz = -2700 - 120 - r() * 500; bw = 20 + r() * 40; bh = 40 + r() * 200; bd = 20 + r() * 30
        building(i, bx, -bz, bw, bh, bd); i += 1
    r2 = M.rng(4242)
    for _ in range(420):                                   # lower blocks close to the viaduct (roadside parallax)
        side = -1 if r2() < 0.5 else 1
        bx = (36 if side > 0 else -50) * 1.0 + side * r2() * 150; by = -350 + r2() * 3300; bw = 12 + r2() * 30; bd = 12 + r2() * 30; bh = -4 + r2() * 26
        building(i, bx, by, bw, bh, bd); i += 1
    mesh_object('Buildings', V, F, mats=[facade, roof], uvs=UV, face_mat=FM)

    ground = simple_material('CityGround', (0.02, 0.02, 0.022), rough=0.9)
    g = Geo(); g.hquad(-1500, 1500, -1200, 4200, gz); g.obj('Ground', ground)
    # street-level lights below the viaduct
    g = Geo(); r3 = M.rng(99)
    for _ in range(2600):
        x = (r3() - 0.5) * 900; y = -400 + r3() * 3400
        if -26 < x < 13:
            continue
        s = 0.5 + r3() * 0.9
        g.hquad(x - s, x + s, y - s, y + s, gz + 0.3 + r3() * 6)
    g.obj('CityLights', emission_material('CityLight', (1.0, 0.70, 0.40), P['city_emit']))
    print('buildings:', i)


# ------------------------------------------------------------------------------------------------
# vehicles
# ------------------------------------------------------------------------------------------------
def import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    bpy.context.view_layer.update()
    meshes = [o for o in new if o.type == 'MESH']
    for o in meshes:                                       # 親Empty(軸変換・スケール)を焼き込む
        mw = o.matrix_world.copy()
        o.parent = None
        o.data.transform(mw)
        o.matrix_world = Matrix.Identity(4)
    for o in new:
        if o.type != 'MESH':
            bpy.data.objects.remove(o)
    if len(meshes) != 1:
        print('WARNING: %s has %d mesh objects; using the largest' % (path, len(meshes)))
        meshes.sort(key=lambda o: -len(o.data.vertices))
    return meshes[0]


def co_array(me):
    a = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', a)
    return a.reshape(-1, 3)


def vehicle_material(mat, kind, tail_mask):
    """Meshyの材質に: 色替え(オブジェクトの色=掛け算、アルファ=色相)、塗装のツヤ、テール発光、かすみ。"""
    nt = mat.node_tree
    b = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if b is None:
        return
    base_links = b.inputs['Base Color'].links
    if base_links:
        src = base_links[0].from_socket
        oi = nt.nodes.new('ShaderNodeObjectInfo')
        if kind == 'body':
            hs = nt.nodes.new('ShaderNodeHueSaturation')
            nt.links.new(src, hs.inputs['Color']); nt.links.new(oi.outputs['Alpha'], hs.inputs['Hue'])
            nt.links.new(vmath(nt, 'MULTIPLY', hs.outputs['Color'], oi.outputs['Color']), b.inputs['Base Color'])
            if tail_mask:                                  # 赤いテールランプの部分を光らせる
                sc = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(src, sc.inputs[0])
                m = math_node(nt, 'SUBTRACT', sc.outputs[0], math_node(nt, 'MAXIMUM', sc.outputs[1], sc.outputs[2]))
                m = math_node(nt, 'MULTIPLY', math_node(nt, 'SUBTRACT', m, 0.32), 6.0, clamp=True)
                nt.links.new(src, b.inputs['Emission Color'])
                nt.links.new(math_node(nt, 'MULTIPLY', m, P['tail_mask_emit']), b.inputs['Emission Strength'])
    if kind == 'body':
        try:
            b.inputs['Coat Weight'].default_value = P['coat']; b.inputs['Coat Roughness'].default_value = 0.06
        except Exception as e:
            print('coat:', e)
        rl = b.inputs['Roughness'].links
        if rl:
            nt.links.new(math_node(nt, 'MULTIPLY', rl[0].from_socket, P['body_rough_mul']), b.inputs['Roughness'])
    add_fog(mat)


_lamp_mats = {}


def lamp_mat(t):
    if t not in _lamp_mats:
        _lamp_mats[t] = emission_material('Lamp_' + t, (1.0, 0.93, 0.80) if t == 'head' else (1.0, 0.03, 0.02), P['head_emit'] if t == 'head' else P['tail_emit'], one_sided=True)
    return _lamp_mats[t]


def build_proto(name):
    """-> dict(body=mesh, wheels=[(mesh, x, y, r)], lamps=mesh, head=(y, z), tail=(y, z), size)"""
    c = CFG[name]; t0 = time.time()
    d = os.path.join(ROOT, 'assets', name)
    rot = {'bike': math.pi, 'car': -math.pi / 2}[c['axes']]
    ob = import_glb(os.path.join(d, '2_nowheel.glb'))
    me = ob.data
    me.transform(Matrix.Rotation(rot, 4, 'Z'))
    co = co_array(me); lo, hi = co.min(0), co.max(0)
    s = c['L'] / (hi[1] - lo[1])
    me.transform(Matrix.Translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])))
    me.transform(Matrix.Diagonal((s, s, s, 1.0)))
    me.transform(Matrix.Translation((0, 0, c['body_z'])))
    me.update()
    for m in me.materials:
        vehicle_material(m, 'body', c.get('tail_mask', False))
    co = co_array(me); lo, hi = co.min(0), co.max(0)
    body_me = me; body_me.name = name + '_body'
    bpy.data.objects.remove(ob)

    # wheels: cut the islands out of the wheel GLB, centre, scale, (flip)
    ow = import_glb(os.path.join(d, '3_wheels.glb'))
    wme = ow.data
    wme.transform(Matrix.Rotation(math.pi / 2, 4, 'Z'))          # axle -> X, glTF x -> Y
    for m in wme.materials:
        vehicle_material(m, 'wheel', False)
    cache = {}; wheels = []
    for w in c['wheels']:
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
            m2.update(); m2.name = '%s_wheel%d%s' % (name, w['island'], 'L' if w.get('flip') else 'R')
            cache[key] = m2
        wheels.append((cache[key], w['x'], w['y'], w['r']))
    bpy.data.objects.remove(ow)

    # lamps: emissive discs placed on the body surface by ray casting from the front / rear
    me = body_me
    bvh = BVHTree.FromPolygons([v.co.copy() for v in me.vertices], [tuple(p.vertices) for p in me.polygons])
    V, F, FM = [], [], []
    ends = {'head': None, 'tail': None}
    for lp in c.get('lamps', []):
        xs = [lp['x'], -lp['x']] if lp['x'] else [0.0]
        sgn = 1.0 if lp['t'] == 'head' else -1.0
        for x in xs:
            hit = bvh.ray_cast(Vector((x, sgn * 30.0, lp['z'])), Vector((0, -sgn, 0)))
            y = hit[0].y if hit[0] is not None else (hi[1] if sgn > 0 else lo[1])
            y += sgn * 0.02
            n = len(V); N = 14
            V.append((x, y, lp['z']))
            for j in range(N):
                a = math.tau * j / N
                V.append((x + math.cos(a) * lp['w'] / 2, y, lp['z'] + math.sin(a) * lp['h'] / 2))
            for j in range(N):                                      # normal faces outwards (+Y for head, -Y for tail)
                tri = (n, n + 1 + (j + 1) % N, n + 1 + j) if lp['t'] == 'head' else (n, n + 1 + j, n + 1 + (j + 1) % N)
                F.append(tri); FM.append(0 if lp['t'] == 'head' else 1)
            if ends[lp['t']] is None or abs(y) > abs(ends[lp['t']][0]):
                ends[lp['t']] = (y, lp['z'])
            print('   lamp %s x=%.2f z=%.2f -> y=%.3f %s' % (lp['t'], x, lp['z'], y, '' if hit[0] is not None else '(no hit)'))
    lamps_me = None
    if V:
        lamps_me = bpy.data.meshes.new(name + '_lamps'); lamps_me.from_pydata(V, [], F)
        lamps_me.materials.append(lamp_mat('head')); lamps_me.materials.append(lamp_mat('tail'))
        lamps_me.polygons.foreach_set('material_index', np.asarray(FM, dtype=np.int32)); lamps_me.update()
    print('proto %s: scale %.3f size %s  wheels %d  (%.1fs)' % (name, s, np.round(hi - lo, 2).tolist(), len(wheels), time.time() - t0))
    return {'name': name, 'body': body_me, 'wheels': wheels, 'lamps': lamps_me, 'head': ends['head'] or (hi[1], 0.7), 'tail': ends['tail'] or (lo[1], 0.8),
            'bike': c['axes'] == 'bike', 'lo': lo, 'hi': hi}


_lights = {}


def light_data(kind):
    if kind in _lights:
        return _lights[kind]
    if kind == 'car_head':
        l = bpy.data.lights.new(kind, 'SPOT'); l.energy = P['head_power']; l.color = (1.0, 0.92, 0.80); l.spot_size = math.radians(62); l.spot_blend = 0.55; l.shadow_soft_size = 0.12
    elif kind == 'bike_head':
        l = bpy.data.lights.new(kind, 'SPOT'); l.energy = P['bike_head_power']; l.color = (1.0, 0.95, 0.86); l.spot_size = math.radians(56); l.spot_blend = 0.6; l.shadow_soft_size = 0.06
    else:
        l = bpy.data.lights.new(kind, 'POINT'); l.energy = P['tail_point']; l.color = (1.0, 0.04, 0.02); l.shadow_soft_size = 0.05
    _lights[kind] = l
    return l


def instance(proto, tag, color=(1, 1, 1, 0.5), head_light=True):
    root = empty(tag)
    body = link(bpy.data.objects.new(tag + '_body', proto['body']), root)
    body.color = color
    wheels = []
    for i, (me, x, y, r) in enumerate(proto['wheels']):
        w = link(bpy.data.objects.new('%s_w%d' % (tag, i), me), root)
        w.location = (x, y, r)
        try:
            w.cycles.motion_steps = 4
        except Exception:
            pass
        wheels.append((w, r))
    if proto['lamps'] is not None:
        link(bpy.data.objects.new(tag + '_lamps', proto['lamps']), root)
    if head_light:
        hy, hz = proto['head']
        o = link(bpy.data.objects.new(tag + '_head', light_data('bike_head' if proto['bike'] else 'car_head')), root)
        o.location = (0, hy + 0.06, hz); o.rotation_euler = (math.pi / 2 - math.radians(5.5 if proto['bike'] else 7.0), 0, 0)
    if proto['bike']:
        ty, tz = proto['tail']
        o = link(bpy.data.objects.new(tag + '_tail', light_data('tail')), root)
        o.location = (0, ty - 0.12, tz)
    return {'root': root, 'wheels': wheels, 'proto': proto}


WHITE_TINTS = [(1, 1, 1), (0.55, 0.57, 0.60), (0.035, 0.035, 0.04), (1, 1, 1), (0.13, 0.14, 0.15), (0.42, 0.03, 0.03), (0.04, 0.08, 0.28), (0.88, 0.86, 0.80)]
SUV_VARS = [(1, 1, 1, 0.5), (0.35, 0.35, 0.35, 0.5), (1, 1, 1, 0.92), (0.2, 0.2, 0.2, 0.5), (1, 1, 1, 0.5), (0.8, 0.8, 0.8, 0.85), (1.6, 1.6, 1.6, 0.5), (1, 1, 1, 0.30)]
TRUCK_HUES = [0.5, 0.5, 1.0, 0.82, 0.5, 0.25, 1.0, 0.5]


def build_vehicles():
    protos = {n: build_proto(n) for n in CFG}
    S = {}
    S['A'] = instance(protos['bike_A'], 'BikeA'); S['B'] = instance(protos['bike_B'], 'BikeB')
    cars = []
    pick = M.rng(555)
    for i, c in enumerate(M.traffic()):
        pal = c['pal']
        if c['kind'] == 'truck':
            name = 'veh_truck'; col = (1, 1, 1, 0.5) if c['hero'] else (1, 1, 1, TRUCK_HUES[pal])
        else:
            q = pick()
            name = 'veh_sedan' if q < 0.4 else ('veh_hatch' if q < 0.7 else 'veh_suv')
            col = SUV_VARS[pal] if name == 'veh_suv' else (*WHITE_TINTS[pal], 0.5)
        inst = instance(protos[name], 'Car%03d' % i, color=col)
        inst['c'] = c
        if c['dir'] > 0:
            inst['root'].rotation_euler = (0, 0, math.pi)
        cars.append(inst)
    S['cars'] = cars
    print('vehicles:', len(cars) + 2)
    return S


# ------------------------------------------------------------------------------------------------
# per-frame state
# ------------------------------------------------------------------------------------------------
def bike_matrix(bid, t):
    p = M.bike_pose(bid, t)
    loc = M.to_blender((p['x'], p['y'], p['z']))
    R = Matrix.Rotation(p['yaw'], 4, 'Z') @ Matrix.Rotation(p['pitch'], 4, 'X') @ Matrix.Rotation(-p['roll'], 4, 'Y')
    return Matrix.Translation(loc) @ R, p['dist']


def camera_matrix(t, k):
    p, l, fov, roll = M.shot_at(t, k)
    pb = Vector(M.to_blender(p)); lb = Vector(M.to_blender(l))
    f = (lb - pb).normalized()
    r = f.cross(Vector((0, 0, 1))).normalized()
    u = r.cross(f)
    R = Matrix((r, u, -f)).transposed().to_4x4()
    return Matrix.Translation(pb) @ R @ Matrix.Rotation(roll, 4, 'Z'), fov


def key(obj, j, loc=None, rot=None):
    if loc is not None:
        obj.location = loc; obj.keyframe_insert('location', frame=j)
    if rot is not None:
        obj.rotation_euler = rot; obj.keyframe_insert('rotation_euler', frame=j)


def apply_frame(S, cam, f, blur):
    t = f / M.FPS; k = M.shot_index(t); dt = 1.0 / M.FPS
    times = ((0, t - dt), (1, t), (2, t + dt)) if blur else ((1, t),)
    campos_y = M.to_blender(M.shot_at(t, k)[0])[1]
    spin = [c for c in S['cars'] if abs(-M.pos_of(c['c'], t) - campos_y) < P['wheel_spin_range']]
    prev = {}
    for j, tt in times:
        for bid in ('A', 'B'):
            m, dist = bike_matrix(bid, tt)
            e = m.to_euler('XYZ')
            if bid in prev:
                e.make_compatible(prev[bid])
            prev[bid] = e
            key(S[bid]['root'], j, loc=m.to_translation(), rot=e)
            for w, r in S[bid]['wheels']:
                key(w, j, rot=(-dist / r, 0, 0))
        for c in S['cars']:
            key(c['root'], j, loc=(c['c']['x'], -M.pos_of(c['c'], tt), 0.0))
        for c in spin:
            d = c['c']['v'] * tt
            for w, r in c['wheels']:
                key(w, j, rot=(-d / r, 0, 0))
        m, fov = camera_matrix(tt, k)
        e = m.to_euler('XYZ')
        if 'cam' in prev:
            e.make_compatible(prev['cam'])
        prev['cam'] = e
        key(cam, j, loc=m.to_translation(), rot=e)
        cam.data.lens = cam.data.sensor_height / 2 / math.tan(math.radians(fov) / 2)
    bpy.context.scene.frame_set(1)
    return t, k


# ------------------------------------------------------------------------------------------------
def setup_render(samples, w, h):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    c = sc.cycles
    c.device = 'CPU'; c.samples = samples
    c.use_adaptive_sampling = True; c.adaptive_threshold = P['adaptive']; c.adaptive_min_samples = min(16, samples)
    c.use_denoising = True
    for attr, val in (('denoiser', 'OPENIMAGEDENOISE'), ('denoising_input_passes', 'RGB_ALBEDO_NORMAL'), ('use_light_tree', True),
                      ('max_bounces', 5), ('diffuse_bounces', 2), ('glossy_bounces', 3), ('transmission_bounces', 2), ('volume_bounces', 0),
                      ('transparent_max_bounces', 4), ('sample_clamp_indirect', P['clamp_indirect']), ('caustics_reflective', False),
                      ('caustics_refractive', False), ('film_exposure', P['film_exposure'])):
        try:
            setattr(c, attr, val)
        except Exception as e:
            print('cycles.%s: %s' % (attr, e))
    r = sc.render
    r.resolution_x = w; r.resolution_y = h; r.resolution_percentage = 100
    r.use_persistent_data = True
    r.use_motion_blur = bool(P['motion_blur'])
    for owner, attr, val in ((r, 'motion_blur_shutter', P['shutter']), (r, 'motion_blur_position', 'CENTER'), (c, 'motion_blur_position', 'CENTER')):
        try:
            setattr(owner, attr, val)
        except Exception as e:
            print('%s: %s' % (attr, e))
    im = r.image_settings
    try:
        im.media_type = 'IMAGE'
    except Exception:
        pass
    im.file_format = 'OPEN_EXR'; im.color_mode = 'RGB'; im.color_depth = '16'; im.exr_codec = 'ZIP'
    sc.frame_start = 0; sc.frame_end = 2; sc.render.fps = M.FPS
    try:
        bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
    except Exception as e:
        print('interp:', e)


def parse_frames(spec):
    out = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-'); out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    a = {'frames': '24', 'out': 'out', 'samples': '48', 'res': '960x540', 'params': '{}'}
    for i in range(0, len(argv) - 1, 2):
        a[argv[i].lstrip('-')] = argv[i + 1]
    P.update(json.loads(a['params'] or '{}'))
    w, h = (int(v) for v in a['res'].split('x'))
    frames = parse_frames(a['frames'])
    os.makedirs(a['out'], exist_ok=True)
    T0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_render(int(a['samples']), w, h)
    build_world(); build_road(); build_city()
    S = build_vehicles()
    camd = bpy.data.cameras.new('Cam'); camd.sensor_fit = 'VERTICAL'; camd.sensor_height = 24.0; camd.clip_start = 0.08; camd.clip_end = 9000
    cam = link(bpy.data.objects.new('Cam', camd)); bpy.context.scene.camera = cam
    build_s = time.time() - T0
    print('BUILD %.1fs  objects %d' % (build_s, len(bpy.data.objects)))
    log = []
    for f in frames:
        t1 = time.time()
        t, k = apply_frame(S, cam, f, bool(P['motion_blur']))
        t2 = time.time()
        bpy.context.scene.render.filepath = os.path.join(os.path.abspath(a['out']), '%04d' % f)
        bpy.ops.render.render(write_still=True)
        dt = time.time() - t2
        log.append({'frame': f, 't': round(t, 3), 'shot': k + 1, 'setup_s': round(t2 - t1, 2), 'render_s': round(dt, 1)})
        print('FRAME %d t=%.3f shot %d  setup %.2fs  render %.1fs' % (f, t, k + 1, t2 - t1, dt), flush=True)
    json.dump({'build_s': round(build_s, 1), 'frames': log, 'samples': int(a['samples']), 'res': a['res'], 'params': P},
              open(os.path.join(a['out'], 'timing_%04d.json' % frames[0]), 'w'), indent=1)
    print('TOTAL %.1fs for %d frames' % (time.time() - T0, len(frames)))


if __name__ == '__main__':
    main()
