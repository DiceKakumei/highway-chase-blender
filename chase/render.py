"""高速チェイス(写実版)の組み立てと描画。 blender --background --factory-startup --python chase/render.py -- [args]
  --frames "24,84,100-110"   描画するコマ(0始まり、t = f/24)
  --out DIR                  EXR の出力先
  --samples N  --res WxH
  --params JSON              P の上書き
Blender の座標: X=右, Y=前, Z=上。動きは motion.py (three.js 座標) を to_blender() で変換する。"""
import bpy, bmesh, json, math, os, sys, time
import numpy as np
from mathutils import Matrix, Vector, Quaternion
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import motion as M   # noqa: E402
import rig as RG     # noqa: E402

CFG = json.load(open(os.path.join(ROOT, 'vehicles.json'), encoding='utf-8'))

P = {
    'world': 0.6,            # HDRI の強さ
    'sky_rot': 4.053,        # 夕焼けを進行方向(+Y)に向ける回転
    'film_exposure': 0.5,
    'lamp_power': 11000.0,   # 街灯(スポット)
    'lamp_color': [1.0, 0.60, 0.27],
    'lamp_emit': 500.0,
    'head_power': 3500.0,   # 車のヘッドライト(1台1灯のスポット)
    'bike_head_power': 1100.0,
    'head_emit': 500.0,
    'tail_emit': 40.0,
    'tail_point': 50.0,
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
    # 道路・ビルの質感(2026-11-18 v2)
    'asph_albedo': 0.55,     # 路面の明るさ(広域写真の平均0.14に掛ける)
    'asph_rough': 0.92,
    'asph_bump': 1.3,
    'patch_prob': 0.22,      # 補修跡が出る割合
    'paint_wear': 0.36,      # 白線のかすれ(大きいほど多い)
    'fac_emit': 3.0,         # 窓あかり
    'glass_wobble': 0.03,    # ガラス1枚ごとの傾き
    'obs_emit': 60.0,        # 航空障害灯
    'reflect_emit': 4.0,     # ガードレールの反射板
    # 街(2026-11-18 v3: 実在の街の配置 + 凹凸)
    'relief_dist': 110.0,    # 高架からこの距離までのビルは、凹凸を形で作る
    'relief_zcut': -4.0,     # これより下の階は平らのまま(高架から見えない)
    'street_emit': 60.0,     # 地上の街灯
    'pool_emit': 2.2,        # 地上の街灯の光だまり
    'street_car_emit': 25.0, # 地上の車のライト
    'dbg_truck': None,       # 確認用カメラ(主役のトラック基準)
    'relax_iters': 3,        # 車体メッシュの凹凸をならす回数(0 = ならさない)
    'dbg_idx': None,         # 確認用カメラの対象(S['cars'] の番号。None = 主役のトラック)
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


def img_node(nt, fn, vec=None, noncolor=False, box=False):
    t = nt.nodes.new('ShaderNodeTexImage')
    t.image = bpy.data.images.load(os.path.join(ROOT, 'assets', 'tex', fn), check_existing=True)
    if noncolor:
        t.image.colorspace_settings.name = 'Non-Color'
    if box:
        t.projection = 'BOX'; t.projection_blend = 0.25
    if vec is not None:
        nt.links.new(vec, t.inputs['Vector'])
    return t


def mapping(nt, vec, scale=(1.0, 1.0, 1.0), rot=(0.0, 0.0, 0.0), loc=(0.0, 0.0, 0.0)):
    mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = scale; mp.inputs['Rotation'].default_value = rot; mp.inputs['Location'].default_value = loc
    nt.links.new(vec, mp.inputs['Vector'])
    return mp.outputs['Vector']


def smooth(nt, v, a, b):
    """smoothstep(a, b, v)。a > b なら逆向き(1 → 0)。"""
    n = nt.nodes.new('ShaderNodeMapRange'); n.interpolation_type = 'SMOOTHSTEP'
    n.inputs['From Min'].default_value = a; n.inputs['From Max'].default_value = b
    n.inputs['To Min'].default_value = 0.0; n.inputs['To Max'].default_value = 1.0
    nt.links.new(v, n.inputs['Value'])
    return n.outputs['Result']


def lerp(nt, a, b, f):
    """a + (b - a) * f (数値かソケット)"""
    return math_node(nt, 'ADD', a, math_node(nt, 'MULTIPLY', math_node(nt, 'SUBTRACT', b, a), f))


def noise(nt, vec, scale, detail=2.0, rough=0.5):
    n = nt.nodes.new('ShaderNodeTexNoise')
    n.inputs['Scale'].default_value = scale; n.inputs['Detail'].default_value = detail; n.inputs['Roughness'].default_value = rough
    nt.links.new(vec, n.inputs['Vector'])
    return n.outputs[0]


def hash2(nt, a, b_):
    """2つの値から乱数。(値, SeparateColor ノード)"""
    comb = nt.nodes.new('ShaderNodeCombineXYZ')
    for i, v in enumerate((a, b_)):
        if isinstance(v, (int, float)):
            comb.inputs[i].default_value = v
        else:
            nt.links.new(v, comb.inputs[i])
    wn = nt.nodes.new('ShaderNodeTexWhiteNoise'); wn.noise_dimensions = '2D'; nt.links.new(comb.outputs[0], wn.inputs['Vector'])
    sc = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(wn.outputs['Color'], sc.inputs[0])
    return wn.outputs['Value'], sc


def luminance(nt, col):
    n = nt.nodes.new('ShaderNodeRGBToBW'); nt.links.new(col, n.inputs[0])
    return n.outputs[0]


def asphalt_material():
    """路面。細かい粒(2 m)× 広域の写真(約34 m、ひび・しみ)に、轍・油じみ・補修跡・伸縮継手・路肩の砂ぼこり・かすれた白線を重ねる。"""
    mat, nt, out = new_material('Asphalt')
    b = principled(nt, out)
    tc = nt.nodes.new('ShaderNodeTexCoord'); pos = tc.outputs['Object']
    sp = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(pos, sp.inputs[0]); x, y = sp.outputs['X'], sp.outputs['Y']
    ROT = 0.31
    m1 = mapping(nt, pos, scale=(0.5, 0.5, 0.5))
    m2 = mapping(nt, pos, scale=(1 / 34.0, 1 / 34.0, 1.0), rot=(0, 0, ROT), loc=(0.37, 0.11, 0))
    d1 = img_node(nt, 'asphalt_diff.jpg', m1); r1 = img_node(nt, 'asphalt_rough.jpg', m1, True); n1 = img_node(nt, 'asphalt_nor.jpg', m1, True)
    d2 = img_node(nt, 'aerial_diff.jpg', m2); r2 = img_node(nt, 'aerial_rough.jpg', m2, True); n2 = img_node(nt, 'aerial_nor.jpg', m2, True)

    # lanes: main carriageway 3.6 m lanes (x = -7.2 .. 7.2), opposite carriageway 4 m lanes
    far = math_node(nt, 'LESS_THAN', x, -11.0)
    lane = lerp(nt, math_node(nt, 'DIVIDE', math_node(nt, 'ADD', x, 7.2), 3.6), math_node(nt, 'DIVIDE', math_node(nt, 'ADD', x, 20.5), 4.0), far)
    lf = math_node(nt, 'FRACT', lane); il = math_node(nt, 'FLOOR', lane); lw = lerp(nt, 3.6, 4.0, far)

    def band(center, w0, w1):
        d = math_node(nt, 'MULTIPLY', math_node(nt, 'ABSOLUTE', math_node(nt, 'SUBTRACT', lf, center)), lw)
        return smooth(nt, d, w0, w1)
    ny = noise(nt, mapping(nt, pos, scale=(0.5, 0.02, 1.0)), 1.0, 2.0)
    wp = math_node(nt, 'MULTIPLY', math_node(nt, 'MAXIMUM', band(0.264, 0.50, 0.12), band(0.736, 0.50, 0.12)),
                   math_node(nt, 'ADD', 0.45, math_node(nt, 'MULTIPLY', ny, 1.0)), clamp=True)          # 轍(タイヤで磨かれた帯)
    oil = band(0.5, 0.36, 0.05)                                                                          # 車線中央の油じみ
    macro = math_node(nt, 'ADD', 0.84, math_node(nt, 'MULTIPLY', noise(nt, pos, 0.045, 3.0), 0.32))

    # 補修跡: 車線 × 60 m のマスごとに、ある確率で四角く舗装し直した跡。ふちは黒いシール材
    cy = math_node(nt, 'DIVIDE', y, 60.0); iy = math_node(nt, 'FLOOR', cy); fy = math_node(nt, 'FRACT', cy)
    hv, hc = hash2(nt, il, iy)
    pc = math_node(nt, 'ADD', 0.3, math_node(nt, 'MULTIPLY', hc.outputs[0], 0.4))
    hl = math_node(nt, 'ADD', 0.07, math_node(nt, 'MULTIPLY', hc.outputs[1], 0.2))
    iny = math_node(nt, 'MULTIPLY', math_node(nt, 'SUBTRACT', hl, math_node(nt, 'ABSOLUTE', math_node(nt, 'SUBTRACT', fy, pc))), 60.0)
    inx = math_node(nt, 'SUBTRACT', math_node(nt, 'MULTIPLY', math_node(nt, 'MINIMUM', lf, math_node(nt, 'SUBTRACT', 1.0, lf)), lw), 0.16)
    dd = math_node(nt, 'MINIMUM', iny, inx)
    exist = math_node(nt, 'LESS_THAN', hv, P['patch_prob'])
    patch = math_node(nt, 'MULTIPLY', exist, math_node(nt, 'GREATER_THAN', dd, 0.0))
    seal = math_node(nt, 'MULTIPLY', exist, math_node(nt, 'LESS_THAN', math_node(nt, 'ABSOLUTE', dd), 0.035))

    # 伸縮継手: 30 m ごとに鋼の帯(幅0.28 m)と、その前後のコンクリート(各0.45 m)
    j = math_node(nt, 'FRACT', math_node(nt, 'DIVIDE', y, 30.0))
    dj = math_node(nt, 'MULTIPLY', math_node(nt, 'MINIMUM', j, math_node(nt, 'SUBTRACT', 1.0, j)), 30.0)
    steel = math_node(nt, 'LESS_THAN', dj, 0.14)
    header = math_node(nt, 'MULTIPLY', math_node(nt, 'LESS_THAN', dj, 0.59), math_node(nt, 'SUBTRACT', 1.0, steel))

    # 路肩・中央帯の砂ぼこり
    da = math_node(nt, 'ABSOLUTE', math_node(nt, 'ADD', x, 0.4)); db = math_node(nt, 'ABSOLUTE', math_node(nt, 'ADD', x, 17.5))
    dust = lerp(nt, smooth(nt, da, 7.85, 8.9), smooth(nt, db, 3.85, 4.7), far)

    # 白線(元の位置・長さのまま)。かすれ = 大きなむら + 細かい欠け
    ax = math_node(nt, 'ABSOLUTE', x)
    dsol = math_node(nt, 'MINIMUM', math_node(nt, 'ABSOLUTE', math_node(nt, 'SUBTRACT', ax, 7.6)),
                     math_node(nt, 'MINIMUM', math_node(nt, 'ABSOLUTE', math_node(nt, 'ADD', x, 21.1)), math_node(nt, 'ABSOLUTE', math_node(nt, 'ADD', x, 11.9))))
    solid = smooth(nt, dsol, 0.112, 0.088)
    dmain = math_node(nt, 'MULTIPLY', math_node(nt, 'ABSOLUTE', math_node(nt, 'SUBTRACT', math_node(nt, 'FRACT', math_node(nt, 'ADD', math_node(nt, 'DIVIDE', x, 3.6), 0.5)), 0.5)), 3.6)
    dmain = math_node(nt, 'ADD', dmain, math_node(nt, 'MULTIPLY', math_node(nt, 'GREATER_THAN', ax, 5.0), 10.0))
    ddash = math_node(nt, 'MINIMUM', dmain, math_node(nt, 'ABSOLUTE', math_node(nt, 'ADD', x, 16.5)))
    dash = math_node(nt, 'LESS_THAN', math_node(nt, 'FRACT', math_node(nt, 'DIVIDE', math_node(nt, 'ADD', y, 301.8), 9.0)), 0.4)
    paint = math_node(nt, 'MULTIPLY', math_node(nt, 'MAXIMUM', solid, math_node(nt, 'MULTIPLY', smooth(nt, ddash, 0.102, 0.078), dash)), math_node(nt, 'SUBTRACT', 1.0, steel))
    wear = smooth(nt, noise(nt, pos, 1.1, 5.0, 0.65), P['paint_wear'] - 0.06, P['paint_wear'] + 0.06)
    speck = smooth(nt, noise(nt, pos, 38.0, 2.0), 0.40, 0.56)
    pa = math_node(nt, 'MULTIPLY', paint, lerp(nt, math_node(nt, 'MULTIPLY', speck, 0.55), math_node(nt, 'ADD', 0.80, math_node(nt, 'MULTIPLY', speck, 0.20)), wear))

    # colour
    fine = vmath(nt, 'SCALE', d1.outputs['Color'], scale=1.0 / 0.015)
    k = math_node(nt, 'MULTIPLY', math_node(nt, 'MULTIPLY', macro, P['asph_albedo']),
                  math_node(nt, 'SUBTRACT', math_node(nt, 'SUBTRACT', 1.0, math_node(nt, 'MULTIPLY', wp, 0.24)), math_node(nt, 'MULTIPLY', oil, 0.10)))
    col = vmath(nt, 'SCALE', vmath(nt, 'MULTIPLY', d2.outputs['Color'], fine), scale=k)
    col = vlerp(nt, col, vmath(nt, 'SCALE', col, scale=0.58), patch)
    col = vlerp(nt, col, (0.150, 0.138, 0.118), math_node(nt, 'MULTIPLY', dust, 0.55))
    col = vlerp(nt, col, (0.010, 0.010, 0.011), seal)
    col = vlerp(nt, col, vmath(nt, 'SCALE', (0.15, 0.147, 0.138), scale=macro), header)
    col = vlerp(nt, col, (0.09, 0.09, 0.10), steel)
    dirt = math_node(nt, 'ADD', 0.72, math_node(nt, 'MULTIPLY', noise(nt, pos, 0.6, 3.0), 0.36))
    col = vlerp(nt, col, vmath(nt, 'SCALE', (0.66, 0.66, 0.63), scale=dirt), pa)
    nt.links.new(col, b.inputs['Base Color'])

    # roughness
    rg = math_node(nt, 'MULTIPLY', math_node(nt, 'MULTIPLY', r1.outputs['Color'], r2.outputs['Color']), P['asph_rough'] / 0.78)
    rg = math_node(nt, 'MULTIPLY', rg, math_node(nt, 'SUBTRACT', math_node(nt, 'SUBTRACT', 1.0, math_node(nt, 'MULTIPLY', wp, 0.28)), math_node(nt, 'MULTIPLY', oil, 0.10)))
    rg = lerp(nt, rg, 0.92, math_node(nt, 'MULTIPLY', dust, 0.6))
    rg = lerp(nt, rg, 0.26, seal)
    rg = lerp(nt, rg, 0.80, header)
    rg = lerp(nt, rg, 0.36, steel)
    rg = lerp(nt, rg, 0.52, pa)
    nt.links.new(rg, b.inputs['Roughness'])
    nt.links.new(steel, b.inputs['Metallic'])

    # normal: 路面は水平なので、法線マップをそのままワールドの向きに直す(UVが要らない)
    def nvec(tex, s, rot=0.0):
        v = vmath(nt, 'MULTIPLY', vmath(nt, 'SUBTRACT', tex.outputs['Color'], (0.5, 0.5, 0.5)), (2 * s, 2 * s, 0.0))
        if rot:
            vr = nt.nodes.new('ShaderNodeVectorRotate'); vr.rotation_type = 'Z_AXIS'; vr.inputs['Angle'].default_value = -rot
            nt.links.new(v, vr.inputs['Vector']); v = vr.outputs[0]
        return v
    flat = math_node(nt, 'SUBTRACT', 1.0, math_node(nt, 'MULTIPLY', math_node(nt, 'MAXIMUM', pa, steel), 0.7))
    nsum = vmath(nt, 'SCALE', vmath(nt, 'ADD', nvec(n1, P['asph_bump']), nvec(n2, 1.0, ROT)), scale=flat)
    nrm = vmath(nt, 'NORMALIZE', vmath(nt, 'ADD', nsum, (0.0, 0.0, 1.0)))
    nt.links.new(nrm, b.inputs['Normal'])
    add_fog(mat)
    return mat


def concrete_material():
    """壁高欄・中央分離帯: 型枠コンクリート + 雨だれ + 目地 + 足元の汚れ。"""
    mat, nt, out = new_material('Concrete')
    b = principled(nt, out)
    tc = nt.nodes.new('ShaderNodeTexCoord'); pos = tc.outputs['Object']
    sp = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(pos, sp.inputs[0]); y, z = sp.outputs['Y'], sp.outputs['Z']
    v = mapping(nt, pos, scale=(1 / 2.71,) * 3)
    d = img_node(nt, 'concrete_diff.jpg', v, box=True); r = img_node(nt, 'concrete_rough.jpg', v, True, box=True)
    streak = smooth(nt, noise(nt, mapping(nt, pos, scale=(1.2, 1.2, 0.045)), 1.0, 4.0, 0.6), 0.40, 0.72)
    top = smooth(nt, z, -0.6, 1.0)
    grime = math_node(nt, 'MULTIPLY', smooth(nt, z, 0.45, 0.0), smooth(nt, z, -0.25, 0.0))
    joint = math_node(nt, 'LESS_THAN', math_node(nt, 'FRACT', math_node(nt, 'DIVIDE', y, 8.0)), 0.004)
    k = math_node(nt, 'MULTIPLY', math_node(nt, 'SUBTRACT', 1.0, math_node(nt, 'MULTIPLY', math_node(nt, 'MULTIPLY', streak, top), 0.5)),
                  math_node(nt, 'MULTIPLY', math_node(nt, 'SUBTRACT', 1.0, math_node(nt, 'MULTIPLY', grime, 0.45)), math_node(nt, 'SUBTRACT', 1.0, math_node(nt, 'MULTIPLY', joint, 0.75))))
    grey = nt.nodes.new('ShaderNodeCombineXYZ')
    lum = luminance(nt, d.outputs['Color'])
    for i_ in range(3):
        nt.links.new(lum, grey.inputs[i_])
    dcol = vlerp(nt, d.outputs['Color'], grey.outputs[0], 0.65)
    nt.links.new(vmath(nt, 'SCALE', dcol, scale=math_node(nt, 'MULTIPLY', k, 0.62)), b.inputs['Base Color'])
    nt.links.new(math_node(nt, 'ADD', r.outputs['Color'], math_node(nt, 'MULTIPLY', streak, 0.1), clamp=True), b.inputs['Roughness'])
    bump = nt.nodes.new('ShaderNodeBump'); bump.inputs['Strength'].default_value = 0.35; bump.inputs['Distance'].default_value = 0.02
    nt.links.new(luminance(nt, d.outputs['Color']), bump.inputs['Height']); nt.links.new(bump.outputs[0], b.inputs['Normal'])
    add_fog(mat)
    return mat


def build_road():
    asphalt = asphalt_material()
    g = Geo(); g.hquad(-22.4, 9.6, Y0 - 20, Y1 + 20, 0.0); g.obj('Deck', asphalt)

    concrete = concrete_material()
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

    # 中央分離帯の上の遮光板(対向車のライトよけ)と、ガードレールの反射板
    g = Geo(); y = Y0
    while y < Y1:
        g.box(-12.7, y, 1.36, 0.34, 0.035, 0.72)
        y += 1.5
    g.obj('GlareSlats', simple_material('GlareSlat', (0.035, 0.085, 0.06), rough=0.5))
    gw = Geo(); gy = Geo(); y = Y0 + 2.0
    while y < Y1:
        gw.box(8.52, y, 0.93, 0.10, 0.02, 0.10); gy.box(-9.32, y, 0.93, 0.10, 0.02, 0.10)
        y += 24.0
    gw.obj('ReflectW', emission_material('ReflectW', (1.0, 0.95, 0.85), P['reflect_emit']))
    gy.obj('ReflectY', emission_material('ReflectY', (1.0, 0.55, 0.10), P['reflect_emit']))

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


# 外壁の写真素材(ambientCG, CC0)。tile = 1枚の実寸(m)、grid = 1枚の中の窓の列数・階数
FACADES = {
    'f018': dict(tile=20.4, grid=(6, 6), emi=True, metal=True, night=False, gain=1.0),     # れんが + 窓
    'f019': dict(tile=20.4, grid=(6, 6), emi=True, metal=True, night=False, gain=1.0),     # 灰色のコンクリート格子
    'f020': dict(tile=20.4, grid=(6, 6), emi=True, metal=True, night=False, gain=1.0),     # 茶色のれんが
    'f017': dict(tile=43.0, grid=(12, 12), emi=True, metal=False, night=False, gain=1.2),  # 縦リブのオフィス
    'f013': dict(tile=112.0, grid=(32, 32), emi=True, metal=False, night=False, gain=1.4), # 格子の高層
    'f009': dict(tile=60.0, grid=(4, 16), emi=True, metal=False, night=True, gain=1.6),    # 暗いガラスのオフィス(夜の写真)
    'f007': dict(tile=160.0, grid=(10, 44), emi=True, metal=False, night=True, gain=2.0),  # 暗いガラスの超高層(夜の写真)
    'f006': dict(tile=28.0, grid=(14, 8), emi=False, metal=True, night=False, gain=1.0),   # ガラス + 白い帯
    'f001': dict(tile=36.0, grid=(16, 10), emi=False, metal=True, night=False, gain=1.0),  # 全面ガラス
}


def bld_attrs(nt):
    """棟ごとの乱数(メッシュの色属性 bld / bld2)。(明るさ, 点灯率, 色味), (種)"""
    a = nt.nodes.new('ShaderNodeAttribute'); a.attribute_name = 'bld'
    sa = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(a.outputs['Color'], sa.inputs[0])
    b2 = nt.nodes.new('ShaderNodeAttribute'); b2.attribute_name = 'bld2'
    sb = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(b2.outputs['Color'], sb.inputs[0])
    return sa.outputs[0], sa.outputs[1], sa.outputs[2], sb.outputs[0]


def facade_material(key):
    c = FACADES[key]
    mat, nt, out = new_material('Facade_' + key)
    b = principled(nt, out)
    tc = nt.nodes.new('ShaderNodeTexCoord')
    t = mapping(nt, tc.outputs['UV'], scale=(1 / c['tile'], 1 / c['tile'], 1.0))
    col = img_node(nt, 'facade/%s_col.jpg' % key, t); rough = img_node(nt, 'facade/%s_rough.jpg' % key, t, True); nor = img_node(nt, 'facade/%s_nor.jpg' % key, t, True)
    br, litf, warm, seed = bld_attrs(nt)
    sp = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(t, sp.inputs[0])
    gx, gy = c['grid']
    px = math_node(nt, 'MULTIPLY', sp.outputs['X'], float(gx)); py = math_node(nt, 'MULTIPLY', sp.outputs['Y'], float(gy))
    hv, hc = hash2(nt, math_node(nt, 'ADD', math_node(nt, 'FLOOR', px), math_node(nt, 'MULTIPLY', seed, 977.0)),
                   math_node(nt, 'ADD', math_node(nt, 'FLOOR', py), math_node(nt, 'MULTIPLY', seed, 613.0)))
    tint = math_node(nt, 'ADD', 0.62, math_node(nt, 'MULTIPLY', br, 0.5))
    base = vmath(nt, 'SCALE', col.outputs['Color'], scale=tint)
    fac = P['fac_emit'] * c['gain']

    if c['metal']:
        metal = img_node(nt, 'facade/%s_metal.jpg' % key, t, True).outputs['Color']
    elif c['night']:
        metal = None
    else:
        metal = math_node(nt, 'MULTIPLY', math_node(nt, 'LESS_THAN', rough.outputs['Color'], 0.12), 0.9)   # 粗さ0 = ガラス

    if c['emi']:
        emi = img_node(nt, 'facade/%s_emi.jpg' % key, t)
        el = luminance(nt, emi.outputs['Color'])
        if c['night']:
            on = math_node(nt, 'ADD', 0.25, math_node(nt, 'MULTIPLY', math_node(nt, 'GREATER_THAN', hv, math_node(nt, 'SUBTRACT', 0.85, math_node(nt, 'MULTIPLY', litf, 0.8))), 1.0))
            base = vmath(nt, 'MAXIMUM', vmath(nt, 'SCALE', base, scale=0.5), (0.085, 0.10, 0.125))
            metal = math_node(nt, 'MULTIPLY', math_node(nt, 'SUBTRACT', 1.0, el, clamp=True), 0.9)
        else:
            on = math_node(nt, 'GREATER_THAN', hv, math_node(nt, 'SUBTRACT', 0.92, math_node(nt, 'MULTIPLY', litf, 0.8)))
            dark = math_node(nt, 'SUBTRACT', 1.0, math_node(nt, 'MULTIPLY', math_node(nt, 'MULTIPLY', el, math_node(nt, 'SUBTRACT', 1.0, on)), 0.88, clamp=True))
            base = vmath(nt, 'SCALE', base, scale=dark)                 # 消灯した窓は、暗いガラスにする
        e = vmath(nt, 'SCALE', emi.outputs['Color'], scale=math_node(nt, 'MULTIPLY', on, math_node(nt, 'ADD', 0.55, math_node(nt, 'MULTIPLY', hc.outputs[2], 0.9))))
    else:
        # 発光マップがない素材: ガラス(金属マップ)のマスごとに点灯。上ほど明るく(天井灯)、暖色/寒色を混ぜる
        hrun, _ = hash2(nt, math_node(nt, 'ADD', math_node(nt, 'FLOOR', math_node(nt, 'DIVIDE', px, 3.0)), math_node(nt, 'MULTIPLY', seed, 811.0)),
                        math_node(nt, 'ADD', math_node(nt, 'FLOOR', py), math_node(nt, 'MULTIPLY', seed, 353.0)))
        on = math_node(nt, 'MULTIPLY', math_node(nt, 'GREATER_THAN', hrun, math_node(nt, 'SUBTRACT', 0.96, math_node(nt, 'MULTIPLY', litf, 0.32))),
                       math_node(nt, 'GREATER_THAN', hv, 0.25))
        gradv = math_node(nt, 'ADD', 0.45, math_node(nt, 'MULTIPLY', math_node(nt, 'FRACT', py), 0.8))
        inter = math_node(nt, 'ADD', 0.55, math_node(nt, 'MULTIPLY', noise(nt, t, c['tile'] * 1.7, 2.0), 0.9))
        lc = vlerp(nt, (1.0, 0.72, 0.42), (0.78, 0.88, 1.0), math_node(nt, 'GREATER_THAN', hc.outputs[1], 0.55))
        e = vmath(nt, 'SCALE', lc, scale=math_node(nt, 'MULTIPLY', math_node(nt, 'MULTIPLY', on, metal), math_node(nt, 'MULTIPLY', math_node(nt, 'MULTIPLY', gradv, inter), 0.24)))

    # ガラスは空を映す(金属扱い)。映り込みの色 = ガラスの色。暗すぎるガラスは少し持ち上げる
    base = vlerp(nt, base, vmath(nt, 'MAXIMUM', base, (0.075, 0.09, 0.115)), metal)
    nt.links.new(base, b.inputs['Base Color'])
    nt.links.new(metal, b.inputs['Metallic'])
    nt.links.new(math_node(nt, 'MAXIMUM', rough.outputs['Color'], 0.045), b.inputs['Roughness'])
    nt.links.new(e, b.inputs['Emission Color']); b.inputs['Emission Strength'].default_value = fac
    # 法線: 素材の法線 + ガラス1枚ごとのわずかな傾き(映り込みがそろわない)
    comb = nt.nodes.new('ShaderNodeCombineXYZ')
    nt.links.new(math_node(nt, 'MULTIPLY', math_node(nt, 'SUBTRACT', hc.outputs[0], 0.5), P['glass_wobble']), comb.inputs[0])
    nt.links.new(math_node(nt, 'MULTIPLY', math_node(nt, 'SUBTRACT', hc.outputs[1], 0.5), P['glass_wobble']), comb.inputs[1])
    nm = nt.nodes.new('ShaderNodeNormalMap'); nm.inputs['Strength'].default_value = 1.0
    nt.links.new(vmath(nt, 'ADD', nor.outputs['Color'], vmath(nt, 'SCALE', comb.outputs[0], scale=metal)), nm.inputs['Color'])
    nt.links.new(nm.outputs['Normal'], b.inputs['Normal'])
    add_fog(mat)
    return mat


def plain_wall_material():
    """窓のない壁(塔屋・立ち上がり・倉庫): 型枠コンクリート。棟ごとに明るさを変える。"""
    mat, nt, out = new_material('PlainWall')
    b = principled(nt, out)
    tc = nt.nodes.new('ShaderNodeTexCoord')
    t = mapping(nt, tc.outputs['UV'], scale=(1 / 2.71, 1 / 2.71, 1.0))
    d = img_node(nt, 'concrete_diff.jpg', t); r = img_node(nt, 'concrete_rough.jpg', t, True)
    br, litf, warm, seed = bld_attrs(nt)
    sp = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(tc.outputs['UV'], sp.inputs[0])
    streak = smooth(nt, noise(nt, mapping(nt, tc.outputs['UV'], scale=(1.1, 0.05, 1.0)), 1.0, 4.0, 0.6), 0.42, 0.74)
    k = math_node(nt, 'MULTIPLY', math_node(nt, 'ADD', 0.30, math_node(nt, 'MULTIPLY', br, 0.45)), math_node(nt, 'SUBTRACT', 1.0, math_node(nt, 'MULTIPLY', streak, 0.4)))
    nt.links.new(vmath(nt, 'SCALE', d.outputs['Color'], scale=k), b.inputs['Base Color'])
    nt.links.new(r.outputs['Color'], b.inputs['Roughness'])
    add_fog(mat)
    return mat


def roof_material():
    mat, nt, out = new_material('Roof')
    b = principled(nt, out)
    tc = nt.nodes.new('ShaderNodeTexCoord'); pos = tc.outputs['Object']
    v = mapping(nt, pos, scale=(1 / 2.5, 1 / 2.5, 1.0))
    d = img_node(nt, 'roof_diff.jpg', v); r = img_node(nt, 'roof_rough.jpg', v, True)
    k = math_node(nt, 'ADD', 0.45, math_node(nt, 'MULTIPLY', noise(nt, pos, 0.11, 3.0), 0.8))
    nt.links.new(vmath(nt, 'SCALE', d.outputs['Color'], scale=k), b.inputs['Base Color'])
    nt.links.new(math_node(nt, 'ADD', r.outputs['Color'], 0.2, clamp=True), b.inputs['Roughness'])
    add_fog(mat)
    return mat


# ------------------------------------------------------------------------------------------------
# vehicles
# ------------------------------------------------------------------------------------------------
def smooth_mesh(me, angle_deg=62.0, weld=2e-5):
    """Meshy のメッシュは、UVの継ぎ目で頂点が分かれていて、継ぎ目ごとに陰影の段差(三角形の切れ目)が出る。
    位置が同じ頂点をつなぎ直し、全面をなめらかにして、角度が大きい辺だけ硬い辺にする。"""
    bm = bmesh.new(); bm.from_mesh(me)
    n0 = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=weld)
    lim = math.radians(angle_deg)
    for e in bm.edges:
        e.smooth = True
        if len(e.link_faces) == 2:
            try:
                if e.calc_face_angle() > lim:
                    e.smooth = False
            except ValueError:
                pass
    for f in bm.faces:
        f.smooth = True
    bm.to_mesh(me); bm.free(); me.update()
    print('   smooth_mesh: verts %d -> %d, faces %d' % (n0, len(me.vertices), len(me.polygons)))


def relax_mesh(me, iters=3, factor=0.5):
    """Meshy の形はボコボコで、アップだと角ばった面が目立つ。硬い辺(と端)に触れていない頂点だけ、まわりの平均へ寄せてならす。"""
    if iters <= 0:
        return
    bm = bmesh.new(); bm.from_mesh(me)
    keep = set()
    for e in bm.edges:
        if (not e.smooth) or len(e.link_faces) != 2:
            keep.add(e.verts[0].index); keep.add(e.verts[1].index)
    bm.verts.ensure_lookup_table()
    verts = [v for v in bm.verts if v.index not in keep]
    p0 = np.array([v.co[:] for v in verts]) if verts else None
    for _ in range(iters):
        bmesh.ops.smooth_vert(bm, verts=verts, factor=factor, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    if verts:
        d = np.linalg.norm(np.array([v.co[:] for v in verts]) - p0, axis=1)
        print('   relax_mesh: %d of %d verts smoothed x%d, mean move %.4f m, max %.4f m' % (len(verts), len(bm.verts), iters, d.mean(), d.max()))
    bm.to_mesh(me); bm.free(); me.update()


def import_glb(path):
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.gltf(filepath=path, import_shading='SMOOTH')
    except TypeError:
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
    smooth_mesh(meshes[0].data)
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
    if name == 'veh_truck':
        import cab
        cab.strip_baked(body_me, d)

    # バイク+ライダーの疑似リグの重み(tools/make_rig_weights.py が作る)を、頂点の位置の最近傍で写す
    rigw = None
    if c['axes'] == 'bike':
        z = np.load(os.path.join(d, 'rig_weights.npz'))
        kd = KDTree(len(z['pos']))
        for i, pp in enumerate(z['pos']):
            kd.insert(pp.tolist(), i)
        kd.balance()
        cb = co_array(body_me); idx = np.empty(len(cb), np.int64); far = 0.0
        for i, pp in enumerate(cb):
            _, j, dd = kd.find(pp.tolist()); idx[i] = j; far = max(far, dd)
        rigw = z['W'].astype(np.float32)[idx]
        print('   rig weights: %d verts, max match distance %.5f m, mean weights %s' % (len(cb), far, dict(zip(RG.BONES, rigw.mean(0).round(3).tolist()))))

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

    relax_mesh(body_me, P['relax_iters'])
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
            'bike': c['axes'] == 'bike', 'lo': lo, 'hi': hi, 'rigw': rigw, 'rig': RG.RIG.get(name)}


_lights = {}


def light_data(kind):
    if kind in _lights:
        return _lights[kind]
    if kind == 'car_head':
        l = bpy.data.lights.new(kind, 'SPOT'); l.energy = P['head_power']; l.color = (1.0, 0.92, 0.80); l.spot_size = math.radians(66); l.spot_blend = 0.85; l.shadow_soft_size = 0.12
    elif kind == 'bike_head':
        l = bpy.data.lights.new(kind, 'SPOT'); l.energy = P['bike_head_power']; l.color = (1.0, 0.95, 0.86); l.spot_size = math.radians(64); l.spot_blend = 0.9; l.shadow_soft_size = 0.06
    else:
        l = bpy.data.lights.new(kind, 'POINT'); l.energy = P['tail_point']; l.color = (1.0, 0.04, 0.02); l.shadow_soft_size = 0.05
    _lights[kind] = l
    return l


def make_rig(tag, root, body, proto):
    """バイク+ライダーのアーマチュア(ボーンは rig.BONES)。頂点グループ = 重み。"""
    cfg = proto['rig']; W = proto['rigw']
    ad = bpy.data.armatures.new(tag + '_rig')
    ao = link(bpy.data.objects.new(tag + '_rig', ad), root)
    bpy.context.view_layer.objects.active = ao
    bpy.ops.object.mode_set(mode='EDIT')
    pivots = {'static': (0, 0, 0), 'frame': cfg['frame_pivot'], 'steer_u': cfg['steer_head'], 'steer_s': cfg['steer_head'], 'torso': cfg['hip'], 'head': cfg['neck']}
    eb = {}
    for bn in RG.BONES:
        e = ad.edit_bones.new(bn); e.head = Vector(pivots[bn]); e.tail = Vector(pivots[bn]) + Vector((0, 0, 0.1)); eb[bn] = e
    for bn in RG.BONES:
        if RG.PARENT[bn]:
            eb[bn].parent = eb[RG.PARENT[bn]]; eb[bn].use_connect = False
    bpy.ops.object.mode_set(mode='OBJECT')
    rest = {bn: np.array(ad.bones[bn].matrix_local) for bn in RG.BONES}
    for i, bn in enumerate(RG.BONES):
        vg = body.vertex_groups.new(name=bn)
        q = np.rint(W[:, i] * 24).astype(np.int32)
        for lvl in range(1, 25):
            ix = np.nonzero(q == lvl)[0]
            if len(ix):
                vg.add(ix.tolist(), lvl / 24.0, 'REPLACE')
    md = body.modifiers.new('Armature', 'ARMATURE'); md.object = ao; md.use_vertex_groups = True
    for bn in RG.BONES:
        ao.pose.bones[bn].rotation_mode = 'QUATERNION'
    print('   rig %s: bones %d, vertex groups %d' % (tag, len(RG.BONES), len(body.vertex_groups)))
    return {'obj': ao, 'rest': rest}


def instance(proto, tag, color=(1, 1, 1, 0.5), head_light=True):
    root = empty(tag)
    is_bike = proto['bike'] and proto.get('rigw') is not None
    fr = steer = None
    if is_bike:
        cfg = proto['rig']
        F = Vector(cfg['frame_pivot']); Hh = Vector(cfg['steer_head'])
        fr = empty(tag + '_frame', root); fr.location = F                      # 車体のばね上(上下・ピッチ)を持つ。ランプとライトを付ける
        steer = empty(tag + '_steer', root); steer.location = Hh; steer.rotation_mode = 'QUATERNION'   # ステアリング軸まわり。前輪を付ける
    body = link(bpy.data.objects.new(tag + '_body', proto['body']), root)
    body.color = color
    wheels = []
    for i, (me, x, y, r) in enumerate(proto['wheels']):
        par = steer if (is_bike and y > 0.3) else root
        w = link(bpy.data.objects.new('%s_w%d' % (tag, i), me), par)
        w.location = (x, y, r) if par is root else (Vector((x, y, r)) - Hh)
        try:
            w.cycles.motion_steps = 4
        except Exception:
            pass
        wheels.append((w, r))
    if proto['lamps'] is not None:
        lo_ = link(bpy.data.objects.new(tag + '_lamps', proto['lamps']), fr if is_bike else root)
        if is_bike:
            lo_.location = -F
    if head_light:
        hy, hz = proto['head']
        o = link(bpy.data.objects.new(tag + '_head', light_data('bike_head' if proto['bike'] else 'car_head')), fr if is_bike else root)
        o.location = (0, hy + 0.06, hz) if not is_bike else (Vector((0, hy + 0.06, hz)) - F)
        o.rotation_euler = (math.pi / 2 - math.radians(5.5 if proto['bike'] else 7.0), 0, 0)
    if proto['bike']:
        ty, tz = proto['tail']
        o = link(bpy.data.objects.new(tag + '_tail', light_data('tail')), fr if is_bike else root)
        o.location = (0, ty - 0.12, tz) if not is_bike else (Vector((0, ty - 0.12, tz)) - F)
    out = {'root': root, 'wheels': wheels, 'proto': proto}
    if proto['name'] == 'veh_truck':
        import cab, zlib
        h = zlib.crc32(tag.encode()) % 1000 / 1000.0
        period = 3.6 + 2.4 * ((h * 7.0) % 1.0)
        out['wipers'] = cab.add_to_truck(sys.modules[__name__], root, cab.BOX_TINTS[zlib.crc32(tag.encode()) % len(cab.BOX_TINTS)])
        out['wiper_phase'] = h * period; out['wiper_period'] = period
    if is_bike:
        out.update({'frame': fr, 'steer': steer, 'rig': make_rig(tag, root, body, proto)})
    return out


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
    print('VEHLIST ' + ' '.join('%d:%s%s' % (i, c['proto']['name'][4:], 'R' if c['c']['dir'] > 0 else '') for i, c in enumerate(cars[:40])))
    return S


# ------------------------------------------------------------------------------------------------
# per-frame state
# ------------------------------------------------------------------------------------------------
def bike_matrix(bid, t):
    p = M.bike_pose(bid, t)
    loc = M.to_blender((p['x'], p['y'], p['z']))
    R = Matrix.Rotation(p['yaw'], 4, 'Z') @ Matrix.Rotation(p['pitch'], 4, 'X') @ Matrix.Rotation(-p['roll'], 4, 'Y')
    return Matrix.Translation(loc) @ R, p['dist'], p


def key_quat(obj, j, q):
    obj.rotation_quaternion = q; obj.keyframe_insert('rotation_quaternion', frame=j)


def key_bike_rig(bk, pose, j):
    """車体のばね上(上下・ピッチ)、ステアリング、ライダー(胴体・頭)の動きをキーに打つ。車輪は路面のまま(root の下で回るだけ)。"""
    cfg = bk['proto']['rig']
    ld = RG.local_deltas(cfg, pose)
    F = Vector(cfg['frame_pivot'])
    fr = bk['frame']
    fr.location = F + Vector((0, 0, pose['body_dz'])); fr.rotation_euler = (pose['body_pitch'], 0, 0)
    fr.keyframe_insert('location', frame=j); fr.keyframe_insert('rotation_euler', frame=j)
    up = Vector(cfg['steer_head']) - Vector(cfg['axle'])
    key_quat(bk['steer'], j, Quaternion(up.normalized(), pose['steer']))
    ao = bk['rig']['obj']; rest = bk['rig']['rest']
    for bn in RG.BONES:
        if bn == 'static':
            continue
        R = rest[bn]
        Bm = Matrix(np.linalg.inv(R) @ ld[bn] @ R)
        loc, rot, _ = Bm.decompose()
        pb = ao.pose.bones[bn]
        pb.location = loc; pb.rotation_quaternion = rot
        pb.keyframe_insert('location', frame=j); pb.keyframe_insert('rotation_quaternion', frame=j)


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
    P['_f'] = f
    times = ((0, t - dt), (1, t), (2, t + dt)) if blur else ((1, t),)
    campos_y = M.to_blender(M.shot_at(t, k)[0])[1]
    spin = [c for c in S['cars'] if abs(-M.pos_of(c['c'], t) - campos_y) < P['wheel_spin_range']]
    prev = {}
    for j, tt in times:
        for bid in ('A', 'B'):
            m, dist, pose = bike_matrix(bid, tt)
            e = m.to_euler('XYZ')
            if bid in prev:
                e.make_compatible(prev[bid])
            prev[bid] = e
            key(S[bid]['root'], j, loc=m.to_translation(), rot=e)
            if 'rig' in S[bid]:
                key_bike_rig(S[bid], pose, j)
            for w, r in S[bid]['wheels']:
                key(w, j, rot=(-dist / r, 0, 0))
        for c in S['cars']:
            key(c['root'], j, loc=(c['c']['x'], -M.pos_of(c['c'], tt), 0.0))
        for c in spin:
            d = c['c']['v'] * tt
            for w, r in c['wheels']:
                key(w, j, rot=(-d / r, 0, 0))
            if c.get('wipers'):
                import cab
                ang = cab.wiper_angle(tt, c['wiper_phase'], c['wiper_period'])
                for w in c['wipers']:
                    key(w, j, rot=(0, 0, ang))
        m, fov = camera_matrix(tt, k)
        if P.get('dbg_truck'):      # 確認用: 主役のトラックを基準にしたカメラ [位置xyz, 見る先xyz, fov](ブレンダー座標、トラックの原点から)
            hero = S['cars'][P['dbg_idx']] if P.get('dbg_idx') is not None else next(c for c in S['cars'] if c['c'].get('hero'))
            tl = Vector((hero['c']['x'], -M.pos_of(hero['c'], tt), 0.0)); dd = P['dbg_truck']
            if isinstance(dd[0], list):
                dd = dd[P.get("_f", 0) % len(dd)]
            cp = tl + Vector(dd[0:3]); lp = tl + Vector(dd[3:6]); fov = dd[6]
            f_ = (lp - cp).normalized(); r_ = f_.cross(Vector((0, 0, 1))).normalized(); u_ = r_.cross(f_)
            m = Matrix.Translation(cp) @ Matrix((r_, u_, -f_)).transposed().to_4x4()
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
    build_world(); build_road()
    import city
    city.build(sys.modules[__name__])
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
