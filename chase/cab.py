"""トラックの運転席(2026-11-18 v4)。
元のトラックのメッシュは、窓にガラスがなく、荷台の白い壁が透けて見えていた。ワイパーは窓のふちに焼き込まれていて動かせない。
 ・焼き込みワイパーの面を消す(assets/veh_truck/wiper_faces.npz = 面の中心の位置)
 ・ガラス(ほぼ透明。ふちが空を映す)を窓に張る
 ・内装: 運転席の後ろの壁、ドアの内側、ダッシュボード、シート、ハンドル、運転手(右ハンドル=車体の右側)
 ・ワイパーを2本作り、ときどき拭く
座標はトラックのプロトタイプ空間(x=右, y=前, z=上)。"""
import math
import numpy as np
import bpy
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree
import bmesh

# ---- 窓の面(ガラス板) ---------------------------------------------------------------------------
WS_TOP = (2.06, 3.131)      # (z, y) フロントガラスの上端(メッシュ内の白い板に合わせた: y = 3.919 - 0.383 z)
WS_BOT = (1.39, 3.387)      # (z, y) 下端
WS_HW = (0.86, 0.88)        # 上端・下端の半幅
SIDE_X = 0.903
SIDE_Y = (2.30, 3.20)
SIDE_Z = (1.42, 2.03)
DRIVER_X = 0.45             # 右ハンドル
PASS_X = -0.38
# ガラス板の座標系: e1=横, e2=窓の上向き(板の上), nout=外向きの法線
_up = np.array([0.0, WS_TOP[1] - WS_BOT[1], WS_TOP[0] - WS_BOT[0]]); _up /= np.linalg.norm(_up)
E1 = np.array([1.0, 0.0, 0.0]); E2 = _up; NOUT = np.cross(E1, E2) * -1.0
if NOUT[1] < 0:
    NOUT = -NOUT
WIPERS = [(0.62, 0.58), (-0.02, 0.58)]      # (ピボットの x, ブレード長)
WIPER_REST = math.radians(174.0)            # 左向き(-x)にほぼ水平
WIPER_SWEEP = math.radians(84.0)
WIPER_Z0 = 1.43                             # ピボットの高さ(ガラスの下端の少し上)


def _wp_pos(px):
    z = WIPER_Z0
    y = WS_BOT[1] - (z - WS_BOT[0]) * (WS_BOT[1] - WS_TOP[1]) / (WS_TOP[0] - WS_BOT[0])
    return np.array([px, y, z]) + NOUT * 0.014


# ---- 形のたくわえ ---------------------------------------------------------------------------------
class Acc:
    def __init__(self):
        self.v = []; self.f = []; self.m = []

    def add(self, verts, faces, mi):
        n = len(self.v)
        self.v.extend(tuple(float(c) for c in p) for p in verts)
        self.f.extend(tuple(n + i for i in f) for f in faces)
        self.m.extend([mi] * len(faces))

    def box(self, c, size, mi, rotx=0.0):
        hx, hy, hz = (s / 2 for s in size)
        pts = np.array([(dx * hx, dy * hy, dz * hz) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)])
        if rotx:
            ca, sa = math.cos(rotx), math.sin(rotx)
            pts = pts @ np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]]).T
        self.add(pts + np.asarray(c), [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)], mi)

    def quad(self, pts, mi):
        self.add(pts, [(0, 1, 2, 3)], mi)

    def sphere(self, c, radii, mi, nu=18, nv=12):
        V = [(0, 0, 1)]
        for j in range(1, nv):
            th = math.pi * j / nv
            for i in range(nu):
                ph = math.tau * i / nu
                V.append((math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)))
        V.append((0, 0, -1))
        F = []
        for i in range(nu):
            F.append((0, 1 + i, 1 + (i + 1) % nu))
        for j in range(nv - 2):
            for i in range(nu):
                a = 1 + j * nu + i; b = 1 + j * nu + (i + 1) % nu
                F.append((a, a + nu, b + nu, b))
        last = len(V) - 1
        for i in range(nu):
            F.append((last, 1 + (nv - 2) * nu + (i + 1) % nu, 1 + (nv - 2) * nu + i))
        self.add(np.array(V) * np.asarray(radii) + np.asarray(c), F, mi)

    def cylinder(self, p0, p1, r0, r1, mi, n=14, caps=True):
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        d = p1 - p0; L = np.linalg.norm(d); d /= L
        a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(d, a); u /= np.linalg.norm(u); w = np.cross(d, u)
        V = []; F = []
        for p, r in ((p0, r0), (p1, r1)):
            for i in range(n):
                ph = math.tau * i / n
                V.append(p + r * (math.cos(ph) * u + math.sin(ph) * w))
        for i in range(n):
            j = (i + 1) % n
            F.append((i, j, n + j, n + i))
        if caps:
            V.append(p0); V.append(p1)
            for i in range(n):
                F.append((2 * n, (i + 1) % n, i)); F.append((2 * n + 1, n + i, n + (i + 1) % n))
        self.add(V, F, mi)

    def torus(self, c, normal, R, r, mi, nu=28, nv=8):
        nrm = np.asarray(normal, float); nrm /= np.linalg.norm(nrm)
        a = np.array([1.0, 0, 0]); u = np.cross(nrm, a); u /= np.linalg.norm(u); w = np.cross(nrm, u)
        V = []; F = []
        for i in range(nu):
            ph = math.tau * i / nu
            ring = math.cos(ph) * u + math.sin(ph) * w
            for j in range(nv):
                th = math.tau * j / nv
                V.append(np.asarray(c) + (R + r * math.cos(th)) * ring + r * math.sin(th) * nrm)
        for i in range(nu):
            for j in range(nv):
                a0 = i * nv + j; a1 = i * nv + (j + 1) % nv; b0 = ((i + 1) % nu) * nv + j; b1 = ((i + 1) % nu) * nv + (j + 1) % nv
                F.append((a0, b0, b1, a1))
        self.add(V, F, mi)


# ---- 材質 -----------------------------------------------------------------------------------------
def glass_material(R):
    mat, nt, out = R.new_material('CabGlass')
    tr = nt.nodes.new('ShaderNodeBsdfTransparent'); tr.inputs['Color'].default_value = (0.66, 0.72, 0.74, 1.0)
    gl = nt.nodes.new('ShaderNodeBsdfGlossy'); gl.inputs['Color'].default_value = (0.92, 0.96, 1.0, 1.0); gl.inputs['Roughness'].default_value = 0.03
    fr = nt.nodes.new('ShaderNodeFresnel'); fr.inputs['IOR'].default_value = 1.5
    fac = R.math_node(nt, 'ADD', R.math_node(nt, 'MULTIPLY', fr.outputs[0], 1.6), 0.03, clamp=True)
    mx = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(fac, mx.inputs[0]); nt.links.new(tr.outputs[0], mx.inputs[1]); nt.links.new(gl.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs['Surface'])
    R.add_fog(mat)
    return mat


def cab_materials(R):
    S = R.simple_material
    return [S('Cab_dark', (0.045, 0.048, 0.054), 0.75), S('Cab_dash', (0.030, 0.030, 0.034), 0.45), S('Cab_seat', (0.065, 0.070, 0.082), 0.9),
            S('Cab_wheel', (0.012, 0.012, 0.014), 0.4), S('Drv_jacket', (0.085, 0.105, 0.145), 0.8), S('Drv_pants', (0.050, 0.054, 0.072), 0.85),
            S('Drv_skin', (0.46, 0.29, 0.22), 0.55), S('Drv_hair', (0.020, 0.016, 0.016), 0.6),
            R.emission_material('Cab_cluster', (1.0, 0.62, 0.25), 4.0, one_sided=False)]


M_DARK, M_DASH, M_SEAT, M_WHEEL, M_JACKET, M_PANTS, M_SKIN, M_HAIR, M_GLOW = range(9)


# ---- 内装 + 運転手 ---------------------------------------------------------------------------------
def build_interior():
    a = Acc()
    # 壁・床・ドアの内側(荷台の白い壁を隠す)
    a.quad([(-0.88, 2.14, 1.0), (0.88, 2.14, 1.0), (0.88, 2.14, 2.04), (-0.88, 2.14, 2.04)], M_DARK)
    a.quad([(-0.88, 2.14, 1.0), (-0.88, 3.30, 1.0), (0.88, 3.30, 1.0), (0.88, 2.14, 1.0)], M_DARK)
    for sx in (-1, 1):
        x = sx * 0.875
        a.quad([(x, 2.14, 1.0), (x, 3.25, 1.0), (x, 3.25, 1.37), (x, 2.14, 1.37)], M_DARK)
        a.quad([(x, 2.14, 1.37), (x, 2.30, 1.37), (x, 2.30, 2.04), (x, 2.14, 2.04)], M_DARK)         # Bピラーの内側
    # ダッシュボード
    a.box((0, 3.20, 1.24), (1.74, 0.30, 0.24), M_DASH)
    a.box((0, 3.10, 1.38), (1.70, 0.18, 0.05), M_DASH)
    a.quad([(DRIVER_X - 0.17, 3.045, 1.355), (DRIVER_X + 0.17, 3.045, 1.355), (DRIVER_X + 0.17, 3.045, 1.43), (DRIVER_X - 0.17, 3.045, 1.43)], M_GLOW)   # メーターのあかり
    # シート
    for x0 in (DRIVER_X, PASS_X):
        a.box((x0, 2.68, 1.07), (0.50, 0.52, 0.14), M_SEAT)
        a.box((x0, 2.42, 1.46), (0.48, 0.13, 0.70), M_SEAT, rotx=math.radians(-9))
        a.box((x0, 2.375, 1.90), (0.30, 0.10, 0.19), M_SEAT, rotx=math.radians(-9))
    # ハンドル
    wc = np.array([DRIVER_X, 3.03, 1.40]); wn = np.array([0.0, -0.70, 0.71]); wn /= np.linalg.norm(wn)
    a.torus(wc, wn, 0.185, 0.017, M_WHEEL)
    a.cylinder(wc - wn * 0.015, wc + wn * -0.0, 0.05, 0.05, M_WHEEL)
    a.cylinder(wc, wc - wn * 0.0 + np.array([0, 0.18, -0.16]), 0.032, 0.036, M_WHEEL)
    for ang in (0.0, 2.1, 4.2):
        u = np.cross(wn, np.array([1.0, 0, 0])); u /= np.linalg.norm(u); w = np.cross(wn, u)
        d = math.cos(ang) * u + math.sin(ang) * w
        a.cylinder(wc, wc + d * 0.18, 0.012, 0.012, M_WHEEL, n=8)
    # 運転手(右ハンドル)
    x0 = DRIVER_X
    hip = np.array([x0, 2.66, 1.17])
    a.sphere((x0, 2.64, 1.40), (0.20, 0.13, 0.27), M_JACKET)                       # 胴
    a.sphere((x0, 2.63, 1.58), (0.25, 0.12, 0.10), M_JACKET)                       # 肩
    a.cylinder((x0, 2.62, 1.64), (x0, 2.60, 1.72), 0.046, 0.044, M_SKIN)           # 首
    a.sphere((x0, 2.60, 1.81), (0.092, 0.108, 0.118), M_SKIN)                      # 頭
    a.sphere((x0, 2.58, 1.86), (0.098, 0.112, 0.085), M_HAIR)                      # 髪(帽子)
    for sx in (-1, 1):
        S = np.array([x0 + sx * 0.25, 2.63, 1.56]); E = np.array([x0 + sx * 0.27, 2.79, 1.36]); H = np.array([x0 + sx * 0.155, 3.0, 1.44])
        a.cylinder(S, E, 0.052, 0.046, M_JACKET); a.sphere(E, (0.05, 0.05, 0.05), M_JACKET)
        a.cylinder(E, H, 0.044, 0.036, M_JACKET); a.sphere(H, (0.045, 0.045, 0.040), M_SKIN)
        K = np.array([x0 + sx * 0.11, 3.02, 1.12])
        a.cylinder(hip + np.array([sx * 0.11, 0, 0]), K, 0.085, 0.07, M_PANTS)     # 太もも
    return a


def build_glass():
    a = Acc()
    a.quad([(-WS_HW[0], WS_TOP[1], WS_TOP[0]), (WS_HW[0], WS_TOP[1], WS_TOP[0]), (WS_HW[1], WS_BOT[1], WS_BOT[0]), (-WS_HW[1], WS_BOT[1], WS_BOT[0])], 0)
    for sx in (-1, 1):
        x = sx * SIDE_X
        a.quad([(x, SIDE_Y[0], SIDE_Z[0]), (x, SIDE_Y[1], SIDE_Z[0]), (x, SIDE_Y[1] - 0.18, SIDE_Z[1]), (x, SIDE_Y[0], SIDE_Z[1])], 0)
    return a


def build_wiper():
    """ワイパー1本(ピボットが原点、腕が +X、長さ 0.58)。材質は M_HAIR 相当の黒"""
    a = Acc()
    a.box((0.30, 0.0, -0.004), (0.60, 0.010, 0.008), 0)        # 腕(外側 = -Z)
    a.box((0.36, 0.0, 0.004), (0.56, 0.016, 0.012), 0)         # ブレード(ゴム。ガラス側 = +Z)
    a.cylinder((0, 0, -0.012), (0, 0, 0.004), 0.016, 0.016, 0, n=10)
    return a


# ---- 荷台(きれいな箱) ---------------------------------------------------------------------------
# 元の荷台の壁は、大きな三角形が少数(1壁 約100枚)で、折れ目が三角形のまま見えていた。平らな箱に置き換える。
BOX = dict(x0=-1.062, x1=1.062, y0=-3.445, y1=1.95, z0=0.94, z1=2.945)
BOX_TINTS = [(0.93, 0.93, 0.93), (0.80, 0.86, 0.95), (0.96, 0.94, 0.84), (0.72, 0.74, 0.76), (0.90, 0.92, 0.93), (0.80, 0.92, 0.84), (0.92, 0.84, 0.80), (0.95, 0.95, 0.95)]


def build_box():
    b = BOX
    a = Acc()
    x0, x1, y0, y1, z0, z1 = b['x0'], b['x1'], b['y0'], b['y1'], b['z0'], b['z1']
    V = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    a.add(V, [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (3, 0, 4, 7)], 0)
    return a


def box_material(R):
    mat, nt, out = R.new_material('TruckBox')
    bsdf = R.principled(nt, out)
    b = BOX
    pos = nt.nodes.new('ShaderNodeTexCoord').outputs['Object']
    sp = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(pos, sp.inputs[0])
    x, y, z = sp.outputs['X'], sp.outputs['Y'], sp.outputs['Z']
    M = R.math_node

    def edist(v, a0, a1):
        return M(nt, 'MINIMUM', M(nt, 'ABSOLUTE', M(nt, 'SUBTRACT', v, a0)), M(nt, 'ABSOLUTE', M(nt, 'SUBTRACT', v, a1)))
    dx, dy, dz = edist(x, b['x0'], b['x1']), edist(y, b['y0'], b['y1']), edist(z, b['z0'], b['z1'])
    ex, ey, ez = R.smooth(nt, dx, 0.045, 0.012), R.smooth(nt, dy, 0.045, 0.012), R.smooth(nt, dz, 0.045, 0.012)
    corner = M(nt, 'ADD', M(nt, 'ADD', M(nt, 'MULTIPLY', ex, ey), M(nt, 'MULTIPLY', ex, ez)), M(nt, 'MULTIPLY', ey, ez), clamp=True)
    side = R.smooth(nt, dx, 0.012, 0.004)
    u = M(nt, 'FRACT', M(nt, 'DIVIDE', M(nt, 'SUBTRACT', y, b['y0']), 0.9))
    dist = M(nt, 'MULTIPLY', M(nt, 'MINIMUM', u, M(nt, 'SUBTRACT', 1.0, u)), 0.9)
    rib = M(nt, 'MULTIPLY', R.smooth(nt, dist, 0.010, 0.004), side)
    rear = R.smooth(nt, M(nt, 'ABSOLUTE', M(nt, 'SUBTRACT', y, b['y0'])), 0.012, 0.004)
    dline = M(nt, 'MULTIPLY', R.smooth(nt, M(nt, 'ABSOLUTE', x), 0.010, 0.004), rear)
    skirt = R.smooth(nt, z, 1.16, 1.08)
    dirt = M(nt, 'MULTIPLY', R.smooth(nt, z, 1.9, 1.0), 0.20)
    nz = R.noise(nt, pos, 2.2, 3.0, 0.55)
    f = M(nt, 'SUBTRACT', 1.0, M(nt, 'ADD', M(nt, 'ADD', M(nt, 'MULTIPLY', rib, 0.14), M(nt, 'MULTIPLY', dline, 0.20)),
                                  M(nt, 'ADD', M(nt, 'ADD', M(nt, 'MULTIPLY', corner, 0.20), dirt), M(nt, 'MULTIPLY', skirt, 0.55))), clamp=True)
    f = M(nt, 'MULTIPLY', f, M(nt, 'ADD', 0.94, M(nt, 'MULTIPLY', nz, 0.10)))
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    col = R.vmath(nt, 'SCALE', oi.outputs['Color'], scale=M(nt, 'MULTIPLY', f, 0.80))
    nt.links.new(col, bsdf.inputs['Base Color'])
    nt.links.new(M(nt, 'ADD', 0.46, M(nt, 'MULTIPLY', skirt, 0.25)), bsdf.inputs['Roughness'])
    try:
        bsdf.inputs['Coat Weight'].default_value = 0.12; bsdf.inputs['Coat Roughness'].default_value = 0.2
    except Exception:
        pass
    R.add_fog(mat)
    return mat


# ---- 焼き込みワイパーの面を消す ------------------------------------------------------------------
def strip_faces(me, npz_path, keys, label):
    """面の中心の位置(npz)に一致する面を消す(元のメッシュに焼き込まれた白いガラス板・ワイパー)"""
    z = np.load(npz_path)
    pts = np.vstack([z[k] for k in keys])
    kd = KDTree(len(pts))
    for i, p in enumerate(pts):
        kd.insert(p.tolist(), i)
    kd.balance()
    bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
    dele = []
    for f in bm.faces:
        _, j, dd = kd.find(f.calc_center_median())
        if dd < 0.004:
            dele.append(f)
    bmesh.ops.delete(bm, geom=dele, context='FACES')
    bm.to_mesh(me); bm.free(); me.update()
    print('   truck: removed %d %s faces (of %d positions)' % (len(dele), label, len(pts)))


def strip_baked(me, d):
    import os
    strip_faces(me, os.path.join(d, 'wiper_faces.npz'), ['L', 'R'], 'baked wiper')
    strip_faces(me, os.path.join(d, 'glass_faces.npz'), ['G'], 'baked glass')
    strip_faces(me, os.path.join(d, 'box_faces.npz'), ['B'], 'cargo-box wall (replaced by a clean box)')


# ---- Blender への組み込み ----------------------------------------------------------------------------
_shared = {}


def shared(R):
    if _shared:
        return _shared
    mats = cab_materials(R)
    def mk(name, acc, ms):
        me = bpy.data.meshes.new(name)
        me.from_pydata(acc.v, [], acc.f)
        for m in ms:
            me.materials.append(m)
        me.polygons.foreach_set('material_index', np.asarray(acc.m, dtype=np.int32))
        me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
        me.update()
        return me
    _shared['interior'] = mk('Cab_interior', build_interior(), mats)
    _shared['glass'] = mk('Cab_glass', build_glass(), [glass_material(R)])
    blk = R.simple_material('Wiper_black', (0.012, 0.012, 0.014), 0.35)
    _shared['wiper'] = mk('Cab_wiper', build_wiper(), [blk])
    _shared['box'] = mk('Cargo_box', build_box(), [box_material(R)])
    return _shared


def add_to_truck(R, root, tint=(0.93, 0.93, 0.93)):
    """トラック1台に、荷台・内装・ガラス・ワイパー2本を付ける。-> ワイパーの物体のリスト"""
    sh = shared(R)
    bx = R.link(bpy.data.objects.new(root.name + '_box', sh['box']), root)
    bx.color = (*tint, 1.0)
    R.link(bpy.data.objects.new(root.name + '_cab', sh['interior']), root)
    R.link(bpy.data.objects.new(root.name + '_glass', sh['glass']), root)
    Z = np.cross(E1, E2)                                                                              # 内向き(右手系にするため)
    basis = Matrix(((E1[0], E2[0], Z[0]), (E1[1], E2[1], Z[1]), (E1[2], E2[2], Z[2])))               # 列 = e1, e2, z
    out = []
    for i, (px, ln) in enumerate(WIPERS):
        wp = R.empty('%s_wp%d' % (root.name, i), root)
        wp.location = Vector(_wp_pos(px)); wp.rotation_euler = basis.to_euler('XYZ')
        w = R.link(bpy.data.objects.new('%s_wiper%d' % (root.name, i), sh['wiper']), wp)
        w.rotation_euler = (0, 0, WIPER_REST)
        out.append(w)
    return out


def wiper_angle(t, phase, period):
    """-> ワイパーの角度(rad)。period 秒に1回、1.3秒かけて往復して拭く。それ以外は停止位置。"""
    s = ((t + phase) % period) / 1.3
    if s >= 1.0:
        return WIPER_REST
    return WIPER_REST - WIPER_SWEEP * math.sin(math.pi * s) ** 2
