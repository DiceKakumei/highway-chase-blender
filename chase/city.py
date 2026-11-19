"""実在の街の配置(assets/city.json = OpenStreetMap を道路沿いの座標に直したもの)から、街を組み立てる。
  ・高架の近くのビル: 窓の奥まり・柱・梁・ベランダ・方立(たてのフィン)を、形として作る(凹凸)
  ・遠くのビル: 平らな壁 + 外壁の写真素材
  ・民家: 屋根(切妻・寄棟・陸屋根)つきの箱
  ・木、地上の道と街灯、緑地、水面、高架の床版と橋脚
render.py から build(R) で呼ぶ(R = render.py のモジュール)。"""
import bpy, bmesh, json, math, os
from array import array
import numpy as np
from mathutils import Vector, noise as mnoise
from mathutils.geometry import tessellate_polygon

R = None
GZ = -12.0

# 外壁の写真素材ごとの、窓の位置(1マスの中の範囲)と、マスの始まりのずれ。tools で測った値
GRID = {
    'f018': dict(phase=(0.5, 0.5), win=(0.141, 0.859, 0.141, 0.859), pil=False),
    'f019': dict(phase=(0.5, 0.5), win=(0.141, 0.859, 0.141, 0.859), pil=True),
    'f020': dict(phase=(0.5, 0.5), win=(0.141, 0.859, 0.141, 0.859), pil=False),
    'f017': dict(phase=(0.0, 0.25), win=(0.125, 0.875, 0.281, 0.719), pil=True),
    'f013': dict(phase=(0.008, 0.258), win=(0.141, 0.875, 0.297, 0.719), pil=True),
}
FINS = {'f006': True, 'f001': True, 'f009': False, 'f007': False}     # True = ガラスの割り付けを素材のマスに合わせる

WALL_COLS = [(0.62, 0.60, 0.56), (0.50, 0.44, 0.36), (0.42, 0.42, 0.42), (0.26, 0.17, 0.12), (0.15, 0.15, 0.16), (0.32, 0.14, 0.10),
             (0.58, 0.54, 0.46), (0.36, 0.33, 0.30), (0.52, 0.50, 0.50)]
HOUSE_COLS = [(0.62, 0.60, 0.56), (0.56, 0.52, 0.44), (0.46, 0.42, 0.36), (0.40, 0.40, 0.40), (0.30, 0.24, 0.18), (0.20, 0.18, 0.17),
              (0.50, 0.52, 0.54), (0.58, 0.50, 0.42)]
ROOF_COLS = [(0.045, 0.047, 0.05), (0.03, 0.03, 0.032), (0.09, 0.05, 0.035), (0.06, 0.075, 0.09), (0.05, 0.07, 0.055), (0.20, 0.21, 0.22),
             (0.11, 0.045, 0.035), (0.07, 0.07, 0.075)]


class Acc:
    """面をためて、1つのメッシュにする(頂点は共有しない)。"""
    def __init__(self):
        self.co = array('f'); self.uv = array('f'); self.a = array('f'); self.b = array('f'); self.ls = array('i'); self.mi = array('i'); self.nl = 0

    def quad(self, p0, p1, p2, p3, uv, mat, a, b=(0.0, 0.0, 0.0, 1.0)):
        co = self.co
        co.extend(p0); co.extend(p1); co.extend(p2); co.extend(p3)
        self.uv.extend(uv); self.a.extend(a * 4); self.b.extend(b * 4)
        self.ls.append(self.nl); self.nl += 4; self.mi.append(mat)

    def tri(self, p0, p1, p2, uv, mat, a, b=(0.0, 0.0, 0.0, 1.0)):
        co = self.co
        co.extend(p0); co.extend(p1); co.extend(p2)
        self.uv.extend(uv); self.a.extend(a * 3); self.b.extend(b * 3)
        self.ls.append(self.nl); self.nl += 3; self.mi.append(mat)

    def faces(self):
        return len(self.ls)

    def obj(self, name, mats):
        n = self.nl
        me = bpy.data.meshes.new(name)
        me.vertices.add(n); me.loops.add(n); me.polygons.add(len(self.ls))
        me.vertices.foreach_set('co', np.frombuffer(self.co, dtype=np.float32))
        me.loops.foreach_set('vertex_index', np.arange(n, dtype=np.int32))
        ls = np.frombuffer(self.ls, dtype=np.int32)
        me.polygons.foreach_set('loop_start', ls)
        try:
            lt = np.diff(np.append(ls, n)).astype(np.int32)
            me.polygons.foreach_set('loop_total', lt)
        except Exception:
            pass
        me.polygons.foreach_set('material_index', np.frombuffer(self.mi, dtype=np.int32))
        uvl = me.uv_layers.new(name='UVMap'); uvl.data.foreach_set('uv', np.frombuffer(self.uv, dtype=np.float32))
        for nm, data in (('bld', self.a), ('bld2', self.b)):
            ca = me.color_attributes.new(nm, 'FLOAT_COLOR', 'CORNER'); ca.data.foreach_set('color', np.frombuffer(data, dtype=np.float32))
        for m in mats:
            me.materials.append(m)
        me.update(calc_edges=True)
        o = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(o)
        return o


# ------------------------------------------------------------------------------------------------
# materials
# ------------------------------------------------------------------------------------------------
def tint_wall_material():
    """色を面ごとに決める壁(マンション・民家・ベランダ)。色 = 色属性 bld、細部 = コンクリートの写真。"""
    mat, nt, out = R.new_material('TintWall')
    b = R.principled(nt, out)
    tc = nt.nodes.new('ShaderNodeTexCoord')
    t = R.mapping(nt, tc.outputs['UV'], scale=(1 / 2.71, 1 / 2.71, 1.0))
    d = R.img_node(nt, 'concrete_diff.jpg', t)
    lum = R.luminance(nt, d.outputs['Color'])
    a = nt.nodes.new('ShaderNodeAttribute'); a.attribute_name = 'bld'
    k = R.math_node(nt, 'ADD', 0.55, R.math_node(nt, 'MULTIPLY', lum, 0.45 / 0.24))
    nt.links.new(R.vmath(nt, 'SCALE', a.outputs['Color'], scale=k), b.inputs['Base Color'])
    b.inputs['Roughness'].default_value = 0.72
    bump = nt.nodes.new('ShaderNodeBump'); bump.inputs['Strength'].default_value = 0.25; bump.inputs['Distance'].default_value = 0.02
    nt.links.new(lum, bump.inputs['Height']); nt.links.new(bump.outputs[0], b.inputs['Normal'])
    R.add_fog(mat)
    return mat


def win_material():
    """窓1枚ごとの明かり。色属性 bld = (明るさ, 色味, カーテンの幅)。上ほど明るく、カーテンの部分は暗く暖色。"""
    mat, nt, out = R.new_material('Window')
    b = R.principled(nt, out)
    a = nt.nodes.new('ShaderNodeAttribute'); a.attribute_name = 'bld'
    sc = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(a.outputs['Color'], sc.inputs[0])
    lit, warm, cur = sc.outputs[0], sc.outputs[1], sc.outputs[2]
    tc = nt.nodes.new('ShaderNodeTexCoord'); sp = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(tc.outputs['UV'], sp.inputs[0])
    grad = R.math_node(nt, 'ADD', 0.5, R.math_node(nt, 'MULTIPLY', sp.outputs['Y'], 0.7))
    curt = R.math_node(nt, 'LESS_THAN', sp.outputs['X'], cur)
    col = R.vlerp(nt, (1.0, 0.72, 0.44), (0.80, 0.90, 1.0), R.math_node(nt, 'GREATER_THAN', warm, 0.72))
    col = R.vlerp(nt, col, (0.9, 0.6, 0.36), curt)
    k = R.math_node(nt, 'MULTIPLY', R.math_node(nt, 'MULTIPLY', lit, grad), R.lerp(nt, 1.0, 0.42, curt))
    nt.links.new(R.vmath(nt, 'SCALE', col, scale=k), b.inputs['Emission Color']); b.inputs['Emission Strength'].default_value = R.P['win_emit']
    b.inputs['Base Color'].default_value = (0.085, 0.10, 0.125, 1.0); b.inputs['Metallic'].default_value = 0.9; b.inputs['Roughness'].default_value = 0.07
    R.add_fog(mat)
    return mat


def house_roof_material():
    """民家の屋根: 色 = 色属性、流れ方向に瓦の段(UV の v = 流れ方向の距離)。"""
    mat, nt, out = R.new_material('HouseRoof')
    b = R.principled(nt, out)
    a = nt.nodes.new('ShaderNodeAttribute'); a.attribute_name = 'bld'
    tc = nt.nodes.new('ShaderNodeTexCoord'); sp = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(tc.outputs['UV'], sp.inputs[0])
    var = R.math_node(nt, 'ADD', 0.75, R.math_node(nt, 'MULTIPLY', R.noise(nt, tc.outputs['UV'], 0.6, 3.0), 0.5))
    nt.links.new(R.vmath(nt, 'SCALE', a.outputs['Color'], scale=var), b.inputs['Base Color'])
    b.inputs['Roughness'].default_value = 0.5
    wave = R.math_node(nt, 'ADD', R.math_node(nt, 'SINE', R.math_node(nt, 'MULTIPLY', sp.outputs['Y'], 2 * math.pi / 0.28)),
                       R.math_node(nt, 'MULTIPLY', R.math_node(nt, 'SINE', R.math_node(nt, 'MULTIPLY', sp.outputs['X'], 2 * math.pi / 0.26)), 0.6))
    bump = nt.nodes.new('ShaderNodeBump'); bump.inputs['Strength'].default_value = 0.5; bump.inputs['Distance'].default_value = 0.03
    nt.links.new(wave, bump.inputs['Height']); nt.links.new(bump.outputs[0], b.inputs['Normal'])
    R.add_fog(mat)
    return mat


def ground_material(name, col, rough, nscale, namp, metallic=0.0):
    mat, nt, out = R.new_material(name)
    b = R.principled(nt, out)
    tc = nt.nodes.new('ShaderNodeTexCoord')
    k = R.math_node(nt, 'ADD', 1.0 - namp / 2, R.math_node(nt, 'MULTIPLY', R.noise(nt, tc.outputs['Object'], nscale, 4.0), namp))
    nt.links.new(R.vmath(nt, 'SCALE', col, scale=k), b.inputs['Base Color'])
    b.inputs['Roughness'].default_value = rough; b.inputs['Metallic'].default_value = metallic
    R.add_fog(mat)
    return mat


def road_material(major):
    """地上の道。UV = (幅方向 0..1, 長さ m)。大きい道には中央線と外側線。"""
    mat, nt, out = R.new_material('StreetMajor' if major else 'StreetMinor')
    b = R.principled(nt, out)
    tc = nt.nodes.new('ShaderNodeTexCoord'); sp = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(tc.outputs['UV'], sp.inputs[0])
    u, v = sp.outputs['X'], sp.outputs['Y']
    k = R.math_node(nt, 'ADD', 0.8, R.math_node(nt, 'MULTIPLY', R.noise(nt, tc.outputs['Object'], 0.35, 4.0), 0.4))
    col = R.vmath(nt, 'SCALE', (0.055, 0.055, 0.058), scale=k)
    if major:
        du = R.math_node(nt, 'ABSOLUTE', R.math_node(nt, 'SUBTRACT', u, 0.5))
        centre = R.math_node(nt, 'MULTIPLY', R.math_node(nt, 'LESS_THAN', du, 0.012), R.math_node(nt, 'LESS_THAN', R.math_node(nt, 'FRACT', R.math_node(nt, 'DIVIDE', v, 10.0)), 0.5))
        edge = R.math_node(nt, 'MULTIPLY', R.math_node(nt, 'GREATER_THAN', du, 0.44), R.math_node(nt, 'LESS_THAN', du, 0.455))
        col = R.vlerp(nt, col, (0.5, 0.5, 0.48), R.math_node(nt, 'MAXIMUM', centre, edge))
    nt.links.new(col, b.inputs['Base Color']); b.inputs['Roughness'].default_value = 0.8
    R.add_fog(mat)
    return mat


def pool_material():
    """地上の街灯の光だまり(丸くぼける)。"""
    mat, nt, out = R.new_material('LampPool')
    tc = nt.nodes.new('ShaderNodeTexCoord')
    d = R.vmath(nt, 'SUBTRACT', tc.outputs['UV'], (0.5, 0.5, 0.0))
    ln = nt.nodes.new('ShaderNodeVectorMath'); ln.operation = 'LENGTH'; nt.links.new(d, ln.inputs[0])
    f = R.math_node(nt, 'POWER', R.smooth(nt, ln.outputs['Value'], 0.5, 0.0), 2.0)
    em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Color'].default_value = (1.0, 0.66, 0.36, 1.0); em.inputs['Strength'].default_value = R.P['pool_emit']
    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    mix = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(R.math_node(nt, 'MULTIPLY', f, 0.85), mix.inputs[0]); nt.links.new(tr.outputs[0], mix.inputs[1]); nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs['Surface'])
    R.add_fog(mat)
    return mat


def leaf_material():
    mat, nt, out = R.new_material('Leaves')
    b = R.principled(nt, out)
    tc = nt.nodes.new('ShaderNodeTexCoord')
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    n1 = R.noise(nt, tc.outputs['Object'], 1.3, 4.0, 0.6)
    k = R.math_node(nt, 'MULTIPLY', R.math_node(nt, 'ADD', 0.45, R.math_node(nt, 'MULTIPLY', n1, 1.1)), R.math_node(nt, 'ADD', 0.7, R.math_node(nt, 'MULTIPLY', oi.outputs['Random'], 0.6)))
    col = R.vlerp(nt, (0.020, 0.040, 0.012), (0.045, 0.055, 0.014), oi.outputs['Random'])
    nt.links.new(R.vmath(nt, 'SCALE', col, scale=k), b.inputs['Base Color'])
    b.inputs['Roughness'].default_value = 0.55
    bump = nt.nodes.new('ShaderNodeBump'); bump.inputs['Strength'].default_value = 0.9; bump.inputs['Distance'].default_value = 0.25
    nt.links.new(R.noise(nt, tc.outputs['Object'], 3.2, 3.0, 0.7), bump.inputs['Height']); nt.links.new(bump.outputs[0], b.inputs['Normal'])
    R.add_fog(mat)
    return mat


# ------------------------------------------------------------------------------------------------
# wall-local helpers  (p0 = 壁の始点 (x, y)、t = 壁の向き、n = 外向きの法線、s = 壁に沿った距離、d = 外への出、z = 高さ)
# ------------------------------------------------------------------------------------------------
def P3(p0, t, n, s, d, z):
    return (p0[0] + t[0] * s + n[0] * d, p0[1] + t[1] * s + n[1] * d, z)


def wbox(acc, p0, t, n, sa, sb, d0, d1, za, zb, mat, a, b2=(0.0, 0.0, 0.0, 1.0), back=False):
    """壁から出っ張る箱(前・上・下・左右。back=True で裏も)。UV はメートル。"""
    A = P3(p0, t, n, sa, d1, za); B = P3(p0, t, n, sb, d1, za); C = P3(p0, t, n, sb, d1, zb); D = P3(p0, t, n, sa, d1, zb)
    E = P3(p0, t, n, sa, d0, za); F = P3(p0, t, n, sb, d0, za); G = P3(p0, t, n, sb, d0, zb); H = P3(p0, t, n, sa, d0, zb)
    w = d1 - d0
    acc.quad(A, B, C, D, (sa, za, sb, za, sb, zb, sa, zb), mat, a, b2)                       # front
    acc.quad(D, C, G, H, (sa, zb, sb, zb, sb, zb + w, sa, zb + w), mat, a, b2)               # top
    acc.quad(E, F, B, A, (sa, za - w, sb, za - w, sb, za, sa, za), mat, a, b2)               # bottom
    acc.quad(E, A, D, H, (sa - w, za, sa, za, sa, zb, sa - w, zb), mat, a, b2)               # left
    acc.quad(B, F, G, C, (sb, za, sb + w, za, sb + w, zb, sb, zb), mat, a, b2)               # right
    if back:
        acc.quad(F, E, H, G, (sb, za, sa, za, sa, zb, sb, zb), mat, a, b2)


def wquad(acc, p0, t, n, sa, sb, d, za, zb, uv, mat, a, b2=(0.0, 0.0, 0.0, 1.0)):
    acc.quad(P3(p0, t, n, sa, d, za), P3(p0, t, n, sb, d, za), P3(p0, t, n, sb, d, zb), P3(p0, t, n, sa, d, zb), uv, mat, a, b2)


def wall_grid(acc, key, MI, p0, t, n, L, z0, zt, zcut, a, b2, rr):
    """写真素材の窓に合わせて、窓を奥まらせる。柱・梁のある素材は、それも出っ張らせる。"""
    F = R.FACADES[key]; G = GRID[key]; mat = MI[key]
    T = F['tile']; gx, gy = F['grid']; pu, pv = G['phase']; wu0, wu1, wv0, wv1 = G['win']
    wb = T / gx; fh = T / gy
    nb = max(1, int(L / wb + 0.5)); nf = max(1, int((zt - z0) / fh + 0.5))
    c0 = int(rr() * gx); r0 = int(rr() * gy)
    ds = L / nb; dz = (zt - z0) / nf
    U = lambda cu: (c0 + cu + pu) * wb
    V = lambda cv: (r0 + cv + pv) * fh
    jcut = 0
    while jcut < nf and z0 + (jcut + 1) * dz <= zcut:
        jcut += 1
    if jcut > 0:                                                                   # 高架より下は平らな1枚
        wquad(acc, p0, t, n, 0.0, L, 0.0, z0, z0 + jcut * dz, (U(0), V(0), U(nb), V(0), U(nb), V(jcut), U(0), V(jcut)), mat, a, b2)
    if L < 2.2:
        if jcut < nf:
            wquad(acc, p0, t, n, 0.0, L, 0.0, z0 + jcut * dz, zt, (U(0) , V(jcut), U(0) + 0.02, V(jcut), U(0) + 0.02, V(nf), U(0), V(nf)), mat, a, b2)
        return
    rd = 0.24
    for j in range(jcut, nf):
        zc0 = z0 + j * dz; zc1 = zc0 + dz; za = zc0 + wv0 * dz; zb = zc0 + wv1 * dz
        v0, v1, va, vb = V(j), V(j + 1), V(j + wv0), V(j + wv1)
        for i in range(nb):
            s0 = i * ds; s1 = s0 + ds; sa = s0 + wu0 * ds; sb = s0 + wu1 * ds
            u0, u1, ua, ub = U(i), U(i + 1), U(i + wu0), U(i + wu1)
            wquad(acc, p0, t, n, s0, sa, 0.0, zc0, zc1, (u0, v0, ua, v0, ua, v1, u0, v1), mat, a, b2)          # 左の壁
            wquad(acc, p0, t, n, sb, s1, 0.0, zc0, zc1, (ub, v0, u1, v0, u1, v1, ub, v1), mat, a, b2)          # 右の壁
            wquad(acc, p0, t, n, sa, sb, 0.0, zc0, za, (ua, v0, ub, v0, ub, va, ua, va), mat, a, b2)           # 腰壁
            wquad(acc, p0, t, n, sa, sb, 0.0, zb, zc1, (ua, vb, ub, vb, ub, v1, ua, v1), mat, a, b2)           # 上の壁
            wquad(acc, p0, t, n, sa, sb, -rd, za, zb, (ua, va, ub, va, ub, vb, ua, vb), mat, a, b2)            # ガラス(奥)
            ue = U(i + wu0 - 0.03); uf = U(i + wu1 + 0.03); ve = V(j + wv0 - 0.03); vf = V(j + wv1 + 0.03)
            A0 = P3(p0, t, n, sa, 0.0, za); A1 = P3(p0, t, n, sa, -rd, za); A2 = P3(p0, t, n, sa, -rd, zb); A3 = P3(p0, t, n, sa, 0.0, zb)
            acc.quad(A0, A1, A2, A3, (ue, va, ua, va, ua, vb, ue, vb), mat, a, b2)                             # 抱き(左)
            B0 = P3(p0, t, n, sb, 0.0, za); B1 = P3(p0, t, n, sb, -rd, za); B2 = P3(p0, t, n, sb, -rd, zb); B3 = P3(p0, t, n, sb, 0.0, zb)
            acc.quad(B1, B0, B3, B2, (ub, va, uf, va, uf, vb, ub, vb), mat, a, b2)                             # 抱き(右)
            acc.quad(A0, B0, B1, A1, (ua, ve, ub, ve, ub, va, ua, va), mat, a, b2)                             # 窓台
            acc.quad(A2, B2, B3, A3, (ua, vb, ub, vb, ub, vf, ua, vf), mat, a, b2)                             # 上端
    if G['pil'] and jcut < nf:
        zc = z0 + jcut * dz
        for i in range(nb + 1):                                                    # 柱
            sc = min(max(i * ds, 0.21), L - 0.21)
            A = P3(p0, t, n, sc - 0.21, 0.14, zc); B = P3(p0, t, n, sc + 0.21, 0.14, zc); C = P3(p0, t, n, sc + 0.21, 0.14, zt); D = P3(p0, t, n, sc - 0.21, 0.14, zt)
            ul, ur = U(i - 0.06), U(i + 0.06)
            acc.quad(A, B, C, D, (ul, V(jcut), ur, V(jcut), ur, V(nf), ul, V(nf)), mat, a, b2)
            E = P3(p0, t, n, sc - 0.21, 0.0, zc); H = P3(p0, t, n, sc - 0.21, 0.0, zt); F2 = P3(p0, t, n, sc + 0.21, 0.0, zc); G2 = P3(p0, t, n, sc + 0.21, 0.0, zt)
            acc.quad(E, A, D, H, (U(i - 0.10), V(jcut), ul, V(jcut), ul, V(nf), U(i - 0.10), V(nf)), mat, a, b2)
            acc.quad(B, F2, G2, C, (ur, V(jcut), U(i + 0.10), V(jcut), U(i + 0.10), V(nf), ur, V(nf)), mat, a, b2)
        for j in range(jcut, nf + 1):                                              # 梁
            zl = z0 + j * dz
            zl0, zl1 = max(zc, zl - 0.18), min(zt, zl + 0.18)
            if zl1 - zl0 < 0.05:
                continue
            A = P3(p0, t, n, 0.0, 0.09, zl0); B = P3(p0, t, n, L, 0.09, zl0); C = P3(p0, t, n, L, 0.09, zl1); D = P3(p0, t, n, 0.0, 0.09, zl1)
            vl, vr = V(j - 0.05), V(j + 0.05)
            acc.quad(A, B, C, D, (U(0), vl, U(nb), vl, U(nb), vr, U(0), vr), mat, a, b2)
            acc.quad(D, C, P3(p0, t, n, L, 0.0, zl1), P3(p0, t, n, 0.0, 0.0, zl1), (U(0), vr, U(nb), vr, U(nb), V(j + 0.08), U(0), V(j + 0.08)), mat, a, b2)
            acc.quad(P3(p0, t, n, 0.0, 0.0, zl0), P3(p0, t, n, L, 0.0, zl0), B, A, (U(0), V(j - 0.08), U(nb), V(j - 0.08), U(nb), vl, U(0), vl), mat, a, b2)


def wall_fins(acc, key, MI, FRAME, p0, t, n, L, z0, zt, zcut, a, b2, rr):
    """ガラス張りの壁: 平らなガラス + たての方立(フィン) + 各階の横の帯。"""
    F = R.FACADES[key]; mat = MI[key]
    T = F['tile']; gx, gy = F['grid']
    if FINS[key]:
        wb = T / gx; fh = T / gy
    else:
        wb = 1.9; fh = 3.7
    nb = max(1, int(L / wb + 0.5)); nf = max(1, int((zt - z0) / fh + 0.5))
    ds = L / nb; dz = (zt - z0) / nf
    if FINS[key]:
        c0 = int(rr() * gx); r0 = int(rr() * gy)
        u0, u1, v0, v1 = c0 * wb, (c0 + nb) * wb, r0 * fh, (r0 + nf) * fh
    else:
        u0 = rr() * 200.0; u1 = u0 + L; v0 = z0 - GZ; v1 = zt - GZ
    wquad(acc, p0, t, n, 0.0, L, 0.0, z0, zt, (u0, v0, u1, v0, u1, v1, u0, v1), mat, a, b2)
    if L < 2.2:
        return
    jcut = 0
    while jcut < nf and z0 + (jcut + 1) * dz <= zcut:
        jcut += 1
    if jcut >= nf:
        return
    zc = z0 + jcut * dz
    fa = (0.0, 0.0, 0.0, 1.0)
    for i in range(nb + 1):
        sc = min(max(i * ds, 0.04), L - 0.04)
        A = P3(p0, t, n, sc - 0.04, 0.17, zc); B = P3(p0, t, n, sc + 0.04, 0.17, zc); C = P3(p0, t, n, sc + 0.04, 0.17, zt); D = P3(p0, t, n, sc - 0.04, 0.17, zt)
        acc.quad(A, B, C, D, (0, 0, 1, 0, 1, 1, 0, 1), FRAME, fa)
        acc.quad(P3(p0, t, n, sc - 0.04, 0.0, zc), A, D, P3(p0, t, n, sc - 0.04, 0.0, zt), (0, 0, 1, 0, 1, 1, 0, 1), FRAME, fa)
        acc.quad(B, P3(p0, t, n, sc + 0.04, 0.0, zc), P3(p0, t, n, sc + 0.04, 0.0, zt), C, (0, 0, 1, 0, 1, 1, 0, 1), FRAME, fa)
    for j in range(jcut, nf + 1):
        zl = z0 + j * dz; zl0, zl1 = max(zc, zl - 0.09), min(zt, zl + 0.09)
        if zl1 - zl0 < 0.04:
            continue
        A = P3(p0, t, n, 0.0, 0.11, zl0); B = P3(p0, t, n, L, 0.11, zl0); C = P3(p0, t, n, L, 0.11, zl1); D = P3(p0, t, n, 0.0, 0.11, zl1)
        acc.quad(A, B, C, D, (0, 0, 1, 0, 1, 1, 0, 1), FRAME, fa)
        acc.quad(D, C, P3(p0, t, n, L, 0.0, zl1), P3(p0, t, n, 0.0, 0.0, zl1), (0, 0, 1, 0, 1, 1, 0, 1), FRAME, fa)
        acc.quad(P3(p0, t, n, 0.0, 0.0, zl0), P3(p0, t, n, L, 0.0, zl0), B, A, (0, 0, 1, 0, 1, 1, 0, 1), FRAME, fa)


def wall_mansion(acc, M_, kind, p0, t, n, L, z0, zt, zcut, lv, col, rail_mat, rail_col, rr):
    """マンションの壁。kind: 'balcony'(ベランダ) / 'corridor'(外廊下) / 'plain'(小窓)。"""
    TINT, WIN, FRAME = M_
    a = (col[0], col[1], col[2], 1.0)
    u0 = rr() * 50.0
    wquad(acc, p0, t, n, 0.0, L, 0.0, z0, zt, (u0, z0, u0 + L, z0, u0 + L, zt, u0, zt), TINT, a)
    if L < 2.5:
        return
    dz = (zt - z0) / lv
    lit_p = 0.5 + rr() * 0.25
    for j in range(lv):
        zf = z0 + j * dz
        if zf + dz <= zcut:
            continue
        if kind == 'plain':
            nw = max(1, int(L / 4.2))
            for k in range(nw):
                sc = (k + 0.5) * L / nw
                lit = (0.5 + rr() * 0.9) if rr() < lit_p * 0.7 else 0.0
                wquad(acc, p0, t, n, sc - 0.7, sc + 0.7, 0.025, zf + 0.95, zf + 2.15, (0, 0, 1, 0, 1, 1, 0, 1), WIN, (lit, rr(), rr() * 0.6, 1.0))
            continue
        dep = 1.25 if kind == 'balcony' else 1.35
        sa, sb = 0.12, L - 0.12
        wbox(acc, p0, t, n, sa, sb, 0.0, dep, zf - 0.17, zf, TINT, a)                                            # 床スラブ
        ra = (rail_col[0], rail_col[1], rail_col[2], 1.0)
        wbox(acc, p0, t, n, sa, sb, dep - 0.06, dep, zf, zf + 1.12, rail_mat, ra, back=True)                     # 手すり
        nu = max(1, int(L / 6.3 + 0.5)); uw = L / nu
        for k in range(nu):
            s0 = k * uw
            if kind == 'balcony':
                if k > 0:
                    wbox(acc, p0, t, n, s0 - 0.03, s0 + 0.03, 0.0, dep - 0.06, zf, zf + 2.0, TINT, a)            # 隔て板
                lit = (0.45 + rr() * 1.0) if rr() < lit_p else 0.0
                wquad(acc, p0, t, n, s0 + 0.7, s0 + uw - 0.7, 0.03, zf + 0.05, zf + 2.15, (0, 0, 1, 0, 1, 1, 0, 1), WIN, (lit, rr(), rr() * 0.8, 1.0))
            else:
                wquad(acc, p0, t, n, s0 + 0.5, s0 + 1.4, 0.03, zf + 0.02, zf + 2.05, (0, 0, 1, 0, 1, 1, 0, 1), FRAME, (0, 0, 0, 1))   # 玄関ドア
                lit = (0.3 + rr() * 0.6) if rr() < 0.3 else 0.0
                wquad(acc, p0, t, n, s0 + 2.2, s0 + 3.1, 0.03, zf + 1.1, zf + 2.0, (0, 0, 1, 0, 1, 1, 0, 1), WIN, (lit, rr(), 0.0, 1.0))
                wquad(acc, p0, t, n, s0 + uw / 2 - 0.25, s0 + uw / 2 + 0.25, 0.04, zf + 2.35, zf + 2.47, (0, 0.5, 1, 0.5, 1, 0.6, 0, 0.6), WIN, (2.2, 0.9, 0.0, 1.0))  # 廊下灯
    wbox(acc, p0, t, n, 0.0, L, 0.0, 0.35, zt - 0.2, zt, TINT, a)                                                # 軒


# ------------------------------------------------------------------------------------------------
# buildings
# ------------------------------------------------------------------------------------------------
def edges_of(P):
    out = []
    m = len(P)
    for i in range(m):
        p0 = P[i]; p1 = P[(i + 1) % m]
        L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if L < 0.05:
            continue
        t = ((p1[0] - p0[0]) / L, (p1[1] - p0[1]) / L)
        out.append((p0, t, (t[1], -t[0]), L))
    return out


def pip(P, x, y):
    inside = False
    m = len(P)
    for i in range(m):
        x1, y1 = P[i]; x2, y2 = P[(i + 1) % m]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def abox(acc, x0, x1, y0, y1, z0, z1, mat, top, a, b2=(0.0, 0.0, 0.0, 1.0)):
    """軸に平行な箱(屋上の設備など)。"""
    for (p0, t, n, L) in edges_of([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]):
        wquad(acc, p0, t, n, 0.0, L, 0.0, z0, z1, (0, z0, L, z0, L, z1, 0, z1), mat, a, b2)
    acc.quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (x0, y0, x1, y0, x1, y1, x0, y1), top, a, b2)


def flat_roof(acc, P, zt, MATS, a, b2, rr, h, near, detail):
    PLAIN, ROOF, METAL, OBS = MATS
    tris = tessellate_polygon([[Vector((p[0], p[1], 0.0)) for p in P]])
    for (i, j, k) in tris:
        A, B, C = P[i], P[j], P[k]
        if (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]) < 0:
            B, C = C, B
        acc.tri((A[0], A[1], zt), (B[0], B[1], zt), (C[0], C[1], zt), (A[0], A[1], B[0], B[1], C[0], C[1]), ROOF, a, b2)
    for (p0, t, n, L) in edges_of(P):                                              # 立ち上がり(パラペット)
        wquad(acc, p0, t, n, 0.0, L, 0.0, zt, zt + 1.0, (0, zt, L, zt, L, zt + 1.0, 0, zt + 1.0), PLAIN, a, b2)
    if not detail:
        return
    xs = [p[0] for p in P]; ys = [p[1] for p in P]
    cx = sum(xs) / len(P); cy = sum(ys) / len(P)
    w = max(xs) - min(xs); d = max(ys) - min(ys)
    if w > 8 and d > 8 and pip(P, cx, cy) and rr() < 0.8:                          # 塔屋
        pw = min(w * 0.3, 7.0) * (0.7 + rr() * 0.5); pd = min(d * 0.3, 7.0) * (0.7 + rr() * 0.5); ph = 3.0 + rr() * 2.8
        ox = cx + (rr() - 0.5) * w * 0.15; oy = cy + (rr() - 0.5) * d * 0.15
        if all(pip(P, ox + sx * pw / 2, oy + sy * pd / 2) for sx in (-1, 1) for sy in (-1, 1)):
            abox(acc, ox - pw / 2, ox + pw / 2, oy - pd / 2, oy + pd / 2, zt, zt + ph, PLAIN, ROOF, a, b2)
    if near:                                                                        # 室外機
        for _ in range(2 + int(rr() * 5)):
            ux = cx + (rr() - 0.5) * w * 0.6; uy = cy + (rr() - 0.5) * d * 0.6
            if pip(P, ux, uy) and pip(P, ux + 1.6, uy + 1.0):
                abox(acc, ux, ux + 1.6, uy, uy + 1.0, zt, zt + 1.5, METAL, METAL, a, b2)
    if h > 100 and rr() < 0.5 and pip(P, cx, cy):                                  # アンテナ
        mh = 10 + rr() * 16
        abox(acc, cx - 0.2, cx + 0.2, cy - 0.2, cy + 0.2, zt, zt + mh, METAL, METAL, a, b2)
        abox(acc, cx - 0.3, cx + 0.3, cy - 0.3, cy + 0.3, zt + mh, zt + mh + 0.6, OBS, OBS, a, b2)
    if h >= 60:                                                                     # 航空障害灯
        ext = [max(P, key=lambda p: p[0] + p[1]), max(P, key=lambda p: p[0] - p[1]), max(P, key=lambda p: -p[0] + p[1]), max(P, key=lambda p: -p[0] - p[1])]
        for (lx, ly) in ext:
            abox(acc, lx - 0.3, lx + 0.3, ly - 0.3, ly + 0.3, zt + 1.0, zt + 1.6, OBS, OBS, a, b2)


def obb(P):
    """最小の外接長方形。(中心, 長辺の向き, 長辺, 短辺)"""
    best = None
    A = np.asarray(P)
    for i in range(len(P)):
        e = A[(i + 1) % len(P)] - A[i]
        L = np.hypot(e[0], e[1])
        if L < 0.3:
            continue
        ux, uy = e / L
        s = A[:, 0] * ux + A[:, 1] * uy; q = -A[:, 0] * uy + A[:, 1] * ux
        w, d = s.max() - s.min(), q.max() - q.min()
        if best is None or w * d < best[0]:
            cs, cq = (s.max() + s.min()) / 2, (q.max() + q.min()) / 2
            best = (w * d, (cs * ux - cq * uy, cs * uy + cq * ux), (ux, uy), w, d)
    _, c, u, w, d = best
    if d > w:
        u = (-u[1], u[0]); w, d = d, w
    return c, u, w, d


def house(acc, b, MATS, rr):
    TINT, WIN, HROOF = MATS
    P = b['p']
    c, u, l, w = obb(P)
    k = math.sqrt(max(0.35, min(1.0, b['a'] / max(1.0, l * w))))
    l *= k; w *= k
    if w < 2.5:
        return
    v = (-u[1], u[0])
    wh = b['lv'] * 2.75 + 0.3
    z0 = GZ; z1 = GZ + wh
    col = HOUSE_COLS[int(rr() * len(HOUSE_COLS))]; sh = 0.75 + rr() * 0.4
    a = (col[0] * sh, col[1] * sh, col[2] * sh, 1.0)
    rc = ROOF_COLS[int(rr() * len(ROOF_COLS))]; ra = (rc[0], rc[1], rc[2], 1.0)
    cor = lambda su, sv, z: (c[0] + u[0] * su + v[0] * sv, c[1] + u[1] * su + v[1] * sv, z)
    hl, hw = l / 2, w / 2
    quad_pts = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
    lit_p = 0.45
    for i in range(4):
        (a0, b0), (a1, b1) = quad_pts[i], quad_pts[(i + 1) % 4]
        L = math.hypot(a1 - a0, b1 - b0)
        acc.quad(cor(a0, b0, z0), cor(a1, b1, z0), cor(a1, b1, z1), cor(a0, b0, z1), (0, 0, L, 0, L, wh, 0, wh), TINT, a)
        p0 = cor(a0, b0, 0)[:2]; p1 = cor(a1, b1, 0)[:2]
        t = ((p1[0] - p0[0]) / L, (p1[1] - p0[1]) / L); n = (t[1], -t[0])
        nw = max(1, int(L / 3.2))
        for j in range(b['lv'] if L > 2.4 else 0):
            for q in range(nw):
                if rr() < 0.25:
                    continue
                sc = (q + 0.5) * L / nw
                lit = (0.4 + rr() * 0.9) if rr() < lit_p else 0.0
                wquad(acc, p0, t, n, sc - 0.75, sc + 0.75, 0.03, z0 + j * 2.75 + 0.95, z0 + j * 2.75 + 2.05, (0, 0, 1, 0, 1, 1, 0, 1), WIN, (lit, rr() * 0.7, rr() * 0.7, 1.0))
    ov = 0.45; slope = 0.42 + rr() * 0.12
    kind = rr()
    if kind < 0.12 or w < 3.5:                                                     # 陸屋根
        acc.quad(cor(-hl, -hw, z1), cor(hl, -hw, z1), cor(hl, hw, z1), cor(-hl, hw, z1), (0, 0, l, 0, l, w, 0, w), TINT, a)
        return
    ze = z1 - ov * slope; zr = z1 + hw * slope; sl = math.hypot(hw + ov, (hw + ov) * slope)
    if kind < 0.62:                                                                 # 切妻
        e = hl + ov
        acc.quad(cor(-e, -hw - ov, ze), cor(e, -hw - ov, ze), cor(e, 0, zr), cor(-e, 0, zr), (0, sl, 2 * e, sl, 2 * e, 0, 0, 0), HROOF, ra)
        acc.quad(cor(e, hw + ov, ze), cor(-e, hw + ov, ze), cor(-e, 0, zr), cor(e, 0, zr), (0, sl, 2 * e, sl, 2 * e, 0, 0, 0), HROOF, ra)
        for sg in (-1, 1):
            acc.tri(cor(sg * hl, -hw * sg, z1), cor(sg * hl, hw * sg, z1), cor(sg * hl, 0, zr), (0, 0, w, 0, w / 2, hw * slope), TINT, a)
    else:                                                                           # 寄棟
        e = hl + ov; rl = max(0.0, hl - hw)
        acc.quad(cor(-e, -hw - ov, ze), cor(e, -hw - ov, ze), cor(rl, 0, zr), cor(-rl, 0, zr), (0, sl, 2 * e, sl, e + rl, 0, e - rl, 0), HROOF, ra)
        acc.quad(cor(e, hw + ov, ze), cor(-e, hw + ov, ze), cor(-rl, 0, zr), cor(rl, 0, zr), (0, sl, 2 * e, sl, e + rl, 0, e - rl, 0), HROOF, ra)
        for sg in (-1, 1):
            acc.tri(cor(sg * e, -sg * (hw + ov), ze), cor(sg * e, sg * (hw + ov), ze), cor(sg * rl, 0, zr), (0, sl, 2 * (hw + ov), sl, hw + ov, 0), HROOF, ra)


def build_buildings(city):
    keys = list(R.FACADES.keys())
    mats = [R.facade_material(k) for k in keys]
    MI = {k: i for i, k in enumerate(keys)}
    n0 = len(mats)
    PLAIN, ROOF, METAL, OBS, TINT, WIN, HROOF, FRAME = range(n0, n0 + 8)
    mats += [R.plain_wall_material(), R.roof_material(), R.simple_material('RoofMetal', (0.34, 0.35, 0.36), rough=0.5, metallic=0.85),
             R.emission_material('ObsLight', (1.0, 0.06, 0.03), R.P['obs_emit']), tint_wall_material(), win_material(), house_roof_material(),
             R.simple_material('Frame', (0.05, 0.055, 0.06), rough=0.38, metallic=0.85)]
    near_acc = Acc(); far_acc = Acc(); house_acc = Acc()
    LOWF = ['f018', 'f019', 'f020', 'f017', 'f018', 'f020']
    MIDF = ['f017', 'f019', 'f009', 'f006', 'f018', 'f001', 'f017', 'f020']
    TALLF = ['f013', 'f007', 'f009', 'f001', 'f006', 'f013']
    cnt = dict(house=0, relief=0, flat=0, skip=0)
    zcut = R.P['relief_zcut']
    for b in city['buildings']:
        P = b['p']
        xs = [p[0] for p in P]; ys = [p[1] for p in P]
        dd = min(max(-22.4 - x, x - 9.6, 0.0) for x in xs)
        rr = R.M.rng(1000 + b['id'])
        h = b['h']; zt = GZ + h
        if b['t'] == 'house':
            if dd > 330 or max(ys) < -500 or min(ys) > 3200:
                cnt['skip'] += 1; continue
            house(house_acc, b, (TINT, WIN, HROOF), rr); cnt['house'] += 1
            continue
        relief = dd < R.P['relief_dist'] and max(ys) > -250 and min(ys) < 2300 and zt > zcut + 3.0
        near = dd < 160
        acc = near_acc if relief else far_acc
        a = (rr(), 0.25 + 0.7 * rr(), rr(), 1.0); b2 = (rr(), rr(), rr(), 1.0)
        E = edges_of(P)
        if b['t'] == 'apt' and relief:
            col = WALL_COLS[int(rr() * len(WALL_COLS))]; sh = 0.7 + rr() * 0.35
            col = (col[0] * sh, col[1] * sh, col[2] * sh)
            glass_rail = rr() < 0.35
            rail_mat = FRAME if glass_rail else TINT
            rail_col = (0, 0, 0) if glass_rail else tuple(min(0.7, c * 1.25) for c in col)
            main = max(E, key=lambda e: e[3])
            lv = max(1, b['lv'])
            for (p0, t, n, L) in E:
                dn = n[0] * main[2][0] + n[1] * main[2][1]
                kind = 'balcony' if (dn > 0.8 and L > 5.0) else ('corridor' if (dn < -0.8 and L > 5.0) else 'plain')
                wall_mansion(acc, (TINT, WIN, FRAME), kind, p0, t, n, L, GZ, zt, zcut, lv, col, rail_mat, rail_col, rr)
            aa = (col[0], col[1], col[2], 1.0)
            flat_roof(acc, P, zt, (TINT, ROOF, METAL, OBS), aa, b2, rr, h, near, True)
            cnt['relief'] += 1
            continue
        if b['t'] == 'tower' or h > 58:
            key = TALLF[int(rr() * len(TALLF))]
        elif h > 26:
            key = MIDF[int(rr() * len(MIDF))]
        else:
            key = LOWF[int(rr() * len(LOWF))]
        if relief:
            for (p0, t, n, L) in E:
                if key in GRID:
                    wall_grid(acc, key, MI, p0, t, n, L, GZ, zt, zcut, a, b2, rr)
                else:
                    wall_fins(acc, key, MI, FRAME, p0, t, n, L, GZ, zt, zcut, a, b2, rr)
            flat_roof(acc, P, zt, (PLAIN, ROOF, METAL, OBS), a, b2, rr, h, near, True)
            cnt['relief'] += 1
        else:
            u = rr() * 300.0
            mat = MI[key]
            for (p0, t, n, L) in E:
                wquad(acc, p0, t, n, 0.0, L, 0.0, GZ, zt, (u, 0.0, u + L, 0.0, u + L, h, u, h), mat, a, b2); u += L
            flat_roof(acc, P, zt, (PLAIN, ROOF, METAL, OBS), a, b2, rr, h, near, near or h >= 60)
            cnt['flat'] += 1
    near_acc.obj('BuildingsNear', mats); far_acc.obj('BuildingsFar', mats); house_acc.obj('Houses', mats)
    print('city buildings:', cnt, 'faces near/far/house:', near_acc.faces(), far_acc.faces(), house_acc.faces())


# ------------------------------------------------------------------------------------------------
# ground: streets, lamps, parks, water, viaduct structure
# ------------------------------------------------------------------------------------------------
def build_ground(city):
    g = R.Geo(); g.hquad(-1800, 1800, -2200, 5600, GZ); g.obj('Ground', ground_material('CityGround', (0.06, 0.057, 0.052), 0.9, 0.08, 0.5))
    acc = Acc()
    MIN, MAJ, RAIL, GRASS, SOIL, WATER, LAMP, POOL, CARB, CARH, CART = range(11)
    mats = [road_material(False), road_material(True), ground_material('RailBed', (0.10, 0.085, 0.07), 0.95, 2.0, 0.5),
            ground_material('Grass', (0.030, 0.050, 0.016), 0.9, 0.25, 0.9), ground_material('Soil', (0.16, 0.13, 0.09), 0.95, 0.2, 0.5),
            ground_material('Water', (0.01, 0.014, 0.018), 0.08, 0.5, 0.3), R.emission_material('StreetLampHead', (1.0, 0.72, 0.42), R.P['street_emit']), pool_material(),
            R.simple_material('StreetCar', (0.10, 0.10, 0.11), rough=0.35, metallic=0.6), R.emission_material('StreetCarHead', (1.0, 0.93, 0.82), R.P['street_car_emit']),
            R.emission_material('StreetCarTail', (1.0, 0.05, 0.03), R.P['street_car_emit'] * 0.35)]
    z = (0.0, 0.0, 0.0, 1.0)
    order = {'rail': 0.03, 'service': 0.05, 'living_street': 0.05, 'residential': 0.06, 'unclassified': 0.06, 'pedestrian': 0.06}
    nl = 0; ncar = 0
    for rd in city['roads']:
        c = rd['c']; w = rd['w']; pts = rd['p']
        zz = GZ + order.get(c, 0.08)
        mat = RAIL if c == 'rail' else (MAJ if w >= 8.0 else MIN)
        v = 0.0
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            L = math.hypot(x2 - x1, y2 - y1)
            if L < 0.3:
                continue
            nx, ny = (y2 - y1) / L * w / 2, -(x2 - x1) / L * w / 2
            ex, ey = (x2 - x1) / L * w * 0.25, (y2 - y1) / L * w * 0.25                 # 継ぎ目のすきまを埋める
            acc.quad((x1 - nx - ex, y1 - ny - ey, zz), (x1 + nx - ex, y1 + ny - ey, zz), (x2 + nx + ex, y2 + ny + ey, zz), (x2 - nx + ex, y2 - ny + ey, zz),
                     (0, v, 1, v, 1, v + L, 0, v + L), mat, z)
            v += L
        if c in ('rail', 'service', 'pedestrian'):
            continue
        if w >= 10:                                                                     # 幹線の車(止まっている。ライトだけ光る)
            rc = R.M.rng(int(abs(pts[0][0]) * 13 + abs(pts[0][1]) * 7) & 0xffffff)
            for lane in (-1, 1):
                pos = rc() * 20.0
                for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
                    L = math.hypot(x2 - x1, y2 - y1)
                    if L < 0.3:
                        continue
                    tx, ty = (x2 - x1) / L, (y2 - y1) / L
                    while pos < L:
                        cx = x1 + tx * pos + ty * w * 0.22 * lane; cy = y1 + ty * pos - tx * w * 0.22 * lane
                        pos += 7.0 + rc() * 26.0
                        if (-27.0 < cx < 14.0) or abs(cx) > 380 or not (-600 < cy < 3300):
                            continue
                        fx, fy = tx * lane, ty * lane                                   # 進む向き(左側通行ではなく、車線ごとに逆向き)
                        p0 = (cx - fx * 2.2 - fy * 0.9, cy - fy * 2.2 + fx * 0.9)
                        for (q0, t_, n_, L_) in edges_of([p0, (p0[0] + fx * 4.4, p0[1] + fy * 4.4), (p0[0] + fx * 4.4 + fy * 1.8, p0[1] + fy * 4.4 - fx * 1.8), (p0[0] + fy * 1.8, p0[1] - fx * 1.8)]):
                            wquad(acc, q0, t_, n_, 0.0, L_, 0.0, zz, zz + 1.35, (0, 0, 1, 0, 1, 1, 0, 1), CARB, z)
                        acc.quad((p0[0], p0[1], zz + 1.35), (p0[0] + fx * 4.4, p0[1] + fy * 4.4, zz + 1.35), (p0[0] + fx * 4.4 + fy * 1.8, p0[1] + fy * 4.4 - fx * 1.8, zz + 1.35),
                                 (p0[0] + fy * 1.8, p0[1] - fx * 1.8, zz + 1.35), (0, 0, 1, 0, 1, 1, 0, 1), CARB, z)
                        for (off_, mat_) in ((4.45, CARH), (-0.05, CART)):
                            for side in (0.25, 1.25):
                                bx = p0[0] + fx * off_ + fy * side; by = p0[1] + fy * off_ - fx * side
                                acc.quad((bx, by, zz + 0.6), (bx + fy * 0.3, by - fx * 0.3, zz + 0.6), (bx + fy * 0.3, by - fx * 0.3, zz + 0.85), (bx, by, zz + 0.85), (0, 0, 1, 0, 1, 1, 0, 1), mat_, z)
                        hx = cx + fx * 7.0; hy = cy + fy * 7.0                           # ヘッドライトが照らす路面
                        acc.quad((hx - 4.5, hy - 4.5, GZ + 0.14 + ncar % 7 * 0.003), (hx + 4.5, hy - 4.5, GZ + 0.14 + ncar % 7 * 0.003), (hx + 4.5, hy + 4.5, GZ + 0.14 + ncar % 7 * 0.003),
                                 (hx - 4.5, hy + 4.5, GZ + 0.14 + ncar % 7 * 0.003), (0, 0, 1, 0, 1, 1, 0, 1), POOL, z)
                        ncar += 1
                    pos -= L
        step, hh, rad = (32.0, 8.5, 9.0) if w >= 10 else ((38.0, 7.0, 7.5) if w >= 8 else (46.0, 5.5, 6.0))
        acc_ = (rd['p'][0][0] * 7.3 + rd['p'][0][1] * 3.1) % step
        k = 0
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            L = math.hypot(x2 - x1, y2 - y1)
            if L < 0.3:
                continue
            while acc_ < L:
                f = acc_ / L; sg = 1 if (k % 2 == 0 or w < 10) else -1; k += 1
                x = x1 + (x2 - x1) * f + (y2 - y1) / L * (w / 2 - 0.4) * sg; y = y1 + (y2 - y1) * f - (x2 - x1) / L * (w / 2 - 0.4) * sg
                acc_ += step
                if not (-27.0 < x < 14.0) and abs(x) < 420 and -700 < y < 3400:
                    s = 0.28
                    acc.quad((x - s, y - s, GZ + hh), (x + s, y - s, GZ + hh), (x + s, y + s, GZ + hh), (x - s, y + s, GZ + hh), (0, 0, 1, 0, 1, 1, 0, 1), LAMP, z)
                    px = x - (y2 - y1) / L * 2.0 * sg; py = y + (x2 - x1) / L * 2.0 * sg
                    acc.quad((px - rad, py - rad, GZ + 0.10 + nl % 9 * 0.003), (px + rad, py - rad, GZ + 0.10 + nl % 9 * 0.003), (px + rad, py + rad, GZ + 0.10 + nl % 9 * 0.003), (px - rad, py + rad, GZ + 0.10 + nl % 9 * 0.003), (0, 0, 1, 0, 1, 1, 0, 1), POOL, z)
                    nl += 1
            acc_ -= L
    for gset, zz in ((city['green'], GZ + 0.02), (city['water'], GZ + 0.04)):
        for gr in gset:
            P = gr['p']
            if gr['k'] == 'stream':
                for (x1, y1), (x2, y2) in zip(P[:-1], P[1:]):
                    L = math.hypot(x2 - x1, y2 - y1)
                    if L < 0.3:
                        continue
                    nx, ny = (y2 - y1) / L * 4.0, -(x2 - x1) / L * 4.0
                    acc.quad((x1 - nx, y1 - ny, zz), (x1 + nx, y1 + ny, zz), (x2 + nx, y2 + ny, zz), (x2 - nx, y2 - ny, zz), (0, 0, 1, 0, 1, 1, 0, 1), WATER, z)
                continue
            if P[0] == P[-1]:
                P = P[:-1]
            if len(P) < 3:
                continue
            mat = WATER if gr['k'] == 'water' else (SOIL if gr['k'] == 'yard' else GRASS)
            for (i, j, k) in tessellate_polygon([[Vector((p[0], p[1], 0.0)) for p in P]]):
                A, B, C = P[i], P[j], P[k]
                if (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]) < 0:
                    B, C = C, B
                acc.tri((A[0], A[1], zz), (B[0], B[1], zz), (C[0], C[1], zz), (0, 0, 1, 0, 0, 1), mat, z)
    acc.obj('Streets', mats)
    print('streets faces:', acc.faces(), 'street lamps:', nl, 'street cars:', ncar)

    # 高架の床版と橋脚
    conc = bpy.data.materials.get('Concrete')
    g = R.Geo()
    g.box(-6.4, R.YC, -1.0, 32.0, R.LEN + 40, 1.9)
    y = R.Y0
    while y < R.Y1:
        g.box(-6.4, y, (GZ - 3.4) / 2, 3.2, 2.4, -3.4 - GZ)          # 柱
        g.box(-6.4, y, -2.7, 27.0, 2.6, 1.5)                          # 横梁
        y += 40.0
    g.obj('Viaduct', conc)


# ------------------------------------------------------------------------------------------------
# trees
# ------------------------------------------------------------------------------------------------
def tree_protos(n=6):
    bark = R.simple_material('Bark', (0.05, 0.04, 0.03), rough=0.9)
    leaf = leaf_material()
    protos = []
    for k in range(n):
        rr = R.M.rng(500 + k)
        bm = bmesh.new()
        th = 3.2 + rr() * 1.5
        bmesh.ops.create_cone(bm, cap_ends=False, segments=6, radius1=0.24, radius2=0.15, depth=th)
        for v in bm.verts:
            v.co.z += th / 2
        for f in bm.faces:
            f.material_index = 0
        nb = 4 + int(rr() * 3)
        for i in range(nb):
            r = 1.5 + rr() * 1.3
            ang = rr() * math.tau; rad = rr() * 1.5
            c = Vector((math.cos(ang) * rad, math.sin(ang) * rad, th + 0.6 + rr() * 3.6))
            res = bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
            for v in res['verts']:
                d = v.co.normalized()
                nz = mnoise.noise(d * 1.7 + Vector((k * 7.1, i * 3.3, 0.0)))
                v.co = Vector((d.x * r, d.y * r, d.z * r * 0.82)) * (1.0 + 0.32 * nz) + c
            for v in res['verts']:
                for f in v.link_faces:
                    f.material_index = 1; f.smooth = True
        me = bpy.data.meshes.new('Tree%d' % k)
        bm.to_mesh(me); bm.free()
        me.materials.append(bark); me.materials.append(leaf)
        zmax = max(v.co.z for v in me.vertices)
        protos.append((me, zmax))
    return protos


def build_trees(city):
    protos = tree_protos()
    n = 0
    for (x, y, size) in city['trees']:
        rr = R.M.rng(int(x * 131 + y * 17) & 0xffffff)
        me, zmax = protos[int(rr() * len(protos))]
        o = bpy.data.objects.new('T', me)
        s = size / zmax
        o.location = (x, y, GZ); o.rotation_euler = (0.0, 0.0, rr() * math.tau); o.scale = (s * (0.85 + rr() * 0.3), s * (0.85 + rr() * 0.3), s)
        bpy.context.scene.collection.objects.link(o)
        n += 1
    print('trees:', n)


def build(render_module):
    global R, GZ
    R = render_module
    GZ = R.P['ground_z']
    if R.P.get('city_kit'):                                   # 新しい街(tools/city_kit)。ジョブのカメラの点の周りだけ near/mid、他は far
        return build_kit()
    city = json.load(open(os.path.join(R.ROOT, 'assets', 'city.json'), encoding='utf-8'))
    build_buildings(city)
    build_ground(city)
    build_trees(city)


def build_kit():
    """新しい街。 R.P['_cam_pts'] = このジョブのカメラの点 [(x, y, z), ...]。 街の中心と半径はその点から決める"""
    import sys
    tools = os.path.join(R.ROOT, 'tools')
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import city_kit as ck
    pts = R.P.get('_cam_pts') or [(0.0, 700.0, 1.0)]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
    radius = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + R.P['kit_radius']
    pol = dict(R.P.get('kit_policy') or {})
    root = ck.build_city(os.path.join(R.ROOT, 'assets', 'city.json'), center, radius, pts, hour=R.P['kit_hour'], seed=1, ground_z=GZ, emit=R.P['kit_emit'],
                         material_hook=R.add_fog, detail_policy=pol, obs_emit=R.P['obs_emit'], street_emit=R.P['street_emit'], pool_emit=R.P['pool_emit'])
    st = ck.last_stats
    print('CITYKIT center %s radius %.0f cams %d  %s' % ([round(c) for c in center], radius, len(pts),
          {k: st.get(k) for k in ('detail_counts', 'houses', 'tris_total', 't_total', 'materials')}))
    return root
