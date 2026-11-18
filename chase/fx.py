"""続編の効果: 薬莢・着弾の火花と砕片・タイヤの煙・路面のスキッド跡。 すべて「プール」(使い回すオブジェクト)で、コマごとにキーを打つ。
時刻 τ(続編の秒)の関数として位置を計算するので、どのコマからでも描ける。座標はブレンダーのワールド。"""
import math
import numpy as np
import bpy
from mathutils import Matrix, Vector

import sequel as SQ

G = 9.81
CASING_SLOTS, IMPACT_SLOTS, SMOKE_N = 10, 4, 56
SPARKS_PER, DEBRIS_PER = 5, 3
SMOKE_LIFE, SMOKE_EVERY = 2.0, 0.045
SKID_STEP = 0.05


def _rng(seed):
    r = np.random.RandomState(seed)
    return r


def impacts(Rm):
    """発砲ごとの着弾点(ワールド、路面)。 Aの近くの路面で、Aを外す。 Rm = render モジュール(bike_matrix を使う)"""
    r = _rng(7); out = []
    for n, ft in enumerate(SQ.FIRES):
        ta = SQ.TOFF + ft + 0.07
        pa = Vector((SQ.xA(ta), -SQ.zA(ta), 0.0))
        side = r.choice([-1.0, 1.0]) * r.uniform(0.8, 2.4)
        back = r.uniform(-7.0, -0.8)
        out.append({'t': ft + 0.07, 'p': pa + Vector((side, back, 0.0)), 'seed': int(r.randint(1 << 30))})
    return out


def casings(Rm, rd):
    """薬莢: 発砲ごとに、銃の位置から後ろ右へ飛ぶ。 -> [{'t': τ, 'p0': Vector, 'v0': Vector}]"""
    import rider as RD
    r = _rng(11); out = []
    for ft in SQ.FIRES:
        t = SQ.TOFF + ft
        m, dist, pz = Rm.bike_matrix('B', t); mA, _, _ = Rm.bike_matrix('A', t)
        aw = mA @ Vector((0, 0.0, 0.95))
        basis, gun_M, vis, fl, muz = RD.pose(rd, pz, np.array(m), np.array(aw), ft)
        gp = m @ Vector(tuple(gun_M[:3, 3]))
        ej = Vector((r.uniform(1.8, 3.0), r.uniform(-2.5, -0.5), r.uniform(1.2, 2.6)))
        out.append({'t': ft, 'p0': gp, 'v0': m.to_3x3() @ ej + Vector((0, SQ_speed('B', t), 0))})
    return out


def SQ_speed(bid, t):
    return SQ._at(bid, 'v', t - SQ.TOFF)


def casing_pos(c, tau):
    """空気抵抗(軽いので前進速度はすぐ失う)と重力、路面ではね返る"""
    age = tau - c['t']
    if age < 0:
        return None
    n = max(1, int(age / 0.01)); dt = age / n
    p = Vector(c['p0']); v = Vector(c['v0'])
    k = 2.6
    for _ in range(n):
        v = v * (1 - k * dt) + Vector((0, 0, -G * dt))
        p = p + v * dt
        if p.z < 0.006:
            p.z = 0.006; v = Vector((v.x * 0.5, v.y * 0.5, -v.z * 0.35))
    return p


def spark_pos(imp, i, age):
    r = _rng(imp['seed'] + i)
    a = r.uniform(0, 2 * math.pi); up = r.uniform(3.0, 8.0); out = r.uniform(2.0, 7.0)
    v = Vector((math.cos(a) * out, math.sin(a) * out + 6.0, up))      # 路面を車体の速度で流れる分(+Y)を少し足す
    return imp['p'] + Vector((0, 0, 0.02)) + v * age - Vector((0, 0, 0.5 * G * age * age))


def build(Rm, rd):
    fx = {}
    # マテリアル
    brass = Rm.simple_material('Brass', (0.75, 0.50, 0.12), rough=0.25, metallic=1.0)
    spark_m = Rm.emission_material('Spark', (1.0, 0.55, 0.15), 160.0)
    debris_m = Rm.simple_material('Debris', (0.06, 0.06, 0.06), rough=0.9)
    skid_m = Rm.simple_material('Skid', (0.012, 0.012, 0.012), rough=0.55)
    smoke_m = smoke_material(Rm)

    def ico(name, r, mat, sub=1):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=sub, radius=r, location=(0, 0, -50))
        o = bpy.context.active_object; o.name = name
        for c in list(o.users_collection):
            c.objects.unlink(o)
        Rm.link(o)
        o.data.materials.append(mat); o.hide_render = True
        return o
    # casings (小さな円筒)
    def cyl(name):
        bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.005, depth=0.019, location=(0, 0, -50))
        o = bpy.context.active_object; o.name = name
        for c in list(o.users_collection):
            c.objects.unlink(o)
        Rm.link(o); o.data.materials.append(brass); o.hide_render = True; o.rotation_mode = 'XYZ'
        return o
    fx['casing'] = [cyl('casing%d' % i) for i in range(CASING_SLOTS)]
    mat = spark_m
    fx['spark'] = [[ico('spark_%d_%d' % (s, i), 0.012, spark_m, 1) for i in range(SPARKS_PER)] for s in range(IMPACT_SLOTS)]
    mat = debris_m
    fx['debris'] = [[ico('debris_%d_%d' % (s, i), 0.022, debris_m, 1) for i in range(DEBRIS_PER)] for s in range(IMPACT_SLOTS)]
    fx['light'] = []
    for s in range(IMPACT_SLOTS):
        ld = bpy.data.lights.new('impL%d' % s, 'POINT'); ld.energy = 140.0; ld.color = (1.0, 0.65, 0.3); ld.shadow_soft_size = 0.1
        lo = Rm.link(bpy.data.objects.new('impL%d' % s, ld)); lo.hide_render = True; fx['light'].append(lo)
    # 煙
    sm_objs = []
    for i in range(SMOKE_N):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1.0, location=(0, 0, -50))
        o = bpy.context.active_object; o.name = 'smoke%d' % i
        for c in list(o.users_collection):
            c.objects.unlink(o)
        Rm.link(o); o.data.materials.append(smoke_m); o.hide_render = True
        sm_objs.append(o)
    fx['smoke'] = sm_objs
    fx['impacts'] = impacts(Rm)
    fx['casings_list'] = casings(Rm, rd)
    # スキッド跡: 後輪の接地点の軌跡(滑り始めから)。 細い四角を並べ、通った時刻から見えるようにする
    segs = []
    t0 = SQ.SLIDE_T0
    ts = np.arange(t0, SQ.SLIDE_T0 + SQ.SLIDE_DUR + 0.6, SKID_STEP)
    pts = []
    for tau in ts:
        m, d, pz = Rm.bike_matrix('B', SQ.TOFF + tau)
        pts.append(m @ Vector((0.0, -0.60, 0.0)))
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        d = b - a
        if d.length < 1e-4:
            continue
        side = Vector((-d.y, d.x, 0)).normalized() * 0.065
        V = [a - side, a + side, b + side, b - side]
        me = bpy.data.meshes.new('skid%d' % i); me.from_pydata([(v.x, v.y, 0.012) for v in V], [], [(0, 1, 2, 3)]); me.materials.append(skid_m); me.update()
        o = Rm.link(bpy.data.objects.new('skid%d' % i, me)); o.hide_render = True
        segs.append((ts[i], o))
    fx['skid'] = segs
    fx['smoke_emit'] = []
    r = _rng(3)
    te = SQ.SLIDE_T0 + 0.05
    while te < SQ.SLIDE_T0 + SQ.SLIDE_DUR:
        m, d, pz = Rm.bike_matrix('B', SQ.TOFF + te)
        p = m @ Vector((r.uniform(-0.1, 0.1), -0.60, 0.15))
        fx['smoke_emit'].append({'t': te, 'p': p, 'seed': int(r.randint(1 << 30)), 'w': r.uniform(-1, 1)})
        te += SMOKE_EVERY
    print('fx: impacts %d, skid segs %d, smoke puffs %d' % (len(fx['impacts']), len(segs), len(fx['smoke_emit'])))
    return fx


def smoke_material(Rm):
    """ボリュームの煙。球の中心ほど濃く(球は半径1、スケールで大きさを変える)、オブジェクトの色のアルファで濃さを時間で変える"""
    mat, nt, out = Rm.new_material('Smoke')
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    tc = nt.nodes.new('ShaderNodeTexCoord')
    ln = nt.nodes.new('ShaderNodeVectorMath'); ln.operation = 'LENGTH'
    nt.links.new(tc.outputs['Object'], ln.inputs[0])
    fall = Rm.math_node(nt, 'SUBTRACT', 1.0, ln.outputs['Value'], clamp=True)
    fall = Rm.math_node(nt, 'POWER', fall, 1.6)
    noi = nt.nodes.new('ShaderNodeTexNoise'); noi.inputs['Scale'].default_value = 3.0; noi.inputs['Detail'].default_value = 3.0
    nt.links.new(tc.outputs['Object'], noi.inputs['Vector'])
    nz = Rm.math_node(nt, 'MAXIMUM', Rm.math_node(nt, 'MULTIPLY', noi.outputs['Fac'], 1.6), 0.25)
    dens = Rm.math_node(nt, 'MULTIPLY', Rm.math_node(nt, 'MULTIPLY', fall, nz), oi.outputs['Alpha'])
    pv = nt.nodes.new('ShaderNodeVolumePrincipled')
    pv.inputs['Color'].default_value = (0.80, 0.80, 0.82, 1.0)
    nt.links.new(Rm.math_node(nt, 'MULTIPLY', dens, 3.2), pv.inputs['Density'])
    try:
        pv.inputs['Anisotropy'].default_value = 0.3
    except Exception:
        pass
    nt.links.new(pv.outputs['Volume'], out.inputs['Volume'])
    return mat


def _key_vis(o, j, vis, loc=None, scale=None, color=None):
    o.hide_render = not vis
    o.keyframe_insert('hide_render', frame=j)
    if vis and loc is not None:
        o.location = loc; o.keyframe_insert('location', frame=j)
    if vis and scale is not None:
        o.scale = scale; o.keyframe_insert('scale', frame=j)
    if vis and color is not None:
        o.color = color; o.keyframe_insert('color', frame=j)


def key(fx, j, tau, gun_world_fn):
    """フレーム j に、時刻 τ の状態のキーを打つ"""
    # 薬莢
    cl = fx['casings_list']
    slots = {}
    for n, c in enumerate(cl):
        age = tau - c['t']
        if 0 <= age < 0.9:
            slots[n % CASING_SLOTS] = (c, age)
    for s, o in enumerate(fx['casing']):
        if s in slots:
            c, age = slots[s]; p = casing_pos(c, tau)
            o.rotation_euler = (age * 31.0 + s, age * 17.0, s * 0.7)
            o.keyframe_insert('rotation_euler', frame=j)
            _key_vis(o, j, True, loc=p)
        else:
            _key_vis(o, j, False)
    # 着弾
    act = {}
    for n, imp in enumerate(fx['impacts']):
        age = tau - imp['t']
        if 0 <= age < 0.40:
            act[n % IMPACT_SLOTS] = (imp, age)
    for s in range(IMPACT_SLOTS):
        if s in act:
            imp, age = act[s]
            for i, o in enumerate(fx['spark'][s]):
                _key_vis(o, j, age < 0.26, loc=spark_pos(imp, i, age))
            for i, o in enumerate(fx['debris'][s]):
                r = _rng(imp['seed'] + 50 + i); a = r.uniform(0, 2 * math.pi)
                v = Vector((math.cos(a) * r.uniform(1, 3), math.sin(a) * r.uniform(1, 3) + 5.0, r.uniform(1.5, 3.5)))
                p = imp['p'] + v * age - Vector((0, 0, 0.5 * G * age * age))
                _key_vis(o, j, p.z > 0.0 and age < 0.4, loc=Vector((p.x, p.y, max(p.z, 0.0))))
            _key_vis(fx['light'][s], j, age < 0.09, loc=imp['p'] + Vector((0, 0, 0.18)))
        else:
            for o in fx['spark'][s] + fx['debris'][s]:
                _key_vis(o, j, False)
            _key_vis(fx['light'][s], j, False)
    # 煙
    ems = fx['smoke_emit']
    live = {}
    for n, e in enumerate(ems):
        age = tau - e['t']
        if 0 <= age < SMOKE_LIFE:
            live[n % SMOKE_N] = (e, age)
    for s, o in enumerate(fx['smoke']):
        if s in live:
            e, age = live[s]; u = age / SMOKE_LIFE
            r = _rng(e['seed']); dr = Vector((r.uniform(-0.4, 0.4) + 0.9 * e['w'], r.uniform(-0.6, 0.2), 0.0))
            p = e['p'] + Vector((0, 0, 0.35 + 0.5 * age)) + dr * age
            rad = 0.32 + 1.35 * math.sqrt(u)
            a = min(1.0, age / 0.12) * (1 - u) ** 1.4
            _key_vis(o, j, True, loc=p, scale=(rad, rad, rad * 0.8), color=(1, 1, 1, a))
        else:
            _key_vis(o, j, False)
    # スキッド跡
    for t0, o in fx['skid']:
        _key_vis(o, j, tau >= t0)
