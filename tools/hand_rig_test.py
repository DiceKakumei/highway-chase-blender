"""指ボーン付きの手(グローブ)を手作りし、銃を握らせて確認する。チェイスのシーンとは無関係の単体テスト。
  blender --background --python tools/hand_rig_test.py -- OUT [params.json]
手のローカル座標(静止姿勢=手を開いて平ら): 原点=手首の中心、+Y=指の向き(前腕に沿う)、手のひらは -X を向く、+Z=親指側(人さし指が上)、単位 m。
銃のローカル座標: x=右、y=銃口、z=上、原点=グリップの中心(chase/rider.py の import_gun と同じ)。
ボーン: Hand, Thumb1-3, Index1-3, Middle1-3, Ring1-3, Pinky1-3。指の曲げ(屈曲) = 手のローカル +Z 回りの正回転(+Y が -X へ倒れる)。
"""
import bpy, bmesh, math, os, sys, json
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

OUT = sys.argv[sys.argv.index('--') + 1]; os.makedirs(OUT, exist_ok=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRM = {}
if '--' in sys.argv and len(sys.argv) > sys.argv.index('--') + 2:
    PRM = json.load(open(sys.argv[sys.argv.index('--') + 2]))

# ------------------------------------------------------------------ 寸法(成人の手、m)
FING = {   # 名前: (MCPのz, 太さ, [基節, 中節, 末節])
    'Index': (0.034, 0.0098, [0.040, 0.025, 0.020]),
    'Middle': (0.012, 0.0100, [0.044, 0.028, 0.022]),
    'Ring': (-0.010, 0.0094, [0.040, 0.026, 0.020]),
    'Pinky': (-0.030, 0.0084, [0.031, 0.020, 0.018]),
}
KNUCKLE_Y = 0.090
KNUCKLE_X = 0.002
THUMB = {'cmc': (-0.004, 0.022, 0.026), 'len': [0.040, 0.031, 0.025], 'r': 0.0132,
         'dir': [(-0.35, 0.62, 0.70), (-0.25, 0.72, 0.65), (-0.15, 0.80, 0.58)]}
BONES = ['Hand'] + [f + str(i) for f in ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky') for i in (1, 2, 3)]

# ------------------------------------------------------------------ 形づくり
V = []; F = []; WT = []          # 頂点, 面, 重み {骨: w}
GROUP = []                        # 頂点がどの指のものか(貫通の集計用)


def smooth(t):
    t = min(max(t, 0.0), 1.0); return t * t * (3 - 2 * t)


def chain_w(s, joints, names, bw):
    """中心線上の位置 s -> 骨の重み。 names[0]=手の骨, joints = 各関節の s"""
    t = [smooth((s - j) / (2 * bw) + 0.5) for j in joints]
    w = {names[0]: 1 - t[0]}
    for k in range(len(joints)):
        nxt = t[k + 1] if k + 1 < len(joints) else 0.0
        w[names[k + 1]] = t[k] * (1 - nxt)
    return w


def add_rings(rings, wfun, grp, cap_start=None, cap_end=None):
    """rings = [(中心, u軸, v軸, ru, rv, s)] 全て同じ頂点数 n の輪を面でつなぐ"""
    n = 14; base = len(V); idx = []
    for (c, u, v, ru, rv, s) in rings:
        row = []
        for i in range(n):
            a = 2 * math.pi * i / n
            V.append(np.asarray(c) + ru * math.cos(a) * np.asarray(u) + rv * math.sin(a) * np.asarray(v)); WT.append(wfun(s, a)); GROUP.append(grp)
            row.append(len(V) - 1)
        idx.append(row)
    for r in range(len(idx) - 1):
        for i in range(n):
            j = (i + 1) % n
            F.append((idx[r][i], idx[r][j], idx[r + 1][j], idx[r + 1][i]))
    for cap, row, rev in ((cap_start, idx[0], True), (cap_end, idx[-1], False)):
        if cap is not None:
            V.append(np.asarray(cap[0])); WT.append(wfun(cap[1], 0.0)); GROUP.append(grp); ci = len(V) - 1
            for i in range(n):
                j = (i + 1) % n
                F.append((ci, row[j], row[i]) if rev else (ci, row[i], row[j]))


def finger_tube(pts, r0, r1, names, joints, bw, grp, ratio=0.92, tipr=True):
    """pts = 中心線(折れ線)の点列 -> 輪を並べる。 pts[0] の手前から。 s = pts[1] からの距離"""
    pts = [np.asarray(p, float) for p in pts]
    # 中心線をなめらかに(Catmull-Rom)サンプル
    P = [pts[0] * 2 - pts[1]] + pts + [pts[-1] * 2 - pts[-2]]
    samp = []
    for k in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[k - 1], P[k], P[k + 1], P[k + 2]
        m = max(2, int(np.linalg.norm(p2 - p1) / 0.0035))
        for q in range(m):
            t = q / m
            samp.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    samp.append(pts[-1])
    samp = np.array(samp)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(samp, axis=0), axis=1))]
    s0 = seg[np.argmin(np.linalg.norm(samp - pts[1], axis=1))]
    s_arr = seg - s0; L = seg[-1] - s0
    rings = []
    u = np.array([1.0, 0, 0]); v = None
    for k in range(len(samp)):
        tg = samp[min(k + 1, len(samp) - 1)] - samp[max(k - 1, 0)]; tg /= np.linalg.norm(tg)
        u = u - (u @ tg) * tg
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(tg, [0, 0, 1.0])
        u /= np.linalg.norm(u); v = np.cross(tg, u)
        fr = min(max(s_arr[k] / L, 0.0), 1.0)
        r = r0 + (r1 - r0) * fr
        rings.append([samp[k], u.copy(), v.copy(), r * ratio, r, s_arr[k]])
    # 先端のまるめ
    tg = samp[-1] - samp[-2]; tg /= np.linalg.norm(tg); rt = rings[-1][4]
    for a in (25, 50, 70, 82):
        c = samp[-1] + tg * rt * math.sin(math.radians(a)) * 0.9
        k = math.cos(math.radians(a))
        rings.append([c, rings[-1][1], rings[-1][2], rt * ratio * k, rt * k, L + rt * math.sin(math.radians(a)) * 0.9])
    tip = samp[-1] + tg * rt * 0.9
    wf = lambda s, a: chain_w(s, joints, names, bw)
    add_rings([tuple(r) for r in rings], wf, grp, cap_start=(samp[0], s_arr[0]), cap_end=(tip, L + rt))
    return samp


def build_geometry():
    # --- 手のひら(輪を y 方向に並べる)。 輪は x(厚み)と z(幅)の楕円
    ys = [-0.045, -0.040, -0.020, 0.0, 0.025, 0.050, 0.072, 0.090, 0.098]
    hz = [0.029, 0.029, 0.027, 0.0285, 0.034, 0.039, 0.0415, 0.0430, 0.034]       # 幅の半分
    zc = [0.0, 0.0, 0.0, 0.0, 0.001, 0.002, 0.002, 0.002, 0.002]
    hx = [0.0195, 0.0190, 0.0165, 0.0165, 0.0160, 0.0150, 0.0140, 0.0125, 0.0085]  # 厚みの半分
    xc = [0.0, 0.0, 0.001, 0.001, 0.0015, 0.002, 0.002, 0.002, 0.002]
    yy = np.arange(-0.045, 0.0981, 0.0075)
    rings = []
    for y in yy:
        rings.append(((np.interp(y, ys, xc), y, np.interp(y, ys, zc)), (1, 0, 0), (0, 0, 1), np.interp(y, ys, hx), np.interp(y, ys, hz), y))
    add_rings(rings, lambda s, a: {'Hand': 1.0}, 'Hand', cap_start=((0, -0.045, 0), -0.045), cap_end=((0.002, 0.098, 0.002), 0.098))
    # --- 4本の指
    for nm, (zf, r, ls) in FING.items():
        names = ['Hand', nm + '1', nm + '2', nm + '3']
        j = [0.0, ls[0], ls[0] + ls[1]]
        p0 = np.array([KNUCKLE_X, KNUCKLE_Y, zf])
        pts = [p0 - np.array([0, 0.012, 0]), p0, p0 + np.array([0, ls[0], 0]), p0 + np.array([0, ls[0] + ls[1], 0]), p0 + np.array([0, sum(ls), 0])]
        finger_tube(pts, r * 1.04, r * 0.82, names, j, 0.0075, nm)
    # --- 親指
    cmc = np.array(THUMB['cmc']); ds = [np.array(d) / np.linalg.norm(d) for d in THUMB['dir']]
    P = [cmc]
    for d, l in zip(ds, THUMB['len']):
        P.append(P[-1] + d * l)
    names = ['Hand', 'Thumb1', 'Thumb2', 'Thumb3']
    j = [0.0, THUMB['len'][0], THUMB['len'][0] + THUMB['len'][1]]
    finger_tube([cmc - ds[0] * 0.02] + P, THUMB['r'] * 1.15, THUMB['r'] * 0.8, names, j, 0.011, 'Thumb', ratio=0.96)
    return P


def make_hand():
    TP = build_geometry()
    me = bpy.data.meshes.new('hand'); me.from_pydata([tuple(map(float, v)) for v in V], [], F); me.update()
    ob = bpy.data.objects.new('hand', me); bpy.context.scene.collection.objects.link(ob)
    for p in me.polygons:
        p.use_smooth = True
    for b in BONES:
        ob.vertex_groups.new(name=b)
    for i, w in enumerate(WT):
        tot = sum(w.values())
        for b, x in w.items():
            if x > 1e-4:
                ob.vertex_groups[b].add([i], x / tot, 'REPLACE')
    # アーマチュア
    arm = bpy.data.armatures.new('handArm'); ao = bpy.data.objects.new('handArm', arm); bpy.context.scene.collection.objects.link(ao)
    bpy.context.view_layer.objects.active = ao; bpy.ops.object.mode_set(mode='EDIT')
    eb = arm.edit_bones
    h = eb.new('Hand'); h.head = (0, 0, 0); h.tail = (0, KNUCKLE_Y, 0); h.roll = 0
    for nm, (zf, r, ls) in FING.items():
        par = h; y = KNUCKLE_Y
        for i, l in enumerate(ls):
            b = eb.new(nm + str(i + 1)); b.head = (KNUCKLE_X, y, zf); b.tail = (KNUCKLE_X, y + l, zf); b.roll = 0; b.parent = par; par = b; y += l
    cmc = np.array(THUMB['cmc']); par = h; pos = cmc
    for i, (d, l) in enumerate(zip(THUMB['dir'], THUMB['len'])):
        d = np.array(d) / np.linalg.norm(d)
        b = eb.new('Thumb%d' % (i + 1)); b.head = tuple(pos); b.tail = tuple(pos + d * l); b.align_roll((-1.0, 0.0, 0.0)); b.parent = par; par = b; pos = pos + d * l
    bpy.ops.object.mode_set(mode='OBJECT')
    ob.parent = ao; ob.matrix_parent_inverse = Matrix.Identity(4)
    md = ob.modifiers.new('arm', 'ARMATURE'); md.object = ao
    for pb in ao.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    return ao, ob


def rotz(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rotvec(v):
    v = np.asarray(v, float); a = np.linalg.norm(v)
    if a < 1e-9:
        return np.eye(3)
    k = v / a; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def set_pose(ao, D):
    """D = {骨: 3x3 (アーマチュア空間の回転。その骨の関節まわり)}。 基底 = R^-1 D R"""
    for pb in ao.pose.bones:
        R = np.array(pb.bone.matrix_local)[:3, :3]
        Dm = D.get(pb.name, np.eye(3))
        pb.rotation_quaternion = Matrix(R.T @ Dm @ R).to_quaternion()
        pb.location = (0, 0, 0)
    bpy.context.view_layer.update()


def swing3(a, b):
    a = np.asarray(a, float) / np.linalg.norm(a); b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b); sn = np.linalg.norm(v)
    if sn < 1e-9:
        return np.eye(3)
    return rotvec(v / sn * math.atan2(sn, c))


def thumb_to_waypoints(ao, wps):
    """親指の3本のボーンを、銃の座標の通過点 wps=[MCP, IP, 先端] へ向ける -> D(手の座標系=アーマチュア空間の回転)。 ao.matrix_world を先に設定しておく"""
    M = np.array(ao.matrix_world); Mi = np.linalg.inv(M)
    cmc = np.array(THUMB['cmc'], float)
    pts = [cmc] + [(Mi @ np.r_[np.asarray(w, float), 1.0])[:3] for w in wps]
    D = {}; Qp = np.eye(3); head = cmc.copy()
    for i in range(3):
        d0 = np.array(THUMB['dir'][i], float); d0 /= np.linalg.norm(d0)
        t = pts[i + 1] - head
        Q = swing3(d0, t)
        D['Thumb%d' % (i + 1)] = Qp.T @ Q
        head = head + Q @ d0 * THUMB['len'][i]; Qp = Q
    return D


def curl_pose(curl, thumb):
    """curl = {'Index': (a1,a2,a3)(度), ...}、 thumb = {'Thumb1': rotvec(度), ...}"""
    D = {}
    for nm in FING:
        for i, a in enumerate(curl.get(nm, (0, 0, 0))):
            D[nm + str(i + 1)] = rotz(math.radians(a))
    for k, v in thumb.items():
        D[k] = rotvec(np.radians(v))
    return D


# ------------------------------------------------------------------ 銃
def import_gun():
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.join(ROOT, 'assets/bike_B2/gun.glb'), import_shading='SMOOTH')
    new = [o for o in bpy.data.objects if o not in before]; bpy.context.view_layer.update()
    o = [x for x in new if x.type == 'MESH'][0]
    mw = o.matrix_world.copy(); o.parent = None; o.data.transform(mw); o.matrix_world = Matrix.Identity(4)
    for x in new:
        if x is not o:
            bpy.data.objects.remove(x)
    me = o.data
    ox = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', ox); ox = ox.reshape(-1, 3)
    ox[:, 0] -= 0.065; ox[:, 2] -= 0.045
    c, s = math.cos(-math.pi / 2), math.sin(-math.pi / 2)
    out = np.c_[c * ox[:, 0] - s * ox[:, 1], s * ox[:, 0] + c * ox[:, 1], ox[:, 2]].astype(np.float32)
    me.vertices.foreach_set('co', out.ravel()); me.update()
    for m in me.materials:
        for n in m.node_tree.nodes:
            if n.type == 'BSDF_PRINCIPLED':
                for l in list(n.inputs['Alpha'].links):
                    m.node_tree.links.remove(l)
                n.inputs['Alpha'].default_value = 1.0
    return o


def glove_material():
    m = bpy.data.materials.new('glove'); m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    b.inputs['Base Color'].default_value = (0.05, 0.10, 0.34, 1); b.inputs['Roughness'].default_value = 0.55
    return m


# ------------------------------------------------------------------ 確認
def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene; sc.render.engine = 'CYCLES'; sc.cycles.samples = int(PRM.get('samples', 24)); sc.cycles.device = 'CPU'
    sc.render.resolution_x = 640; sc.render.resolution_y = 640
    w = bpy.data.worlds.new('w'); w.use_nodes = True; w.node_tree.nodes['Background'].inputs[1].default_value = 1.1; w.node_tree.nodes['Background'].inputs[0].default_value = (0.8, 0.82, 0.86, 1); sc.world = w
    for loc, e in (((-0.6, -0.8, 1.0), 120), ((0.8, -0.4, 0.7), 80), ((0.0, 0.8, 0.6), 60)):
        ld = bpy.data.lights.new('l', 'AREA'); ld.energy = e; ld.size = 0.8; lo = bpy.data.objects.new('l', ld); lo.location = loc; sc.collection.objects.link(lo)
        lo.rotation_euler = (Vector((0, 0, 0)) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.lens = 80
    return sc, cam


def shoot(sc, cam, name, center, d, dist=0.42, up=(0, 0, 1)):
    d = Vector(d).normalized(); cam.location = Vector(center) + d * dist
    cam.rotation_euler = (Vector(center) - cam.location).to_track_quat('-Z', 'Y' if abs(d.z) < 0.9 else 'X').to_euler()
    sc.render.filepath = os.path.join(OUT, name + '.png'); bpy.ops.render.render(write_still=True)


def fk_points(nm, ang, M):
    """指 nm を、屈曲角 ang=(a1,a2,a3)度 で曲げたときの、中心線上のサンプル点(銃の座標)と半径 -> [(点, 半径, 区間番号)]"""
    zf, r, ls = FING[nm]; p = np.array([KNUCKLE_X, KNUCKLE_Y, zf]); a = 0.0; out = []
    for k in range(3):
        a += math.radians(ang[k]); d = np.array([-math.sin(a), math.cos(a), 0.0])
        for q in (0.25, 0.5, 0.75, 1.0):
            pt = p + d * ls[k] * q
            out.append(((M @ np.r_[pt, 1.0])[:3], r * (1.0 - 0.15 * (k + q) / 3), k))
        p = p + d * ls[k]
    return out


def gap(bvh, pt):
    loc, nrm, idx, dist = bvh.find_nearest(Vector(pt), 0.05)
    if loc is None:
        return 0.05
    return dist if (Vector(pt) - loc).dot(nrm) >= 0 else -dist


def fit_fingers(bvh, M, a0):
    """中指・薬指・小指の曲げ角を探す(貫通しない範囲で、グリップに沿って巻く)。 -> (角度の辞書, 点数)"""
    best = {}; total = 0.0
    for nm in ('Middle', 'Ring', 'Pinky'):
        bs = None
        for a1 in range(10, 100, 10):
            for a2 in range(20, 120, 10):
                for a3 in range(0, 90, 10):
                    pts = fk_points(nm, (a1, a2, a3), M)
                    sc = 0.0; segmin = [0.05, 0.05, 0.05]
                    for (pt, r, k) in pts:
                        g = gap(bvh, pt) - r
                        if g < -0.001:
                            sc += 40 * (-g - 0.001) * 1000 / 10
                        segmin[k] = min(segmin[k], max(g, 0))
                    sc += sum(segmin) * 300
                    tip = pts[-1][0]
                    if tip[0] > -0.004:
                        sc += 0.8 + (tip[0] + 0.004) * 20
                    sc += 0.0006 * (abs(a1 - a0[nm][0]) + abs(a2 - a0[nm][1]) + abs(a3 - a0[nm][2]))
                    if bs is None or sc < bs[0]:
                        bs = (sc, (a1, a2, a3))
        best[nm] = bs[1]; total += bs[0]
    return best, total


def palm_penalty(bvh, M):
    pen = 0.0
    for y in (0.0, 0.03, 0.06, 0.088):
        for z in (-0.03, 0.0, 0.03):
            for x in (-0.0165, -0.012):
                g = gap(bvh, (M @ np.array([x, y, z, 1.0]))[:3])
                if g < 0.0005:
                    pen += (0.0005 - g) * 1000
    return pen


def pen_report(ob, gun, ao):
    dg = bpy.context.evaluated_depsgraph_get(); ev = ob.evaluated_get(dg); me = ev.to_mesh()
    M = ob.matrix_world
    bvh = BVHTree.FromObject(gun, dg)
    res = {}
    for i, v in enumerate(me.vertices):
        p = M @ v.co
        loc, nrm, idx, dist = bvh.find_nearest(p, 0.06)
        if loc is None:
            continue
        hits = 0; q = p.copy(); dirn = Vector((1, 0.0013, 0.0007))
        for _ in range(12):
            h = bvh.ray_cast(q, dirn)
            if h[0] is None:
                break
            hits += 1; q = h[0] + dirn.normalized() * 1e-5
        inside = hits % 2 == 1
        g = GROUP[i]
        r = res.setdefault(g, [0, 0.0, 1e9])
        if inside:
            r[0] += 1; r[1] = max(r[1], dist)
        r[2] = min(r[2], dist if not inside else -dist)
    ev.to_mesh_clear()
    return res


def main():
    sc, cam = setup_scene()
    ao, ob = make_hand(); ob.data.materials.append(glove_material())
    print('hand verts', len(V), 'faces', len(F), 'bones', len(ao.data.bones))
    # --- 1. 静止(開いた手)と、全部曲げた拳
    ao.matrix_world = Matrix.Identity(4)
    c = (0.0, 0.05, 0.0)
    set_pose(ao, {}); shoot(sc, cam, '1a_rest_palm', c, (-1, 0, 0.2), 0.40); shoot(sc, cam, '1b_rest_back', c, (1, 0, 0.2), 0.40); shoot(sc, cam, '1c_rest_top', c, (0, 0, 1), 0.40)
    fist = curl_pose({k: (85, 95, 65) for k in FING}, {'Thumb1': (0, 0, 0)})
    set_pose(ao, fist); shoot(sc, cam, '2a_fist_side', c, (-1, 0.3, 0.2), 0.40); shoot(sc, cam, '2b_fist_front', c, (0, 1, 0.2), 0.40)
    # --- 2. 銃を握る
    gun = import_gun()
    P = dict(PRM)
    dg0 = bpy.context.evaluated_depsgraph_get(); bvh = BVHTree.FromObject(gun, dg0)
    a0 = {'Middle': (60, 90, 50), 'Ring': (60, 90, 50), 'Pinky': (55, 85, 45)}
    if 'pos' in P:
        cand = [(tuple(P['pos']), P.get('yaw', 0.0))]
    else:
        cand = [((x, y, -0.020), yw) for yw in (0, 12, 24, 36) for x in (0.034, 0.038, 0.042) for y in (-0.085, -0.070, -0.055, -0.040)]
    bestc = None
    for (pos, yw) in cand:
        Mh = np.array(Matrix.Translation(pos) @ Matrix.Rotation(math.radians(yw), 4, 'Z'))
        ang, sc = fit_fingers(bvh, Mh, a0)
        sc += palm_penalty(bvh, Mh) * 3
        if bestc is None or sc < bestc[0]:
            bestc = (sc, pos, yw, ang)
        print('cand pos %s yaw %d score %.2f' % (np.round(pos, 3), yw, sc))
    best_sc, pos, yaw, ang = bestc; roll = P.get('roll', 0.0)
    print('hand BEST pos', pos, 'yaw', yaw, 'score %.2f' % best_sc, 'angles', ang)
    ao.matrix_world = Matrix.Translation(pos) @ Matrix.Rotation(math.radians(yaw), 4, 'Z') @ Matrix.Rotation(math.radians(roll), 4, 'Y')
    curl = {'Index': (8, 14, 8)}; curl.update(ang)
    curl.update({k: tuple(v) for k, v in P.get('curl', {}).items()})
    thumb = P.get('thumb', {'Thumb1': (0, 0, 50), 'Thumb2': (0, 0, 30), 'Thumb3': (0, 0, 15)})
    Dg = curl_pose({k: tuple(v) for k, v in curl.items()}, {})
    wps = P.get('thumb_wp', [(-0.010, -0.054, 0.020), (-0.031, -0.022, 0.030), (-0.031, 0.004, 0.030)])
    Dg.update(thumb_to_waypoints(ao, wps))
    set_pose(ao, Dg)
    rep = pen_report(ob, gun, ao)
    for g, r in rep.items():
        print('PEN %-7s inside=%4d maxdepth=%.1fmm mindist=%.1fmm' % (g, r[0], r[1] * 1000, r[2] * 1000))
    ctr = (0.0, 0.0, -0.005)
    for nm, d in (('3a_grip_right', (1, 0, 0.15)), ('3b_grip_left', (-1, 0, 0.15)), ('3c_grip_front', (0, 1, 0.1)), ('3d_grip_rear', (0, -1, 0.1)),
                  ('3e_grip_top', (0, 0, 1)), ('3f_grip_34', (0.7, -0.7, 0.35)), ('3g_grip_34f', (-0.7, 0.7, 0.35))):
        shoot(sc, cam, nm, ctr, d, 0.45)
    # 銃を隠して、手だけも(つかむ形が見える)
    gun.hide_render = True
    shoot(sc, cam, '4a_hand_only_right', ctr, (1, 0, 0.15), 0.40); shoot(sc, cam, '4b_hand_only_left', ctr, (-1, 0, 0.15), 0.40)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, 'hand_test.blend'))


main()
