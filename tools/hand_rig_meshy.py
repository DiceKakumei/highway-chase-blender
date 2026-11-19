"""Meshy製のグローブの手(assets/bike_B2/hand_open.glb、手を開いた静止姿勢)に指ボーンを入れて、銃を握らせる単体テスト。チェイスのシーンとは無関係。
  blender --background --python tools/hand_rig_meshy.py -- OUT
メッシュの座標(glTF): 指 = +Y、親指 = -X 側、手のひら = -Z、手の甲 = +Z。 手首のあたりが y=0、指先が y=0.28 (m)。 原点は中指の軸(x=0.0145)へずらす。
ボーン: Hand, Thumb1-3, Index1-3, Middle1-3, Ring1-3, Pinky1-3。 屈曲 = X 軸まわりに -θ(+Y が -Z=手のひら側へ倒れる)。
"""
import bpy, os, sys, math, json
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hand_rig_test as H

OUT = H.OUT; ROOT = H.ROOT
C0 = np.array([0.0145, 0.0, 0.0])
FINGERS = {   # mcp(付け根の関節), end(先端の中心線の終点), 半径
    'Index': dict(mcp=(-0.025, 0.172, 0.016), end=(-0.025, 0.258, 0.002), r=0.0125),
    'Middle': dict(mcp=(0.0145, 0.177, 0.018), end=(0.0145, 0.269, 0.002), r=0.0130),
    'Ring': dict(mcp=(0.050, 0.172, 0.014), end=(0.050, 0.258, -0.004), r=0.0125),
    'Pinky': dict(mcp=(0.070, 0.165, 0.010), end=(0.078, 0.229, -0.012), r=0.0110),
}
RATIO = (0.43, 0.30, 0.27)
THUMB_J = [(-0.030, 0.085, 0.000), (-0.052, 0.125, -0.012), (-0.068, 0.160, -0.022), (-0.080, 0.180, -0.027)]
THUMB_R = 0.016
HAND_HEAD = (0.0145, 0.03, 0.005); HAND_TAIL = (0.0170, 0.17, 0.012)
MH = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1.0]])      # メッシュ座標 -> 手の座標(手のひら=-X, 親指側=+Z, 指=+Y)。 銃の座標と向きがそろう


def sh(p):
    return np.asarray(p, float) - C0


def chain_joints(f):
    a = sh(f['mcp']); b = sh(f['end']); d = b - a
    pts = [a]; acc = 0.0
    for r in RATIO:
        acc += r; pts.append(a + d * acc)
    return pts                      # 4点: MCP, PIP, DIP, 先端


JOINTS = {n: chain_joints(f) for n, f in FINGERS.items()}
JOINTS['Thumb'] = [sh(p) for p in THUMB_J]
CHAINS = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
BONES = ['Hand'] + [c + str(i) for c in CHAINS for i in (1, 2, 3)]
RAD = {n: f['r'] for n, f in FINGERS.items()}; RAD['Thumb'] = THUMB_R


def smooth(t):
    t = min(max(t, 0.0), 1.0); return t * t * (3 - 2 * t)


def seg_dist(P, a, b):
    """点群 P(N,3) と線分 ab の距離と、線分上のパラメータ位置(a からの長さ)"""
    ab = b - a; L = np.linalg.norm(ab); u = ab / L
    t = np.clip((P - a) @ u, 0, L)
    return np.linalg.norm(P - (a + np.outer(t, u)), axis=1), t, L


def compute_weights(P):
    """P = 頂点(ずらし後) -> W (N, 骨数)。 いちばん近い指の中心線に属する頂点は、その指の骨(関節の近くでなめらかにつなぐ)。 ほかは Hand"""
    N = len(P); W = np.zeros((N, len(BONES))); bi = {b: i for i, b in enumerate(BONES)}
    best = np.full(N, 1e9); owner = np.full(N, -1); sarr = np.zeros(N)
    for ci, c in enumerate(CHAINS):
        J = JOINTS[c]; cum = 0.0; dmin = np.full(N, 1e9); smin = np.zeros(N)
        for k in range(3):
            d, t, L = seg_dist(P, J[k], J[k + 1])
            m = d < dmin; dmin = np.where(m, d, dmin); smin = np.where(m, cum + t, smin); cum += L
        m = dmin < best; best = np.where(m, dmin, best); owner = np.where(m, ci, owner); sarr = np.where(m, smin, sarr)
    own_final = np.full(N, -1)
    for i in range(N):
        W[i, 0] = 1.0
        ci = owner[i]; c = CHAINS[ci]; R = RAD[c] + (0.013 if c != 'Thumb' else 0.018)
        s = sarr[i]
        if best[i] > R or s < -0.012:
            continue
        own_final[i] = ci
        J = JOINTS[c]; l1 = np.linalg.norm(J[1] - J[0]); l2 = np.linalg.norm(J[2] - J[1])
        bw = 0.011 if c == 'Thumb' else 0.008
        t0 = smooth((s - 0.0) / (2 * bw) + 0.5); t1 = smooth((s - l1) / (2 * bw) + 0.5); t2 = smooth((s - l1 - l2) / (2 * bw) + 0.5)
        W[i, 0] = 1 - t0
        W[i, bi[c + '1']] = t0 * (1 - t1); W[i, bi[c + '2']] = t1 * (1 - t2); W[i, bi[c + '3']] = t2
    return W, own_final, best


def build():
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.join(ROOT, 'assets/bike_B2/hand_open.glb'), import_shading='SMOOTH')
    new = [o for o in bpy.data.objects if o not in before]; bpy.context.view_layer.update()
    ob = [x for x in new if x.type == 'MESH'][0]
    mw = ob.matrix_world.copy(); ob.parent = None; ob.data.transform(mw); ob.matrix_world = Matrix.Identity(4)
    for x in new:
        if x is not ob:
            bpy.data.objects.remove(x)
    me = ob.data
    # Blender(Z上) -> glTF(Y上) に戻し、 C0 だけずらす。 法線ごと変換するため、行列で変換する
    Mfix = Matrix(((1, 0, 0, -C0[0]), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))
    me.transform(Mfix); me.update()
    P = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', P); P = P.reshape(-1, 3).astype(float)
    print('hand mesh verts', len(P), 'bbox', P.min(0), P.max(0))
    W, owner, best = compute_weights(P)
    for b in BONES:
        ob.vertex_groups.new(name=b)
    for bi, b in enumerate(BONES):
        vg = ob.vertex_groups[b]
        for i in np.nonzero(W[:, bi] > 1e-3)[0]:
            vg.add([int(i)], float(W[i, bi]), 'REPLACE')
    arm = bpy.data.armatures.new('handArm'); ao = bpy.data.objects.new('handArm', arm); bpy.context.scene.collection.objects.link(ao)
    bpy.context.view_layer.objects.active = ao; bpy.ops.object.mode_set(mode='EDIT'); eb = arm.edit_bones
    h = eb.new('Hand'); h.head = tuple(sh(HAND_HEAD)); h.tail = tuple(sh(HAND_TAIL)); h.align_roll((0, 0, 1))
    for c in CHAINS:
        par = h; J = JOINTS[c]
        for i in range(3):
            b = eb.new(c + str(i + 1)); b.head = tuple(J[i]); b.tail = tuple(J[i + 1]); b.align_roll((0, 0, 1)); b.parent = par; par = b
    bpy.ops.object.mode_set(mode='OBJECT')
    ob.parent = ao; ob.matrix_parent_inverse = Matrix.Identity(4)
    md = ob.modifiers.new('arm', 'ARMATURE'); md.object = ao
    for pb in ao.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    return ao, ob, P, W, owner


def Rx(a):
    return H.rotvec((a, 0, 0))


def flex_D(curl):
    D = {}
    for c in ('Index', 'Middle', 'Ring', 'Pinky'):
        for i, a in enumerate(curl.get(c, (0, 0, 0))):
            D[c + str(i + 1)] = Rx(-math.radians(a))
    return D


def finger_points(c, ang, M):
    J = JOINTS[c]; p = J[0].copy(); cum = 0.0; out = []
    for k in range(3):
        d0 = J[k + 1] - J[k]; L = np.linalg.norm(d0); d0 = d0 / L
        cum += math.radians(ang[k]); d = Rx(-cum) @ d0
        for q in (0.25, 0.5, 0.75, 1.0):
            out.append(((M @ np.r_[p + d * L * q, 1.0])[:3], RAD[c], k))
        p = p + d * L
    return out


def gap(bvh, pt):
    return H.gap(bvh, pt)


def fit_fingers(bvh, M, a0):
    best = {}; total = 0.0
    for c in ('Middle', 'Ring', 'Pinky'):
        bs = None
        for a1 in range(10, 100, 10):
            for a2 in range(20, 120, 10):
                for a3 in range(0, 90, 10):
                    pts = finger_points(c, (a1, a2, a3), M)
                    sc = 0.0; segmin = [0.05, 0.05, 0.05]
                    for (pt, r, k) in pts:
                        g = gap(bvh, pt) - r
                        if g < -0.001:
                            sc += 4 * (-g - 0.001) * 1000
                        segmin[k] = min(segmin[k], max(g, 0))
                    sc += sum(segmin) * 300
                    tip = pts[-1][0]
                    if tip[0] > -0.004:
                        sc += 0.8 + (tip[0] + 0.004) * 20
                    sc += 0.0006 * (abs(a1 - a0[c][0]) + abs(a2 - a0[c][1]) + abs(a3 - a0[c][2]))
                    if bs is None or sc < bs[0]:
                        bs = (sc, (a1, a2, a3))
        best[c] = bs[1]; total += bs[0]
    return best, total


def thumb_D(M, wps):
    Mi = np.linalg.inv(M); J = JOINTS['Thumb']
    pts = [(Mi @ np.r_[np.asarray(w, float), 1.0])[:3] for w in wps]
    D = {}; Qp = np.eye(3); head = J[0].copy()
    for i in range(3):
        d0 = J[i + 1] - J[i]; L = np.linalg.norm(d0); d0 = d0 / L
        Q = H.swing3(d0, pts[i] - head)
        D['Thumb%d' % (i + 1)] = Qp.T @ Q; head = head + Q @ d0 * L; Qp = Q
    return D


def fit_thumb(bvh, M, target, rng):
    J = JOINTS['Thumb']; L = [np.linalg.norm(J[i + 1] - J[i]) for i in range(3)]
    cmc = (M @ np.r_[J[0], 1.0])[:3]; rest_d = [(M[:3, :3] @ (J[i + 1] - J[i])) / L[i] for i in range(3)]

    def pts_of(dirs):
        p = cmc.copy(); P = [p]; samples = []
        for i in range(3):
            q = p + dirs[i] * L[i]
            for t in (0.25, 0.5, 0.75, 1.0):
                samples.append(p + (q - p) * t)
            p = q; P.append(p)
        return P, samples

    def score(dirs):
        P, S = pts_of(dirs); sc = 0.0
        for s_ in S:
            g = gap(bvh, s_) - THUMB_R
            if g < -0.0005:
                sc += (-g - 0.0005) * 4000
        sc += np.linalg.norm(P[3] - target) * 80
        for i in range(2):
            ang = math.acos(max(-1, min(1, float(dirs[i] @ dirs[i + 1]))))
            sc += max(0.0, ang - 1.0) * 5
        for i in range(3):
            ang = math.acos(max(-1, min(1, float(dirs[i] @ rest_d[i]))))
            sc += max(0.0, ang - 1.7) * 5 + ang * 0.05
        return sc

    def rnd_dir(base, spread):
        v = base + rng.normal(0, spread, 3); return v / np.linalg.norm(v)

    best = None
    for it in range(9000):
        dirs = [rnd_dir(rest_d[i], 0.9) for i in range(3)]
        sc = score(dirs)
        if best is None or sc < best[0]:
            best = (sc, dirs)
    sc, dirs = best
    for it in range(4000):
        k = it % 3; nd = list(dirs); nd[k] = rnd_dir(dirs[k], 0.12)
        s2 = score(nd)
        if s2 < sc:
            sc, dirs = s2, nd
    P, S = pts_of(dirs)
    print('thumb fit score %.3f tip %s target %s' % (sc, np.round(P[3], 3), np.round(target, 3)))
    return P[1:], sc


def inside_report(ob, owner, gun):
    dg = bpy.context.evaluated_depsgraph_get(); ev = ob.evaluated_get(dg); me = ev.to_mesh(); M = ob.matrix_world
    bvh = BVHTree.FromObject(gun, dg); res = {}
    for i, v in enumerate(me.vertices):
        p = M @ v.co
        loc, nrm, idx, dist = bvh.find_nearest(p, 0.04)
        if loc is None:
            continue
        hits = 0; q = p.copy(); dn = Vector((1, 0.0013, 0.0007))
        for _ in range(12):
            h = bvh.ray_cast(q, dn)
            if h[0] is None:
                break
            hits += 1; q = h[0] + dn.normalized() * 1e-5
        name = 'palm' if owner[i] < 0 else CHAINS[owner[i]]
        r = res.setdefault(name, [0, 0.0])
        if hits % 2 == 1:
            r[0] += 1; r[1] = max(r[1], dist)
    ev.to_mesh_clear()
    return res


def main():
    sc, cam = H.setup_scene()
    ao, ob, P, W, owner = build()
    print('weights: verts per owner', {('palm' if k < 0 else CHAINS[k]): int((owner == k).sum()) for k in range(-1, 5)})
    # ---- 1. 静止(開いた手)と、拳(リグの確認)。 メッシュ座標のまま撮る
    ao.matrix_world = Matrix.Identity(4)
    c = (0.0, 0.14, 0.0)
    H.set_pose(ao, {})
    H.shoot(sc, cam, '1a_rest_back', c, (0, 0.05, 1), 0.55); H.shoot(sc, cam, '1b_rest_palm', c, (0, 0.05, -1), 0.55); H.shoot(sc, cam, '1c_rest_side', c, (1, 0.1, 0.1), 0.55)
    fist = flex_D({k: (85, 95, 65) for k in ('Index', 'Middle', 'Ring', 'Pinky')})
    fist.update({'Thumb1': H.rotvec((0.0, 0.5, 0.0))})
    H.set_pose(ao, fist)
    H.shoot(sc, cam, '2a_fist_back', c, (0, 0.05, 1), 0.55); H.shoot(sc, cam, '2b_fist_palm', c, (0, 0.05, -1), 0.55); H.shoot(sc, cam, '2c_fist_side', c, (1, 0.3, 0.2), 0.55)
    # ---- 2. 銃を握る: 手の位置・向きと指の曲げを、銃のメッシュに合わせて探す
    gun = H.import_gun()
    dg0 = bpy.context.evaluated_depsgraph_get(); bvh = BVHTree.FromObject(gun, dg0)
    palm_idx = np.nonzero((W[:, 0] > 0.99) & (owner < 0))[0]; palm_idx = palm_idx[::25]
    palm_pts = P[palm_idx]
    a0 = {'Middle': (60, 90, 50), 'Ring': (60, 90, 50), 'Pinky': (55, 85, 45)}
    cand = []
    for yw in (-8, 0, 10, 20):
        for x in (0.032, 0.040, 0.048, 0.056):
            for y in (-0.120, -0.104, -0.088, -0.072, -0.056):
                for z in (-0.035, -0.022, -0.010):
                    cand.append(((x, y, z), yw))
    bestc = None
    for (pos, yw) in cand:
        Mh = np.array(Matrix.Translation(pos) @ Matrix.Rotation(math.radians(yw), 4, 'Z')) @ MH
        pen = 0.0; gs = []
        for q in palm_pts:
            g = gap(bvh, (Mh @ np.r_[q, 1.0])[:3]); gs.append(g)
            if g < 0.0005:
                pen += (0.0005 - g) * 1000
        gs = np.sort(gs); pen += float(np.mean(np.maximum(gs[:max(3, len(gs) // 8)] - 0.002, 0))) * 3000   # 手のひらを銃に接触させる
        if bestc is not None and pen * 3 > bestc[0]:
            continue
        ang, scr = fit_fingers(bvh, Mh, a0)
        scr += pen * 3
        if bestc is None or scr < bestc[0]:
            bestc = (scr, pos, yw, ang, Mh); print('cand pos %s yaw %d score %.2f palmpen %.2f' % (np.round(pos, 3), yw, scr, pen))
    scr, pos, yw, ang, Mh = bestc
    print('hand BEST pos', pos, 'yaw', yw, 'score %.2f' % scr, 'angles', ang)
    ao.matrix_world = Matrix(Mh)
    rng = np.random.default_rng(3)
    wps, tscore = fit_thumb(bvh, Mh, np.array([-0.030, 0.012, 0.030]), rng)
    curl = {'Index': (8, 14, 8)}; curl.update(ang)
    D = flex_D(curl); D.update(thumb_D(Mh, wps))
    H.set_pose(ao, D)
    rep = inside_report(ob, owner, gun)
    for g, r in rep.items():
        print('PEN %-7s inside=%4d maxdepth=%.1fmm' % (g, r[0], r[1] * 1000))
    ctr = (0.0, 0.0, -0.005)
    for nm, d in (('3a_grip_right', (1, 0, 0.15)), ('3b_grip_left', (-1, 0, 0.15)), ('3c_grip_front', (0, 1, 0.1)), ('3d_grip_rear', (0, -1, 0.1)),
                  ('3e_grip_top', (0, 0, 1)), ('3f_grip_34', (0.7, -0.7, 0.35)), ('3g_grip_34f', (-0.7, 0.7, 0.35))):
        H.shoot(sc, cam, nm, ctr, d, 0.5)
    gun.hide_render = True
    H.shoot(sc, cam, '4a_hand_only_right', ctr, (1, 0, 0.15), 0.5); H.shoot(sc, cam, '4b_hand_only_left', ctr, (-1, 0, 0.15), 0.5)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, 'hand_rig_meshy.blend'))


main()
