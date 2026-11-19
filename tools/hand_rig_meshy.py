"""Meshy製のグローブの手(assets/bike_B2/hand_open.glb、手を開いた静止姿勢)に指ボーンを入れて、拳銃を握らせる単体テスト。チェイスのシーンとは無関係。
  blender --background --python tools/hand_rig_meshy.py -- OUT
計算(握りの探索・仕様)は tools/hand_grip_core.py。 仕様の出典: リサーチ/2026-11-19_拳銃の握り方_現実の挙動.md
メッシュの座標(glTF): 指 = +Y、親指 = -X 側、手のひら = -Z、手の甲 = +Z。 手首のあたりが y=0、指先が y=0.28 (m)。 原点は中指の軸(x=0.0145)へずらす。
ボーン: Hand, Thumb1-3, Index1-3, Middle1-3, Ring1-3, Pinky1-3。 屈曲 = X 軸まわりに -θ(+Y が -Z=手のひら側へ倒れる)。
出力: OUT/hand_grip_poses.json (2状態: ready=構え(発砲前)、fire=発砲中)。 リポジトリの assets/bike_B2/hand_grip_poses.json にコピーして使う。
"""
import bpy, os, sys, math, json
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hand_rig_test as H
import hand_grip_core as C

OUT = H.OUT; ROOT = H.ROOT
CHAINS = C.CHAINS; BONES = C.BONES; JOINTS = C.JOINTS


def build(wvar='w0'):
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
    Mfix = Matrix(((1, 0, 0, -C.C0[0]), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))
    me.transform(Mfix); me.update()
    P = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', P); P = P.reshape(-1, 3).astype(float)
    print('hand mesh verts', len(P), 'bbox', P.min(0), P.max(0))
    W, owner, best = C.compute_weights(P, C.WEIGHT_VARIANTS[wvar])
    set_weights(ob, W)
    arm = bpy.data.armatures.new('handArm'); ao = bpy.data.objects.new('handArm', arm); bpy.context.scene.collection.objects.link(ao)
    bpy.context.view_layer.objects.active = ao; bpy.ops.object.mode_set(mode='EDIT'); eb = arm.edit_bones
    h = eb.new('Hand'); h.head = tuple(C.sh(C.HAND_HEAD)); h.tail = tuple(C.sh(C.HAND_TAIL)); h.align_roll((0, 0, 1))
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


def set_weights(ob, W):
    for g in list(ob.vertex_groups):
        ob.vertex_groups.remove(g)
    for bi, b in enumerate(BONES):
        vg = ob.vertex_groups.new(name=b)
        for i in np.nonzero(W[:, bi] > 1e-3)[0]:
            vg.add([int(i)], float(W[i, bi]), 'REPLACE')


def blender_sdf(gun):
    dg = bpy.context.evaluated_depsgraph_get(); bvh = BVHTree.FromObject(gun, dg)

    def dist(pts):
        out = np.empty(len(pts))
        for i in range(len(pts)):
            p = Vector(pts[i]); loc, nrm, idx, d = bvh.find_nearest(p, 0.04)
            out[i] = 0.04 if loc is None else (d if (p - loc).dot(nrm) >= 0 else -d)
        return out
    return C.SDF.build(C.SDF_ORG, C.SDF_STEP, C.SDF_SHAPE, dist), bvh


def inside_report(ob, owner, bvh):
    """頂点ごとの貫通。 normal = 最寄りの面の法線の符号(深さ=最寄り面までの距離)、 ray = 3方向のレイの偶奇の多数決(2/3)。 銃のメッシュは内部の面があり、 レイ1本だと偽陽性が出るため"""
    dg = bpy.context.evaluated_depsgraph_get(); ev = ob.evaluated_get(dg); me = ev.to_mesh(); M = ob.matrix_world
    res = {}
    dirs = [Vector((1, 0.0013, 0.0007)), Vector((0.0009, 1, 0.0011)), Vector((0.0008, 0.0012, 1))]
    for i, v in enumerate(me.vertices):
        p = M @ v.co
        loc, nrm, idx, dist = bvh.find_nearest(p, 0.04)
        if loc is None:
            continue
        name = 'palm' if owner[i] < 0 else CHAINS[owner[i]]
        r = res.setdefault(name, [0, 0.0, 0, 0.0])
        if (p - loc).dot(nrm) < 0:
            r[0] += 1; r[1] = max(r[1], dist)
        odd = 0
        for dn in dirs:
            hits = 0; q = p.copy()
            for _ in range(12):
                h = bvh.ray_cast(q, dn)
                if h[0] is None:
                    break
                hits += 1; q = h[0] + dn.normalized() * 1e-5
            odd += hits % 2
        if odd >= 2:
            r[2] += 1; r[3] = max(r[3], dist)
    ev.to_mesh_clear()
    return res


def pose_dict(S, state):
    curl = {k: tuple(v) for k, v in S['curl'][state].items()}
    D = C.flex_D(curl); D.update(S['Dth'])
    return curl, D


def deg(v):
    return [float(x) for x in v]


def main():
    sc, cam = H.setup_scene()
    ao, ob, P, W, owner = build('w0')
    print('weights: verts per owner', {('palm' if k < 0 else CHAINS[k]): int((owner == k).sum()) for k in range(-1, 5)})
    # ---- 1. 静止(開いた手)と拳(リグの確認)。 メッシュ座標のまま撮る
    ao.matrix_world = Matrix.Identity(4)
    c0 = (0.0, 0.14, 0.0)
    H.set_pose(ao, {}); H.shoot(sc, cam, '1a_rest_back', c0, (0, 0.05, 1), 0.55)
    fist = C.flex_D({k: (85, 95, 65) for k in ('Index', 'Middle', 'Ring', 'Pinky')}); fist.update({'Thumb1': H.rotvec((0.0, 0.5, 0.0))})
    H.set_pose(ao, fist); H.shoot(sc, cam, '2b_fist_palm', c0, (0, 0.05, -1), 0.55)
    # ---- 2. 銃のメッシュから距離の格子を作り、仕様に沿って握りを探す
    gun = H.import_gun()
    sdf, bvh = blender_sdf(gun)
    rng = np.random.default_rng(3)
    S = C.solve(sdf, P, owner, W, rng, log=lambda s_: print('hand', s_))
    x, M, wps, info, cost = S['x'], S['M'], S['wps'], S['info'], S['cost']
    best = dict(Middle=S['curl']['fire']['Middle'], Ring=S['curl']['fire']['Ring'], Pinky=S['curl']['fire']['Pinky'], IndexFire=S['curl']['fire']['Index'], IndexReady=S['curl']['ready']['Index'])
    ao.matrix_world = Matrix(M)
    # ---- 3. 2状態: ready(構え) / fire(発砲中)。 他の指・親指・手のひらは同じ
    states = {}
    ctr = (0.0, 0.0, 0.0)
    views = (('right', (1, 0, 0.15)), ('left', (-1, 0, 0.15)), ('front', (0, 1, 0.1)), ('top', (0, 0, 1)), ('rear34', (0.7, -0.7, 0.35)))
    close = (('web_rear', (0.0, -0.025, 0.035), (0.15, -1, 0.35), 0.3), ('thumb_left', (-0.02, -0.03, 0.03), (-1, -0.5, 0.3), 0.3), ('thumb_leftfront', (-0.02, 0.0, 0.04), (-1, 0.5, 0.3), 0.3))
    for st in ('ready', 'fire'):
        curl, Dm = pose_dict(S, st)
        H.set_pose(ao, Dm)
        npen, _, Qn = C.mesh_pen(sdf, P, W, owner, Dm, M)
        print('PEN numpy-skin %s %s' % (st, {k: (v[0], round(v[1], 2)) for k, v in npen.items()}))
        ev_ = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()); evm = ev_.to_mesh(); cb = np.empty(len(evm.vertices) * 3, np.float32); evm.vertices.foreach_get('co', cb); cb = cb.reshape(-1, 3).astype(float)
        Qb = cb @ np.array(ob.matrix_world)[:3, :3].T + np.array(ob.matrix_world)[:3, 3]; ev_.to_mesh_clear()
        print('hand numpy-vs-blender skin max diff mm %.3f' % (np.abs(Qb - Qn).max() * 1000))
        rep = inside_report(ob, owner, bvh)
        for g, r in rep.items():
            print('PEN %s %-7s normal=%4d maxdepth=%.1fmm | ray2of3=%4d maxdist=%.1fmm' % (st, g, r[0], r[1] * 1000, r[2], r[3] * 1000))
        for nm, d in views:
            H.shoot(sc, cam, '3%s_%s' % (st, nm), ctr, d, 0.5)
        if st == 'fire':
            for nm, cc, d, dist in close:
                H.shoot(sc, cam, '3%s_%s' % (st, nm), cc, d, dist)
        gun.hide_render = True
        H.shoot(sc, cam, '4%s_hand_right' % st, ctr, (1, 0, 0.15), 0.5); H.shoot(sc, cam, '4%s_hand_left' % st, ctr, (-1, 0, 0.15), 0.5)
        gun.hide_render = False
        quats = {pb.name: [float(v) for v in pb.rotation_quaternion] for pb in ao.pose.bones}
        states[st] = dict(angles_deg={k: deg(v) for k, v in curl.items()}, bone_rotation_armature_space={k: [[float(e) for e in row] for row in v] for k, v in Dm.items()},
                          pose_bone_quaternion_wxyz=quats, pen={g: [int(r[0]), float(r[1] * 1000)] for g, r in rep.items()}, pen_numpy={k: [v[0], v[1]] for k, v in npen.items()})
    # ---- 4. 親指の付け根のしわ: 重みの比較(fire の姿勢で、右・後ろ・上から)
    curl, Dm = pose_dict(S, 'fire')
    for wv in ('w0', 'w3', 'w4'):
        Wv, ownv, _ = C.compute_weights(P, C.WEIGHT_VARIANTS[wv]); set_weights(ob, Wv); H.set_pose(ao, Dm)
        gun.hide_render = True
        for nm, d in (('left', (-1, -0.5, 0.3)), ('rear', (-0.3, -1, 0.4)), ('palm', (-0.6, -0.2, -0.7))):
            H.shoot(sc, cam, '5%s_thumb_%s' % (wv, nm), (-0.01, -0.04, 0.03), d, 0.3)
        gun.hide_render = False
        rep = inside_report(ob, ownv, bvh)
        print('PEN weights %s %s' % (wv, {g: (r[0], round(r[1] * 1000, 1), r[2]) for g, r in rep.items()}))
    # ---- 5. 仕様との照合に使う数値
    R3 = M[:3, :3]; axis = R3 @ np.array([0, 1.0, 0])
    web = info['web']; mid = (M @ np.r_[JOINTS['Middle'][0], 1.0])[:3]
    z_c = {c: float((M @ np.r_[JOINTS[c][0], 1.0])[2]) for c in ('Index', 'Middle', 'Ring', 'Pinky')}
    ptsI, padI, _ = C.finger_batch('Index', np.array([best['IndexFire']]), M); ptsR, _, _ = C.finger_batch('Index', np.array([best['IndexReady']]), M)
    gR = sdf(ptsR[0]) - C.RAD['Index']
    spec = dict(
        wrist_pitch_yaw_roll_deg=[float(x[4]), float(x[3]), float(x[5])], hand_axis_vs_barrel_deg=float(math.degrees(math.acos(float(axis @ np.array([0, 1.0, 0]))))),
        forearm_axis_height_above_bore_m=float(0.068 - (M @ np.r_[C.sh(C.HAND_HEAD), 1.0])[2]),
        web_pos=deg(web), web_gap_mm=float(info['web_gap'] * 1000), middle_top_z=float(mid[2] + C.RAD['Middle']), guard_base_z=C.GUARD_BASE_Z,
        finger_center_z=z_c, thumb_waypoints=[deg(p) for p in wps], thumb_tip_target=deg(C.THUMB_T3),
        index_fire_pad=deg(padI[0]), index_fire_pad_to_trigger_mm=float(np.linalg.norm(padI[0] - C.TRIG_FRONT) * 1000),
        index_ready_tip=deg(ptsR[0][-1]), index_ready_min_gap_mm=float(gR.min() * 1000), index_ready_mean_gap_mm=float(gR[:8].mean() * 1000))
    Dm = pose_dict(S, 'fire')[1]; _, gq, Qq = C.mesh_pen(sdf, P, W, owner, Dm, M)
    near_web = np.nonzero(np.linalg.norm(P - C.WEB[None, :], axis=1) < 0.02)[0]; j = near_web[np.argmin(gq[near_web])]
    spec['mesh_web_min_gap_mm'] = float(gq[j] * 1000); spec['mesh_web_contact_pos'] = deg(Qq[j])
    for k_, nm_ in ((2, 'Middle'), (3, 'Ring'), (4, 'Pinky'), (1, 'Index'), (0, 'Thumb')):
        m_ = owner == k_; spec['mesh_min_gap_mm_' + nm_] = float(gq[m_].min() * 1000); spec['mesh_zmax_' + nm_] = float(Qq[m_][:, 2].max()); spec['mesh_zmin_' + nm_] = float(Qq[m_][:, 2].min())
    print('hand SPEC', json.dumps(spec))
    out = dict(version=1, note='右手。 hand_to_gun = メッシュ(C0ずらし後)座標 -> 銃の座標(x右,y銃口,z上,原点=グリップ中心)の4x4。 bone_rotation_armature_space = tools/hand_rig_test.set_pose の D。 pose_bone_quaternion_wxyz は、hand_rig_meshy.build() で作ったアーマチュアの pose bone へそのまま入れられる。',
               source='tools/hand_rig_meshy.py + tools/hand_grip_core.py、 仕様 リサーチ/2026-11-19_拳銃の握り方_現実の挙動.md',
               hand_to_gun=[[float(e) for e in r] for r in M], hand_pose=dict(pos=deg(x[:3]), yaw_pitch_roll_deg=[float(x[3]), float(x[4]), float(x[5])]),
               weights_variant='w0', states=states, spec_check=spec)
    json.dump(out, open(os.path.join(OUT, 'hand_grip_poses.json'), 'w'), ensure_ascii=False, indent=1)
    set_weights(ob, W); H.set_pose(ao, pose_dict(S, 'fire')[1])
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, 'hand_rig_meshy.blend'))


main()
