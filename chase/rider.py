"""続編: Bのライダー(Meshyの自動リグ付き単体モデル)と銃。バイクのシートに座り、右手で銃を抜いて撃つ。
ボーンの動きは、Blender の IK を使わず、numpy で計算する(2関節の IK + 背骨の FK)。座標は「バイクのプロトタイプ空間」(x=右, y=前, z=上, 路面 z=0)。
  X_b = ボーン b の剛体の動き(プロトタイプ空間の 4x4)。 ポーズ行列 = X_b @ 静止行列。 基底(pose bone の loc/rot) = inv(rest_b) @ (inv(X_親) @ X_b) @ rest_b  (rig.py と同じ式)
"""
import math, os, sys
import numpy as np
import bpy
from mathutils import Matrix, Vector

import rig as RG
import sequel as SQ

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAIN = ['Hips', 'Spine02', 'Spine01', 'Spine', 'neck', 'Head']
ARMS = {'L': ('LeftArm', 'LeftForeArm', 'LeftHand'), 'R': ('RightArm', 'RightForeArm', 'RightHand')}
LEGS = {'L': ('LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'), 'R': ('RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase')}

# 調整値(確認描画で合わせる)。 バイクのプロトタイプ空間の m / rad
K = {
    'seat': (0.0, -0.20, 0.93),          # 腰(Hips の関節)の位置
    'pelvis_pitch': 0.55,                # 骨盤の前傾
    'tuck': 1.30,                        # 走行中の上体の前傾(鉛直から。腰から上の合計)
    'aim_lean': 0.52,                    # 銃を構えるときの前傾
    'grip': (0.31, 0.40, 1.02),         # 手の位置(ハンドルのグリップ。右、x は左右対称)
    'peg': (0.215, -0.10, 0.45),         # くるぶしの位置(ステップの上)
    'knee_pole': (0.45, 0.9, 0.15),      # 膝の向き(前・外)
    'elbow_pole': (0.55, -0.35, -0.55),  # 肘の向き(外・後ろ・下)
    'chest_off': (0.10, 0.15, -0.14),    # 銃を抜くとき、右手が入る位置(胸の関節から。上着の内側。胴体に付いて動く)
    'gun_grip_off': (0.0, 0.105, 0.012),  # 手首から銃のグリップ中心まで(前腕の向きに、上に)。小さすぎると銃が腕にめり込む
    'gun_scale': 1.0,
    'foot_down': (-0.46, 0.0),           # 横滑りのとき、左足を着く位置(バイクの左, 前)。ここは地面
}


def rot_swing(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b); s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        ax = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0]); ax /= np.linalg.norm(ax)
        return RG.rot3(ax, math.pi)
    return RG.rot3(v / s, math.atan2(s, c))


def tr(p):
    m = np.eye(4); m[:3, 3] = p; return m


def rotm(R):
    m = np.eye(4); m[:3, :3] = R; return m


def at(R3, pivot):
    return RG.about(R3, pivot)


def two_bone(sh, T, l1, l2, pole):
    """肩 sh, 手首の目標 T, 上腕 l1, 前腕 l2, 肘の向き pole -> 肘の位置(目標が遠すぎるときは腕を伸ばした位置にそろえる)"""
    d = T - sh; dist = np.linalg.norm(d)
    dist_c = min(max(dist, abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4)
    u = d / max(dist, 1e-9)
    a = (l1 * l1 - l2 * l2 + dist_c * dist_c) / (2 * dist_c)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    pv = np.asarray(pole, float); pv = pv - (pv @ u) * u
    if np.linalg.norm(pv) < 1e-6:
        pv = np.cross(u, [0, 0, 1.0])
    pv /= np.linalg.norm(pv)
    return sh + u * a + pv * h, sh + u * dist_c


def make_opaque(mat):
    """Meshy の材質は、テクスチャのアルファで半透明になることがある(ライダーが透ける)。 アルファへのつなぎを切って 1 にする"""
    nt = mat.node_tree
    for n in nt.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            for l in list(n.inputs['Alpha'].links):
                nt.links.remove(l)
            n.inputs['Alpha'].default_value = 1.0
    for attr, val in (('surface_render_method', 'DITHERED'), ('blend_method', 'OPAQUE')):
        try:
            setattr(mat, attr, val)
        except Exception:
            pass


def build(R, root, proto, tag='BikeB'):
    """リグ付きライダーを読み込み、バイクの root の下に置く。 R = render モジュール"""
    d = os.path.join(ROOT, 'assets', 'bike_B2')
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.join(d, 'rider_rigged.glb'), import_shading='SMOOTH')
    new = [o for o in bpy.data.objects if o not in before]
    ao = next(o for o in new if o.type == 'ARMATURE'); body = next(o for o in new if o.type == 'MESH')
    bpy.context.view_layer.update()
    for o in new:                                            # 読み込みで作られた線の置き物などは捨てる
        if o.type not in ('ARMATURE', 'MESH'):
            bpy.data.objects.remove(o)
    for m in body.data.materials:
        make_opaque(m)
        R.vehicle_material(m, 'wheel', False)
    try:
        body.cycles.use_deform_motion = bool(R.P.get('rider_deform_blur', False))      # 既定は切る。スキンメッシュの変形ブラーを入れると、ライダーが半透明にぶれる(2026-11-19 切り分けで確認。ブラーを切ると不透明)
    except Exception as e:
        print('use_deform_motion:', e)
    # 静止姿勢のボーン位置(アーマチュア空間)と静止行列
    arm = ao.data
    bones = {b.name: b for b in arm.bones}
    A0 = ao.matrix_world.copy(); A0.translation = (0, 0, 0)
    rest_arm = {n: np.array(b.head_local) for n, b in bones.items()}
    rest_mat = {n: np.array(b.matrix_local) for n, b in bones.items()}
    hips_w = A0 @ Vector(rest_arm['Hips'])
    seat = Vector(K['seat'])
    Mw = Matrix.Translation(seat - Matrix.Rotation(math.pi, 4, 'Z') @ hips_w) @ Matrix.Rotation(math.pi, 4, 'Z') @ A0       # 腕・脚はそのまま。180°回して前向きにする
    ao.parent = root; ao.matrix_parent_inverse = Matrix.Identity(4); ao.matrix_basis = Mw
    for o in new:
        o.hide_viewport = False
    M = np.array(Mw)                                         # アーマチュア空間 -> プロトタイプ空間
    Mi = np.linalg.inv(M)
    rest_p = {n: (M @ np.r_[p, 1.0])[:3] for n, p in rest_arm.items()}          # プロトタイプ空間の静止位置
    for pb in ao.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    # 銃
    gun = import_gun(R, d)
    flash = make_flash(R)
    rd = {'ao': ao, 'body': body, 'M': M, 'Mi': Mi, 'rest_p': rest_p, 'rest_mat': rest_mat, 'bones': bones, 'gun': gun, 'flash': flash, 'root': root,
          'cfg': proto['rig'], 'len': {}, 'parent': {n: (b.parent.name if b.parent else None) for n, b in bones.items()}}
    L = {}
    for s in 'LR':
        a, f, h = ARMS[s]; L['arm' + s] = (np.linalg.norm(rest_p[f] - rest_p[a]), np.linalg.norm(rest_p[h] - rest_p[f]))
        u, k, an, to = LEGS[s]; L['leg' + s] = (np.linalg.norm(rest_p[k] - rest_p[u]), np.linalg.norm(rest_p[an] - rest_p[k]))
    rd['len'] = L
    print('   rider: bones %d, verts %d, arm %s leg %s, hips(proto) %s' % (len(bones), len(body.data.vertices), np.round(L['armL'], 3), np.round(L['legL'], 3), np.round(rest_p['Hips'], 3)))
    return rd


def import_gun(R, d):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.join(d, 'gun.glb'), import_shading='SMOOTH')
    new = [o for o in bpy.data.objects if o not in before]
    bpy.context.view_layer.update()
    ms = [o for o in new if o.type == 'MESH']
    o = ms[0]
    mw = o.matrix_world.copy(); o.parent = None; o.data.transform(mw); o.matrix_world = Matrix.Identity(4)
    for x in new:
        if x is not o:
            bpy.data.objects.remove(x)
    me = o.data
    # glTF(長軸 x, 上 y, 銃口 -x, グリップは +x 側) を、原点=グリップ中心、銃口 +Y、上 +Z にそろえる
    ox = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', ox); ox = ox.reshape(-1, 3)
    # import は (x, -z, y)。 グリップ中心(glTF x=0.065, y=0.045) を原点へ
    ox[:, 0] -= 0.065; ox[:, 2] -= 0.045
    # 銃口は -X。 Z 回り -90° で +Y へ
    c, s = math.cos(-math.pi / 2), math.sin(-math.pi / 2)
    nx = c * ox[:, 0] - s * ox[:, 1]; ny = s * ox[:, 0] + c * ox[:, 1]
    out = np.c_[nx, ny, ox[:, 2]].astype(np.float32) * K['gun_scale']
    me.vertices.foreach_set('co', out.ravel()); me.update()
    for m in me.materials:
        make_opaque(m)
        R.vehicle_material(m, 'wheel', False)
    R.smooth_mesh(me, 40.0, 2e-5)
    o.rotation_mode = 'QUATERNION'
    print('   gun: verts %d size %s' % (len(me.vertices), np.round(np.ptp(out, axis=0), 3)))
    return o


def make_flash(R):
    """マズルフラッシュ(銃口から前へ伸びる光る星 + 点光源)。銃口の向き = +Y"""
    mat = R.emission_material('MuzzleFlash', (1.0, 0.72, 0.30), 110.0)
    V, F = [], []
    def tri(a, b, c): n = len(V); V.extend([a, b, c]); F.append((n, n + 1, n + 2))
    seg = 8
    for i in range(seg):                                   # 前へ伸びる円錐
        a0 = 2 * math.pi * i / seg; a1 = 2 * math.pi * (i + 1) / seg
        tri((0.022 * math.cos(a0), 0, 0.022 * math.sin(a0)), (0.022 * math.cos(a1), 0, 0.022 * math.sin(a1)), (0, 0.15, 0))
    for k in range(8):                                     # 横へ出るとげ
        a = 2 * math.pi * k / 8
        tip = (0.085 * math.cos(a), 0.02, 0.085 * math.sin(a)); w = 0.016
        tri((-w * math.sin(a), 0.0, w * math.cos(a)), (w * math.sin(a), 0.0, -w * math.cos(a)), tip)
    for k in range(4):                                     # 斜め前へのとげ
        a = 2 * math.pi * k / 4 + 0.4
        tri((0.02, 0.0, 0.0), (-0.02, 0.0, 0.0), (0.05 * math.cos(a), 0.09, 0.05 * math.sin(a)))
    me = bpy.data.meshes.new('flash'); me.from_pydata(V, [], F); me.materials.append(mat); me.update()
    o = R.link(bpy.data.objects.new('flash', me))
    ld = bpy.data.lights.new('flashL', 'POINT'); ld.energy = 260.0; ld.color = (1.0, 0.75, 0.4); ld.shadow_soft_size = 0.05
    lo = bpy.data.objects.new('flashL', ld); R.link(lo, o); lo.location = (0, 0.12, 0)
    o.rotation_mode = 'QUATERNION'; o.scale = (0.001, 0.001, 0.001)
    try:
        o.cycles.use_motion_blur = False                    # 一瞬の光なので、にじませない(にじむと長い筋になる)
    except Exception:
        pass
    return {'obj': o, 'light': lo, 'mat': mat}


def pose(rd, pz, root_M, A_world, tau):
    """-> (bone_basis: {name: (loc, quat)}, gun_M(4x4 プロトタイプ空間), gun_vis, flash_on, muzzle_M)
    pz = sequel.bike_pose('B', t) の結果、root_M = バイクの root のワールド行列(numpy 4x4)、A_world = Aの胴体のワールド位置(Blender座標)"""
    cfg = rd['cfg']; rp = rd['rest_p']; Lr = rd['len']
    gs = SQ.gun_state(tau)
    ld = RG.local_deltas(cfg, pz)
    Xfr = ld['frame']                                                    # 車体のばね上(上下・ピッチ)。腰は座席に乗っている
    Xst = RG.total_deltas(cfg, pz)['steer_s']                            # ハンドル(ばね上 + ステアリング)
    slide = pz.get('slide', 0.0)
    # --- 背骨 ---
    aim_k = gs['aim']; chest_k = gs['chest']
    lean = K['tuck'] * (1 - aim_k) + K['aim_lean'] * aim_k - 0.12 * chest_k - 0.30 * min(1.0, slide / 1.2)
    lean = lean + pz['torso_pitch'] * 3.0
    pel = K['pelvis_pitch']
    rel = lean - pel
    # 銃を向ける方向(Aの胴体)を、バイクのプロトタイプ空間へ
    Ai = np.linalg.inv(root_M)
    a_loc = (Ai @ np.r_[np.asarray(A_world, float), 1.0])[:3]
    sh_est = np.array([0.18, 0.10, 1.12])
    to_a = a_loc - sh_est
    yaw_aim = math.atan2(-to_a[0], to_a[1])                              # + = 左(Z 回り)
    yaw_aim = max(-1.2, min(1.2, yaw_aim))
    sp_yaw = 0.35 * yaw_aim * aim_k
    Xh = Xfr @ at(RG.rot3((1, 0, 0), -pel) @ RG.rot3((0, 1, 0), pz['torso_roll']), rp['Hips'])
    X = {'Hips': Xh}
    shares = {'Spine02': 0.40, 'Spine01': 0.35, 'Spine': 0.25}
    prev = 'Hips'
    for n in ('Spine02', 'Spine01', 'Spine'):
        Rn = RG.rot3((1, 0, 0), -rel * shares[n]) @ RG.rot3((0, 0, 1), sp_yaw * shares[n])
        X[n] = X[prev] @ at(Rn, rp[n]); prev = n
    nk = 0.20
    X['neck'] = X['Spine'] @ at(RG.rot3((1, 0, 0), +0.30 * lean), rp['neck'])
    hy = pz['head_yaw'] + 0.65 * yaw_aim * aim_k
    X['Head'] = X['neck'] @ at(RG.rot3((0, 0, 1), hy) @ RG.rot3((0, 1, 0), pz['head_roll']) @ RG.rot3((1, 0, 0), +0.30 * lean), rp['Head'])
    # --- 腕 ---
    out_pos = {}
    # 右手の目標: grip → chest → aim
    sh_R = (X['Spine'] @ np.r_[rp['RightArm'], 1.0])[:3]
    sh_L = (X['Spine'] @ np.r_[rp['LeftArm'], 1.0])[:3]
    grip_R = (Xst @ np.r_[np.array(K['grip']), 1.0])[:3]
    grip_L = (Xst @ np.r_[np.array([-K['grip'][0], K['grip'][1], K['grip'][2]]), 1.0])[:3]
    chest_h = (X['Spine'] @ np.r_[rp['Spine'] + np.array(K['chest_off']), 1.0])[:3]
    reach_R = float(sum(Lr['armR'])) * 0.985
    aimdir = to_a / np.linalg.norm(to_a)
    aim_t = sh_R + aimdir * reach_R * 0.985 + np.array([0.0, 0.0, 0.0])
    gw = np.array([gs['grip'], gs['chest'], gs['aim']]); gw = gw / max(gw.sum(), 1e-6)
    TR = gw[0] * grip_R + gw[1] * chest_h + gw[2] * aim_t
    rec = gs['recoil']
    TR = TR - aimdir * 0.06 * rec * aim_k + np.array([0, 0, 0.025 * rec])
    targets_arm = {'R': TR, 'L': grip_L}
    shoulder = {'R': sh_R, 'L': sh_L}
    el_pole = {'R': np.array([K['elbow_pole'][0], K['elbow_pole'][1], K['elbow_pole'][2]]), 'L': np.array([-K['elbow_pole'][0], K['elbow_pole'][1], K['elbow_pole'][2]])}
    forearm_dir = {}
    for s in 'RL':
        a, f, h = ARMS[s]
        l1, l2 = Lr['arm' + s]
        p_el, p_wr = two_bone(shoulder[s], targets_arm[s], l1, l2, el_pole[s])
        d_u0 = rp[f] - rp[a]; d_f0 = rp[h] - rp[f]
        X[a] = tr(shoulder[s]) @ rotm(rot_swing(d_u0, p_el - shoulder[s])) @ tr(-rp[a])
        Rf = rot_swing(d_f0, p_wr - p_el)
        X[f] = tr(p_el) @ rotm(Rf) @ tr(-rp[f])
        X[h] = tr(p_wr) @ rotm(Rf) @ tr(-rp[h])
        forearm_dir[s] = (p_wr - p_el) / np.linalg.norm(p_wr - p_el)
        out_pos['wr' + s] = p_wr
    # --- 脚 ---
    foot_down = min(1.0, max(0.0, (slide - 0.25) / 0.8))
    for s in 'LR':
        u, k, an, to = LEGS[s]
        sg = 1.0 if s == 'R' else -1.0
        peg = np.array([sg * K['peg'][0], K['peg'][1], K['peg'][2]])
        T_an = (Xfr @ np.r_[peg, 1.0])[:3]
        if s == 'L' and foot_down > 0:
            gx, gy = K['foot_down']
            wp = (root_M @ np.r_[gx, gy, 0.0, 1.0])[:3]; wp[2] = 0.11                  # 路面(くるぶしの高さ)
            T_an = (1 - foot_down) * T_an + foot_down * (Ai @ np.r_[wp, 1.0])[:3]
        hip_j = (X['Hips'] @ np.r_[rp[u], 1.0])[:3]
        l1, l2 = Lr['leg' + s]
        p_kn, p_an = two_bone(hip_j, T_an, l1, l2, np.array([sg * K['knee_pole'][0], K['knee_pole'][1], K['knee_pole'][2]]))
        X[u] = tr(hip_j) @ rotm(rot_swing(rp[k] - rp[u], p_kn - hip_j)) @ tr(-rp[u])
        X[k] = tr(p_kn) @ rotm(rot_swing(rp[an] - rp[k], p_an - p_kn)) @ tr(-rp[k])
        toe_dir = np.array([sg * 0.05, 1.0, -0.25])
        Rfoot = rot_swing(rp[to] - rp[an], toe_dir)
        X[an] = tr(p_an) @ rotm(Rfoot) @ tr(-rp[an])
        X[to] = X[an]
    # --- ボーンの基底へ ---
    parent = rd['parent']
    basis = {}
    M, Mi = rd['M'], rd['Mi']
    for n in X:
        ld_ = np.linalg.inv(parent_X(X, parent, n)) @ X[n]
        ldA = Mi @ ld_ @ M
        Rm = rd['rest_mat'][n]
        Bm = Matrix(np.linalg.inv(Rm) @ ldA @ Rm)
        loc, rot, _ = Bm.decompose()
        basis[n] = (loc, rot)
    # --- 銃 ---
    fR = forearm_dir['R']; wrR = out_pos['wrR']
    up = np.array([0.0, 0.0, 1.0]); zz = up - (up @ fR) * fR
    if np.linalg.norm(zz) < 1e-4:
        zz = np.array([0.0, -1.0, 0.0])
    zz /= np.linalg.norm(zz); xx = np.cross(fR, zz)
    Rg = np.stack([xx, fR, zz], axis=1)
    rk = RG.rot3(xx, 0.20 * rec)
    Rg = rk @ Rg
    gun_M = np.eye(4); gun_M[:3, :3] = Rg; gun_M[:3, 3] = wrR + fR * K['gun_grip_off'][1] + Rg @ np.array([0.0, 0.0, K['gun_grip_off'][2]])
    muz = np.eye(4); muz[:3, :3] = Rg; muz[:3, 3] = gun_M[:3, 3] + Rg @ np.array([0.0, 0.175, 0.07])
    flash_on = any(abs(tau - ft) < 0.5 / SQ.FPS for ft in SQ.FIRES)
    return basis, gun_M, gs['vis'], flash_on, muz


def parent_X(X, parent, n):
    """n の親の X。親が制御していないボーン(肩など)は、その親の X をたどる。ルート(Hips の親)は単位行列"""
    p = parent[n]
    while p is not None and p not in X:
        p = parent.get(p)
    return X[p] if p is not None else np.eye(4)


def _cont(rd, name, j, q):
    """クォータニオンの符号をそろえる(Blender のキー補間は成分ごと。符号が反転すると、間で大きく回って残像になる)"""
    pq = rd.setdefault('_pq', {})
    if j == 0:
        pq.clear()
    last = pq.get(name)
    if last is not None and q.dot(last) < 0:
        q = -q
    pq[name] = q.copy()
    return q


def key(rd, j, pz, root_M, A_world, tau):
    """ボーンと銃のキーを打つ(フレーム j)"""
    basis, gun_M, vis, flash_on, muz = pose(rd, pz, root_M, A_world, tau)
    ao = rd['ao']
    for n, (loc, rot) in basis.items():
        pb = ao.pose.bones[n]
        pb.location = loc; pb.rotation_quaternion = _cont(rd, n, j, rot)
        pb.keyframe_insert('location', frame=j); pb.keyframe_insert('rotation_quaternion', frame=j)
    # 銃は root の子ではなく、ワールドで直接キーする(root_M @ gun_M)
    gw = root_M @ gun_M
    g = rd['gun']; mm = Matrix(gw)
    g.location = mm.to_translation(); g.rotation_quaternion = _cont(rd, 'gun', j, mm.to_quaternion())
    g.keyframe_insert('location', frame=j); g.keyframe_insert('rotation_quaternion', frame=j)
    g.hide_render = vis < 0.5; g.keyframe_insert('hide_render', frame=j)
    fl = rd['flash']['obj']; mw = Matrix(root_M @ muz)
    fl.location = mw.to_translation(); fl.rotation_quaternion = _cont(rd, 'flash', j, mw.to_quaternion())
    fl.keyframe_insert('location', frame=j); fl.keyframe_insert('rotation_quaternion', frame=j)
    # 表示/非表示のキーを打つと、モーションブラーで途中の位置に長い筋が出る → 常に表示し、大きさ(と光の強さ)で出し入れする
    k = 1.0 if flash_on else 0.001
    fl.scale = (k, k, k); fl.keyframe_insert('scale', frame=j)
    ld = rd['flash']['light'].data; ld.energy = 260.0 * (1.0 if flash_on else 0.0); ld.keyframe_insert('energy', frame=j)
