"""バイク+ライダー(1枚のメッシュ)の疑似リグ。bpy を使わない(numpy だけ)ので、手元でも重みと姿勢を確かめられる。
座標はプロトタイプ空間(build_proto の出力): x=右, y=前, z=上(m)、車輪の接地面 z=0。

ボーン(親 → 子):
  static                      動かない(車輪と同じ。路面に接している側)
  frame                       車体のばね上(上下 + ピッチ)。車輪は動かさない = タイヤは路面から離れない
    steer_s                   ハンドル・フォークの上側(ばね上)。ステアリング軸まわりに回る
    torso                     ライダーの胴体(腰が支点。車体より遅れて傾く)
      head                    ヘルメット(首が支点。水平を保ち、進行方向を見る)
  steer_u                     フォークの下側(車輪と同じ。ステアリング軸まわりに回る)
腕は、肩(torso)から手(steer_s)へ重みを滑らかに移す。 → ハンドルを切ると手が付いていき、腕が曲がる。
"""
import math
import numpy as np

BONES = ['static', 'frame', 'steer_u', 'steer_s', 'torso', 'head']
PARENT = {'static': None, 'frame': 'static', 'steer_u': 'static', 'steer_s': 'frame', 'torso': 'frame', 'head': 'torso'}

RIG = {
    'bike_A': dict(bike_col='red', hip=(0.0, -0.28, 0.82), neck=(0.0, 0.28, 1.09), head_c=(0.0, 0.335, 1.22), head_r=(0.17, 0.21, 0.19),
                   steer_head=(0.0, 0.50, 0.88), axle=(0.0, 0.777, 0.30), shoulder_y=0.10, grip_y=0.52, frame_pivot=(0.0, 0.08, 0.55),
                   wheel_f=(0.777, 0.30, 0.30), wheel_r=(-0.62, 0.31, 0.31)),
    'bike_B': dict(bike_col='teal', hip=(0.0, -0.25, 0.86), neck=(0.0, 0.27, 1.15), head_c=(0.0, 0.325, 1.29), head_r=(0.17, 0.20, 0.19),
                   steer_head=(0.0, 0.52, 0.92), axle=(0.0, 0.78, 0.30), shoulder_y=0.10, grip_y=0.52, frame_pivot=(0.0, 0.08, 0.55),
                   wheel_f=(0.78, 0.30, 0.30), wheel_r=(-0.60, 0.305, 0.305)),
    # 続編のB: ライダー無しのバイク単体(ライダーは別のアーマチュア)。位置の値は bike_B と同じ(同じ画像から作ったので重なる)
    'bike_B2': dict(bike_col='teal', rider=False, hip=(0.0, -0.25, 0.86), neck=(0.0, 0.27, 1.15), head_c=(0.0, 0.325, 1.29), head_r=(0.17, 0.20, 0.19),
                    steer_head=(0.0, 0.52, 0.92), axle=(0.0, 0.78, 0.30), shoulder_y=0.10, grip_y=0.52, frame_pivot=(0.0, 0.08, 0.55),
                    wheel_f=(0.78, 0.30, 0.30), wheel_r=(-0.60, 0.305, 0.305)),
}


def sm(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def hue_sat(C):
    """RGB(0..255) -> 色相(度, 0..360), 彩度(0..1)"""
    c = C.astype(np.float32) / 255.0
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    mx = c.max(1); mn = c.min(1); d = mx - mn
    sat = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    dd = np.maximum(d, 1e-6)
    h = np.where(mx == r, ((g - b) / dd) % 6.0, np.where(mx == g, (b - r) / dd + 2.0, (r - g) / dd + 4.0)) * 60.0
    return h % 360.0, sat


def body_colour_likelihood(C, kind):
    """テクスチャ色から、「バイクの車体色(ライダーではない)」らしさ 0..1。
    A: 車体=赤(色相 0°付近)、ヘルメット=オレンジ(20〜30°)。  B: 車体=青緑(色相 165〜205°)、ライダー=濃い青(210〜240°)"""
    h, sat = hue_sat(C)
    if kind == 'red':
        dh = np.minimum(h, 360.0 - h)
        return sm((sat - 0.4) / 0.2) * (1 - sm((dh - 8.0) / 10.0))
    return sm((sat - 0.35) / 0.2) * (1 - sm((np.abs(h - 185.0) - 20.0) / 10.0))


def compute_weights(P, C, cfg):
    """-> (n, 6) float32。行の合計 = 1"""
    n = len(P)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    isb = body_colour_likelihood(C, cfg['bike_col'])
    H = np.array(cfg['steer_head'], np.float32); A = np.array(cfg['axle'], np.float32)
    L = float(np.linalg.norm(A - H)); u = (A - H) / L
    d = P - H; t = d @ u; rad = np.linalg.norm(d - t[:, None] * u[None, :], axis=1)
    # ステアリング部品: フォーク(軸から半径0.11m、ヘッドから車軸の少し先まで) + ハンドル(ヘッド付近の横長の箱)
    wf = (1 - sm((rad - 0.115) / 0.05)) * sm((t + 0.06) / 0.06) * (1 - sm((t - (L + 0.12)) / 0.10))
    yh, zh = H[1], H[2]
    wb = (1 - sm((np.abs(x) - 0.36) / 0.05)) * sm((y - (yh - 0.12)) / 0.04) * (1 - sm((y - (yh + 0.14)) / 0.04)) \
        * sm((z - (zh - 0.10)) / 0.04) * (1 - sm((z - (zh + 0.14)) / 0.04))
    wst = np.maximum(wf, wb) * (1 - 0.9 * isb)
    lam = t / L
    sprung = np.where(t < 0, 1.0, 1 - sm((lam - 0.30) / 0.45))      # ヘッド側ほど「ばね上」、車軸側ほど「ばね下」
    # ライダー: 腰より上(腰=支点、胸で重み1)、前のカウル・後ろのテールは除く、車体色は除く
    hip = np.array(cfg['hip'], np.float32)
    rz = sm((z - (hip[2] - 0.02)) / 0.16)
    ry = sm((y + 0.55) / 0.10) * (1 - sm((y - 0.50) / 0.10))
    rider = rz * ry * (1 - 0.85 * isb) * (1 - wst)
    if cfg.get('rider') is False:
        rider = rider * 0.0
    # 頭: ヘルメットの楕円体(外側へ滑らかに)
    hc = np.array(cfg['head_c'], np.float32); hr = np.array(cfg['head_r'], np.float32)
    e = np.linalg.norm((P - hc) / hr, axis=1)
    wh_raw = 1 - sm((e - 1.0) / 0.45)
    w_head = np.minimum(1.0, wh_raw * (1 - wst) * (1 - 0.9 * isb)) * (0.0 if cfg.get('rider') is False else 1.0)
    # 腕: 体の横(|x|>0.1)、肩より前 → 肩(torso)から手(steer_s)へ
    ys, yg = cfg['shoulder_y'], cfg['grip_y']
    arm = sm((np.abs(x) - 0.10) / 0.07) * sm((y - (ys - 0.05)) / 0.10) * sm((z - 0.82) / 0.06)
    s_arm = sm((y - ys) / (yg - ys))
    rider_rest = np.maximum(rider - w_head, 0.0)
    w_arm = rider_rest * arm * s_arm
    w_torso = rider_rest - w_arm
    w_su = wst * (1 - sprung)
    w_ss = wst * sprung + w_arm
    used = w_su + w_ss + w_torso + w_head
    rem = np.maximum(0.0, 1.0 - used)
    wz = sm((z - 0.42) / 0.30)
    W = np.stack([rem * (1 - wz), rem * wz, w_su, w_ss, w_torso, w_head], axis=1).astype(np.float32)
    W /= np.maximum(W.sum(1, keepdims=True), 1e-6)
    # タイヤの上のフェンダー(泥よけ)は車輪と一緒に動く(ばね下)。車体の上下に付いていくと、タイヤとの間隔がビヨンビヨン変わって見える。
    #   前: フォークに付く = ステアリングだけ(steer_u)。 後ろ: 動かない(static)。
    zf = wheel_zone(P, cfg['wheel_f']); zr = wheel_zone(P, cfg['wheel_r'])
    W *= (1 - zf)[:, None]; W[:, 2] += zf
    W *= (1 - zr)[:, None]; W[:, 0] += zr
    return W / np.maximum(W.sum(1, keepdims=True), 1e-6)


def wheel_zone(P, wheel):
    """車輪の中心(y, z, 半径 r)のまわり、車軸より上の、タイヤから約10cmまでの帯 = フェンダー。1 = 完全に車輪側"""
    yc, zc, r = wheel
    dy = P[:, 1] - yc; dz = P[:, 2] - zc
    d = np.hypot(dy, dz)
    band = 1 - sm((d - (r + 0.10)) / 0.05)
    above = sm((dz + 0.04) / 0.06)
    along = 1 - sm((np.abs(dy) - (r + 0.05)) / 0.06)
    return (band * above * along).astype(np.float32)


# ---- 姿勢(各ボーンの「剛体の動き」: プロトタイプ空間の 4x4) ---------------------------------------------
def T(v):
    m = np.eye(4); m[:3, 3] = v; return m


def rot3(axis, ang):
    a = np.asarray(axis, np.float64); a = a / np.linalg.norm(a)
    c, s = math.cos(ang), math.sin(ang); C = 1 - c
    x, y, z = a
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def about(R3, pivot):
    m = np.eye(4); m[:3, :3] = R3
    pv = np.asarray(pivot, np.float64)
    m[:3, 3] = pv - R3 @ pv
    return m


def local_deltas(cfg, pose):
    """各ボーンの(親から見た)動き。pose = motion.bike_pose の結果"""
    H = np.array(cfg['steer_head'], np.float64); A = np.array(cfg['axle'], np.float64)
    up = -(A - H)                                       # 車軸 → ヘッド(上向き)。左に切る = この軸まわりに +
    Ds = about(rot3(up, pose['steer']), H)
    Dfr = T((0, 0, pose['body_dz'])) @ about(rot3((1, 0, 0), pose['body_pitch']), cfg['frame_pivot'])
    Dt = T((0, 0, pose['torso_dz'])) @ about(rot3((0, 1, 0), pose['torso_roll']) @ rot3((1, 0, 0), pose['torso_pitch']), cfg['hip'])
    Dh = about(rot3((0, 0, 1), pose['head_yaw']) @ rot3((0, 1, 0), pose['head_roll']), cfg['neck'])
    return {'static': np.eye(4), 'frame': Dfr, 'steer_u': Ds, 'steer_s': Ds, 'torso': Dt, 'head': Dh}


def total_deltas(cfg, pose):
    ld = local_deltas(cfg, pose); tot = {}
    for b in BONES:
        p = PARENT[b]
        tot[b] = ld[b] if p is None else tot[p] @ ld[b]
    return tot


def skin(P, W, cfg, pose):
    """線形ブレンドスキニング(Blender の Armature と同じ)"""
    tot = total_deltas(cfg, pose)
    Ph = np.c_[P, np.ones(len(P))]
    out = np.zeros((len(P), 3))
    for i, b in enumerate(BONES):
        out += W[:, i:i + 1] * (Ph @ tot[b].T)[:, :3]
    return out
