"""動く拳銃キット(tools/gun_kit.py)の確認描画と、.blend / .glb の書き出し。 チェイス本編とは無関係。
  blender --background --python tools/gun_preview.py -- OUT [params.json]
params.json: {"samples": 32, "w": 480, "h": 360, "stretch": 3.5}
出力 OUT/*.png (一覧画像は 1920 幅前後)、 OUT/gun_kit.blend、 OUT/gun_kit.glb、 OUT/parts_report.json
"""
import bpy, os, sys, json, math, zlib, struct
import numpy as np
from mathutils import Vector, Matrix

ARGV = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else ['out']
OUT = ARGV[0]; os.makedirs(OUT, exist_ok=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import gun_kit as GK
PRM = json.load(open(ARGV[1])) if len(ARGV) > 1 else {}
SAMPLES = int(PRM.get('samples', 32)); TW = int(PRM.get('w', 480)); TH = int(PRM.get('h', 360)); STRETCH = float(PRM.get('stretch', 3.5))
GLB = os.path.join(ROOT, 'assets', 'bike_B2', 'gun.glb')
TMP = os.path.join(OUT, '_tiles'); os.makedirs(TMP, exist_ok=True)


def log(*a):
    print('hand gun:', *a, flush=True)


# ------------------------------------------------------------------ PNG の読み書き(Blender に PIL が無いので)
def png_read(path):
    b = open(path, 'rb').read(); pos = 8; idat = b''; w = h = ct = bd = 0
    while pos < len(b):
        ln, tp = struct.unpack('>I4s', b[pos:pos + 8]); d = b[pos + 8:pos + 8 + ln]; pos += 12 + ln
        if tp == b'IHDR':
            w, h, bd, ct = struct.unpack('>IIBB', d[:10])
        elif tp == b'IDAT':
            idat += d
    ch = {0: 1, 2: 3, 4: 2, 6: 4}[ct]; bpp = ch * (bd // 8); raw = zlib.decompress(idat); stride = w * bpp
    out = np.zeros((h, stride), np.uint8); prev = np.zeros(stride, np.int32); p = 0
    for y in range(h):
        f = raw[p]; line = np.frombuffer(raw, np.uint8, stride, p + 1).astype(np.int32); p += stride + 1
        if f == 0:
            cur = line
        elif f == 1:
            cur = line.copy()
            for i in range(bpp, stride):
                cur[i] = (cur[i] + cur[i - bpp]) & 255
        elif f == 2:
            cur = (line + prev) & 255
        elif f == 3:
            cur = line.copy()
            for i in range(stride):
                a = cur[i - bpp] if i >= bpp else 0
                cur[i] = (cur[i] + ((a + prev[i]) >> 1)) & 255
        else:
            cur = line.copy()
            for i in range(stride):
                a = cur[i - bpp] if i >= bpp else 0; c = prev[i - bpp] if i >= bpp else 0; bq = prev[i]
                pa = abs(bq - c); pb = abs(a - c); pc = abs(a + bq - 2 * c)
                pr = a if (pa <= pb and pa <= pc) else (bq if pb <= pc else c)
                cur[i] = (cur[i] + pr) & 255
        out[y] = cur; prev = cur
    img = out.reshape(h, w, ch)
    return img[:, :, :3] if ch >= 3 else np.repeat(img[:, :, :1], 3, 2)


def png_write(path, img):
    h, w, _ = img.shape
    raw = b''.join(b'\x00' + img[y].tobytes() for y in range(h))

    def chunk(t, d):
        c = struct.pack('>I', len(d)) + t + d
        return c + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    open(path, 'wb').write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def montage(files, cols, name):
    ims = [png_read(f) for f in files]; h, w, _ = ims[0].shape; rows = (len(ims) + cols - 1) // cols
    M = np.full((rows * h, cols * w, 3), 40, np.uint8)
    for i, im in enumerate(ims):
        r, c = divmod(i, cols); M[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    png_write(os.path.join(OUT, name + '.png'), M); log('montage', name, M.shape)


# ------------------------------------------------------------------ シーン
sc = None; cam = None; tilec = 0


def setup():
    global sc, cam
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene; sc.render.engine = 'CYCLES'; sc.cycles.samples = SAMPLES; sc.cycles.device = 'CPU'
    try:
        sc.cycles.use_denoising = True
    except Exception:
        pass
    sc.render.resolution_x = TW; sc.render.resolution_y = TH; sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = 'Standard'
    w = bpy.data.worlds.new('w'); w.use_nodes = True; w.node_tree.nodes['Background'].inputs[0].default_value = (0.62, 0.64, 0.68, 1); w.node_tree.nodes['Background'].inputs[1].default_value = 0.35; sc.world = w
    st = sc.render
    st.use_stamp = True; st.use_stamp_note = True; st.use_stamp_labels = False; st.stamp_font_size = max(12, TW // 32)
    for a in ('use_stamp_date', 'use_stamp_time', 'use_stamp_render_time', 'use_stamp_frame', 'use_stamp_frame_range', 'use_stamp_memory', 'use_stamp_hostname',
              'use_stamp_camera', 'use_stamp_lens', 'use_stamp_scene', 'use_stamp_marker', 'use_stamp_filename', 'use_stamp_sequencer_strip'):
        try:
            setattr(st, a, False)
        except Exception:
            pass
    st.stamp_foreground = (1, 1, 0.2, 1); st.stamp_background = (0, 0, 0, 0.55)


def rig():
    global cam
    for loc, e in (((-0.5, -0.9, 0.9), 55), ((0.9, -0.5, 0.7), 45), ((0.2, 0.9, 0.6), 25), ((0.0, 0.0, 1.2), 14)):
        ld = bpy.data.lights.new('l', 'AREA'); ld.energy = e; ld.size = 0.9; lo = bpy.data.objects.new('l', ld); lo.location = loc; sc.collection.objects.link(lo)
        lo.rotation_euler = (Vector((0, 0.05, 0.03)) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'


def look(loc, tgt, scale):
    cam.location = loc; cam.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler(); cam.data.ortho_scale = scale
    cam.data.clip_start = 0.01; cam.data.clip_end = 20


def dirloc(tgt, d, dist=1.2):
    d = Vector(d).normalized(); return Vector(tgt) + d * dist


def shoot(tag, note=''):
    global tilec
    sc.render.stamp_note_text = note
    f = os.path.join(TMP, '%s_%02d.png' % (tag, tilec)); tilec += 1
    sc.render.filepath = f; bpy.ops.render.render(write_still=True); return f


def set_casing(ob, c, t, floor_z=-0.6, stretch=STRETCH, base_vel=(0, 0, 0)):
    st = GK.casing_state(c, t, stretch, floor_z=floor_z, base_vel=base_vel) if t is not None else None
    if st is None:
        ob.hide_render = True; return False
    p, R = st
    R0 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])            # 薬莢の長軸(+z)を、排莢直後は銃口の向き(+Y)にそろえる
    M = np.eye(4); M[:3, :3] = R @ R0; M[:3, 3] = p
    ob.matrix_world = Matrix(M.tolist()); ob.hide_render = False; return True


def clear_anim(obs):
    for k, o in obs.items():
        if o.animation_data:
            o.animation_data_clear()


# ------------------------------------------------------------------ 本体
setup()
parts, rep = GK.build_parts(GLB)
json.dump(rep, open(os.path.join(OUT, 'parts_report.json'), 'w'), ensure_ascii=False, indent=1)
for k in GK.PART_NAMES:
    log('part %-14s tris=%5d boundary_edges=%4d bbox=%s' % (k, rep[k]['tris'], rep[k]['boundary'], rep[k]['bbox']))

obs = GK.build_blend(bpy, GLB, parts)
cas = GK.add_casing_object(bpy, 'casing'); cas.hide_render = True
# 書き出し用: 動き(アクション)をつけて .blend と .glb を保存。 casing はキットに含める(静止位置)
try:
    GK.apply_pose(obs, GK.gun_pose(None, 0, 6, 0.0))
    GK.make_actions(bpy, obs, fps=240, stretch=STRETCH, total_rounds=6)
    log('actions', [a.name for a in bpy.data.actions][:14])
except Exception as e:
    import traceback; traceback.print_exc(); log('ERR actions', e)
cas.hide_render = False; cas.location = (0.012, 0.04, 0.0725); cas.name = 'casing_sample'
try:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, 'gun_kit.blend'), compress=True)
    log('saved gun_kit.blend')
except Exception as e:
    log('ERR save blend', e)
# glb は、ライト・カメラを除いて書き出す(部品と薬莢だけ選択)
try:
    for o in bpy.data.objects:
        o.select_set(o.type in ('MESH', 'EMPTY'))
    GK.export_glb(bpy, os.path.join(OUT, 'gun_kit.glb'), animation=True)
except Exception as e:
    import traceback; traceback.print_exc(); log('ERR glb', e)
clear_anim(obs); sc.frame_set(1); rig()
sc.render.fps = 24
cas.name = 'casing'; cas.hide_render = True

TGT = (0.0, 0.07, 0.035)
# ---- (1) 発砲サイクル(引き伸ばし)。 右側から / 斜め後ろ上から
MS = [0.0, 2.0, 4.0, 6.5, 9.0, 10.5, 14.0, 22.0, 30.0, 38.0, 46.0, 54.0]
cl = GK.casing_launch(3)
VIEWS = {'right': (dirloc(TGT, (1.0, -0.28, 0.20)), 0.34), 'rearup': (dirloc(TGT, (0.55, -0.80, 0.60)), 0.34)}
for vn, (loc, scl) in VIEWS.items():
    look(loc, TGT, scl); tilec = 0; fs = []
    for ms in MS:
        t = ms / 1000.0 * STRETCH
        GK.apply_pose(obs, GK.gun_pose(t, 1, 6, 1.0, STRETCH))
        set_casing(cas, cl, t, floor_z=-0.6)
        fs.append(shoot('cyc_' + vn, 't=%.1fms (x%.1f)' % (ms, STRETCH)))
    montage(fs, 4, '1_cycle_%s' % vn)
cas.hide_render = True

# ---- (2) 弾切れの最後の1発のあと: スライドが後ろで止まる(右・上・斜め後ろ上)+ 撃つ前
tilec = 0; fs = []
for (loc, scl), (fired, t, note) in zip([VIEWS['right'], VIEWS['right'], VIEWS['rearup'], (dirloc(TGT, (0.1, -0.2, 1.0)), 0.30)],
                                         [(0, None, 'before: closed'), (6, None, 'after last shot: locked back'), (6, None, 'locked back (rear up)'), (6, None, 'locked back (top)')]):
    look(loc, TGT, scl); GK.apply_pose(obs, GK.gun_pose(t, fired, 6, 0.0, STRETCH)); fs.append(shoot('lock', note))
montage(fs, 2, '2_slide_lock')

# ---- (3) 分解図(色分け)。 部品を離して置く
for o in obs.values():
    o.hide_render = True
OFF = {'frame': (0, 0, 0), 'slide': (0, 0.0, 0.07), 'barrel': (0, 0.0, 0.125), 'trigger': (0.0, 0.03, -0.045), 'slide_stop': (-0.05, 0, 0.02), 'magazine_base': (0, 0, -0.05)}
xs = []
for nm in GK.PART_NAMES:
    o = GK.mesh_to_object(bpy, 'x_' + nm, parts[nm], GK.PIVOT[nm], GK.COLORS[nm]); o.location = tuple(np.array(GK.PIVOT[nm]) + np.array(OFF[nm])); xs.append(o)
tilec = 0; fs = []
T2 = (0.0, 0.07, 0.07)
for loc, scl in ((dirloc(T2, (0.9, -0.5, 0.45)), 0.46), (dirloc(T2, (-0.9, -0.5, 0.45)), 0.46), (dirloc(T2, (1.0, -0.15, 0.1)), 0.46), (dirloc(T2, (0.2, -0.3, 1.0)), 0.46)):
    look(loc, T2, scl); fs.append(shoot('expl', 'parts (colors)'))
montage(fs, 2, '3_exploded')
for o in xs:
    bpy.data.objects.remove(o)
for o in obs.values():
    o.hide_render = False

# ---- (4) スライド後退の最大位置の拡大(発砲 10.5ms)。 バレル・排莢口の中・フレームのレール面
GK.apply_pose(obs, GK.gun_pose(0.0105 * STRETCH, 1, 6, 1.0, STRETCH)); tilec = 0; fs = []
T4 = (0.0, 0.075, 0.062)
for loc, tg, scl, note in ((dirloc((0.0, 0.03, 0.07), (0.8, -0.3, 0.7)), (0.0, 0.03, 0.07), 0.15, 'port + chamber (right up)'),
                           (dirloc((0.0, 0.07, 0.06), (0.05, 0.0, 1.0)), (0.0, 0.07, 0.06), 0.30, 'top: barrel + rail'),
                           (dirloc((0.0, 0.13, 0.06), (0.7, 0.8, 0.45)), (0.0, 0.13, 0.06), 0.16, 'front right: rail face + barrel'),
                           (dirloc((0.0, 0.07, 0.05), (-0.6, -0.5, -0.5)), (0.0, 0.07, 0.05), 0.32, 'underside (left low)')):
    look(loc, tg, scl); fs.append(shoot('zoom', note))
montage(fs, 2, '4_slide_back_zoom')

# ---- (5) 薬莢の軌道(実時間 stretch=1。複数の位置を重ねる)。 路面 -0.6m
tilec = 0
GK.apply_pose(obs, GK.gun_pose(None, 0, 6, 0.0))
cs = []
for i in range(14):
    c = GK.add_casing_object(bpy, 'cs%02d' % i); cs.append(c)
fl = bpy.data.meshes.new('floor'); fo = bpy.data.objects.new('floor', fl); sc.collection.objects.link(fo)
fl.from_pydata([(-1, -1, -0.6), (1, -1, -0.6), (1, 1, -0.6), (-1, 1, -0.6)], [], [(0, 1, 2, 3)]); fl.update(); fo.data.materials.append(GK._material(bpy, 1))
fs = []
for k, (vn, loc, scl) in enumerate((('right', dirloc((0.0, -0.2, -0.2), (1.0, -0.05, 0.1)), 1.3), ('top', dirloc((0.0, -0.2, -0.2), (0.15, -0.1, 1.0)), 1.3))):
    look(loc, (0.0, -0.2, -0.2), scl)
    for i, c in enumerate(cs):
        set_casing(c, cl, 0.0065 + 0.035 * (i + 1) ** 1.05 / 1.0 * 0.5 + 0.004 * i, floor_z=-0.6 + GK.CASING_FLOOR_Z, stretch=1.0)
    fs.append(shoot('flight', 'casing path, real time (every ~20-35ms)'))
montage(fs, 2, '5_casing_flight')
log('done')
