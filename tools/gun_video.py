"""動く拳銃(tools/gun_kit.py)の単品動画: 12秒・24fps・4カット・6発・効果音つき。 チェイス本編とは無関係。
  描画 (Blender):  blender --background --factory-startup --python tools/gun_video.py -- render --frames "0,12,13" --out OUT --samples 24 --res 960x540
  音   (numpy):    python tools/gun_video.py audio OUT.wav [SFX_DIR]          (効果音は アセット庫/小道具/銃/音 の WAV)
  結合 (ffmpeg):   python tools/gun_video.py mux PNG_DIR AUDIO.wav OUT.mp4
  構図確認(numpy): python tools/gun_video.py check          (カメラの画角に銃・薬莢が入っているか)
  同期確認(numpy): python tools/gun_video.py sync [AUDIO.wav]
時刻はすべて「動画の時刻(秒) = コマ番号 / 24」。 発砲(撃針が雷管を打つ瞬間)は、コマの真ん中の時刻に合わせる(フラッシュを出すコマ = F)。
銃の動きは gun_kit.gun_pose / casing_state をそのまま使う(このファイルは時刻と見せ方だけ決める)。"""
import os, sys, json, math, subprocess, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import gun_kit as GK

FPS = 24
NF = 288                       # 12 秒
TOTAL_ROUNDS = 6
STRETCH = 3.5
GLB = os.path.join(ROOT, 'assets', 'bike_B2', 'gun.glb')

# ---- カット(コマ番号 [f0, f1))
CUTS = [dict(name='右側から全体', f0=0, f1=78), dict(name='斜め後ろ上から', f0=78, f1=156),
        dict(name='スライドと排莢口のアップ', f0=156, f1=216), dict(name='弾切れ・スライドストップ(回り込み)', f0=216, f1=288)]
# ---- 発砲。 F = 発砲のコマ(フラッシュ・銃声)。 s = 時間の引き伸ばし(1 = 実時間)
SHOTS = [dict(n=1, F=12, s=STRETCH), dict(n=2, F=54, s=1.0), dict(n=3, F=90, s=STRETCH), dict(n=4, F=132, s=1.0),
         dict(n=5, F=168, s=STRETCH), dict(n=6, F=228, s=STRETCH)]
# ---- 画面の隅のラベル(コマ [a, b))
LABELS = [(0, 40, '×3.5(引き伸ばし)'), (40, 78, '×1(実時間)'), (78, 118, '×3.5(引き伸ばし)'), (118, 156, '×1(実時間)'),
          (156, 216, '×3.5(引き伸ばし)'), (216, 288, '×3.5(引き伸ばし)')]

TABLE_Z = -0.0448               # 台の上面(弾倉の底板 -0.045 に接する)
CASING_FLOOR = TABLE_Z + GK.CASING_FLOOR_Z
CASING_CAP = 0.8                # 薬莢は排莢から実時間で 0.8 秒たったら、その状態で止める(転がり続けない)
TRIG_LEAD = 0.030               # 引き金を引き始める(実時間)。 発砲の 30 ms 前から 0→1
POSE_END = 0.075                # 発砲から実時間で 75 ms 後には、キットの動きは終わっている
GUN_C = np.array([0.0, 0.065, 0.02])
MUZZLE = (0.0, 0.1665, 0.0675)


def smooth(u):
    u = min(max(u, 0.0), 1.0); return u * u * (3 - 2 * u)


# ============================================================ カメラ
def _dir(az, el):
    a, e = math.radians(az), math.radians(el); return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def cam_at(f):
    """コマ f -> (位置, 注視点, 焦点距離mm)。 方位 az は +X(銃の右側)から +Y(銃口側)へ測る度数"""
    cut = [i for i, c in enumerate(CUTS) if c['f0'] <= f < c['f1']][0]
    c = CUTS[cut]; u = (f - c['f0']) / max(1, c['f1'] - c['f0'] - 1)
    if cut == 0:        # 右側の前寄りから全体。 薬莢は画面の左へ飛ぶ。 銃は画面の右寄り
        d = _dir(65, 16); R = 1.15 * (1 - 0.04 * u); lens = 35
        fwd = -d; rt = np.cross(fwd, [0, 0, 1]); rt /= np.linalg.norm(rt)
        tgt = GUN_C - 0.30 * rt + np.array([0, 0, 0.02]); return tgt + d * R, tgt, lens
    if cut == 1:        # 斜め後ろ上から
        d = _dir(-48, 36); R = 0.62 * (1 - 0.06 * u); tgt = np.array([0.02, 0.03, 0.03]); return tgt + d * R, tgt, 40
    if cut == 2:        # スライドと排莢口のアップ
        d = _dir(52, 13); R = 0.38 * (1 - 0.05 * u); tgt = np.array([0.012, 0.045, 0.068]); return tgt + d * R, tgt, 50
    az = 20 + 155 * smooth(u)           # 右前寄り -> 正面 -> 左(スライドストップのレバー側)。 薬莢の飛ぶ側(右後ろ)は通らない
    d = _dir(az, 22); tgt = np.array([0.0, 0.04, 0.03]); return tgt + d * 0.46, tgt, 50


def project(loc, tgt, lens, P):
    """3D 点 -> 画面(横幅を 1 としたときの x(-0.5〜0.5), y(縦 ±0.28), 奥行き)。 センサー幅 36 mm"""
    fwd = tgt - loc; fwd = fwd / np.linalg.norm(fwd); rt = np.cross(fwd, [0, 0, 1]); rt /= np.linalg.norm(rt); up = np.cross(rt, fwd)
    v = np.asarray(P, float) - loc; z = v @ fwd
    return (v @ rt) / z * lens / 36.0, (v @ up) / z * lens / 36.0, z


# ============================================================ 時刻から銃の姿勢
def pose_at(T):
    """動画の時刻 T[秒] の銃の部品の姿勢。 発砲の少し前から引き金を引き、発砲後は gun_pose(キット)に任せる"""
    last = None
    for sh in SHOTS:
        if T >= sh['F'] / FPS - TRIG_LEAD * sh['s'] - 1e-9:
            last = sh
    if last is None:
        return GK.gun_pose(None, 0, TOTAL_ROUNDS, 0.0, STRETCH)
    ft = last['F'] / FPS; s = last['s']; n = last['n']
    if T < ft:
        u = (T - (ft - TRIG_LEAD * s)) / (TRIG_LEAD * s)
        return GK.gun_pose(None, n - 1, TOTAL_ROUNDS, smooth(u), s)
    return GK.gun_pose(min(T - ft, POSE_END * s), n, TOTAL_ROUNDS, 1.0, s)


def casing_at(sh, T, c):
    """発砲 sh の薬莢の (位置, 回転行列) か None(まだ出ていない)。 実時間 CASING_CAP 秒で止める"""
    ft = sh['F'] / FPS; s = sh['s']
    t = min(T - ft, (CASING_CAP + c['t_ms'] / 1000.0) * s)
    return GK.casing_state(c, t, s, floor_z=CASING_FLOOR)


def casing_land_age(c):
    """薬莢が台に最初に当たるまで(排莢からの実時間[秒])"""
    for i in range(1, 2000):
        a = i * 0.001
        st = GK.casing_state(c, a + c['t_ms'] / 1000.0, 1.0, floor_z=CASING_FLOOR)
        if st is not None and st[0][2] <= CASING_FLOOR + 1e-9:
            return a
    return None


# ============================================================ 構図の確認(numpy)
def check():
    pts = []
    bb = [(-0.017, -0.035, -0.045), (0.017, 0.1655, 0.085)]
    for x in (bb[0][0], bb[1][0]):
        for y in (bb[0][1], bb[1][1]):
            for z in (bb[0][2], bb[1][2]):
                pts.append((x, y, z))
    rows = []
    for ci, c in enumerate(CUTS):
        shots = [sh for sh in SHOTS if c['f0'] <= sh['F'] < c['f1']]
        for f in (c['f0'], (c['f0'] + c['f1']) // 2, c['f1'] - 1):
            loc, tgt, lens = cam_at(f)
            g = [project(loc, tgt, lens, p) for p in pts]
            gx = [a[0] for a in g]; gy = [a[1] for a in g]
            rows.append('cut%d f%3d cam=%s lens=%d gun x[%.2f,%.2f] y[%.2f,%.2f]' % (ci + 1, f, np.round(loc, 2), lens, min(gx), max(gx), min(gy), max(gy)))
        for sh in shots:
            cl = GK.casing_launch(sh['n']); tot = None; inside = 0; n = 0; dmin = 9
            for k in range(0, int((CASING_CAP + 0.007) * sh['s'] * FPS)):
                T = sh['F'] / FPS + k / FPS; f = int(round(T * FPS))
                if f >= c['f1']: break
                st = casing_at(sh, T, cl)
                if st is None: continue
                loc, tgt, lens = cam_at(f); x, y, z = project(loc, tgt, lens, st[0]); n += 1
                dmin = min(dmin, np.linalg.norm(st[0] - loc))
                if abs(x) < 0.5 and abs(y) < 0.28 and z > 0: inside += 1
            rows.append('   shot%d(x%.1f): casing in-frame %d/%d frames, min dist to camera %.2f m' % (sh['n'], sh['s'], inside, n, dmin))
    print('\n'.join(rows))


# ============================================================ 音
def build_cues(sfx_dir):
    import sfx_cues as SC
    tmp = tempfile.mkdtemp(prefix='gunsfx_')
    import soundfile as sf, shutil
    for fn in os.listdir(sfx_dir):
        if fn.endswith('.wav'):
            shutil.copy(os.path.join(sfx_dir, fn), os.path.join(tmp, fn))
    x, sr = sf.read(os.path.join(sfx_dir, 'slide_cycle.wav'), always_2d=True)
    i0 = int(0.0535 * sr); tail = x[i0:].copy(); fi = int(0.002 * sr); tail[:fi] *= np.linspace(0, 1, fi)[:, None]
    sf.write(os.path.join(tmp, 'slide_close.wav'), tail, sr, subtype='PCM_24')     # 閉鎖音だけ(引き伸ばし用)
    dist = 0.7; dly = dist / SC.SOUND_SPEED
    cues = []; info = []
    for sh in SHOTS:
        ft = sh['F'] / FPS; s = sh['s']; n = sh['n']
        cs = SC.gun_cues([ft], total_rounds=(1 if n == TOTAL_ROUNDS else 99), camera_dist=dist, trigger=True)
        cl = GK.casing_launch(n); la = casing_land_age(cl)
        for c in cs:
            k = c['kind']
            if k == 'gunshot':
                pass                                                          # 発砲の瞬間 = ft(+音速の遅れ 2 ms)
            elif k == 'slide_cycle':
                if s > 1:                                                     # 引き伸ばし: 後退音と閉鎖音を別々に、映像の時刻へ
                    c['file'] = 'slide_back.wav'; c['t'] = ft + dly + SC.T_SLIDE_REAR * s
                    cues.append(dict(kind='slide_close', t=ft + dly + SC.T_SLIDE_CLOSE * s, file='slide_close.wav', gain_db=c['gain_db'], pan=c['pan'], note='閉鎖(引き伸ばし)'))
            elif k == 'slide_back':
                c['t'] = ft + dly + SC.T_SLIDE_REAR * s
            elif k == 'slide_stop':
                c['t'] = ft + dly + 0.011 * s                                 # スライドストップのレバーが上がりきる(キット: 11 ms)
            elif k == 'casing':
                c['t'] = ft + (la + cl['t_ms'] / 1000.0) * s + dly; c['file'] = 'casing_drop_%d.wav' % (1 + (n - 1) % 3)
            elif k == 'trigger':
                c['t'] = ft - 0.004 * s
            cues.append(c)
        info.append(dict(n=n, F=sh['F'], s=s, land_age=la))
    return sorted(cues, key=lambda c: c['t']), tmp, info


def build_audio(out_wav, sfx_dir=None):
    import sfx_cues as SC, soundfile as sf
    sfx_dir = sfx_dir or os.environ.get('SFX_DIR') or os.path.join('G:\\', 'マイドライブ', 'newCom', '1_アプリ開発', 'AIの作業場', 'アセット庫', '小道具', '銃', '音')
    cues, tmp, info = build_cues(sfx_dir)
    y = SC.render_cues(cues, NF / FPS, 48000, sfx_dir=tmp)
    sf.write(out_wav, y, 48000, subtype='PCM_16')
    json.dump(dict(cues=cues, shots=info), open(os.path.splitext(out_wav)[0] + '_cues.json', 'w'), ensure_ascii=False, indent=1, default=float)
    print('audio', out_wav, y.shape, 'peak %.3f' % np.abs(y).max())
    return cues


def sync(wav=None, sfx_dir=None):
    """映像側の時刻(発砲のコマ F・スライドが最後尾に来る時刻)と、音の山(WAV の実測)を並べる"""
    import soundfile as sf, sfx_cues as SC
    wav = wav or os.path.join(ROOT, 'assets', 'gun_video_audio.wav')
    y, sr = sf.read(wav, always_2d=True); m = np.abs(y).max(1)
    sfx_dir = sfx_dir or os.path.join('G:' + os.sep, 'マイドライブ', 'newCom', '1_アプリ開発', 'AIの作業場', 'アセット庫', '小道具', '銃', '音')
    cues, tmp, info = build_cues(sfx_dir)
    ys = SC.render_cues([c for c in cues if c['kind'] in ('slide_cycle', 'slide_back', 'slide_stop')], NF / FPS, 48000, sfx_dir=tmp, peak_dbfs=-1.0)
    ms = np.abs(ys).max(1)                                                # スライド系の音だけを別に作って、立ち上がりを測る
    yc = SC.render_cues([c for c in cues if c['kind'] == 'casing'], NF / FPS, 48000, sfx_dir=tmp); mc = np.abs(yc).max(1)
    print('音: %s %.2f s(WAV実測。立ち上がり = 山の50%%に最初に届く点)' % (wav, len(y) / sr))
    for sh, inf in zip(SHOTS, info):
        ft = sh['F'] / FPS; s = sh['s']
        a0, a1 = int((ft - 0.02) * sr), int((ft + 0.03) * sr); seg = m[a0:a1]; on = (a0 + int(np.argmax(seg > seg.max() * 0.5))) / sr
        ts = np.arange(0, 0.06 * s, 0.0005)
        back = [-GK.gun_pose(t, sh['n'], TOTAL_ROUNDS, 1.0, s)['slide'][1] for t in ts]
        vt = ft + ts[int(np.argmax(np.array(back) > max(back) * 0.97))]            # 映像: スライドが最後尾の 97% に着く時刻
        a2, a3 = int((vt - 0.03) * sr), int((vt + 0.03) * sr); sm = ms[a2:a3]; sk = (a2 + int(np.argmax(sm > sm.max() * 0.5))) / sr
        cl = GK.casing_launch(sh['n']); land = ft + (casing_land_age(cl) + cl['t_ms'] / 1000.0) * s
        a4, a5 = int((land - 0.03) * sr), int((land + 0.05) * sr); cm = mc[a4:a5]; ck = (a4 + int(np.argmax(cm > cm.max() * 0.5))) / sr
        print('発砲%d(x%.1f) 発砲コマF=%d(%.4fs): 銃声 %.4fs(%+.1f ms) | スライド最後尾 映像 %.4fs(コマ%.2f)・音 %.4fs(%+.1f ms) | 薬莢着地 映像 %.4fs(コマ%.2f)・音 %.4fs(%+.1f ms)'
              % (sh['n'], s, sh['F'], ft, on, (on - ft) * 1e3, vt, vt * FPS, sk, (sk - vt) * 1e3, land, land * FPS, ck, (ck - land) * 1e3))


# ============================================================ ffmpeg で連結(drawtext のラベル + 音声 AAC)
def mux(png_dir, audio, out):
    font = os.environ.get('GUN_FONT')
    if not font:
        try:
            font = subprocess.run(['fc-match', '-f', '%{file}', 'Noto Sans CJK JP:bold'], capture_output=True, text=True).stdout.strip()
        except OSError:
            font = ''
        font = font or '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
    esc = lambda q: q.replace(chr(92), '/').replace(':', chr(92) + ':')       # ffmpeg のフィルタ内のパス(Windows の C: 対策)
    font = esc(font)
    tmp = tempfile.mkdtemp(prefix='gunlab_'); parts = []
    for i, (a, b, txt) in enumerate(LABELS):
        p = os.path.join(tmp, 'l%d.txt' % i); open(p, 'w', encoding='utf-8').write(txt)
        parts.append("drawtext=fontfile='%s':textfile='%s':x=22:y=18:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.45:boxborderw=7:enable='between(t,%.4f,%.4f)'" % (font, esc(p), a / FPS, (b - 0.01) / FPS))
    fs = ','.join(parts)
    cmd = ['ffmpeg', '-loglevel', 'error', '-y', '-framerate', str(FPS), '-start_number', '0', '-i', os.path.join(png_dir, '%04d.png'), '-i', audio,
           '-vf', fs, '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p',
           '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-shortest', '-movflags', '+faststart', out]
    print(' '.join(cmd)); subprocess.check_call(cmd)


# ============================================================ Blender
def render_main(argv):
    import bpy, time
    from mathutils import Vector, Matrix
    ar = {}
    k = 0
    while k < len(argv):
        if argv[k].startswith('--'):
            ar[argv[k][2:]] = argv[k + 1]; k += 2
        else:
            k += 1
    frames = [int(x) for x in ar['frames'].split(',') if x.strip()]
    OUT = ar.get('out', 'out'); os.makedirs(OUT, exist_ok=True)
    SAMPLES = int(ar.get('samples', 24)); RW, RH = [int(v) for v in ar.get('res', '960x540').split('x')]
    t00 = time.time()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'; sc.cycles.samples = SAMPLES; sc.cycles.device = 'CPU'
    try:
        sc.cycles.use_denoising = True; sc.cycles.denoiser = 'OPENIMAGEDENOISE'
    except Exception:
        pass
    sc.render.resolution_x = RW; sc.render.resolution_y = RH; sc.render.resolution_percentage = 100
    sc.render.fps = FPS; sc.frame_start = 0; sc.frame_end = NF - 1
    sc.view_settings.view_transform = 'Standard'
    sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.6
    sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGB'
    try:
        bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
    except Exception as e:
        print('WARNING interpolation pref', e)
    w = bpy.data.worlds.new('w'); w.use_nodes = True
    w.node_tree.nodes['Background'].inputs[0].default_value = (0.040, 0.042, 0.047, 1); w.node_tree.nodes['Background'].inputs[1].default_value = 1.0; sc.world = w

    # ---- 台(暗い灰の面。広くして、遠くは光が届かず暗く落ちる)
    me = bpy.data.meshes.new('table'); R = 14.0
    me.from_pydata([(-R, -R, TABLE_Z), (R, -R, TABLE_Z), (R, R, TABLE_Z), (-R, R, TABLE_Z)], [], [(0, 1, 2, 3)]); me.update()
    tb = bpy.data.objects.new('table', me); sc.collection.objects.link(tb)
    mt = bpy.data.materials.new('table_m'); mt.use_nodes = True; b = mt.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (0.055, 0.057, 0.062, 1); b.inputs['Roughness'].default_value = 0.5; b.inputs['Metallic'].default_value = 0.0
    me.materials.append(mt)

    # ---- 3 灯(キー・フィル・リム)。 灯りは映らないようにする
    def area(name, loc, energy, size, color=(1, 1, 1)):
        ld = bpy.data.lights.new(name, 'AREA'); ld.energy = energy; ld.size = size; ld.color = color
        lo = bpy.data.objects.new(name, ld); lo.location = loc; sc.collection.objects.link(lo)
        lo.rotation_euler = (Vector(tuple(GUN_C)) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
        try:
            lo.visible_camera = False
        except Exception:
            pass
        return lo
    area('key', (0.75, 0.55, 0.85), 85, 0.5, (1.0, 0.96, 0.90))
    area('fill', (-0.85, 0.20, 0.45), 22, 0.8, (0.85, 0.92, 1.0))
    area('rim', (-0.15, -0.85, 0.65), 60, 0.35, (1.0, 1.0, 1.0))

    # ---- 銃(キットそのまま)
    parts, rep = GK.build_parts(GLB)
    obs = GK.build_blend(bpy, GLB, parts)
    KEYED = ['slide', 'barrel', 'trigger', 'slide_stop']
    GK.apply_pose(obs, GK.gun_pose(None, 0, TOTAL_ROUNDS, 0.0, STRETCH))

    def key_pose(T):
        fr = T * FPS
        GK.apply_pose(obs, pose_at(T))
        for nm in KEYED:
            ob = obs[nm]; ob.keyframe_insert('location', frame=fr); ob.keyframe_insert('rotation_euler', index=0, frame=fr)
    key_pose(0.0)
    for sh in SHOTS:
        ft = sh['F'] / FPS; s = sh['s']; T0 = ft - TRIG_LEAD * s - 0.002; T1 = ft + (POSE_END + 0.004) * s
        # 引き伸ばしは 1/480 s(0.05 コマ)刻み、 実時間は 1/960 s 刻み。 発砲の瞬間 ft を必ず含める
        step = (1 / 960.0) if s == 1 else (1 / 480.0)
        ts = list(np.arange(T0, T1, step)) + [ft, T1]
        for T in sorted(set(round(float(t), 6) for t in ts)):
            key_pose(T)
    key_pose(NF / FPS)

    # ---- 薬莢(発砲ごとに 1 個。 出るまでは大きさ 0。 台の上に残る)
    R0 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])
    for sh in SHOTS:
        c = GK.casing_launch(sh['n']); ob = GK.add_casing_object(bpy, 'casing%d' % sh['n']); ob.rotation_mode = 'QUATERNION'
        ft = sh['F'] / FPS; s = sh['s']; eject = ft + c['t_ms'] / 1000.0 * s
        fr_e = eject * FPS; fr_end = (eject + CASING_CAP * s) * FPS
        live = any(fr_e - 2 <= f <= fr_end + 2 for f in frames)        # このプロセスで描くコマに、飛んでいる薬莢が写るときだけ、軌道を細かく作る
        done = any(f > fr_end + 2 for f in frames)                      # 台の上で止まった薬莢だけ写るなら、最後の位置に置くだけ
        ob.scale = (0.0, 0.0, 0.0); ob.keyframe_insert('scale', frame=0)
        if not live and not done:
            continue
        ob.scale = (0.0, 0.0, 0.0); ob.keyframe_insert('scale', frame=(eject - 0.002) * FPS)
        ob.scale = (1.0, 1.0, 1.0); ob.keyframe_insert('scale', frame=eject * FPS)
        dage = 0.001 if s == 1 else 0.002; qprev = None
        ages = list(np.arange(0.0, CASING_CAP + dage, dage)) if live else [CASING_CAP]
        for a in ages:
            T = eject + a * s
            st = casing_at(sh, T, c)
            if st is None: continue
            p, Rm = st; Mx = np.eye(4); Mx[:3, :3] = Rm @ R0; Mx[:3, 3] = p
            q = Matrix(Mx.tolist()).to_quaternion()
            if qprev is not None and q.dot(qprev) < 0: q = -q
            qprev = q
            ob.location = tuple(p); ob.rotation_quaternion = q
            if not live:                                               # 止まった薬莢: 出る時刻から最後の位置にある(写らない時間は関係ない)
                ob.keyframe_insert('location', frame=eject * FPS); ob.keyframe_insert('rotation_quaternion', frame=eject * FPS); continue
            ob.keyframe_insert('location', frame=T * FPS); ob.keyframe_insert('rotation_quaternion', frame=T * FPS)

    # ---- マズルフラッシュ(chase/rider.py の make_flash と同じ形。 発砲のコマだけ出る)
    mf = bpy.data.materials.new('MuzzleFlash'); mf.use_nodes = True; nt = mf.node_tree
    for n_ in list(nt.nodes):
        nt.nodes.remove(n_)
    em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Color'].default_value = (1.0, 0.72, 0.30, 1); em.inputs['Strength'].default_value = 110.0
    oo = nt.nodes.new('ShaderNodeOutputMaterial'); nt.links.new(em.outputs[0], oo.inputs['Surface'])
    V, Fc = [], []

    def tri(a_, b_, c_):
        n_ = len(V); V.extend([a_, b_, c_]); Fc.append((n_, n_ + 1, n_ + 2))
    for i in range(8):
        a0 = 2 * math.pi * i / 8; a1 = 2 * math.pi * (i + 1) / 8
        tri((0.022 * math.cos(a0), 0, 0.022 * math.sin(a0)), (0.022 * math.cos(a1), 0, 0.022 * math.sin(a1)), (0, 0.15, 0))
    for k_ in range(8):
        a = 2 * math.pi * k_ / 8; tip = (0.085 * math.cos(a), 0.02, 0.085 * math.sin(a)); w_ = 0.016
        tri((-w_ * math.sin(a), 0.0, w_ * math.cos(a)), (w_ * math.sin(a), 0.0, -w_ * math.cos(a)), tip)
    for k_ in range(4):
        a = 2 * math.pi * k_ / 4 + 0.4; tri((0.02, 0.0, 0.0), (-0.02, 0.0, 0.0), (0.05 * math.cos(a), 0.09, 0.05 * math.sin(a)))
    fme = bpy.data.meshes.new('flash'); fme.from_pydata(V, [], Fc); fme.materials.append(mf); fme.update()
    fo = bpy.data.objects.new('flash', fme); sc.collection.objects.link(fo); fo.location = MUZZLE
    fl = bpy.data.lights.new('flashL', 'POINT'); fl.energy = 0.0; fl.color = (1.0, 0.75, 0.4); fl.shadow_soft_size = 0.03
    flo = bpy.data.objects.new('flashL', fl); sc.collection.objects.link(flo); flo.parent = fo; flo.location = (0, 0.12, 0)
    FS = 1.0; FE = 1.5
    fo.scale = (0.001,) * 3; fo.keyframe_insert('scale', frame=-5); fl.energy = 0.0; fl.keyframe_insert('energy', frame=-5)
    for sh in SHOTS:
        F = sh['F']
        for fr, sv, ev in ((F - 0.5, 0.001, 0.0), (F - 0.46, FS, FE), (F + 0.46, FS, FE), (F + 0.5, 0.001, 0.0)):
            fo.scale = (sv,) * 3; fo.keyframe_insert('scale', frame=fr); fl.energy = ev; fl.keyframe_insert('energy', frame=fr)

    # ---- カメラ
    cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.sensor_fit = 'HORIZONTAL'; cam.data.sensor_width = 36.0; cam.data.clip_start = 0.01; cam.data.clip_end = 30
    print('BUILD', round(time.time() - t00, 1), 's')

    timing = []
    for f in frames:
        t1 = time.time()
        loc, tgt, lens = cam_at(f)
        cam.location = tuple(loc); cam.data.lens = lens
        cam.rotation_euler = (Vector(tuple(tgt)) - Vector(tuple(loc))).to_track_quat('-Z', 'Y').to_euler()
        sc.frame_set(f)
        sc.render.filepath = os.path.join(OUT, '%04d.png' % f)
        bpy.ops.render.render(write_still=True)
        dt = time.time() - t1; timing.append(dict(frame=f, render_s=round(dt, 2))); print('FRAME', f, round(dt, 1), 's', flush=True)
    json.dump(dict(build_s=round(time.time() - t00 - sum(t['render_s'] for t in timing), 1), frames=timing), open(os.path.join(OUT, 'timing_%s.json' % ar.get('idx', '000')), 'w'))
    print('TOTAL', round(time.time() - t00, 1))


if __name__ == '__main__':
    try:
        import bpy
        IN_BPY = True
    except ImportError:
        IN_BPY = False
    if IN_BPY:
        render_main(sys.argv[sys.argv.index('--') + 2:])         # blender ... -- render --frames ...
    else:
        a = sys.argv[1:]
        if a[0] == 'check': check()
        elif a[0] == 'audio': build_audio(a[1], a[2] if len(a) > 2 else None)
        elif a[0] == 'mux': mux(a[1], a[2], a[3])
        elif a[0] == 'sync': sync(a[1] if len(a) > 1 else None)
