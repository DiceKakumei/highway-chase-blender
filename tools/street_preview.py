"""street_preview.py — 道路の断面キット(street_kit)と足元(building_base)の見本画像・書き出し。チェイス本編とは無関係。
  blender --background --factory-startup --python tools/street_preview.py -- OUT [params.json]
  params.json: {"part": "all"|"section"|"views"|"before"|"samples", "samples": 24, "res": [960, 540], "shots": ["02", ...]}
出力 OUT/01_section.png(断面図) 02-05 地上の目線・高架の目線(夕方・夜) 06-07 従来相当との比較 sample_*.png / street_samples.blend / sample_*.glb / spec_check.json
hand_view.yml で動かす。 座標: 道路の中心線 = +X、建物は +Y 側(正面 -Y)。 z=0 = 車道面。"""
import bpy, bmesh, math, os, sys, json, time, random
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import street_kit as sk
import building_base as bb
try:
    import building_kit as bk
except Exception as e:
    bk = None
    print('hand building_kit import failed', e)

ARGV = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else ['out']
OUT = ARGV[0]; os.makedirs(OUT, exist_ok=True)
PRM = json.load(open(ARGV[1])) if len(ARGV) > 1 else {}
PART = PRM.get('part', 'all'); SAMPLES = int(PRM.get('samples', 24)); RES = PRM.get('res', [960, 540]); ONLY = PRM.get('shots')


def log(*a):
    print('hand', *a, flush=True)


# ------------------------------------------------------------------ シーン(building_preview と同じ作り)
def fresh(res=None):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = SAMPLES
    sc.cycles.use_adaptive_sampling = True; sc.cycles.adaptive_threshold = 0.03; sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = 'OPENIMAGEDENOISE'
    except Exception:
        pass
    sc.cycles.max_bounces = 8; sc.cycles.glossy_bounces = 6; sc.cycles.transmission_bounces = 8; sc.cycles.transparent_max_bounces = 24
    sc.render.resolution_x, sc.render.resolution_y = res or RES; sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    try:
        sc.view_settings.view_transform = 'AgX'
    except Exception:
        pass
    return sc


def make_world(sc, mode, sun_dir):
    w = bpy.data.worlds.new('W')
    try:
        w.use_nodes = True
    except Exception:
        pass
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputWorld'); bg = nt.nodes.new('ShaderNodeBackground')
    nb = sk.NB(nt)
    tc = nt.nodes.new('ShaderNodeTexCoord')
    nv = nt.nodes.new('ShaderNodeVectorMath'); nv.operation = 'NORMALIZE'; nt.links.new(tc.outputs['Generated'], nv.inputs[0])
    x, y, z = nb.sep(nv.outputs[0])
    if mode == 'day':
        hor, zen, gro, glow = (1.0, 0.52, 0.26), (0.07, 0.15, 0.40), (0.05, 0.04, 0.04), (1.0, 0.42, 0.12)
    else:
        hor, zen, gro, glow = (0.030, 0.040, 0.085), (0.003, 0.006, 0.020), (0.004, 0.004, 0.005), (0.0, 0.0, 0.0)
    col = nb.vmix(nb.rgb(hor), nb.rgb(zen), nb.sstep(z, 0.0, 0.55))
    dot = nt.nodes.new('ShaderNodeVectorMath'); dot.operation = 'DOT_PRODUCT'; nt.links.new(nv.outputs[0], dot.inputs[0]); dot.inputs[1].default_value = tuple(sun_dir)
    g = nb.m('POWER', nb.m('MAXIMUM', dot.outputs['Value'], 0.0), 6.0)
    col = nb.vop('ADD', col, nb.vscale(nb.rgb(glow), nb.m('MULTIPLY', g, 1.6)))
    col = nb.vmix(col, nb.rgb(gro), nb.sstep(z, 0.0, -0.08))
    nt.links.new(col, bg.inputs['Color']); bg.inputs['Strength'].default_value = 1.0
    nt.links.new(bg.outputs[0], out.inputs['Surface'])
    sc.world = w


def add_sun(sc, direction_to_sun, strength, color, angle=0.03):
    ld = bpy.data.lights.new('sun', 'SUN'); ld.energy = strength; ld.color = color; ld.angle = angle
    lo = bpy.data.objects.new('sun', ld); sc.collection.objects.link(lo)
    lo.rotation_euler = (-Vector(direction_to_sun)).to_track_quat('-Z', 'Y').to_euler()


def add_ground(sc, z=-0.02, size=900.0, color=(0.05, 0.05, 0.055)):
    me = bpy.data.meshes.new('ground'); s = size / 2
    me.from_pydata([(-s, -s, z), (s, -s, z), (s, s, z), (-s, s, z)], [], [(0, 1, 2, 3)])
    o = bpy.data.objects.new('ground', me); sc.collection.objects.link(o)
    mat, nt, out_ = sk.new_material('ground')
    b = sk.principled(nt, **{'Base Color': (*color, 1.0), 'Roughness': 0.85}); nt.links.new(b.outputs[0], out_.inputs['Surface']); me.materials.append(mat)


def set_cam(sc, pos, target, lens):
    cd = bpy.data.cameras.new('cam'); cd.lens = lens; cd.clip_end = 3000.0; cd.sensor_width = 36
    co = bpy.data.objects.new('cam', cd); sc.collection.objects.link(co); sc.camera = co
    co.location = pos; co.rotation_euler = (Vector(target) - Vector(pos)).to_track_quat('-Z', 'Y').to_euler()


def render(sc, name):
    sc.render.filepath = os.path.join(OUT, name + '.png'); t0 = time.time()
    bpy.ops.render.render(write_still=True)
    log('rendered', name, '%.0fs' % (time.time() - t0))


SUN_DUSK = Vector((0.45, -0.80, 0.22)).normalized()
SUN_NIGHT = Vector((-0.4, -0.5, 0.75)).normalized()


def scene_basic(mode, res=None):
    sc = fresh(res); sd = SUN_DUSK if mode == 'day' else SUN_NIGHT
    make_world(sc, mode, sd)
    if mode == 'day':
        add_sun(sc, sd, 3.8, (1.0, 0.68, 0.40))
    else:
        add_sun(sc, sd, 0.05, (0.65, 0.75, 1.0), angle=0.06)
    add_ground(sc)
    return sc


# ------------------------------------------------------------------ 街のサンプル(通り + 1階 + 上の階)
BW, BD = 19.5, 11.7          # ビルの幅(zakkyo の5モジュール)・奥行き
SEGS = [
    dict(name='low', kw=dict(foundation='low'),
         rows=[[('glass', 11.5), ('entrance', 4.0), ('shutter', 4.0)], [('tile', 9.5), ('parking', 5.0), ('shutter', 5.0)]]),
    dict(name='wainscot', kw=dict(foundation='wainscot', found_mat='stone'),
         rows=[[('glass', 12.0), ('tile', 7.5)], [('entrance', 6.0), ('glass', 13.5)]]),
    dict(name='flush', kw=dict(foundation='flush', setback=1.5),
         rows=[[('entrance', 5.5), ('glass', 14.0)], [('shutter', 6.0), ('parking', 5.5), ('tile', 8.0)]]),
]
SEGLEN = 39.0
G1 = 4.2        # building_kit zakkyo の1階の高さ(上の階はここから)


def kit_upper(x, y, seed, detail, night, zcut, name):
    """building_kit の建物を置き、1階(zcut より下)の面を消す。 無ければ箱。 -> root"""
    if bk is not None:
        try:
            r = bk.make_building('zakkyo', BW, BD, 7, seed=seed, detail=detail, night=night, hour=20.5, name=name)
            r.location = (x, y, 0)
            body = [c for c in r.children if c.type == 'MESH'][0]
            bm = bmesh.new(); bm.from_mesh(body.data); bm.faces.ensure_lookup_table()
            dele = [f for f in bm.faces if f.calc_center_median().z < zcut - 0.01]
            bmesh.ops.delete(bm, geom=dele, context='FACES'); bm.to_mesh(body.data); bm.free()
            return r
        except Exception as e:
            log('kit_upper failed', repr(e))
    me = bpy.data.meshes.new(name); s = (BW / 2, BD / 2); z0 = zcut; z1 = zcut + 25.0
    V = [(-s[0], -s[1], z0), (s[0], -s[1], z0), (s[0], s[1], z0), (-s[0], s[1], z0), (-s[0], -s[1], z1), (s[0], -s[1], z1), (s[0], s[1], z1), (-s[0], s[1], z1)]
    me.from_pydata(V, [], [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    o = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(o); o.location = (x, y, 0)
    m, nt, out = sk.new_material(name + 'm'); b = sk.principled(nt, **{'Base Color': (0.4, 0.38, 0.35, 1), 'Roughness': 0.9}); nt.links.new(b.outputs[0], out.inputs['Surface']); me.materials.append(m)
    return o


def build_segment(i, seg, x_off, detail, night, upper_detail, seed=1, lights=True, collection=None):
    """通り1区間(SEGLEN)を作る。 x_off だけ +X へずらす。 -> dict(street, rows, info)"""
    kw = dict(seg['kw']); kw.update(night=night, lights=lights)
    _, I0, _, _ = sk.build_geometry('tokyo246', None, SEGLEN, seed + i, 'far', **kw)         # 壁の位置・床の高さだけ先に知る
    bz, gl, wp = I0['base_z'], I0['gl'], I0['wall_plane']
    H1 = G1 - bz
    gaps = []; drives = []; rows = []
    for k, spec in enumerate(seg['rows']):
        cx = BW * (k + 0.5) + 0.2
        r = bb.make_ground_floor_row(spec, H1, BD, bz, gl, seed=seed * 10 + i * 3 + k, detail=detail, night=night, position=(x_off + cx, wp + BD / 2), name='f1_%d_%d' % (i, k))
        inf = bb.floor_info(r); rows.append(r)
        for (a, b), kd in zip(inf['gaps_x'], inf['gaps_kind']):
            gaps.append((cx + a, cx + b))
            if kd == 'parking':
                drives.append((cx + a, cx + b))
        kit_upper(x_off + cx, wp + BD / 2, seed + i * 7 + k, upper_detail, night, G1, 'up_%d_%d' % (i, k))
    kw.update(found_gaps=gaps, driveways=drives)
    st = sk.make_street_section('tokyo246', None, SEGLEN, seed + i, detail, name='street_%d' % i, **kw)
    st.location = (x_off, 0, 0)
    return dict(street=st, rows=rows, info=sk.street_info(st), gaps=gaps, drives=drives)


def view_scene(mode, near_seg):
    sc = scene_basic(mode)
    night = mode == 'night'
    segs = []
    for i, seg in enumerate(SEGS):
        det = 'near' if i == near_seg else ('mid' if abs(i - near_seg) == 1 else 'far')
        up = 'mid' if det != 'far' else 'far'
        segs.append(build_segment(i, seg, i * SEGLEN, det, night, up))
    # 反対側(道の向こう)は暗い地面のまま(見える範囲が狭いので省略)
    return sc, segs


def shot_eye(name, mode):
    sc, segs = view_scene(mode, 0)
    I = segs[0]['info']
    y_s = I['sidewalk_o0'] + 1.5
    set_cam(sc, (2.0, y_s - 0.4, 1.6), (13.0, I['wall_plane'] + 0.0, 1.7), 21)
    render(sc, name)


def shot_eye2(name, mode):
    sc, segs = view_scene(mode, 1)
    I = segs[1]['info']
    set_cam(sc, (SEGLEN + 6.0, I['sidewalk_o0'] + 1.2, 1.5), (SEGLEN + 14.0, I['wall_plane'] + 0.0, 1.0), 24)
    render(sc, name)


def shot_aerial(name, mode):
    sc, segs = view_scene(mode, 0)
    set_cam(sc, (-8.0, 1.0, 10.5), (46.0, 10.0, 0.5), 22)
    render(sc, name)


def shot_aerial2(name, mode):
    sc, segs = view_scene(mode, 0)
    set_cam(sc, (4.0, 0.5, 11.0), (34.0, 13.0, 0.5), 24)
    render(sc, name)


def shot_before(name, mode):
    """従来相当: 道路(車道)から直接ビルの壁が立つ。 city.py と同じ作り(歩道・基礎・1階の作り込みなし)を、同じ視点で再現した比較用"""
    sc = scene_basic(mode); night = mode == 'night'
    _, I0, _, P0 = sk.build_geometry('tokyo246', None, SEGLEN, 1, 'far', night=night)
    Wr = I0['half_road']
    md = sk.MD(); L = SEGLEN * len(SEGS)
    md.face([(0, 0, 0), (L, 0, 0), (L, Wr + 1.0, 0), (0, Wr + 1.0, 0)], 'road', [(0, 0), (L, 0), (L, Wr + 1.0), (0, Wr + 1.0)])
    me = bpy.data.meshes.new('oldroad'); me.from_pydata(md.V, [], md.F)
    uvl = me.uv_layers.new(name='UVMap'); uvl.data.foreach_set('uv', np.asarray(md.UV, np.float32).ravel())
    me.polygons.foreach_set('material_index', np.zeros(len(md.F), np.int32))
    info = dict(I0); info['sidewalk_o0'] = Wr; info['boundary_o'] = Wr + 1.0; info['tactile'] = (Wr, Wr)
    me.materials.append(sk.build_material('road', P0, info, 'oldroad_m'))
    o = bpy.data.objects.new('oldroad', me); sc.collection.objects.link(o)
    for i in range(len(SEGS) * 2):
        if bk is not None:
            r = bk.make_building('zakkyo', BW, BD, 8, seed=3 + i, detail='mid' if i < 4 else 'far', night=night, hour=20.5, name='old_%d' % i)
            r.location = (BW * (i + 0.5) + 0.2, Wr + 1.0 + BD / 2, 0)
    if night:      # 従来は街灯の光だまりだけ
        for lx in (10.0, 50.0, 90.0):
            ld = bpy.data.lights.new('l', 'POINT'); ld.energy = 3000.0; ld.color = (1.0, 0.75, 0.45); lo = bpy.data.objects.new('l', ld); lo.location = (lx, Wr + 0.5, 8.0); sc.collection.objects.link(lo)
    set_cam(sc, (-8.0, 1.0, 10.5), (46.0, 10.0, 0.5), 22)
    render(sc, name)


# ------------------------------------------------------------------ 断面図(2D描画。メッシュを面で切って、その切り口を描く)
def get_pil():
    try:
        from PIL import Image, ImageDraw, ImageFont
        return Image, ImageDraw, ImageFont
    except Exception:
        import subprocess
        tgt = '/tmp/pil_pkg'
        subprocess.call([sys.executable, '-m', 'ensurepip', '--default-pip'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', '--target', tgt, 'pillow'])
        sys.path.insert(0, tgt)
        from PIL import Image, ImageDraw, ImageFont
        return Image, ImageDraw, ImageFont


def slice_md(md, x0, dy=0.0):
    segs = []; V = np.asarray(md.V, float)
    for f, mi in zip(md.F, md.MI):
        pts = V[list(f)]
        if pts[:, 0].max() < x0 or pts[:, 0].min() > x0:
            continue
        cr = []; n = len(f)
        for i in range(n):
            a = pts[i]; b = pts[(i + 1) % n]; da = a[0] - x0; db = b[0] - x0
            if (da <= 0) != (db <= 0):
                t = da / (da - db); p = a + t * (b - a); cr.append((p[1] + dy, p[2]))
        if len(cr) >= 2:
            segs.append((cr[0][0], cr[0][1], cr[1][0], cr[1][1], md.roles[mi]))
    return segs


ROLE_COL = {'road': (70, 70, 78), 'gutter': (110, 110, 120), 'curb': (190, 190, 185), 'sidewalk': (160, 150, 135), 'tactile': (245, 200, 20), 'tactile_bar': (245, 200, 20),
            'boundary': (150, 150, 145), 'setback': (135, 130, 120), 'inuhashiri': (160, 160, 155), 'found_conc': (120, 120, 115), 'found_stone': (45, 45, 55), 'found_tile': (140, 80, 55),
            'flashing': (230, 230, 235), 'soil': (110, 78, 50), 'trunk': (100, 70, 40), 'leaf': (80, 150, 60), 'fence_dark': (50, 50, 60), 'fence_galv': (170, 175, 180), 'glass': (110, 190, 235),
            'fascia': (130, 70, 150), 'stainless': (200, 205, 215), 'rail_metal': (90, 90, 100), 'kick': (60, 60, 70), 'room_floor': (230, 215, 170), 'room_ceiling': (230, 230, 225),
            'room_wall': (250, 230, 170), 'room_light': (255, 245, 160), 'step': (170, 170, 165), 'entry_floor': (90, 90, 100), 'wall': (180, 170, 150), 'wall_tile': (150, 85, 60), 'stone': (60, 60, 70),
            'door_frame': (70, 50, 40), 'handrail': (210, 215, 225), 'pole_paint': (110, 120, 130), 'lamp_emit': (255, 190, 90), 'grating': (40, 40, 40), 'manhole': (60, 40, 30),
            'stone_cap': (120, 120, 125), 'furniture': (160, 120, 80)}
FILL_ROLES = {'road', 'gutter', 'curb', 'sidewalk', 'tactile', 'boundary', 'setback', 'inuhashiri', 'soil'}


def find_font(Fnt, size):
    for p in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if os.path.exists(p):
            try:
                return Fnt.truetype(p, size), ('CJK' in p)
            except Exception:
                pass
    return Fnt.load_default(), False


def section_data(foundation, found_mat='conc', setback=0.0, kind='glass', seed=3):
    kw = dict(foundation=foundation, found_mat=found_mat, setback=setback, night=True)
    md, I, lamps, P = sk.build_geometry('tokyo246', None, 40.0, 1, 'near', **kw)
    H1 = G1 - I['base_z']
    fmd, FI = bb.build_geometry(kind, 12.0, H1, seed, 'near', depth=BD, base_z=I['base_z'], gl=I['gl'], night=True)
    return md, I, fmd, FI


def draw_section(path):
    Image, ImageDraw, ImageFont = get_pil()
    f_s, ja = find_font(ImageFont, 15); f_m, _ = find_font(ImageFont, 19); f_l, _ = find_font(ImageFont, 26)
    L = (lambda j, e: j) if ja else (lambda j, e: e)
    W_, H_ = 1700, 2400
    img = Image.new('RGB', (W_, H_), (245, 243, 238)); d = ImageDraw.Draw(img)
    d.text((24, 14), L('道路の断面キット: 車道の中心から1階の壁まで(切り口。マットなベタ塗り・寸法はm)', 'street_kit cross-section: road centre to ground-floor wall (m)'), fill=(20, 20, 20), font=f_l)
    # ---- 全体
    md, I, fmd, FI = section_data('low')
    x0 = 15.0                                      # 木の桝を通る位置
    seg = slice_md(md, x0)
    pitch = FI['dims']['mull_pitch']; xl = -6.0 + pitch * 0.5
    seg += slice_md(fmd, xl, dy=I['wall_plane'] + BD / 2)
    sc_ = 62.0; ox, oz = 40, 640                  # y=0 -> ox, z=0 -> oz (px)
    P_ = lambda y, z: (ox + y * sc_, oz - z * sc_)
    d.rectangle([ox - 20, 60, ox + 22 * sc_ + 20, 700], outline=(180, 180, 180))
    d.text((ox, 66), L('全体(縮尺 1m=62px)。左=道路の中心線。縁石・歩道・植樹桝・柵・基礎・1階(ガラス張り)', 'overall (62px/m). left = road centre line'), fill=(60, 60, 60), font=f_s)
    d.line([P_(-0.5, 0), P_(22, 0)], fill=(200, 200, 200), width=1)
    for (y0, z0, y1, z1, role) in seg:
        if role in FILL_ROLES and abs(y1 - y0) > 1e-4 and abs(z1 - z0) < 0.6 * abs(y1 - y0):
            a, b = P_(y0, z0), P_(y1, z1)
            d.polygon([a, b, P_(y1, -0.45), P_(y0, -0.45)], fill=tuple(int(c * 0.55 + 255 * 0.45) for c in ROLE_COL.get(role, (150, 150, 150))))
    # 基礎は info の値から塗る(切り口は壁の表面だけなので)
    fy0, fy1 = I['found_front'], I['wall_plane']
    d.rectangle([P_(fy0, I['found_top'])[0], P_(fy0, I['found_top'])[1], P_(fy1 + 0.02, I['gl'])[0], P_(fy1, I['gl'])[1]], fill=ROLE_COL[I['found_role']])
    for (y0, z0, y1, z1, role) in seg:
        d.line([P_(y0, z0), P_(y1, z1)], fill=ROLE_COL.get(role, (120, 120, 120)), width=1 if role == 'leaf' else 3)
    # 寸法(全体)
    def hd(y0, y1, z, text, dz=0, col=(30, 30, 160)):
        a, b = P_(y0, z), P_(y1, z)
        d.line([a, b], fill=col, width=1); d.line([(a[0], a[1] - 5), (a[0], a[1] + 5)], fill=col); d.line([(b[0], b[1] - 5), (b[0], b[1] + 5)], fill=col)
        d.text(((a[0] + b[0]) / 2 - len(text) * 4.2, a[1] - 22 - dz), text, fill=col, font=f_s)
    hd(0, I['half_road'], -0.65 + 0.0, L('車道 %.2fm(%d車線)' % (I['half_road'], I['lanes']), 'carriageway %.2fm' % I['half_road']))
    hd(I['half_road'], I['curb_o'], -0.65, L('路肩 0.30', 'gutter 0.30'), dz=24)
    hd(I['sidewalk_o0'], I['boundary_o'], 5.0 * 0 + 6.2, L('歩道 %.1fm(セミフラット)' % I['sidewalk_w'], 'sidewalk %.1fm' % I['sidewalk_w']))
    hd(I['boundary_o'], I['wall_plane'], 7.4, L('境界〜壁 %.2f' % (I['wall_plane'] - I['boundary_o']), 'boundary-wall %.2f' % (I['wall_plane'] - I['boundary_o'])))
    hd(0, I['wall_plane'], 8.6, L('道路中心から壁面まで %.2fm' % I['wall_plane'], 'centre to wall %.2fm' % I['wall_plane']))
    d.text((P_(I['wall_plane'] + 0.4, 4.7)[0], P_(0, 4.7)[1]), L('1階: ガラス張り(下枠・縦桟・腰板・看板帯)', '1F glass storefront'), fill=(60, 60, 60), font=f_s)
    # ---- 基礎3種(拡大)
    types = [('low', 'conc', 0.0, 'A: 低い立ち上がり GL+0.20(推0.15〜0.3)', 'A: low riser GL+0.20'),
             ('wainscot', 'stone', 0.0, 'B: 化粧の腰(石) GL+0.70(推0.45〜0.9)', 'B: stone wainscot GL+0.70'),
             ('flush', 'conc', 1.5, 'C: 歩道と同じ高さ 段差2cm・セットバック1.5m', 'C: flush, 2cm step, setback 1.5m')]
    kinds = ['glass', 'tile', 'entrance']
    zs = 330.0; top = 720; PH = 560
    for n, (fd, fm, sbk, ja_t, en_t) in enumerate(types):
        md, I, fmd, FI = section_data(fd, fm, sbk, kinds[n])
        seg = slice_md(md, 15.0)
        y_lo = I['curb_o'] - 0.35; y_hi = I['wall_plane'] + 1.25; scz = min(1500.0 / (y_hi - y_lo), 200.0)
        ox2 = 60; oz2 = top + n * PH + 520; scx = scz; scz = scz * 2.5
        Q_ = lambda y, z: (ox2 + (y - y_lo) * scx, oz2 - z * scz)
        d.rectangle([ox2 - 30, top + n * PH - 8, ox2 + 1580, top + n * PH + PH - 20], outline=(180, 180, 180))
        d.text((ox2 - 20, top + n * PH - 4), L(ja_t, en_t) + L('   + 1階 %s   (縦だけ2.5倍に拡大)' % kinds[n], '   + ground floor %s   (vertical x2.5)' % kinds[n]), fill=(20, 20, 20), font=f_m)
        d.line([Q_(y_lo, 0), Q_(y_hi, 0)], fill=(215, 215, 215), width=1)
        for (y0, z0, y1, z1, role) in seg:
            if role in FILL_ROLES and abs(y1 - y0) > 1e-4 and abs(z1 - z0) < 0.6 * abs(y1 - y0):
                d.polygon([Q_(y0, z0), Q_(y1, z1), Q_(y1, -0.12), Q_(y0, -0.12)], fill=tuple(int(c * 0.55 + 255 * 0.45) for c in ROLE_COL.get(role, (150, 150, 150))))
        fy0, fy1 = I['found_front'], I['wall_plane']
        if fd != 'flush':
            d.rectangle([Q_(fy0, I['found_top'])[0], Q_(fy0, I['found_top'])[1], Q_(fy1, I['gl'])[0], Q_(fy1, I['gl'])[1]], fill=ROLE_COL[I['found_role']])
        seg += slice_md(fmd, -6.0 + FI['dims'].get('mull_pitch', 1.5) * 0.5 if kinds[n] == 'glass' else (-4.0 if kinds[n] == 'tile' else 0.0), dy=I['wall_plane'] + BD / 2)
        for (y0, z0, y1, z1, role) in seg:
            if Q_(y1, z1)[0] < ox2 - 5 or Q_(y0, z0)[0] > ox2 + 1560 or max(z0, z1) > 0.98 and role not in ('fence_dark', 'fence_galv'):
                continue
            d.line([Q_(y0, z0), Q_(y1, z1)], fill=ROLE_COL.get(role, (120, 120, 120)), width=3)
        def vd(y, z0, z1, text, dx=0, col=(160, 30, 30)):
            a, b = Q_(y, z0), Q_(y, z1)
            d.line([a, b], fill=col, width=1); d.line([(a[0] - 5, a[1]), (a[0] + 5, a[1])], fill=col); d.line([(b[0] - 5, b[1]), (b[0] + 5, b[1])], fill=col)
            d.text((a[0] + 8 + dx, (a[1] + b[1]) / 2 - 8), text, fill=col, font=f_s)
        def hd2(y0, y1, z, text, up=24, col=(30, 30, 160)):
            a, b = Q_(y0, z), Q_(y1, z)
            d.line([a, b], fill=col, width=1); d.line([(a[0], a[1] - 5), (a[0], a[1] + 5)], fill=col); d.line([(b[0], b[1] - 5), (b[0], b[1] + 5)], fill=col)
            d.text((a[0] + 2, a[1] - up), text, fill=col, font=f_s)
        vd(I['curb_o'] + 0.09, 0, I['curb_h'], L('縁石 0.15(歩道面から0.10)', 'curb 0.15 (0.10 above walk)'), dx=-60 - 40)
        vd(I['sidewalk_o0'] + 0.35, 0, I['sw_h'], L('歩道面 +0.05', 'walk +0.05'), dx=30)
        hd2(I['curb_o'], I['sidewalk_o0'], 1.0, L('縁石厚 0.18', 'curb 0.18'), up=24)
        hd2(I['tactile'][0], I['tactile'][1], 0.62, L('点字 0.30', 'tactile 0.30'), up=22, col=(150, 110, 0))
        hd2(I['sidewalk_o0'], I['tactile'][0], 0.42, L('縁石から0.6m', '0.6m from curb'), up=22, col=(150, 110, 0))
        hd2(I['pit'][0], I['pit'][1], 0.42, L('植樹桝 0.9', 'tree pit 0.9'), up=22, col=(40, 120, 40))
        hd2(I['boundary_o'], I['boundary_o'] + 0.12, 0.62, L('境界 0.12', 'boundary'), up=22)
        vd(I['sidewalk_o0'] + 0.12 - 0.05, I['sw_h'], I['sw_h'] + 0.9, L('柵(ガードパイプ) 上端0.9', 'guard pipe top 0.9'), dx=-70)
        if fd != 'flush':
            vd(I['found_front'] - 0.1, I['gl'], I['found_top'], L('基礎 +%.2f' % (I['found_top'] - I['gl']), 'found +%.2f' % (I['found_top'] - I['gl'])), dx=-110)
            hd2(I['wall_plane'] - 0.45 - I['found_prot'], I['wall_plane'] - I['found_prot'], I['gl'] - 0.06, L('犬走り 0.45', 'apron 0.45'), up=-8)
        else:
            vd(I['found_front'] - 0.05, I['gl'], I['found_top'], L('段差 0.02', 'step 0.02'), dx=-90)
            hd2(I['boundary_o'] + 0.12, I['wall_plane'], 0.62, L('セットバック 1.5', 'setback 1.5'), up=22)
        d.text((Q_(I['wall_plane'] + 0.15, 0.55)[0], Q_(0, 0.55)[1]), L('壁面 ' + ('GL+%.2f' % (I['base_z'] - I['gl'])), 'wall base'), fill=(80, 80, 80), font=f_s)
        # 柵(0.9m)の凡例
    img.save(path)
    log('section saved', path, img.size, 'ja' if ja else 'en')


# ------------------------------------------------------------------ サンプル(書き出し)
SAMPLE_DEFS = [
    ('S1_low_glass', dict(foundation='low'), [('glass', 12.0)]),
    ('S2_wainscot_tile', dict(foundation='wainscot', found_mat='stone'), [('tile', 12.0)]),
    ('S3_flush_entrance', dict(foundation='flush', setback=1.5), [('entrance', 12.0)]),
    ('S4_low_shutter_parking', dict(foundation='low'), [('shutter', 6.0), ('parking', 6.0)]),
]


def make_sample(name, kw, spec, night, x_off, detail='near', collection=None):
    L = 30.0
    _, I0, _, _ = sk.build_geometry('tokyo246', None, L, 1, 'far', night=night, **kw)
    bz, gl, wp = I0['base_z'], I0['gl'], I0['wall_plane']; H1 = G1 - bz
    cx = L / 2
    row = bb.make_ground_floor_row(spec, H1, 8.0, bz, gl, seed=11, detail=detail, night=night, position=(x_off + cx, wp + 4.0), name=name + '_f1', collection=collection)
    inf = bb.floor_info(row)
    gaps = [(cx + a, cx + b) for a, b in inf['gaps_x']]; drives = [(cx + a, cx + b) for (a, b), k in zip(inf['gaps_x'], inf['gaps_kind']) if k == 'parking']
    kw2 = dict(kw); kw2.update(night=night, found_gaps=gaps, driveways=drives)
    st = sk.make_street_section('tokyo246', None, L, 1, detail, name=name + '_street', collection=collection, **kw2)
    st.location = (x_off, 0, 0)
    return st, row


def samples_part():
    sc = scene_basic('night', (640, 640)); out = {}
    colls = []
    for i, (name, kw, spec) in enumerate(SAMPLE_DEFS):
        c = bpy.data.collections.new(name); sc.collection.children.link(c); colls.append(c)
        st, row = make_sample(name, kw, spec, True, i * 40.0, 'near', c)
        out[name] = dict(street=sk.street_info(st), floor=bb.floor_info(row))
    json.dump(out, open(os.path.join(OUT, 'catalog_info.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, 'street_samples.blend'))
    log('saved blend')
    # 見本画像(夕方) ふつうの目線の斜め
    for i, (name, kw, spec) in enumerate(SAMPLE_DEFS):
        sc2 = scene_basic('day', (800, 600))
        st, row = make_sample(name, kw, spec, False, 0.0, 'near')
        I = sk.street_info(st)
        set_cam(sc2, (6.5, I['sidewalk_o0'] + 2.5, 1.7), (16.0, I['wall_plane'], 1.9), 20)
        render(sc2, 'sample_%s' % name)
        sc3 = scene_basic('night', (800, 600))
        st, row = make_sample(name, kw, spec, True, 0.0, 'near')
        set_cam(sc3, (6.5, I['sidewalk_o0'] + 2.5, 1.7), (16.0, I['wall_plane'], 1.9), 20)
        render(sc3, 'sample_night_%s' % name)
    # glb(材質を単純化)。 .blend 保存のあと、別シーンで1つずつ
    for i, (name, kw, spec) in enumerate(SAMPLE_DEFS):
        fresh((640, 640))
        st, row = make_sample(name, kw, spec, True, 0.0, 'near')
        sk.simplify_materials(st); bb.simplify_materials(row)
        for o in bpy.data.objects:
            o.select_set(False)
        objs = [st, row] + list(st.children_recursive) + list(row.children_recursive)
        for o in objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = st
        bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, 'sample_%s.glb' % name), export_format='GLB', use_selection=True)
    log('glb done')


def main():
    t0 = time.time()
    steps = []
    if PART in ('all', 'section'):
        steps.append(('01', lambda: draw_section(os.path.join(OUT, '01_section.png'))))
    if PART in ('all', 'views'):
        steps += [('02', lambda: shot_eye('02_eye_dusk', 'day')), ('03', lambda: shot_eye('03_eye_night', 'night')),
                  ('04', lambda: shot_aerial('04_aerial_dusk', 'day')), ('05', lambda: shot_aerial('05_aerial_night', 'night')),
                  ('08', lambda: shot_eye2('08_eye2_night', 'night')),
                  ('09', lambda: shot_aerial2('09_aerial_oblique_dusk', 'day')), ('10', lambda: shot_aerial2('10_aerial_oblique_night', 'night'))]
    if PART in ('all', 'before'):
        steps += [('06', lambda: shot_before('06_before_dusk', 'day')), ('07', lambda: shot_before('07_before_night', 'night'))]
    for key, fn in steps:
        if ONLY and key not in ONLY:
            continue
        try:
            fn()
        except Exception as e:
            import traceback; traceback.print_exc(); log('ERR', key, repr(e))
    if PART in ('all', 'samples'):
        try:
            samples_part()
        except Exception as e:
            import traceback; traceback.print_exc(); log('ERR samples', repr(e))
    log('total %.0fs' % (time.time() - t0))


main()
