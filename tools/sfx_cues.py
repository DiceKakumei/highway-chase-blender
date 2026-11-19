"""拳銃の効果音セット: 発砲時刻のリスト -> 音の時刻リスト(cue) -> 1本のWAV。 bpy非依存(numpy + scipy + soundfile)。

使い方:
    import sfx_cues as SC
    cues = SC.gun_cues(FIRES, total_rounds=6)                    # 近い耳(銃口のそば)
    cues = SC.gun_cues(FIRES, 6, camera_dist=25.0, bike_speed=30) # 25m後ろの耳。音速の遅れ(25/340秒)つき・遠い銃声
    cues += SC.handling_cues(t_chest=1.9, t_drawn=2.6, t_holster0=10.8, t_holster1=11.4)
    y = SC.render_cues(cues, duration=15.0)                       # (n, 2) float, ピーク-1 dBFS以下
    SC.write_tables('<音フォルダ>')                               # 銃_効果音表.json / 銃_取り扱い音表.json を作り直す

音の置き場: 環境変数 SFX_DIR > このファイルの隣の `音` > アセット庫/小道具/銃/音。WAVが無い音は合成で代用する(下の _synth)。
時刻は「発砲 t=0 = 撃針が雷管を打つ瞬間」からの相対秒。cue['t'] はその音の「主な山」(銃声のピーク・薬莢が路面に当たる瞬間)の絶対時刻で、
WAVの頭ではない(render_cues が頭の余白を引いて置く)。出典: リサーチ/2026-11-19_拳銃の機構と発砲サイクル_動く部品.md。"""
import os, json, math
import numpy as np
from scipy import signal
import soundfile as sf

SOUND_SPEED = 340.0
# ---- 発砲サイクル(0 = 撃針が雷管を打つ)。出典: 拳銃の機構と発砲サイクル 1-3, 1-4, 1-5
T_EJECT = 0.006          # 薬莢が排莢口から出る(5〜8 ms。推)
T_SLIDE_REAR = 0.010     # スライドが最後尾に当たる(9〜12 ms。★)。カシャッの「後退」側
T_SLIDE_CLOSE = 0.055    # スライドが前で止まる(45〜57 ms。★)。slide_cycle.wav は後退の45ms後に閉鎖音が入る
T_SLIDE_STOP = 0.035     # 弾切れ: スライドストップのカチッ(最後尾に着いてレバーが入る。聞こえる位置を銃声の直後に置く。推)
T_TRIGGER = -0.030       # 引き金のクリック(発火の少し前。推)
CASING_H = 1.2           # 排莢口の高さ[m](バイクに乗った手の高さ。指示値)
CASING_VZ0 = 1.9         # 薬莢の上向き初速[m/s](chase/fx.py の ej.z の平均 1.2〜2.6)
CASING_K = 2.6           # 空気抵抗(chase/fx.py と同じ)
CASING_E = 0.35          # 路面ではね返る反発係数(chase/fx.py と同じ)
GAIN = dict(gun_near=0.0, gun_far=-4.0, slide=-14.0, slide_stop=-10.0, casing=-18.0, trigger=-26.0,
            draw=-15.0, holster=-13.0)
PAN = dict(gun_near=-0.2, gun_far=0.0, slide=0.2, casing=0.35, trigger=0.2, draw=0.2, holster=0.2)


def casing_fall(h=CASING_H, vz0=CASING_VZ0, k=CASING_K, e=CASING_E, g=9.8, n_bounce=3, dt=0.0005):
    """薬莢の落下。排莢から路面に当たる時刻[秒]と当たる速さ[m/s]のリスト(最大 n_bounce 回)。chase/fx.py の casing_pos と同じ式。"""
    z, v, t, out = h, vz0, 0.0, []
    while len(out) < n_bounce and t < 3.0:
        v = v * (1 - k * dt) - g * dt
        z += v * dt; t += dt
        if z < 0.006:
            z = 0.006
            if v < -0.3: out.append((t, -v))
            v = -v * e
    return out


def casing_lag(t_fall, bike_speed, k=CASING_K):
    """薬莢が落ちた場所が、排莢位置のどれだけ後ろか[m]。走行中は薬莢の前進速度が空気抵抗で消え、路面に置いていかれる。"""
    return bike_speed * t_fall - bike_speed / k * (1 - math.exp(-k * t_fall))


# ---- 音ファイル
def _dir(sfx_dir=None):
    if sfx_dir: return sfx_dir
    if os.environ.get('SFX_DIR'): return os.environ['SFX_DIR']
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), '音')
    if os.path.isdir(here): return here
    return os.path.join('G:\\', 'マイドライブ', 'newCom', '1_アプリ開発', 'AIの作業場', 'アセット庫', '小道具', '銃', '音')


_cache = {}
def _anchor(name, x, sr):
    """cue の時刻に合わせる位置(音の主な山)。銃声・薬莢・スライド等は立ち上がり、抜く/しまうは最大の山(カチャッ)。"""
    m = np.abs(x).max(1) if x.ndim == 2 else np.abs(x)
    if name.startswith('draw'): return int(0.25 * sr + np.argmax(m[int(0.25 * sr):]))   # 後半の金属のカチャッ
    if name.startswith('holster'): return int(np.argmax(m))
    return int(np.argmax(m > m.max() * 0.5))


def load(name, sr=48000, sfx_dir=None):
    """-> (stereo float配列, 山の位置[サンプル])。WAVが無ければ合成。"""
    key = (name, sr, sfx_dir)
    if key in _cache: return _cache[key]
    p = os.path.join(_dir(sfx_dir), name)
    if os.path.exists(p):
        x, s0 = sf.read(p, always_2d=True)
        if s0 != sr: x = signal.resample_poly(x, sr, s0, axis=0)
    else:
        x = _synth(name, sr)
    if x.shape[1] == 1: x = np.repeat(x, 2, 1)
    _cache[key] = (x, _anchor(name, x, sr))
    return _cache[key]


# ---- 合成(WAVが無いときの代用。実録音に比べて薄い。audio_sequel.py の銃声と同じ作り: ノイズのクラック + 胴鳴り + 尾)
def _synth(name, sr=48000):
    r = np.random.default_rng(abs(hash(name)) % (1 << 31))
    T = lambda d: np.arange(int(d * sr)) / sr
    bp = lambda x, lo, hi: signal.sosfilt(signal.butter(2, [lo, hi], 'band', fs=sr, output='sos'), x)
    lp = lambda x, f: signal.sosfilt(signal.butter(2, f, 'low', fs=sr, output='sos'), x)
    nm = lambda y, pk=.7: y * (pk / (np.abs(y).max() + 1e-9))
    if name.startswith('gunshot'):
        t = T(1.2); n = r.standard_normal(len(t))
        y = bp(n, 1800, 7000) * np.exp(-t / .016) + lp(n, 1100) * np.exp(-t / .055) * .75 \
            + np.sin(2 * np.pi * np.cumsum(48 + 130 * np.exp(-t * 28)) / sr) * np.exp(-t / .085) + lp(r.standard_normal(len(t)), 2400) * np.exp(-t / .34) * .16
        y = np.tanh(1.6 * y)
        if 'far' in name: y = lp(y, 2600) * 0.6 + np.roll(lp(y, 1700), int(.09 * sr)) * .3
    elif name.startswith('casing'):
        t = T(.3); f = r.uniform(3600, 5200); y = np.zeros(len(t))
        for d, g in [(0, 1), (.2, .45), (.29, .2)]:
            i = int(d * sr); u = t[:len(t) - i]
            y[i:] += g * (np.sin(2 * np.pi * f * u) * np.exp(-u / .035) + .4 * np.sin(2 * np.pi * f * 1.62 * u) * np.exp(-u / .025))
    elif name.startswith('slide_stop') or name.startswith('trigger'):
        t = T(.03); y = bp(r.standard_normal(len(t)), 2500, 9000) * np.exp(-t / .004)
    elif name.startswith('slide'):
        t = T(.13); y = np.zeros(len(t))
        for d in (0.0, .045) if 'cycle' in name else (0.0,):
            i = int(d * sr); u = t[:len(t) - i]; y[i:] += bp(r.standard_normal(len(u)), 1500, 9000) * np.exp(-u / .008)
    elif name.startswith('draw') or name.startswith('holster'):
        t = T(.5); y = bp(r.standard_normal(len(t)), 900, 4500) * np.exp(-((t - .25) / .1) ** 2) * .5
        i = int(.4 * sr); u = t[:len(t) - i]; y[i:] += bp(r.standard_normal(len(u)), 2000, 9000) * np.exp(-u / .01)
    else:
        raise FileNotFoundError(name)
    return np.repeat(nm(y)[:, None], 2, 1)


# ---- 発砲 -> cue
def gun_cues(fire_times, total_rounds=None, camera_dist=None, listener=None, bike_speed=None, trigger=False, seed=0):
    """発砲時刻のリスト[秒] -> 音のcueのリスト(t順)。 cue = dict(t, file, gain_db, pan, note, kind)。
    total_rounds: 弾倉の弾数(省略=len(fire_times))。最後の1発だけ、スライドが後ろで止まる(閉鎖音なし + スライドストップのカチッ)。
    camera_dist: 銃口から耳(カメラ)までの距離[m]。指定すると、音速(340 m/s)の遅れ distance/340 を足す。
    listener: 'near'(銃口のそば。既定)か 'far'(後ろを走るバイク)。camera_dist>=12 なら自動で 'far'。
    bike_speed: 走行速度[m/s]。指定すると、薬莢が路面に落ちた場所が後ろへ流れた分(m)だけ、薬莢音を小さく・遅らせる。"""
    n = len(fire_times); total = n if total_rounds is None else total_rounds
    if listener is None: listener = 'far' if (camera_dist is not None and camera_dist >= 12) else 'near'
    dly = (camera_dist or 0.0) / SOUND_SPEED
    far = listener == 'far'
    rng = np.random.default_rng(seed)
    out = []
    for i, ft in enumerate(sorted(fire_times)):
        last = (i + 1) >= total
        out.append(dict(kind='gunshot', t=ft + dly, file='gunshot_far.wav' if far else 'gunshot_near.wav',
                        gain_db=GAIN['gun_far' if far else 'gun_near'], pan=PAN['gun_far' if far else 'gun_near'],
                        note='発砲%d/%d。t=発砲時刻%s' % (i + 1, total, ' + 音速の遅れ %.3f s' % dly if dly else '')))
        sg = GAIN['slide'] - (10 if far else 0)
        if last:
            out.append(dict(kind='slide_back', t=ft + T_SLIDE_REAR + dly, file='slide_back.wav', gain_db=sg, pan=PAN['slide'],
                            note='最後の1発: スライドは後退するだけで戻らない(閉鎖音なし)'))
            out.append(dict(kind='slide_stop', t=ft + T_SLIDE_STOP + dly, file='slide_stop.wav',
                            gain_db=GAIN['slide_stop'] - (12 if far else 0), pan=PAN['slide'], note='弾切れ: スライドストップが入る「カチッ」'))
        else:
            out.append(dict(kind='slide_cycle', t=ft + T_SLIDE_REAR + dly, file='slide_cycle.wav', gain_db=sg, pan=PAN['slide'],
                            note='スライド後退(+%.0f ms)と閉鎖(+%.0f ms)の「カシャッ」' % (T_SLIDE_REAR * 1e3, T_SLIDE_CLOSE * 1e3)))
        vz = CASING_VZ0 + rng.uniform(-0.6, 0.7)                      # 発砲ごとの初速のばらつき(1.2〜2.6)
        imp = casing_fall(vz0=vz)
        tf = imp[0][0]
        lag = casing_lag(tf, bike_speed) if bike_speed else 0.0
        cg = GAIN['casing'] - (10 * math.log10(max(lag, 1.5) / 1.5) if lag else 0.0)   # 後ろへ流れた距離の分だけ、距離減衰の半分(-10 dB/10倍)
        cg -= 10 if far else 0
        out.append(dict(kind='casing', t=ft + T_EJECT + tf + (dly + lag / SOUND_SPEED if camera_dist else 0.0),
                        file='casing_drop_%d.wav' % (1 + i % 3), gain_db=cg, pan=PAN['casing'],
                        note='薬莢が路面に当たる(落下%.2f s、初速vz=%.1f)。録音自体が2〜3回はねる。物理のはね返り時刻: %s%s' % (
                            tf, vz, ', '.join('%.2f' % (x[0] + T_EJECT) for x in imp), '。後ろへ%.1f m流れた分だけ小さく' % lag if lag else '')))
        if trigger:
            out.append(dict(kind='trigger', t=ft + T_TRIGGER + dly, file='trigger_click.wav', gain_db=GAIN['trigger'], pan=PAN['trigger'], note='引き金のクリック'))
    return sorted(out, key=lambda c: c['t'])


def handling_cues(t_chest=None, t_drawn=None, t_holster0=None, t_holster1=None, camera_dist=None):
    """銃を抜く・しまう音。draw.wav の最大の山(金属のカチャッ)を t_drawn - 0.05 に、holster.wav の山を t_holster1 - 0.02 に合わせる。
    (draw.wavは山の約0.4秒前から革・布のこすれが始まる = 上着の中へ手が入る T_CHEST 付近)。t_chest, t_holster0 は未使用(互換のため受け取る)。"""
    dly = (camera_dist or 0.0) / SOUND_SPEED
    out = []
    if t_drawn is not None:
        out.append(dict(kind='draw', t=t_drawn - 0.05 + dly, file='draw.wav', gain_db=GAIN['draw'], pan=PAN['draw'], note='銃を抜く(革・布のこすれ→構えるカチャッ)'))
    if t_holster1 is not None:
        out.append(dict(kind='holster', t=t_holster1 - 0.02 + dly, file='holster.wav', gain_db=GAIN['holster'], pan=PAN['holster'], note='銃をしまう(こすれ→ホルスターにおさまる音)'))
    return sorted(out, key=lambda c: c['t'])


# ---- 合成して1本に
def render_cues(cues, duration, sr=48000, sfx_dir=None, peak_dbfs=-1.0):
    """cue のリストを足し合わせて (n,2) の float 配列にする。左右は等パワーのパン。ピークが peak_dbfs を超えたら全体を下げる(クリップ防止)。"""
    y = np.zeros((int(round(duration * sr)), 2))
    for c in cues:
        x, a = load(c['file'], sr, sfx_dir)
        p = float(np.clip(c.get('pan', 0.0), -1, 1)); th = (p + 1) * math.pi / 4
        g = 10 ** (c.get('gain_db', 0.0) / 20)
        i0 = int(round(c['t'] * sr)) - a
        s0, e0 = max(0, i0), min(len(y), i0 + len(x))
        if e0 <= s0: continue
        seg = x[s0 - i0:e0 - i0]
        y[s0:e0, 0] += seg[:, 0] * g * math.cos(th) * math.sqrt(2)
        y[s0:e0, 1] += seg[:, 1] * g * math.sin(th) * math.sqrt(2)
    pk = np.abs(y).max(); lim = 10 ** (peak_dbfs / 20)
    if pk > lim: y *= lim / pk
    return y


# ---- 時刻表(JSON)。上の定数と同じ値から作る(定数を直したら write_tables を走らせ直す)
def write_tables(out_dir):
    imp = casing_fall()
    fire = {
        '説明': '起点の動き = 発砲 t=0(撃針が雷管を打つ瞬間)からの相対時刻[秒]。t は音の「主な山」の時刻で、WAVの頭ではない。gain_db は WAV(ピーク-3〜-9 dBFS)に足す値。pan は -1(左)〜+1(右)。',
        '出典': 'リサーチ/2026-11-19_拳銃の機構と発砲サイクル_動く部品.md 1-3(タイムライン), 1-4(薬莢), 1-5(スライドストップ)',
        '使い方': 'tools/sfx_cues.py の gun_cues(発砲時刻のリスト, 弾数) が、この表を絶対時刻に直して返す。render_cues で1本のWAVにする。',
        'cues': [
            dict(kind='gunshot', t=0.0, file='gunshot_near.wav', gain_db=GAIN['gun_near'], pan=PAN['gun_near'], speed_of_sound_delay=True,
                 note='銃口のそばの耳(ライダー側)。カメラまでの距離÷340 m/s を動画側で足す(gun_cues(camera_dist=) が足す)'),
            dict(kind='gunshot', t=0.0, file='gunshot_far.wav', gain_db=GAIN['gun_far'], pan=PAN['gun_far'], speed_of_sound_delay=True,
                 note='後ろを走るバイクの耳(12m以上は自動でこちら)。高域が落ち、高架の反射が尾に乗る'),
            dict(kind='slide_cycle', t=T_SLIDE_REAR, file='slide_cycle.wav', gain_db=GAIN['slide'], pan=PAN['slide'], speed_of_sound_delay=True,
                 note='スライド後退(+10 ms)→閉鎖(+55 ms)の小さな金属音。銃声に重なる。遠い耳ではさらに-10 dB'),
            dict(kind='casing', t=round(T_EJECT + imp[0][0], 3), file='casing_drop_1..3.wav', gain_db=GAIN['casing'], pan=PAN['casing'], speed_of_sound_delay=True,
                 note='薬莢が路面に当たる。排莢(+%.0f ms)+落下(高さ%.1f m・初速vz %.1f m/s・空気抵抗k=%.1f から %.2f s)。はね返りは %s。走行中は後ろへ流れるので小さく(bike_speed指定で距離減衰)' % (
                     T_EJECT * 1e3, CASING_H, CASING_VZ0, CASING_K, imp[0][0], ', '.join('+%.2f s' % (x[0] + T_EJECT) for x in imp))),
            dict(kind='slide_stop', t=T_SLIDE_STOP, file='slide_stop.wav', gain_db=GAIN['slide_stop'], pan=PAN['slide'], only_last_round=True, speed_of_sound_delay=True,
                 note='最後の1発だけ。スライドは戻らず後ろで止まる(slide_cycle の代わりに slide_back.wav を使い、閉鎖音を出さない)'),
            dict(kind='trigger', t=T_TRIGGER, file='trigger_click.wav', gain_db=GAIN['trigger'], pan=PAN['trigger'], optional=True,
                 note='引き金のクリック(任意。gun_cues(trigger=True))'),
        ]}
    hand = {
        '説明': '銃を抜く・しまう。動画側のイベント時刻(続編の chase/sequel.py の T_DRAWN, T_HOLSTER1)からの相対秒。t は音の最大の山の時刻。',
        'cues': [
            dict(kind='draw', base='T_DRAWN', t=-0.05, file='draw.wav', gain_db=GAIN['draw'], pan=PAN['draw'],
                 note='構えた瞬間の少し前に金属のカチャッ。その約0.4秒前から革・布のこすれ(T_CHEST=上着の中へ手が入る時刻の付近)'),
            dict(kind='holster', base='T_HOLSTER1', t=-0.02, file='holster.wav', gain_db=GAIN['holster'], pan=PAN['holster'],
                 note='銃がホルスターに収まる音。その約0.14秒前からこすれ'),
        ]}
    for name, d in (('銃_効果音表.json', fire), ('銃_取り扱い音表.json', hand)):
        json.dump(d, open(os.path.join(out_dir, name), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == '--write-tables': write_tables(sys.argv[2])
    else: print(__doc__)
