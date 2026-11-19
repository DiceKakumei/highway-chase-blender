"""追跡(30秒)+続編(15秒)の全体45秒の音を、1本のアシッドとして作る。 python tools/audio_full45.py [--stage render|compose] -> assets/audio_full45.wav
追跡と続編は1繋ぎのシーン。暗転・無音・BGMの切れ目を入れない(ユーザー指示 2026-11-19)。

作り方(要点)
  1. 音楽: 追跡の音(audio_acid_v5.wav)は recipes.acid(duration=30, hits=追跡のカット, A minor) で作られていた(132.45 bpm、ドロップ13.477秒。
     キックのグリッドを音から実測して一致を確認)。同じテンポ・同じ拍・同じコード進行のまま、45秒の1本として作り直す
     (アウトロを44.0秒=続編の終わりへ移す。ツマミ[フィルタ]の曲線は28.869秒までは前作と同じ)。
  2. 続編の効果音(銃声・タイヤ・エンジンなど)は audio_sequel.py と同じ中身を、時刻+30秒で重ねる。続編の音楽(別テンポ139.9bpm)は使わない。
     続編の頭の0.8秒の明け(フェードイン)もやめる。
  3. 追跡の効果音(エンジン・カットの切れ目)は、前作の audio_acid_v5.wav を、暗転のフェードを割り戻して使い、
     つなぎ目の少し手前(拍に合わせた位置)から1.5拍かけて、作り直した音へクロスフェードする。
音楽が追跡と同じ曲なので、クロスフェードしても拍・音程がずれない。"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIB = os.path.expanduser('~/video-refs/mg-styles-ja/lib/audio')
sys.path.insert(0, LIB)
sys.path.insert(0, os.path.join(ROOT, 'chase'))
import mgaudio as mg
from mgaudio import recipes, sfx, osc as O, filters as F
from mgaudio.recipes import club as C
from mgaudio.recipes.base import Form, new_mix
import sequel as SQ

SR = 48000
OFF = 30.0                      # 続編の頭 = 全体の30.0秒
TOTAL = 45.0
CACHE = os.environ.get('FULL45_CACHE', os.path.join(ROOT, '_cache_full45'))
os.makedirs(CACHE, exist_ok=True)

CHASE_OUTRO = 26.84            # 前作の音のアウトロ(音から実測。拍グリッド上の8分音符)。ツマミの曲線を前作と同じにするために使う
SHIFT = 240                    # 作り直した音は、前作の音より5ms(240サンプル)遅れていたので、その分前へ詰める(音から実測。全区間で一定)
CHASE_HITS = (4.417, 6.0, 8.0, 9.625, 11.625, 13.0, 14.417, 16.0, 17.5, 19.0, 21.0, 23.417, 25.0)   # 追跡のカットの切れ目(追跡の音の作り方)
SEQ_CUTS = [OFF + c for c in SQ.CUTS]
PROG = 'i i bVI bVII'


def _chase_form():
    return Form(30.0, None, None, None, None, CHASE_HITS, None, 'A', 'minor', 'i i bVI bVII', chord_beats=4, octave=4,
                bpm_default=130, bpm_range=(124, 140))


_F0 = _chase_form()
CF_A = _F0.drop + 27 * _F0.beat       # 前作の音 → 作り直した音 のクロスフェード(1拍。どちらも拍の頭: ドロップ+27拍=25.708秒、+28拍=26.161秒)
CF_B = _F0.drop + 28 * _F0.beat


def make_form():
    """追跡の音と同じテンポ・同じ拍・同じコード進行の Form(45秒)。"""
    f0 = Form(30.0, None, None, None, None, CHASE_HITS, None, 'A', 'minor', PROG, chord_beats=4, octave=4,
              bpm_default=130, bpm_range=(124, 140))
    O_END = OFF + 14.0           # 続編の音楽のアウトロ(止まった姿の直前)
    f = Form(TOTAL, f0.bpm, f0.drop, f0.build, O_END, (), None, 'A', 'minor', PROG, chord_beats=4, octave=4,
             bpm_default=130, bpm_range=(124, 140))
    return f0, f


def acid45(f, chase_outro):
    """recipes.acid(club.py)と同じ中身。違いは、ツマミの曲線が追跡のアウトロ(28.869秒)の先も2200Hz前後で続くことだけ。"""
    m = new_mix(f, -20.0, 3.0, master=dict(warmth=1.0))
    g = m.grid
    D, S_, seq = C.D, C.S, C.seq
    LV = C.LV
    k = D.kit('house')
    B, Dp, Ou, E = f.build, f.drop, f.outro, f.end
    pat = C._ACID
    steps = []
    st = f.bar / 16
    t_start = f.bars(0.0, Ou)[0]
    nsteps = int(round((Ou - t_start) / st))
    for i in range(nsteps):
        t = t_start + i * st
        it = pat[i % 16]
        if it is None:
            steps.append(None)
            continue
        semi, fl = it
        root = f.bass_at(t + 1e-4, 2)
        steps.append((root + semi, 'A' in fl, 'S' in fl))
    steps.append((f.bass_at(Ou, 2), True, False))
    pts = [(0.0, 190), (B, 420), (Dp - 0.02, 2600), (Dp, 750), (Dp + (chase_outro - Dp) * 0.5, 1500), (chase_outro, 2200),
           (Ou, 2300), (E, 900)]
    cut = [(0.0 - t_start + a, v) for a, v in pts]
    ac = S_.acid_line(steps, f.bpm, t0=0.0, cutoff=260, res=0.86, env_mod=2.6, decay=0.24, drive=2.4, cutoff_curve=cut)
    tb = m.track('303', level=LV['lead'] - 0.5, sends={'delay': -12}, pan=0.05)
    tb.add(ac, t_start)
    tb.duck(by='kick', depth=4, release=0.1)
    kick = lambda v=1.0: D.kick('techno', v, decay=0.42, drive=3.0)
    kk = m.track('kick', level=LV['kick'])
    kk.loop(kick, 'x...x...x...x...', 0.0, B)
    kk.loop(kick, 'x...x...x...x...', Dp, Ou)
    kk.add(kick(1.0), Ou)
    m.track('clap', level=LV['clap'], sends={'room': -10}).loop(k['clap'], '....x.......x...', Dp, Ou)
    m.track('ohat', level=LV['ohat'] + 1).loop(lambda v: D.hat('open', v, decay=0.2), '..x...x...x...x.', 0.0, Ou)
    m.track('hat', level=LV['hat'], pan=0.2).loop(lambda v: D.hat('closed', v), 'xxxxxxxxxxxxxxxx', B, Ou, vel=0.55, vel_jitter=0.2)
    m.track('ride', level=LV['crash'], pan=-0.3).loop(lambda v: D.cymbal('ride', v, decay=1.2), 'x.x.x.x.x.x.x.x.', Dp, Ou, vel=0.6)
    if Dp - B > 0.3:
        m.track('snare', level=LV['snare'] - 1).hits(lambda v: D.snare('tight', v, decay=0.12),
                                                     seq.roll(B, Dp - 0.01, 8, 32, g, 0.2, 0.9))
    rs = m.track('stab', level=LV['chords'], sends={'hall': -9})
    rs.add(S_.stab(f.chord_at(Dp), 0.4, 1.0, 'rave'), Dp)
    rs.add(S_.stab(f.chord_at(Ou), 0.8, 1.0, 'rave'), Ou)
    C._transition(m, f, riser='noise', reverse=True)
    m['crash'].add(D.crash(0.8), Ou)
    return m


def add_sequel_sfx(m):
    """audio_sequel.py の効果音と同じ中身。時刻は +30秒。続編の頭のフェードインはなし。エンジンは追跡の終わり(25.7秒)から入る。"""
    S = lambda snd, at, **kw: m.sfx(snd, at=at + OFF, **kw)
    T = lambda d: np.arange(int(round(d * SR))) / SR
    norm = lambda y, pk=.9: y * (pk / (np.max(np.abs(y)) + 1e-9))
    DUR = SQ.DUR
    SLIDE = SQ.SLIDE_T0
    CUTS = SQ.CUTS
    tt = T(DUR)                                                         # 続編の時刻 τ(0〜15)
    vA = np.interp(tt, SQ.TS + 0.0, SQ.tables('A')['v'])
    vB = np.interp(tt, SQ.TS + 0.0, SQ.tables('B')['v'])
    for i, c in enumerate(CUTS):
        S(sfx.whoosh(0.35, 'swoosh', direction=1 if i % 2 == 0 else -1, peak=.5, doppler=.3, seed=110 + i), at=c - 0.10, gain=-9, pan=0.3 if i % 2 == 0 else -0.3)

    # エンジン(クロスフェードの頭から。追跡の終わりから続けて入れる)
    TAU0 = CF_A - OFF - 0.0
    ta = np.arange(int(round(TAU0 * SR)), int(round(DUR * SR))) / SR
    vA_ = np.interp(ta, SQ.TS + 0.0, SQ.tables('A')['v'])
    vB_ = np.interp(ta, SQ.TS + 0.0, SQ.tables('B')['v'])

    def engine(base, rpm, cutoff, amp, seed):
        fq = base * rpm * (1 + .004 * np.sin(2 * np.pi * 7.3 * ta + seed))
        y = O.saw(fq) * .5 + O.saw(fq * 1.007) * .45 + O.sine(fq * .5) * .8 + O.sine(fq * 2) * .15
        y *= .8 + .2 * np.sin(np.cumsum(2 * np.pi * fq * .75) / SR)
        y = F.svf(y, cutoff, .9, 'lp')
        return y * amp

    sm_ = lambda x: (lambda u: u * u * (3 - 2 * u))(np.clip(x, 0, 1))
    pre = sm_((ta - (CF_A - OFF)) / (CF_B - CF_A))                    # 前作の音(エンジン込み)が消えるのと入れ替わりに入る(つなぎ目で途切れさせない)
    rpmA = .55 + .5 * (vA_ / 46.0)
    rpmB = .55 + .55 * (vB_ / 46.0)
    rpmB = rpmB + 0.25 * np.exp(-((ta - 10.6) / 0.5) ** 2)
    ampA = np.interp(ta, [0, 1, 9.0, 12.0, 14.5, 15.0], [.55, .6, .6, .8, .35, .12])
    ampB = np.interp(ta, [0, 1, 9.0, 11.5, 12.2, 14.5, 15.0], [.6, .7, .7, 1.0, .9, .25, .15])
    cutA = np.interp(ta, [0, 5, 8, 12, 15], [1300, 1200, 800, 700, 400])
    cutB = np.interp(ta, [0, 5, 10, 12, 15], [1100, 1300, 1700, 1500, 500])
    S(norm(engine(82.0, rpmA, cutA, ampA, 0.0) * pre), at=TAU0, gain=-18, pan=0.0)
    S(norm(engine(68.0, rpmB, cutB, ampB, 1.7) * pre), at=TAU0, gain=-17, pan=-0.2)

    def gunshot(seed, size=1.0):
        t = T(0.7); r = np.random.default_rng(seed); n = r.standard_normal(len(t))
        crack = F.bpf(n, 1800, 7000) * np.exp(-t / 0.016) * (1 - np.exp(-t * 4000))
        body = F.lpf(n, 1100) * np.exp(-t / 0.055)
        thump = O.sine(np.full(len(t), 48.0) + 130 * np.exp(-t * 28)) * np.exp(-t / 0.085)
        tail = F.lpf(r.standard_normal(len(t)), 2400) * np.exp(-t / 0.34) * 0.16
        y = crack * 1.0 + body * 0.75 + thump * 1.0 + tail
        return norm(np.tanh(1.6 * y * size))

    def casing_ping(seed):
        t = T(0.25); r = np.random.default_rng(seed); f = r.uniform(3600, 5200)
        return norm(O.sine(np.full(len(t), f)) * np.exp(-t / 0.035) + 0.4 * O.sine(np.full(len(t), f * 1.62)) * np.exp(-t / 0.025))

    def ricochet(seed):
        t = T(0.3); r = np.random.default_rng(seed)
        f = 3200 * np.exp(-t * 4.0) + 1200
        y = O.sine(f) * np.exp(-t / 0.09) * 0.5 + F.bpf(r.standard_normal(len(t)), 2500, 8000) * np.exp(-t / 0.02) * (1 - np.exp(-t * 3000))
        return norm(y)

    for n, ft in enumerate(SQ.FIRES):
        S(gunshot(200 + n, 1.0 + 0.12 * ((n * 7) % 3)), at=ft, gain=-5 - (n % 2), pan=-0.2)
        S(casing_ping(300 + n), at=ft + 0.19 + 0.03 * (n % 3), gain=-24, pan=0.45)
        S(ricochet(400 + n), at=ft + 0.075, gain=-19, pan=0.15 if n % 2 else -0.15)
    u = T(0.35); sl = F.bpf(np.random.default_rng(5).standard_normal(len(u)), 900, 4500) * np.exp(-((u - .18) / .09) ** 2)
    S(norm(sl), at=SQ.T_CHEST - 0.15, gain=-17, pan=0.2)
    S(sfx.click('soft'), at=SQ.T_DRAWN - 0.1, gain=-13, pan=0.2)
    S(sfx.click('soft'), at=SQ.T_HOLSTER0 - 0.02, gain=-12, pan=0.2)
    S(sfx.click('soft'), at=SQ.T_HOLSTER1 - 0.05, gain=-14, pan=0.2)
    tp = 11.9
    S(sfx.whoosh(0.9, 'heavy', direction=1, peak=.6, doppler=.6, seed=150), at=tp - 0.55, gain=-4, pan=0.4)
    ts_ = T(SQ.SLIDE_DUR + 0.5); sv = np.interp(ts_ + SLIDE, tt, vB) / 35.0
    r = np.random.default_rng(77); nz = r.standard_normal(len(ts_))
    fc = 1100 + 2600 * sv
    sq = F.svf(nz, fc, 2.2, 'bp') * np.clip(sv * 1.6, 0, 1) ** 0.7 * (1 + .35 * np.sin(2 * np.pi * 41 * ts_))
    whine = O.sine(600 + 1100 * sv) * .10 * sv ** 1.2 * (1 + .5 * np.sin(2 * np.pi * 6.2 * ts_))
    att = np.clip(ts_ / 0.06, 0, 1)
    S(norm((sq + whine) * att), at=SLIDE, gain=-8, pan=0.0)
    HIT = SQ.HIT_T
    u = T(0.9); r_ = np.random.default_rng(91)
    burst = np.tanh(2.0 * (F.lpf(r_.standard_normal(len(u)), 1400) * np.exp(-u / 0.05) + O.sine(np.full(len(u), 60.0) + 90 * np.exp(-u * 30)) * np.exp(-u / 0.10)))
    hiss = F.bpf(r_.standard_normal(len(u)), 5000, 11000) * np.exp(-u / 0.35) * 0.35
    S(norm(burst + hiss), at=HIT + 0.01, gain=-6, pan=-0.2)
    tr = T(2.5); sa = np.interp(tr + HIT, tt, vA) / 28.0
    nzr = np.random.default_rng(92).standard_normal(len(tr))
    grind = F.svf(nzr, 3000 + 3500 * sa, 1.8, 'bp') * (0.6 + 0.4 * np.sin(2 * np.pi * 52 * tr)) * np.clip(sa * 1.6, 0, 1) ** 0.7 * np.clip(tr / 0.04, 0, 1)
    grind = grind + 0.25 * O.saw(np.full(len(tr), 180.0) * (0.6 + 0.6 * sa)) * np.clip(sa * 1.5, 0, 1) * np.clip(tr / 0.06, 0, 1)
    S(norm(grind), at=HIT, gain=-10, pan=-0.3)
    S(sfx.boom(1.6, f0=42.0), at=SLIDE + SQ.SLIDE_DUR - 0.1, gain=-12)
    wind = F.lpf(np.random.default_rng(79).standard_normal(len(tt)), 900) * np.interp(tt, [0, 12, 14.5, 15], [.0, .0, .25, .12])
    S(norm(wind), at=0.0, gain=-26, pan=0.0)
    close = np.clip(1.0 - np.abs(np.interp(tt, SQ.TS + SQ.TOFF * 0, SQ.tables('B')['z'] - SQ.tables('A')['z'])) / 22.0, 0.0, 1.0)
    rate = 1.6 + (12.0 - 1.6) * close ** 1.5
    phi = np.cumsum(rate) / SR; frac = phi % 1.0
    fq = np.where(np.floor(phi) % 2 == 0, 110.0, 116.54)
    env = np.exp(-frac * 9.0) * (1 - np.exp(-frac * 500.0))
    osc = O.saw(fq) * .6 + O.saw(fq * 2.003) * .25
    osc = F.svf(osc, 500 + 3500 * close ** 1.2, 1.3, 'lp')
    gate = np.clip((SLIDE + 0.1 - tt) / 0.2, 0, 1)
    ost = np.tanh(3.0 * osc) * env * (0.25 + 0.75 * close ** 0.7) * gate
    S(norm(ost), at=0.0, gain=-14, pan=0.0)
    S(sfx.impact('cinematic', size=.9), at=SLIDE, gain=-5)
    m.group('music').region('mute', OFF + 14.15, 0.85)                  # 音楽は止まる少し前に切る(静けさで終わる。続編の元の音と同じ)


def stage_render():
    f0, f = make_form()
    print('chase form', f0.summary())
    print('full form', f.summary())
    m = acid45(f, CHASE_OUTRO)
    add_sequel_sfx(m)
    y = m.master(lufs=-14.0, tp=-3.0)
    y = np.asarray(y)
    np.save(os.path.join(CACHE, 'new45.npy'), y)
    # 音楽だけ(後の突き合わせ用)
    mm = m.stems.get('group.music')
    if mm is not None:
        np.save(os.path.join(CACHE, 'music45.npy'), np.asarray(mm))
    print('rendered', y.shape)


def stage_compose(out_path):
    import soundfile as sf
    new = np.load(os.path.join(CACHE, 'new45.npy')).T                     # (n, 2)
    new = np.concatenate([new[SHIFT:], np.repeat(new[-1:], SHIFT, axis=0)], axis=0)
    orig, sr = sf.read(os.path.join(ROOT, 'assets', 'audio_acid_v5.wav'))
    assert sr == SR and len(orig) == 30 * SR
    n = int(TOTAL * SR)
    new = new[:n]
    # 音量合わせ: クロスフェード直前(21〜25.7秒)の 120〜2000Hz で、前作の音に合わせる(±1.5dBに制限)
    from scipy.signal import butter, sosfilt
    sos = butter(4, [120, 2000], 'band', fs=SR, output='sos')
    a = sosfilt(sos, orig.mean(1))[21 * SR:int(CF_A * SR)]
    b = sosfilt(sos, new.mean(1))[21 * SR:int(CF_A * SR)]
    g = float(np.clip(np.sqrt((a @ a) / (b @ b)), 10 ** (-1.5 / 20), 10 ** (1.5 / 20)))
    print('level match gain', round(20 * np.log10(g), 2), 'dB')
    new = new * g
    out = new.copy()
    ia, ib = int(round(CF_A * SR)), int(round(CF_B * SR))
    w = np.zeros(n)
    w[ia:ib] = np.linspace(0.0, 1.0, ib - ia, endpoint=False)
    w[ib:] = 1.0
    out[:ib] = orig[:ib] * (1 - w[:ib, None]) + new[:ib] * w[:ib, None]
    out[ib:] = new[ib:]
    # 全体を -14 LUFS(統合)へ。ピークは -3 dBTP 以下に
    import pyloudnorm as pyln
    L = pyln.Meter(SR).integrated_loudness(out)
    gain = 10 ** ((-14.0 - L) / 20)
    pk = np.max(np.abs(out)) * gain
    if pk > 10 ** (-3.0 / 20):
        gain *= 10 ** (-3.0 / 20) / pk
    out = out * gain
    print('loudness before', round(L, 2), 'LUFS; applied', round(20 * np.log10(gain), 2), 'dB')
    fo = int(0.01 * SR)
    out[-fo:] *= np.linspace(1, 0, fo)[:, None]
    sf.write(out_path, out, SR, subtype='PCM_24')
    print('written', out_path, len(out) / SR, 's')


if __name__ == '__main__':
    st = 'render'
    if '--stage' in sys.argv:
        st = sys.argv[sys.argv.index('--stage') + 1]
    if st == 'render':
        stage_render()
    if st in ('render', 'compose'):
        stage_compose(os.path.join(ROOT, 'assets', 'audio_full45.wav'))
