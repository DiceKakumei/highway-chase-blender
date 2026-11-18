"""続編「打ち合い」(15秒)の音。 python tools/audio_sequel.py -> assets/audio_sequel.wav (48k stereo, 15 s, -14 LUFS)
前作と同じアシッド(A minor)の上に、エンジン2台(速度に追従)、銃声、薬莢、着弾、追い越し、後輪ロックのスキッド音、止まった後の静けさ。
ライブラリ: ~/video-refs/mg-styles-ja/lib/audio (mgaudio)。前作の audio.py と同じ使い方。"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIB = os.path.expanduser('~/video-refs/mg-styles-ja/lib/audio')
sys.path.insert(0, LIB)
sys.path.insert(0, os.path.join(ROOT, 'chase'))
import mgaudio as mg
from mgaudio import recipes, sfx, osc as O, filters as F
import sequel as SQ

SR = 48000
DUR = SQ.DUR
T = lambda d: np.arange(int(round(d * SR))) / SR
ramp = lambda t, pts: np.interp(t, [p[0] for p in pts], [p[1] for p in pts])
norm = lambda y, pk=.9: y * (pk / (np.max(np.abs(y)) + 1e-9))
CUTS = SQ.CUTS
SLIDE = SQ.SLIDE_T0

m = recipes.acid(duration=DUR, key='A', mode='minor', hits=tuple(CUTS), drop=2.0, build=1.0, outro=14.0)
print(m.form.summary())
S = m.sfx

# ---- 速度(エンジンとスキッド音に使う)
tt = T(DUR)
vA = np.interp(tt, SQ.TS + 0.0, SQ.tables('A')['v'])
vB = np.interp(tt, SQ.TS + 0.0, SQ.tables('B')['v'])

# ---- カット前のウーッシュ
for i, c in enumerate(CUTS):
    S(sfx.whoosh(0.35, 'swoosh', direction=1 if i % 2 == 0 else -1, peak=.5, doppler=.3, seed=110 + i), at=c - 0.10, gain=-9, pan=0.3 if i % 2 == 0 else -0.3)


# ---- エンジン
def engine(base, rpm, cutoff, amp, seed):
    fq = base * rpm * (1 + .004 * np.sin(2 * np.pi * 7.3 * tt + seed))
    y = O.saw(fq) * .5 + O.saw(fq * 1.007) * .45 + O.sine(fq * .5) * .8 + O.sine(fq * 2) * .15
    y *= .8 + .2 * np.sin(np.cumsum(2 * np.pi * fq * .75) / SR)
    y = F.svf(y, cutoff, .9, 'lp')
    return y * amp


fade_in = np.clip(tt / 0.8, 0, 1)
rpmA = .55 + .5 * (vA / 46.0)
rpmB = .55 + .55 * (vB / 46.0)
rpmB = rpmB + 0.25 * np.exp(-((tt - 10.6) / 0.5) ** 2)                 # 追い越しの前に吹かす
ampA = np.interp(tt, [0, 1, 9.0, 12.0, 14.5, 15.0], [.55, .6, .6, .8, .35, .12])
ampB = np.interp(tt, [0, 1, 9.0, 11.5, 12.2, 14.5, 15.0], [.6, .7, .7, 1.0, .9, .25, .15])
cutA = np.interp(tt, [0, 5, 8, 12, 15], [1300, 1200, 800, 700, 400])
cutB = np.interp(tt, [0, 5, 10, 12, 15], [1100, 1300, 1700, 1500, 500])
S(norm(engine(82.0, rpmA, cutA, ampA, 0.0) * fade_in), at=0.0, gain=-18, pan=0.0)
S(norm(engine(68.0, rpmB, cutB, ampB, 1.7) * fade_in), at=0.0, gain=-17, pan=-0.2)


# ---- 銃声
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
# 銃を抜く音(上着のこすれ、金属のかちゃ)
u = T(0.35); sl = F.bpf(np.random.default_rng(5).standard_normal(len(u)), 900, 4500) * np.exp(-((u - .18) / .09) ** 2)
S(norm(sl), at=SQ.T_CHEST - 0.15, gain=-17, pan=0.2)
S(sfx.click('soft'), at=SQ.T_DRAWN - 0.1, gain=-13, pan=0.2)
# 弾切れ(スライドが止まるかちゃ)と、しまう音
S(sfx.click('soft'), at=SQ.T_HOLSTER0 - 0.02, gain=-12, pan=0.2)
S(sfx.click('soft'), at=SQ.T_HOLSTER1 - 0.05, gain=-14, pan=0.2)

# ---- 追い越し: 近づく → 通過(ドップラー)
tp = 11.9
S(sfx.whoosh(0.9, 'heavy', direction=1, peak=.6, doppler=.6, seed=150), at=tp - 0.55, gain=-4, pan=0.4)

# ---- 横滑り: 後輪ロックの音(ゴムが滑る摩擦音)。速度に追従。止まると消える
ts_ = T(SQ.SLIDE_DUR + 0.5); sv = np.interp(ts_ + SLIDE, tt, vB) / 35.0
r = np.random.default_rng(77); nz = r.standard_normal(len(ts_))
fc = 1100 + 2600 * sv
sq = F.svf(nz, fc, 2.2, 'bp') * np.clip(sv * 1.6, 0, 1) ** 0.7 * (1 + .35 * np.sin(2 * np.pi * 41 * ts_))
whine = O.sine(600 + 1100 * sv) * .10 * sv ** 1.2 * (1 + .5 * np.sin(2 * np.pi * 6.2 * ts_))
att = np.clip(ts_ / 0.06, 0, 1)
S(norm((sq + whine) * att), at=SLIDE, gain=-8, pan=0.0)
# Aの後輪に命中: バーストの音(重い破裂 + シュー)と、リムが路面をこする金属音(速度に追従)
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
# 止まった瞬間の低い余韻と、風
S(sfx.boom(1.6, f0=42.0), at=SLIDE + SQ.SLIDE_DUR - 0.1, gain=-12)
wind = F.lpf(np.random.default_rng(79).standard_normal(len(tt)), 900) * np.interp(tt, [0, 12, 14.5, 15], [.0, .0, .25, .12])
S(norm(wind), at=0.0, gain=-26, pan=0.0)

# ---- 緊張(近づくほど速くなる刻み)。Bが追い越す瞬間まで高まり、ロックで途切れる
close = np.clip(1.0 - np.abs(np.interp(tt, SQ.TS + SQ.TOFF * 0, SQ.tables('B')['z'] - SQ.tables('A')['z'])) / 22.0, 0.0, 1.0)
rate = 1.6 + (12.0 - 1.6) * close ** 1.5
phi = np.cumsum(rate) / SR; frac = phi % 1.0
fq = np.where(np.floor(phi) % 2 == 0, 110.0, 116.54)
env = np.exp(-frac * 9.0) * (1 - np.exp(-frac * 500.0))
osc = O.saw(fq) * .6 + O.saw(fq * 2.003) * .25
osc = F.svf(osc, 500 + 3500 * close ** 1.2, 1.3, 'lp')
gate = np.clip((SLIDE + 0.1 - tt) / 0.2, 0, 1)
ost = np.tanh(3.0 * osc) * env * (0.25 + 0.75 * close ** 0.7) * gate * fade_in
S(norm(ost), at=0.0, gain=-14, pan=0.0)
# ロックの瞬間の一撃
S(sfx.impact('cinematic', size=.9), at=SLIDE, gain=-5)
# 音楽は、止まる少し前に切る(静けさで終わる)
m.group('music').region('mute', 14.15, 0.85)

out = os.path.join(ROOT, 'assets', 'audio_sequel.wav')
res = m.export(out, lufs=-14, tp=-3.0)
print({k: res[k] for k in ('duration', 'lufs', 'true_peak', 'warnings') if k in res})
import soundfile as sf
x, sr = sf.read(out)
t = np.arange(len(x)) / sr
g = np.clip(t / 0.8, 0, 1); g = g * g * (3 - 2 * g)                    # 映像の暗転明けと同じ(0.8秒)
sf.write(out, x * (g[:, None] if x.ndim == 2 else g), sr, subtype='PCM_24')
print('written', out, len(x) / sr, 's')
