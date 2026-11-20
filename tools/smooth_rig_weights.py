"""疑似リグの重み(assets/<名前>/rig_weights.npz)を、メッシュの隣り合う頂点どうしで滑らかにする。
原因: make_rig_weights.py の重みは、テクスチャ色と位置から頂点ごとに決めるため、隣の頂点(3cm未満)で重みが正反対になる所が約3,700辺ある。
      ライダーが揺れると(胴体のロール・ハンドルの切れ)、その辺が最大10倍以上に伸び縮みして、テクスチャが乱れて見える。
やること: 位置が同じ頂点(UVの継ぎ目で複製された頂点)は1つにまとめ、辺でつながる頂点の重みの平均へ寄せる操作を何回か繰り返す(行の合計は1のまま)。
usage: python tools/smooth_rig_weights.py [bike_A bike_B] [--iters 40] [--lam 0.5]   元の重みは rig_weights_raw.npz に1回だけ保存する(上書きしない)"""
import json, os, sys, shutil
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(os.path.join(ROOT, 'vehicles.json'), encoding='utf-8'))


def faces_of(name):
    c = CFG[name]
    sc = trimesh.load(os.path.join(ROOT, 'assets', name, c.get('body_file', '2_nowheel.glb')), force='scene', process=False)
    return np.asarray(list(sc.geometry.values())[0].faces)


def smooth(P, W, F, iters, lam):
    key = np.round(P * 1e4).astype(np.int64)
    _, uid, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel(); n = len(uid)
    Wu = np.zeros((n, W.shape[1]), np.float64); cnt = np.zeros(n)
    np.add.at(Wu, inv, W); np.add.at(cnt, inv, 1); Wu /= cnt[:, None]
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); a = inv[E[:, 0]]; b = inv[E[:, 1]]
    m = a != b; a = a[m]; b = b[m]
    pair = np.unique(np.sort(np.stack([a, b], 1), axis=1), axis=0); a, b = pair[:, 0], pair[:, 1]
    deg = np.bincount(np.r_[a, b], minlength=n).astype(np.float64)
    for _ in range(iters):
        S = np.zeros_like(Wu); np.add.at(S, a, Wu[b]); np.add.at(S, b, Wu[a])
        Wu = (1 - lam) * Wu + lam * S / np.maximum(deg, 1)[:, None]
    out = Wu[inv]; out /= np.maximum(out.sum(1, keepdims=True), 1e-9)
    return out.astype(np.float32)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    iters = int(sys.argv[sys.argv.index('--iters') + 1]) if '--iters' in sys.argv else 40
    lam = float(sys.argv[sys.argv.index('--lam') + 1]) if '--lam' in sys.argv else 0.5
    args = [a for a in args if a not in (str(iters), str(lam))]
    for name in (args or ['bike_A', 'bike_B']):
        d = os.path.join(ROOT, 'assets', name)
        raw = os.path.join(d, 'rig_weights_raw.npz')
        if not os.path.exists(raw):
            shutil.copy(os.path.join(d, 'rig_weights.npz'), raw)
        z = np.load(raw); P = z['pos']; W = z['W'].astype(np.float32)
        W2 = smooth(P, W, faces_of(name), iters, lam)
        np.savez_compressed(os.path.join(d, 'rig_weights.npz'), pos=P, W=W2.astype(np.float16))
        print(name, len(P), 'verts smoothed x%d lam %.2f' % (iters, lam))
