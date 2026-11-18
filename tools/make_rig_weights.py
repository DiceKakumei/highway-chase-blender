"""バイク(ライダー付き1枚メッシュ)の疑似リグの重みを作る。 usage: python tools/make_rig_weights.py [bike_A bike_B]
GLB を Blender と同じ変換でプロトタイプ空間(x=右, y=前, z=上, m)に置き、テクスチャ色と位置から chase/rig.py の重みを計算して
assets/<名前>/rig_weights.npz (pos, W) に保存する。render.py は頂点の位置で最近傍を引いて重みを写す(頂点の並びが違っても動く)。"""
import json, math, os, sys
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'chase'))
import rig  # noqa: E402

CFG = json.load(open(os.path.join(ROOT, 'vehicles.json'), encoding='utf-8'))


def load_proto(name):
    c = CFG[name]
    sc = trimesh.load(os.path.join(ROOT, 'assets', name, '2_nowheel.glb'), force='scene', process=False)
    V, C = [], []
    for nm in sc.graph.nodes_geometry:
        T, gname = sc.graph[nm]; g = sc.geometry[gname]
        v = trimesh.transformations.transform_points(np.asarray(g.vertices), T)
        im = np.asarray(g.visual.material.baseColorTexture.convert('RGB')); uv = np.asarray(g.visual.uv)
        h, w = im.shape[:2]
        x = np.clip((uv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(int); y = np.clip((1 - (uv[:, 1] % 1.0)) * (h - 1), 0, h - 1).astype(int)
        V.append(v); C.append(im[y, x])
    v = np.vstack(V); col = np.vstack(C)
    b = np.c_[v[:, 0], -v[:, 2], v[:, 1]]                       # glTF(Y上) -> Blender(Z上)
    rot = {'bike': math.pi, 'car': -math.pi / 2}[c['axes']]
    ca, sa = math.cos(rot), math.sin(rot)
    b = np.c_[ca * b[:, 0] - sa * b[:, 1], sa * b[:, 0] + ca * b[:, 1], b[:, 2]]
    lo, hi = b.min(0), b.max(0)
    s = c['L'] / (hi[1] - lo[1])
    b = (b - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])) * s
    b[:, 2] += c['body_z']
    return b.astype(np.float32), col


if __name__ == '__main__':
    for name in (sys.argv[1:] or ['bike_A', 'bike_B']):
        pos, col = load_proto(name)
        W = rig.compute_weights(pos, col, rig.RIG[name])
        out = os.path.join(ROOT, 'assets', name, 'rig_weights.npz')
        np.savez_compressed(out, pos=pos, W=W.astype(np.float16))
        print(name, len(pos), 'verts ->', out, os.path.getsize(out) // 1024, 'KB', dict(zip(rig.BONES, W.mean(0).round(3).tolist())))
