# highway-chase-blender

バイクの高速道路チェイス(30秒・24fps・15カット)を Blender (Cycles) で描画する。
GitHub Actions 上の headless Blender で、コマを分割して並列に描画する。

- `chase/motion.py` 動き(純関数)
- `chase/render.py` シーンの組み立てと描画(EXR)
- `tools/post.py` 仕上げ(にじみ・トーンマップ・題字・暗転)
- `vehicles.json` 車体と車輪の合体の値
- `assets/` 車両モデル(Meshy AI で生成)、路面と空(Poly Haven, CC0)、音声
