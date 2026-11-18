# highway-chase-blender

バイクの高速道路チェイス(30秒・24fps・15カット)を Blender (Cycles) で描画する。
GitHub Actions 上の headless Blender で、コマを分割して並列に描画する。

- `chase/motion.py` 動き(純関数)
- `chase/render.py` シーンの組み立てと描画(EXR)
- `tools/post.py` 仕上げ(にじみ・トーンマップ・題字・暗転)
- `vehicles.json` 車体と車輪の合体の値
- `chase/city.py` 街の組み立て(建物の凹凸・民家・木・地上の道)
- `tools/osm/`、`tools/prep_city.py` 地図データの取得と、道路沿いの座標への変換
- `assets/` 車両モデル(Meshy AI で生成)、路面と空(Poly Haven, CC0)、外壁(ambientCG, CC0)、音声
- `assets/city.json` 街の配置(首都高3号渋谷線沿い)

## 地図データ

街の配置は OpenStreetMap のデータを加工したもの。© OpenStreetMap contributors (ODbL 1.0)
https://www.openstreetmap.org/copyright

この動画を公開するときは、説明欄に「© OpenStreetMap contributors」を入れる。
