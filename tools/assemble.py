"""描き済みEXRを、Releaseから必要なコマだけ集めて exr/ に並べる(描画しない)。
usage: python tools/assemble.py '<segments JSON>' <exr_dir> <manifest.json> [repo]
segments = [{"tag":"r31","src":"0-719","dst":0}, ...]  重ねる順。後ろが前を上書き。
  src: "0-719" や "125,151,228-247"(そのタグ内のコマ番号)。tarの中に無いコマは飛ばす(上書きしない)。
  dst: src の最初のコマを、出力のこのコマ番号に置く(出力コマ = dst + (src - 最初のsrc))。
exr_dir に 0000.exr ... を作る。manifest.json に 出力コマ -> 元タグ・元コマ・解像度 を書く。"""
import sys, os, json, re, subprocess, tarfile, shutil, glob


def parse(spec):
    fr = []
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            fr += list(range(int(a), int(b) + 1))
        else:
            fr.append(int(part))
    return sorted(set(fr))


def size_of(p):
    try:
        import OpenEXR
        dw = OpenEXR.File(p).header()['dataWindow']
        return [int(dw[1][0] - dw[0][0] + 1), int(dw[1][1] - dw[0][1] + 1)]
    except Exception:
        return None


def ranges(nums):
    out = []
    for n in sorted(nums):
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return ','.join(str(a) if a == b else '%d-%d' % (a, b) for a, b in out)


def main():
    segs = json.loads(sys.argv[1])
    exr = sys.argv[2]
    man_path = sys.argv[3]
    repo = sys.argv[4] if len(sys.argv) > 4 else os.environ.get('GITHUB_REPOSITORY', '')
    os.makedirs(exr, exist_ok=True)
    owner = {}                                   # 出力コマ -> {tag, src}
    for si, sg in enumerate(segs):
        tag = sg['tag']
        want = parse(sg['src'])
        dst0 = int(sg['dst'])
        first = want[0]
        wset = set(want)
        dl = 'dl_%d' % si
        os.makedirs(dl, exist_ok=True)
        subprocess.check_call(['gh', 'release', 'download', tag, '-p', 'exr-*.tar', '-D', dl, '--repo', repo])
        got = []
        for tp in sorted(glob.glob(os.path.join(dl, '*.tar'))):
            with tarfile.open(tp) as tf:
                for m in tf:
                    mm = re.fullmatch(r'(?:\./)?(\d{4})\.exr', m.name)
                    if not mm or not m.isfile():
                        continue
                    f = int(mm.group(1))
                    if f not in wset:
                        continue
                    d = dst0 + (f - first)
                    with tf.extractfile(m) as src, open(os.path.join(exr, '%04d.exr' % d), 'wb') as out:
                        shutil.copyfileobj(src, out)
                    owner[d] = {'tag': tag, 'src': f}
                    got.append(f)
        shutil.rmtree(dl)
        miss = sorted(wset - set(got))
        print('segment %d: tag %s src %s -> dst %d..: took %d of %d frames (src %s)%s' % (
            si, tag, sg['src'], dst0, len(got), len(want), ranges(got) if got else '-',
            ('  skipped(not in tar): ' + ranges(miss)) if miss else ''), flush=True)
        if not got:
            sys.exit('segment %d (%s) has no frames' % (si, tag))
    n = sorted(owner)
    gaps = sorted(set(range(0, n[-1] + 1)) - set(n))
    print('assembled', len(n), 'frames, last', n[-1], 'gaps', gaps[:20])
    if gaps:
        sys.exit('missing output frames: ' + ranges(gaps))
    # 出力コマ -> 元(連続する同じタグ・同じ解像度のコマをまとめる)
    rows = []
    for d in n:
        o = owner[d]
        sz = size_of(os.path.join(exr, '%04d.exr' % d))
        r = rows[-1] if rows else None
        if r and r['tag'] == o['tag'] and r['size'] == sz and r['src_to'] + 1 == o['src'] and r['dst_to'] + 1 == d:
            r['dst_to'] = d
            r['src_to'] = o['src']
        else:
            rows.append({'tag': o['tag'], 'dst_from': d, 'dst_to': d, 'src_from': o['src'], 'src_to': o['src'], 'size': sz})
    json.dump({'segments': segs, 'frames': len(n), 'ranges': rows}, open(man_path, 'w'), ensure_ascii=False, indent=1)
    for r in rows:
        print('  out %d-%d <- %s %d-%d  %s' % (r['dst_from'], r['dst_to'], r['tag'], r['src_from'], r['src_to'], r['size']))


if __name__ == '__main__':
    main()
