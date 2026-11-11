"""COCO 2017 から、熊のいない「町の画像」（車・犬・人・自転車・ベンチなどが写ったもの）を負例として取ってくる。

    python coco_negs.py annotations_trainval2017.zip --out coco_neg [--per-cat 350] [--seed 0]

自動撮影カメラの負例は森や草原ばかりで、町なかの物（とくに黒い車）を熊と見る誤報が出た。
そこで、町に出てくる物の種類ごとに上限 per_cat 枚ずつ、熊（category 23）が写っていない画像を選ぶ。
同じ画像が複数の種類に入ったら 1 回だけ数える。val2017 からは per_cat の 1/10 ずつ。
出力: out/images/*.jpg、out/split.csv（train2017 / val2017 の別）。ラベルは付けない（全部「熊なし」）。
COCO の注釈は CC BY 4.0、画像は Flickr の撮影者ごとのライセンス。再配布しない。
"""
import argparse
import json
import os
import random
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

BEAR = 23
# 町なかで熊と見間違えそうな物、町の背景によく出る物
TOWN = ["car", "truck", "bus", "motorcycle", "bicycle", "dog", "cat", "person", "horse", "cow", "sheep",
        "bench", "fire hydrant", "stop sign", "parking meter", "suitcase", "backpack", "potted plant",
        "chair", "couch", "umbrella", "handbag", "traffic light"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ann_zip")
    ap.add_argument("--out", default="coco_neg")
    ap.add_argument("--per-cat", type=int, default=350)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    z = zipfile.ZipFile(a.ann_zip)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    jobs = []
    for part, cap in (("train2017", a.per_cat), ("val2017", max(1, a.per_cat // 10))):
        d = json.load(z.open(f"annotations/instances_{part}.json"))
        name = {c["id"]: c["name"] for c in d["categories"]}
        imgs = {im["id"]: im for im in d["images"]}
        cats = {}
        for an in d["annotations"]:
            cats.setdefault(an["image_id"], set()).add(an["category_id"])
        no_bear = [i for i, cs in sorted(cats.items()) if BEAR not in cs]
        rnd.shuffle(no_bear)
        taken, count = set(), {c: 0 for c in TOWN}
        for i in no_bear:
            hit = [name[c] for c in cats[i] if name[c] in count and count[name[c]] < cap]
            if not hit:
                continue
            for n in hit:
                count[n] += 1
            taken.add(i)
        for i in sorted(taken):
            jobs.append((imgs[i], part))
        print(f"{part}: {len(taken)} images, " + ", ".join(f"{k} {v}" for k, v in count.items()), file=sys.stderr)

    def fetch(job):
        im, part = job
        p = os.path.join(a.out, "images", os.path.splitext(im["file_name"])[0] + ".jpg")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return True
        try:
            open(p, "wb").write(urlopen(im["coco_url"], timeout=60).read())
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(a.workers) as ex:
        ok = sum(ex.map(fetch, jobs))
    with open(os.path.join(a.out, "split.csv"), "w") as f:
        f.write("file,split\n")
        for im, part in jobs:
            f.write(f"{im['file_name']},{part}\n")
    print(f"done: {ok}/{len(jobs)} images", file=sys.stderr)


if __name__ == "__main__":
    main()
