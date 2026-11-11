"""COCO 2017 から熊（category 23 "bear"）が写った画像だけを取り出し、YOLO 形式で書く（枠は人が付けたもの）。

    python coco_bears.py annotations_trainval2017.zip --out coco

COCO 全体（19 GB）は落とさず、注釈ファイルだけを読み、熊の画像を coco_url から 1 枚ずつ取ってくる。
出力: out/images/*.jpg, out/labels/*.txt（クラス 0 = bear）、out/split.csv（train2017 / val2017 の別）
COCO の注釈は CC BY 4.0、画像は Flickr の撮影者ごとのライセンス。
"""
import argparse
import io
import json
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

BEAR = 23


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ann_zip")
    ap.add_argument("--out", default="coco")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    z = zipfile.ZipFile(a.ann_zip)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "labels"), exist_ok=True)
    jobs, split = [], []
    for part in ("train2017", "val2017"):
        d = json.load(z.open(f"annotations/instances_{part}.json"))
        imgs = {im["id"]: im for im in d["images"]}
        boxes = {}
        for an in d["annotations"]:
            if an["category_id"] == BEAR and not an.get("iscrowd", 0):
                boxes.setdefault(an["image_id"], []).append(an["bbox"])
        for iid, bs in boxes.items():
            im = imgs[iid]
            jobs.append((im, bs))
            split.append((im["file_name"], part))
        print(f"{part}: {len(boxes)} images with bears", file=sys.stderr)

    def fetch(job):
        im, bs = job
        stem = os.path.splitext(im["file_name"])[0]
        p = os.path.join(a.out, "images", stem + ".jpg")
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            try:
                data = urlopen(im["coco_url"], timeout=60).read()
                open(p, "wb").write(data)
            except Exception:
                return False
        W, H = im["width"], im["height"]
        with open(os.path.join(a.out, "labels", stem + ".txt"), "w") as f:
            for x, y, w, h in bs:
                f.write(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n")
        return True

    with ThreadPoolExecutor(a.workers) as ex:
        ok = sum(ex.map(fetch, jobs))
    with open(os.path.join(a.out, "split.csv"), "w") as f:
        f.write("file,split\n")
        for fn, part in split:
            f.write(f"{fn},{part}\n")
    print(f"done: {ok}/{len(jobs)} images", file=sys.stderr)


if __name__ == "__main__":
    main()
