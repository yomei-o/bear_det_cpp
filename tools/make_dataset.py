"""LILA（MegaDetector の枠）と COCO（人の枠）から、熊の検出器の学習データ（YOLO 形式、1 クラス bear）を作る。

    python -m megadetector.detection.run_detector_batch MDV5A lila lila_md.json --recursive --output_relative_filenames
    python make_dataset.py --lila lila --md lila_md.json --coco coco --out dataset

  正例（LILA）: MegaDetector の「動物」の枠（信頼度 >= box_conf）を熊の枠にする。
                信頼度 keep_conf 以上の動物が 1 つも無い画像は、熊が写っていないコマ（ラベルが連続撮影単位のため）として落とす
  負例（LILA）: 枠なし（熊はいない）。人や他の動物が写っていても熊ではない
  正例（COCO）: 人が付けた熊の枠
  学習 / 検証の分け方:
    LILA … カメラの設置場所（location_id）ごとに 2 割を検証へ（同じ場所の画像が両方に入らないように）
    COCO … 元の train2017 / val2017
  町の負例（2 版で追加。自動撮影カメラの負例は森ばかりで、夜の黒い車を熊と見た）:
    --coco-neg   … coco_negs.py で取った、熊のいない町の画像（車・犬・人など）。COCO の train / val のまま
    --extra-neg  … 「フォルダ:train|val:上限」の形で、画像だけのフォルダを負例に足す（例: 首都高の夜の録画のコマ）
  町なかの熊（iNaturalist の urban）:
    --urban-train-frac … 観察ごとに、この割合を学習（2 割はさらに検証）へ回し、残りを test_urban に残す
  出力: out/{train,val}/{images,labels}、out/val_asian/（検証のうちツキノワグマだけ）、out/data.yaml、out/stats.json
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter


def is_val(location):
    return int(hashlib.md5(location.encode("utf-8")).hexdigest(), 16) % 10 < 2


def frac_of(key):
    """key ごとに決まる 0〜1 の値（is_val とは別の鍵で、偏りが重ならないように）"""
    return int(hashlib.md5(("frac/" + key).encode("utf-8")).hexdigest(), 16) % 1000 / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lila", default="lila")
    ap.add_argument("--md", default="lila_md.json")
    ap.add_argument("--coco", default="coco")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--inat", default="inat")
    ap.add_argument("--inat-md", default="inat_md.json")
    ap.add_argument("--box-conf", type=float, default=0.35)
    ap.add_argument("--keep-conf", type=float, default=0.5)
    ap.add_argument("--coco-neg", default="", help="coco_negs.py の出力フォルダ")
    ap.add_argument("--extra-neg", action="append", default=[], help="フォルダ:train|val:上限（何回でも）")
    ap.add_argument("--urban-train-frac", type=float, default=0.0, help="町なかの熊の観察のうち学習へ回す割合")
    a = ap.parse_args()

    md = json.load(open(a.md, encoding="utf-8"))
    det = {im["file"].replace("\\", "/"): im.get("detections") or [] for im in md["images"]}
    for s in ("train", "val", "val_asian"):
        for d in ("images", "labels"):
            os.makedirs(os.path.join(a.out, s, d), exist_ok=True)
    st = Counter()

    def put(split, src, stem, lines):
        shutil.copyfile(src, os.path.join(a.out, split, "images", stem + ".jpg"))
        with open(os.path.join(a.out, split, "labels", stem + ".txt"), "w") as f:
            f.write("".join(lines))

    for row in csv.DictReader(open(os.path.join(a.lila, "selected.csv"), encoding="utf-8")):
        rel = row["file"].replace("\\", "/")
        src = os.path.join(a.lila, rel)
        if not os.path.exists(src):
            continue
        split = "val" if is_val(row["dataset"] + "/" + row["location_id"]) else "train"
        stem = "lila_" + os.path.splitext(os.path.basename(rel))[0]
        if row["kind"] == "pos":
            ds = [d for d in det.get(rel, []) if d["category"] == "1"]
            if not any(d["conf"] >= a.keep_conf for d in ds):
                st["pos_dropped_no_animal"] += 1
                continue
            lines = [f"0 {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}\n" for d in ds if d["conf"] >= a.box_conf
                     for x, y, w, h in [d["bbox"]]]
            put(split, src, stem, lines)
            st[f"{split}_pos"] += 1
            st[f"{split}_boxes"] += len(lines)
            if row["common_name"].lower() == "asian black bear" and split == "val":
                put("val_asian", src, stem, lines)
                st["val_asian"] += 1
        else:
            put(split, src, stem, [])
            st[f"{split}_neg"] += 1

    split_of = {}
    sp = os.path.join(a.coco, "split.csv")
    for row in csv.DictReader(open(sp)):
        split_of[os.path.splitext(row["file"])[0]] = "val" if row["split"] == "val2017" else "train"
    for fn in os.listdir(os.path.join(a.coco, "images")):
        stem = os.path.splitext(fn)[0]
        lab = os.path.join(a.coco, "labels", stem + ".txt")
        if not os.path.exists(lab):
            continue
        split = split_of.get(stem, "train")
        lines = open(lab).readlines()
        put(split, os.path.join(a.coco, "images", fn), "coco_" + stem, lines)
        st[f"{split}_pos"] += 1
        st[f"{split}_boxes"] += len(lines)

    # iNaturalist：MegaDetector で動物が見つかった写真だけ（糞・足跡・何も写っていない写真を落とす）
    #   asian / japan / general … 観察ごとに 2 割を検証へ
    #   urban                   … 学習に使わず「町なかテスト」（test_urban）へ
    if a.inat and os.path.exists(a.inat_md):
        imd = json.load(open(a.inat_md, encoding="utf-8"))
        idet = {im["file"].replace("\\", "/").split("/")[-1]: im.get("detections") or [] for im in imd["images"]}
        for s in ("test_urban",):
            for d in ("images", "labels"):
                os.makedirs(os.path.join(a.out, s, d), exist_ok=True)
        for row in csv.DictReader(open(os.path.join(a.inat, "inat.csv"), encoding="utf-8")):
            src = os.path.join(a.inat, "images", row["file"])
            if not os.path.exists(src):
                continue
            ds = [d for d in idet.get(row["file"], []) if d["category"] == "1"]
            if not any(d["conf"] >= a.keep_conf for d in ds):
                st["inat_dropped_no_animal"] += 1
                continue
            lines = [f"0 {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}\n" for d in ds if d["conf"] >= a.box_conf for x, y, w, h in [d["bbox"]]]
            if row["group"] == "urban" and frac_of("inat/" + row["observation"]) >= a.urban_train_frac:
                split = "test_urban"
            else:
                split = "val" if is_val("inat/" + row["observation"]) else "train"
            stem = "inat_" + os.path.splitext(row["file"])[0]
            put(split, src, stem, lines)
            st[f"{split}_pos"] += 1
            st[f"{split}_boxes"] += len(lines)
            st[f"inat_{row['group']}"] += 1
            if row["group"] == "asian" and split == "val":
                put("val_asian", src, stem, lines)
                st["val_asian"] += 1

    # 町の負例（熊なし、ラベルは空）
    if a.coco_neg:
        for row in csv.DictReader(open(os.path.join(a.coco_neg, "split.csv"))):
            stem = os.path.splitext(row["file"])[0]
            src = os.path.join(a.coco_neg, "images", stem + ".jpg")
            if not os.path.exists(src):
                continue
            split = "val" if row["split"] == "val2017" else "train"
            put(split, src, "coconeg_" + stem, [])
            st[f"{split}_neg"] += 1
            st["coco_neg"] += 1
    for spec in a.extra_neg:
        folder, split, cap = spec.rsplit(":", 2)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png")))
        step = max(1, len(files) // int(cap))          # 連続したコマなので、間隔をあけて取る
        tag = "xneg_" + hashlib.md5(folder.encode("utf-8")).hexdigest()[:6] + "_"
        for fn in files[::step][:int(cap)]:
            src = os.path.join(folder, fn)
            if fn.lower().endswith(".png"):
                from PIL import Image
                Image.open(src).convert("RGB").save(os.path.join(a.out, split, "images", tag + os.path.splitext(fn)[0] + ".jpg"), quality=92)
                open(os.path.join(a.out, split, "labels", tag + os.path.splitext(fn)[0] + ".txt"), "w").close()
            else:
                put(split, src, tag + os.path.splitext(fn)[0], [])
            st[f"{split}_neg"] += 1
            st["extra_neg"] += 1

    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: train/images\nval: val/images\nnames:\n  0: bear\n")
    json.dump(dict(st), open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(json.dumps(dict(st), indent=1))


if __name__ == "__main__":
    main()
