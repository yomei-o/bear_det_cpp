"""LILA BC（自動撮影カメラの画像集）から、熊の正例と、熊のいない負例を選んでダウンロードする。

    python select_lila.py lila_image_urls_and_labels.csv.zip --out lila [--seed 0]

LILA の CSV（全データセットの画像 URL と、共通の動物名 common_name）を 1 回読み、
  正例: 名前に bear を含む画像（bearded は除く）
        ツキノワグマ（asian black bear）・マレーグマ・メガネグマは全部、北米のクロクマはデータセットごとに上限 1,200
        同じ連続撮影（sequence_id）からは 1 枚だけ（似た画像が並ばないように）
  負例: 熊が出るデータセットの、熊以外の動物と空の画像
        動物は (データセット, 動物名) ごとに上限 150、間違えやすい動物（イノシシ・犬・鹿・猿・人など）は 400
        空の画像はデータセットごとに上限 300
  を選び、out/pos/ と out/neg/ に保存して、out/selected.csv に一覧を書く。
ラベルは連続撮影単位のものもあるので、正例でも熊が写っていないコマが混じる。それは make_dataset.py が
MegaDetector で動物が見つからない画像として落とす。
LILA のデータはデータセットごとにライセンスが違う（多くは CDLA-Permissive）。再配布はしない。
"""
import argparse
import csv
import io
import os
import random
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

FULL = {"asian black bear", "sun bear", "spectacled bear", "bear"}
LOOKALIKE = ("boar", "pig", "dog", "deer", "muntjac", "macaque", "monkey", "human", "person", "badger", "raccoon dog",
             "serow", "goat", "cattle", "cow", "porcupine", "wolverine", "fisher", "coyote", "wolf", "wild boar")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_zip")
    ap.add_argument("--out", default="lila")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap-pos", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--neg-total", type=int, default=9000, help="負例の総数の上限")
    ap.add_argument("--per-seq-rare", type=int, default=5, help="珍しい熊は 1 回の連続撮影から何枚まで取るか")
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    z = zipfile.ZipFile(a.csv_zip)
    f = io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8")
    # 1 回目：熊の出るデータセットを知るため、まず全部読んで熊と候補を分ける（貯水池抽出）
    pos_seq = defaultdict(list)          # (dataset, seq) -> rows
    neg_pool = defaultdict(list)         # (dataset, name) -> rows（貯水池）
    neg_seen = defaultdict(int)
    bear_ds = set()
    rows_all = 0
    for row in csv.DictReader(f):
        rows_all += 1
        cn = (row.get("common_name") or "").lower()
        ds = row["dataset_name"]
        if "bear" in cn and "bearded" not in cn:
            bear_ds.add(ds)
            # 連続撮影の番号が「不明」のデータセット（NACTI・ENA24 など）は 1 枚ずつ別に扱う
            seq = row["sequence_id"] or ""
            if not seq or seq.lower().endswith("unknown"):
                seq = row["image_id"]
            pos_seq[(ds, seq)].append(row)
            continue
        key = (ds, cn or "empty")
        cap = 300 if cn in ("empty", "") else (400 if any(k in cn for k in LOOKALIKE) else 150)
        neg_seen[key] += 1
        lst = neg_pool[key]
        if len(lst) < cap:
            lst.append(row)
        else:
            j = rnd.randrange(neg_seen[key])
            if j < cap:
                lst[j] = row
    print(f"{rows_all} rows, bear datasets: {sorted(bear_ds)}", file=sys.stderr)

    # 正例：連続撮影から 1 枚、データセット・種類ごとの上限
    by_group = defaultdict(list)
    for (ds, seq), rows in pos_seq.items():
        cn = rows[0]["common_name"].lower()
        if cn in FULL:
            # 珍しい熊（ツキノワグマなど）は連続撮影からも何枚か取る（コマの間隔をあけて）
            k = min(len(rows), a.per_seq_rare)
            picks = [rows[int(i * len(rows) / k)] for i in range(k)]
        else:
            picks = [rnd.choice(rows)]
        for r in picks:
            by_group[(ds, cn if cn in FULL else "black bear")].append(r)
    pos = []
    for (ds, cn), rows in sorted(by_group.items()):
        rnd.shuffle(rows)
        take = rows if cn in FULL else rows[:a.cap_pos]
        pos += take
        print(f"  pos {ds} / {cn}: {len(take)} of {len(rows)} sequences", file=sys.stderr)
    neg = []
    for (ds, cn), rows in sorted(neg_pool.items()):
        if ds in bear_ds:
            neg += [(ds, cn, r) for r in rows]
    # 総数を抑える：間違えやすい動物と空の画像は多めに残す
    def weight(t):
        cn = t[1]
        return 3.0 if any(k in cn for k in LOOKALIKE) else (1.5 if cn == "empty" else 1.0)
    if len(neg) > a.neg_total:
        keyed = sorted(neg, key=lambda t: rnd.random() ** (1.0 / weight(t)), reverse=True)
        neg = keyed[:a.neg_total]
    neg = [r for _, _, r in neg]
    print(f"selected {len(pos)} positive, {len(neg)} negative images", file=sys.stderr)

    os.makedirs(os.path.join(a.out, "pos"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "neg"), exist_ok=True)
    items = [("pos", r) for r in pos] + [("neg", r) for r in neg]

    def fname(kind, r):
        safe = "".join(ch if ch.isalnum() else "_" for ch in r["image_id"])[-120:]
        return os.path.join(a.out, kind, safe + ".jpg")

    def fetch(item):
        kind, r = item
        p = fname(kind, r)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return True
        for key in ("url_aws", "url_gcp", "url_azure"):
            u = r.get(key)
            if not u:
                continue
            try:
                data = urlopen(u, timeout=60).read()
                with open(p, "wb") as o:
                    o.write(data)
                return True
            except Exception:
                continue
        return False

    ok = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, good in enumerate(ex.map(fetch, items)):
            ok += good
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(items)} downloaded ({ok} ok)", file=sys.stderr, flush=True)
    with open(os.path.join(a.out, "selected.csv"), "w", newline="", encoding="utf-8") as o:
        w = csv.writer(o)
        w.writerow(["kind", "file", "dataset", "common_name", "image_id", "sequence_id", "location_id"])
        for kind, r in items:
            p = fname(kind, r)
            if os.path.exists(p):
                w.writerow([kind, os.path.relpath(p, a.out), r["dataset_name"], r["common_name"], r["image_id"], r["sequence_id"], r["location_id"]])
    print(f"done: {ok}/{len(items)} images", file=sys.stderr)


if __name__ == "__main__":
    main()
