"""iNaturalist から熊の観察写真を集める（野生のもの、写真付き、CC ライセンスのもの）。

    python inat_bears.py --out inat [--licenses cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa]

集めるもの（それぞれ別の「グループ」として一覧に残す）:
  asian   … ツキノワグマ（Ursus thibetanus）全部
  japan   … 日本のヒグマ（Ursus arctos、place=日本）全部
  urban   … アメリカクロクマで、説明文に町の言葉（庭・近所・道路・私道・ゴミ・玄関・家・町）があるもの
  general … アメリカクロクマ一般（上限 --general 件）
観察記録には糞や足跡の写真もあるので、あとで MegaDetector で動物が写っていない写真を落とす。
出力: out/images/<観察ID>_<写真ID>.jpg と out/inat.csv（ライセンス・撮影者の表示・説明文・場所・グループ）。
写真の著作権は撮影者にある。CC BY 系は表示が必要、NC は非商用だけ。学習に使った写真の一覧（inat.csv）は必ず残す。
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://api.inaturalist.org/v1/observations"
UA = {"User-Agent": "bear-detector-research (non-commercial)"}
URBAN = ["backyard", "yard", "neighborhood", "neighbourhood", "street", "driveway", "trash", "garbage", "porch", "house", "town",
         "deck", "garage", "residential", "subdivision", "doorbell", "ring camera", "security camera", "parking"]


def get(params):
    url = API + "?" + urllib.parse.urlencode(params)
    for k in range(5):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90))
        except Exception as e:
            time.sleep(5 * (k + 1))
    raise RuntimeError("api failed: " + url)


def collect(base, limit):
    """id_above で順にたどる（ページ番号は 10,000 件までしか使えないので）"""
    out, last = [], 0
    while len(out) < limit:
        p = dict(base, per_page=200, order_by="id", order="asc", id_above=last)
        res = get(p)["results"]
        if not res:
            break
        out += res
        last = res[-1]["id"]
        time.sleep(1.1)   # API は 1 秒 1 回くらいまでにする
    return out[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="inat")
    ap.add_argument("--licenses", default="cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa")
    ap.add_argument("--general", type=int, default=3000)
    a = ap.parse_args()
    base = dict(photos="true", captive="false", photo_license=a.licenses, quality_grade="research,needs_id")
    groups = [("asian", dict(base, taxon_id=41647), 100000),
              ("japan", dict(base, taxon_id=41641, place_id=6737), 100000)]
    for w in URBAN:
        groups.append(("urban", dict(base, taxon_id=41638, q=w, search_on="description"), 100000))
    groups.append(("general", dict(base, taxon_id=41638), a.general))

    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    seen = {}
    for g, params, lim in groups:
        obs = collect(params, lim)
        n_new = 0
        for o in obs:
            if o["id"] in seen:
                continue
            seen[o["id"]] = (g, o)
            n_new += 1
        print(f"{g} {params.get('q', '')}: {len(obs)} observations ({n_new} new)", file=sys.stderr, flush=True)

    # 写真の一覧：珍しい熊（asian / japan）と町（urban）は 1 観察から 3 枚まで、一般は 1 枚
    todo = []
    for oid, (g, o) in seen.items():
        for ph in o.get("photos", [])[:(1 if g == "general" else 3)]:
            if ph.get("license_code"):
                todo.append((oid, g, o, ph))

    def fetch(t):
        oid, g, o, ph = t
        fn = f"{oid}_{ph['id']}.jpg"
        p = os.path.join(a.out, "images", fn)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            try:
                url = ph["url"].replace("/square.", "/large.")
                data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90).read()
                open(p, "wb").write(data)
            except Exception:
                return None
        return {"file": fn, "group": g, "observation": oid, "license": ph["license_code"], "attribution": ph.get("attribution", ""),
                "place": o.get("place_guess") or "", "observed_on": o.get("observed_on") or "",
                "description": (o.get("description") or "").replace("\n", " ")[:300],
                "url": f"https://www.inaturalist.org/observations/{oid}"}

    rows = []
    # 写真は静的なファイル置き場（API ではない）なので並行で取る
    with ThreadPoolExecutor(16) as ex:
        for i, r in enumerate(ex.map(fetch, todo)):
            if r:
                rows.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(todo)} photos", file=sys.stderr, flush=True)
    with open(os.path.join(a.out, "inat.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"done: {len(seen)} observations, {len(rows)} photos", file=sys.stderr)


if __name__ == "__main__":
    main()
