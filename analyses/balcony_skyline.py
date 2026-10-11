#!/usr/bin/env python3
"""Predicted skyline from the home balcony, to check against a photo or video panorama.

Casts rays every 0.1 deg across the balcony's sector (plus 20 deg either side) over the surface
model (LandsD terrain + building tops + canopy; Garden Vista Block B itself left out) and draws the
highest elevation angle in each direction, coloured by what forms it (terrain, tree canopy or a
building), with the terrain-only skyline dashed behind it. Named peaks and buildings that form the
skyline are labelled; key relay sites are marked as visible (filled) or hidden (hollow).

Usage: python analyses/balcony_skyline.py --registry out/nodes.geojson --out out/balcony_skyline.png
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import hkgeo  # noqa: E402
from tools.dsm import building_tops  # noqa: E402

R_EFF = 6371000.0 * 4 / 3
RMAX, STEP, DAZ, MARGIN = 20000.0, 5.0, 0.1, 20.0
HOME_BLOCK = "Garden Vista Block B"


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rasterio.features import rasterize
    from scipy.ndimage import map_coordinates

    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(ROOT / "out" / "nodes.geojson"))
    ap.add_argument("--spots", default=str(ROOT / "out" / "spots" / "ranked.json"))
    ap.add_argument("--out", default=str(ROOT / "out" / "balcony_skyline.png"))
    a = ap.parse_args(argv)

    reg = json.loads(Path(a.registry).read_text())
    home = next(f["properties"] for f in reg["features"] if "home" in f["properties"]["roles"])
    ant = next(x for x in home["antennas"] if x["band"] == "lora")
    e0, n0, h0 = home["hk80"]["e"], home["hk80"]["n"], ant["mpd"]
    az0 = ant["azimuth_deg"] - ant["visible_arc_deg"] / 2 - MARGIN
    az1 = ant["azimuth_deg"] + ant["visible_arc_deg"] / 2 + MARGIN

    B = (e0 - RMAX - 100, n0 - RMAX - 100, e0 + RMAX + 100, n0 + RMAX + 100)
    dtm, tr = hkgeo.load_dem(B)
    _, cmean, _ = hkgeo.load_canopy(B)
    bld = building_tops(B)
    bld = bld[bld["BuildingNameEN"].fillna("").str.lower() != HOME_BLOCK.lower()].sort_values("top_mpd").reset_index(drop=True)
    tops = rasterize(((g, float(h)) for g, h in zip(bld.geometry, bld["top_mpd"])), out_shape=dtm.shape, transform=tr,
                     fill=np.nan, dtype="float32")
    bid = rasterize(((g, i) for i, g in enumerate(bld.geometry)), out_shape=dtm.shape, transform=tr, fill=-1, dtype="int32")
    ground = np.where(np.isfinite(dtm), dtm, 0.0)
    canopy = ground + np.nan_to_num(cmean, nan=0.0)
    surf = np.fmax(canopy, np.nan_to_num(tops, nan=-1e9))

    az = np.arange(az0, az1 + 1e-9, DAZ)
    s = np.arange(STEP, RMAX, STEP)
    rad = np.radians(az)
    pe = e0 + np.sin(rad)[:, None] * s[None, :]
    pn = n0 + np.cos(rad)[:, None] * s[None, :]
    col = (pe - tr.c) / tr.a - 0.5
    row = (pn - tr.f) / tr.e - 0.5

    def ang(grid, order=1):
        z = map_coordinates(grid, [row.ravel(), col.ravel()], order=order, mode="nearest").reshape(pe.shape)
        return np.degrees(np.arctan((z - h0) / s[None, :] - s[None, :] / (2 * R_EFF)))

    a_all = ang(surf, order=0)
    a_ter = ang(ground)
    a_can = ang(canopy, order=0)
    k = a_all.argmax(axis=1)
    sky = a_all[np.arange(len(az)), k]
    sky_ter = a_ter.max(axis=1)
    dist = s[k]
    rr = np.clip(np.round(row[np.arange(len(az)), k]).astype(int), 0, dtm.shape[0] - 1)
    cc = np.clip(np.round(col[np.arange(len(az)), k]).astype(int), 0, dtm.shape[1] - 1)
    b_at = bid[rr, cc]
    is_bld = (b_at >= 0) & (np.nan_to_num(tops[rr, cc], nan=-1e9) >= canopy[rr, cc] - 0.5)
    is_tree = ~is_bld & (a_can[np.arange(len(az)), k] - a_ter[np.arange(len(az)), k] > 0.05) & (np.nan_to_num(cmean[rr, cc]) > 2)

    peaks = hkgeo.load_vector("osm", B, layer="peaks", pad=0)
    pk_name = [x if isinstance(x, str) else (y if isinstance(y, str) else None) for x, y in zip(peaks["name:en"], peaks["name"])]
    pxy = np.column_stack([peaks.geometry.x, peaks.geometry.y])

    # label runs of the same skyline source
    labels = []
    src_key = np.where(is_bld, b_at, -1)
    i = 0
    while i < len(az):
        j = i
        while j + 1 < len(az) and src_key[j + 1] == src_key[i] and (src_key[i] >= 0 or True):
            if src_key[i] < 0 and abs(dist[j + 1] - dist[j]) > 400:
                break
            j += 1
        seg = slice(i, j + 1)
        m = i + int(np.argmax(sky[seg]))
        width = (j - i + 1) * DAZ
        if src_key[i] >= 0:
            nm = bld.iloc[src_key[i]]["BuildingNameEN"]
            if isinstance(nm, str) and width >= 0.6:
                labels.append((az[m], sky[m], nm.title() if nm.isupper() else nm, dist[m], "b"))
        else:
            ex, ny = pe[m, k[m]], pn[m, k[m]]
            dpk = np.hypot(pxy[:, 0] - ex, pxy[:, 1] - ny)
            jn = np.argsort(dpk)
            for q in jn[:5]:
                if dpk[q] > 350:
                    break
                if pk_name[q] and not any("一" <= ch <= "鿿" for ch in pk_name[q]):
                    labels.append((az[m], sky[m], pk_name[q], dist[m], "t"))
                    break
        i = j + 1
    # one label per estate / peak, at its highest point
    best = {}
    for b_, el_, nm_, d_, kd_ in labels:
        key = re.sub(r"\s+(tower|block|blk|house)\s*[\w']+$", "", nm_, flags=re.I) if kd_ == "b" else nm_
        if key not in best or el_ > best[key][1]:
            best[key] = (b_, el_, key, d_, kd_)
    labels = sorted(best.values(), key=lambda L: L[0])

    # key sites: visible or hidden from the balcony
    sites = []
    for f in reg["features"]:
        p = f["properties"]
        if "home" in p["roles"]:
            continue
        an = next((x for x in p.get("antennas", []) if x.get("band") == "lora"), None)
        sites.append((p["id"] + " " + p["name"], p["hk80"]["e"], p["hk80"]["n"], an["mpd"] if an else p["ground_mpd"]))
    try:
        rk = json.loads(Path(a.spots).read_text())
        for r in rk["hills"] + rk["roofs"]:
            if r.get("relay_pick") and not r.get("registry_site"):
                sites.append((r["sid"] + " " + r["name"].replace("Unnamed building near ", "bldg near ").replace(" Block ", " Blk "), r["e"], r["n"], r["antenna_mpd"]))
    except FileNotFoundError:
        pass
    marks = []
    for name, e, n, h in sites:
        d = float(np.hypot(e - e0, n - n0))
        b = float(np.degrees(np.arctan2(e - e0, n - n0)) % 360)
        bb = b if b >= az0 else b + 360
        if bb > az1:
            continue
        el = float(np.degrees(np.arctan((h - h0) / d - d / (2 * R_EFF))))
        ia = int(round((bb - az0) / DAZ))
        kd = max(int((d - 30) / STEP) - 1, 0)
        block = float(a_all[ia, :kd].max()) if kd > 0 else -90
        marks.append((bb, el, name, el > block, d))

    # draw
    fig, ax = plt.subplots(figsize=(22, 6.2), dpi=130)
    x = az
    lo = min(-4.0, float(np.percentile(sky, 1)) - 1)
    hi = max(16.0, float(np.percentile(sky, 99.5)) + 2.5)
    ax.fill_between(x, lo, sky, color="#c9d3cf", zorder=1)
    for mask, c in ((~is_bld & ~is_tree, "#7f8c86"), (is_tree, "#4f8a5b"), (is_bld, "#47607a")):
        y = np.where(mask, sky, np.nan)
        ax.plot(x, y, color=c, lw=2.2, zorder=3)
    ax.plot(x, sky_ter, color="#7f8c86", lw=1.0, ls="--", zorder=2, label="terrain only (no buildings or trees)")
    for edge in (ant["azimuth_deg"] - ant["visible_arc_deg"] / 2, ant["azimuth_deg"] + ant["visible_arc_deg"] / 2):
        ax.axvline(edge, color="#9b3d8a", ls=":", lw=1.4)
    ax.axvline(ant["azimuth_deg"], color="#9b3d8a", ls="-", lw=0.6, alpha=0.5)
    ax.axhline(0, color="#999", lw=0.6)
    deg_per_char = (az1 - az0) / 22 / 13.0          # ~13 characters per inch at 7 pt
    levels = []                                       # per level: list of (centre, half-width)
    for (b, el, nm, d, kind) in labels:
        text = f"{nm} ({d / 1000:.1f} km)"
        half = len(text) * deg_per_char / 2 + 0.4
        base = max(el, float(sky[max(0, int((b - az0) / DAZ) - 30):int((b - az0) / DAZ) + 30].max()))
        lv = 0
        while lv < len(levels) and any(abs(b - c) < half + h for c, h in levels[lv]):
            lv += 1
        if lv == len(levels):
            levels.append([])
        levels[lv].append((b, half))
        ax.annotate(text, (b, el), xytext=(b, base + 1.0 + lv * 1.25), fontsize=7, ha="center",
                    color="#2b3a4a" if kind == "b" else "#2f4a35",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#999"), zorder=4)
    marks.sort(key=lambda m: m[0])
    last = -999
    flip = False
    for b, el, nm, vis, d in marks:
        flip = (not flip) if b - last < 6 else False
        last = b
        ax.plot(b, el, marker="o", ms=8, mfc="#d7301f" if vis else "none", mec="#d7301f", mew=1.6, zorder=5)
        ax.annotate(f"{nm} · {'visible' if vis else 'hidden'}", (b, el), xytext=(4, -24 if flip else -12),
                    textcoords="offset points", fontsize=7.5, color="#a01c0f", zorder=5)
    ticks = np.arange(np.ceil(az0 / 10) * 10, az1 + 0.1, 10)
    ax.set_xticks(ticks)
    names = {0: "N", 45: "NE", 90: "E", 180: "S", 225: "SW", 270: "W", 315: "NW"}
    ax.set_xticklabels([f"{int(t % 360)}°" + (f"\n{names[int(t % 360)]}" if int(t % 360) in names else "") for t in ticks])
    ax.set_xlim(az0, az1)
    ax.set_ylim(lo, hi)
    ax.set_ylabel("elevation angle (deg)")
    ax.set_title(f"Predicted view from the balcony antenna ({h0:.0f} mPD, 13/F Block B): skyline coloured terrain grey, trees green, "
                 f"buildings blue; dotted purple = assumed clear-sector edges; red = relay sites (filled visible, hollow hidden)",
                 fontsize=9)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out)
    print("saved", a.out, "| labels:", [L[2] for L in labels][:40])
    print("marks:", [(round(b % 360, 1), round(el, 1), nm, vis) for b, el, nm, vis, d in marks])


if __name__ == "__main__":
    main()
