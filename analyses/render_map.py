#!/usr/bin/env python3
"""Render map layers for the planning page: theme-neutral hillshade + water, contours, and the
home screening overlay, plus a data.json with sites, links, targets and per-target margins.

Usage: python analyses/render_map.py --registry out/nodes.geojson --out out/map
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analyses"))
from tools import hkgeo  # noqa: E402
from home_relays import MARGIN_OK_DB, RELAY, F_MHZ, SENS_DBM, VALLEY, KOWLOON  # noqa: E402

VIEW = (826000, 814000, 850000, 836000)   # HK1980 bounds of the rendered map
RES = 10.0                                # metres per pixel


def block_mean(z, f):
    h, w = (z.shape[0] // f) * f, (z.shape[1] // f) * f
    return np.nanmean(z[:h, :w].reshape(h // f, f, w // f, f), axis=(1, 3))


def terrain_rgba(z):
    """Hillshade as black shadows / white highlights on transparent, so it works on light and dark pages."""
    zz = np.where(np.isfinite(z), z, 0.0)
    gy, gx = np.gradient(zz * 1.3, RES)          # 1.3x vertical exaggeration
    dzdx, dzdy = gx, -gy                          # rows run south, so flip for north
    nrm = np.sqrt(dzdx ** 2 + dzdy ** 2 + 1)
    az, alt = np.radians(315.0), np.radians(45.0)
    lx, ly, lz = np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)
    shade = (-dzdx * lx - dzdy * ly + lz) / nrm
    delta = shade - np.sin(alt)
    rgba = np.zeros(z.shape + (4,), dtype=np.uint8)
    shadow = np.clip(-delta * 1.5, 0, 0.62)
    light = np.clip(delta * 1.3, 0, 0.32)
    a = np.where(delta < 0, shadow, light)
    rgba[..., :3] = np.where((delta < 0)[..., None], 0, 255)
    rgba[..., 3] = (np.round(a * 255 / 6) * 6).clip(0, 255).astype(np.uint8)   # quantised for compression
    water = ~np.isfinite(z) | (z <= 1.0)
    rgba[water] = (61, 126, 166, 82)
    return rgba


def contours_rgba(z, x0, y0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h, w = z.shape
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    xs = x0 + (np.arange(w) + 0.5) * RES
    ys = y0 - (np.arange(h) + 0.5) * RES
    zz = np.where(np.isfinite(z), z, -1)
    ax.contour(xs, ys, zz, levels=[l for l in range(100, 1000, 100) if l % 500], colors=[(0.5, 0.5, 0.5, 0.55)], linewidths=0.45)
    ax.contour(xs, ys, zz, levels=[500], colors=[(0.5, 0.5, 0.5, 0.8)], linewidths=1.0)
    ax.set_xlim(x0, x0 + w * RES)
    ax.set_ylim(y0 - h * RES, y0)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img


def screen_rgba(npy: Path, meta_json: Path):
    meta = json.loads(meta_json.read_text())
    g = np.load(npy)
    x0, y0, res = meta["x0"], meta["y0"], meta["res"]
    c0, c1 = int((VIEW[0] - x0) / res), int((VIEW[2] - x0) / res)
    r0, r1 = int((y0 - VIEW[3]) / res), int((y0 - VIEW[1]) / res)
    g = g[r0:r1, c0:c1]
    rgba = np.zeros(g.shape + (4,), dtype=np.uint8)
    rgba[g >= 40] = (31, 158, 137, 120)
    rgba[(g >= MARGIN_OK_DB) & (g < 40)] = (31, 158, 137, 62)
    rgba[(g >= 0) & (g < MARGIN_OK_DB)] = (208, 138, 44, 80)
    return rgba


def buildings_rgba():
    """Footprints at 5 m, neutral grey; taller buildings more opaque, so towers read at a glance."""
    from rasterio.features import rasterize
    from tools.dsm import building_tops
    z5, tr = hkgeo.load_dem(VIEW)
    b = building_tops(VIEW)
    rp = b.geometry.representative_point()
    h = (b["top_mpd"] - hkgeo.dem_at(rp.x.values, rp.y.values)).clip(lower=0)
    alpha = np.clip(70 + h * 1.6, 70, 215)
    b = b.assign(alpha=alpha).sort_values("alpha")
    a = rasterize(((g, int(v)) for g, v in zip(b.geometry, b["alpha"])), out_shape=z5.shape, transform=tr, fill=0, dtype="uint8")
    rgba = np.zeros(a.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = 112
    rgba[..., 3] = a
    return rgba


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--check", default=str(ROOT / "out" / "building_check"))
    ap.add_argument("--relays", default=str(ROOT / "out" / "home_relays"))
    ap.add_argument("--out", default=str(ROOT / "out" / "map"))
    ap.add_argument("--skip-base", action="store_true", help="reuse terrain.png and contours.png")
    a = ap.parse_args(argv)
    out, chk, rel = Path(a.out), Path(a.check), Path(a.relays)
    out.mkdir(parents=True, exist_ok=True)

    if not a.skip_base or not (out / "terrain.png").exists():
        z5, _ = hkgeo.load_dem(VIEW)
        z = block_mean(z5, int(RES / 5))
        Image.fromarray(terrain_rgba(z), "RGBA").save(out / "terrain.png", optimize=True)
        Image.fromarray(contours_rgba(z, VIEW[0], VIEW[3]), "RGBA").save(out / "contours.png", optimize=True)
    Image.fromarray(buildings_rgba(), "RGBA").save(out / "buildings.png", optimize=True)
    Image.fromarray(screen_rgba(chk / "home_screen_bldg.npy", chk / "home_screen_bldg.json"), "RGBA").save(out / "screen.png", optimize=True)

    reg = json.loads(Path(a.registry).read_text())
    roof = json.loads((chk / "rooftops.json").read_text())
    summ = json.loads((chk / "summary.json").read_text())
    cands = {(f["properties"].get("name") or ""): f["properties"]
             for f in json.loads((rel / "candidates.geojson").read_text())["features"]}
    sites = []
    for f in reg["features"]:
        p = f["properties"]
        mesh = (p.get("coverage") or {}).get("mesh") or {}
        sites.append({"id": p["id"], "name": p["name"], "status": p["status"], "roles": p["roles"],
                      "e": p["hk80"]["e"], "n": p["hk80"]["n"], "ground": p.get("ground_mpd"), "access": p.get("access"),
                      "notes": p.get("notes"), "valley": mesh.get("valley_frac"), "kowloon": mesh.get("kowloon_frac"),
                      "registry": True})
    reg_names = {s["name"] for s in sites}
    for name, (e, n) in roof["sites"].items():
        if name in reg_names:
            continue
        c = cands.get(name, {})
        sites.append({"id": None, "name": name, "status": "analysed", "roles": [], "e": e, "n": n,
                      "ground": c.get("ground"), "access": None, "registry": False})
    # rooftop targets: drop cells without a rooftop, keep margins aligned
    targets, margins = {}, {nm: {} for nm in roof["margins"]}
    for area in ("valley", "kowloon"):
        keep = [i for i, r in enumerate(roof["roofs"][area]) if r is not None]
        targets[area] = [[round(roof["roofs"][area][i]["e"]), round(roof["roofs"][area][i]["n"]),
                          round(roof["roofs"][area][i]["top"]), roof["roofs"][area][i]["name"]] for i in keep]
        for nm in roof["margins"]:
            margins[nm][area] = [roof["margins"][nm][area][i] for i in keep]
    data = {
        "view": VIEW, "res": RES,
        "home": roof["home"],
        "sites": sites,
        "targets": targets,
        "margins": margins,
        "pairs": summ["pairs"],
        "home_pairs": summ["home_pairs"],
        "threshold_db": MARGIN_OK_DB,
        "assumptions": {"freq_mhz": F_MHZ, "sens_dbm": SENS_DBM, "relay": RELAY, "rooftop": "tallest measured building within 150 m, +3 m",
                        "valley_area": VALLEY, "kowloon_area": KOWLOON},
    }
    (out / "data.json").write_text(json.dumps(data, separators=(",", ":"), default=lambda v: None))
    for f in ("terrain.png", "contours.png", "buildings.png", "screen.png", "data.json"):
        print(f, round((out / f).stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
