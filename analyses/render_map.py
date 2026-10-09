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
from home_relays import MARGIN_OK_DB, RELAY, ROOFTOP, F_MHZ, SENS_DBM, VALLEY, KOWLOON  # noqa: E402

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


def screen_rgba(out_dir_relays: Path):
    meta = json.loads((out_dir_relays / "home_screen.json").read_text())
    g = np.load(out_dir_relays / "home_screen.npy")
    x0, y0, res = meta["x0"], meta["y0"], meta["res"]
    c0, c1 = int((VIEW[0] - x0) / res), int((VIEW[2] - x0) / res)
    r0, r1 = int((y0 - VIEW[3]) / res), int((y0 - VIEW[1]) / res)
    g = g[r0:r1, c0:c1]
    rgba = np.zeros(g.shape + (4,), dtype=np.uint8)
    rgba[g >= 40] = (31, 158, 137, 120)
    rgba[(g >= MARGIN_OK_DB) & (g < 40)] = (31, 158, 137, 62)
    rgba[(g >= 0) & (g < MARGIN_OK_DB)] = (208, 138, 44, 80)
    return rgba


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--relays", default=str(ROOT / "out" / "home_relays"))
    ap.add_argument("--out", default=str(ROOT / "out" / "map"))
    a = ap.parse_args(argv)
    out, rel = Path(a.out), Path(a.relays)
    out.mkdir(parents=True, exist_ok=True)

    z5, _ = hkgeo.load_dem(VIEW)
    z = block_mean(z5, int(RES / 5))
    Image.fromarray(terrain_rgba(z), "RGBA").save(out / "terrain.png", optimize=True)
    Image.fromarray(contours_rgba(z, VIEW[0], VIEW[3]), "RGBA").save(out / "contours.png", optimize=True)
    Image.fromarray(screen_rgba(rel), "RGBA").save(out / "screen.png", optimize=True)

    reg = json.loads(Path(a.registry).read_text())
    cands = {(f["properties"].get("name") or ""): f["properties"]
             for f in json.loads((rel / "candidates.geojson").read_text())["features"]}
    sets = json.loads((rel / "relay_set_margins.json").read_text())
    sites = []
    for f in reg["features"]:
        p = f["properties"]
        e, n = p["hk80"]["e"], p["hk80"]["n"]
        mesh = (p.get("coverage") or {}).get("mesh") or {}
        sites.append({"id": p["id"], "name": p["name"], "status": p["status"], "roles": p["roles"], "e": e, "n": n,
                      "ground": p.get("ground_mpd"), "access": p.get("access"), "notes": p.get("notes"),
                      "valley": mesh.get("valley_frac"), "kowloon": mesh.get("kowloon_frac"),
                      "links": p.get("links", []), "registry": True})
    reg_names = {s["name"] for s in sites}
    for name, (e, n) in sets["sites"].items():
        if name in reg_names:
            continue
        c = cands.get(name, {})
        sites.append({"id": None, "name": name, "status": "analysed", "roles": [], "e": e, "n": n,
                      "ground": c.get("ground"), "home_margin": c.get("home_margin_db"),
                      "valley": c.get("valley_cov"), "kowloon": c.get("kowloon_cov"), "registry": False})
    for s in sites:
        c = cands.get(s["name"])
        if c is not None:
            s["home_margin"] = c.get("home_margin_db")
            s["az_from_home"] = c.get("az_from_home")
            s["home_d_km"] = c.get("home_d_km")
    home = next(f for f in reg["features"] if f["properties"]["id"] == "S-001")["properties"]
    ant = home["antennas"][0]
    data = {
        "view": VIEW, "res": RES,
        "home": {"e": home["hk80"]["e"], "n": home["hk80"]["n"], "azimuth": ant["azimuth_deg"],
                 "arc": ant["visible_arc_deg"], "mpd": ant.get("mpd")},
        "sites": sites,
        "targets": {"valley": sets["valley_points"], "kowloon": sets["kowloon_points"]},
        "margins": sets["margins"],
        "threshold_db": MARGIN_OK_DB,
        "assumptions": {"freq_mhz": F_MHZ, "sens_dbm": SENS_DBM, "relay": RELAY, "rooftop": ROOFTOP,
                        "valley_area": VALLEY, "kowloon_area": KOWLOON},
        "generated": "2026-10-09",
    }
    (out / "data.json").write_text(json.dumps(data, separators=(",", ":"), default=lambda v: None))
    for f in ("terrain.png", "contours.png", "screen.png", "data.json"):
        print(f, round((out / f).stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
