#!/usr/bin/env python3
"""Re-check the home station and the relay shortlist with buildings (DTM + LandsD building tops).

1. Put the home antenna on Garden Vista Block B's river-facing facade.
2. Home -> candidate links with buildings. Block B itself is left out of the surface model and
   the balcony sector stands in for it; every other building blocks normally.
3. Rooftop targets: in each 300 m sample cell of the valley and Kowloon, the tallest building
   with a measured top within 150 m, antenna 3 m above its roof.
4. Relay -> rooftop coverage with buildings, union coverage of relay sets, and direct
   home -> rooftop coverage.
5. Pairwise links among the mapped sites with buildings.

Usage: python analyses/building_check.py --registry out/nodes.geojson
Outputs: out/building_check/
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analyses"))
from tools import hkgeo  # noqa: E402
from tools.dsm import building_tops, load_dsm  # noqa: E402
from tools.rf import Antenna, Terrain, bearing_deg, in_sector, link, screen  # noqa: E402
from home_relays import (F_MHZ, KOWLOON, MARGIN_OK_DB, REGION, RELAY, SENS_DBM, VALLEY,  # noqa: E402
                         load_home, target_points)

HOME_BLOCK = "Garden Vista Block B"
FLOOR = 13
ROOF_ALLOWANCE_M = 3.0      # parapet and roof structures included in TopHeight
ANTENNA_ABOVE_SLAB_M = 1.5
ROOFTOP_SEARCH_M = 150.0
ROOFTOP_MAST_M = 3.0


def facade_point(poly, az_hint, offset=3.0):
    """Outward normal of the footprint edge facing az_hint, and a point `offset` m outside it."""
    from shapely.geometry import LineString, Point
    mrr = poly.minimum_rotated_rectangle
    xs, ys = mrr.exterior.coords.xy
    c = poly.centroid
    best = None
    for i in range(4):
        (x0, y0), (x1, y1) = (xs[i], ys[i]), (xs[i + 1], ys[i + 1])
        nx, ny = (y1 - y0), -(x1 - x0)                     # a normal of the edge
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if (mx - c.x) * nx + (my - c.y) * ny < 0:           # make it point outward
            nx, ny = -nx, -ny
        az = np.degrees(np.arctan2(nx, ny)) % 360
        diff = abs((az - az_hint + 180) % 360 - 180)
        if best is None or diff < best[0]:
            best = (diff, az)
    az = best[1]
    r = np.radians(az)
    ray = LineString([(c.x, c.y), (c.x + np.sin(r) * 500, c.y + np.cos(r) * 500)])
    hit = ray.intersection(poly.boundary)
    pts = [hit] if hit.geom_type == "Point" else list(getattr(hit, "geoms", []))
    far = max(pts, key=lambda p: Point(c.x, c.y).distance(p))
    return float(az), (far.x + np.sin(r) * offset, far.y + np.cos(r) * offset)


def rooftop_targets(points, bld):
    """For each sample point, the tallest measured building within ROOFTOP_SEARCH_M (or None)."""
    from shapely.strtree import STRtree
    meas = bld[bld["top_source"] == "TopHeight"]
    rp = meas.geometry.representative_point()
    tree = STRtree(list(rp))
    out = []
    from shapely.geometry import Point
    for e, n in points:
        idx = tree.query(Point(e, n).buffer(ROOFTOP_SEARCH_M))
        if len(idx) == 0:
            out.append(None)
            continue
        sub = meas.iloc[idx]
        j = int(np.argmax(sub["top_mpd"].values))
        p = rp.iloc[idx[j]]
        out.append({"e": float(p.x), "n": float(p.y), "top": float(sub["top_mpd"].values[j]),
                    "name": sub["BuildingNameEN"].values[j] if isinstance(sub["BuildingNameEN"].values[j], str) else None})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(ROOT / "out" / "nodes.geojson"))
    ap.add_argument("--relays", default=str(ROOT / "out" / "home_relays"))
    ap.add_argument("--out", default=str(ROOT / "out" / "building_check"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    home_pin, home_ant0, _ = load_home(Path(a.registry))

    # 1. home block and antenna position
    near = building_tops((home_pin[0] - 400, home_pin[1] - 400, home_pin[0] + 400, home_pin[1] + 400))
    blk = near[near["BuildingNameEN"].fillna("").str.lower() == HOME_BLOCK.lower()]
    if blk.empty:
        raise SystemExit(f"{HOME_BLOCK} not found in the building data")
    blk = blk.iloc[0]
    poly = blk.geometry
    facade_az, ant_en = facade_point(poly, home_ant0.azimuth_deg)
    base, top, storeys = float(blk["BaseHeight"]), float(blk["TopHeight"]), int(blk["Storeys"])
    storey_h = (top - base - ROOF_ALLOWANCE_M) / storeys
    ant_mpd = base + FLOOR * storey_h + ANTENNA_ABOVE_SLAB_M
    ant_mpd_skip4 = base + (FLOOR - 1) * storey_h + ANTENNA_ABOVE_SLAB_M
    print(f"{HOME_BLOCK}: base {base} top {top} storeys {storeys} -> {storey_h:.2f} m/storey; "
          f"13/F antenna {ant_mpd:.1f} mPD ({ant_mpd_skip4:.1f} if 4/F is skipped); facade faces {facade_az:.0f} deg; "
          f"antenna at E{ant_en[0]:.0f} N{ant_en[1]:.0f}")
    home_ant = Antenna(gain_dbi=home_ant0.gain_dbi, hpbw_deg=home_ant0.hpbw_deg, agl_m=home_ant0.agl_m,
                       tx_dbm=home_ant0.tx_dbm, cable_db=home_ant0.cable_db,
                       azimuth_deg=home_ant0.azimuth_deg, arc_deg=home_ant0.arc_deg, amsl_m=round(ant_mpd, 1))

    # surface model
    bld = building_tops(REGION)
    dsm, tr, info = load_dsm(REGION, exclude=[poly.buffer(1.0)], buildings=bld)
    T = Terrain(dsm, tr)
    dtm, _ = hkgeo.load_dem(REGION)
    T0 = Terrain(dtm, tr)
    print("surface model:", info)

    relay = Antenna(**RELAY)
    # 2. home -> candidates (all from the terrain run), with and without buildings
    cands = json.loads((Path(a.relays) / "candidates.geojson").read_text())["features"]
    home_rows = []
    for f in cands:
        p = f["properties"]
        az = float(bearing_deg(ant_en[0], ant_en[1], p["e"], p["n"]))
        if not in_sector(az, home_ant.azimuth_deg, home_ant.arc_deg):
            continue
        Lb = link(T, ant_en, (p["e"], p["n"]), home_ant, relay, F_MHZ, SENS_DBM)
        Lt = link(T0, ant_en, (p["e"], p["n"]), home_ant, relay, F_MHZ, SENS_DBM)
        home_rows.append({"name": p.get("name"), "kind": p["kind"], "e": p["e"], "n": p["n"], "ground": p["ground"],
                          "az": round(az, 1), "d_km": round(Lb.d_km, 2),
                          "terrain_margin": round(Lt.margin_db, 1), "terrain_los": Lt.los,
                          "bldg_margin": round(Lb.margin_db, 1), "bldg_los": Lb.los, "bldg_diff": round(Lb.diffraction_db, 1),
                          "obstruction": Lb.obstruction})

    # 3. rooftop targets
    valley_pts = target_points(T0, VALLEY)
    kowloon_pts = target_points(T0, KOWLOON)
    roofs = {"valley": rooftop_targets(valley_pts, bld), "kowloon": rooftop_targets(kowloon_pts, bld)}
    for k, v in roofs.items():
        tops = [r["top"] for r in v if r]
        print(f"{k}: {len(v)} cells, {sum(r is not None for r in v)} with a rooftop; median roof {np.median(tops):.0f} mPD")

    # 4. relay -> rooftop coverage with buildings
    sets = json.loads((Path(a.relays) / "relay_set_margins.json").read_text())
    sites = {n: tuple(en) for n, en in sets["sites"].items()}
    margins = {}
    for name, en in sites.items():
        margins[name] = {}
        for area in ("valley", "kowloon"):
            row = []
            for r in roofs[area]:
                if r is None:
                    row.append(None)
                    continue
                tgt = Antenna(gain_dbi=6.0, hpbw_deg=25.0, agl_m=0.0, tx_dbm=28.0, cable_db=1.0, amsl_m=r["top"] + ROOFTOP_MAST_M)
                row.append(round(link(T, en, (r["e"], r["n"]), relay, tgt, F_MHZ, SENS_DBM, step_m=10).margin_db, 1))
            margins[name][area] = row
        print(f"  coverage computed: {name}")
    # home -> rooftops directly (sector applies)
    margins["HOME"] = {}
    for area in ("valley", "kowloon"):
        row = []
        for r in roofs[area]:
            if r is None:
                row.append(None)
                continue
            az = float(bearing_deg(ant_en[0], ant_en[1], r["e"], r["n"]))
            if not in_sector(az, home_ant.azimuth_deg, home_ant.arc_deg):
                row.append(-99.0)
                continue
            tgt = Antenna(gain_dbi=6.0, hpbw_deg=25.0, agl_m=0.0, tx_dbm=28.0, cable_db=1.0, amsl_m=r["top"] + ROOFTOP_MAST_M)
            row.append(round(link(T, ant_en, (r["e"], r["n"]), home_ant, tgt, F_MHZ, SENS_DBM, step_m=10).margin_db, 1))
        margins["HOME"][area] = row

    def frac(names, area):
        n_ok = n_all = 0
        for i, r in enumerate(roofs[area]):
            if r is None:
                continue
            n_all += 1
            best = max(margins[nm][area][i] for nm in names)
            n_ok += best >= MARGIN_OK_DB
        return n_ok / max(n_all, 1)

    names = list(sites)
    single = {n: (round(frac([n], "valley"), 2), round(frac([n], "kowloon"), 2)) for n in names + ["HOME"]}
    combos = []
    for r in (1, 2, 3):
        for ns in itertools.combinations(names, r):
            combos.append((frac(ns, "valley"), frac(ns, "kowloon"), ns))
    best_sets = {r: [{"sites": ns, "valley": round(v, 3), "kowloon": round(k, 3)}
                     for v, k, ns in sorted([c for c in combos if len(c[2]) == r], key=lambda c: -(c[0] + c[1]))[:6]]
                 for r in (1, 2, 3)}

    # 5. pairwise links among mapped sites, with buildings, plus home -> every mapped site
    pairs = []
    for x, y in itertools.combinations(names, 2):
        L = link(T, sites[x], sites[y], relay, relay, F_MHZ, SENS_DBM)
        pairs.append({"a": x, "b": y, "d_km": round(L.d_km, 2), "los": L.los, "margin_db": round(L.margin_db, 1)})
    home_pairs = []
    for x in names:
        az = float(bearing_deg(ant_en[0], ant_en[1], sites[x][0], sites[x][1]))
        L = link(T, ant_en, sites[x], home_ant, relay, F_MHZ, SENS_DBM)
        blocked = not bool(in_sector(az, home_ant.azimuth_deg, home_ant.arc_deg))
        home_pairs.append({"a": "HOME", "b": x, "az": round(az, 1), "d_km": round(L.d_km, 2), "los": L.los,
                           "blocked": blocked, "terrain_margin_db": round(link(T0, ant_en, sites[x], home_ant, relay, F_MHZ, SENS_DBM).margin_db, 1),
                           "margin_db": None if blocked else round(L.margin_db, 1)})

    # screening map from the balcony, with buildings
    grid, geo, _ = screen(T, ant_en, home_ant, relay, r_max_m=25000, step_m=10, daz_deg=0.1, out_res_m=25,
                          f_mhz=F_MHZ, sens_dbm=SENS_DBM)
    np.save(out / "home_screen_bldg.npy", grid)
    (out / "home_screen_bldg.json").write_text(json.dumps({"x0": geo[0], "y0": geo[1], "res": geo[2], "shape": list(grid.shape)}))

    summary = {
        "home": {"block": HOME_BLOCK, "base_mpd": base, "top_mpd": top, "storeys": storeys, "storey_m": round(storey_h, 2),
                 "antenna_mpd": round(ant_mpd, 1), "antenna_mpd_if_4F_skipped": round(ant_mpd_skip4, 1),
                 "facade_az": round(facade_az, 1), "antenna_e": round(ant_en[0], 1), "antenna_n": round(ant_en[1], 1)},
        "surface_model": info,
        "rooftop_cells": {k: sum(r is not None for r in v) for k, v in roofs.items()},
        "home_links": sorted(home_rows, key=lambda r: -r["bldg_margin"]),
        "single_site_coverage": single,
        "best_sets": best_sets,
        "pairs": pairs,
        "home_pairs": home_pairs,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda v: None if v is None else str(v)))
    (out / "rooftops.json").write_text(json.dumps({"roofs": roofs, "margins": margins, "sites": sites,
                                                   "home": {"e": ant_en[0], "n": ant_en[1], "azimuth": home_ant.azimuth_deg,
                                                            "arc": home_ant.arc_deg, "mpd": home_ant.amsl_m}}))
    print("done")


if __name__ == "__main__":
    main()
