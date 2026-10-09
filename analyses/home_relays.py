#!/usr/bin/env python3
"""What can the home station (S-001) reach, and which hilltops make the best first relays?

Terrain-only analysis on the LandsD 5 m DTM; buildings are not modelled yet.

Steps
  1. Screening margin map from the home balcony across its clear sector.
  2. Candidate relay sites: DTM high points, named OSM peaks, OSM masts/towers.
  3. Exact link (P.526 Bullington) from every candidate to home.
  4. For every candidate: how much of the Sha Tin valley and of Kowloon it reaches.
  5. Single-relay and two-relay routes from home toward Kowloon.
  6. DTM check against OSM peak heights.

Usage: python analyses/home_relays.py --registry out/nodes.geojson --out out/home_relays
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import hkgeo  # noqa: E402
from tools.rf import Antenna, Terrain, bearing_deg, in_sector, link, screen  # noqa: E402

# ---------------------------------------------------------------- assumptions (all in one place)
F_MHZ = 915.0                 # worst case of the 868-915 MHz band
SENS_DBM = -134.0             # LONG_MODERATE: 156 dB budget at 22 dBm, 0 dBi (Meshtastic docs)
MARGIN_OK_DB = 20.0           # screening margin needed on terrain-only numbers (clutter, noise, fade)
RELAY = dict(gain_dbi=6.0, hpbw_deg=25.0, agl_m=4.0, tx_dbm=28.0, cable_db=1.0)    # pole on a hilltop
ROOFTOP = dict(gain_dbi=6.0, hpbw_deg=25.0, agl_m=25.0, tx_dbm=28.0, cable_db=1.0)  # valley / Kowloon station
HOME_HPBW = {7: 20.0, 8: 16.0, 9: 13.0}   # vertical HPBW of a collinear at that gain (approximate)
SEARCH_RADIUS_M = 20000.0     # candidate relays within this distance of home
REGION = (816000, 810000, 864000, 850000)  # HK1980 bounds loaded from the DTM
# Target areas sampled as "a rooftop station could be here": low-lying land only.
VALLEY = dict(bounds=(833000, 823500, 846500, 833500), z=(2.0, 40.0), step=300)
KOWLOON = dict(bounds=(832000, 816500, 846000, 823000), z=(2.0, 60.0), step=300)


def load_home(registry: Path):
    reg = json.loads(registry.read_text())
    f = next(x for x in reg["features"] if x["properties"]["id"] == "S-001")
    p = f["properties"]
    a = next(x for x in p["antennas"] if x["band"] == "lora")
    gain = float(a.get("gain_dbi") or 7)
    ant = Antenna(gain_dbi=gain, hpbw_deg=HOME_HPBW.get(int(round(gain)), 20.0), agl_m=float(a["agl_m"]),
                  tx_dbm=float(p["radio"]["lora_tx_dbm"]), cable_db=1.0,
                  azimuth_deg=a.get("azimuth_deg"), arc_deg=a.get("visible_arc_deg"),
                  amsl_m=a.get("mpd"))
    e, n = hkgeo.to_hk80(*f["geometry"]["coordinates"])
    return (float(e), float(n)), ant, p


def target_points(terrain: Terrain, spec):
    x0, y0, x1, y1 = spec["bounds"]
    es, ns = np.meshgrid(np.arange(x0 + spec["step"] / 2, x1, spec["step"]),
                         np.arange(y0 + spec["step"] / 2, y1, spec["step"]))
    es, ns = es.ravel(), ns.ravel()
    t = terrain.transform
    col = ((es - t.c) / t.a).astype(int)
    row = ((ns - t.f) / t.e).astype(int)
    z = terrain.z[row, col]
    keep = np.isfinite(z) & (z > spec["z"][0]) & (z < spec["z"][1])
    return np.column_stack([es[keep], ns[keep]])


def high_points(terrain: Terrain, home, radius, min_z=100.0, window=161, spacing=500.0):
    from scipy.ndimage import maximum_filter
    z = np.where(np.isfinite(terrain.z), terrain.z, -1e9)
    peak = (z == maximum_filter(z, size=window)) & (z > min_z)
    rows, cols = np.nonzero(peak)
    t = terrain.transform
    e = t.c + (cols + 0.5) * t.a
    n = t.f + (rows + 0.5) * t.e
    zz = z[rows, cols]
    near = np.hypot(e - home[0], n - home[1]) <= radius
    e, n, zz = e[near], n[near], zz[near]
    order = np.argsort(-zz)
    kept = []
    for i in order:  # greedy thinning, highest first
        if all(np.hypot(e[i] - e[j], n[i] - n[j]) >= spacing for j in kept):
            kept.append(i)
    return [(float(e[i]), float(n[i]), float(zz[i])) for i in kept]


def osm_points(layer, bounds):
    try:
        g = hkgeo.load_vector("osm", bounds, layer=layer, pad=0)
    except FileNotFoundError:
        return None
    return g


def nearest_way(ways, e, n):
    """Distance (m) and highway type of the nearest OSM way."""
    from shapely.geometry import Point
    if ways is None or ways.empty:
        return None, None
    idx = ways.sindex.nearest(Point(e, n), return_all=False)
    j = int(np.asarray(idx)[1][0])
    w = ways.iloc[j]
    return float(w.geometry.distance(Point(e, n))), w.get("highway")


def clean(v):
    """JSON-safe scalar: numpy -> python, NaN -> None."""
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 1)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", default=str(ROOT / "out" / "home_relays"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    home, home_ant, home_props = load_home(Path(a.registry))
    print(f"home {home[0]:.0f} {home[1]:.0f}, antenna {home_ant}")
    z, tr = hkgeo.load_dem(REGION)
    terrain = Terrain(z, tr)
    home_ground = terrain.sample(*home)
    print(f"DEM {z.shape}, DTM at home pin {home_ground:.1f} m, antenna at {home_ant.height(home_ground):.1f} m")

    relay = Antenna(**RELAY)
    rooftop = Antenna(**ROOFTOP)

    # 1. screening map from home toward hilltop relays
    grid, geo, h_ts = screen(terrain, home, home_ant, relay, r_max_m=25000, step_m=10, daz_deg=0.1,
                             out_res_m=25, f_mhz=F_MHZ, sens_dbm=SENS_DBM)
    np.save(out / "home_screen.npy", grid)
    (out / "home_screen.json").write_text(json.dumps({"x0": geo[0], "y0": geo[1], "res": geo[2],
                                                      "shape": list(grid.shape), "target": RELAY}))

    # 2. candidates
    ways = osm_points("ways", REGION)
    peaks = osm_points("peaks", REGION)
    masts = osm_points("masts", REGION)
    cands = []
    for e, n, zz in high_points(terrain, home, SEARCH_RADIUS_M):
        cands.append({"e": e, "n": n, "ground": zz, "kind": "high point", "name": None})

    def attach(g, kind):
        if g is None:
            return
        for _, r in g.iterrows():
            e, n = r.geometry.x, r.geometry.y
            if np.hypot(e - home[0], n - home[1]) > SEARCH_RADIUS_M:
                continue
            name = r.get("name:en") or r.get("name")
            best = min(cands, key=lambda c: np.hypot(c["e"] - e, c["n"] - n), default=None)
            if best is not None and np.hypot(best["e"] - e, best["n"] - n) < 150:
                best["name"] = best["name"] or name
                best["kind"] = kind if best["kind"] == "high point" else best["kind"]
                if kind == "osm peak":
                    best["osm_ele"] = r.get("ele")
                if kind == "mast/tower":
                    best["mast"] = True
            else:
                cands.append({"e": e, "n": n, "ground": terrain.sample(e, n), "kind": kind, "name": name,
                              "osm_ele": r.get("ele") if kind == "osm peak" else None,
                              "mast": kind == "mast/tower"})

    attach(peaks, "osm peak")
    attach(masts, "mast/tower")
    print(f"{len(cands)} candidates")

    # 3. link to home (weaker direction is field -> home), sector enforced
    for c in cands:
        az = float(bearing_deg(home[0], home[1], c["e"], c["n"]))
        c["az_from_home"] = round(az, 1)
        c["in_sector"] = bool(in_sector(az, home_ant.azimuth_deg, home_ant.arc_deg))
        L = link(terrain, home, (c["e"], c["n"]), home_ant, relay, F_MHZ, SENS_DBM)
        c["home_d_km"] = round(L.d_km, 2)
        c["home_los"] = L.los
        c["home_diff_db"] = round(L.diffraction_db, 1)
        c["home_margin_db"] = round(L.margin_db, 1) if c["in_sector"] else None
        c["home_pattern_db"] = round(L.pattern_a_db, 1)
        d, hw = nearest_way(ways, c["e"], c["n"])
        c["way_m"] = None if d is None else round(d)
        c["way_type"] = hw

    # 4. coverage of target areas
    valley = target_points(terrain, VALLEY)
    kowloon = target_points(terrain, KOWLOON)
    print(f"targets: valley {len(valley)}, kowloon {len(kowloon)}")

    def coverage(c, pts):
        m = np.array([link(terrain, (c["e"], c["n"]), (p[0], p[1]), relay, rooftop, F_MHZ, SENS_DBM, step_m=10).margin_db
                      for p in pts])
        return m

    for i, c in enumerate(cands):
        mv = coverage(c, valley)
        mk = coverage(c, kowloon)
        c["valley_cov"] = round(float((mv >= MARGIN_OK_DB).mean()), 3)
        c["kowloon_cov"] = round(float((mk >= MARGIN_OK_DB).mean()), 3)
        c["_mv"], c["_mk"] = mv, mk
        if i % 25 == 0:
            print(f"  coverage {i}/{len(cands)}")

    # 5. routes toward Kowloon
    linked = [c for c in cands if c["home_margin_db"] is not None and c["home_margin_db"] >= MARGIN_OK_DB]
    single = sorted(linked, key=lambda c: (-c["kowloon_cov"], -c["valley_cov"]))
    bridges = sorted(cands, key=lambda c: -c["kowloon_cov"])[:25]
    pairs = []
    for r1 in sorted(linked, key=lambda c: -c["home_margin_db"])[:40]:
        for r2 in bridges:
            if r2 is r1:
                continue
            L = link(terrain, (r1["e"], r1["n"]), (r2["e"], r2["n"]), relay, relay, F_MHZ, SENS_DBM)
            if L.margin_db >= MARGIN_OK_DB:
                pairs.append({"r1": r1, "r2": r2, "r1_r2_margin_db": round(L.margin_db, 1),
                              "kowloon_cov": r2["kowloon_cov"],
                              "chain_min_margin_db": round(min(L.margin_db, r1["home_margin_db"]), 1)})
    pairs.sort(key=lambda p: (-p["kowloon_cov"], -p["chain_min_margin_db"]))

    # 6. DTM vs OSM peak heights
    checks = []
    if peaks is not None:
        for _, r in peaks.iterrows():
            try:
                ele = float(str(r.get("ele")).replace("m", "").strip())
            except ValueError:
                continue
            e, n = r.geometry.x, r.geometry.y
            t = terrain.transform
            col, row = int((e - t.c) / t.a), int((n - t.f) / t.e)
            if not (10 <= row < terrain.z.shape[0] - 10 and 10 <= col < terrain.z.shape[1] - 10):
                continue
            win = terrain.z[row - 10:row + 11, col - 10:col + 11]  # +/- 50 m
            if np.isfinite(win).any():
                checks.append({"name": r.get("name:en") or r.get("name"), "osm_ele": ele,
                               "dtm_max_50m": round(float(np.nanmax(win)), 1)})

    # write results
    lon, lat = hkgeo.to_wgs84(np.array([c["e"] for c in cands]), np.array([c["n"] for c in cands]))
    feats = []
    for c, lo, la in zip(cands, lon, lat):
        props = {k: clean(v) for k, v in c.items() if not k.startswith("_")}
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(float(lo), 6), round(float(la), 6)]},
                      "properties": props})
    (out / "candidates.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}, default=str))
    np.save(out / "valley_targets.npy", valley)
    np.save(out / "kowloon_targets.npy", kowloon)

    def brief(c):
        return {k: clean(c.get(k)) for k in ("name", "kind", "e", "n", "ground", "home_d_km", "az_from_home", "home_los",
                                       "home_diff_db", "home_margin_db", "home_pattern_db", "valley_cov",
                                       "kowloon_cov", "way_m", "way_type", "mast")}

    summary = {
        "assumptions": {"f_mhz": F_MHZ, "sens_dbm": SENS_DBM, "margin_ok_db": MARGIN_OK_DB, "relay": RELAY,
                        "rooftop_target": ROOFTOP, "home_antenna": home_ant.__dict__, "valley": VALLEY,
                        "kowloon": KOWLOON},
        "home": {"e": home[0], "n": home[1], "dtm_at_pin_m": round(home_ground, 1), "antenna_m": round(float(h_ts), 1)},
        "counts": {"candidates": len(cands), "linked_to_home": len(linked), "valley_targets": len(valley),
                   "kowloon_targets": len(kowloon)},
        "best_single_relays": [brief(c) for c in single[:15]],
        "best_kowloon_bridges": [brief(c) for c in bridges[:15]],
        "best_pairs": [{"r1": brief(p["r1"]), "r2": brief(p["r2"]), "r1_r2_margin_db": p["r1_r2_margin_db"],
                        "chain_min_margin_db": p["chain_min_margin_db"]} for p in pairs[:15]],
        "dtm_vs_osm_peaks": checks,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=clean, allow_nan=False))
    print(json.dumps(summary["counts"]))


if __name__ == "__main__":
    main()
