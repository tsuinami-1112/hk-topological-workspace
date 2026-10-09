#!/usr/bin/env python3
"""Exact links among shortlisted sites, from the candidates produced by analyses/home_relays.py."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import hkgeo  # noqa: E402
from tools.rf import Antenna, Terrain, link  # noqa: E402
sys.path.insert(0, str(ROOT / "analyses"))
from home_relays import RELAY, ROOFTOP, F_MHZ, SENS_DBM, REGION, MARGIN_OK_DB, load_home, target_points, VALLEY, KOWLOON  # noqa: E402

names = sys.argv[2:]
cands = json.load(open(sys.argv[1]))["features"]
def find(n):
    hits = [f for f in cands if (f["properties"].get("name") or "").lower() == n.lower()]
    return max(hits, key=lambda f: f["properties"]["ground"]) if hits else None
z, tr = hkgeo.load_dem(REGION); T = Terrain(z, tr)
home, home_ant, _ = load_home(ROOT / "out" / "nodes.geojson")
relay = Antenna(**RELAY)
sites = {"HOME": (home, home_ant)}
for n in names:
    f = find(n)
    if f is None:
        print("not found:", n); continue
    p = f["properties"]
    sites[n] = ((p["e"], p["n"]), relay)
    print(f"{n:<16} z{p['ground']:>4.0f}  E{p['e']:.0f} N{p['n']:.0f}  home {p['home_margin_db']}  valley {p['valley_cov']}  kln {p['kowloon_cov']}  "
          f"access {p['way_m']} m ({p['way_type']}){'  MAST' if p.get('mast') else ''}")
print()
keys = list(sites)
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        (pa, aa), (pb, ab) = sites[a], sites[b]
        L = link(T, pa, pb, aa, ab, F_MHZ, SENS_DBM)
        sec = ""
        if a == "HOME":
            from tools.rf import bearing_deg, in_sector
            az = float(bearing_deg(pa[0], pa[1], pb[0], pb[1]))
            sec = f" az {az:5.1f}" + ("" if in_sector(az, home_ant.azimuth_deg, home_ant.arc_deg) else " OUTSIDE SECTOR")
        print(f"{a:>14} - {b:<14} {L.d_km:5.1f} km {'LOS ' if L.los else 'nlos'} diff {L.diffraction_db:5.1f} dB  margin {L.margin_db:5.1f} dB{sec}")
