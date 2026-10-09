#!/usr/bin/env python3
"""Union coverage of small relay sets over the Sha Tin valley and Kowloon target points."""
import itertools, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analyses"))
from tools import hkgeo  # noqa: E402
from tools.rf import Antenna, Terrain, link  # noqa: E402
from home_relays import RELAY, ROOFTOP, F_MHZ, SENS_DBM, REGION, MARGIN_OK_DB, VALLEY, KOWLOON, target_points  # noqa: E402

cands = json.load(open(ROOT / "out/home_relays/candidates.geojson"))["features"]
def site(n):
    hits = [f["properties"] for f in cands if (f["properties"].get("name") or "").lower() == n.lower()]
    p = max(hits, key=lambda p: p["ground"]); return (p["e"], p["n"])
names = sys.argv[1:]
z, tr = hkgeo.load_dem(REGION); T = Terrain(z, tr)
valley, kowloon = target_points(T, VALLEY), target_points(T, KOWLOON)
relay, roof = Antenna(**RELAY), Antenna(**ROOFTOP)
cov = {}
for n in names:
    s = site(n)
    mv = np.array([link(T, s, tuple(p), relay, roof, F_MHZ, SENS_DBM, step_m=10).margin_db for p in valley])
    mk = np.array([link(T, s, tuple(p), relay, roof, F_MHZ, SENS_DBM, step_m=10).margin_db for p in kowloon])
    cov[n] = (mv, mk)
out = {"valley_points": valley.tolist(), "kowloon_points": kowloon.tolist(), "sites": {n: list(site(n)) for n in names},
       "margins": {n: {"valley": cov[n][0].round(1).tolist(), "kowloon": cov[n][1].round(1).tolist()} for n in names}}
(ROOT / "out/home_relays/relay_set_margins.json").write_text(json.dumps(out))
def union(ns):
    v = np.max([cov[n][0] for n in ns], axis=0); k = np.max([cov[n][1] for n in ns], axis=0)
    return (v >= MARGIN_OK_DB).mean(), (k >= MARGIN_OK_DB).mean(), v, k
rows = []
for r in (1, 2, 3):
    for ns in itertools.combinations(names, r):
        uv, uk, _, _ = union(ns); rows.append((uv, uk, ns))
print("best sets by valley + kowloon coverage (terrain only, >= 20 dB)")
for r in (1, 2, 3):
    best = sorted([x for x in rows if len(x[2]) == r], key=lambda x: -(x[0] + x[1]))[:5]
    for uv, uk, ns in best:
        print(f"  {r}: valley {uv:.0%}  kowloon {uk:.0%}  {' + '.join(ns)}")
