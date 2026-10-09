#!/usr/bin/env python3
"""Build out/map/index.html from map_template.html and the rendered layers (render_map.py).

Adds WGS84 coordinates, every pairwise link among the mapped sites (P.526, terrain only; the
balcony's sector enforced) and the selection presets, then inlines it all into the page.

Usage: python analyses/build_map_page.py --map out/map
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analyses"))
from tools import hkgeo  # noqa: E402
from tools.rf import Antenna, Terrain, bearing_deg, in_sector, link  # noqa: E402
from home_relays import F_MHZ, RELAY, REGION, SENS_DBM, load_home  # noqa: E402

PRESETS = [
    {"label": "Proposed: Grassy Hill + Kowloon Peak", "sites": ["Grassy Hill", "Kowloon Peak"]},
    {"label": "+ Ma On Shan", "sites": ["Grassy Hill", "Kowloon Peak", "Ma On Shan"]},
    {"label": "+ Needle Hill as backup", "sites": ["Grassy Hill", "Kowloon Peak", "Needle Hill"]},
    {"label": "With rooftop: Tate's Cairn alone", "sites": ["Tate's Cairn"]},
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(ROOT / "out" / "map"))
    ap.add_argument("--registry", default=str(ROOT / "out" / "nodes.geojson"))
    a = ap.parse_args(argv)
    out = Path(a.map)
    data = json.loads((out / "data.json").read_text())

    z, tr = hkgeo.load_dem(REGION)
    terrain = Terrain(z, tr)
    home_en, home_ant, _ = load_home(Path(a.registry))
    relay = Antenna(**RELAY)

    for s in data["sites"]:
        lon, lat = hkgeo.to_wgs84(s["e"], s["n"])
        s["lon"], s["lat"] = round(float(lon), 6), round(float(lat), 6)

    sites = data["sites"]
    pairs = []
    for i, sa in enumerate(sites):
        for sb in sites[i + 1:]:
            a_home, b_home = "home" in sa["roles"], "home" in sb["roles"]
            pa, pb = (sa["e"], sa["n"]), (sb["e"], sb["n"])
            if b_home:
                sa, sb, pa, pb, a_home, b_home = sb, sa, pb, pa, b_home, a_home
            L = link(terrain, pa, pb, home_ant if a_home else relay, relay, F_MHZ, SENS_DBM)
            rec = {"a": sa["name"], "b": sb["name"], "d_km": round(L.d_km, 2), "los": L.los,
                   "terrain_margin_db": round(L.margin_db, 1)}
            blocked = False
            if a_home:
                az = float(bearing_deg(pa[0], pa[1], pb[0], pb[1]))
                blocked = not bool(in_sector(az, home_ant.azimuth_deg, home_ant.arc_deg))
                rec["az"] = round(az, 1)
            rec["blocked"] = blocked
            rec["margin_db"] = None if blocked else round(L.margin_db, 1)
            pairs.append(rec)
            if a_home:
                sb["home_link"] = {"los": L.los, "blocked": blocked, "margin_db": rec["margin_db"],
                                   "terrain_margin_db": rec["terrain_margin_db"], "az": rec["az"]}
    data["pairs"] = pairs
    data["presets"] = PRESETS

    tpl = (ROOT / "analyses" / "map_template.html").read_text()
    css = (ROOT / "analyses" / "vendor" / "leaflet-1.9.4.css").read_text()
    html = tpl.replace("/*__LEAFLET_CSS__*/", css).replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    (out / "index.html").write_text(html)
    print("index.html", round(len(html) / 1e3), "kB;", len(pairs), "pairs")


if __name__ == "__main__":
    main()
