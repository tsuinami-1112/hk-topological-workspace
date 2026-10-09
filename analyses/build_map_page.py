#!/usr/bin/env python3
"""Build out/map/index.html from map_template.html and the layers/data written by render_map.py.

Adds WGS84 coordinates, the balcony link for each site and the selection presets, then inlines
everything into the page. Links come from analyses/building_check.py (building-aware).

Usage: python analyses/build_map_page.py --map out/map
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import hkgeo  # noqa: E402

PRESETS = [
    {"label": "Proposed: Needle Hill + Kowloon Peak", "sites": ["Needle Hill", "Kowloon Peak"]},
    {"label": "+ Ma On Shan", "sites": ["Needle Hill", "Kowloon Peak", "Ma On Shan"]},
    {"label": "+ Grassy Hill for redundancy", "sites": ["Needle Hill", "Kowloon Peak", "Ma On Shan", "Grassy Hill"]},
    {"label": "With rooftop: Tate's Cairn alone", "sites": ["Tate's Cairn"]},
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(ROOT / "out" / "map"))
    a = ap.parse_args(argv)
    out = Path(a.map)
    data = json.loads((out / "data.json").read_text())
    home_name = next(s["name"] for s in data["sites"] if "home" in s["roles"])
    # the home site sits at the antenna position on the facade
    for s in data["sites"]:
        if s["name"] == home_name:
            s["e"], s["n"] = data["home"]["e"], data["home"]["n"]
        lon, lat = hkgeo.to_wgs84(s["e"], s["n"])
        s["lon"], s["lat"] = round(float(lon), 6), round(float(lat), 6)
    pairs = []
    for p in data["pairs"]:
        pairs.append({**p, "blocked": False, "terrain_margin_db": p["margin_db"]})
    by_name = {s["name"]: s for s in data["sites"]}
    for p in data["home_pairs"]:
        pairs.append({**p, "a": home_name})
        if p["b"] in by_name:
            by_name[p["b"]]["home_link"] = {"los": p["los"], "blocked": p["blocked"], "margin_db": p["margin_db"],
                                            "terrain_margin_db": p["terrain_margin_db"], "az": p["az"]}
    data["pairs"] = pairs
    data.pop("home_pairs", None)
    data["presets"] = PRESETS
    tpl = (ROOT / "analyses" / "map_template.html").read_text()
    css = (ROOT / "analyses" / "vendor" / "leaflet-1.9.4.css").read_text()
    html = tpl.replace("/*__LEAFLET_CSS__*/", css).replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    (out / "index.html").write_text(html)
    print("index.html", round(len(html) / 1e3), "kB;", len(pairs), "links")


if __name__ == "__main__":
    main()
