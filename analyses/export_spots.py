#!/usr/bin/env python3
"""Score and tier the spots from analyses/spots.py and export overlay files.

Outputs (out/spots/):
  sentinel_spots.kmz      Google Earth / Google My Maps: tiered hill spots with their open-ground
                          zones, tiered rooftops with footprints, planned sites, the balcony sector
  sentinel_spots.geojson  the same plus every scored candidate (tier C), WGS84, for QGIS / Felt /
                          geojson.io
  ranked.json             all candidates with scores, for later sessions

Hill score (0-100) = 20 link reach + 15 LoS reach + 5 airspace view + 15 open horizon + 15 not
overgrown + 10 altitude + 10 solar + 10 access. Rooftop score = 25 link reach + 15 LoS reach +
10 airspace view + 25 open horizon + 10 height + 15 solar. Reach and view terms are scaled to the
95th percentile of their kind; altitude/height is a percentile rank.
Tiers: A = top 20 of each kind, B = next 40, plus the top 6 for each area (Sha Tin valley,
Kowloon, Hong Kong Island; scored with that area's reach) and, for rooftops, the top 5 located in
each area, lifted to at least B. C = the rest.
Balcony relay picks: the 8 spots (hill or rooftop) with the best score + 0.5 x (home margin - 20,
capped at 70) + 5 for line of sight, among spots the balcony reaches with >= 20 dB. They get their
own folder; every spot appears once in the KMZ.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import hkgeo  # noqa: E402

AREA_LABEL = {"valley": "Sha Tin valley", "kowloon": "Kowloon", "hk_island": "HK Island"}
AOI_B = (824000.0, 806000.0, 852000.0, 838000.0)
LOCATED = {"Sha Tin valley": (833000, 823500, 846500, 833500), "Kowloon": (832000, 816600, 846000, 823500),
           "HK Island": (826000, 806000, 850000, 816600)}
N_LOCAL, N_RELAY = 5, 8
N_A, N_B, N_AREA = 20, 40, 6
ZONE_R = 200.0
DATE = "2026-10-11"


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def scale95(v):
    v = np.asarray(v, float)
    ref = np.nanpercentile(v, 95) if np.isfinite(v).any() else 1.0
    return clip01(v / ref) if ref > 0 else np.zeros_like(v)


def rank01(v):
    v = np.asarray(v, float)
    order = v.argsort().argsort()
    return order / max(len(v) - 1, 1)


def score(rows, kind, reach_key="reach_all"):
    g = lambda k: np.array([np.nan if r.get(k) is None else r[k] for r in rows], float)  # noqa: E731
    link, los, air = scale95(g(reach_key)), scale95(g("reach_los")), scale95(g("air_view"))
    opn = clip01(g("open_frac"))
    solar = clip01((g("solar_frac") - 0.8) / 0.2)
    if kind == "hill":
        veg = 1 - clip01((g("canopy_25m") - 2) / 8)
        alt = rank01(g("ground"))
        path = g("path_m")
        acc = np.where(path <= 25, 1.0, 1.0 - 0.5 * clip01((path - 25) / 95))
        s = 20 * link + 15 * los + 5 * air + 15 * opn + 15 * veg + 10 * alt + 10 * solar + 10 * acc
    else:
        alt = rank01(g("antenna_mpd"))
        s = 25 * link + 15 * los + 10 * air + 25 * opn + 10 * alt + 15 * solar
    return np.round(s, 1)


def compass(az):
    return ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"][int((az % 360) / 22.5 + 0.5) % 16]


def title_case(s):
    if not s:
        return s
    return s.title() if s.isupper() else s


PEAKS = None


def nearest_peak(e, n, max_m=2000.0):
    global PEAKS
    if PEAKS is None:
        g = hkgeo.load_vector("osm", (AOI_B[0] - 2000, AOI_B[1] - 2000, AOI_B[2] + 2000, AOI_B[3] + 2000), layer="peaks", pad=0)
        nm = [a if isinstance(a, str) else (b if isinstance(b, str) else None) for a, b in zip(g["name:en"], g["name"])]
        keep = [i for i, x in enumerate(nm) if x and not any("\u4e00" <= ch <= "\u9fff" for ch in x)]
        PEAKS = (np.array([g.geometry.x.values[i] for i in keep]), np.array([g.geometry.y.values[i] for i in keep]),
                 [nm[i] for i in keep])
    px, py, names = PEAKS
    dd = np.hypot(px - e, py - n)
    j = int(np.argmin(dd))
    if dd[j] > max_m:
        return None, None, None
    return names[j], float(dd[j]), float(np.degrees(np.arctan2(e - px[j], n - py[j])) % 360)


def where(e, n):
    pk, dist, az = nearest_peak(e, n)
    if not pk:
        return None
    if dist <= 120:
        return pk
    d = f"{dist / 1000:.1f} km" if dist >= 1000 else f"{round(dist, -1):.0f} m"
    return f"{pk}, {d} {compass(az)}"


def name_spot(r, i):
    if r["kind"] == "hill":
        return where(r["e"], r["n"]) or f"Hill spot {i}"
    b = title_case(r.get("building"))
    if b:
        return b
    w = where(r["e"], r["n"])
    st = r.get("storeys")
    base = "Unnamed building" + (f" ({st} storeys)" if st else "")
    return f"{base} near {w}" if w else base


def tiers(rows, kind):
    s = score(rows, kind)
    for r, v in zip(rows, s):
        r["score"] = float(v)
    order = np.argsort(-s)
    tier = np.array(["C"] * len(rows), dtype=object)
    tier[order[:N_A]] = "A"
    tier[order[N_A:N_A + N_B]] = "B"
    best_for = [[] for _ in rows]
    for area in AREA_LABEL:
        sa = score(rows, kind, f"reach_{area}")
        for i in np.argsort(-sa)[:N_AREA]:
            best_for[i].append(AREA_LABEL[area])
            if tier[i] == "C":
                tier[i] = "B"
    if kind == "roof":
        for label, (x0, y0, x1, y1) in LOCATED.items():
            inside = [i for i, r in enumerate(rows) if x0 <= r["e"] < x1 and y0 <= r["n"] < y1]
            for i in sorted(inside, key=lambda i: -s[i])[:N_LOCAL]:
                best_for[i].append(f"rooftop in {label}")
                if tier[i] == "C":
                    tier[i] = "B"
    for r, t, b in zip(rows, tier, best_for):
        r["tier"], r["best_for"] = t, b
    rows.sort(key=lambda r: (-{"A": 2, "B": 1, "C": 0}[r["tier"]], -r["score"]))
    prefix = "H" if kind == "hill" else "R"
    for i, r in enumerate(rows, 1):
        r["sid"] = f"{prefix}{i:03d}"
        r["name"] = name_spot(r, r["sid"])
    return rows


# ------------------------------------------------------------------ geometry

def to_ll(e, n):
    lon, lat = hkgeo.to_wgs84(e, n)
    return float(lon), float(lat)


def poly_ll(geom):
    """Shapely (Multi)Polygon in HK1980 -> list of rings (lists of lon/lat) per polygon."""
    from shapely.geometry import MultiPolygon
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    out = []
    for p in polys:
        rings = []
        for ring in [p.exterior] + list(p.interiors):
            xs, ys = ring.coords.xy
            lon, lat = hkgeo.to_wgs84(np.array(xs), np.array(ys))
            rings.append([[round(float(a), 6), round(float(b), 6)] for a, b in zip(lon, lat)])
        out.append(rings)
    return out


def zones_for(rows, mask_path):
    """Open-ground area within ZONE_R of each tier A/B hill spot, as polygons (HK1980)."""
    import rasterio
    from rasterio.features import shapes
    from rasterio.windows import from_bounds
    from shapely.geometry import Point, shape
    from shapely.ops import unary_union
    out = {}
    with rasterio.open(mask_path) as src:
        for r in rows:
            if r["tier"] not in ("A", "B"):
                continue
            e, n = r["e"], r["n"]
            win = from_bounds(e - ZONE_R - 20, n - ZONE_R - 20, e + ZONE_R + 20, n + ZONE_R + 20,
                              transform=src.transform).round_offsets().round_lengths()
            m = src.read(1, window=win, boundless=True, fill_value=0)
            tr = src.window_transform(win)
            polys = [shape(g) for g, v in shapes(m, mask=m == 1, transform=tr) if v == 1]
            if not polys:
                continue
            disk = Point(e, n).buffer(ZONE_R, 32)
            u = unary_union(polys).intersection(disk)
            if u.is_empty:
                continue
            parts = list(getattr(u, "geoms", [u]))
            keep = [p for p in parts if p.distance(Point(e, n)) < 15 or p.area >= 2000]
            if not keep:
                continue
            z = unary_union(keep).simplify(4, preserve_topology=True)
            out[r["sid"]] = (z, round(float(z.area)))
    return out


def footprints_for(rows):
    out = {}
    want = [r for r in rows if r["tier"] in ("A", "B")]
    for r in want:
        e, n = r["e"], r["n"]
        b = hkgeo.load_vector("buildings", (e - 5, n - 5, e + 5, n + 5), pad=300)
        b = b[b["OBJECTID"] == r["objectid"]]
        if len(b):
            out[r["sid"]] = b.geometry.iloc[0]
    return out


# ------------------------------------------------------------------ text

def pct(v):
    return "–" if v is None else f"{round(100 * v)}%"


def db(v):
    return "–" if v is None else f"{v:.0f} dB"


def describe(r, zone_area=None):
    lon, lat = to_ll(r["e"], r["n"])
    rows = []
    if r["kind"] == "hill":
        rows.append(("Ground", f"{r['ground']:.0f} mPD (antenna on a 4 m pole: {r['antenna_mpd']:.0f} mPD)"))
        rows.append(("Relief", f"{r['relief_m']:.0f} m above the surrounding 300 m · slope {r['slope_deg']:.0f}°"))
        rows.append(("Vegetation", f"tallest canopy within 10 m: {r['canopy_10m']:.0f} m · within 25 m: {r['canopy_25m']:.0f} m"
                     + (f" · PlanD land-use map: {r['landuse']}" if r.get("landuse") else "")
                     + (f" · woodland within 50 m: {pct(r.get('woodland_50m'))}" if r.get("woodland_50m") else "")))
        rows.append(("Access", f"{r['path_m']} m to a mapped {r.get('path_type') or 'path'}"))
        if zone_area:
            rows.append(("Open ground", f"{zone_area:,} m² of open, accessible ground within 200 m (the shaded zone)"))
        if r.get("structures_40m"):
            rows.append(("Nearby", f"{r['structures_40m']} structure(s) within 40 m" + (f": {r['structure_names']}" if r.get("structure_names") else "")))
    else:
        nm = r.get("building_tc")
        rows.append(("Roof", f"{r['roof']:.0f} mPD" + (f" · {r['storeys']} storeys" if r.get("storeys") else "")
                     + f" · ground {r['ground']:.0f} mPD · footprint {r['footprint_m2']:,.0f} m²" + (f" · {nm}" if nm else "")
                     + (f" · PlanD land-use map: {r['landuse']}" if r.get("landuse") else "")))
        rows.append(("Antenna", f"2 m mast on the roof: {r['antenna_mpd']:.0f} mPD"))
    rows.append(("Link reach", f"rooftops reached with ≥20 dB: valley {pct(r['reach_valley'])}, Kowloon {pct(r['reach_kowloon'])}, "
                 f"HK Island {pct(r['reach_hk_island'])}, whole area {pct(r['reach_all'])}"))
    rows.append(("Line of sight", f"clear LoS to {pct(r['reach_los'])} of rooftops; sees {pct(r['air_view'])} of the airspace 60 m up within 3 km"))
    ob = r["worst_obstr_deg"]
    rows.append(("Horizon", f"{pct(r['open_frac'])} clear within 300 m"
                 + (f" (worst: {ob:.0f}° above the antenna to the {compass(r['worst_obstr_az'])})" if ob > 1 else "")))
    rows.append(("Solar", f"{pct(r['solar_frac'])} of clear-sky sun on a south-facing panel"))
    hd = r.get("home_db")
    rows.append(("Home link", f"{db(hd)}{' line of sight' if r.get('home_los') else (' diffracted' if hd is not None else '')}"
                 if hd is not None else f"outside the balcony's clear sector (bearing {r['az_from_home']:.0f}° from home)"))
    if r.get("backbone_site"):
        rows.append(("Best planned relay", f"{r['backbone_site']}: {db(r['backbone_db'])}{' LoS' if r.get('backbone_los') else ''}"))
    if r.get("best_for"):
        rows.append(("Top pick for", ", ".join(r["best_for"])))
    if r.get("registry_site"):
        rows.append(("Registry", f"same spot as {r['registry_site']}"))
    rows.append(("Position", f"{lat:.6f}, {lon:.6f} · HK1980 E{r['e']:.0f} N{r['n']:.0f}"))
    gm = f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"
    html = (f"<p><b>{escape(r['sid'])}</b> · Tier {r['tier']} · score {r['score']:.0f}/100</p><table>"
            + "".join(f"<tr><td valign='top'><b>{escape(k)}</b></td><td>{escape(v)}</td></tr>" for k, v in rows)
            + f"</table><p><a href='{gm}'>Open in Google Maps</a></p>")
    return html, rows


FIELDS = ["sid", "name", "kind", "tier", "score", "ground", "antenna_mpd", "roof", "storeys", "reach_all", "reach_valley",
          "reach_kowloon", "reach_hk_island", "reach_los", "air_view", "open_frac", "worst_obstr_deg", "worst_obstr_az",
          "solar_frac", "canopy_10m", "canopy_25m", "landuse", "woodland_50m", "relief_m", "slope_deg", "path_m",
          "path_type", "home_db", "home_los", "az_from_home", "home_km", "backbone_site", "backbone_db", "structures_40m",
          "structure_names", "masts_100m", "peak_name", "building", "building_tc", "registry_site"]


# ------------------------------------------------------------------ KML

def kml_color(hex_rgb, alpha):
    r, g, b = hex_rgb[1:3], hex_rgb[3:5], hex_rgb[5:7]
    return f"{alpha:02x}{b}{g}{r}"


STYLES = {
    "hillA": ("#1a9850", "placemark_circle", 1.2, 0.9), "hillB": ("#f4a300", "placemark_circle", 0.9, 0.0),
    "hillC": ("#9e9e9e", "placemark_circle", 0.6, 0.0),
    "roofA": ("#2c7fb8", "placemark_square", 1.1, 0.9), "roofB": ("#7fcdbb", "placemark_square", 0.85, 0.0),
    "site": ("#9b3d8a", "star", 1.3, 1.0), "relay": ("#d7301f", "target", 1.3, 0.9),
}


def kml_styles():
    out = []
    for sid, (col, icon, scale, label) in STYLES.items():
        out.append(f"""<Style id="{sid}"><IconStyle><color>{kml_color(col, 255)}</color><scale>{scale}</scale>
<Icon><href>https://maps.google.com/mapfiles/kml/shapes/{icon}.png</href></Icon></IconStyle>
<LabelStyle><scale>{label}</scale></LabelStyle><BalloonStyle><text>$[description]</text></BalloonStyle></Style>""")
    for sid, col, fill, width in (("zoneA", "#1a9850", 0x55, 2), ("zoneB", "#f4a300", 0x40, 1.5),
                                  ("footA", "#2c7fb8", 0x80, 2), ("footB", "#7fcdbb", 0x70, 1.5),
                                  ("sector", "#9b3d8a", 0x26, 1.5)):
        out.append(f"""<Style id="{sid}"><LineStyle><color>{kml_color(col, 255)}</color><width>{width}</width></LineStyle>
<PolyStyle><color>{kml_color(col, fill)}</color></PolyStyle><BalloonStyle><text>$[description]</text></BalloonStyle></Style>""")
    return "\n".join(out)


def kml_point(name, style, desc, lon, lat, data=None, visible=True):
    ext = ""
    if data:
        ext = "<ExtendedData>" + "".join(f'<Data name="{escape(k)}"><value>{escape(str(v))}</value></Data>'
                                         for k, v in data.items() if v is not None) + "</ExtendedData>"
    return (f"<Placemark><name>{escape(name)}</name><visibility>{int(visible)}</visibility><styleUrl>#{style}</styleUrl>"
            f"<description><![CDATA[{desc}]]></description>{ext}"
            f"<Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point></Placemark>")


def kml_polygon(name, style, desc, polys, visible=True):
    geoms = []
    for rings in polys:
        outer = " ".join(f"{a},{b},0" for a, b in rings[0])
        inner = "".join(f"<innerBoundaryIs><LinearRing><coordinates>{' '.join(f'{a},{b},0' for a, b in r)}</coordinates></LinearRing></innerBoundaryIs>"
                        for r in rings[1:])
        geoms.append(f"<Polygon><tessellate>1</tessellate><outerBoundaryIs><LinearRing><coordinates>{outer}</coordinates></LinearRing></outerBoundaryIs>{inner}</Polygon>")
    g = geoms[0] if len(geoms) == 1 else "<MultiGeometry>" + "".join(geoms) + "</MultiGeometry>"
    return (f"<Placemark><name>{escape(name)}</name><visibility>{int(visible)}</visibility><styleUrl>#{style}</styleUrl>"
            f"<description><![CDATA[{desc}]]></description>{g}</Placemark>")


def folder(name, items, open_=False, visible=True, desc=None):
    d = f"<description><![CDATA[{desc}]]></description>" if desc else ""
    return f"<Folder><name>{escape(name)}</name><open>{int(open_)}</open><visibility>{int(visible)}</visibility>{d}{''.join(items)}</Folder>"


def sector_polygon(home):
    e, n = home["e"], home["n"]
    az0, az1 = home["az"] - home["arc"] / 2, home["az"] + home["arc"] / 2
    pts = [(e, n)] + [(e + 3000 * math.sin(math.radians(a)), n + 3000 * math.cos(math.radians(a)))
                      for a in np.linspace(az0, az1, 41)] + [(e, n)]
    lon, lat = hkgeo.to_wgs84(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    return [[[round(float(a), 6), round(float(b), 6)] for a, b in zip(lon, lat)]]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spots", default=str(ROOT / "out" / "spots"))
    ap.add_argument("--registry", default=str(ROOT / "out" / "nodes.geojson"))
    a = ap.parse_args(argv)
    sp = Path(a.spots)
    data = json.loads((sp / "candidates.json").read_text())
    rows = data["rows"]
    hills = tiers([r for r in rows if r["kind"] == "hill"], "hill")
    roofs = tiers([r for r in rows if r["kind"] == "roof"], "roof")
    pool = [r for r in hills + roofs if (r.get("home_db") or -99) >= 20]
    for r in pool:
        r["relay_score"] = round(r["score"] + 0.5 * (min(r["home_db"], 70) - 20) + (5 if r.get("home_los") else 0), 1)
    picks = sorted(pool, key=lambda r: -r["relay_score"])[:N_RELAY]
    for r in picks:
        r["relay_pick"] = True
        r["best_for"] = ["first relay from the balcony"] + list(r.get("best_for") or [])
        if r["tier"] == "C":
            r["tier"] = "B"
    reg0 = json.loads(Path(a.registry).read_text())
    for r in hills + roofs:
        for f in reg0["features"]:
            q = f["properties"]
            if np.hypot(q["hk80"]["e"] - r["e"], q["hk80"]["n"] - r["n"]) <= 60:
                r["registry_site"] = f"{q['id']} {q['name']}"
    zones = zones_for(hills, sp / "open_ground_10m.tif")
    foots = footprints_for(roofs)
    reg = json.loads(Path(a.registry).read_text())

    # ---------------- KML
    parts = []
    reg_items = []
    home_info = None
    for f in reg["features"]:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        ant = next((x for x in p.get("antennas", []) if x.get("band") == "lora"), {})
        desc = (f"<p><b>{escape(p['id'])}</b> · {escape(p['status'])} · {escape(', '.join(p['roles']))}</p>"
                f"<p>{escape(p.get('notes') or '')}</p><p>Antenna {ant.get('mpd')} mPD, {ant.get('gain_dbi')} dBi</p>")
        reg_items.append(kml_point(f"{p['id']} {p['name']}", "site", desc, lon, lat))
        if "home" in p["roles"]:
            home_info = {"e": p["hk80"]["e"], "n": p["hk80"]["n"], "az": ant["azimuth_deg"], "arc": ant["visible_arc_deg"]}
    if home_info:
        reg_items.append(kml_polygon("Balcony clear sector (3 km)", "sector",
                                     f"Planning sector {home_info['az'] - home_info['arc'] / 2:.0f}° to {(home_info['az'] + home_info['arc'] / 2) % 360:.0f}° from the balcony.",
                                     [sector_polygon(home_info)]))
    parts.append(folder("Planned sites (registry)", reg_items, open_=True))

    def spot_items(rs, tier, kind, relay=False):
        items, zitems = [], []
        for r in rs:
            if bool(r.get("relay_pick")) != relay or (tier and r["tier"] != tier):
                continue
            z = zones.get(r["sid"])
            desc, _ = describe(r, z[1] if z else None)
            lon, lat = to_ll(r["e"], r["n"])
            data_ = {k: r.get(k) for k in FIELDS}
            t = r["tier"] if r["tier"] in ("A", "B") else "B"
            items.append(kml_point(f"{r['sid']} {r['name']}", "relay" if relay else f"{kind}{t}", desc, lon, lat, data_))
            if kind == "hill" and z:
                zitems.append(kml_polygon(f"{r['sid']} open ground", f"zone{t}", desc, poly_ll(z[0])))
            if kind == "roof" and r["sid"] in foots:
                zitems.append(kml_polygon(f"{r['sid']} footprint", f"foot{t}", desc, poly_ll(foots[r["sid"]])))
        return items, zitems

    pH, pzH = spot_items(hills, None, "hill", relay=True)
    pR, pzR = spot_items(roofs, None, "roof", relay=True)
    parts.append(folder("First relay from the balcony · picks", pH + pR + pzH + pzR, open_=True,
                        desc="Spots the balcony reaches with at least 20 dB (its clear sector enforced), ranked by overall score plus link strength. Candidates for the first hop out of the home station."))
    hA, zA = spot_items(hills, "A", "hill")
    hB, zB = spot_items(hills, "B", "hill")
    rA, fA = spot_items(roofs, "A", "roof")
    rB, fB = spot_items(roofs, "B", "roof")
    parts.append(folder("Hill spots · Tier A", hA + zA, open_=True,
                        desc="Open (no canopy over 3 m within ~15 m), accessible (≤120 m to a mapped path or road) high ground. Shaded zones: open, accessible ground within 200 m of the spot."))
    parts.append(folder("Hill spots · Tier B", hB + zB))
    parts.append(folder("Rooftops · Tier A", rA + fA, open_=True, desc="Rooftops that dominate their 120 m surroundings; footprint outlined."))
    parts.append(folder("Rooftops · Tier B", rB + fB))
    method = ("Scores combine link reach (≥20 dB to the tallest rooftop in each 300 m cell), direct line of sight, "
              "airspace view, open horizon within 300 m, vegetation, altitude, solar access and path access. "
              "Surface model: LandsD 5 m terrain + LandsD building heights + Meta/WRI canopy height; "
              "Meshtastic LONG_MODERATE at 915 MHz, 6 dBi antennas, 28 dBm.")
    kml = (f'<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
           f"<name>Sentinel spots of interest ({DATE})</name><open>1</open><description><![CDATA[{method}]]></description>"
           + kml_styles() + "".join(parts) + "</Document></kml>")
    kmz = sp / "sentinel_spots.kmz"
    with zipfile.ZipFile(kmz, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml)

    # ---------------- GeoJSON
    feats = []
    for f in reg["features"]:
        p = f["properties"]
        feats.append({"type": "Feature", "geometry": f["geometry"],
                      "properties": {"layer": "planned site", "sid": p["id"], "name": p["name"], "status": p["status"],
                                     "roles": ", ".join(p["roles"])}})
    if home_info:
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": sector_polygon(home_info)},
                      "properties": {"layer": "balcony sector", "name": "Balcony clear sector (3 km)"}})
    for kind, rs in (("hill spot", hills), ("rooftop", roofs)):
        for r in rs:
            lon, lat = to_ll(r["e"], r["n"])
            props = {"layer": kind, **{k: r.get(k) for k in FIELDS}, "best_for": ", ".join(r.get("best_for") or []) or None,
                     "relay_pick": bool(r.get("relay_pick")), "relay_score": r.get("relay_score")}
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                          "properties": props})
            if kind == "hill spot" and r["sid"] in zones:
                pl = poly_ll(zones[r["sid"]][0])
                geom = {"type": "Polygon", "coordinates": pl[0]} if len(pl) == 1 else {"type": "MultiPolygon", "coordinates": pl}
                feats.append({"type": "Feature", "geometry": geom,
                              "properties": {"layer": "open ground zone", "sid": r["sid"], "name": r["name"], "tier": r["tier"],
                                             "area_m2": zones[r["sid"]][1]}})
            if kind == "rooftop" and r["sid"] in foots:
                pl = poly_ll(foots[r["sid"]])
                geom = {"type": "Polygon", "coordinates": pl[0]} if len(pl) == 1 else {"type": "MultiPolygon", "coordinates": pl}
                feats.append({"type": "Feature", "geometry": geom,
                              "properties": {"layer": "rooftop footprint", "sid": r["sid"], "name": r["name"], "tier": r["tier"]}})
    gj = {"type": "FeatureCollection", "name": f"sentinel_spots_{DATE}", "features": feats}
    (sp / "sentinel_spots.geojson").write_text(json.dumps(gj, ensure_ascii=False, default=lambda v: None))
    (sp / "ranked.json").write_text(json.dumps({"meta": data["meta"], "hills": hills, "roofs": roofs}, ensure_ascii=False,
                                               default=lambda v: None))
    n = lambda rs, t: sum(r["tier"] == t for r in rs)  # noqa: E731
    print(f"hills: A {n(hills, 'A')}, B {n(hills, 'B')}, C {n(hills, 'C')}; zones {len(zones)}")
    print(f"roofs: A {n(roofs, 'A')}, B {n(roofs, 'B')}, C {n(roofs, 'C')}; footprints {len(foots)}")
    print("relay picks:", [(r["sid"], r["name"], r.get("home_db"), r.get("home_los"), r["relay_score"]) for r in picks])
    print(f"{kmz.name}: {kmz.stat().st_size / 1e6:.2f} MB; geojson: {(sp / 'sentinel_spots.geojson').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
