#!/usr/bin/env python3
"""Spots of interest: open high ground and rooftops across Sha Tin, Kowloon and Hong Kong Island.

Criteria (2026-10-11): not overgrown (signal and solar), high, good direct line-of-sight coverage
with no surrounding obstructions; rooftops considered; hill spots reachable on foot or by road.

Surface model: LandsD 5 m DTM (bare ground in practice: surveyed peaks under >10 m of canopy read
only ~2 m high) + Meta/WRI canopy height (mean per 5 m cell) + LandsD building tops. Trees count as
solid obstructions. Garden Vista Block B is left out so the home antenna on its facade is not
blocked by its own building.

Stage 1, every 10 m cell of the AOI. Open ground = land, no building in the cell, no canopy over
OPEN_CANOPY_M within ~15 m, slope <= 30 deg, >= 10 m above its ~300 m surroundings, >= 50 mPD,
within ACCESS_M of a mapped path or road. Hill spots = the highest open cell in each 250 m.
Rooftops = buildings with a measured top, the tallest (within 3 m) in 120 m, footprint >= 200 m2,
one per 300 m (so every district is represented), at most ROOF_CAP.

Stage 2, every candidate, a radial sweep of the surface model (720 rays, 10 m steps, 20 km):
  reach_*     share of rooftop targets (tallest roof in each 300 m cell, +3 m) reached with
              >= 20 dB: P.526 knife edge at the dominant horizon + Bullington correction,
              6 dBi / 28 dBm / 1 dB cables at both ends, LONG_MODERATE -134 dBm
  air_view    share of the airspace 60 m above ground within 3 km in direct line of sight
  open_frac   share of the horizon with nothing more than 1 deg above the antenna within 300 m
  solar_frac  share of a year's clear-sky direct sun on a 22 deg south-facing panel not blocked
              by terrain, buildings or trees (horizon out to 5 km)
  home_db     margin to the home balcony (its sector enforced), exact 5 m profile
  backbone_*  best margin to a planned relay (registry hilltops), exact 5 m profile
plus canopy within 10 / 25 m, access, nearest named peak, nearby installations.

Usage: python analyses/spots.py --registry out/nodes.geojson [--out out/spots] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import hkgeo  # noqa: E402
from tools.dsm import building_tops  # noqa: E402
from tools.rf import (CE, C_MPS, Antenna, Terrain, bearing_deg, bullington_total,  # noqa: E402
                      elevation_penalty_db, fspl_db, in_sector, knife_edge_db, link)

AOI = (824000.0, 806000.0, 852000.0, 838000.0)
PAD = 5000.0
RES = 10.0
F_MHZ, SENS_DBM, MARGIN_OK = 915.0, -134.0, 20.0
GAIN, HPBW, TX, CABLE = 6.0, 25.0, 28.0, 1.0
POLE_M, ROOF_MAST_M, TARGET_MAST_M = 4.0, 2.0, 3.0
DAZ, STEP, RMAX = 0.5, 10.0, 20000.0
OPEN_R, OPEN_TOL_DEG = 300.0, 1.0
AIR_R, AIR_AGL, AIR_GRID = 3000.0, 60.0, 100.0
SOLAR_R, PANEL_TILT = 5000.0, 22.0
OPEN_CANOPY_M, ACCESS_M = 3.0, 120.0
HILL_SPACING, ROOF_SPACING, ROOF_CAP = 250.0, 300.0, 2500
TARGET_CELL = 300.0
AREAS = {"valley": (833000, 823500, 846500, 833500), "kowloon": (832000, 816600, 846000, 823500)}
HK_ISLAND_SEED = (833600.0, 815400.0)       # Victoria Peak
WALK_EXCLUDE = {"motorway", "motorway_link", "trunk", "trunk_link", "construction", "proposed",
                "raceway", "bus_guideway", "elevator", "platform"}
HOME_BLOCK = "Garden Vista Block B"

G: dict = {}   # read-only globals shared with worker processes (fork)


# ------------------------------------------------------------------ helpers

def pool2(a, fn):
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a[:h, :w].reshape(h // 2, 2, w // 2, 2), axis=(1, 3))


def rc(e, n):
    """Row/col in the 10 m grids for HK1980 coordinates (clipped)."""
    x0, y0, H, W = G["x0"], G["y0"], G["H"], G["W"]
    c = np.clip(((np.asarray(e) - x0) / RES).astype(np.int64), 0, W - 1)
    r = np.clip(((y0 - np.asarray(n)) / RES).astype(np.int64), 0, H - 1)
    return r, c


def thin(points, spacing):
    """Greedy thinning: keep points (already sorted best-first) at least `spacing` apart."""
    kept, cells = [], {}
    for i, (e, n) in enumerate(points):
        ci, cj = int(e // spacing), int(n // spacing)
        ok = True
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for (ke, kn) in cells.get((ci + di, cj + dj), ()):
                    if (ke - e) ** 2 + (kn - n) ** 2 < spacing ** 2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            kept.append(i)
            cells.setdefault((ci, cj), []).append((e, n))
    return kept


def sun_table(lat=22.35, lon=114.17, tz=8.0):
    """Sun azimuth/elevation and clear-sky beam weight on a south-facing PANEL_TILT panel, over a year."""
    days = np.arange(1, 366, 3, dtype=float)
    hours = np.arange(5.0, 19.76, 0.25)
    D, Hh = np.meshgrid(days, hours, indexing="ij")
    gam = 2 * np.pi / 365 * (D - 1 + (Hh - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(gam) - 0.032077 * np.sin(gam)
                    - 0.014615 * np.cos(2 * gam) - 0.040849 * np.sin(2 * gam))
    decl = (0.006918 - 0.399912 * np.cos(gam) + 0.070257 * np.sin(gam) - 0.006758 * np.cos(2 * gam)
            + 0.000907 * np.sin(2 * gam) - 0.002697 * np.cos(3 * gam) + 0.00148 * np.sin(3 * gam))
    tst = Hh * 60 + eqt + 4 * lon - 60 * tz
    ha = np.radians(tst / 4 - 180)
    la = np.radians(lat)
    E = -np.cos(decl) * np.sin(ha)
    N = np.sin(decl) * np.cos(la) - np.cos(decl) * np.cos(ha) * np.sin(la)
    U = np.sin(decl) * np.sin(la) + np.cos(decl) * np.cos(ha) * np.cos(la)
    el = np.degrees(np.arcsin(np.clip(U, -1, 1)))
    az = np.degrees(np.arctan2(E, N)) % 360
    up = el > 0.5
    el, az = el[up], az[up]
    zen = 90 - el
    am = 1 / (np.cos(np.radians(zen)) + 0.50572 * (96.07995 - zen) ** -1.6364)
    dni = 1353 * 0.7 ** (am ** 0.678)
    b = np.radians(PANEL_TILT)
    cos_inc = np.cos(np.radians(zen)) * np.cos(b) + np.sin(np.radians(zen)) * np.sin(b) * np.cos(np.radians(az - 180))
    w = dni * np.clip(cos_inc, 0, None)
    keep = w > 0
    return az[keep], el[keep], w[keep]


# ------------------------------------------------------------------ inputs

def load_inputs(registry: Path):
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from scipy import ndimage

    t0 = time.time()
    R = (AOI[0] - PAD, AOI[1] - PAD, AOI[2] + PAD, AOI[3] + PAD)
    dtm5, tr5 = hkgeo.load_dem(R)
    cmax5, cmean5, _ = hkgeo.load_canopy(R)
    bld = building_tops(R)
    home_blk = bld[bld["BuildingNameEN"].fillna("").str.lower() == HOME_BLOCK.lower()]
    bld = bld[~bld.index.isin(home_blk.index)].sort_values("top_mpd")
    tops5 = rasterize(((g, float(h)) for g, h in zip(bld.geometry, bld["top_mpd"])), out_shape=dtm5.shape,
                      transform=tr5, fill=np.nan, dtype="float32")
    surf5 = np.fmax(dtm5 + np.nan_to_num(cmean5, nan=0.0), tops5)
    surf5[np.isnan(dtm5)] = np.nan
    print(f"inputs loaded in {time.time() - t0:.0f} s: grid {dtm5.shape}, {len(bld)} buildings", flush=True)

    G["T5"] = Terrain(surf5, tr5)                                   # exact links
    G["dsm10"] = pool2(np.nan_to_num(surf5, nan=1.0), np.max).astype("float32")
    G["dtm10"] = pool2(dtm5, np.nanmean).astype("float32")
    G["cmax10"] = pool2(np.nan_to_num(cmax5, nan=0.0), np.max).astype("float32")
    G["bmask10"] = pool2(np.isfinite(tops5), np.any)
    del cmax5, cmean5, tops5, surf5
    G["H"], G["W"] = G["dsm10"].shape
    G["x0"], G["y0"] = tr5.c, tr5.f
    G["tr10"] = Affine(RES, 0, tr5.c, 0, -RES, tr5.f)

    dtm10 = G["dtm10"]
    land = np.isfinite(dtm10) & (dtm10 > 1.5)
    zf = np.where(np.isfinite(dtm10), dtm10, 0.0)
    gy, gx = np.gradient(zf, RES)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    tpi = zf - ndimage.uniform_filter(zf, size=61)
    open_canopy = ndimage.maximum_filter(G["cmax10"], size=3) <= OPEN_CANOPY_M
    G["c25"] = ndimage.maximum_filter(G["cmax10"], size=5)
    G["c10"] = ndimage.maximum_filter(G["cmax10"], size=3)

    ways = hkgeo.load_vector("osm", R, layer="ways", pad=0)
    walk = ways[~ways["highway"].isin(WALK_EXCLUDE)]
    wr = rasterize(((g, 1) for g in walk.geometry), out_shape=dtm10.shape, transform=G["tr10"], fill=0,
                   dtype="uint8", all_touched=True)
    G["dist_way"] = (ndimage.distance_transform_edt(wr == 0) * RES).astype("float32")
    G["ways"] = walk[["highway", "name", "geometry"]].reset_index(drop=True)

    r0, c0 = rc(AOI[0], AOI[3])
    r1, c1 = rc(AOI[2], AOI[1])
    in_aoi = np.zeros_like(land)
    in_aoi[r0:r1, c0:c1] = True
    cand = (land & ~G["bmask10"] & open_canopy & (slope <= 30) & (tpi >= 10) & (zf >= 50)
            & (G["dist_way"] <= ACCESS_M) & in_aoi)
    G["slope"], G["tpi"] = slope.astype("float32"), tpi.astype("float32")
    lab, _ = ndimage.label(land, structure=np.ones((3, 3)))
    G["land_lab"] = lab
    G["hki_lab"] = int(lab[rc(*HK_ISLAND_SEED)])
    G["bld"] = bld
    G["lu"] = load_landuse(R, dtm10.shape)
    lu_water = np.isin(G["lu"], [0, 62, 74, 91, 92])
    G["open_mask"] = cand & ~lu_water
    G["peaks"] = hkgeo.load_vector("osm", R, layer="peaks", pad=0)
    G["masts"] = hkgeo.load_vector("osm", R, layer="masts", pad=0)

    # home and planned relays from the registry
    reg = json.loads(registry.read_text())
    relays = []
    for f in reg["features"]:
        p = f["properties"]
        a = next((x for x in p.get("antennas", []) if x.get("band") == "lora"), None)
        if a is None or a.get("mpd") is None:
            continue
        if "home" in p["roles"]:
            G["home"] = {"id": p["id"], "en": (p["hk80"]["e"], p["hk80"]["n"]),
                         "ant": Antenna(gain_dbi=a["gain_dbi"], hpbw_deg=20.0, agl_m=0.0,
                                        tx_dbm=p["radio"]["lora_tx_dbm"], cable_db=1.0,
                                        azimuth_deg=a["azimuth_deg"], arc_deg=a["visible_arc_deg"], amsl_m=a["mpd"])}
        elif p["site_type"] == "hilltop":
            relays.append({"id": p["id"], "name": p["name"], "en": (p["hk80"]["e"], p["hk80"]["n"]),
                           "ant": Antenna(gain_dbi=a.get("gain_dbi") or GAIN, hpbw_deg=HPBW, agl_m=0.0,
                                          tx_dbm=TX, cable_db=a.get("cable_loss_db") or CABLE, amsl_m=a["mpd"])})
    G["relays"] = relays

    # sweep geometry and sun table
    G["az"] = np.arange(0.0, 360.0, DAZ)
    G["sin"], G["cos"] = np.sin(np.radians(G["az"])), np.cos(np.radians(G["az"]))
    G["s"] = np.arange(STEP, RMAX + STEP / 2, STEP)
    G["d"] = G["s"] / 1000.0
    g = np.arange(-AIR_R, AIR_R + 1, AIR_GRID)
    ox, oy = np.meshgrid(g, g)
    rr = np.hypot(ox, oy)
    k = (rr <= AIR_R) & (rr >= 50)
    G["air_dx"], G["air_dy"] = ox[k], oy[k]
    G["sun_az"], G["sun_el"], G["sun_w"] = sun_table()
    print(f"grids ready in {time.time() - t0:.0f} s; open cells {int(cand.sum())}", flush=True)


def load_landuse(R, shape):
    """PlanD land-use codes on the 10 m analysis grid (nearest), -128 where missing."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    import yaml
    out = np.full(shape, -128, np.int16)
    try:
        idx = hkgeo.index("landuse")
    except FileNotFoundError:
        print("land use: not fetched, skipping", flush=True)
        return out
    for t in idx["tiles"]:
        b = t["bounds"]
        if b[0] >= R[2] or b[2] <= R[0] or b[1] >= R[3] or b[3] <= R[1]:
            continue
        with rasterio.open(ROOT / "data" / "landuse" / t["file"]) as src:
            a = src.read(1).astype(np.int16)
            tmp = np.full(shape, -128, np.int16)
            reproject(a, tmp, src_transform=src.transform, src_crs="EPSG:2326", dst_transform=G["tr10"],
                      dst_crs="EPSG:2326", resampling=Resampling.nearest, src_nodata=-128, dst_nodata=-128)
            out = np.where(tmp != -128, tmp, out)
    G["lu_names"] = {int(k): v["name"] for k, v in
                     yaml.safe_load((ROOT / "config" / "landuse_classes.yml").read_text())["classes"].items()}
    return out


def make_targets():
    """Tallest measured roof in each 300 m cell of the AOI, +3 m, labelled by area."""
    b = G["bld"]
    b = b[(b["top_source"] == "TopHeight") & (b.geometry.area >= 100)]
    rp = b.geometry.representative_point()
    inside = (rp.x >= AOI[0]) & (rp.x < AOI[2]) & (rp.y >= AOI[1]) & (rp.y < AOI[3])
    b, rp = b[inside], rp[inside]
    key = (np.floor(rp.x / TARGET_CELL).astype(int) * 100000 + np.floor(rp.y / TARGET_CELL).astype(int)).values
    order = np.lexsort((-b["top_mpd"].values, key))
    first = np.ones(len(order), bool)
    first[1:] = key[order][1:] != key[order][:-1]
    sel = order[first]
    te, tn = rp.x.values[sel], rp.y.values[sel]
    th = b["top_mpd"].values[sel] + TARGET_MAST_M
    r, c = rc(te, tn)
    lab = G["land_lab"][r, c]
    area = np.full(len(sel), "other", dtype=object)
    for name, (x0, y0, x1, y1) in AREAS.items():
        area[(te >= x0) & (te < x1) & (tn >= y0) & (tn < y1)] = name
    area[lab == G["hki_lab"]] = "hk_island"
    G["te"], G["tn"], G["th"], G["tarea"] = te, tn, th, area
    print("targets:", {a: int((area == a).sum()) for a in ("valley", "kowloon", "hk_island", "other")}, flush=True)


def hill_candidates():
    from scipy import ndimage
    m = G["open_mask"]
    z = np.where(m, G["dtm10"], -np.inf)
    mx = ndimage.maximum_filter(z, size=int(HILL_SPACING / RES))
    rows, cols = np.nonzero(m & (z == mx))
    order = np.argsort(-z[rows, cols])
    rows, cols = rows[order], cols[order]
    e = G["x0"] + (cols + 0.5) * RES
    n = G["y0"] - (rows + 0.5) * RES
    keep = thin(list(zip(e, n)), HILL_SPACING)
    out = []
    for i in keep:
        ground = float(G["dtm10"][rows[i], cols[i]])
        out.append({"kind": "hill", "e": float(e[i]), "n": float(n[i]), "ground": ground, "h0": ground + POLE_M})
    return out


def roof_candidates():
    from scipy.spatial import cKDTree
    b = G["bld"]
    b = b[(b["top_source"] == "TopHeight") & (b.geometry.area >= 200)].copy()
    rp = b.geometry.representative_point()
    inside = (rp.x >= AOI[0]) & (rp.x < AOI[2]) & (rp.y >= AOI[1]) & (rp.y < AOI[3])
    b, rp = b[inside], rp[inside]
    xy = np.column_stack([rp.x.values, rp.y.values])
    top = b["top_mpd"].values
    tree = cKDTree(xy)
    dom = np.ones(len(b), bool)
    for i, nb in enumerate(tree.query_ball_point(xy, 120.0)):
        others = [j for j in nb if j != i]
        if others and top[i] < top[others].max() - 3.0:
            dom[i] = False
    idx = np.where(dom)[0]
    idx = idx[np.argsort(-top[idx])]
    keep = thin([tuple(xy[i]) for i in idx], ROOF_SPACING)
    idx = idx[keep][:ROOF_CAP]
    out = []
    for i in idx:
        row = b.iloc[i]
        name = row["BuildingNameEN"] if isinstance(row["BuildingNameEN"], str) else None
        out.append({"kind": "roof", "e": float(xy[i, 0]), "n": float(xy[i, 1]), "ground": float(row["BaseHeight"]),
                    "roof": float(top[i]), "h0": float(top[i]) + ROOF_MAST_M, "building": name,
                    "building_tc": row["BuildingNameTC"] if isinstance(row["BuildingNameTC"], str) else None,
                    "storeys": None if row["Storeys"] != row["Storeys"] else int(row["Storeys"]),
                    "objectid": int(row["OBJECTID"]), "footprint_m2": float(row.geometry.area)})
    return out


# ------------------------------------------------------------------ per-candidate analysis

def analyse(c):
    e0, n0, h0 = c["e"], c["n"], c["h0"]
    s, d = G["s"], G["d"]
    pe = e0 + G["sin"][:, None] * s[None, :]
    pn = n0 + G["cos"][:, None] * s[None, :]
    r, cc = rc(pe, pn)
    z = G["dsm10"][r, cc]
    g = (z - h0) / d[None, :] - 500.0 * CE * d[None, :]
    gmax = np.maximum.accumulate(g, axis=1)
    ns = len(s)
    idx = np.maximum.accumulate(np.where(g >= gmax, np.arange(ns)[None, :], 0), axis=1)
    nr = len(G["az"])

    def lookup(te, tn, th, buffer_m):
        dx, dy = te - e0, tn - n0
        dist = np.hypot(dx, dy)
        ray = np.round((np.degrees(np.arctan2(dx, dy)) % 360) / DAZ).astype(int) % nr
        k = np.clip(np.floor((dist - buffer_m) / STEP).astype(int) - 1, 0, ns - 1)
        dt = dist / 1000.0
        gt = (th - h0) / np.maximum(dt, 1e-6) - 500.0 * CE * dt
        return dist, ray, k, dt, gt

    out = dict(c)
    # mesh reach to rooftop targets
    dist, ray, k, dt, gt = lookup(G["te"], G["tn"], G["th"], 2 * STEP)
    use = (dist > 60) & (dist <= RMAX)
    gprev = gmax[ray, k]
    ip = idx[ray, k]
    di, hi = d[ip], z[ray, ip]
    th = G["th"]
    lam = C_MPS / (F_MHZ * 1e6)
    dd = np.maximum(dt - di, 1e-6)
    H = hi + 500 * CE * di * (dt - di) - (h0 * (dt - di) + th * di) / np.maximum(dt, 1e-6)
    v = np.where(k > 0, H * np.sqrt(0.002 * dt / (lam * np.maximum(di, 1e-6) * dd)), -10.0)
    diff = bullington_total(knife_edge_db(v), dt)
    drop = 500 * CE * dt * dt
    pen = (elevation_penalty_db(np.degrees(np.arctan2(th - h0 - drop, dist)), HPBW)
           + elevation_penalty_db(np.degrees(np.arctan2(h0 - th - drop, dist)), HPBW))
    margin = TX - 2 * CABLE + 2 * GAIN - pen - fspl_db(dt, F_MHZ) - diff - SENS_DBM
    ok = use & (margin >= MARGIN_OK)
    for a in ("valley", "kowloon", "hk_island"):
        m = use & (G["tarea"] == a)
        out[f"reach_{a}"] = round(float(ok[m].sum() / max(m.sum(), 1)), 3)
    out["reach_all"] = round(float(ok.sum() / max(use.sum(), 1)), 3)
    out["reach_los"] = round(float((use & (gprev < gt)).sum() / max(use.sum(), 1)), 3)

    # airspace 60 m above ground within 3 km
    ae, an = e0 + G["air_dx"], n0 + G["air_dy"]
    ar, ac = rc(ae, an)
    ground = np.nan_to_num(G["dtm10"][ar, ac], nan=0.0)
    ah = ground + AIR_AGL
    free = G["dsm10"][ar, ac] < ah
    dist, ray, k, dt, gt = lookup(ae, an, ah, 0.0)
    vis = gmax[ray, k] < gt
    out["air_view"] = round(float(vis[free].mean()) if free.any() else 0.0, 3)

    # local openness
    n_open = int(OPEN_R / STEP)
    g300 = gmax[:, n_open - 1]
    tol = np.tan(np.radians(OPEN_TOL_DEG)) * 1000
    out["open_frac"] = round(float((g300 <= tol).mean()), 3)
    worst = int(np.argmax(g300))
    out["worst_obstr_deg"] = round(float(np.degrees(np.arctan(g300[worst] / 1000))), 1)
    out["worst_obstr_az"] = float(G["az"][worst])

    # solar access (panel 1.5-2 m below the antenna)
    hp = h0 - (2.0 if c["kind"] == "hill" else 1.5)
    n_sol = int(SOLAR_R / STEP)
    gp = (z[:, :n_sol] - hp) / d[None, :n_sol] - 500.0 * CE * d[None, :n_sol]
    hor = np.degrees(np.arctan(gp.max(axis=1) / 1000))
    sray = np.round(G["sun_az"] / DAZ).astype(int) % nr
    lit = G["sun_el"] > hor[sray]
    out["solar_frac"] = round(float((G["sun_w"] * lit).sum() / G["sun_w"].sum()), 3)
    out["horizon_south_deg"] = round(float(hor[int(180 / DAZ)]), 1)

    # exact links: home (sector enforced) and planned relays
    ant = Antenna(gain_dbi=GAIN, hpbw_deg=HPBW, agl_m=0.0, tx_dbm=TX, cable_db=CABLE, amsl_m=h0)
    hm = G["home"]
    az_h = float(bearing_deg(hm["en"][0], hm["en"][1], e0, n0))
    out["az_from_home"] = round(az_h, 1)
    out["home_km"] = round(float(np.hypot(e0 - hm["en"][0], n0 - hm["en"][1])) / 1000, 2)
    if in_sector(az_h, hm["ant"].azimuth_deg, hm["ant"].arc_deg) and out["home_km"] > 0.05:
        L = link(G["T5"], (e0, n0), hm["en"], ant, hm["ant"], F_MHZ, SENS_DBM)
        out["home_db"], out["home_los"] = round(L.margin_db, 1), bool(L.los)
    else:
        out["home_db"], out["home_los"] = None, None
    best = (None, -999.0, None)
    for rl in G["relays"]:
        dd_ = np.hypot(e0 - rl["en"][0], n0 - rl["en"][1])
        if dd_ < 60:
            continue
        L = link(G["T5"], (e0, n0), rl["en"], ant, rl["ant"], F_MHZ, SENS_DBM)
        if L.margin_db > best[1]:
            best = (rl["id"] + " " + rl["name"], round(L.margin_db, 1), bool(L.los))
    out["backbone_site"], out["backbone_db"], out["backbone_los"] = best

    # surroundings
    r1, c1 = rc(e0, n0)
    out["canopy_10m"] = float(G["c10"][r1, c1])
    out["canopy_25m"] = float(G["c25"][r1, c1])
    out["slope_deg"] = round(float(G["slope"][r1, c1]), 1)
    out["relief_m"] = round(float(G["tpi"][r1, c1]), 1)
    out["path_m"] = round(float(G["dist_way"][r1, c1]))
    out["antenna_mpd"] = round(h0, 1)
    return out


def annotate(rows):
    """Names, nearest path type and nearby installations (main process, vectorised)."""
    from shapely.geometry import Point
    peaks = G["peaks"]
    pk_name = peaks["name:en"].where(peaks["name:en"].apply(lambda v: isinstance(v, str)), peaks["name"])
    pk_xy = np.column_stack([peaks.geometry.x, peaks.geometry.y])
    masts = G["masts"]
    mxy = np.column_stack([masts.geometry.x, masts.geometry.y]) if len(masts) else np.zeros((0, 2))
    bld = G["bld"]
    ways = G["ways"]
    wsi = ways.sindex
    for o in rows:
        e, n = o["e"], o["n"]
        dpk = np.hypot(pk_xy[:, 0] - e, pk_xy[:, 1] - n)
        named = np.array([isinstance(x, str) for x in pk_name])
        if named.any():
            j = int(np.argmin(np.where(named, dpk, np.inf)))
            o["peak_name"], o["peak_dist_m"] = (pk_name.iloc[j], round(float(dpk[j]))) if dpk[j] <= 400 else (None, None)
        if len(mxy):
            dm = np.hypot(mxy[:, 0] - e, mxy[:, 1] - n)
            o["masts_100m"] = int((dm <= 100).sum())
        if o["kind"] == "hill":
            near = bld.iloc[list(bld.sindex.query(Point(e, n).buffer(40)))]
            near = near[near.geometry.distance(Point(e, n)) <= 40]
            o["structures_40m"] = int(len(near))
            names = [x for x in near["BuildingNameEN"] if isinstance(x, str)]
            o["structure_names"] = "; ".join(sorted(set(names)))[:120] or None
        if "lu_names" in G:
            r, c = rc(e, n)
            code = int(G["lu"][r, c])
            o["landuse"] = G["lu_names"].get(code, str(code))
            win = G["lu"][max(r - 5, 0):r + 6, max(c - 5, 0):c + 6]
            o["woodland_50m"] = round(float((win == 71).mean()), 2)
        j = wsi.nearest(Point(e, n), return_all=False)
        j = int(np.asarray(j)[1][0])
        hw = ways.iloc[j]["highway"]
        o["path_type"] = hw if isinstance(hw, str) else None
    return rows


def init_worker():
    import os
    os.environ["OMP_NUM_THREADS"] = "1"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(ROOT / "out" / "nodes.geojson"))
    ap.add_argument("--out", default=str(ROOT / "out" / "spots"))
    ap.add_argument("--limit", type=int, default=0, help="analyse only the first N of each kind (testing)")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    load_inputs(Path(a.registry))
    make_targets()
    hills, roofs = hill_candidates(), roof_candidates()
    print(f"candidates: {len(hills)} hill spots, {len(roofs)} rooftops", flush=True)
    if a.limit:
        hills, roofs = hills[:a.limit], roofs[:a.limit]
    cands = hills + roofs
    ctx = mp.get_context("fork")
    with ctx.Pool(a.procs, initializer=init_worker) as pool:
        rows = []
        for i, r in enumerate(pool.imap(analyse, cands, chunksize=8), 1):
            rows.append(r)
            if i % 250 == 0:
                print(f"  analysed {i}/{len(cands)} ({time.time() - t0:.0f} s)", flush=True)
    rows = annotate(rows)
    import rasterio
    with rasterio.open(out / "open_ground_10m.tif", "w", driver="GTiff", width=G["W"], height=G["H"], count=1,
                       dtype="uint8", crs="EPSG:2326", transform=G["tr10"], compress="deflate", nodata=0) as dst:
        dst.write(G["open_mask"].astype("uint8"), 1)
    meta = {"aoi": AOI, "res_m": RES, "criteria": {
        "open_canopy_m": OPEN_CANOPY_M, "max_slope_deg": 30, "min_relief_m": 10, "min_ground_mpd": 50,
        "max_path_m": ACCESS_M, "hill_spacing_m": HILL_SPACING, "roof_spacing_m": ROOF_SPACING, "roof_cap": ROOF_CAP},
        "radio": {"f_mhz": F_MHZ, "sens_dbm": SENS_DBM, "margin_ok_db": MARGIN_OK, "gain_dbi": GAIN, "tx_dbm": TX,
                  "cable_db": CABLE, "pole_m": POLE_M, "roof_mast_m": ROOF_MAST_M, "target_mast_m": TARGET_MAST_M},
        "targets": {a_: int((G["tarea"] == a_).sum()) for a_ in ("valley", "kowloon", "hk_island", "other")},
        "sweep": {"daz_deg": DAZ, "step_m": STEP, "rmax_m": RMAX}, "runtime_s": round(time.time() - t0)}
    (out / "candidates.json").write_text(json.dumps({"meta": meta, "rows": rows}, default=lambda v: None if v != v else v))
    print(f"done: {len(rows)} candidates in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
