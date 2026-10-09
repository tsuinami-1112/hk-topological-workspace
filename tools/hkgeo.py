"""Helpers for planning sessions: CRS transforms and tile-aware loading of data/.

All bounds are HK1980 Grid (EPSG:2326) as (minx, miny, maxx, maxy) in metres.

    from tools.hkgeo import to_hk80, load_dem, dem_at, load_vector
    e, n = to_hk80(114.2067, 22.3932)
    dem, transform = load_dem((e - 5000, n - 5000, e + 5000, n + 5000))
    bldg = load_vector("buildings", (e - 500, n - 500, e + 500, n + 500))
    masts = load_vector("osm", bounds, layer="masts")
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HK80 = "EPSG:2326"


@lru_cache(maxsize=None)
def _transformer(src: str, dst: str):
    from pyproj import Transformer
    return Transformer.from_crs(src, dst, always_xy=True)


def to_hk80(lon, lat):
    """WGS84 lon/lat -> HK1980 Grid easting/northing (scalars or arrays)."""
    return _transformer("EPSG:4326", HK80).transform(lon, lat)


def to_wgs84(e, n):
    """HK1980 Grid easting/northing -> WGS84 lon/lat (scalars or arrays)."""
    return _transformer(HK80, "EPSG:4326").transform(e, n)


@lru_cache(maxsize=None)
def index(dataset: str) -> dict:
    p = DATA / dataset / "index.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run the fetch-data workflow for '{dataset}' first")
    return json.loads(p.read_text())


def tiles(dataset: str, bounds, pad: float = 0.0) -> list[Path]:
    minx, miny, maxx, maxy = bounds
    minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad
    out = []
    for t in index(dataset)["tiles"]:
        bx0, by0, bx1, by1 = t["bounds"]
        if bx0 < maxx and bx1 > minx and by0 < maxy and by1 > miny:
            out.append(DATA / dataset / t["file"])
    return out


def load_dem(bounds, res: float | None = None):
    """Mosaic of the 5 m DTM over bounds. Returns (float32 array with NaN for no data, affine transform)."""
    import rasterio
    from rasterio.merge import merge

    files = tiles("dtm5m", bounds)
    if not files:
        raise ValueError(f"no DTM tiles intersect {bounds}")
    srcs = [rasterio.open(f) for f in files]
    try:
        nodata = srcs[0].nodata
        arr, transform = merge(srcs, bounds=tuple(bounds), res=res or srcs[0].res, nodata=nodata, dtype="float32")
    finally:
        for s in srcs:
            s.close()
    a = arr[0]
    if nodata is not None:
        a[a == nodata] = np.nan
    return a, transform


def dem_at(e, n):
    """DTM height (m) at HK1980 points, nearest cell. NaN where there is no data."""
    import rasterio

    e = np.atleast_1d(np.asarray(e, dtype=float))
    n = np.atleast_1d(np.asarray(n, dtype=float))
    out = np.full(e.shape, np.nan)
    for t in index("dtm5m")["tiles"]:
        bx0, by0, bx1, by1 = t["bounds"]
        m = (e >= bx0) & (e < bx1) & (n >= by0) & (n < by1) & np.isnan(out)
        if not m.any():
            continue
        with rasterio.open(DATA / "dtm5m" / t["file"]) as src:
            vals = np.array([v[0] for v in src.sample(zip(e[m], n[m]))], dtype=float)
            if src.nodata is not None:
                vals[vals == src.nodata] = np.nan
        out[m] = vals
    return out


def load_vector(dataset: str, bounds, layer: str | None = None, pad: float = 1000.0):
    """Features from a tiled vector dataset ('buildings' or 'osm') intersecting bounds.

    pad widens the tile search because each feature is stored once, in the tile holding its
    representative point; long roads can belong to a neighbouring tile.
    """
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    layer = layer or {"buildings": "buildings", "osm": "ways"}.get(dataset)
    parts = []
    for f in tiles(dataset, bounds, pad):
        try:
            g = gpd.read_file(f, layer=layer, bbox=tuple(bounds))
        except Exception:  # tile has no such layer (e.g. no masts in it)
            continue
        if not g.empty:
            parts.append(g)
    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs=HK80)
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=HK80)
    return g[g.intersects(box(*bounds))].reset_index(drop=True)
