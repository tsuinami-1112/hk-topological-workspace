#!/usr/bin/env python3
"""Fetch planning data and normalise it into data/ as HK1980 Grid, 10 km tiles.

Datasets (run in this order, so later ones can reuse the DTM tile index):
  dtm5m      LandsD 5 m DTM            -> data/dtm5m/dtm5m_E###N###.tif   float32, metres
  buildings  LandsD Building (CSDI)    -> data/buildings/bldg_E###N###.gpkg  layer "buildings"
  osm        OSM ways + masts/towers   -> data/osm/osm_E###N###.gpkg        layers "ways", "masts"

Tile keys name the tile's south-west corner in km of HK1980 Grid: E830N820 covers
E 830000-840000, N 820000-830000. Each dataset writes data/<name>/index.json and an
entry in data/MANIFEST.json (source, retrieval time, sha256 per file). A dataset already
fetched from the same source is skipped unless --force is given.

Usage:
  python tools/fetch.py --datasets dtm5m,buildings,osm [--force]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HK80 = "EPSG:2326"
# A grid without a CRS is accepted as HK1980 Grid only if it falls inside this window.
HK80_SANITY = (780000, 780000, 890000, 870000)
RASTER_EXT = {".asc", ".tif", ".tiff", ".img", ".bil", ".flt"}
UA = {"User-Agent": "hk-topological-workspace/1.0 (RF site planning; data fetch)"}
RETRY_STATUS = {429, 500, 502, 503, 504}


def log(msg: str) -> None:
    print(msg, flush=True)


def note(msg: str, level: str = "notice") -> None:
    """Log, and on GitHub Actions also raise an annotation (readable through the checks API)."""
    log(msg)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level} title=fetch::{msg}", flush=True)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_cfg(path: str | None = None) -> dict:
    return yaml.safe_load(Path(path or ROOT / "config" / "sources.yml").read_text())


# ---------------------------------------------------------------- tiling

def tile_key(e0: float, n0: float) -> str:
    return f"E{int(round(e0)) // 1000:03d}N{int(round(n0)) // 1000:03d}"


def tile_grid(bounds, size):
    """10 km cells (minx, miny, maxx, maxy) covering bounds, aligned to multiples of size."""
    minx, miny, maxx, maxy = bounds
    cells = []
    e = math.floor(minx / size) * size
    while e < maxx:
        n = math.floor(miny / size) * size
        while n < maxy:
            cells.append((e, n, e + size, n + size))
            n += size
        e += size
    return cells


def query_tiles(cfg) -> list[tuple]:
    """Tiles worth querying for vector data: DTM tiles with land if fetched, else the HK grid."""
    size = cfg["tile_size_m"]
    idx = DATA / "dtm5m" / "index.json"
    if idx.exists():
        keys = {t["key"] for t in json.loads(idx.read_text())["tiles"]}
        cells = [c for c in tile_grid(cfg["hk_bounds_hk80"], size) if tile_key(c[0], c[1]) in keys]
        if cells:
            return cells
    return tile_grid(cfg["hk_bounds_hk80"], size)


def write_tiled(gdf, out_dir: Path, prefix: str, size: int, layer: str, files: dict) -> None:
    """Write each feature once, into the tile holding its representative point."""
    if gdf.empty:
        return
    pts = gdf.geometry.representative_point()
    ex = (np.floor(pts.x / size) * size).astype(int)
    nx = (np.floor(pts.y / size) * size).astype(int)
    for (e0, n0), part in gdf.groupby([ex.values, nx.values]):
        path = out_dir / f"{prefix}_{tile_key(e0, n0)}.gpkg"
        part.to_file(path, layer=layer, driver="GPKG")
        files.setdefault(path.name, {"key": tile_key(e0, n0), "file": path.name,
                                     "bounds": [int(e0), int(n0), int(e0 + size), int(n0 + size)],
                                     "counts": {}})
        files[path.name]["counts"][layer] = int(len(part))


# ---------------------------------------------------------------- manifest

def manifest_load() -> dict:
    p = DATA / "MANIFEST.json"
    return json.loads(p.read_text()) if p.exists() else {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(name: str, source: str, files, extra: dict | None = None) -> None:
    m = manifest_load()
    m[name] = {
        "source": source,
        "retrieved": now(),
        "files": {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                  for p in sorted(files)},
        **(extra or {}),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "MANIFEST.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


def up_to_date(name: str, source: str) -> bool:
    e = manifest_load().get(name)
    return bool(e) and e.get("source") == source and all((ROOT / f).exists() for f in e.get("files", {}))


def reset_dir(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)


# ---------------------------------------------------------------- http

def session() -> requests.Session:
    s = requests.Session()
    s.headers.update(UA)
    return s


def http(s, method, url, *, tries=5, timeout=180, **kw):
    for i in range(tries):
        try:
            r = s.request(method, url, timeout=timeout, **kw)
            if r.status_code in RETRY_STATUS:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as ex:
            resp = getattr(ex, "response", None)
            if resp is not None and resp.status_code not in RETRY_STATUS:
                raise
            if i == tries - 1:
                raise
            wait = 5 * 2 ** i
            log(f"  retry {i + 1}/{tries - 1} in {wait}s: {ex}")
            time.sleep(wait)


def download(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if url.startswith("gh-release:"):
        tag, asset = url[len("gh-release:"):].split("/", 1)
        subprocess.run(["gh", "release", "download", tag, "-p", asset, "-D", str(dest_dir), "--clobber"],
                       check=True)
        return dest_dir / asset
    local = Path(url[len("file://"):]) if url.startswith("file://") else None
    if local is not None:
        out = dest_dir / local.name
        shutil.copyfile(local, out)
        return out
    out = dest_dir / (url.rstrip("/").split("/")[-1] or "download")
    # Probe first, so an unreachable host fails in seconds instead of hanging the job.
    try:
        with session() as s:
            h = s.head(url, allow_redirects=True, timeout=(15, 30))
        size = h.headers.get("content-length")
        note(f"probe {url}: HTTP {h.status_code}, "
             f"{(int(size) / 2**20):.0f} MB" if size else f"probe {url}: HTTP {h.status_code}, size unknown")
    except requests.RequestException as ex:
        raise RuntimeError(f"cannot reach {url} from this runner: {type(ex).__name__}: {ex}") from ex
    t0 = time.time()
    for attempt in range(3):
        try:
            with session() as s:
                r = http(s, "GET", url, stream=True, timeout=(30, 300), tries=3)
                total = int(r.headers.get("content-length") or 0)
                got, last = 0, 0
                with open(out, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        got += len(chunk)
                        if got - last >= 100 << 20:
                            rate = got / 2**20 / max(time.time() - t0, 1)
                            log(f"  {got >> 20} MB" + (f" of {total >> 20} MB" if total else "") + f" ({rate:.1f} MB/s)")
                            last = got
            if total and got != total:
                raise IOError(f"short read: {got} of {total} bytes")
            break
        except (requests.RequestException, IOError) as ex:
            if attempt == 2:
                raise
            log(f"  download failed ({ex}), retrying")
            time.sleep(15)
    note(f"downloaded {out.name}: {out.stat().st_size / 2**20:.1f} MB in {time.time() - t0:.0f} s")
    return out


# ---------------------------------------------------------------- DTM

def _unpack(archive: Path, dest: Path) -> None:
    """Extract zips, including zips nested inside the download."""
    pending = [archive]
    while pending:
        a = pending.pop()
        if zipfile.is_zipfile(a):
            target = dest / a.stem
            with zipfile.ZipFile(a) as z:
                z.extractall(target)
            if a != archive:
                a.unlink()
            pending += [p for p in target.rglob("*.zip")]
        elif a == archive:
            shutil.copy(a, dest / a.name)


def _is_hk80(crs, bounds) -> tuple[bool, str]:
    inside = (HK80_SANITY[0] <= bounds.left and bounds.right <= HK80_SANITY[2]
              and HK80_SANITY[1] <= bounds.bottom and bounds.top <= HK80_SANITY[3])
    if crs is not None and crs.to_epsg() == 2326:
        return True, "EPSG:2326"
    wkt = (crs.to_wkt() if crs is not None else "").lower().replace("_", " ")
    if crs is not None and "hong kong 1980" in wkt and inside:
        return True, "HK1980 Grid (non-EPSG WKT)"
    if crs is None and inside:
        return True, "no CRS in source; bounds fit HK1980 Grid"
    return False, f"CRS {crs} with bounds {tuple(bounds)}"


def _normalise(src_path: Path, out_path: Path):
    """Copy one source grid to a float32 GeoTIFF tagged EPSG:2326, in row strips."""
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(src_path) as src:
        ok, why = _is_hk80(src.crs, src.bounds)
        if not ok:
            raise RuntimeError(f"{src_path.name}: not recognisably HK1980 Grid ({why}); refusing to guess")
        log(f"  {src_path.name}: {src.width}x{src.height} px, res {src.res}, {why}, nodata {src.nodata}")
        nodata = float(src.nodata) if src.nodata is not None else -9999.0
        prof = dict(driver="GTiff", width=src.width, height=src.height, count=1, dtype="float32",
                    crs=HK80, transform=src.transform, nodata=nodata, tiled=True,
                    blockxsize=512, blockysize=512, compress="deflate", predictor=3, BIGTIFF="IF_SAFER")
        with rasterio.open(out_path, "w", **prof) as dst:
            for row in range(0, src.height, 1024):
                w = Window(0, row, src.width, min(1024, src.height - row))
                dst.write(src.read(1, window=w).astype("float32"), 1, window=w)
        return src.res, nodata


def fetch_dtm(cfg: dict, force: bool) -> str:
    import rasterio
    from rasterio.merge import merge

    url = (cfg.get("dtm5m") or {}).get("url")
    if not url:
        log("dtm5m: no url configured, skipping")
        return "skipped (no url)"
    if not force and up_to_date("dtm5m", url):
        log("dtm5m: already fetched from this source, skipping (use --force to refetch)")
        return "skipped (up to date)"
    size = int(cfg["tile_size_m"])
    out_dir = DATA / "dtm5m"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        archive = download(url, td / "dl")
        src_dir = td / "src"
        src_dir.mkdir()
        _unpack(archive, src_dir)
        rasters = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in RASTER_EXT)
        if not rasters:
            raise RuntimeError("no raster grid found in the download")
        norm, res, nodata = [], None, None
        for i, p in enumerate(rasters):
            r, nd = _normalise(p, td / f"norm_{i}.tif")
            if res is None:
                res, nodata = r, nd
            elif tuple(r) != tuple(res):
                raise RuntimeError(f"mixed resolutions in source: {res} vs {r}")
            norm.append(td / f"norm_{i}.tif")

        srcs = [rasterio.open(p) for p in norm]
        try:
            left = min(s.bounds.left for s in srcs)
            bottom = min(s.bounds.bottom for s in srcs)
            right = max(s.bounds.right for s in srcs)
            top = max(s.bounds.top for s in srcs)
            ox, oy = srcs[0].transform.c, srcs[0].transform.f
            rx, ry = res

            def sx(x):  # snap to the source pixel grid so tiles are cut, never resampled
                return ox + round((x - ox) / rx) * rx

            def sy(y):
                return oy + round((y - oy) / ry) * ry

            reset_dir(out_dir)
            tiles, written = [], []
            for e0, n0, e1, n1 in tile_grid((left, bottom, right, top), size):
                tb = (sx(e0), sy(n0), sx(e1), sy(n1))
                arr, tr = merge(srcs, bounds=tb, res=res, nodata=nodata, dtype="float32")
                a = arr[0]
                valid = (a != nodata) & np.isfinite(a)
                if not valid.any():
                    continue
                key = tile_key(e0, n0)
                path = out_dir / f"dtm5m_{key}.tif"
                with rasterio.open(path, "w", driver="GTiff", width=a.shape[1], height=a.shape[0], count=1,
                                   dtype="float32", crs=HK80, transform=tr, nodata=nodata, tiled=True,
                                   blockxsize=256, blockysize=256, compress="deflate", predictor=3) as dst:
                    dst.write(a, 1)
                tiles.append({"key": key, "file": path.name, "bounds": [round(v, 3) for v in tb],
                              "valid_fraction": round(float(valid.mean()), 4),
                              "min_m": round(float(a[valid].min()), 2), "max_m": round(float(a[valid].max()), 2)})
                written.append(path)
                log(f"  tile {key}: {valid.mean():.0%} valid, {a[valid].min():.1f}..{a[valid].max():.1f} m")
        finally:
            for s in srcs:
                s.close()

    index = {"dataset": "dtm5m", "crs": HK80, "res_m": list(res), "nodata": nodata, "units": "metres",
             "vertical_datum": "assumed mPD (HK Principal Datum), LandsD's standard height datum; "
                               "not stated on the dataset page",
             "notes": "LandsD 5 m DTM, +/-5 m. Includes elevated roads, bridges and vegetation canopy; "
                      "buildings are not modelled.",
             "source": url, "tiles": tiles}
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    record("dtm5m", url, written + [out_dir / "index.json"], {"tiles": len(tiles)})
    note(f"dtm5m: {len(tiles)} tiles, res {list(res)}, nodata {nodata}")
    return f"ok, {len(tiles)} tiles"


# ---------------------------------------------------------------- buildings (ArcGIS REST)

def _arcgis(s, layer_url, bbox, **params):
    q = {"where": "1=1", "geometry": ",".join(f"{v:.3f}" for v in bbox),
         "geometryType": "esriGeometryEnvelope", "inSR": 2326,
         "spatialRel": "esriSpatialRelIntersects", **params}
    j = http(s, "GET", layer_url + "/query", params=q).json()
    if isinstance(j, dict) and "error" in j:
        raise RuntimeError(f"ArcGIS query error: {j['error']}")
    return j


def _ring_area(r):
    a = np.asarray(r, dtype=float)
    return 0.5 * float(np.sum(a[:-1, 0] * a[1:, 1] - a[1:, 0] * a[:-1, 1]))


def _esri_polygon(rings):
    """Esri JSON rings -> shapely geometry. Clockwise rings are exteriors, the rest holes."""
    from shapely.geometry import MultiPolygon, Polygon
    outers, holes = [], []
    for r in rings:
        if len(r) < 4:
            continue
        (outers if _ring_area(r) < 0 else holes).append(r)
    polys = [[o, []] for o in outers]
    for h in holes:
        hp = Polygon(h)
        for p in polys:
            if Polygon(p[0]).contains(hp.representative_point()):
                p[1].append(h)
                break
    shapes = [Polygon(o, hs) for o, hs in polys]
    if not shapes:
        return None
    return shapes[0] if len(shapes) == 1 else MultiPolygon(shapes)


def _fetch_features(s, layer_url, bbox, maxrec, fmt):
    if fmt == "geojson":
        j = _arcgis(s, layer_url, bbox, outFields="*", returnGeometry="true", outSR=2326, f="geojson")
        feats = j.get("features", [])
        exceeded = j.get("exceededTransferLimit") or (j.get("properties") or {}).get("exceededTransferLimit")
    else:
        from shapely.geometry import mapping
        j = _arcgis(s, layer_url, bbox, outFields="*", returnGeometry="true", outSR=2326, f="json")
        feats = []
        for f in j.get("features", []):
            g = _esri_polygon((f.get("geometry") or {}).get("rings", []))
            if g is not None:
                feats.append({"type": "Feature", "properties": f.get("attributes", {}), "geometry": mapping(g)})
        exceeded = j.get("exceededTransferLimit")
    return feats, bool(exceeded)


def _fetch_bbox(s, layer_url, bbox, maxrec, fmt, depth=0):
    """Fetch every feature intersecting bbox, splitting it until each request is under the server cap."""
    n = _arcgis(s, layer_url, bbox, returnCountOnly="true", f="json").get("count", 0)
    if n == 0:
        return []
    minx, miny, maxx, maxy = bbox
    splittable = (maxx - minx) > 50
    if n > int(maxrec * 0.9) and splittable:
        mx, my = (minx + maxx) / 2, (miny + maxy) / 2
        out = []
        for q in ((minx, miny, mx, my), (mx, miny, maxx, my), (minx, my, mx, maxy), (mx, my, maxx, maxy)):
            out += _fetch_bbox(s, layer_url, q, maxrec, fmt, depth + 1)
        return out
    feats, exceeded = _fetch_features(s, layer_url, bbox, maxrec, fmt)
    if exceeded:
        if splittable:
            mx, my = (minx + maxx) / 2, (miny + maxy) / 2
            out = []
            for q in ((minx, miny, mx, my), (mx, miny, maxx, my), (minx, my, mx, maxy), (mx, my, maxx, maxy)):
                out += _fetch_bbox(s, layer_url, q, maxrec, fmt, depth + 1)
            return out
        log(f"  warning: transfer limit hit on a {maxx - minx:.0f} m box; some features may be missing")
    time.sleep(0.1)
    return feats


def fetch_buildings(cfg: dict, force: bool) -> str:
    import geopandas as gpd

    url = ((cfg.get("buildings") or {}).get("url") or "").strip().rstrip("/")
    if not url:
        log("buildings: no service URL in config/sources.yml, skipping")
        return "skipped (no url)"
    if url.endswith("/query"):
        url = url[: -len("/query")]
    if not force and up_to_date("buildings", url):
        log("buildings: already fetched from this source, skipping (use --force to refetch)")
        return "skipped (up to date)"
    size = int(cfg["tile_size_m"])
    s = session()
    meta = http(s, "GET", url, params={"f": "json"}).json()
    if "error" in meta:
        raise RuntimeError(f"service error: {meta['error']}")
    if "fields" not in meta:
        raise RuntimeError("URL is not a feature layer (no 'fields'); it should end in /FeatureServer/<n> or /MapServer/<n>")
    maxrec = int(meta.get("maxRecordCount") or 1000)
    oid = meta.get("objectIdField") or next(
        (f["name"] for f in meta["fields"] if f.get("type") == "esriFieldTypeOID"), None)
    log(f"  layer '{meta.get('name')}', {len(meta['fields'])} fields, maxRecordCount {maxrec}, oid {oid}")

    fmt = "geojson"
    probe = query_tiles(cfg)[0]
    try:
        _arcgis(s, url, probe, outFields="*", returnGeometry="true", outSR=2326, f="geojson", resultRecordCount=1)
    except Exception as ex:  # older servers: no f=geojson on this layer
        log(f"  f=geojson not supported ({ex}); using Esri JSON")
        fmt = "json"

    feats = {}
    cells = query_tiles(cfg)
    for i, cell in enumerate(cells, 1):
        got = _fetch_bbox(s, url, cell, maxrec, fmt)
        for f in got:
            props = f.get("properties") or {}
            key = props.get(oid) if oid else None
            if key is None:
                key = f.get("id")
            if key is None:
                key = hashlib.md5(json.dumps(f.get("geometry"), sort_keys=True).encode()).hexdigest()
            feats[key] = f
        log(f"  cell {tile_key(cell[0], cell[1])} ({i}/{len(cells)}): {len(got)} features, {len(feats)} unique so far")
    if not feats:
        raise RuntimeError("service returned no features for Hong Kong's extent; check the URL")

    gdf = gpd.GeoDataFrame.from_features(list(feats.values()), crs=HK80)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    out_dir = DATA / "buildings"
    reset_dir(out_dir)
    (out_dir / "layer_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    files: dict = {}
    write_tiled(gdf, out_dir, "bldg", size, "buildings", files)
    index = {"dataset": "buildings", "crs": HK80, "source": url, "layer": meta.get("name"),
             "fields": [f["name"] for f in meta["fields"]], "features": int(len(gdf)),
             "tiles": sorted(files.values(), key=lambda t: t["key"])}
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    record("buildings", url, [out_dir / f for f in files] + [out_dir / "index.json", out_dir / "layer_meta.json"],
           {"features": int(len(gdf))})
    return f"ok, {len(gdf)} buildings in {len(files)} tiles"


# ---------------------------------------------------------------- OSM (Overpass)

OVERPASS = """[out:json][timeout:300];
(
  way["highway"]({s},{w},{n},{e});
  node["man_made"~"^(mast|tower|communications_tower)$"]({s},{w},{n},{e});
  way["man_made"~"^(mast|tower|communications_tower)$"]({s},{w},{n},{e});
  node["tower:type"="communication"]({s},{w},{n},{e});
  node["natural"="peak"]({s},{w},{n},{e});
);
out geom;"""

WAY_TAGS = ["highway", "name", "name:en", "access", "foot", "motor_vehicle", "vehicle", "service",
            "surface", "tracktype", "sac_scale", "trail_visibility", "lit"]
MAST_TAGS = ["man_made", "tower:type", "tower:construction", "height", "operator", "name", "name:en"]
PEAK_TAGS = ["name", "name:en", "name:zh", "ele"]


def _osm_frames(elements):
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    ways, masts, peaks = [], [], []
    for el in elements:
        tags = el.get("tags", {})
        is_mast = tags.get("man_made") in ("mast", "tower", "communications_tower") or tags.get("tower:type") == "communication"
        if el["type"] == "node":
            geom = Point(el["lon"], el["lat"])
        elif el["type"] == "way" and len(el.get("geometry") or []) >= 2:
            geom = LineString([(p["lon"], p["lat"]) for p in el["geometry"]])
        else:
            continue
        row = {"osm_type": el["type"], "osm_id": el["id"], "tags": json.dumps(tags, ensure_ascii=False)}
        if tags.get("natural") == "peak" and el["type"] == "node":
            row.update({k: tags.get(k) for k in PEAK_TAGS})
            peaks.append({**row, "geometry": geom})
        elif is_mast:
            row.update({k: tags.get(k) for k in MAST_TAGS})
            masts.append({**row, "geometry": geom.centroid if el["type"] == "way" else geom})
        elif "highway" in tags and el["type"] == "way":
            row.update({k: tags.get(k) for k in WAY_TAGS})
            ways.append({**row, "geometry": geom})

    def frame(rows):
        if not rows:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_crs(HK80)
        return gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(HK80)

    return frame(ways), frame(masts), frame(peaks)


def fetch_osm(cfg: dict, force: bool) -> str:
    from pyproj import Transformer

    url = (cfg.get("osm") or {}).get("overpass_url")
    if not url:
        log("osm: no overpass_url configured, skipping")
        return "skipped (no url)"
    query_sha = hashlib.sha256(OVERPASS.encode()).hexdigest()[:16]
    prev = manifest_load().get("osm")
    if (not force and prev and prev.get("query_sha") == query_sha
            and all((ROOT / f).exists() for f in prev.get("files", {}))):
        age = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(prev["retrieved"].replace("Z", "+00:00"))
        if age < dt.timedelta(days=7):
            log(f"osm: fetched {age.days} days ago, skipping (use --force to refetch)")
            return "skipped (fresh)"
    size = int(cfg["tile_size_m"])
    to_wgs = Transformer.from_crs(HK80, "EPSG:4326", always_xy=True)
    s = session()
    elements = {}
    cells = query_tiles(cfg)
    for i, (e0, n0, e1, n1) in enumerate(cells, 1):
        lons, lats = to_wgs.transform([e0, e1, e0, e1], [n0, n0, n1, n1])
        q = OVERPASS.format(s=min(lats), w=min(lons), n=max(lats), e=max(lons))
        j = http(s, "POST", url, data={"data": q}, timeout=400).json()
        for el in j.get("elements", []):
            elements[(el["type"], el["id"])] = el
        log(f"  cell {tile_key(e0, n0)} ({i}/{len(cells)}): {len(j.get('elements', []))} elements")
        time.sleep(2)
    ways, masts, peaks = _osm_frames(elements.values())
    out_dir = DATA / "osm"
    reset_dir(out_dir)
    files: dict = {}
    write_tiled(ways, out_dir, "osm", size, "ways", files)
    write_tiled(masts, out_dir, "osm", size, "masts", files)
    write_tiled(peaks, out_dir, "osm", size, "peaks", files)
    index = {"dataset": "osm", "crs": HK80, "source": url, "retrieved": now(),
             "attribution": "(c) OpenStreetMap contributors, ODbL 1.0",
             "ways": int(len(ways)), "masts": int(len(masts)), "peaks": int(len(peaks)),
             "tiles": sorted(files.values(), key=lambda t: t["key"])}
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    record("osm", url, [out_dir / f for f in files] + [out_dir / "index.json"],
           {"ways": int(len(ways)), "masts": int(len(masts)), "peaks": int(len(peaks)), "query_sha": query_sha})
    return f"ok, {len(ways)} ways, {len(masts)} masts/towers, {len(peaks)} peaks"


# ---------------------------------------------------------------- main

FETCHERS = {"dtm5m": fetch_dtm, "buildings": fetch_buildings, "osm": fetch_osm}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="dtm5m,buildings,osm")
    ap.add_argument("--force", action="store_true", help="refetch even if MANIFEST says up to date")
    ap.add_argument("--config", help="alternate sources.yml")
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config)
    want = [d.strip() for d in a.datasets.split(",") if d.strip()]
    unknown = set(want) - set(FETCHERS)
    if unknown:
        ap.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    results, failed = {}, False
    for name in FETCHERS:  # fixed order: the DTM index steers the vector queries
        if name not in want:
            continue
        log(f"== {name}")
        try:
            results[name] = FETCHERS[name](cfg, a.force)
        except Exception as ex:
            failed = True
            results[name] = f"FAILED: {ex}"
            traceback.print_exc()
            note(f"{name} failed: {type(ex).__name__}: {ex}", "error")
    for k, v in results.items():
        if not v.startswith("FAILED"):
            note(f"{k}: {v}")
    lines = ["| dataset | result |", "|---|---|"] + [f"| {k} | {v} |" for k, v in results.items()]
    log("\n".join(lines))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("## Data fetch\n\n" + "\n".join(lines) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
