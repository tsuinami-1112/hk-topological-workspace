#!/usr/bin/env python3
"""Probe candidate data endpoints from GitHub's runner and save what they return under meta/probe/.

Each entry in config/probe.yml is one of:
  kind: http      GET a URL; saves status, headers, and the body (JSON pretty-printed, or the
                  first 200 kB of text). Optional 'grep': keep only JSON list items / lines that
                  contain this string (for huge listings).
  kind: s3list    list a public S3 bucket prefix (ListObjectsV2), all pages, optional 'match'
                  substrings; saves keys and sizes.
  kind: gdal      open a raster through /vsicurl/; saves driver, CRS, size, transform, dtype,
                  nodata, block size, overviews, tags, and a few sample values near HK.
Every probe is independent: a failure is recorded and the rest still run.
"""
from __future__ import annotations

import json
import os
import re
import sys
import traceback
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "meta" / "probe"
UA = {"User-Agent": "hk-topological-workspace/1.0 (RF site planning; data probe)"}


def note(msg, level="notice"):
    print(msg, flush=True)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level} title=probe::{msg}", flush=True)


def probe_http(p):
    r = requests.get(p["url"], headers=UA, timeout=(20, 120), allow_redirects=True)
    rec = {"url": p["url"], "final_url": r.url, "status": r.status_code,
           "content_type": r.headers.get("content-type"), "bytes": len(r.content),
           "headers": dict(r.headers)}
    body = None
    try:
        body = r.json()
    except ValueError:
        pass
    grep = p.get("grep")
    if body is not None:
        if grep:
            def walk(o):
                if isinstance(o, dict):
                    return {k: walk(v) for k, v in o.items()}
                if isinstance(o, list):
                    return [walk(x) for x in o if grep.lower() in json.dumps(x, ensure_ascii=False).lower()]
                return o
            body = walk(body)
        rec["json"] = body
    else:
        text = r.text
        if grep:
            text = "\n".join(l for l in text.splitlines() if grep.lower() in l.lower())
        rec["text"] = text[:200_000]
        rec["urls_found"] = sorted(set(re.findall(r"https?://[^\s\"'<>]+", r.text)))[:500]
    return rec, f"HTTP {r.status_code}, {r.headers.get('content-type')}, {len(r.content)} bytes"


def probe_s3list(p):
    bucket, prefix = p["bucket"], p.get("prefix", "")
    base = f"https://{bucket}.s3.amazonaws.com/"
    keys, prefixes, token = [], [], None
    for _ in range(int(p.get("max_pages", 20))):
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if p.get("delimiter"):
            params["delimiter"] = p["delimiter"]
        if token:
            params["continuation-token"] = token
        r = requests.get(base, params=params, headers=UA, timeout=(20, 120))
        r.raise_for_status()
        x = r.text
        keys += [{"key": k, "size": int(s)} for k, s in re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", x, re.S)]
        prefixes += re.findall(r"<Prefix>([^<]+)</Prefix>", x)[1:] if "<CommonPrefixes>" in x else []
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", x)
        if not m:
            break
        token = m.group(1)
    match = p.get("match")
    if match:
        keys = [k for k in keys if any(s in k["key"] for s in match)]
    rec = {"bucket": bucket, "prefix": prefix, "keys": keys[:2000], "n_keys": len(keys), "common_prefixes": prefixes[:500]}
    return rec, f"{len(keys)} keys, {len(prefixes)} prefixes"


def probe_gdal(p):
    import rasterio
    from rasterio.windows import from_bounds
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.vrt")
    url = p["url"]
    path = url if url.startswith("/vsi") else "/vsicurl/" + url
    with rasterio.open(path) as src:
        rec = {"url": url, "driver": src.driver, "crs": str(src.crs), "width": src.width, "height": src.height,
               "count": src.count, "dtype": src.dtypes[0], "nodata": src.nodata, "res": list(src.res),
               "bounds": list(src.bounds), "transform": list(src.transform)[:6], "block_shapes": src.block_shapes[:1],
               "overviews": src.overviews(1), "tags": src.tags(), "band_tags": src.tags(1),
               "compression": str(src.compression), "interleaving": str(src.interleaving)}
        lonlat = p.get("sample_lonlat")
        if lonlat:
            from rasterio.warp import transform as wt
            xs, ys = wt("EPSG:4326", src.crs, [c[0] for c in lonlat], [c[1] for c in lonlat])
            rec["samples"] = [{"lonlat": c, "value": [float(v) for v in val]}
                              for c, val in zip(lonlat, src.sample(list(zip(xs, ys))))]
        wb = p.get("window_lonlat")
        if wb:
            from rasterio.warp import transform_bounds
            b = transform_bounds("EPSG:4326", src.crs, *wb)
            win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
            arr = src.read(1, window=win, out_shape=(min(int(win.height), 400), min(int(win.width), 400)))
            import numpy as np
            vals, counts = np.unique(arr, return_counts=True)
            order = np.argsort(-counts)[:30]
            rec["window_value_counts"] = [[float(vals[i]), int(counts[i])] for i in order]
    return rec, f"{rec['driver']} {rec['crs']} {rec['width']}x{rec['height']} res {rec['res']} {rec['dtype']} ovr {rec['overviews']}"


KINDS = {"http": probe_http, "s3list": probe_s3list, "gdal": probe_gdal}


def main():
    cfg = yaml.safe_load((ROOT / "config" / "probe.yml").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for p in cfg.get("probes", []):
        name = p["name"]
        try:
            rec, msg = KINDS[p["kind"]](p)
            note(f"{name}: {msg}")
        except Exception as ex:
            rec = {"error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-3000:], "probe": p}
            note(f"{name}: FAILED {type(ex).__name__}: {str(ex)[:200]}", "warning")
        (OUT / f"{name}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
