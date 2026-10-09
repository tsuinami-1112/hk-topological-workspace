#!/usr/bin/env python3
"""Describe ArcGIS REST services listed in config/inspect.yml and save the results under meta/<name>/.

For each service (a FeatureServer/MapServer root or a single layer) it saves:
  service.json          the root description (?f=pjson)
  layer_<id>.json       each layer's description: fields, limits, query formats, spatial reference
  sample_<id>.geojson   up to 50 features in the sample box (HK1980), all fields
  summary.json          per layer: name, geometry, feature count, limits, fields, and how many
                        features have each height-like field empty

Run on GitHub's runner by .github/workflows/inspect-service.yml, which commits meta/.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.fetch import http, note, session  # noqa: E402

HEIGHTISH = re.compile(r"height|level|storey|story|floor|elev|top|base|hgt|mpd", re.I)


def get(s, url, **params):
    j = http(s, "GET", url, params={"f": "pjson", **params}, timeout=(30, 180), tries=3).json()
    if isinstance(j, dict) and "error" in j:
        raise RuntimeError(f"{url}: {j['error']}")
    return j


def query(s, layer_url, **params):
    return get(s, layer_url + "/query", **{"where": "1=1", **params})


def inspect(entry: dict) -> dict:
    s = session()
    url = entry["url"].rstrip("/")
    out = ROOT / "meta" / entry["name"]
    out.mkdir(parents=True, exist_ok=True)
    root = get(s, url)
    (out / "service.json").write_text(json.dumps(root, indent=2, ensure_ascii=False))
    if "fields" in root:  # the URL is already a layer
        layer_urls = {url.rsplit("/", 1)[-1]: url}
    else:
        layer_urls = {str(l["id"]): f"{url}/{l['id']}" for l in root.get("layers", [])}
    summary = {"url": url, "service_description": root.get("serviceDescription") or root.get("description"),
               "spatialReference": root.get("spatialReference"), "maxRecordCount": root.get("maxRecordCount"),
               "layers": []}
    for lid, lurl in layer_urls.items():
        meta = get(s, lurl)
        (out / f"layer_{lid}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        info = {"id": lid, "url": lurl, "name": meta.get("name"), "geometryType": meta.get("geometryType"),
                "maxRecordCount": meta.get("maxRecordCount"),
                "supportedQueryFormats": meta.get("supportedQueryFormats"),
                "supportsPagination": (meta.get("advancedQueryCapabilities") or {}).get("supportsPagination"),
                "spatialReference": (meta.get("extent") or {}).get("spatialReference"),
                "objectIdField": meta.get("objectIdField"),
                "fields": [{"name": f["name"], "type": f["type"], "alias": f.get("alias")} for f in meta.get("fields", [])]}
        try:
            info["count"] = query(s, lurl, returnCountOnly="true").get("count")
        except Exception as ex:
            info["count_error"] = str(ex)
        empties = {}
        for f in meta.get("fields", []):
            if HEIGHTISH.search(f["name"]) or HEIGHTISH.search(f.get("alias") or ""):
                try:
                    empties[f["name"]] = query(s, lurl, where=f"{f['name']} IS NULL", returnCountOnly="true").get("count")
                except Exception as ex:
                    empties[f["name"]] = f"error: {ex}"
        info["height_fields_null_count"] = empties
        box = entry.get("sample_hk80")
        if box and meta.get("geometryType"):
            try:
                g = query(s, lurl, geometry=",".join(map(str, box)), geometryType="esriGeometryEnvelope", inSR=2326,
                          spatialRel="esriSpatialRelIntersects", outFields="*", outSR=2326, f="geojson",
                          resultRecordCount=50)
                (out / f"sample_{lid}.geojson").write_text(json.dumps(g, ensure_ascii=False))
                info["sample_features"] = len(g.get("features", []))
            except Exception as ex:
                info["sample_error"] = str(ex)
        summary["layers"].append(info)
        note(f"{entry['name']} layer {lid} '{info['name']}': {info.get('count')} features, "
             f"{info['geometryType']}, maxRecordCount {info['maxRecordCount']}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "inspect.yml").read_text())
    failed = False
    for entry in cfg.get("services", []):
        try:
            inspect(entry)
        except Exception as ex:
            failed = True
            note(f"inspect {entry.get('name')} failed: {type(ex).__name__}: {ex}", "error")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
