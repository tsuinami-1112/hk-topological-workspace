"""Surface model = LandsD 5 m DTM + LandsD building tops, for building-aware line of sight.

    from tools.dsm import load_dsm
    dsm, transform, info = load_dsm(bounds)                  # all buildings
    dsm, transform, info = load_dsm(bounds, exclude=[poly])  # e.g. without the observer's own block

Building heights: TopHeight (mPD) where LandsD gives one; that covers every Tower and Podium.
Temporary and open-sided structures mostly lack it and get ground + GAP_HEIGHT_M. Where
footprints overlap (a tower on its podium) the higher top wins. Cells keep the DTM wherever it
is higher than a building top (hillside buildings), so the result is max(DTM, building top).
"""
from __future__ import annotations

import numpy as np

from tools import hkgeo

GAP_HEIGHT_M = 4.0   # assumed height of structures with no TopHeight (sheds, canopies, yards)


def building_tops(bounds, pad: float = 0.0):
    """Buildings intersecting bounds with a 'top_mpd' column and how it was obtained."""
    b = hkgeo.load_vector("buildings", bounds, pad=pad or 1000.0)
    if b.empty:
        return b
    b = b[b.geometry.notna() & ~b.geometry.is_empty].copy()
    rp = b.geometry.representative_point()
    ground = hkgeo.dem_at(rp.x.values, rp.y.values)
    base = b["BaseHeight"].astype(float).where(b["BaseHeight"].notna(), ground)
    top = b["TopHeight"].astype(float)
    measured = top.notna() & (top > base)
    b["top_mpd"] = np.where(measured, top, base + GAP_HEIGHT_M)
    b["top_source"] = np.where(measured, "TopHeight", "gap+4m")
    return b


def load_dsm(bounds, exclude=None, buildings=None):
    """(surface array float32 with NaN for no data, affine transform, info dict) over bounds."""
    from rasterio.features import rasterize

    dtm, tr = hkgeo.load_dem(bounds)
    b = buildings if buildings is not None else building_tops(bounds)
    if exclude:
        from shapely.ops import unary_union
        ex = unary_union(exclude)
        b = b[~b.geometry.intersects(ex)]
    b = b.sort_values("top_mpd")            # later shapes overwrite earlier: highest top wins
    tops = rasterize(((g, float(h)) for g, h in zip(b.geometry, b["top_mpd"])), out_shape=dtm.shape,
                     transform=tr, fill=np.nan, dtype="float32")
    dsm = np.fmax(dtm, tops).astype("float32")
    dsm[np.isnan(dtm)] = np.nan
    info = {"buildings": int(len(b)), "measured": int((b["top_source"] == "TopHeight").sum()),
            "gap_filled": int((b["top_source"] != "TopHeight").sum()),
            "cells_raised": int(np.sum(np.isfinite(tops) & (tops > dtm)))}
    return dsm, tr, info
