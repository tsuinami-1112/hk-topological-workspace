"""LoRa link and coverage estimates on the HK 5 m DTM (terrain only).

Model
-----
Path loss = free-space loss + diffraction loss from the ITU-R P.526 Bullington method on the
actual terrain profile, including P.526's empirical correction for multiple edges
(Luc + (1 - exp(-Luc/6)) * (10 + 0.02 d)). Effective earth radius k = 4/3.

Antennas are vertical collinear omnis. Their elevation pattern is approximated as a parabola,
penalty = min(12 * (theta / HPBW)^2, 15) dB, so steep paths (a hilltop looking down at a
nearby rooftop) lose gain the way real collinears do.

Not modelled: buildings (until data/buildings exists), clutter and foliage beyond the DTM's
canopy surface, fading, and the local noise floor. Margins are against LoRa's thermal
sensitivity, so treat them as upper bounds and calibrate with measured SNR.

All coordinates are HK1980 Grid (EPSG:2326) metres; heights are metres (DTM datum, mPD).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import map_coordinates

C_MPS = 299_792_458.0
R_EFF_KM = 6371.0 * 4.0 / 3.0
CE = 1.0 / R_EFF_KM                 # effective earth curvature, 1/km
WATER_MPD = 1.0                     # approximate water surface: used for no-data cells and below-water
                                    # (bathymetric) cells, which the LandsD DTM stores as negative heights


# ------------------------------------------------------------------ basic terms

def fspl_db(d_km, f_mhz):
    return 32.45 + 20 * np.log10(f_mhz) + 20 * np.log10(np.maximum(d_km, 1e-3))


def knife_edge_db(v):
    """ITU-R P.526 single knife-edge loss J(v); 0 for v <= -0.78."""
    v = np.asarray(v, dtype=float)
    out = np.zeros_like(v)
    m = v > -0.78
    vm = v[m]
    out[m] = 6.9 + 20 * np.log10(np.sqrt((vm - 0.1) ** 2 + 1) + vm - 0.1)
    return out if out.ndim else float(out)


def bullington_total(luc, d_km):
    """P.526 correction applied to the Bullington knife-edge loss for real multi-edge terrain."""
    luc = np.asarray(luc, dtype=float)
    return luc + (1 - np.exp(-luc / 6.0)) * (10 + 0.02 * np.asarray(d_km, dtype=float))


def elevation_penalty_db(theta_deg, hpbw_deg):
    return np.minimum(12.0 * (np.asarray(theta_deg) / hpbw_deg) ** 2, 15.0)


# ------------------------------------------------------------------ terrain sampling

@dataclass
class Terrain:
    """A DEM array (NaN = no data) with its rasterio affine transform."""
    z: np.ndarray
    transform: object
    _filled: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._filled = np.where(np.isfinite(self.z), np.maximum(self.z, WATER_MPD), WATER_MPD).astype("float32")

    @property
    def res(self):
        return self.transform.a

    def sample(self, e, n):
        """Bilinear terrain height at HK1980 points; WATER_MPD where the DTM has no data."""
        scalar = np.ndim(e) == 0
        t = self.transform
        col = (np.atleast_1d(np.asarray(e, float)) - t.c) / t.a - 0.5
        row = (np.atleast_1d(np.asarray(n, float)) - t.f) / t.e - 0.5
        out = map_coordinates(self._filled, [row, col], order=1, mode="nearest")
        return float(out[0]) if scalar else out


# ------------------------------------------------------------------ point to point

@dataclass
class Antenna:
    gain_dbi: float
    hpbw_deg: float            # vertical half-power beamwidth
    agl_m: float
    tx_dbm: float = 28.0
    cable_db: float = 1.0
    azimuth_deg: float | None = None    # centre of usable sector (None = omni)
    arc_deg: float | None = None        # width of usable sector
    amsl_m: float | None = None         # absolute antenna height (mPD); overrides ground + agl_m

    def height(self, ground_m: float) -> float:
        return self.amsl_m if self.amsl_m is not None else ground_m + self.agl_m


def bearing_deg(e0, n0, e1, n1):
    return np.degrees(np.arctan2(np.asarray(e1) - e0, np.asarray(n1) - n0)) % 360.0


def in_sector(az, centre, width):
    if centre is None or width is None:
        return np.ones_like(np.asarray(az), dtype=bool)
    diff = (np.asarray(az) - centre + 180.0) % 360.0 - 180.0
    return np.abs(diff) <= width / 2.0


def profile(terrain: Terrain, e0, n0, e1, n1, step_m=5.0):
    dist = float(np.hypot(e1 - e0, n1 - n0))
    n = max(int(np.ceil(dist / step_m)), 2)
    s = np.linspace(0.0, dist, n + 1)
    e = e0 + (e1 - e0) * s / dist
    nn = n0 + (n1 - n0) * s / dist
    return s, terrain.sample(e, nn), e, nn


@dataclass
class Link:
    d_km: float
    los: bool
    v: float
    diffraction_db: float
    fspl_db: float
    pattern_a_db: float
    pattern_b_db: float
    loss_db: float
    margin_ab_db: float        # a transmits, b receives
    margin_ba_db: float
    obstruction: tuple | None  # (e, n, terrain height) of the main obstruction, if any

    @property
    def margin_db(self):
        return min(self.margin_ab_db, self.margin_ba_db)


def link(terrain: Terrain, a_en, b_en, ant_a: Antenna, ant_b: Antenna, f_mhz=915.0,
         sens_dbm=-134.0, step_m=5.0) -> Link:
    """Terrain link estimate between a and b (HK1980 e, n), both directions."""
    s, z, pe, pn = profile(terrain, a_en[0], a_en[1], b_en[0], b_en[1], step_m)
    lam = C_MPS / (f_mhz * 1e6)
    d = s[-1] / 1000.0
    h_ts = ant_a.height(z[0])
    h_rs = ant_b.height(z[-1])
    di, hi = s[1:-1] / 1000.0, z[1:-1]
    bulge = 500.0 * CE * di * (d - di)
    s_tim_all = (hi + bulge - h_ts) / di
    s_tim = s_tim_all.max()
    s_tr = (h_rs - h_ts) / d
    obstruction = None
    if s_tim < s_tr:  # line of sight: worst Fresnel intrusion
        vv = (hi + bulge - (h_ts * (d - di) + h_rs * di) / d) * np.sqrt(0.002 * d / (lam * di * (d - di)))
        k = int(np.argmax(vv))
        v = float(vv[k])
        los = True
        if v > -0.78:
            obstruction = (float(pe[k + 1]), float(pn[k + 1]), float(hi[k]))
    else:
        s_rim_all = (hi + bulge - h_rs) / (d - di)
        s_rim = s_rim_all.max()
        d_bp = (h_rs - h_ts + s_rim * d) / (s_tim + s_rim)
        v = float((h_ts + s_tim * d_bp - (h_ts * (d - d_bp) + h_rs * d_bp) / d)
                  * np.sqrt(0.002 * d / (lam * d_bp * (d - d_bp))))
        los = False
        k = int(np.argmax(s_tim_all))
        obstruction = (float(pe[k + 1]), float(pn[k + 1]), float(hi[k]))
    luc = knife_edge_db(v)
    diff = float(bullington_total(luc, d))
    fs = float(fspl_db(d, f_mhz))
    # elevation angle of the path at each end (each end sees the other lowered by the earth bulge)
    drop = 500.0 * CE * d * d
    theta_a = np.degrees(np.arctan2(h_rs - h_ts - drop, d * 1000))
    theta_b = np.degrees(np.arctan2(h_ts - h_rs - drop, d * 1000))
    pa = float(elevation_penalty_db(theta_a, ant_a.hpbw_deg))
    pb = float(elevation_penalty_db(theta_b, ant_b.hpbw_deg))
    loss = fs + diff
    common = ant_a.gain_dbi - pa + ant_b.gain_dbi - pb - loss - sens_dbm
    m_ab = ant_a.tx_dbm - ant_a.cable_db - ant_b.cable_db + common
    m_ba = ant_b.tx_dbm - ant_b.cable_db - ant_a.cable_db + common
    return Link(d, los, v, diff, fs, pa, pb, loss, float(m_ab), float(m_ba), obstruction)


# ------------------------------------------------------------------ sector screening raster

def screen(terrain: Terrain, obs_en, ant_obs: Antenna, ant_tgt: Antenna, r_max_m=25000.0,
           step_m=10.0, daz_deg=0.1, out_res_m=25.0, f_mhz=915.0, sens_dbm=-134.0, chunk=240):
    """Fast margin map from one observer to targets on every cell (ant_tgt.agl_m above ground).

    Diffraction uses a single knife-edge at the observer's horizon point for each target, a
    screening shortcut: it is exact for one dominant ridge and optimistic when the main
    obstruction sits near the target. Use link() for any site that matters.
    Margin is for the weaker direction. Returns (grid, transform-like tuple (x0, y0, res)).
    """
    e0, n0 = obs_en
    lam = C_MPS / (f_mhz * 1e6)
    h_ts = ant_obs.height(terrain.sample(e0, n0))
    if ant_obs.azimuth_deg is not None and ant_obs.arc_deg is not None:
        az = np.arange(ant_obs.azimuth_deg - ant_obs.arc_deg / 2, ant_obs.azimuth_deg + ant_obs.arc_deg / 2 + 1e-9, daz_deg)
    else:
        az = np.arange(0.0, 360.0, daz_deg)
    s = np.arange(step_m, r_max_m + step_m / 2, step_m)
    d = s / 1000.0
    size = int(np.ceil(2 * r_max_m / out_res_m))
    x0, y0 = e0 - r_max_m, n0 + r_max_m
    grid = np.full((size, size), -np.inf, dtype="float32")
    has_edge = (np.arange(len(s)) >= 1)[None, :]   # the first sample has nothing in front of it
    for c0 in range(0, len(az), chunk):
        a = np.radians(az[c0:c0 + chunk])[:, None]
        pe = e0 + np.sin(a) * s[None, :]
        pn = n0 + np.cos(a) * s[None, :]
        z = terrain.sample(pe.ravel(), pn.ravel()).reshape(pe.shape)
        g = (z - h_ts) / d - 500.0 * CE * d                 # separable part of S_tim
        gmax = np.maximum.accumulate(g, axis=1)
        idx = np.where(g >= gmax, np.arange(g.shape[1])[None, :], 0)
        idx = np.maximum.accumulate(idx, axis=1)
        iprev = np.concatenate([np.zeros((g.shape[0], 1), dtype=int), idx[:, :-1]], axis=1)
        h_rs = z + ant_tgt.agl_m
        di = d[iprev]
        hi = np.take_along_axis(z, iprev, axis=1)
        valid = np.broadcast_to(has_edge, di.shape)
        di_safe = np.where(valid, di, 1e-6)
        dd = np.maximum(d - di_safe, 1e-6)
        H = hi + 500.0 * CE * di_safe * (d - di_safe) - (h_ts * (d - di_safe) + h_rs * di_safe) / d
        v = np.where(valid, H * np.sqrt(0.002 * d / (lam * di_safe * dd)), -10.0)
        luc = knife_edge_db(v)
        diff = bullington_total(luc, d[None, :])
        loss = fspl_db(d, f_mhz)[None, :] + diff
        drop = 500.0 * CE * d * d
        theta_o = np.degrees(np.arctan2(h_rs - h_ts - drop, s[None, :]))
        theta_t = np.degrees(np.arctan2(h_ts - h_rs - drop, s[None, :]))
        pen = elevation_penalty_db(theta_o, ant_obs.hpbw_deg) + elevation_penalty_db(theta_t, ant_tgt.hpbw_deg)
        common = ant_obs.gain_dbi + ant_tgt.gain_dbi - pen - loss - sens_dbm - ant_obs.cable_db - ant_tgt.cable_db
        margin = np.minimum(ant_obs.tx_dbm, ant_tgt.tx_dbm) + common
        col = ((pe - x0) / out_res_m).astype(int)
        row = ((y0 - pn) / out_res_m).astype(int)
        ok = (col >= 0) & (col < size) & (row >= 0) & (row < size)
        np.maximum.at(grid, (row[ok], col[ok]), margin[ok].astype("float32"))
    grid[~np.isfinite(grid)] = np.nan
    return grid, (x0, y0, out_res_m), h_ts
