# hk-topological-workspace

Terrain, building and access data for planning where drone-sentinel nodes go in Hong Kong,
plus the tools that analyse it. Planning sessions clone this repo, load what they need from
`data/`, and write their site decisions to the site registry, which lives in the claude.ai
"Mesh planning" Project (`registry/nodes.geojson`), not here.

Keep this repo **private**. The source data is public, but analysis outputs and notes made
here can reveal where stations are.

## Layout

```
config/sources.yml      where each dataset comes from (edit this, not the script)
tools/fetch.py          downloads and normalises the data (run by the workflow)
tools/hkgeo.py          loaders for analysis: to_hk80, load_dem, dem_at, load_vector
tools/rf.py             LoRa link model: free space + ITU-R P.526 Bullington diffraction
analyses/               one script per planning question; outputs go to out/
data/                   fetched data, committed by the workflow; never edit by hand
  MANIFEST.json         source, retrieval time and sha256 of every file
  dtm5m/                LandsD 5 m DTM, float32 GeoTIFF tiles + index.json
  buildings/            LandsD building footprints with heights, GeoPackage tiles + index.json
  osm/                  OSM roads/tracks/paths ("ways"), masts/towers ("masts"), named peaks ("peaks")
out/                    analysis outputs (git-ignored, regenerated per session)
```

## Conventions

- **CRS:** everything in `data/` is HK1980 Grid, EPSG:2326, metres. Use `tools/hkgeo.py` to
  convert to and from WGS84.
- **Tiles:** 10 km squares named by their south-west corner in km: `E830N820` covers
  E 830000-840000, N 820000-830000. Vector features are stored once, in the tile holding
  their representative point.
- **Heights:** metres, assumed mPD (Hong Kong Principal Datum). The DTM's dataset page does
  not state its datum; check against a known benchmark before trusting sub-metre detail.
- **DTM content:** bare terrain plus elevated roads, bridges and vegetation canopy (per
  LandsD). Buildings are not in it; combine with `buildings/` for urban line of sight.

## Refreshing data

The `fetch-data` workflow runs on GitHub's runner, which can reach LandsD, CSDI and
Geofabrik. Each dataset is fetched and committed in its own step. It runs when `config/sources.yml`, `tools/fetch.py` or the workflow changes on
`main`, or manually from the Actions tab (pick datasets, optionally force a refetch).
Datasets already fetched from the same source are skipped; OSM is skipped if under 7 days old.
The `probe-sources` workflow checks in a minute whether each source is reachable from the runner.

If the runner can't download the DTM from LandsD, download `Whole_HK_DTM_5m.zip` yourself,
attach it to a release of this repo tagged `src`, and set the DTM url in
`config/sources.yml` to `gh-release:src/Whole_HK_DTM_5m.zip`.

## Planning workflow

1. Copy the registry from the claude.ai Project (`registry/nodes.geojson`) to `out/nodes.geojson`.
2. Run the analysis for the question at hand, e.g.
   `python analyses/home_relays.py --registry out/nodes.geojson` (relay candidates around home),
   `python analyses/relay_sets.py <site> <site> ...` (union coverage of relay sets),
   `python analyses/shortlist_links.py out/home_relays/candidates.geojson <site> ...` (pairwise links).
3. Write decisions back to the registry in the Project.
4. Rebuild the planning map: `python analyses/render_map.py --registry out/nodes.geojson` then
   `python analyses/build_map_page.py`, and republish `out/map/` to the existing map artifact
   (its URL is in the registry under `x_registry.map_artifact`).

## Sources and terms

| Dataset | Provider | Terms |
|---|---|---|
| Digital Terrain Model, 5 m grid | Lands Department, via DATA.GOV.HK | DATA.GOV.HK Terms and Conditions of Use (free re-use) |
| Building | Lands Department, via CSDI Portal | DATA.GOV.HK / CSDI terms |
| Leaflet 1.9.4 stylesheet (analyses/vendor) | Volodymyr Agafonkin and contributors | BSD 2-Clause |
| Roads, paths, masts, peaks | OpenStreetMap contributors, via Geofabrik's Hong Kong extract | ODbL 1.0, attribution required |
