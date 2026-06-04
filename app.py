import math
import os
import shutil
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import anyio
import geopandas as gpd
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import zarr
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from PIL import Image
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.coords import BoundingBox
from rasterio.warp import reproject, transform_bounds, Resampling as WarpRes
from rasterio.windows import from_bounds as window_from_bounds

matplotlib.use("Agg")

# Constants
_data_dir             = os.environ["FOREST_BROWNING_DATA_DIR"]
FOREST_MASK           = f"{_data_dir}/forest_mask.npy"
TEMPORAL_DATASET_ZARR = f"{_data_dir}/ndvi_dataset_temporal.zarr"
SPATIAL_DATASET_ZARR  = f"{_data_dir}/ndvi_dataset_spatial.zarr"
REF_BBOX  = BoundingBox(left=2474090.0, bottom=1065110.0, right=2851370.0, top=1310530.0)
REF_WIDTH = 37728

STATIC_DIR     = Path(__file__).parent / "static"
COG_CACHE_DIR  = Path(__file__).parent / "cog_cache"
TILE_CACHE_DIR = Path(__file__).parent / "tile_cache"
COG_CACHE_DIR.mkdir(exist_ok=True)
TILE_CACHE_DIR.mkdir(exist_ok=True)

SCORE_THRESHOLD   = -1.5
VMIN, VMAX        = -7.0, -1.5
TILE_SIZE         = 256
WEB_MERCATOR_HALF = 20037508.342789244
TARGET_ZOOM = 15

# uint8 encoding stored inside the COG (one byte per pixel):
#   0      → NODATA (background or non-anomalous forest pixel)
#   1      → cloud / no valid observation
#   2..255 → anomaly score linearly mapped from [VMIN, VMAX]
QUANT_NODATA = 0
QUANT_CLOUD  = 1
QUANT_BIN_LO = 2
QUANT_BIN_HI = 255
QUANT_RANGE  = QUANT_BIN_HI - QUANT_BIN_LO  # 253


# App setup
@asynccontextmanager
async def lifespan(app):
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = 32
    yield


app = FastAPI(lifespan=lifespan)


# Mercator grid helpers
def _mercator_pixel_resolution(z: int) -> float:
    """Pixel size in EPSG:3857 meters at zoom z (256-pixel tiles)."""
    return (WEB_MERCATOR_HALF * 2) / (TILE_SIZE * 2 ** z)


def _snap_to_mercator_grid(left, bottom, right, top, pixel_res):
    """Expand bounds outward to land exactly on the Mercator pixel grid at this resolution."""
    aligned_left   = math.floor((left   + WEB_MERCATOR_HALF) / pixel_res) * pixel_res - WEB_MERCATOR_HALF
    aligned_right  = math.ceil ((right  + WEB_MERCATOR_HALF) / pixel_res) * pixel_res - WEB_MERCATOR_HALF
    aligned_top    = WEB_MERCATOR_HALF - math.floor((WEB_MERCATOR_HALF - top)    / pixel_res) * pixel_res
    aligned_bottom = WEB_MERCATOR_HALF - math.ceil ((WEB_MERCATOR_HALF - bottom) / pixel_res) * pixel_res
    return aligned_left, aligned_bottom, aligned_right, aligned_top


# Startup: load data into memory
print("Loading forest mask...", flush=True)
_mask = np.load(str(FOREST_MASK))
_flat = np.flatnonzero(_mask == 1)
del _mask

_rows = (_flat // REF_WIDTH).astype(np.int32)
_cols = (_flat % REF_WIDTH).astype(np.int32)
del _flat

ROW0, ROW1 = int(_rows.min()), int(_rows.max())
COL0, COL1 = int(_cols.min()), int(_cols.max())
CROP_H = ROW1 - ROW0 + 1
CROP_W = COL1 - COL0 + 1

FOREST_ROWS = _rows - ROW0
FOREST_COLS = _cols - COL0
del _rows, _cols

_res = (REF_BBOX.right - REF_BBOX.left) / REF_WIDTH
CROP_LEFT   = REF_BBOX.left + COL0 * _res
CROP_RIGHT  = REF_BBOX.left + (COL1 + 1) * _res
CROP_TOP    = REF_BBOX.top  - ROW0 * _res
CROP_BOTTOM = REF_BBOX.top  - (ROW1 + 1) * _res

print("Opening zarr datasets...", flush=True)
_ds_spatial  = zarr.open_group(str(SPATIAL_DATASET_ZARR),  mode="r")
_ds_temporal = zarr.open_group(str(TEMPORAL_DATASET_ZARR), mode="r")
SCORES = _ds_spatial["anomaly_scores"]
_dates_raw = pd.to_datetime([d.decode("utf-8") for d in _ds_temporal["dates"][:]])

_sort_order = np.argsort(_dates_raw)
DATES    = _dates_raw[_sort_order]
ZARR_IDX = np.asarray(_sort_order, dtype=int)
print(f"{len(DATES)} dates available ({DATES[0].date()} → {DATES[-1].date()}).", flush=True)

# Colormap
_c = plt.cm.inferno_r(np.linspace(0, 0.85, 256))
_c[-1, :] = [0, 0, 0, 1]
_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom", _c)
HEX_COLORS = [matplotlib.colors.to_hex(_cmap(i / 255)) for i in range(256)]
CMAP_RGB = np.array(
    [[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for c in HEX_COLORS],
    dtype=np.uint8,
)

# Switzerland outline
print("Loading Switzerland outline...", flush=True)
_gdf = gpd.read_file(
    Path(__file__).parent / "data" / "N2020_Revision_BiogeoRegion.shp"
)
CH_GEOJSON = _gdf.dissolve()[["geometry"]].to_crs(epsg=4326).to_json()
del _gdf
print("Ready.", flush=True)


# COG generation: reproject anomaly scores to a Web Mercator COG at TARGET_ZOOM
_cog_locks: dict[int, threading.Lock] = {}
_cog_locks_mutex = threading.Lock()


def _get_cog_lock(idx: int) -> threading.Lock:
    with _cog_locks_mutex:
        if idx not in _cog_locks:
            _cog_locks[idx] = threading.Lock()
        return _cog_locks[idx]


def _generate_cog(date_idx: int, final_path: Path) -> None:
    zarr_idx = int(ZARR_IDX[date_idx])
    col = SCORES[zarr_idx, :].astype(np.float32)

    src = np.zeros((CROP_H, CROP_W), dtype=np.uint8)  # 0 = NODATA everywhere
    mask_cloud   = np.isnan(col)
    mask_anomaly = ~mask_cloud & (col <= SCORE_THRESHOLD)

    src[FOREST_ROWS[mask_cloud], FOREST_COLS[mask_cloud]] = QUANT_CLOUD

    t = np.clip((col[mask_anomaly] - VMIN) / (VMAX - VMIN), 0.0, 1.0)
    src[FOREST_ROWS[mask_anomaly], FOREST_COLS[mask_anomaly]] = (
        QUANT_BIN_LO + t * QUANT_RANGE
    ).astype(np.uint8)

    src_transform = from_bounds(
        CROP_LEFT, CROP_BOTTOM, CROP_RIGHT, CROP_TOP,
        width=CROP_W, height=CROP_H,
    )
    src_crs = CRS.from_epsg(2056)
    dst_crs = CRS.from_epsg(3857)

    src_left_m, src_bottom_m, src_right_m, src_top_m = transform_bounds(
        src_crs, dst_crs, CROP_LEFT, CROP_BOTTOM, CROP_RIGHT, CROP_TOP,
    )
    pixel_res = _mercator_pixel_resolution(TARGET_ZOOM)
    dst_left, dst_bottom, dst_right, dst_top = _snap_to_mercator_grid(
        src_left_m, src_bottom_m, src_right_m, src_top_m, pixel_res,
    )
    dst_w = int(round((dst_right - dst_left) / pixel_res))
    dst_h = int(round((dst_top   - dst_bottom) / pixel_res))
    dst_transform = from_bounds(dst_left, dst_bottom, dst_right, dst_top, dst_w, dst_h)

    dst = np.zeros((dst_h, dst_w), dtype=np.uint8)
    reproject(
        source=src, destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        src_nodata=QUANT_NODATA, dst_nodata=QUANT_NODATA,
        resampling=WarpRes.nearest,
        num_threads=8,
    )

    tmp = final_path.with_suffix(".tmp.tif")
    try:
        with rasterio.open(
            str(tmp), "w",
            driver="COG",
            height=dst_h, width=dst_w,
            count=1, dtype="uint8",
            crs=dst_crs, transform=dst_transform,
            nodata=QUANT_NODATA,
            compress="deflate",
            predictor=2,
            zlevel=9,
            blocksize=256,
            overview_resampling="nearest",
        ) as out:
            out.write(dst, 1)
        shutil.move(str(tmp), str(final_path))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _ensure_cog(date_idx: int) -> Path:
    date_str = DATES[date_idx].strftime("%Y-%m-%d")
    path = COG_CACHE_DIR / f"{date_str}.tif"
    with _get_cog_lock(date_idx):
        if not path.exists():
            _generate_cog(date_idx, path)
    return path


# Tile rendering: slice a COG window and encode as PNG
def _tile_bounds_3857(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    n = 2 ** z
    tile_size = (WEB_MERCATOR_HALF * 2) / n
    minx = x * tile_size - WEB_MERCATOR_HALF
    maxy = WEB_MERCATOR_HALF - y * tile_size
    return minx, maxy - tile_size, minx + tile_size, maxy


def _data_to_rgba(data: np.ndarray) -> np.ndarray:
    """uint8 raster → RGBA. 0 → transparent, 1 → cloud, 2..255 → colormap."""
    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    rgba[data == QUANT_CLOUD] = [70, 70, 70, 230]

    anomaly_mask = data >= QUANT_BIN_LO
    if anomaly_mask.any():
        rgba[anomaly_mask, :3] = CMAP_RGB[data[anomaly_mask] - QUANT_BIN_LO]
        rgba[anomaly_mask, 3]  = 255

    return rgba


@lru_cache(maxsize=32)
def _open_cog(path_str: str):
    return rasterio.open(path_str)


def _render_tile(date_idx: int, z: int, x: int, y: int) -> bytes:
    cog_path = _ensure_cog(date_idx)
    tile_minx, tile_miny, tile_maxx, tile_maxy = _tile_bounds_3857(z, x, y)

    cog = _open_cog(str(cog_path))
    b = cog.bounds
    if (tile_maxx <= b.left or tile_minx >= b.right
            or tile_maxy <= b.bottom or tile_miny >= b.top):
        rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    else:
        window = window_from_bounds(
            tile_minx, tile_miny, tile_maxx, tile_maxy,
            transform=cog.transform,
        )
        data = cog.read(
            1, window=window,
            out_shape=(TILE_SIZE, TILE_SIZE),
            resampling=Resampling.nearest,
            boundless=True, fill_value=QUANT_NODATA,
        )
        rgba = _data_to_rgba(data)

    img = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    img.save(buf, "PNG", compress_level=1)
    return buf.getvalue()


# Background precompute: generate all COGs at startup
def _precompute_all_cogs():
    for idx in range(len(DATES)):
        try:
            _ensure_cog(idx)
        except Exception as e:
            print(f"precompute failed for date {idx}: {e}", flush=True)


threading.Thread(target=_precompute_all_cogs, daemon=True).start()


# Routes
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/config")
def config():
    return {"colors": HEX_COLORS, "vmin": VMIN, "vmax": VMAX}


@app.get("/dates")
def dates():
    return {"dates": DATES.strftime("%Y-%m-%d").tolist()}


@app.get("/switzerland")
def switzerland():
    return Response(CH_GEOJSON, media_type="application/geo+json")


@app.get("/tiles/{date_idx}/{z}/{x}/{y}.png")
def tile(date_idx: int, z: int, x: int, y: int):
    if not (0 <= date_idx < len(DATES)):
        raise HTTPException(status_code=404, detail="Date index out of range")
    if not (0 <= z <= 18 and 0 <= x < 2**z and 0 <= y < 2**z):
        raise HTTPException(status_code=400, detail="Tile coords out of range")

    date_str = DATES[date_idx].strftime("%Y-%m-%d")
    cached = TILE_CACHE_DIR / date_str / str(z) / str(x) / f"{y}.png"

    if cached.exists():
        return FileResponse(str(cached), media_type="image/png")

    png_bytes = _render_tile(date_idx, z, x, y)

    cached.parent.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(".tmp.png")
    tmp.write_bytes(png_bytes)
    shutil.move(str(tmp), str(cached))

    return Response(content=png_bytes, media_type="image/png")
