Source code for the interactive forest browning anomaly map at [forest-monitoring.org](https://forest-monitoring.org).

## Setup

Requires Python 3.12. Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Set the data directory environment variable:

```bash
export FOREST_BROWNING_DATA_DIR=/path/to/data
```

The required datasets (`forest_mask.npy`, `ndvi_dataset_temporal.zarr`, `ndvi_dataset_spatial.zarr`) are produced by [s2-forest-browning-monitoring](https://github.com/samanthabiegel/s2-forest-browning-monitoring).

## Running

```bash
uv run uvicorn app:app
```
