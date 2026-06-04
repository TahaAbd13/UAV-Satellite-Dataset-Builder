# UAV Satellite Dataset Builder

Create UAV-to-satellite retrieval dataset manifests from large GeoTIFF satellite
images and UAV frame coordinates.

The main script, `CreateRetrievalDataset.py`, does two things:

1. Tiles one or more large GeoTIFF satellite images into square GeoTIFF tiles.
2. Assigns each UAV frame the nearest `k` tile centers as positive satellite
   images.

It writes dataset manifests and generated tiles. It does not train a model or
run image matching.

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:

- `numpy`
- `pandas`
- `rasterio`
- `pyproj`
- `tqdm`

## Expected Input Layout

Example layout:

```text
UAV_VisLoc_dataset/
  01/
    01.csv
    drone/
      01_0001.JPG
      01_0002.JPG
      ...

satellite_maps/
  satellite01.tif
```

The UAV scene CSV must contain:

```text
filename,lat,lon
```

Optional columns copied into `pairs.csv` if present:

```text
num,date,height,Omega,Kappa,Phi1,Phi2
```

Input GeoTIFF images must have a valid CRS.

## Usage

Tile one large GeoTIFF into `1024 x 1024` pixel tiles and create top-4 positive
labels:

```bash
python CreateRetrievalDataset.py \
  --geotiff_paths UAV_VisLoc_dataset/01/satellite01.tif \
  --scene_dir UAV_VisLoc_dataset/01 \
  --csv_path UAV_VisLoc_dataset/01/01.csv \
  --output_dir RetrievalDataset/01_from_geotiff \
  --tile_length 1024 \
  --num_positives 4
```

Tile every GeoTIFF in a directory:

```bash
python CreateRetrievalDataset.py \
  --geotiff_dir satellite_maps \
  --scene_dir UAV_VisLoc_dataset/01 \
  --csv_path UAV_VisLoc_dataset/01/01.csv \
  --output_dir RetrievalDataset/01_from_geotiffs \
  --tile_length 1024 \
  --num_positives 4
```

Use overlapping tiles:

```bash
python CreateRetrievalDataset.py \
  --geotiff_paths UAV_VisLoc_dataset/01/satellite01.tif \
  --scene_dir UAV_VisLoc_dataset/01 \
  --csv_path UAV_VisLoc_dataset/01/01.csv \
  --output_dir RetrievalDataset/01_overlap \
  --tile_length 1024 \
  --overlap 0.5 \
  --num_positives 4
```

`--overlap 0.5` means the stride is half the tile length.

## Outputs

The output directory contains:

```text
RetrievalDataset/01_from_geotiff/
  tiles/
    satellite01_tile_00_0000_0000.tif
    satellite01_tile_00_0000_0001.tif
    ...
  tiles.csv
  pairs.csv
  stats.csv
  missing_drone_images.txt  # only created if needed
```

## Output Files

`tiles.csv` describes every generated satellite tile:

```text
sat_id
sat_filename
source_sat_path
source_geotiff_path
tile_left_px
tile_top_px
tile_length_px
tile_stride_px
tile_overlap
sat_center_lat
sat_center_lon
```

`pairs.csv` contains one row per UAV query-positive satellite pair:

```text
query_id
query_filename
query_path
positive_rank
positive_sat_id
positive_sat_filename
positive_sat_path
source_geotiff_path
query_lat
query_lon
sat_center_lat
sat_center_lon
center_distance_m
tile_left_px
tile_top_px
tile_length_px
```

If `--num_positives 4`, each UAV frame appears four times in `pairs.csv`.
`positive_rank=1` is the nearest tile center.

`stats.csv` summarizes the generated dataset:

```text
num_queries
num_source_geotiffs
num_gallery_tiles
positives_per_query
num_pairs
num_unique_positive_tiles
tile_length_px
tile_stride_px
tile_overlap
mean_center_distance_m
median_center_distance_m
max_center_distance_m
rank1_mean_center_distance_m
rankK_mean_center_distance_m
missing_drone_images
```

## Notes

- Tile centers are converted to WGS84 (`EPSG:4326`) before distance calculation.
- Distance is computed with the haversine formula in meters.
- The script creates dataset labels from coordinates, not pixel-level alignment.
- Large GeoTIFFs, generated tiles, and full generated datasets should usually
  stay outside git.

## Small Example Files

The `examples/` folder contains tiny sample manifests:

```text
examples/
  pairs_sample.csv
  stats_sample.csv
```

These are intended for documentation and schema reference only.
