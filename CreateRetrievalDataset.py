from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
GEOTIFF_EXTS = {".tif", ".tiff"}


def list_images(folder: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images[path.name.lower()] = path
    return images


def list_geotiffs(paths: list[Path], geotiff_dir: Path | None = None) -> list[Path]:
    geotiffs = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in GEOTIFF_EXTS:
            geotiffs.append(path)
    if geotiff_dir is not None:
        if not geotiff_dir.is_dir():
            raise ValueError(f"GeoTIFF directory not found: {geotiff_dir}")
        geotiffs.extend(
            p for p in sorted(geotiff_dir.iterdir())
            if p.is_file() and p.suffix.lower() in GEOTIFF_EXTS
        )
    geotiffs = sorted(set(geotiffs))
    if not geotiffs:
        raise ValueError("No input GeoTIFF images found.")
    return geotiffs


def tile_positions(size: int, tile_length: int, stride: int) -> list[int]:
    if size < tile_length:
        raise ValueError(
            f"GeoTIFF side length {size} is smaller than tile_length={tile_length}"
        )
    positions = list(range(0, size - tile_length + 1, stride))
    last = size - tile_length
    if positions[-1] != last:
        positions.append(last)
    return positions


def tile_geotiffs(
    geotiff_paths: list[Path],
    tiles_dir: Path,
    tile_length: int,
    overlap: float = 0.0,
) -> pd.DataFrame:
    if tile_length < 1:
        raise ValueError("tile_length must be >= 1")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0.0, 1.0)")

    tiles_dir.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(round(tile_length * (1.0 - overlap))))
    rows = []
    sat_id = 0

    for image_index, image_path in enumerate(geotiff_paths):
        with rasterio.open(image_path) as src:
            if src.crs is None:
                raise ValueError(f"GeoTIFF has no CRS: {image_path}")

            x_positions = tile_positions(src.width, tile_length, stride)
            y_positions = tile_positions(src.height, tile_length, stride)
            iterator = tqdm(
                enumerate(y_positions),
                total=len(y_positions),
                desc=f"Tiling {image_path.name}",
            )

            for y_idx, top in iterator:
                for x_idx, left in enumerate(x_positions):
                    window = Window(left, top, tile_length, tile_length)
                    tile_data = src.read(window=window)
                    tile_transform = src.window_transform(window)

                    meta = src.meta.copy()
                    meta.update(
                        height=tile_length,
                        width=tile_length,
                        transform=tile_transform,
                        driver="GTiff",
                    )

                    tile_name = (
                        f"{image_path.stem}_tile_{image_index:02d}_"
                        f"{y_idx:04d}_{x_idx:04d}.tif"
                    )
                    tile_path = tiles_dir / tile_name
                    with rasterio.open(tile_path, "w", **meta) as dst:
                        dst.write(tile_data)

                    rows.append(
                        {
                            "sat_id": sat_id,
                            "sat_filename": tile_name,
                            "source_sat_path": tile_path.as_posix(),
                            "source_geotiff_path": image_path.as_posix(),
                            "tile_left_px": int(left),
                            "tile_top_px": int(top),
                            "tile_length_px": int(tile_length),
                            "tile_stride_px": int(stride),
                            "tile_overlap": float(overlap),
                        }
                    )
                    sat_id += 1

    tile_df = pd.DataFrame(rows)
    if tile_df.empty:
        raise ValueError("No tiles were created.")
    return tile_df


def geotiff_center_latlon(path: Path) -> tuple[float, float]:
    with rasterio.open(path) as ds:
        if ds.crs is None:
            raise ValueError(f"GeoTIFF has no CRS: {path}")

        bounds = ds.bounds
        center_x = (bounds.left + bounds.right) / 2.0
        center_y = (bounds.bottom + bounds.top) / 2.0

        if ds.crs.to_epsg() == 4326:
            return float(center_y), float(center_x)

        transformer = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(center_x, center_y)
        return float(lat), float(lon)


def add_tile_centers(tile_df: pd.DataFrame) -> pd.DataFrame:
    out = tile_df.copy()
    centers = []
    for path in tqdm(out["source_sat_path"].tolist(), desc="Reading tile centers"):
        centers.append(geotiff_center_latlon(Path(path)))
    out["sat_center_lat"] = [lat for lat, _ in centers]
    out["sat_center_lon"] = [lon for _, lon in centers]
    return out


def haversine_m(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    radius_m = 6371000.0
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * radius_m * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def build_pairs(
    tile_df: pd.DataFrame,
    scene_dir: Path,
    csv_path: Path,
    output_dir: Path,
    num_positives: int,
) -> pd.DataFrame:
    drone_dir = scene_dir / "drone"
    if not drone_dir.is_dir():
        raise ValueError(f"Drone image folder not found: {drone_dir}")
    if not csv_path.is_file():
        raise ValueError(f"CSV not found: {csv_path}")
    if num_positives < 1:
        raise ValueError("num_positives must be >= 1")
    if num_positives > len(tile_df):
        raise ValueError(
            f"num_positives={num_positives} but only {len(tile_df)} tiles were created."
        )

    frame_df = pd.read_csv(csv_path)
    required_cols = {"filename", "lat", "lon"}
    missing = required_cols - set(frame_df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    tile_lats = tile_df["sat_center_lat"].to_numpy(dtype=float)
    tile_lons = tile_df["sat_center_lon"].to_numpy(dtype=float)
    drone_images = list_images(drone_dir)
    pair_rows = []
    missing_images = []

    for query_id, row in tqdm(
        frame_df.reset_index(drop=True).iterrows(),
        total=len(frame_df),
        desc="Matching frames to nearest tiles",
    ):
        filename = str(row["filename"])
        source_query_path = drone_images.get(filename.lower())
        if source_query_path is None:
            missing_images.append(filename)
            continue

        query_lat = float(row["lat"])
        query_lon = float(row["lon"])
        distances = haversine_m(
            np.full_like(tile_lats, query_lat),
            np.full_like(tile_lons, query_lon),
            tile_lats,
            tile_lons,
        )
        nearest_indices = np.argsort(distances)[:num_positives]

        for positive_rank, nearest_idx in enumerate(nearest_indices, start=1):
            nearest_tile = tile_df.iloc[int(nearest_idx)]
            pair_row = {
                "query_id": query_id,
                "query_filename": source_query_path.name,
                "query_path": source_query_path.as_posix(),
                "positive_rank": positive_rank,
                "positive_sat_id": int(nearest_tile["sat_id"]),
                "positive_sat_filename": str(nearest_tile["sat_filename"]),
                "positive_sat_path": str(nearest_tile["source_sat_path"]),
                "source_geotiff_path": str(nearest_tile["source_geotiff_path"]),
                "query_lat": query_lat,
                "query_lon": query_lon,
                "sat_center_lat": float(nearest_tile["sat_center_lat"]),
                "sat_center_lon": float(nearest_tile["sat_center_lon"]),
                "center_distance_m": float(distances[int(nearest_idx)]),
                "tile_left_px": int(nearest_tile["tile_left_px"]),
                "tile_top_px": int(nearest_tile["tile_top_px"]),
                "tile_length_px": int(nearest_tile["tile_length_px"]),
            }

            for optional_col in ["num", "date", "height", "Omega", "Kappa", "Phi1", "Phi2"]:
                if optional_col in frame_df.columns:
                    pair_row[optional_col] = row[optional_col]

            pair_rows.append(pair_row)

    pairs_df = pd.DataFrame(pair_rows)
    if pairs_df.empty:
        raise ValueError("No frame/tile pairs were created.")

    pairs_df.to_csv(output_dir / "pairs.csv", index=False)

    stats = {
        "num_queries": int(pairs_df["query_id"].nunique()),
        "num_source_geotiffs": int(tile_df["source_geotiff_path"].nunique()),
        "num_gallery_tiles": int(len(tile_df)),
        "positives_per_query": int(num_positives),
        "num_pairs": int(len(pairs_df)),
        "num_unique_positive_tiles": int(pairs_df["positive_sat_id"].nunique()),
        "tile_length_px": int(tile_df["tile_length_px"].iloc[0]),
        "tile_stride_px": int(tile_df["tile_stride_px"].iloc[0]),
        "tile_overlap": float(tile_df["tile_overlap"].iloc[0]),
        "mean_center_distance_m": float(pairs_df["center_distance_m"].mean()),
        "median_center_distance_m": float(pairs_df["center_distance_m"].median()),
        "max_center_distance_m": float(pairs_df["center_distance_m"].max()),
        "rank1_mean_center_distance_m": float(
            pairs_df.loc[pairs_df["positive_rank"] == 1, "center_distance_m"].mean()
        ),
        f"rank{num_positives}_mean_center_distance_m": float(
            pairs_df.loc[pairs_df["positive_rank"] == num_positives, "center_distance_m"].mean()
        ),
        "missing_drone_images": int(len(missing_images)),
    }
    pd.DataFrame([stats]).to_csv(output_dir / "stats.csv", index=False)

    if missing_images:
        (output_dir / "missing_drone_images.txt").write_text(
            "\n".join(missing_images) + "\n",
            encoding="utf-8",
        )

    return pairs_df


def create_dataset_from_geotiffs(
    geotiff_paths: list[Path],
    scene_dir: Path,
    csv_path: Path,
    output_dir: Path,
    tile_length: int,
    num_positives: int = 4,
    overlap: float = 0.0,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = output_dir / "tiles"
    tile_df = tile_geotiffs(
        geotiff_paths=geotiff_paths,
        tiles_dir=tiles_dir,
        tile_length=tile_length,
        overlap=overlap,
    )
    tile_df = add_tile_centers(tile_df)
    tile_df.to_csv(output_dir / "tiles.csv", index=False)
    pairs_df = build_pairs(
        tile_df=tile_df,
        scene_dir=scene_dir,
        csv_path=csv_path,
        output_dir=output_dir,
        num_positives=num_positives,
    )

    stats = pd.read_csv(output_dir / "stats.csv").iloc[0].to_dict()
    print(f"\nDataset written to: {output_dir}")
    print(f"Tiles: {len(tile_df)} -> {tiles_dir}")
    print(f"Queries: {pairs_df['query_id'].nunique()}")
    print(f"Pairs: {len(pairs_df)} ({num_positives} positives per query)")
    print(f"Mean positive center distance: {stats['mean_center_distance_m']:.2f} m")
    print(f"Median positive center distance: {stats['median_center_distance_m']:.2f} m")
    print(f"Max positive center distance: {stats['max_center_distance_m']:.2f} m")
    return pairs_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tile large GeoTIFF image(s), then create a UAV-to-satellite "
            "retrieval dataset manifest from nearest tile centers."
        )
    )
    parser.add_argument(
        "--geotiff_paths",
        type=Path,
        nargs="*",
        default=[],
        help="One or more large GeoTIFF images to tile.",
    )
    parser.add_argument(
        "--geotiff_dir",
        type=Path,
        default=None,
        help="Optional directory containing large GeoTIFF images to tile.",
    )
    parser.add_argument("--scene_dir", type=Path, default=Path("UAV_VisLoc_dataset/01"))
    parser.add_argument("--csv_path", type=Path, default=Path("UAV_VisLoc_dataset/01/01.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("RetrievalDataset/01_from_geotiff"))
    parser.add_argument("--tile_length", type=int, required=True, help="Square tile side length in pixels.")
    parser.add_argument("--num_positives", type=int, default=4)
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Fractional tile overlap in [0, 1). Example: 0.5 means 50 percent overlap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geotiff_paths = list_geotiffs(args.geotiff_paths, args.geotiff_dir)
    create_dataset_from_geotiffs(
        geotiff_paths=geotiff_paths,
        scene_dir=args.scene_dir,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        tile_length=args.tile_length,
        num_positives=args.num_positives,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
