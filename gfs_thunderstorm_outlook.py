from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import requests

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError as exc:
    raise SystemExit(
        "cartopy is required for map plotting. Install it with: pip install cartopy"
    ) from exc

try:
    import pygrib
except ImportError as exc:
    raise SystemExit(
        "pygrib is required to read GRIB2 files. Install it with: pip install pygrib"
    ) from exc

NOMADS_FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
CYCLE_HOURS = (18, 12, 6, 0)
USA_EXTENT = (-125.0, -66.0, 24.0, 50.0)
DATA_EXTENT = (-127.0, -64.0, 23.0, 51.0)
FORECAST_HOURS = tuple(range(0, 385, 6))
DISPLAY_TIMEZONE = dt.timezone(dt.timedelta(hours=2), name="UTC+2")
BASE_DIR = Path("/var/data")
ARCHIVE_RETENTION = dt.timedelta(days=15)

@dataclass(frozen=True)
class RunSpec:
    run_date: dt.date
    cycle_hour: int

    @property
    def cycle_time(self) -> dt.datetime:
        return dt.datetime.combine(
            self.run_date,
            dt.time(hour=self.cycle_hour, tzinfo=dt.timezone.utc),
        )

    @property
    def run_stamp(self) -> str:
        return self.cycle_time.strftime("%Y%m%d_%Hz")

    def build_params(self, forecast_hour: int) -> dict[str, str]:
        cycle_label = f"{self.cycle_hour:02d}"
        return {
            "dir": f"/gfs.{self.run_date:%Y%m%d}/{cycle_label}/atmos",
            "file": f"gfs.t{cycle_label}z.pgrb2.0p25.f{forecast_hour:03d}",
            "var_4LFTX": "on",
            "lev_surface": "on",
            "var_CAPE": "on",
            "lev_255-0_mb_above_ground": "on",
            "var_DZDT": "on",
            "lev_750_mb": "on",
            "var_ACPCP": "on",
            "lev_surface": "on",
        }

    def build_download_url(self, forecast_hour: int) -> str:
        return requests.Request(
            "GET", NOMADS_FILTER_URL, params=self.build_params(forecast_hour)
        ).prepare().url

@dataclass(frozen=True)
class OutlookCategory:
    dn: int
    label: str
    label2: str
    stroke: str
    fill: str

OUTLOOK_CATEGORIES = (
    OutlookCategory(2, "SMALL", "Small Risk", "#1E90FF", "#B0E2FF"),
    OutlookCategory(3, "MEDIUM", "Medium Risk", "#FFA500", "#FFDAB9"),
    OutlookCategory(4, "HIGH", "High Risk", "#CC0000", "#FFB6B6"),
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a model-derived thunderstorm outlook from GFS lifted index, CAPE, vertical velocity, and convective precip."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BASE_DIR,
        help="Folder that will hold the run directory. Default: /var/data",
    )
    parser.add_argument(
        "--lookback-cycles",
        type=int,
        default=12,
        help="How many cycles to inspect while searching for the latest complete run.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds for availability checks and downloads.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not re-download or re-plot files that already exist.",
    )
    return parser.parse_args()

def candidate_runs(reference_time: dt.datetime, lookback_cycles: int) -> Iterable[RunSpec]:
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=dt.timezone.utc)
    else:
        reference_time = reference_time.astimezone(dt.timezone.utc)

    current_day = reference_time.date()
    emitted = 0

    while emitted < lookback_cycles:
        for cycle_hour in CYCLE_HOURS:
            cycle_time = dt.datetime.combine(
                current_day,
                dt.time(hour=cycle_hour, tzinfo=dt.timezone.utc),
            )
            if cycle_time <= reference_time:
                yield RunSpec(run_date=current_day, cycle_hour=cycle_hour)
                emitted += 1
                if emitted >= lookback_cycles:
                    return
        current_day -= dt.timedelta(days=1)

def check_url_available(session: requests.Session, url: str, timeout: int) -> bool:
    try:
        response = session.head(url, timeout=timeout, allow_redirects=True)
        if response.ok:
            return True
        if response.status_code not in {403, 404, 405, 500, 502, 503, 504}:
            return False
        response = session.get(url, timeout=timeout, stream=True)
        try:
            return response.ok and "text/html" not in response.headers.get("Content-Type", "")
        finally:
            response.close()
    except requests.RequestException:
        return False

def find_latest_complete_run(
    session: requests.Session,
    reference_time: dt.datetime,
    lookback_cycles: int,
    timeout: int,
) -> RunSpec:
    for run_spec in candidate_runs(reference_time=reference_time, lookback_cycles=lookback_cycles):
        first_url = run_spec.build_download_url(FORECAST_HOURS[0])
        last_url = run_spec.build_download_url(FORECAST_HOURS[-1])
        if check_url_available(session, first_url, timeout) and check_url_available(
            session, last_url, timeout
        ):
            print(f"Selected run {run_spec.run_stamp}.")
            return run_spec

    raise RuntimeError(
        "Could not find a complete GFS run with both f000 and f384 available "
        f"within the last {lookback_cycles} cycles."
    )

def make_output_dirs(output_root: Path, run_spec: RunSpec) -> tuple[Path, Path, Path, Path]:
    run_root = output_root / f"gfs_thunderstorm_outlook_{run_spec.run_stamp}"
    grib_dir = run_root / "grib"
    png_dir = run_root / "png"
    geojson_dir = run_root / "geojson"
    grib_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    geojson_dir.mkdir(parents=True, exist_ok=True)
    return run_root, grib_dir, png_dir, geojson_dir

def parse_run_dir_cycle_time(path: Path) -> dt.datetime | None:
    prefix = "gfs_thunderstorm_outlook_"
    if not path.is_dir() or not path.name.startswith(prefix):
        return None
    stamp = path.name[len(prefix) :]
    try:
        return dt.datetime.strptime(stamp, "%Y%m%d_%Hz").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None

def prune_old_archives(output_root: Path, reference_time: dt.datetime) -> None:
    cutoff_time = reference_time - ARCHIVE_RETENTION
    if not output_root.exists():
        return
    for child in output_root.iterdir():
        cycle_time = parse_run_dir_cycle_time(child)
        if cycle_time is None or cycle_time >= cutoff_time:
            continue
        shutil.rmtree(child)
        print(f"Removed archived run {child.name} older than {ARCHIVE_RETENTION.days} days.")

def download_file(
    session: requests.Session,
    run_spec: RunSpec,
    forecast_hour: int,
    destination: Path,
    timeout: int,
) -> None:
    url = run_spec.build_download_url(forecast_hour)
    response = session.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    with destination.open("wb") as file_handle:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if chunk:
                file_handle.write(chunk)

def find_matching_message(grib_file: pygrib.open, short_names: set[str], name_fragments: tuple[str, ...]):
    for message in grib_file:
        short_name = getattr(message, "shortName", "").lower()
        name = getattr(message, "name", "").lower()
        if short_name in short_names:
            return message
        if any(fragment in name for fragment in name_fragments):
            return message
    raise KeyError(f"Could not find GRIB field matching {sorted(short_names)}.")

def read_field(message) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat1 = DATA_EXTENT[2]
    lat2 = DATA_EXTENT[3]
    lon1 = DATA_EXTENT[0] % 360.0
    lon2 = DATA_EXTENT[1] % 360.0
    values, lats, lons = message.data(lat1=lat1, lat2=lat2, lon1=lon1, lon2=lon2)
    lons = np.where(lons > 180.0, lons - 360.0, lons)
    return values.astype(np.float64), lats, lons

def read_outlook_fields(grib_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dt.datetime]:
    with pygrib.open(str(grib_path)) as grib_file:
        messages = list(grib_file)

    def lookup(short_names: set[str], name_fragments: tuple[str, ...]):
        for message in messages:
            short_name = getattr(message, "shortName", "").lower()
            name = getattr(message, "name", "").lower()
            if short_name in short_names or any(fragment in name for fragment in name_fragments):
                return message
        raise KeyError(f"Could not find GRIB field matching {sorted(short_names)}.")

    def lookup_optional(short_names: set[str], name_fragments: tuple[str, ...]):
        for message in messages:
            short_name = getattr(message, "shortName", "").lower()
            name = getattr(message, "name", "").lower()
            if short_name in short_names or any(fragment in name for fragment in name_fragments):
                return message
        return None

    lifted_message = lookup({"4lftx", "lftx"}, ("lifted index", "4-layer lifted index"))
    cape_message = lookup({"cape"}, ("convective available potential energy",))
    dzdt_message = lookup({"dzdt"}, ("vertical velocity", "geopotential tendency"))
    acpcp_message = lookup_optional({"acpcp"}, ("convective precipitation",))

    lifted, lats, lons = read_field(lifted_message)
    cape, _, _ = read_field(cape_message)
    dzdt, _, _ = read_field(dzdt_message)
    if acpcp_message is not None:
        acpcp, _, _ = read_field(acpcp_message)
    else:
        acpcp = np.zeros_like(lifted)
    valid_time = lifted_message.validDate.replace(tzinfo=dt.timezone.utc)

    fields = {
        "lifted": lifted,
        "cape": cape,
        "dzdt": dzdt,
        "acpcp": acpcp,
    }
    return fields, lats, lons, valid_time

def smooth_field(values: np.ndarray, passes: int = 2) -> np.ndarray:
    smoothed = values.astype(np.float64, copy=True)
    for _ in range(passes):
        padded = np.pad(smoothed, 1, mode="edge")
        smoothed = (
            padded[:-2, :-2]
            + 2.0 * padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + 2.0 * padded[1:-1, :-2]
            + 4.0 * padded[1:-1, 1:-1]
            + 2.0 * padded[1:-1, 2:]
            + padded[2:, :-2]
            + 2.0 * padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 16.0
    return smoothed

def expand_mask(mask: np.ndarray, threshold: int = 1) -> np.ndarray:
    padded = np.pad(mask.astype(np.int8), 1, mode="edge")
    neighborhood = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return neighborhood >= threshold

def erode_mask(mask: np.ndarray, threshold: int = 9) -> np.ndarray:
    padded = np.pad(mask.astype(np.int8), 1, mode="edge")
    neighborhood = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return neighborhood >= threshold

def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    filled = mask.copy()
    rows, cols = filled.shape
    visited = np.zeros_like(filled, dtype=bool)
    stack: list[tuple[int, int]] = []

    for row in range(rows):
        for col in (0, cols - 1):
            if not filled[row, col] and not visited[row, col]:
                stack.append((row, col))
    for col in range(cols):
        for row in (0, rows - 1):
            if not filled[row, col] and not visited[row, col]:
                stack.append((row, col))

    while stack:
        row, col = stack.pop()
        if row < 0 or row >= rows or col < 0 or col >= cols:
            continue
        if visited[row, col] or filled[row, col]:
            continue
        visited[row, col] = True
        stack.append((row - 1, col))
        stack.append((row + 1, col))
        stack.append((row, col - 1))
        stack.append((row, col + 1))

    holes = (~filled) & (~visited)
    filled[holes] = True
    return filled

def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    kept = np.zeros_like(mask, dtype=bool)
    visited = np.zeros_like(mask, dtype=bool)
    rows, cols = mask.shape

    for start_row in range(rows):
        for start_col in range(cols):
            if visited[start_row, start_col] or not mask[start_row, start_col]:
                continue

            stack = [(start_row, start_col)]
            component: list[tuple[int, int]] = []
            visited[start_row, start_col] = True

            while stack:
                row, col = stack.pop()
                component.append((row, col))
                for row_offset in (-1, 0, 1):
                    for col_offset in (-1, 0, 1):
                        if row_offset == 0 and col_offset == 0:
                            continue
                        next_row = row + row_offset
                        next_col = col + col_offset
                        if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                            continue
                        if visited[next_row, next_col] or not mask[next_row, next_col]:
                            continue
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))

            if len(component) >= min_size:
                for row, col in component:
                    kept[row, col] = True

    return kept

def derive_outlook_categories(fields: dict[str, np.ndarray], forecast_hour: int) -> np.ndarray:
    lifted = fields["lifted"]
    cape = fields["cape"]
    dzdt = fields["dzdt"]
    acpcp = fields["acpcp"]

    if forecast_hour == 0:
        # For F000, do not use convective precip in logic
        small_mask = (lifted <= 0.0) & (cape >= 100.0)
        medium_mask = (lifted <= -2.0) & (cape >= 500.0) & (dzdt >= 0.01)
        high_mask = (lifted <= -4.0) & (cape >= 1500.0) & (dzdt >= 0.03)
    else:
        small_mask = (lifted <= 0.0) & (cape >= 100.0) & (acpcp >= 0.1)
        medium_mask = (lifted <= -2.0) & (cape >= 500.0) & (dzdt >= 0.01) & (acpcp >= 0.25)
        high_mask = (lifted <= -4.0) & (cape >= 1500.0) & (dzdt >= 0.03) & (acpcp >= 0.5)

    small_mask = fill_mask_holes(small_mask)
    medium_mask = fill_mask_holes(medium_mask)
    high_mask = fill_mask_holes(high_mask)

    categories = np.zeros_like(lifted, dtype=np.int16)
    categories[small_mask] = 2
    categories[medium_mask] = 3
    categories[high_mask] = 4
    return categories

def isoformat_compact(timestamp: dt.datetime) -> str:
    return timestamp.strftime("%Y%m%d%H%M")

def format_display_time(timestamp: dt.datetime) -> str:
    # If the time is exactly 12Z or 18Z, use 8am/2pm local time for display
    hour_utc = timestamp.hour
    if hour_utc == 12:
        label = "8 AM"
    elif hour_utc == 18:
        label = "2 PM"
    else:
        local_time = timestamp.astimezone(DISPLAY_TIMEZONE)
        label = local_time.strftime("%I %p").lstrip("0")
    day = timestamp.astimezone(DISPLAY_TIMEZONE).strftime('%A')
    return f"{day} {label}"

def contour_segments_to_multipolygon(segments) -> list[list[list[list[float]]]]:
    multipolygon: list[list[list[list[float]]]] = []
    for segment in segments:
        polygon = np.asarray(segment, dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[0] < 4 or polygon.shape[1] != 2:
            continue
        if not np.allclose(polygon[0], polygon[-1]):
            polygon = np.vstack([polygon, polygon[0]])
        multipolygon.append([polygon.tolist()])
    return multipolygon

def build_geojson(
    contour_set,
    category: OutlookCategory,
    valid_time: dt.datetime,
    expire_time: dt.datetime,
    issue_time: dt.datetime,
) -> dict:
    features = []
    for segments in contour_set.allsegs:
        geometry = contour_segments_to_multipolygon(segments)
        if not geometry:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "MultiPolygon", "coordinates": geometry},
                "properties": {
                    "DN": category.dn,
                    "VALID": isoformat_compact(valid_time),
                    "EXPIRE": isoformat_compact(expire_time),
                    "ISSUE": isoformat_compact(issue_time),
                    "VALID_ISO": valid_time.isoformat(),
                    "EXPIRE_ISO": expire_time.isoformat(),
                    "ISSUE_ISO": issue_time.isoformat(),
                    "FORECASTER": "GPT-5.4 model-derived",
                    "LABEL": category.label,
                    "LABEL2": category.label2,
                    "stroke": category.stroke,
                    "fill": category.fill,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}

def build_level_display_mask(categories: np.ndarray, level: int) -> np.ndarray:
    level_mask = fill_mask_holes(categories >= level)
    min_component_size = {
        2: 90,
        3: 70,
        4: 45,
    }.get(level, 25)
    level_mask = remove_small_components(level_mask, min_size=min_component_size)
    if not level_mask.any():
        return np.zeros_like(categories, dtype=np.float64)
    if level >= 3:
        eroded_mask = erode_mask(level_mask)
        if eroded_mask.any():
            level_mask = eroded_mask
    level_domain = fill_mask_holes(expand_mask(level_mask, threshold=1))
    return np.where(level_domain, smooth_field(level_mask.astype(float), passes=3), 0.0)

def render_outlook(
    png_path: Path,
    geojson_path: Path,
    run_spec: RunSpec,
    forecast_hour: int,
    lats: np.ndarray,
    lons: np.ndarray,
    categories: np.ndarray,
    valid_time: dt.datetime,
) -> None:
    figure = plt.figure(figsize=(15, 9))
    axis = plt.axes(projection=ccrs.LambertConformal(central_longitude=-96, central_latitude=38))
    axis.set_extent(USA_EXTENT, crs=ccrs.PlateCarree())

    axis.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f7f4ef", zorder=0)
    axis.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dfeefa", zorder=0)
    axis.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#dfeefa", edgecolor="none")
    axis.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7)
    axis.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.7)
    axis.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.45, edgecolor="#4a4a4a")

    contour_sets = []
    for index, category in enumerate(OUTLOOK_CATEGORIES, start=2):
        band_mask = build_level_display_mask(categories, index)
        if np.max(band_mask) < 0.45:
            continue

        contour_set = axis.contourf(
            lons,
            lats,
            band_mask,
            levels=[0.45, 1.01],
            colors=[category.fill],
            alpha=0.38,
            antialiased=True,
            transform=ccrs.PlateCarree(),
            zorder=2 + index,
        )
        contour_sets.append((category, contour_set))
        axis.contour(
            lons,
            lats,
            band_mask,
            levels=[0.5],
            colors=[category.stroke],
            linewidths=1.0,
            transform=ccrs.PlateCarree(),
            zorder=10 + index,
        )

    init_label = format_display_time(run_spec.cycle_time)
    valid_label = format_display_time(valid_time)
    run_label = run_spec.run_stamp.replace('_', ' ').replace('z', 'Z')
    axis.set_title(
        f"GFS Model-Derived Thunderstorm Outlook\nRun: {run_label}   Init: {init_label}   Valid: {valid_label}   Forecast Hour: F{forecast_hour:03d}",
        fontsize=16,
        weight="bold",
        loc="left",
    )

    legend_handles = [
        plt.Line2D([0], [0], color=category.stroke, linewidth=8, solid_capstyle="butt")
        for category in OUTLOOK_CATEGORIES
    ]
    legend_labels = [category.label for category in OUTLOOK_CATEGORIES]
    axis.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        framealpha=0.92,
        title="Category",
    )

    plt.tight_layout()
    figure.savefig(png_path, dpi=150, bbox_inches="tight")

    expire_time = valid_time + dt.timedelta(hours=6)
    feature_collection = {"type": "FeatureCollection", "features": []}
    for category, contour_set in contour_sets:
        geojson = build_geojson(
            contour_set=contour_set,
            category=category,
            valid_time=valid_time,
            expire_time=expire_time,
            issue_time=run_spec.cycle_time,
        )
        if geojson["features"]:
            for feature in geojson["features"]:
                feature["properties"]["DN"] = category.dn
                feature["properties"]["LABEL"] = category.label
                feature["properties"]["LABEL2"] = category.label2
                feature["properties"]["stroke"] = "#000000"
                feature["properties"]["fill"] = category.fill
            feature_collection["features"].extend(geojson["features"])
    geojson_path.write_text(json.dumps(feature_collection, indent=2), encoding="utf-8")
    plt.close(figure)

def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser()
    session = requests.Session()
    session.headers.update({"User-Agent": "gfs-thunderstorm-outlook/1.0"})

    try:
        run_spec = find_latest_complete_run(
            session=session,
            reference_time=dt.datetime.now(tz=dt.timezone.utc),
            lookback_cycles=args.lookback_cycles,
            timeout=args.timeout,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    run_root, grib_dir, png_dir, geojson_dir = make_output_dirs(output_root, run_spec)
    prune_old_archives(output_root, run_spec.cycle_time)
    print(f"Writing outputs under {run_root.resolve()}")

    for forecast_hour in FORECAST_HOURS:
        base_name = f"gfs_thunderstorm_outlook_{run_spec.run_stamp}_f{forecast_hour:03d}"
        grib_path = grib_dir / f"{base_name}.grib2"
        png_path = png_dir / f"{base_name}.png"
        geojson_path = geojson_dir / f"{base_name}.geojson"

        if not (args.skip_existing and grib_path.exists()):
            print(f"Downloading F{forecast_hour:03d} -> {grib_path.name}")
            try:
                download_file(
                    session=session,
                    run_spec=run_spec,
                    forecast_hour=forecast_hour,
                    destination=grib_path,
                    timeout=args.timeout,
                )
            except requests.RequestException as exc:
                print(
                    f"Failed to download forecast hour F{forecast_hour:03d}: {exc}",
                    file=sys.stderr,
                )
                return 1

        if args.skip_existing and png_path.exists() and geojson_path.exists():
            continue

        print(f"Building thunderstorm outlook for F{forecast_hour:03d}")
        try:
            fields, lats, lons, valid_time = read_outlook_fields(grib_path)
            categories = derive_outlook_categories(fields, forecast_hour)
            render_outlook(
                png_path=png_path,
                geojson_path=geojson_path,
                run_spec=run_spec,
                forecast_hour=forecast_hour,
                lats=lats,
                lons=lons,
                categories=categories,
                valid_time=valid_time,
            )
        except (OSError, RuntimeError, ValueError, IndexError, KeyError) as exc:
            print(f"Failed to generate outlook for {grib_path.name}: {exc}", file=sys.stderr)
            return 1

    print("Finished generating model-derived thunderstorm outlooks.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
