# GeoTagger

GeoTagger is a desktop georeferencing tool for matching timestamped images, usually GoPro survey images, to a GPS track. It can load GPX, CSV, or BlueBoat MAVLink TLOG GPS data, align image timestamps to the track, display the result on an interactive map, write GPS EXIF tags into image copies, and extract images that fall along defined transects.

The application is aimed at field survey workflows where image timestamps, GPS tracks, and transect locations need to be reviewed quickly and repeatably.

## Features

- Load GPS tracks from `.gpx`, `.csv`, or `.txt` files.
- Select and merge multiple BlueBoat MAVLink `.tlog` files directly, without first exporting CSV.
- Map CSV columns for latitude, longitude, and timestamps.
- Recursively scan image folders for EXIF timestamps.
- Support sub-second EXIF timestamps where cameras provide them.
- Synchronise image times to GPS using manual offset, sync-point calibration, or visual image-to-map sync.
- Interpolate image positions along the GPS track.
- Display GPS points, geotagged image positions, and transects over selectable
  Esri latest/clarity satellite imagery or a street map.
- Filter GPS display and transect work by local date.
- Save geotagged image copies with GPS EXIF latitude, longitude, altitude, GPS date, and GPS time.
- Define transects manually on the map.
- Load transects from JSON or CSV.
- Batch-extract images near transect lines using a configurable buffer distance.
- Export the results table to CSV and save the map as standalone HTML.
- Responsive desktop workspace with scrollable **Setup & Sync** and
  **Transects** workflows, a persistent processing toolbar, and saved window
  and splitter sizing.

## Requirements

Python 3.9+ is recommended.

Install GeoTagger and its required packages from the repository:

```bash
python -m pip install .
```

Package roles:

| Package | Purpose |
| --- | --- |
| `PyQt6`, `PyQt6-WebEngine` | Desktop GUI and embedded map display |
| `pandas`, `numpy` | Data processing |
| `folium` | Interactive Leaflet map generation |
| `pytz` | Timezone handling |
| `piexif` | Reading and writing EXIF timestamps/GPS data |
| `gpxpy` | GPX parsing |
| `Pillow` | Reading existing EXIF GPS data from images |
| `geopy` | Transect length and distance calculations |
| `scipy` | Fast nearest-point lookups with KD-tree indexing |

`geopy` and `scipy` are optional in parts of the code, but they are strongly recommended. Without them, some distance-based transect features are disabled or slower.

## Run

From the folder containing `main.py` and `georeference_engine.py`:

```bash
python main.py
```

After installation, it can also be launched as:

```bash
geotagger
```

## Typical Workflow

1. Click **Load GPS Log(s)...** and select one GPX/CSV track or one or more TLOG files.
2. If loading a CSV/TXT track, map the latitude, longitude, and timestamp columns.
3. Click **Load Images...** and select the parent folder containing your images.
4. Set **Data Timezone** to the timezone used for your laptop/camera-time workflow.
5. Click **Analyze Folder Offsets** when the selected parent folder contains
   separate camera sessions, then review each suggested offset and its
   equal-score range.
6. For an ambiguous row, use **Visual Sync** on one recognisable image from
   that folder and click its matching GPS point.
7. Click **Process Data**.
8. Review image positions on the map and in the results table.
9. Define transects or load transects from JSON/CSV.
10. Extract images, export the table, save the map, or save geotagged image copies.

## Timestamp Workflow

GeoTagger expects image timestamps from EXIF to be naive camera times. The **Data Timezone** setting provides the timezone context used for display, GPS CSV parsing, transect CSV parsing, date filtering, and sync workflows.

In the field workflow this project was built around, GoPro and GPS time reference points are checked against laptop local time. When entering sync times, enter the times as they appear in that laptop/Data Timezone workflow, and keep **Data Timezone** set accordingly.

### Manual Offset

Manual offset is entered in seconds. It is used as the correction applied to media time before matching against GPS time.

Use a positive offset when the media clock is behind the GPS reference, and a negative offset when the media clock is ahead.

Examples:

- Camera time is 3 seconds behind the GPS reference: enter `3`.
- Camera time is 2.5 seconds ahead of the GPS reference: enter `-2.5`.

### Sync Point Calibration

Use sync-point calibration when you know a matching media time and GPS reference time for the same event. Enter both using your selected Data Timezone/laptop-time workflow.

### Visual Sync

Visual sync lets you select an image, then click the corresponding GPS point
on the map. The application calculates an exact offset for that image's
top-level camera folder. Other folder offsets are left unchanged.

### Per-Folder Offset Analysis

**Analyze Folder Offsets** treats each top-level folder beneath the selected
image directory as a separate camera-time group. It searches for the offset
that places the most image timestamps inside actual GPS logging sessions.

The table shows GPS coverage and an **Equal-Score Range**. If the camera
starts after GPS and stops before GPS, multiple offsets can achieve the same
coverage. A wide range is therefore labelled ambiguous and is not proof of
one exact correction. Use a recognisable image/map pair to visually calibrate
that folder. Suggested rows are not applied until **Use** is ticked. The
editable offset column also allows a measured correction to be entered
directly.

**Analyze nearby folders as one continuous camera clock** is enabled by
default. When one camera folder begins within 30 minutes of the preceding
folder, GeoTagger combines their timestamps and scores one shared offset using
the whole camera run. Rows are reported as **Shared**. A long folder can
therefore resolve several equally plausible GPS sessions for a short folder,
without inventing a clock change at a battery stop. Disable the option when
adjacent folders genuinely came from independently set camera clocks.

Select a folder row and click **Review First Photos** to inspect the first 12
frames as a timestamped filmstrip. Choose the first frame that clearly shows
the boat entering or moving in the water, then click **Use Selected Photo for
Visual Sync**. GeoTagger opens the map, colours every GPS point allowed by the
equal-score offset range amber, and zooms to that candidate section. Match the
selected frame to a durable feature such as a ramp, beach corner, headland, or
reef edge, then click the corresponding GPS point. That pair replaces the
ambiguous timing range with a visually confirmed folder offset.

The cyan-ring markers show the beginning of each contiguous GPS logging
interval. They are orientation cues only; a logging start is not assumed to be
the moment the boat enters the water.

### Satellite Map

**Satellite - Latest** is the default map. For remote sites, GeoTagger
overzooms the last reliable imagery tile instead of requesting a deeper zoom
that may be replaced by a data-unavailable tile. Use the layer control at the
top right to try **Satellite - Clarity**, which can make reef, surf, and beach
edges easier to distinguish, or turn on **Place labels**.

The shoreline in a satellite image is not an exact water-level measurement:
imagery can have been captured on a different date, tide, or sea state. Use it
to identify fixed coastal geometry, and use the amber candidate track plus a
recognisable camera event to establish the exact time match.

## GPS Inputs

### BlueBoat MAVLink TLOG

Select one or more `.tlog` files together in the GPS file picker. GeoTagger reads the timestamped MAVLink 1 or MAVLink 2 records directly, merges the selected logs chronologically, and removes duplicate timestamps. A selected file with no valid GPS positions is skipped with a warning instead of failing the rest of the batch.

For each file, GeoTagger prefers valid `GPS_RAW_INT` fixes because they include GPS fix quality, satellite count, and GPS altitude. If a log contains no valid raw GPS fixes, it falls back to `GLOBAL_POSITION_INT`.

The TLOG record timestamp comes from the computer that recorded the telemetry and is stored internally as UTC epoch time. Camera EXIF timestamps are interpreted in the selected **Data Timezone**, then converted to UTC before matching. Map popups and the rest of the interface display local time, so a Sydney laptop workflow appears as Sydney local time while the georeferencing calculations retain unambiguous UTC timestamps.

GeoTagger defaults **Data Timezone** to `Australia/Sydney`. The timezone remains selectable for exceptional datasets recorded using a different camera/laptop timezone.

GPS sessions separated by more than 30 seconds are treated as distinct.
GeoTagger will not draw an artificial straight-line interpolation across an
internal logging gap; images falling in such a gap remain ungeoreferenced.

No intermediate CSV export or column mapping is required.

### GPX

GPX files are parsed using `gpxpy`. GPX timestamps are treated as UTC.

### CSV/TXT

CSV/TXT GPS logs must contain latitude, longitude, and timestamp data. Timestamp data can be one combined datetime column or separate date and time columns.

When parsing CSV/TXT GPS logs, choose the matching timestamp format string, for example:

```text
%Y-%m-%d %H:%M:%S
```

The selected **Data Timezone** is used to localize naive CSV timestamps before converting them internally for processing.

Rows with valid latitude/longitude but unparseable timestamps are still shown on the map. Their popups display **Time was not identified**, and those points are excluded from time-based georeferencing, transect picks, and date filtering.

Rows with missing or out-of-range coordinates are ignored. Ambiguous or nonexistent daylight-saving wall times are retained as unidentified timestamps instead of aborting the entire import.

## Image Inputs

GeoTagger recursively scans the selected image folder for supported image types:

- `.jpg`
- `.jpeg`
- `.tif`
- `.tiff`

Images without usable EXIF capture timestamps are skipped during normal GPS-track processing.

Generated GeoTagger output directories are excluded from recursive scans so that extracted or geotagged copies are not loaded again as source images.

## Transects

GeoTagger supports three transect workflows.

### Manual Transects

After processing GPS and image data, click **Define Transect Start/End**, then click the start and end GPS points on the map. When prompted, name the transect. Images with corrected timestamps inside that start/end time window are copied to:

```text
transects_output/<transect_name>/
```

### JSON Transects

JSON transect sets define transects by coordinates:

```json
[
  {
    "name": "Site1_T1",
    "start_lat": -33.8500,
    "start_lon": 151.2100,
    "end_lat": -33.8501,
    "end_lon": 151.2101,
    "length": 10.0
  }
]
```

Required fields:

- `name`
- `start_lat`
- `start_lon`
- `end_lat`
- `end_lon`

Optional field:

- `length`

### CSV Transects

Transects can also be loaded from CSV. The app supports two styles:

- Coordinate-based CSVs with transect name, start latitude/longitude, and end latitude/longitude.
- Time-based CSVs with transect name, start time, end time, and optional length.

For time-based transect CSVs, timestamps are interpreted using the selected **Data Timezone**.

## Batch Extraction

Batch extraction currently copies images whose georeferenced points fall within a configurable buffer distance of each transect line. The default buffer is 2 metres.

Output is written to:

```text
transects_output_batch/<transect_name>/
```

If expected images are missing from a transect, increase the buffer slightly or check the time sync and GPS/image alignment. If too many images are included, reduce the buffer or inspect the transect coordinates.

## Output Tools

### Export Table CSV

Exports the full results table, including identifiers, file paths, timestamps, latitude, and longitude.

### Save Map HTML

Saves the current interactive map as a standalone HTML file.

### Save Geotagged

Copies JPEG images to an output folder and writes GPS EXIF tags into the copies. Source images are not modified, and the destination is replaced only after EXIF writing succeeds.

TIFF files can be loaded and georeferenced, but GPS EXIF writing is currently limited to `.jpg` and `.jpeg`. TIFFs are skipped with a clear log message rather than producing an incomplete output file.

### Extract Selected Images

Copies only the currently selected result-table images into a chosen output folder.

## Notes And Limitations

- Very large GPS tracks can create large map HTML files because each clickable GPS point is rendered as a map marker.
- If different subfolders contain images with the same filename, the first copy keeps its basename and later copies receive a deterministic suffix. Existing identical copies are reused.
- Keep `main.py` and `georeference_engine.py` in the same folder unless packaging the project differently.
- The application writes logs to `geotagger_app.log` when run.

## Development

Install the development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the automated checks:

```bash
python -m ruff check .
python -m pytest
```

GitHub Actions runs the same checks on Windows and Linux with supported Python versions.

## Project Files

```text
main.py
georeference_engine.py
file_utils.py
README.md
pyproject.toml
tests/
```

Do not commit `__pycache__` or generated output folders.
