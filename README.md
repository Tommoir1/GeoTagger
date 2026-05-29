# GeoTagger

GeoTagger is a desktop georeferencing tool for matching timestamped images, usually GoPro survey images, to a GPS track. It can load GPX or CSV GPS data, align image timestamps to the track, display the result on an interactive map, write GPS EXIF tags into image copies, and extract images that fall along defined transects.

The application is aimed at field survey workflows where image timestamps, GPS tracks, and transect locations need to be reviewed quickly and repeatably.

## Features

- Load GPS tracks from `.gpx`, `.csv`, or `.txt` files.
- Map CSV columns for latitude, longitude, and timestamps.
- Recursively scan image folders for EXIF timestamps.
- Support sub-second EXIF timestamps where cameras provide them.
- Synchronise image times to GPS using manual offset, sync-point calibration, or visual image-to-map sync.
- Interpolate image positions along the GPS track.
- Display GPS points, geotagged image positions, and transects on a Folium map.
- Filter GPS display and transect work by local date.
- Save geotagged image copies with GPS EXIF latitude, longitude, altitude, GPS date, and GPS time.
- Define transects manually on the map.
- Load transects from JSON or CSV.
- Batch-extract images near transect lines using a configurable buffer distance.
- Export the results table to CSV and save the map as standalone HTML.

## Requirements

Python 3.9+ is recommended.

Install the required packages:

```bash
pip install PyQt6 PyQt6-WebEngine pandas numpy folium pytz piexif gpxpy Pillow geopy scipy
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

## Typical Workflow

1. Click **Load GPS...** and select a GPX, CSV, or TXT GPS track.
2. If loading a CSV/TXT track, map the latitude, longitude, and timestamp columns.
3. Click **Load Images...** and select the parent folder containing your images.
4. Set **Data Timezone** to the timezone used for your laptop/camera-time workflow.
5. Choose a time synchronisation method.
6. Click **Process Data**.
7. Review image positions on the map and in the results table.
8. Define transects or load transects from JSON/CSV.
9. Extract images, export the table, save the map, or save geotagged image copies.

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

Visual sync lets you select an image, then click the corresponding GPS point on the map. The application calculates the offset from that pair and fills the manual offset control.

## GPS Inputs

### GPX

GPX files are parsed using `gpxpy`. GPX timestamps are treated as UTC.

### CSV/TXT

CSV/TXT GPS logs must contain latitude, longitude, and timestamp data. Timestamp data can be one combined datetime column or separate date and time columns.

When parsing CSV/TXT GPS logs, choose the matching timestamp format string, for example:

```text
%Y-%m-%d %H:%M:%S
```

The selected **Data Timezone** is used to localize naive CSV timestamps before converting them internally for processing.

## Image Inputs

GeoTagger recursively scans the selected image folder for supported image types:

- `.jpg`
- `.jpeg`
- `.tif`
- `.tiff`

Images without usable EXIF capture timestamps are skipped during normal GPS-track processing.

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

Copies images to an output folder and writes GPS EXIF tags into the copies. Source images are not modified.

### Extract Selected Images

Copies only the currently selected result-table images into a chosen output folder.

## Notes And Limitations

- Very large GPS tracks can create large map HTML files because each clickable GPS point is rendered as a map marker.
- If different subfolders contain images with the same filename, extraction outputs may overwrite files when copying by basename.
- Keep `main.py` and `georeference_engine.py` in the same folder unless packaging the project differently.
- The application writes logs to `geotagger_app.log` when run.

## Project Files

```text
main.py
georeference_engine.py
README.md
```

Do not commit `__pycache__` or generated output folders.
