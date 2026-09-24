from datetime import datetime, timedelta
from pathlib import Path
import shutil

import pandas as pd
import piexif
from PIL import Image
from PyQt6.QtWidgets import QApplication

import main as main_module
from main import (
    DEFAULT_DATA_TIMEZONE,
    GeoTaggerApp,
    SequenceReviewDialog,
    WorkerThread,
    _continuous_camera_group_chains,
    _correct_media_timestamp_utc,
    _format_map_gps_timestamp,
    _gps_parse_cache_key,
)


def test_default_data_timezone_is_sydney():
    assert DEFAULT_DATA_TIMEZONE == "Australia/Sydney"


def test_tlog_media_time_is_converted_from_laptop_timezone_to_utc():
    camera_time = datetime(2026, 3, 27, 7, 38, 49)

    corrected = _correct_media_timestamp_utc(
        camera_time,
        timedelta(0),
        "Australia/Sydney",
        interpret_media_time_as_local=True,
    )
    legacy = _correct_media_timestamp_utc(
        camera_time,
        timedelta(0),
        "Australia/Sydney",
        interpret_media_time_as_local=False,
    )

    assert corrected == pd.Timestamp("2026-03-26T20:38:49Z")
    assert legacy == pd.Timestamp("2026-03-27T07:38:49Z")


def test_map_gps_timestamp_uses_selected_laptop_timezone():
    timestamp = pd.Timestamp("2026-03-26T20:10:24.768Z")

    assert _format_map_gps_timestamp(timestamp, "Australia/Sydney") == (
        "2026-03-27 07:10:24.768 (Australia/Sydney)"
    )


def test_tlog_parsing_params_keep_all_selected_files_and_laptop_timezone():
    class AppStub:
        gps_file_path = "first.tlog"
        gps_file_paths = ["first.tlog", "second.tlog"]
        current_timezone_str = "Australia/Sydney"

    params = GeoTaggerApp._get_gps_parsing_params_from_ui(AppStub())

    assert params == {
        "type": "tlog",
        "paths": ["first.tlog", "second.tlog"],
        "boat_timezone": "Australia/Sydney",
    }


def test_tlog_parse_cache_is_not_invalidated_by_display_timezone(tmp_path):
    tlog_path = tmp_path / "survey.tlog"
    sydney = {
        "type": "tlog",
        "paths": [str(tlog_path)],
        "boat_timezone": "Australia/Sydney",
    }
    perth = {
        "type": "tlog",
        "paths": [str(tlog_path)],
        "boat_timezone": "Australia/Perth",
    }

    assert _gps_parse_cache_key(str(tlog_path), sydney) == _gps_parse_cache_key(
        str(tlog_path),
        perth,
    )


def test_worker_reuses_preliminary_gps_and_media_scans(monkeypatch):
    gps_times = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-12-31T12:59:59Z"),
            pd.Timestamp("2025-12-31T13:00:01Z"),
        ]
    )
    gps = pd.DataFrame(
        {
            "latitude": [-33.0, -32.999],
            "longitude": [151.0, 151.001],
        },
        index=gps_times,
    )
    media_items = [
        {
            "identifier": "frame.jpg",
            "file_path": "frame.jpg",
            "image_time": datetime(2026, 1, 1, 0, 0, 0),
            "time_group": ".",
        }
    ]

    def unexpected_rescan(*_args, **_kwargs):
        raise AssertionError("validated preliminary inputs should be reused")

    monkeypatch.setattr(
        main_module.engine,
        "parse_mavlink_tlog_files",
        unexpected_rescan,
    )
    monkeypatch.setattr(
        main_module.engine,
        "get_image_files_and_times",
        unexpected_rescan,
    )
    worker = WorkerThread(
        "survey.tlog",
        {
            "type": "tlog",
            "paths": ["survey.tlog"],
            "boat_timezone": "Australia/Sydney",
        },
        "images",
        "images",
        "manual",
        {"offset_seconds": 0},
        preloaded_gps_df=gps,
        preloaded_media_items=media_items,
    )
    monkeypatch.setattr(worker, "generate_combined_map", lambda *_args: None)

    worker.run()

    assert worker.results_df is not None
    assert len(worker.results_df) == 1
    assert worker.results_df.iloc[0]["Latitude"] == -32.9995


def test_show_all_dates_timezone_change_updates_labels_without_map_rebuild():
    class LabelStub:
        def setText(self, text):
            self.text = text

    class ComboStub:
        def __init__(self):
            self.items = ["Show All Dates"]
            self.index = 0

        def blockSignals(self, _blocked):
            pass

        def currentText(self):
            return self.items[self.index]

        def clear(self):
            self.items = []
            self.index = 0

        def addItem(self, item):
            self.items.append(item)

        def findText(self, text):
            return self.items.index(text) if text in self.items else -1

        def setCurrentIndex(self, index):
            self.index = index

        def currentIndex(self):
            return self.index

        def setEnabled(self, _enabled):
            pass

    class AppStub:
        current_timezone_str = "Australia/Sydney"
        processed_gps_df = pd.DataFrame(
            {"latitude": [-33.0], "longitude": [151.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-01-01T00:00:00Z")]),
        )
        date_filter_label = LabelStub()
        header_timezone_label = LabelStub()
        date_filter_combo = ComboStub()
        media_timezone_input = ComboStub()
        map_label_updates = 0

        def log_message(self, *_args):
            pass

        def _update_map_timezone_labels(self):
            self.map_label_updates += 1

        def regenerate_map_display(self):
            raise AssertionError("Show All Dates must not rebuild the full map")

        def update_transect_ui(self):
            pass

        def _populate_loaded_transects_list(self):
            pass

    app = AppStub()
    GeoTaggerApp.update_current_timezone(app, "Australia/Perth")

    assert app.current_timezone_str == "Australia/Perth"
    assert app.map_label_updates == 1
    assert app.filtered_gps_by_date_df is app.processed_gps_df


def test_extraction_writes_gps_and_upgrades_existing_untagged_copy(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "frame.jpg"
    exif_bytes = piexif.dump(
        {
            "0th": {},
            "Exif": {
                piexif.ExifIFD.DateTimeOriginal: b"2026:03:27 07:00:00",
            },
            "GPS": {},
            "1st": {},
            "thumbnail": None,
        }
    )
    Image.new("RGB", (16, 12), "white").save(source, exif=exif_bytes)
    source_bytes = source.read_bytes()
    result_row = pd.Series(
        {
            "Latitude": -29.0551234,
            "Longitude": 167.9534567,
            "Corrected Timestamp (UTC)": pd.Timestamp(
                "2026-03-26T20:00:00.500Z"
            ),
        }
    )

    class AppStub:
        media_path = str(source_dir)
        _extract_exif_gps = GeoTaggerApp._extract_exif_gps
        _geotag_values_from_result_row = staticmethod(
            GeoTaggerApp._geotag_values_from_result_row
        )
        _existing_extraction_matches_geotag = (
            GeoTaggerApp._existing_extraction_matches_geotag
        )
        _write_geotagged_extraction = GeoTaggerApp._write_geotagged_extraction

        def log_message(self, *_args):
            pass

    app = AppStub()
    output_dir = tmp_path / "transect"
    output_dir.mkdir()
    untagged_output = output_dir / source.name
    shutil.copy2(source, untagged_output)

    first = app._write_geotagged_extraction(
        result_row,
        str(source),
        str(output_dir),
    )
    output_gps = piexif.load(first["destination_path"])["GPS"]

    assert first["written"]
    assert piexif.GPSIFD.GPSLatitude in output_gps
    assert piexif.GPSIFD.GPSLongitude in output_gps
    assert piexif.GPSIFD.GPSDateStamp in output_gps
    assert source.read_bytes() == source_bytes
    assert len(list(output_dir.glob("*.jpg"))) == 1

    second = app._write_geotagged_extraction(
        result_row,
        str(source),
        str(output_dir),
    )

    assert second["already_present"]
    assert not second["written"]
    assert len(list(output_dir.glob("*.jpg"))) == 1


def test_only_geotagger_generated_maps_are_treated_as_temporary():
    assert GeoTaggerApp._is_temporary_map_file("/tmp/geotagger_map_app_123.html")
    assert GeoTaggerApp._is_temporary_map_file("/tmp/geotagger_map_worker_123.html")
    assert not GeoTaggerApp._is_temporary_map_file("/tmp/survey_map.html")


def test_initial_map_regeneration_is_deferred_to_the_worker():
    class AppStub:
        _awaiting_worker_map = True

        def __init__(self):
            self.messages = []

        def log_message(self, message, _level):
            self.messages.append(message)

    app_stub = AppStub()

    GeoTaggerApp.regenerate_map_display(app_stub)

    assert any("Deferring duplicate map regeneration" in message for message in app_stub.messages)


def test_worker_map_escapes_image_identifiers_and_prefers_canvas():
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    gps = pd.DataFrame(
        {"latitude": [-33.0], "longitude": [151.0]},
        index=pd.DatetimeIndex([timestamp]),
    )
    results = pd.DataFrame(
        {
            "Identifier": ['<img src=x onerror="alert(1)">'],
            "Corrected Timestamp (UTC)": [timestamp],
            "Latitude": [-33.0],
            "Longitude": [151.0],
        }
    )
    worker = WorkerThread("", {}, "", "", "manual", {})

    map_path = worker.generate_combined_map(gps, results)
    assert map_path is not None
    try:
        html = Path(map_path).read_text(encoding="utf-8")
        assert '<img src=x onerror="alert(1)">' not in html
        assert "\\u003cimg" in html
        assert '"preferCanvas": true' in html
        assert "const gpsData =" in html
        assert "const imageData =" in html
        assert "updateMapTimezone" in html
        assert "2026-01-01 00:00:00.000 (UTC)" in html
        assert "Satellite - Latest" in html
        assert "Satellite - Clarity" in html
        assert "Place labels" in html
        assert "focusGpsTimeWindow" in html
        assert "GPS Logging Starts" in html
        assert '"maxNativeZoom": 18' in html
    finally:
        Path(map_path).unlink(missing_ok=True)


def test_sequence_review_pages_thumbnails_and_selects_a_sync_frame(tmp_path):
    qt_app = QApplication.instance() or QApplication([])
    media_items = []
    for index in range(14):
        image_path = tmp_path / f"frame_{index:02d}.jpg"
        Image.new("RGB", (24, 16), (index * 10, 40, 80)).save(image_path)
        media_items.append(
            {
                "file_path": str(image_path),
                "image_time": datetime(2026, 3, 27, 7, 0, index),
                "time_group": "SB_WS",
            }
        )

    dialog = SequenceReviewDialog(
        media_items,
        "SB_WS",
        "Australia/Sydney",
        offset_analysis={
            "suggested_offset_seconds": -3646,
            "best_offset_min_seconds": -3837,
            "best_offset_max_seconds": -3646,
        },
        display_name="South Bay",
    )

    assert "South Bay" in dialog.windowTitle()
    assert len(dialog.thumbnail_buttons) == 12
    assert "1–12 of 14" in dialog.page_label.text()

    selected = media_items[3]
    dialog._select_item(selected)
    assert dialog.selected_item == selected
    assert dialog.use_selected_button.isEnabled()

    dialog._next_page()
    assert len(dialog.thumbnail_buttons) == 2
    assert "13–14 of 14" in dialog.page_label.text()
    dialog.close()
    qt_app.processEvents()


def test_nearby_camera_folders_are_grouped_but_later_session_is_separate():
    groups = {
        "001_0017": [
            datetime(2026, 1, 1, 8, 0),
            datetime(2026, 1, 1, 8, 10),
        ],
        "001_0018": [
            datetime(2026, 1, 1, 8, 17),
            datetime(2026, 1, 1, 8, 25),
        ],
        "next_site": [
            datetime(2026, 1, 1, 10, 0),
            datetime(2026, 1, 1, 10, 10),
        ],
    }

    chains = _continuous_camera_group_chains(groups)

    assert [[name for name, _ in chain] for chain in chains] == [
        ["001_0017", "001_0018"],
        ["next_site"],
    ]
