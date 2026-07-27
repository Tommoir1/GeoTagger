from datetime import datetime, timedelta
from pathlib import Path
import struct

import numpy as np
import pandas as pd
import piexif
import pytz
from PIL import Image

import georeference_engine as engine


def _mavlink2_tlog_record(timestamp, message_id, payload, sequence=0):
    timestamp_bytes = struct.pack(">Q", int(timestamp * 1_000_000))
    header = bytes(
        [
            len(payload),
            0,
            0,
            sequence,
            1,
            1,
        ]
    ) + message_id.to_bytes(3, byteorder="little")
    return timestamp_bytes + b"\xfd" + header + payload + b"\x00\x00"


def _gps_raw_payload(latitude, longitude, altitude_m, fix_type=3, satellites=15):
    return struct.pack(
        "<QiiiHHHHBB",
        1_000_000,
        round(latitude * 1e7),
        round(longitude * 1e7),
        round(altitude_m * 1000),
        75,
        100,
        0,
        0,
        fix_type,
        satellites,
    )


def _global_position_payload(latitude, longitude, altitude_m):
    return struct.pack(
        "<IiiiihhhH",
        1000,
        round(latitude * 1e7),
        round(longitude * 1e7),
        round(altitude_m * 1000),
        0,
        0,
        0,
        0,
        0,
    )


def _write_timestamped_jpeg(path: Path, timestamp: str = "2026:01:15 10:00:00") -> None:
    exif_bytes = piexif.dump(
        {
            "0th": {},
            "Exif": {piexif.ExifIFD.DateTimeOriginal: timestamp.encode("ascii")},
            "GPS": {},
            "1st": {},
            "thumbnail": None,
        }
    )
    Image.new("RGB", (8, 8), "white").save(path, format="JPEG", exif=exif_bytes)


def test_manual_time_offset_preserves_existing_semantics():
    assert engine.calculate_time_offset("manual", offset_seconds=2.5) == timedelta(seconds=2.5)


def test_sync_point_uses_selected_laptop_timezone():
    camera_time = datetime(2026, 1, 15, 10, 0, 0)
    same_instant_utc = pytz.utc.localize(datetime(2026, 1, 14, 23, 0, 0))

    offset = engine.calculate_time_offset(
        "sync_point",
        gopro_sync_time=camera_time,
        gps_sync_time=same_instant_utc,
        boat_timezone_str="Australia/Sydney",
    )

    assert offset == timedelta(0)


def test_csv_invalid_coordinates_are_removed(tmp_path):
    csv_path = tmp_path / "track.csv"
    csv_path.write_text(
        "time,lat,lon\n"
        "2026-01-01 00:00:00,-33.0,151.0\n"
        "2026-01-01 00:00:01,95.0,181.0\n",
        encoding="utf-8",
    )

    result = engine.parse_boat_log_csv(
        csv_path,
        lat_col="lat",
        lon_col="lon",
        boat_timezone_str="UTC",
        datetime_col="time",
        datetime_format="%Y-%m-%d %H:%M:%S",
    )

    assert len(result) == 1
    assert result.iloc[0]["latitude"] == -33.0
    assert result.attrs["invalid_coordinate_rows"] == 1


def test_csv_dst_ambiguity_is_kept_as_an_unidentified_time(tmp_path):
    csv_path = tmp_path / "track.csv"
    csv_path.write_text(
        "time,lat,lon\n2026-04-05 02:30:00,-33.0,151.0\n",
        encoding="utf-8",
    )

    result = engine.parse_boat_log_csv(
        csv_path,
        lat_col="lat",
        lon_col="lon",
        boat_timezone_str="Australia/Sydney",
        datetime_col="time",
        datetime_format="%Y-%m-%d %H:%M:%S",
    )

    assert len(result) == 1
    assert result.index.isna().all()
    assert not result.iloc[0]["gps_time_identified"]


def test_csv_timezone_aware_timestamp_keeps_its_embedded_offset(tmp_path):
    csv_path = tmp_path / "track.csv"
    csv_path.write_text(
        "time,lat,lon\n2026-01-15T10:00:00+1100,-33.0,151.0\n",
        encoding="utf-8",
    )

    result = engine.parse_boat_log_csv(
        csv_path,
        lat_col="lat",
        lon_col="lon",
        boat_timezone_str="UTC",
        datetime_col="time",
        datetime_format="%Y-%m-%dT%H:%M:%S%z",
    )

    assert result.index[0] == pd.Timestamp("2026-01-14T23:00:00Z")


def test_tlog_multi_file_import_prefers_raw_gps_and_best_duplicate_fix(tmp_path):
    first_path = tmp_path / "first.tlog"
    second_path = tmp_path / "second.tlog"
    first_timestamp = pd.Timestamp("2026-03-26T00:44:03.866Z").timestamp()
    second_timestamp = first_timestamp + 2

    first_path.write_bytes(
        _mavlink2_tlog_record(
            first_timestamp,
            24,
            _gps_raw_payload(-29.0, 167.9, 3.5, fix_type=3, satellites=12),
        )
        + _mavlink2_tlog_record(
            first_timestamp + 1,
            33,
            _global_position_payload(-30.0, 168.0, 4.0),
            sequence=1,
        )
    )
    second_path.write_bytes(
        _mavlink2_tlog_record(
            first_timestamp,
            24,
            _gps_raw_payload(-29.1, 167.8, 2.5, fix_type=5, satellites=20),
        )
        + _mavlink2_tlog_record(
            second_timestamp,
            24,
            _gps_raw_payload(-29.2, 167.7, 2.0, fix_type=3, satellites=18),
            sequence=1,
        )
    )

    result = engine.parse_mavlink_tlog_files([first_path, second_path])

    assert len(result) == 2
    assert str(result.index.tz) == "UTC"
    assert result.iloc[0]["latitude"] == -29.1
    assert result.iloc[0]["fix_type"] == 5
    assert result.iloc[0]["satellites_visible"] == 20
    assert result.iloc[0]["source_message"] == "GPS_RAW_INT"
    assert result.attrs["duplicate_timestamp_rows"] == 1
    assert result.index[0].tz_convert("Australia/Sydney") == pd.Timestamp(
        "2026-03-26T11:44:03.866+11:00"
    )


def test_tlog_falls_back_to_global_position_when_raw_gps_is_absent(tmp_path):
    tlog_path = tmp_path / "fused-only.tlog"
    timestamp = pd.Timestamp("2026-03-26T20:10:24.768Z").timestamp()
    tlog_path.write_bytes(
        _mavlink2_tlog_record(
            timestamp,
            33,
            _global_position_payload(-29.06, 167.96, -1.5),
        )
    )

    result = engine.parse_mavlink_tlog_files(tlog_path)

    assert len(result) == 1
    assert result.iloc[0]["latitude"] == -29.06
    assert result.iloc[0]["longitude"] == 167.96
    assert result.iloc[0]["elevation"] == -1.5
    assert result.iloc[0]["source_message"] == "GLOBAL_POSITION_INT"


def test_tlog_batch_skips_a_file_without_positions(tmp_path):
    position_path = tmp_path / "positions.tlog"
    empty_path = tmp_path / "telemetry-only.tlog"
    timestamp = pd.Timestamp("2026-03-26T20:10:24.768Z").timestamp()
    position_path.write_bytes(
        _mavlink2_tlog_record(
            timestamp,
            24,
            _gps_raw_payload(-29.06, 167.96, 2.0),
        )
    )
    empty_path.write_bytes(
        _mavlink2_tlog_record(timestamp, 0, b"\x00" * 9)
    )

    result = engine.parse_mavlink_tlog_files([position_path, empty_path])

    assert len(result) == 1
    assert result.attrs["skipped_files"] == [str(empty_path)]


def test_interpolation_and_extrapolation_guard():
    index = pd.DatetimeIndex(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:10Z",
        ]
    )
    gps = pd.DataFrame(
        {
            "latitude": [0.0, 10.0],
            "longitude": [100.0, 110.0],
        },
        index=index,
    )

    midpoint = engine.interpolate_gps_position(
        gps,
        datetime(2026, 1, 1, 0, 0, 5, tzinfo=pytz.utc),
    )
    too_early = engine.interpolate_gps_position(
        gps,
        datetime(2025, 12, 31, 23, 59, 0, tzinfo=pytz.utc),
    )

    assert midpoint == (5.0, 105.0)
    assert too_early is None


def test_interpolation_refuses_to_bridge_separate_gps_sessions():
    gps = pd.DataFrame(
        {
            "latitude": [0.0, 1.0, 20.0, 21.0],
            "longitude": [100.0, 101.0, 120.0, 121.0],
        },
        index=pd.DatetimeIndex(
            [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:10Z",
                "2026-01-01T02:00:00Z",
                "2026-01-01T02:00:10Z",
            ]
        ),
    )

    in_session = engine.interpolate_gps_position(
        gps,
        datetime(2026, 1, 1, 0, 0, 5, tzinfo=pytz.utc),
    )
    in_gap = engine.interpolate_gps_position(
        gps,
        datetime(2026, 1, 1, 1, 0, 0, tzinfo=pytz.utc),
    )
    exact_second_session_start = engine.interpolate_gps_position(
        gps,
        datetime(2026, 1, 1, 2, 0, 0, tzinfo=pytz.utc),
    )

    assert in_session == (0.5, 100.5)
    assert in_gap is None
    assert exact_second_session_start == (20.0, 120.0)


def test_camera_offset_analysis_reports_an_ambiguous_full_coverage_range():
    gps_index = pd.date_range(
        "2026-01-01T00:00:00Z",
        "2026-01-01T01:00:00Z",
        freq="30s",
    )
    gps = pd.DataFrame(
        {
            "latitude": np.linspace(0.0, 1.0, len(gps_index)),
            "longitude": np.linspace(100.0, 101.0, len(gps_index)),
        },
        index=gps_index,
    )
    camera_times = [
        datetime(2026, 1, 1, 13, 10, 0),
        datetime(2026, 1, 1, 13, 30, 0),
        datetime(2026, 1, 1, 13, 50, 0),
    ]

    analysis = engine.analyze_camera_time_offset(
        camera_times,
        gps,
        timezone_str="Australia/Sydney",
    )

    assert analysis["suggested_offset_seconds"] == -7200
    assert analysis["matched_count"] == 3
    assert analysis["coverage_fraction"] == 1.0
    assert analysis["best_offset_min_seconds"] == -7800
    assert analysis["best_offset_max_seconds"] == -6600
    assert analysis["ambiguity_seconds"] == 1200
    assert analysis["evidence"].startswith("Ambiguous")


def test_camera_offset_coverage_can_evaluate_a_continuity_offset():
    gps_index = pd.date_range(
        "2026-01-01T00:00:00Z",
        "2026-01-01T01:05:00Z",
        freq="1min",
    )
    gps = pd.DataFrame(
        {
            "latitude": np.zeros(len(gps_index)),
            "longitude": np.ones(len(gps_index)),
        },
        index=gps_index,
    )

    matched, total = engine.count_camera_time_offset_coverage(
        [
            datetime(2026, 1, 1, 2, 0),
            datetime(2026, 1, 1, 2, 10),
        ],
        gps,
        -3600,
        timezone_str="UTC",
    )

    assert (matched, total) == (1, 2)


def test_combined_camera_run_resolves_short_folder_offset_ambiguity():
    first_gps = pd.date_range(
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:40:00Z",
        freq="1min",
    )
    second_gps = pd.date_range(
        "2026-01-01T00:45:00Z",
        "2026-01-01T02:00:00Z",
        freq="1min",
    )
    gps_index = first_gps.append(second_gps)
    gps = pd.DataFrame(
        {
            "latitude": np.zeros(len(gps_index)),
            "longitude": np.ones(len(gps_index)),
        },
        index=gps_index,
    )
    combined_camera_times = [
        datetime(2026, 1, 1, 2, 5),
        datetime(2026, 1, 1, 2, 20),
        datetime(2026, 1, 1, 2, 25),
        datetime(2026, 1, 1, 2, 35),
        datetime(2026, 1, 1, 2, 45),
        datetime(2026, 1, 1, 4, 0),
    ]

    analysis = engine.analyze_camera_time_offset(
        combined_camera_times,
        gps,
        timezone_str="UTC",
    )

    assert analysis["suggested_offset_seconds"] == -7200
    assert analysis["matched_count"] == len(combined_camera_times)


def test_scanner_ignores_generated_output_directories(tmp_path):
    source_dir = tmp_path / "survey"
    generated_dir = source_dir / "transects_output" / "T1"
    source_dir.mkdir()
    generated_dir.mkdir(parents=True)
    _write_timestamped_jpeg(source_dir / "source.jpg")
    _write_timestamped_jpeg(generated_dir / "copy.jpg")

    found = engine.get_image_files_and_times(source_dir)

    assert [Path(item["file_path"]).name for item in found] == ["source.jpg"]


def test_scanner_keeps_flattened_identifiers_unique(tmp_path):
    source_dir = tmp_path / "survey"
    first_dir = source_dir / "a_b"
    second_dir = source_dir / "a"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    _write_timestamped_jpeg(first_dir / "c.jpg")
    _write_timestamped_jpeg(second_dir / "b_c.jpg")

    found = engine.get_image_files_and_times(source_dir)
    identifiers = [item["identifier"] for item in found]

    assert len(identifiers) == 2
    assert len(set(identifiers)) == 2
    assert any("__" in identifier for identifier in identifiers)


def test_scanner_assigns_top_level_camera_folders_as_time_groups(tmp_path):
    source_dir = tmp_path / "survey"
    camera_dir = source_dir / "Camera_A" / "day_1"
    camera_dir.mkdir(parents=True)
    _write_timestamped_jpeg(source_dir / "root.jpg")
    _write_timestamped_jpeg(camera_dir / "nested.jpg")

    found = engine.get_image_files_and_times(source_dir)
    groups_by_name = {
        Path(item["file_path"]).name: item["time_group"] for item in found
    }

    assert groups_by_name == {"root.jpg": ".", "nested.jpg": "Camera_A"}


def test_jpeg_geotagging_omits_unknown_altitude_and_is_atomic(tmp_path):
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    _write_timestamped_jpeg(source)
    original_source_bytes = source.read_bytes()

    assert engine.set_gps_location(
        source,
        -33.0,
        151.0,
        None,
        output,
        gps_time_utc=datetime(2026, 1, 15, 0, 0, tzinfo=pytz.utc),
    )

    gps = piexif.load(str(output))["GPS"]
    assert piexif.GPSIFD.GPSAltitude not in gps
    assert piexif.GPSIFD.GPSAltitudeRef not in gps
    assert piexif.GPSIFD.GPSDateStamp in gps
    assert source.read_bytes() == original_source_bytes
    assert not list(tmp_path.glob("*.tmp"))


def test_negative_altitude_uses_below_sea_level_reference(tmp_path):
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    _write_timestamped_jpeg(source)

    assert engine.set_gps_location(source, -33.0, 151.0, -12.5, output)

    gps = piexif.load(str(output))["GPS"]
    assert gps[piexif.GPSIFD.GPSAltitudeRef] == 1
    assert gps[piexif.GPSIFD.GPSAltitude] == (1250, 100)


def test_tiff_geotagging_fails_cleanly_without_creating_output(tmp_path):
    source = tmp_path / "source.tif"
    output = tmp_path / "output.tif"
    Image.new("RGB", (8, 8), "white").save(source, format="TIFF")

    assert not engine.set_gps_location(source, -33.0, 151.0, None, output)
    assert not output.exists()


def test_invalid_geotag_does_not_replace_an_existing_output(tmp_path):
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    _write_timestamped_jpeg(source)
    output.write_bytes(b"existing output")

    assert not engine.set_gps_location(source, 95.0, 151.0, None, output)
    assert output.read_bytes() == b"existing output"
