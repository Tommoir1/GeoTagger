"""
GeoTagger Engine - Core data processing functions
-------------------------------------------------
Handles parsing of GPS data (GPX, CSV), finding and reading image timestamps,
calculating time offsets, interpolating positions, and writing EXIF GPS data.
"""

import os
import shutil
import tempfile
import hashlib
import struct
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import pytz

from file_utils import is_generated_output_directory

# --- Optional Dependencies with User-Friendly Error Messages ---
try:
    import gpxpy
    import gpxpy.gpx
    GPXPY_AVAILABLE = True
except ImportError:
    print("WARNING: 'gpxpy' library not found. GPX file parsing will be disabled.")
    print("         Install using: pip install gpxpy")
    GPXPY_AVAILABLE = False

try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    print("WARNING: 'piexif' library not found. Image timestamp reading and geotagging will be disabled.")
    print("         Install using: pip install piexif")
    PIEXIF_AVAILABLE = False


# --- Core Functionality ---

def get_image_files_and_times(directory_path, cancel_check=None):
    """
    Recursively finds all supported image files in a directory and its subdirectories,
    and extracts their creation timestamps from EXIF data.

    Args:
        directory_path (str): The path to the parent directory to search.

    Returns:
        list: A list of dictionaries, where each dictionary represents an image
              and contains 'identifier', 'file_path', and 'image_time' (a naive datetime object).
              Returns an empty list if piexif is not available.
    """
    if not PIEXIF_AVAILABLE:
        print("ERROR: Cannot get image times because 'piexif' library is not installed.")
        return []
        
    image_list = []
    used_identifiers = set()
    supported_extensions = ('.jpg', '.jpeg', '.tif', '.tiff')
    
    print(f"Recursively scanning for images in: {directory_path}")
    
    # os.walk traverses the directory tree top-down
    for dirpath, dirnames, filenames in os.walk(directory_path):
        if cancel_check and cancel_check():
            print("Image scan cancelled.")
            break
        # Output folders live beneath the selected survey directory in several
        # workflows. Prune them so a subsequent scan cannot ingest generated
        # copies as if they were new source images.
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not is_generated_output_directory(dirname)
        ]
        dirnames.sort(key=str.casefold)
        filenames.sort(key=str.casefold)
        for filename in filenames:
            if cancel_check and cancel_check():
                print("Image scan cancelled.")
                return sorted(image_list, key=lambda x: x['image_time'])
            # macOS AppleDouble sidecar files can look like JPEGs but are not
            # source images and make EXIF scanning needlessly expensive.
            if (
                not filename.startswith('._')
                and filename.lower().endswith(supported_extensions)
            ):
                file_path = os.path.join(dirpath, filename)
                try:
                    exif_dict = piexif.load(file_path)
                    # The standard EXIF tag for original creation time (whole-second precision)
                    datetime_str = exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal].decode('utf-8')
                    image_time = datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')

                    # GoPro and many modern cameras additionally write SubSecTimeOriginal,
                    # which holds the fractional-seconds digits as a string (e.g. "750" -> 0.750 s).
                    # On a moving boat this matters: at 5 kt (~2.6 m/s) one second is ~2.6 m of position.
                    try:
                        subsec_bytes = exif_dict['Exif'].get(piexif.ExifIFD.SubSecTimeOriginal)
                        if subsec_bytes:
                            # Strip whitespace and any trailing NUL padding cameras occasionally write.
                            subsec_str = subsec_bytes.decode('utf-8', errors='ignore').strip().rstrip('\x00')
                            if subsec_str.isdigit():
                                # The string is the fractional part: "1" -> 0.1 s, "75" -> 0.75 s,
                                # "750" -> 0.750 s, "1234567" -> 0.1234567 s (truncate to micro).
                                microseconds = int(subsec_str.ljust(6, '0')[:6])
                                image_time = image_time.replace(microsecond=microseconds)
                    except (AttributeError, ValueError, UnicodeDecodeError):
                        # No usable subsec data — fall back to whole-second image_time.
                        pass

                    # Create a unique identifier from the path relative to the base search directory
                    relative_path = os.path.relpath(file_path, directory_path)
                    # Replace path separators for a clean, OS-agnostic ID
                    identifier = relative_path.replace(os.sep, '_')
                    if identifier in used_identifiers:
                        identifier_stem, identifier_extension = os.path.splitext(identifier)
                        path_hash = hashlib.sha256(
                            relative_path.replace(os.sep, '/').encode('utf-8')
                        ).hexdigest()[:10]
                        identifier = f"{identifier_stem}__{path_hash}{identifier_extension}"
                        counter = 2
                        while identifier in used_identifiers:
                            identifier = (
                                f"{identifier_stem}__{path_hash}_{counter}{identifier_extension}"
                            )
                            counter += 1
                    used_identifiers.add(identifier)

                    relative_parts = os.path.normpath(relative_path).split(os.sep)
                    time_group = relative_parts[0] if len(relative_parts) > 1 else "."
                    image_list.append({
                        'identifier': identifier,
                        'file_path': file_path,
                        'image_time': image_time,  # Stored as a naive datetime object
                        # A camera folder can have a different clock offset from another
                        # folder even when both are beneath the selected survey directory.
                        'time_group': time_group,
                    })
                except (KeyError, ValueError):
                    # This can happen if the image has no valid EXIF timestamp.
                    # We could fall back to file modification time, but EXIF is more reliable.
                    # print(f"Warning: Could not read EXIF timestamp for {file_path}. Skipping.")
                    pass
                except piexif.InvalidImageDataError:
                    # print(f"Warning: Invalid image data or not a valid JPEG in {file_path}. Skipping.")
                    pass
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    
    print(f"Found {len(image_list)} images with valid timestamps.")
    # Sort by time, which is helpful for chronological processing
    return sorted(image_list, key=lambda x: x['image_time'])


def parse_gpx_file(gpx_file_path):
    """
    Parses a GPX file and returns its track points as a pandas DataFrame.

    Args:
        gpx_file_path (str): The path to the GPX file.

    Returns:
        pd.DataFrame: A DataFrame with a DatetimeIndex (UTC) and columns for
                      'latitude', 'longitude', and 'elevation'. Returns None on error.
    """
    if not GPXPY_AVAILABLE:
        raise ImportError("Cannot parse GPX file because 'gpxpy' library is not installed.")

    points = []
    with open(gpx_file_path, 'r', encoding='utf-8-sig') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    points.append({
                        'time': point.time,
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'elevation': point.elevation
                    })

    if not points:
        return pd.DataFrame()

    df = pd.DataFrame(points)
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df['gps_time_identified'] = df['time'].notna()
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    valid_coordinate_mask = (
        df['latitude'].between(-90, 90, inclusive='both')
        & df['longitude'].between(-180, 180, inclusive='both')
    )
    invalid_coordinate_rows = int((~valid_coordinate_mask).sum())
    df = df.loc[valid_coordinate_mask].copy()
    df = df.set_index('time')
    df.sort_index(inplace=True)
    df.attrs['invalid_coordinate_rows'] = invalid_coordinate_rows
    return df


def _read_tlog_frame(log_file, file_path):
    """Read one timestamped MAVLink 1/2 frame from a standard telemetry log."""
    record_offset = log_file.tell()
    timestamp_bytes = log_file.read(8)
    if not timestamp_bytes:
        return None
    if len(timestamp_bytes) != 8:
        raise ValueError(
            f"Truncated TLOG timestamp in '{file_path}' at byte {record_offset}."
        )

    magic = log_file.read(1)
    if magic == b'\xfd':  # MAVLink 2
        header = log_file.read(9)
        if len(header) != 9:
            raise ValueError(
                f"Truncated MAVLink 2 header in '{file_path}' at byte {record_offset + 8}."
            )
        payload_length = header[0]
        incompatibility_flags = header[1]
        message_id = int.from_bytes(header[6:9], byteorder='little')
        trailer_length = 2 + (13 if incompatibility_flags & 0x01 else 0)
    elif magic == b'\xfe':  # MAVLink 1
        header = log_file.read(5)
        if len(header) != 5:
            raise ValueError(
                f"Truncated MAVLink 1 header in '{file_path}' at byte {record_offset + 8}."
            )
        payload_length = header[0]
        message_id = header[4]
        trailer_length = 2
    else:
        found = magic.hex() if magic else "end of file"
        raise ValueError(
            f"Unsupported or damaged TLOG record in '{file_path}' at byte "
            f"{record_offset}: expected a MAVLink frame, found {found}."
        )

    payload = log_file.read(payload_length)
    trailer = log_file.read(trailer_length)
    if len(payload) != payload_length or len(trailer) != trailer_length:
        raise ValueError(
            f"Truncated MAVLink frame in '{file_path}' at byte {record_offset}."
        )

    timestamp_usec = struct.unpack('>Q', timestamp_bytes)[0]
    return timestamp_usec, message_id, payload


def _valid_tlog_coordinate(latitude, longitude):
    return (
        np.isfinite(latitude)
        and np.isfinite(longitude)
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
        and not (latitude == 0 and longitude == 0)
    )


def _tlog_timestamp(timestamp_usec):
    try:
        timestamp = pd.Timestamp(timestamp_usec, unit='us', tz='UTC')
    except (OverflowError, ValueError, OSError):
        return None
    if timestamp.year < 1980 or timestamp.year > 2200:
        return None
    return timestamp


def _decode_tlog_gps_raw(timestamp_usec, payload, source_file):
    """Decode MAVLink GPS_RAW_INT (message 24), including truncated v2 payloads."""
    if len(payload) < 29:
        return None, "damaged"
    padded_payload = payload.ljust(30, b'\0')
    (
        _onboard_time_usec,
        latitude_raw,
        longitude_raw,
        altitude_raw,
        eph_raw,
        _epv_raw,
        _velocity_raw,
        _course_raw,
        fix_type,
        satellites_visible,
    ) = struct.unpack_from('<QiiiHHHHBB', padded_payload)

    if fix_type < 2:
        return None, "no_fix"

    latitude = latitude_raw / 1e7
    longitude = longitude_raw / 1e7
    timestamp = _tlog_timestamp(timestamp_usec)
    if timestamp is None:
        return None, "timestamp"
    if not _valid_tlog_coordinate(latitude, longitude):
        return None, "coordinate"

    elevation = altitude_raw / 1000
    if abs(elevation) > 100_000:
        elevation = np.nan
    hdop = eph_raw / 100 if eph_raw != 65535 else np.nan
    return {
        'time': timestamp,
        'latitude': latitude,
        'longitude': longitude,
        'elevation': elevation,
        'gps_time_identified': True,
        'raw_timestamp': timestamp.isoformat(),
        'fix_type': fix_type,
        'satellites_visible': satellites_visible,
        'hdop': hdop,
        'source_file': os.path.basename(source_file),
        'source_message': 'GPS_RAW_INT',
    }, None


def _decode_tlog_global_position(timestamp_usec, payload, source_file):
    """Decode MAVLink GLOBAL_POSITION_INT (message 33) as a fallback track."""
    if len(payload) < 12:
        return None, "damaged"
    padded_payload = payload.ljust(28, b'\0')
    (
        _time_boot_ms,
        latitude_raw,
        longitude_raw,
        altitude_raw,
        _relative_altitude_raw,
        _velocity_x,
        _velocity_y,
        _velocity_z,
        _heading,
    ) = struct.unpack_from('<IiiiihhhH', padded_payload)

    latitude = latitude_raw / 1e7
    longitude = longitude_raw / 1e7
    timestamp = _tlog_timestamp(timestamp_usec)
    if timestamp is None:
        return None, "timestamp"
    if not _valid_tlog_coordinate(latitude, longitude):
        return None, "coordinate"

    elevation = altitude_raw / 1000
    if abs(elevation) > 100_000:
        elevation = np.nan
    return {
        'time': timestamp,
        'latitude': latitude,
        'longitude': longitude,
        'elevation': elevation,
        'gps_time_identified': True,
        'raw_timestamp': timestamp.isoformat(),
        'fix_type': np.nan,
        'satellites_visible': np.nan,
        'hdop': np.nan,
        'source_file': os.path.basename(source_file),
        'source_message': 'GLOBAL_POSITION_INT',
    }, None


def parse_mavlink_tlog_files(tlog_file_paths, cancel_check=None):
    """
    Parse one or more timestamped MAVLink TLOG files into a UTC GPS track.

    The timestamp stored before each MAVLink packet is the logging computer's
    epoch timestamp. GPS_RAW_INT fixes are preferred because they include GPS
    fix quality and satellite count. GLOBAL_POSITION_INT is used only for files
    that contain no valid raw GPS fixes.
    """
    if isinstance(tlog_file_paths, (str, bytes, os.PathLike)):
        tlog_file_paths = [tlog_file_paths]
    paths = [os.fspath(path) for path in tlog_file_paths]
    if not paths:
        raise ValueError("Select at least one TLOG file.")

    all_points = []
    invalid_coordinate_rows = 0
    invalid_timestamp_rows = 0
    ignored_no_fix_rows = 0
    damaged_position_rows = 0
    records_scanned = 0
    source_message_types = {}
    skipped_files = []

    for file_path in paths:
        if cancel_check and cancel_check():
            raise InterruptedError("TLOG import cancelled.")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"TLOG file not found: {file_path}")

        raw_gps_points = []
        global_position_frames = []
        with open(file_path, 'rb') as log_file:
            while True:
                if records_scanned % 4096 == 0 and cancel_check and cancel_check():
                    raise InterruptedError("TLOG import cancelled.")
                frame = _read_tlog_frame(log_file, file_path)
                if frame is None:
                    break
                records_scanned += 1
                timestamp_usec, message_id, payload = frame

                point = None
                rejection_reason = None
                if message_id == 24:
                    point, rejection_reason = _decode_tlog_gps_raw(
                        timestamp_usec, payload, file_path
                    )
                    if point:
                        if not raw_gps_points:
                            global_position_frames.clear()
                        raw_gps_points.append(point)
                elif message_id == 33 and not raw_gps_points:
                    # Defer decoding the much higher-rate fused position stream.
                    # Most BlueBoat logs contain GPS_RAW_INT, so this avoids
                    # constructing tens of thousands of unused point objects.
                    global_position_frames.append((timestamp_usec, payload))

                if rejection_reason == "coordinate":
                    invalid_coordinate_rows += 1
                elif rejection_reason == "timestamp":
                    invalid_timestamp_rows += 1
                elif rejection_reason == "no_fix":
                    ignored_no_fix_rows += 1
                elif rejection_reason == "damaged":
                    damaged_position_rows += 1

        selected_points = raw_gps_points
        if not selected_points:
            global_position_points = []
            for timestamp_usec, payload in global_position_frames:
                point, rejection_reason = _decode_tlog_global_position(
                    timestamp_usec, payload, file_path
                )
                if point:
                    global_position_points.append(point)
                elif rejection_reason == "coordinate":
                    invalid_coordinate_rows += 1
                elif rejection_reason == "timestamp":
                    invalid_timestamp_rows += 1
                elif rejection_reason == "damaged":
                    damaged_position_rows += 1
            selected_points = global_position_points
        if not selected_points:
            skipped_files.append(file_path)
            continue
        source_message_types[os.path.basename(file_path)] = selected_points[0]['source_message']
        all_points.extend(selected_points)

    if not all_points:
        selected_names = ", ".join(os.path.basename(path) for path in paths)
        raise ValueError(
            f"No valid GPS positions were found in the selected TLOG file(s): {selected_names}."
        )

    points_df = pd.DataFrame(all_points)
    points_df.sort_values(
        ['time', 'fix_type', 'satellites_visible'],
        ascending=[True, False, False],
        na_position='last',
        inplace=True,
    )
    duplicate_timestamp_rows = int(points_df['time'].duplicated(keep='first').sum())
    points_df.drop_duplicates(subset=['time'], keep='first', inplace=True)
    points_df.set_index('time', inplace=True)
    points_df.sort_index(inplace=True)
    points_df.attrs['invalid_coordinate_rows'] = invalid_coordinate_rows
    points_df.attrs['invalid_timestamp_rows'] = invalid_timestamp_rows
    points_df.attrs['ignored_no_fix_rows'] = ignored_no_fix_rows
    points_df.attrs['damaged_position_rows'] = damaged_position_rows
    points_df.attrs['duplicate_timestamp_rows'] = duplicate_timestamp_rows
    points_df.attrs['tlog_records_scanned'] = records_scanned
    points_df.attrs['source_files'] = paths
    points_df.attrs['skipped_files'] = skipped_files
    points_df.attrs['source_message_types'] = source_message_types
    return points_df


def parse_boat_log_csv(csv_file_path, lat_col, lon_col, boat_timezone_str,
                         date_col=None, time_col=None, datetime_col=None,
                         alt_col=None, datetime_format=None, delimiter=','):
    """
    Parses a generic CSV log file into a standardized pandas DataFrame.

    Args:
        csv_file_path (str): Path to the CSV file.
        lat_col (str): Column name for latitude.
        lon_col (str): Column name for longitude.
        boat_timezone_str (str): The timezone of the naive timestamps in the CSV (e.g., 'America/New_York').
        date_col (str, optional): Column name for date. Used with time_col.
        time_col (str, optional): Column name for time. Used with date_col.
        datetime_col (str, optional): Column name for a combined timestamp.
        alt_col (str, optional): Column name for altitude/elevation.
        datetime_format (str): The strptime format string for parsing the timestamp.
        delimiter (str): The delimiter for the CSV file.

    Returns:
        pd.DataFrame: A DataFrame with a DatetimeIndex (UTC) and columns for
                      'latitude', 'longitude', and 'elevation'.
    """
    df = pd.read_csv(csv_file_path, delimiter=delimiter, skipinitialspace=True)

    # --- Timestamp processing ---
    if datetime_col:
        # Use a single timestamp column
        timestamp_series = df[datetime_col].astype(str)
    elif date_col and time_col:
        # Combine separate date and time columns
        timestamp_series = df[date_col].astype(str) + ' ' + df[time_col].astype(str)
    else:
        raise ValueError("Either datetime_col or both date_col and time_col must be provided.")

    # Convert to datetime objects, coercing errors to NaT (Not a Time).
    # Rows with unparseable timestamps are kept so their coordinates can still
    # be shown on the map; time-based interpolation ignores NaT-indexed rows.
    parsed_timestamps = pd.to_datetime(timestamp_series, format=datetime_format, errors='coerce')

    # Localize the naive timestamps to the specified timezone, then convert to UTC
    tz = pytz.timezone(boat_timezone_str)
    valid_timestamp_mask = parsed_timestamps.notna()
    timestamps_utc = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns, UTC]')
    if valid_timestamp_mask.any():
        parsed_timezone = getattr(parsed_timestamps.dt, 'tz', None)
        if parsed_timezone is not None:
            localized_timestamps = parsed_timestamps.loc[valid_timestamp_mask].dt.tz_convert('UTC')
        else:
            # An isolated ambiguous/nonexistent DST wall time cannot be inferred
            # safely. Keep its coordinates and mark its time as unidentified,
            # matching the existing handling for other unparseable timestamps.
            localized_timestamps = (
                parsed_timestamps.loc[valid_timestamp_mask]
                .dt.tz_localize(tz, ambiguous='NaT', nonexistent='NaT')
                .dt.tz_convert('UTC')
            )
        timestamps_utc.loc[localized_timestamps.index] = localized_timestamps

    # --- GPS data processing ---
    df_out = df[[lat_col, lon_col]].copy()
    df_out.rename(columns={lat_col: 'latitude', lon_col: 'longitude'}, inplace=True)
    df_out['gps_time_identified'] = timestamps_utc.notna()
    df_out['raw_timestamp'] = timestamp_series
    df_out.index = pd.DatetimeIndex(timestamps_utc, name='time')
    
    if alt_col and alt_col in df.columns:
        df_out['elevation'] = pd.to_numeric(df[alt_col], errors='coerce')
    else:
        df_out['elevation'] = np.nan

    df_out['latitude'] = pd.to_numeric(df_out['latitude'], errors='coerce')
    df_out['longitude'] = pd.to_numeric(df_out['longitude'], errors='coerce')

    valid_coordinate_mask = (
        df_out['latitude'].between(-90, 90, inclusive='both')
        & df_out['longitude'].between(-180, 180, inclusive='both')
    )
    invalid_coordinate_rows = int((~valid_coordinate_mask).sum())
    unidentified_timestamp_rows = int((~df_out['gps_time_identified']).sum())
    df_out = df_out.loc[valid_coordinate_mask].copy()

    # Sort by time
    df_out.sort_index(inplace=True)
    df_out.attrs['invalid_coordinate_rows'] = invalid_coordinate_rows
    df_out.attrs['unidentified_timestamp_rows'] = unidentified_timestamp_rows
    
    return df_out


def calculate_time_offset(method, **kwargs):
    """
    Calculates the time offset needed to sync media time with GPS time (UTC).

    Args:
        method (str): The calculation method, either 'manual' or 'sync_point'.
        **kwargs:
            For 'manual': `offset_seconds` (float).
            For 'sync_point': `gopro_sync_time` (naive datetime),
                              `gps_sync_time` (UTC datetime),
                              `boat_timezone_str` (str).

    Returns:
        timedelta: The offset to be added to the media time.
    """
    if method == 'manual':
        offset_seconds = kwargs.get('offset_seconds', 0.0)
        return timedelta(seconds=offset_seconds)

    elif method == 'sync_point':
        media_time_naive = kwargs.get('gopro_sync_time')
        gps_time_utc = kwargs.get('gps_sync_time')
        tz_str = kwargs.get('boat_timezone_str')

        if not all([media_time_naive, gps_time_utc, tz_str]):
            raise ValueError("Missing parameters for 'sync_point' offset calculation.")

        # Localize the naive media time to its original timezone
        media_tz = pytz.timezone(tz_str)
        media_time_aware = media_tz.localize(media_time_naive)

        # The offset is the difference between the true UTC time and the media's time
        offset = gps_time_utc - media_time_aware
        return offset
    else:
        raise ValueError(f"Unknown time offset calculation method: {method}")


def get_gps_time_intervals(gps_df, max_gap_seconds=30):
    """
    Return contiguous UTC GPS time intervals, split wherever logging stopped.

    The intervals are used both by interpolation and by the camera-offset analyser.
    Treating a multi-hour logging gap as a valid straight track would create plausible
    looking, but entirely fictional, coordinates between two survey sessions.
    """
    if (
        gps_df is None
        or gps_df.empty
        or not isinstance(gps_df.index, pd.DatetimeIndex)
    ):
        return []

    timestamps = gps_df.index[~gps_df.index.isna()].sort_values().unique()
    if len(timestamps) == 0:
        return []

    if timestamps.tz is None:
        timestamps = timestamps.tz_localize("UTC")
    else:
        timestamps = timestamps.tz_convert("UTC")

    gap_ns = float(max_gap_seconds) * 1_000_000_000
    timestamp_ns = timestamps.asi8
    split_after = np.flatnonzero(np.diff(timestamp_ns) > gap_ns)
    starts = np.r_[0, split_after + 1]
    ends = np.r_[split_after, len(timestamps) - 1]
    return [(timestamps[start], timestamps[end]) for start, end in zip(starts, ends)]


def _camera_times_to_utc_nanoseconds(image_times, timezone_str):
    try:
        local_timezone = pytz.timezone(timezone_str)
    except pytz.UnknownTimeZoneError as exc:
        raise ValueError(f"Unknown camera timezone: {timezone_str}") from exc

    image_time_ns = []
    for raw_time in image_times:
        timestamp = pd.Timestamp(raw_time)
        if pd.isna(timestamp):
            continue
        try:
            if timestamp.tzinfo is None:
                aware = local_timezone.localize(
                    timestamp.to_pydatetime(),
                    is_dst=None,
                )
                timestamp = pd.Timestamp(aware)
            else:
                timestamp = timestamp.tz_convert(local_timezone)
        except (pytz.AmbiguousTimeError, pytz.NonExistentTimeError):
            continue
        image_time_ns.append(timestamp.tz_convert("UTC").value)
    return np.asarray(image_time_ns, dtype=np.int64)


def _count_offset_coverage(
    images_ns,
    interval_starts_ns,
    interval_ends_ns,
    offset_seconds,
):
    if len(images_ns) == 0 or len(interval_starts_ns) == 0:
        return 0
    shifted_ns = images_ns + int(round(offset_seconds)) * 1_000_000_000
    interval_indexes = np.searchsorted(
        interval_starts_ns,
        shifted_ns,
        side="right",
    ) - 1
    valid_indexes = interval_indexes >= 0
    if not np.any(valid_indexes):
        return 0
    valid_targets = shifted_ns[valid_indexes]
    valid_interval_indexes = interval_indexes[valid_indexes]
    return int(
        np.count_nonzero(
            valid_targets <= interval_ends_ns[valid_interval_indexes]
        )
    )


def count_camera_time_offset_coverage(
    image_times,
    gps_df,
    offset_seconds,
    timezone_str="Australia/Sydney",
    max_gap_seconds=30,
):
    """Count image timestamps covered by real GPS intervals at one offset."""
    images_ns = _camera_times_to_utc_nanoseconds(image_times, timezone_str)
    intervals = get_gps_time_intervals(
        gps_df,
        max_gap_seconds=max_gap_seconds,
    )
    interval_starts_ns = np.asarray(
        [start.value for start, _ in intervals],
        dtype=np.int64,
    )
    interval_ends_ns = np.asarray(
        [end.value for _, end in intervals],
        dtype=np.int64,
    )
    matched_count = _count_offset_coverage(
        images_ns,
        interval_starts_ns,
        interval_ends_ns,
        offset_seconds,
    )
    return matched_count, int(len(images_ns))


def analyze_camera_time_offset(
    image_times,
    gps_df,
    timezone_str="Australia/Sydney",
    search_hours=4,
    max_gap_seconds=30,
):
    """
    Find camera-clock offsets that place the most images inside real GPS sessions.

    This is deliberately a constraint analyser, not a claim that a timestamp-only
    match is exact. If the camera starts after GPS and stops before GPS, many nearby
    offsets can produce the same coverage. The returned best range and evidence label
    make that ambiguity visible so a known image/GPS pair can be used to confirm it.
    """
    images_ns = _camera_times_to_utc_nanoseconds(
        image_times,
        timezone_str,
    )

    intervals = get_gps_time_intervals(gps_df, max_gap_seconds=max_gap_seconds)
    result = {
        "suggested_offset_seconds": None,
        "matched_count": 0,
        "image_count": len(images_ns),
        "coverage_fraction": 0.0,
        "best_offset_min_seconds": None,
        "best_offset_max_seconds": None,
        "ambiguity_seconds": None,
        "evidence": "No usable timestamps",
        "confirmation_required": True,
    }
    if len(images_ns) == 0 or not intervals:
        return result

    interval_starts_ns = np.asarray([start.value for start, _ in intervals], dtype=np.int64)
    interval_ends_ns = np.asarray([end.value for _, end in intervals], dtype=np.int64)

    def coverage_count(offset_seconds):
        return _count_offset_coverage(
            images_ns,
            interval_starts_ns,
            interval_ends_ns,
            offset_seconds,
        )

    search_limit_seconds = max(1, int(round(float(search_hours) * 3600)))
    coarse_offsets = np.arange(
        -search_limit_seconds,
        search_limit_seconds + 1,
        60,
        dtype=int,
    )
    coarse_counts = np.asarray(
        [coverage_count(offset) for offset in coarse_offsets],
        dtype=int,
    )

    # Refine several distinct high-scoring regions. This avoids missing a narrow
    # one-second optimum that lies between the one-minute coarse samples.
    ranked_indexes = sorted(
        range(len(coarse_offsets)),
        key=lambda index: (
            -coarse_counts[index],
            abs(coarse_offsets[index] - round(coarse_offsets[index] / 3600) * 3600),
            abs(coarse_offsets[index]),
        ),
    )
    candidate_centres = []
    for index in ranked_indexes:
        offset = int(coarse_offsets[index])
        if all(abs(offset - existing) > 240 for existing in candidate_centres):
            candidate_centres.append(offset)
        if len(candidate_centres) == 5:
            break

    refined_counts = {}
    for centre in candidate_centres:
        for offset in range(
            max(-search_limit_seconds, centre - 90),
            min(search_limit_seconds, centre + 90) + 1,
        ):
            if offset not in refined_counts:
                refined_counts[offset] = coverage_count(offset)

    best_count = max(refined_counts.values(), default=0)
    if best_count == 0:
        result["evidence"] = "No overlap in search range"
        return result

    best_offsets = [
        offset for offset, count in refined_counts.items() if count == best_count
    ]
    suggested_offset = min(
        best_offsets,
        key=lambda offset: (
            abs(offset - round(offset / 3600) * 3600),
            abs(offset),
        ),
    )

    # Expand from the selected optimum to reveal the full equal-score plateau.
    plateau_min = suggested_offset
    while (
        plateau_min > -search_limit_seconds
        and coverage_count(plateau_min - 1) == best_count
    ):
        plateau_min -= 1
    plateau_max = suggested_offset
    while (
        plateau_max < search_limit_seconds
        and coverage_count(plateau_max + 1) == best_count
    ):
        plateau_max += 1

    nearest_hour = int(round(suggested_offset / 3600) * 3600)
    if (
        plateau_min <= nearest_hour <= plateau_max
        and coverage_count(nearest_hour) == best_count
    ):
        suggested_offset = nearest_hour

    coverage_fraction = best_count / len(images_ns)
    ambiguity_seconds = plateau_max - plateau_min
    if ambiguity_seconds > 120:
        evidence = "Ambiguous timing range — verify visually"
    elif coverage_fraction >= 0.95 and ambiguity_seconds <= 5:
        evidence = "Strong timing constraint — confirm visually"
    elif coverage_fraction >= 0.80 and ambiguity_seconds <= 60:
        evidence = "Moderate timing constraint — confirm visually"
    else:
        evidence = "Weak timing constraint — verify visually"

    result.update(
        {
            "suggested_offset_seconds": int(suggested_offset),
            "matched_count": int(best_count),
            "coverage_fraction": float(coverage_fraction),
            "best_offset_min_seconds": int(plateau_min),
            "best_offset_max_seconds": int(plateau_max),
            "ambiguity_seconds": int(ambiguity_seconds),
            "evidence": evidence,
        }
    )
    return result


def interpolate_gps_position(
    gps_df,
    target_time_utc,
    max_extrapolation_seconds=30,
    max_interpolation_gap_seconds=30,
):
    """
    Interpolates the GPS position (lat, lon) for a specific time.

    Args:
        gps_df (pd.DataFrame): DataFrame of GPS points with a DatetimeIndex (UTC).
        target_time_utc (datetime): The UTC time for which to find a position.
        max_extrapolation_seconds (float): If the target time falls outside the GPS
            track range by more than this many seconds, returns None instead of snapping
            to the nearest endpoint. This prevents photos taken before/after the GPS
            recording window from being silently mis-tagged at the start or end position
            of the track. Default 30 s allows for small clock-sync slop at the boundaries.
            Pass float('inf') to restore the original "always snap to nearest end" behaviour.
        max_interpolation_gap_seconds (float): Refuse to interpolate between adjacent
            GPS points farther apart than this. This prevents straight-line positions
            being invented across separate logging sessions. Default 30 s.

    Returns:
        tuple: A tuple of (latitude, longitude), or None if interpolation is not possible
               (empty track) or the target time is too far outside the track range.
    """
    if not isinstance(target_time_utc, datetime) or target_time_utc.tzinfo is None:
        raise ValueError("target_time_utc must be a timezone-aware datetime object.")

    if gps_df.empty:
        return None

    if isinstance(gps_df.index, pd.DatetimeIndex) and gps_df.index.hasnans:
        gps_df = gps_df[~gps_df.index.isna()].copy()
        if gps_df.empty:
            return None

    track_min = gps_df.index.min()
    track_max = gps_df.index.max()

    # Edge case: target at or before the start of the track
    if target_time_utc <= track_min:
        gap_seconds = (track_min - target_time_utc).total_seconds()
        if gap_seconds > max_extrapolation_seconds:
            return None
        return (gps_df.iloc[0]['latitude'], gps_df.iloc[0]['longitude'])

    # Edge case: target at or after the end of the track
    if target_time_utc >= track_max:
        gap_seconds = (target_time_utc - track_max).total_seconds()
        if gap_seconds > max_extrapolation_seconds:
            return None
        return (gps_df.iloc[-1]['latitude'], gps_df.iloc[-1]['longitude'])

    # Find the indices of the points just before and after the target time
    idx = gps_df.index.searchsorted(target_time_utc)
    if idx < len(gps_df) and gps_df.index[idx] == target_time_utc:
        exact_point = gps_df.iloc[idx]
        return (exact_point['latitude'], exact_point['longitude'])

    # Get the two points that bracket our target time
    p1 = gps_df.iloc[idx - 1]
    p2 = gps_df.iloc[idx]

    # Calculate the time proportion between the two points
    time_diff_total = (p2.name - p1.name).total_seconds()
    if time_diff_total == 0:
        return (p1['latitude'], p1['longitude']) # Points have same timestamp
    if time_diff_total > max_interpolation_gap_seconds:
        return None

    time_diff_target = (target_time_utc - p1.name).total_seconds()
    proportion = time_diff_target / time_diff_total

    # Linear interpolation for latitude and longitude
    lat = p1['latitude'] + proportion * (p2['latitude'] - p1['latitude'])
    lon = p1['longitude'] + proportion * (p2['longitude'] - p1['longitude'])

    return (lat, lon)


# --- EXIF Writing Functionality ---

def _decimal_to_dms(degrees_decimal):
    """Converts decimal degrees to the DMS format required by EXIF."""
    degrees_decimal = abs(degrees_decimal)
    degrees = int(degrees_decimal)
    minutes_decimal = (degrees_decimal - degrees) * 60
    minutes = int(minutes_decimal)
    seconds_decimal = (minutes_decimal - minutes) * 60
    # Store as rational numbers (numerator, denominator)
    return (
        (degrees, 1),
        (minutes, 1),
        (int(seconds_decimal * 10000), 10000)
    )

def set_gps_location(image_path, lat, lon, alt, output_path, gps_time_utc=None):
    """
    Writes GPS coordinates to the EXIF data of an image file.

    Args:
        image_path (str): Path to the source image.
        lat (float): Latitude in decimal degrees.
        lon (float): Longitude in decimal degrees.
        alt (float, optional): Altitude in meters. When None/NaN, altitude tags
            are omitted rather than writing a false zero.
        output_path (str): Path to save the new image with GPS data.
        gps_time_utc (datetime, optional): The corrected UTC timestamp of when the photo
            was taken (i.e. the time used for GPS interpolation). Written into the EXIF
            GPSDateStamp and GPSTimeStamp fields. Must be a timezone-aware datetime; naive
            datetimes are assumed to already be UTC. If None, the GPS timestamp tags are
            NOT written -- this is intentional, as writing the current wall-clock time
            (the previous behaviour) silently corrupts the EXIF for any downstream tool
            that builds a track from the geotagged JPEGs.

    Returns:
        bool: True on success, False on failure.
    """
    if not PIEXIF_AVAILABLE:
        print("ERROR: Cannot set GPS location because 'piexif' is not installed.")
        return False

    image_path = os.fspath(image_path)
    output_path = os.fspath(output_path)
    source_extension = os.path.splitext(image_path)[1].lower()
    if source_extension not in {'.jpg', '.jpeg'}:
        print(
            f"ERROR: GPS EXIF writing is supported for JPEG files only; "
            f"'{source_extension or '<no extension>'}' cannot be written safely."
        )
        return False

    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        print("ERROR: Latitude and longitude must be numeric.")
        return False

    if not np.isfinite(lat) or not np.isfinite(lon) or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        print(f"ERROR: Invalid GPS coordinates: latitude={lat}, longitude={lon}")
        return False

    if os.path.abspath(image_path) == os.path.abspath(output_path):
        print("ERROR: Source and output paths must be different; source images are never modified.")
        return False

    temp_output_path = None
    try:
        output_directory = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_directory, exist_ok=True)
        file_descriptor, temp_output_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(output_path)}.",
            suffix=".tmp",
            dir=output_directory,
        )
        os.close(file_descriptor)

        # Work on a temporary copy, then atomically replace the destination only
        # after EXIF serialization succeeds.
        shutil.copy2(image_path, temp_output_path)

        # Load EXIF data from the new file
        exif_dict = piexif.load(temp_output_path)
        
        # Determine reference (N/S/E/W)
        lat_ref = b'N' if lat >= 0 else b'S'
        lon_ref = b'E' if lon >= 0 else b'W'

        # Convert to DMS format
        dms_lat = _decimal_to_dms(lat)
        dms_lon = _decimal_to_dms(lon)
        
        # Create the GPS IFD (Image File Directory) dictionary
        gps_ifd = {
            piexif.GPSIFD.GPSLatitudeRef: lat_ref,
            piexif.GPSIFD.GPSLatitude: dms_lat,
            piexif.GPSIFD.GPSLongitudeRef: lon_ref,
            piexif.GPSIFD.GPSLongitude: dms_lon,
        }

        if alt is not None:
            try:
                altitude = float(alt)
            except (TypeError, ValueError):
                altitude = np.nan
            if np.isfinite(altitude):
                gps_ifd[piexif.GPSIFD.GPSAltitudeRef] = 0 if altitude >= 0 else 1
                gps_ifd[piexif.GPSIFD.GPSAltitude] = (int(round(abs(altitude) * 100)), 100)

        # Write the time the photo was actually taken (in UTC), NOT the current wall-clock.
        # If no timestamp is supplied, skip these tags entirely rather than write a wrong one.
        if gps_time_utc is not None:
            if gps_time_utc.tzinfo is None:
                # Defensively assume UTC if a naive datetime slipped through.
                gps_time_utc = gps_time_utc.replace(tzinfo=timezone.utc)
            else:
                # Normalise to UTC if the caller passed a different aware tz.
                gps_time_utc = gps_time_utc.astimezone(timezone.utc)

            # GPSTimeStamp: hours, minutes, seconds as rationals. Sub-second precision
            # is encoded by using a non-1 denominator on the seconds field.
            sec_numerator = gps_time_utc.second * 1_000_000 + gps_time_utc.microsecond
            sec_denominator = 1_000_000

            gps_ifd[piexif.GPSIFD.GPSTimeStamp] = (
                (gps_time_utc.hour, 1),
                (gps_time_utc.minute, 1),
                (sec_numerator, sec_denominator),
            )
            gps_ifd[piexif.GPSIFD.GPSDateStamp] = gps_time_utc.strftime('%Y:%m:%d')

        # Add the GPS data to the main EXIF dictionary
        exif_dict['GPS'] = gps_ifd
        
        # Dump the dictionary to bytes and insert it into the image file
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, temp_output_path)
        os.replace(temp_output_path, output_path)
        temp_output_path = None
        
        return True

    except Exception as e:
        print(f"Error setting GPS for {os.path.basename(image_path)}: {e}")
        return False
    finally:
        if temp_output_path and os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
            except OSError:
                pass


if __name__ == '__main__':
    # This block allows for testing the engine functions directly
    print("GeoTagger Engine script. This file is intended to be imported, not run directly.")
    print("You can add test code here to validate functions.")

    # Example: Test recursive image search
    # test_image_dir = "path/to/your/test/image_folder"
    # if os.path.exists(test_image_dir):
    #     images = get_image_files_and_times(test_image_dir)
    #     print(f"\n--- Testing get_image_files_and_times ---")
    #     if images:
    #         print(f"Found {len(images)} images.")
    #         print("First image found:")
    #         print(images[0])
    #     else:
    #         print("No images found in test directory.")
