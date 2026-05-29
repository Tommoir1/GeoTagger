"""
GeoTagger Engine - Core data processing functions
-------------------------------------------------
Handles parsing of GPS data (GPX, CSV), finding and reading image timestamps,
calculating time offsets, interpolating positions, and writing EXIF GPS data.
"""

import os
import shutil
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import pytz

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

def get_image_files_and_times(directory_path):
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
    supported_extensions = ('.jpg', '.jpeg', '.tif', '.tiff')
    
    print(f"Recursively scanning for images in: {directory_path}")
    
    # os.walk traverses the directory tree top-down
    for dirpath, _, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.lower().endswith(supported_extensions):
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

                    image_list.append({
                        'identifier': identifier,
                        'file_path': file_path,
                        'image_time': image_time  # Stored as a naive datetime object
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
    with open(gpx_file_path, 'r') as gpx_file:
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
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time')
    return df


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

    # Convert to datetime objects, coercing errors to NaT (Not a Time)
    timestamps_naive = pd.to_datetime(timestamp_series, format=datetime_format, errors='coerce')

    # Drop rows where timestamp parsing failed
    df = df[timestamps_naive.notna()]
    timestamps_naive = timestamps_naive[timestamps_naive.notna()]

    # Localize the naive timestamps to the specified timezone, then convert to UTC
    tz = pytz.timezone(boat_timezone_str)
    timestamps_utc = timestamps_naive.dt.tz_localize(tz, ambiguous='infer').dt.tz_convert('UTC')

    df.index = timestamps_utc

    # --- GPS data processing ---
    df_out = df[[lat_col, lon_col]].copy()
    df_out.rename(columns={lat_col: 'latitude', lon_col: 'longitude'}, inplace=True)
    
    if alt_col and alt_col in df.columns:
        df_out['elevation'] = pd.to_numeric(df[alt_col], errors='coerce')
    else:
        df_out['elevation'] = np.nan

    df_out['latitude'] = pd.to_numeric(df_out['latitude'], errors='coerce')
    df_out['longitude'] = pd.to_numeric(df_out['longitude'], errors='coerce')

    # Drop any rows where essential data is missing after conversion
    df_out.dropna(subset=['latitude', 'longitude'], inplace=True)

    # Sort by time
    df_out.sort_index(inplace=True)
    
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


def interpolate_gps_position(gps_df, target_time_utc, max_extrapolation_seconds=30):
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

    Returns:
        tuple: A tuple of (latitude, longitude), or None if interpolation is not possible
               (empty track) or the target time is too far outside the track range.
    """
    if not isinstance(target_time_utc, datetime) or target_time_utc.tzinfo is None:
        raise ValueError("target_time_utc must be a timezone-aware datetime object.")

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

    # Get the two points that bracket our target time
    p1 = gps_df.iloc[idx - 1]
    p2 = gps_df.iloc[idx]

    # Calculate the time proportion between the two points
    time_diff_total = (p2.name - p1.name).total_seconds()
    if time_diff_total == 0:
        return (p1['latitude'], p1['longitude']) # Points have same timestamp

    time_diff_target = (target_time_utc - p1.name).total_seconds()
    proportion = time_diff_target / time_diff_total

    # Linear interpolation for latitude and longitude
    lat = p1['latitude'] + proportion * (p2['latitude'] - p1['latitude'])
    lon = p1['longitude'] + proportion * (p2['longitude'] - p1['longitude'])

    return (lat, lon)


# --- EXIF Writing Functionality ---

def _decimal_to_dms(degrees_decimal):
    """Converts decimal degrees to the DMS format required by EXIF."""
    is_positive = degrees_decimal >= 0
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
        alt (float): Altitude in meters.
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

    try:
        # Copy the original file to the output path to avoid modifying the source
        shutil.copy2(image_path, output_path)

        # Load EXIF data from the new file
        exif_dict = piexif.load(output_path)
        
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
            piexif.GPSIFD.GPSAltitudeRef: 0, # 0 = Above sea level
            piexif.GPSIFD.GPSAltitude: (int(alt * 100), 100), # As a rational
        }

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
        piexif.insert(exif_bytes, output_path)
        
        return True

    except Exception as e:
        print(f"Error setting GPS for {os.path.basename(image_path)}: {e}")
        return False


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