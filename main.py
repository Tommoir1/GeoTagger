import sys
import os
import tempfile
from datetime import datetime, timedelta, timezone
import logging
import shutil
import csv
import traceback
import json # Required for JS communication AND transect save/load

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    print("WARNING: Optional library 'scipy' not found. Transect loading and other distance-based calculations will be slow.")
    print("         Install using: pip install scipy")
    cKDTree = None
    SCIPY_AVAILABLE = False

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QFileDialog,
                             QSpinBox, QDateTimeEdit, QRadioButton, QGroupBox,
                             QTableWidget, QTableWidgetItem, QMessageBox,
                             QProgressBar, QTabWidget, QTextEdit, QComboBox,
                             QGridLayout, QDoubleSpinBox, QCompleter, QAbstractItemView,
                             QSizePolicy, QInputDialog, QCheckBox, QListWidget, QListWidgetItem,
                             QSplitter) # Added QSplitter
from PyQt6.QtCore import (QObject, pyqtSlot, QThread, pyqtSignal, Qt, QUrl,
                          QTimer, QDateTime, QItemSelectionModel, QItemSelection,
                          QItemSelectionRange, QEvent)
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings, QWebEngineProfile, QWebEnginePage
from PyQt6.QtGui import QPalette, QColor, QIcon, QPixmap

# --- Data Handling ---
import pandas as pd
import numpy as np
import folium
import folium.plugins
import pytz

# --- Spatial Calculation ---
try:
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    print("WARNING: Optional library 'geopy' not found. Transect length calculations and preset length features will be disabled.")
    print("         Install using: pip install geopy")
    geodesic = None
    GEOPY_AVAILABLE = False

# --- Import Custom Engine ---
try:
    import georeference_engine as engine
    ENGINE_AVAILABLE = True
except ImportError:
    print("ERROR: georeference_engine.py not found.")
    print("       Please ensure it is in the same directory as this script or in the Python path.")
    engine = None
    ENGINE_AVAILABLE = False

# --- Set up logging for GUI ---
class QTextEditLogger(logging.Handler):
    def __init__(self, text_edit_widget):
        super().__init__()
        self.widget = text_edit_widget
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        try:
            self.widget.append(msg)
            self.widget.ensureCursorVisible()
        except Exception as e:
            print(f"ERROR logging to GUI: {e}\nLog Message: {msg}")

class MapInteractionHandler(QObject):
    gpsPointClicked = pyqtSignal(str, bool, bool)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent_app = parent
    def log(self, message, level=logging.INFO):
        if self._parent_app and hasattr(self._parent_app, 'log_message'):
            self._parent_app.log_message(f"[JS Handler] {message}", level)
        else:
            print(f"HANDLER FALLBACK LOG: {logging.getLevelName(level)} - {message}")
    @pyqtSlot(str, bool, bool)
    def handleGpsPointClick(self, timestamp_str_iso, ctrl_pressed, shift_pressed):
        self.log(f"Handler emitting signal for: {timestamp_str_iso}", logging.DEBUG)
        self.gpsPointClicked.emit(timestamp_str_iso, ctrl_pressed, shift_pressed)

class WorkerThread(QThread):
    progress = pyqtSignal(int)
    log_message = pyqtSignal(str)
    results_ready = pyqtSignal(object, list, object)
    error_occurred = pyqtSignal(str)
    map_ready = pyqtSignal(str)

    def __init__(self, gps_path, gps_parsing_params, media_path, media_type, frame_interval, sync_method, sync_params):
        super().__init__()
        self.gps_path = gps_path
        self.gps_parsing_params = gps_parsing_params
        self.media_path = media_path
        self.media_type = media_type
        self.frame_interval = frame_interval
        self.sync_method = sync_method
        self.sync_params = sync_params
        self.results_df = None
        self.original_media_list = []
        self.map_file = None
        self.temp_frame_dir = None
        self.final_gps_df_for_app = None
        

    def log(self, message, level=logging.INFO):
        log_entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - {logging.getLevelName(level)} - [Worker] {message}"
        self.log_message.emit(log_entry)

    def run(self):
        gps_df_final = None
        if not ENGINE_AVAILABLE:
            self.error_occurred.emit("Georeference engine (georeference_engine.py) not loaded.")
            return

        try:
            self.log("Worker thread started.")
            self.progress.emit(5)

            self.log("Step 1: Loading GPS data...")
            gps_df = None
            parse_type = self.gps_parsing_params.get('type', 'gpx')
            tz_str = self.gps_parsing_params.get('boat_timezone', 'UTC')

            if parse_type == 'gpx':
                try:
                    gps_df = engine.parse_gpx_file(self.gps_path)
                except Exception as e:
                    err_msg = f"Failed to parse GPX: {e}"
                    self.log(err_msg, logging.ERROR)
                    self.error_occurred.emit(err_msg)
                    return
            elif parse_type == 'csv':
                delimiter = ','
                try:
                    with open(self.gps_path, 'r', errors='ignore') as f:
                        sample = "".join(line for line in (f.readline() for _ in range(10)) if line and line.strip())
                        dialect = csv.Sniffer().sniff(sample)
                        delimiter = dialect.delimiter
                        self.log(f"Detected delimiter: '{repr(delimiter)}'")
                except Exception as e:
                    self.log(f"Delimiter sniff error ({e}), using ','.", logging.WARNING)

                try:
                    gps_df = engine.parse_boat_log_csv(
                        csv_file_path=self.gps_path,
                        date_col=self.gps_parsing_params['date_col'],
                        time_col=self.gps_parsing_params['time_col'],
                        lat_col=self.gps_parsing_params['lat_col'],
                        lon_col=self.gps_parsing_params['lon_col'],
                        alt_col=None,
                        boat_timezone_str=tz_str,
                        datetime_format=self.gps_parsing_params['datetime_format'],
                        datetime_col=self.gps_parsing_params.get('datetime_col'),
                        delimiter=delimiter
                    )
                except Exception as e:
                    err_msg = f"Failed to parse CSV: {e}"
                    self.log(err_msg, logging.ERROR)
                    self.log(traceback.format_exc(), logging.DEBUG)
                    self.error_occurred.emit(err_msg)
                    return
            else:
                err_msg = f"Unsupported GPS type: {parse_type}"
                self.log(err_msg, logging.ERROR)
                self.error_occurred.emit(err_msg)
                return

            if gps_df is None:
                err_msg = "GPS data load failed (returned None)."
                self.log(err_msg, logging.ERROR)
                self.error_occurred.emit(err_msg)
                return
            if gps_df.empty:
                self.log("GPS data is empty after parsing.", logging.WARNING)
            else:
                self.log(f"GPS data loaded: {len(gps_df)} points.")
                if not all(col in gps_df.columns for col in ['latitude', 'longitude']):
                    err_msg = "GPS data missing required 'latitude' or 'longitude' columns."
                    self.log(err_msg, logging.ERROR)
                    self.error_occurred.emit(err_msg)
                    return
                if not isinstance(gps_df.index, pd.DatetimeIndex):
                    err_msg = "GPS DataFrame index is not a DatetimeIndex."
                    self.log(err_msg, logging.ERROR)
                    self.error_occurred.emit(err_msg)
                    return
                if gps_df.index.tz is None:
                    self.log("GPS index is naive. Assuming UTC.", logging.WARNING)
                    try:
                        gps_df.index = gps_df.index.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
                        gps_df.dropna(subset=[gps_df.index.name], inplace=True)
                    except Exception as tz_err:
                        err_msg = f"Failed to localize naive GPS index to UTC: {tz_err}"
                        self.log(err_msg, logging.ERROR)
                        self.error_occurred.emit(err_msg)
                        return
                elif gps_df.index.tz.utcoffset(gps_df.index.min()) != timedelta(0):
                    self.log("GPS index not UTC. Converting...", logging.WARNING)
                    try:
                        gps_df.index = gps_df.index.tz_convert('UTC')
                    except Exception as idx_e:
                        err_msg = f"UTC conversion failed: {idx_e}"
                        self.log(err_msg, logging.ERROR)
                        self.error_occurred.emit(err_msg)
                        return
                if not gps_df.index.is_monotonic_increasing:
                    self.log("Sorting GPS data by time.", logging.INFO)
                    gps_df.sort_index(inplace=True)

            gps_df_final = gps_df
            self.progress.emit(15)

            self.log("Step 2: Loading image files...")
            media_items = []
            self.original_media_list = []
            if self.media_path and self.media_type == 'images':
                self.log(f"Scanning image dir: {self.media_path}")
                try:
                    media_items = engine.get_image_files_and_times(self.media_path)
                    self.original_media_list = media_items
                except Exception as e:
                    self.log(f"Error scanning images: {e}", logging.ERROR)
            if not media_items:
                self.log("No compatible images found.", logging.WARNING)
            else:
                self.log(f"Found {len(media_items)} images.", logging.INFO)
            self.progress.emit(30)

            self.log("Step 3: Calculating time offset...")
            time_offset = timedelta(0)
            if media_items:
                sync_params_with_tz = self.sync_params.copy()
                sync_params_with_tz['boat_timezone_str'] = tz_str
                try:
                    time_offset = engine.calculate_time_offset(self.sync_method, **sync_params_with_tz)
                    self.log(f"Time offset calculated: {time_offset}")
                except ValueError as e:
                    err_msg = f"Offset calc error: {e}"
                    self.log(err_msg, logging.ERROR)
                    self.error_occurred.emit(err_msg)
                    return
                except Exception as e:
                    err_msg = f"Offset calc error: {e}"
                    self.log(err_msg, logging.ERROR)
                    self.log(traceback.format_exc(), logging.DEBUG)
                    self.error_occurred.emit(err_msg)
                    return
            else:
                self.log("Skipping offset calculation (no media).")
            self.progress.emit(40)

            self.log("Step 4: Georeferencing images...")
            georeferenced_data = []
            skipped_interpolation_count = 0
            total_items = len(media_items)
            if not media_items:
                self.log("No images to georeference.")
                self.results_df = pd.DataFrame(columns=['Identifier', 'File Path', 'Original Timestamp', 'Corrected Timestamp (UTC)', 'Latitude', 'Longitude'])
            else:
                media_data_tz_obj = pytz.timezone(tz_str)
                valid_gps_for_interp = gps_df_final is not None and not gps_df_final.empty

                for i, item in enumerate(media_items):
                    lat, lon = None, None
                    id_val = item.get('identifier', f'Item_{i}')
                    fp = item.get('file_path', 'N/A')
                    orig_ts_naive = item.get('image_time')
                    orig_disp = "N/A"
                    corr_utc_iso = "N/A"

                    try:
                        if orig_ts_naive and isinstance(orig_ts_naive, datetime):
                            try:
                                localized_orig_ts = media_data_tz_obj.localize(orig_ts_naive, is_dst=None)
                                orig_disp = localized_orig_ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + f" ({tz_str})"
                            except Exception as disp_err:
                                self.log(f"Error formatting original time display {orig_ts_naive} for {id_val}: {disp_err}", logging.WARNING)
                                orig_disp = orig_ts_naive.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + " (Naive, Format Error)"

                            corr_naive = orig_ts_naive + time_offset
                            corr_utc = pytz.utc.localize(corr_naive)
                            corr_utc_iso = corr_utc.isoformat(timespec='milliseconds').replace('+00:00', 'Z')

                            if valid_gps_for_interp:
                                try:
                                    pos_result = engine.interpolate_gps_position(gps_df_final, corr_utc)
                                    if pos_result is not None:
                                        if isinstance(pos_result, (tuple, list)) and len(pos_result) >= 2:
                                            lat, lon = pos_result[0], pos_result[1]
                                        elif isinstance(pos_result, dict):
                                            lat, lon = pos_result.get('latitude'), pos_result.get('longitude')
                                    if lat is None or lon is None:
                                        skipped_interpolation_count += 1
                                except Exception as interp_err:
                                    self.log(f"Error during interpolation for {id_val} at {corr_utc}: {interp_err}", logging.WARNING)
                                    skipped_interpolation_count += 1
                                    lat, lon = None, None
                            else:
                                skipped_interpolation_count += 1
                        else:
                            self.log(f"Skipping item '{id_val}': Missing or invalid original timestamp.", logging.DEBUG)
                            skipped_interpolation_count += 1

                        georeferenced_data.append({
                            'Identifier': id_val, 'File Path': fp, 'Original Timestamp': orig_disp,
                            'Corrected Timestamp (UTC)': corr_utc_iso if corr_utc_iso != "N/A" else pd.NA,
                            'Latitude': lat if pd.notna(lat) else np.nan,
                            'Longitude': lon if pd.notna(lon) else np.nan
                        })
                    except Exception as e:
                        self.log(f"Error processing media item '{id_val}': {e}", logging.ERROR)
                        georeferenced_data.append({
                            'Identifier': id_val, 'File Path': fp, 'Original Timestamp': 'ERROR',
                            'Corrected Timestamp (UTC)': 'ERROR', 'Latitude': np.nan, 'Longitude': np.nan
                        })
                    prog = int(40 + (i + 1) / total_items * 55) if total_items > 0 else 95
                    self.progress.emit(prog)

                self.results_df = pd.DataFrame(georeferenced_data)
                if 'Corrected Timestamp (UTC)' in self.results_df.columns:
                    self.results_df['Corrected Timestamp (UTC)'] = pd.to_datetime(
                        self.results_df['Corrected Timestamp (UTC)'].astype(str).str.replace('Z','+00:00', regex=False),
                        errors='coerce', utc=True
                    )
                if 'Altitude' in self.results_df.columns:
                    self.results_df.drop(columns=['Altitude'], inplace=True, errors='ignore')
                if skipped_interpolation_count > 0:
                    self.log(f"{skipped_interpolation_count} items could not be georeferenced.", logging.WARNING)
            self.log("Image georeferencing complete.")

            self.final_gps_df_for_app = gps_df_final
            self.results_ready.emit(self.results_df, self.original_media_list, gps_df_final)
            self.progress.emit(95)

            self.log("Step 5: Generating initial map...")
            self.map_file = self.generate_combined_map(gps_df_final, self.results_df)
            if self.map_file and os.path.exists(self.map_file):
                self.map_ready.emit(self.map_file)
                self.log("Initial map generated.")
            else:
                self.log("Initial map generation failed.", logging.WARNING)
            self.progress.emit(100)

        except Exception as e:
            error_message = f"Critical worker error: {e}"
            self.log(error_message, logging.CRITICAL)
            self.log(traceback.format_exc(), logging.ERROR)
            self.error_occurred.emit(error_message)
            self.progress.emit(0)
        finally:
            self.log("Worker thread finished.", logging.DEBUG)
            if self.temp_frame_dir and os.path.exists(self.temp_frame_dir):
                try:
                    shutil.rmtree(self.temp_frame_dir)
                    self.log("Cleaned temp dir.", logging.DEBUG)
                except Exception as err:
                    self.log(f"Error removing temp dir: {err}", logging.WARNING)

    def generate_combined_map(self, gps_df_to_map, results_df_to_map):
        self.log("[Worker] --- Generating combined map ---")
        has_valid_gps = False
        valid_gps_df = pd.DataFrame()
        if gps_df_to_map is not None and not gps_df_to_map.empty:
            try:
                gps_map_df = gps_df_to_map.copy()
                req = ['latitude','longitude']
                if not all(c in gps_map_df.columns for c in req):
                    self.log(f"[Worker] GPS map cols missing: {req}", logging.ERROR)
                    return None
                gps_map_df['latitude'] = pd.to_numeric(gps_map_df['latitude'], errors='coerce')
                gps_map_df['longitude'] = pd.to_numeric(gps_map_df['longitude'], errors='coerce')
                valid_gps_df = gps_map_df.dropna(subset=req).copy()
                if not isinstance(valid_gps_df.index, pd.DatetimeIndex) or valid_gps_df.index.tz is None or valid_gps_df.index.tz.utcoffset(valid_gps_df.index.min()) != timedelta(0):
                    self.log("[Worker] GPS index invalid map.", logging.ERROR)
                    return None
                if not valid_gps_df.index.is_monotonic_increasing:
                    valid_gps_df.sort_index(inplace=True)
                if not valid_gps_df.empty:
                    has_valid_gps = True
                    self.log(f"[Worker] Using {len(valid_gps_df)} valid GPS points for map.")
            except Exception as e:
                self.log(f"[Worker] Error prep GPS map: {e}", logging.ERROR)
                has_valid_gps = False
        else:
            self.log("[Worker] No GPS data for map.", logging.INFO)

        has_valid_results = False
        valid_results_df = pd.DataFrame()
        if results_df_to_map is not None and not results_df_to_map.empty:
            try:
                res_map_df = results_df_to_map.copy()
                req = ['Latitude','Longitude','Corrected Timestamp (UTC)','Identifier']
                if all(c in res_map_df.columns for c in req):
                    if not pd.api.types.is_datetime64_any_dtype(res_map_df['Corrected Timestamp (UTC)']):
                        res_map_df['TS_dt'] = pd.to_datetime(
                            res_map_df['Corrected Timestamp (UTC)'].astype(str).str.replace('Z','+00:00', regex=False),
                            errors='coerce', utc=True
                        )
                    else:
                        res_map_df['TS_dt'] = res_map_df['Corrected Timestamp (UTC)']
                    res_map_df['Latitude']=pd.to_numeric(res_map_df['Latitude'], errors='coerce')
                    res_map_df['Longitude']=pd.to_numeric(res_map_df['Longitude'], errors='coerce')
                    valid_results_df = res_map_df.dropna(subset=['Latitude','Longitude','TS_dt']).copy()
                    if not valid_results_df.empty:
                        has_valid_results = True
                        self.log(f"[Worker] Using {len(valid_results_df)} valid results for map.")
                else:
                    self.log(f"[Worker] Results missing map cols: {req}", logging.WARNING)
            except Exception as e:
                self.log(f"[Worker] Error prep results map: {e}", logging.ERROR)
                has_valid_results = False
        else:
            self.log("[Worker] No results data for map.", logging.INFO)

        if not has_valid_gps and not has_valid_results:
            self.log("[Worker] No valid data to plot.", logging.WARNING)
            return None

        center, zoom = [0,0], 2
        try:
            tgt_df, lat_c, lon_c = (valid_gps_df, 'latitude', 'longitude') if has_valid_gps else \
                                (valid_results_df, 'Latitude', 'Longitude') if has_valid_results else \
                                (None, None, None)
            if tgt_df is not None and not tgt_df.empty:
                mean_lat, mean_lon = tgt_df[lat_c].mean(), tgt_df[lon_c].mean()
                if pd.notna(mean_lat) and pd.notna(mean_lon):
                    center, zoom = [mean_lat, mean_lon], 15
        except Exception as e:
            self.log(f"[Worker] Center calc error: {e}", logging.WARNING)

        map_path = None
        try:
            m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Tiles © Esri', name='Esri Satellite', overlay=False, control=True
            ).add_to(m)

            js_code = """
            <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script>
window.savedTransectLines = window.savedTransectLines || []; // Ensure it's initialized
window.addEventListener("DOMContentLoaded", function() {
    if (window.qt && qt.webChannelTransport) {
        new QWebChannel(qt.webChannelTransport, function(channel) {
            window.pyHandler = channel.objects.pyHandler;
            console.log("JS: pyHandler connected.");
        });
    }
});
function sendClickedGpsTimestampToPython(ts, ctrl, shift) {
    if (window.pyHandler && window.pyHandler.handleGpsPointClick) {
        window.pyHandler.handleGpsPointClick(ts, ctrl, shift);
    }
}
function highlightTransectRange(timestamps) {
    var map = Object.values(window).find(x => x instanceof L.Map);
    if (!map) return;
    map.eachLayer(function(layer) {
        if (layer.options && layer.options.customTimestamp) { // Check if it's a GPS point marker
            var isHl = timestamps.indexOf(layer.options.customTimestamp) !== -1;
            layer.setStyle({
                color: isHl? 'lime':'blue', // Highlighted vs default GPS point color
                fillColor: isHl? 'lime':'blue',
                fillOpacity: isHl? 0.9:0.6,
                weight: isHl? 3:1 // Thicker if highlighted
            });
            if (layer.setRadius) layer.setRadius(isHl? 6:4); // Larger if highlighted
        }
    });
}
// This now only handles the temporary YELLOW line during definition
function drawTransectLine(lat1, lon1, lat2, lon2) {
    var map = Object.values(window).find(x => x instanceof L.Map);
    if (!map) return;
    if (window.transectLine) { // Always clear the previous temp line
        map.removeLayer(window.transectLine);
    }
    window.transectLine = L.polyline([[lat1, lon1], [lat2, lon2]], { color: 'yellow', weight: 2, interactive: false }).addTo(map);
}
// NEW FUNCTION (this was the missing one): Draws a permanent ORANGE line for a saved transect
function drawSavedTransectLine(lat1, lon1, lat2, lon2, name) {
    var map = Object.values(window).find(x => x instanceof L.Map);
    if (!map) return;
    const newLine = L.polyline([[lat1, lon1], [lat2, lon2]], { color: 'orange', weight: 3, opacity: 0.8 })
        .bindTooltip("Saved: " + name)
        .addTo(map);
    window.savedTransectLines.push(newLine); // Add to our list to keep track
}
function clearTransectLine() { // Clears the single, active (yellow) transect line
    var map = Object.values(window).find(x => x instanceof L.Map);
    if (map && window.transectLine) {
        map.removeLayer(window.transectLine);
        window.transectLine = null;
    }
}
function drawPresetLengthCircle(lat, lon, r, action) {
    var map = Object.values(window).find(x => x instanceof L.Map);
    if (!map) return;
    if (window.presetRadiusCircle) { // Clear existing circle first
        map.removeLayer(window.presetRadiusCircle);
        window.presetRadiusCircle = null;
    }
    if (action==='draw' && lat!=null && lon!=null && r>0) {
        window.presetRadiusCircle = L.circle([lat, lon], {
            radius: r, // in meters
            color: 'orange', dashArray: '5,5', weight:2,
            fillColor: 'orange', fillOpacity:0.1,
            interactive: false
        }).addTo(map);
    }
}
function clearAllSavedTransectLines() {
    var map = Object.values(window).find(x => x instanceof L.Map);
    if (!map) return;
    if (window.savedTransectLines && window.savedTransectLines.length > 0) {
        for (var i = 0; i < window.savedTransectLines.length; i++) {
            map.removeLayer(window.savedTransectLines[i]);
        }
        window.savedTransectLines = []; // Clear the array
    }
    if (window.transectLine) {
        map.removeLayer(window.transectLine);
        window.transectLine = null;
    }
}
</script>
            """
            m.get_root().html.add_child(folium.Element(js_code))
            # --- Bind direct clicks on the blue‐dot markers for visual sync ---
            click_binding = """
            <script>
            window.addEventListener("DOMContentLoaded", function() {
            // find the Leaflet map instance
            var map = Object.values(window).find(x => x instanceof L.Map);
            if (!map) return;
            // walk every layer; if it has our customTimestamp, hook click
            map.eachLayer(function(layer) {
                if (layer.options && layer.options.customTimestamp) {
                layer.on('click', function(e) {
                    // forward the click straight to Python
                    window.pyHandler.handleGpsPointClick(
                    layer.options.customTimestamp,
                    e.originalEvent.ctrlKey,
                    e.originalEvent.shiftKey
                    );
                });
                }
            });
            });
            </script>
            """
            m.get_root().html.add_child(folium.Element(click_binding))


            if has_valid_gps:
                gps_group = folium.FeatureGroup(
                    name="GPS Track (Click for Transect)",
                    show=True,
                    overlay=True
                ).add_to(m)
                for ts, row in valid_gps_df.iterrows():
                    if not isinstance(ts, pd.Timestamp):
                        continue
                    ts_iso = ts.isoformat(timespec='milliseconds').replace('+00:00', 'Z')
                    ts_tip = ts.strftime('%H:%M:%S.%f')[:-3] + 'Z'
                    js_call = (
                        f"sendClickedGpsTimestampToPython('{ts_iso}', event.ctrlKey, event.shiftKey); "
                        "var m=null;for(var k in window){if(window[k] instanceof L.Map){m=window[k];break;}} "
                        "if(m){m.closePopup();};"
                    )
                    escaped_js = js_call.replace('"', '&quot;')
                    popup_html = (
                        f'<b>GPS Point</b><br>'
                        f'Time: {ts_tip}<br>'
                        f'Lat: {row["latitude"]:.6f}, Lon: {row["longitude"]:.6f}<br>'
                        f'<button type="button" onclick="{escaped_js}">Select This Point</button>'
                    )
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=4,
                        color='blue',
                        weight=1,
                        fill=True,
                        fill_color='blue',
                        fill_opacity=0.6,
                        popup=folium.Popup(popup_html, max_width=250),
                        tooltip=f"Click to open: {ts_tip}",
                        options={'customTimestamp': ts_iso}
                    ).add_to(gps_group)
            if has_valid_results:
                img_cluster = folium.plugins.MarkerCluster(name="Images", show=True, overlay=True).add_to(m)
                for _, row_res in valid_results_df.iterrows():
                    ts_utc_res = row_res['TS_dt']
                    pop = f"ID: {row_res['Identifier']}<br>Time: {ts_utc_res.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ' UTC' if pd.notna(ts_utc_res) else 'N/A'}<br>Lat: {row_res['Latitude']:.7f}, Lon: {row_res['Longitude']:.7f}"
                    folium.CircleMarker(location=[row_res['Latitude'], row_res['Longitude']], radius=4, color='red', weight=1, fill=True,
                                        fill_color='red', fill_opacity=0.7, popup=folium.Popup(pop, max_width=300),
                                        tooltip=f"Img: {row_res['Identifier']}").add_to(img_cluster)

            folium.LayerControl().add_to(m)
            fd, map_path_temp = tempfile.mkstemp(suffix=".html", prefix="geotagger_map_worker_")
            os.close(fd)
            map_path = map_path_temp
            m.save(map_path)
            self.log(f"[Worker] Map saved to temporary file: {map_path}")
            return map_path
        except Exception as e:
            self.log(f"[Worker] Map generation failed: {e}", logging.CRITICAL)
            self.log(traceback.format_exc(), logging.ERROR)
            if map_path and os.path.exists(map_path):
                try:
                    os.remove(map_path)
                except Exception as rm_err:
                    self.log(f"[Worker] Error removing partial map file: {rm_err}", logging.WARNING)
            return None

class GeoTaggerApp(QWidget):
    PRESET_LENGTH_TOLERANCE_METERS = 0.5

    def __init__(self):
        super().__init__()
        self.gps_file_path = None
        self.media_path = None
        self.media_type = None
        self.transect_csv_file_path = None

        self.results_df = None
        self.original_media_list = []
        self.identifier_to_path_map = {}
        self.processed_gps_df = None
        self.preliminary_gps_df = None
        self.preliminary_media_list = []
        self.sync_image_info = None
        self.sync_gps_point_info = None

        self.worker_thread = None
        self.geotag_output_dir_base = "geotagged_images"
        self._map_file_path = None

        self.transect_definition_mode = None
        self.transect_start_time = None
        self.transect_end_time = None
        self.transect_length_meters = None
        self.current_timezone_str = "UTC"
        self.transects_json_dir_name = "transects_json" 
        self.gps_kdtree = None
        self.gps_coordinates_for_kdtree = None
        self.gps_kdtree_df_map = None # Maps index from kdtree back to original DataFrame
        

        self.available_dates = []
        self.selected_filter_date = None
        self.filtered_gps_by_date_df = None

        self.saved_transects = []
        self.autosave_json_path = None

        self.setWindowTitle(f"GeoTagger - v1.8.5{' (Geopy Disabled)' if not GEOPY_AVAILABLE else ''}")
        self.setGeometry(100, 100, 1300, 1000)

        # --- Main Layout (will contain the splitter) ---
        self.main_layout = QVBoxLayout(self) # Changed from QHBoxLayout to QVBoxLayout for splitter
        self.setLayout(self.main_layout)

        # --- Top layout for inputs and config (will go into left panel of splitter) ---
        top_layout = QHBoxLayout()
        self.input_layout = QVBoxLayout()
        self.config_layout = QVBoxLayout()
        # process_layout and export_layout will also go into the left panel

        # --- Input Group and Layout (self.input_layout) ---
        self.input_group = QGroupBox("Input Data")
        input_group_layout = QVBoxLayout()

        gps_layout = QHBoxLayout()
        self.gps_label = QLabel("GPS File (GPX/CSV):")
        gps_layout.addWidget(self.gps_label)
        self.gps_path_display = QLineEdit()
        self.gps_path_display.setPlaceholderText("Load GPS track...")
        self.gps_path_display.setReadOnly(True)
        gps_layout.addWidget(self.gps_path_display, 1)
        self.gps_load_button = QPushButton("Load GPS...")
        self.gps_load_button.setIcon(QIcon.fromTheme("document-open"))
        self.gps_load_button.clicked.connect(self.load_gps_data)
        gps_layout.addWidget(self.gps_load_button)
        input_group_layout.addLayout(gps_layout)

        # Find this section
        media_layout = QHBoxLayout()
        # CHANGE THE LABEL TEXT
        self.media_label = QLabel("Image Parent Directory (Recursive):") 
        # ADD A TOOLTIP TO THE LABEL
        self.media_label.setToolTip("Select the main folder containing your images.\nThe application will search through all subfolders.")
        media_layout.addWidget(self.media_label)
        self.media_path_display = QLineEdit()
        self.media_path_display.setPlaceholderText("Load parent image directory...") # Optional text change
        self.media_path_display.setReadOnly(True)
        media_layout.addWidget(self.media_path_display, 1)
        self.media_load_button = QPushButton("Load Images...")
        self.media_load_button.setIcon(QIcon.fromTheme("folder-image"))
        # ADD/UPDATE THE BUTTON'S TOOLTIP
        self.media_load_button.setToolTip("Select the main folder; all images in it and its subfolders will be loaded.")
        self.media_load_button.clicked.connect(self.load_media)
        media_layout.addWidget(self.media_load_button)

        self.media_exif_button = QPushButton("Load GPS-Tagged Images...")
        self.media_exif_button.setIcon(QIcon.fromTheme("folder-image"))
        self.media_exif_button.setToolTip(
            "Load a folder of images that already have GPS coordinates embedded in EXIF.\n"
            "Skips the GPS-file load and time-sync steps — you can go straight to defining transects."
        )
        self.media_exif_button.clicked.connect(self.load_images_with_embedded_gps)
        media_layout.addWidget(self.media_exif_button)

        input_group_layout.addLayout(media_layout)

        tz_layout = QHBoxLayout()
        self.media_timezone_label = QLabel("Data Timezone:")
        tz_layout.addWidget(self.media_timezone_label)
        self.media_timezone_input = QComboBox()
        self.media_timezone_input.setToolTip("Timezone for: Image EXIF/GPS CSV (if naive), Transect CSV (if naive), Transect Display, Date Filter.")
        tz_layout.addWidget(self.media_timezone_input, 1)
        all_tz_list = ["UTC"]
        try:
            all_tz_list = sorted(pytz.common_timezones)
            self.media_timezone_input.addItems(all_tz_list)
            try:
                local_name = datetime.now().astimezone().tzname()
                idx = self.media_timezone_input.findText(local_name, Qt.MatchFlag.MatchContains|Qt.MatchFlag.MatchFixedString)
                self.media_timezone_input.setCurrentIndex(idx if idx >=0 else self.media_timezone_input.findText("UTC", Qt.MatchFlag.MatchFixedString))
            except Exception:
                self.media_timezone_input.setCurrentIndex(self.media_timezone_input.findText("UTC", Qt.MatchFlag.MatchFixedString))
            self.current_timezone_str = self.media_timezone_input.currentText()
            self.media_timezone_input.currentTextChanged.connect(self.update_current_timezone)
        except Exception as tz_err:
            self.media_timezone_input.addItems(["UTC"])
            self.current_timezone_str = "UTC"
        self.media_timezone_input.setEditable(True)
        self.media_timezone_input.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        completer = QCompleter(all_tz_list, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.media_timezone_input.setCompleter(completer)
        input_group_layout.addLayout(tz_layout)

        date_filter_layout = QHBoxLayout()
        self.date_filter_label = QLabel(f"Filter GPS by Date ({self.current_timezone_str}):")
        date_filter_layout.addWidget(self.date_filter_label)
        self.date_filter_combo = QComboBox()
        self.date_filter_combo.setToolTip("Filter displayed GPS track & transect ops to a single local date (using Data Timezone).")
        self.date_filter_combo.addItem("Show All Dates")
        self.date_filter_combo.setEnabled(False)
        self.date_filter_combo.currentTextChanged.connect(self.on_date_filter_changed)
        date_filter_layout.addWidget(self.date_filter_combo, 1)
        input_group_layout.addLayout(date_filter_layout)
        self.input_group.setLayout(input_group_layout)
        self.input_layout.addWidget(self.input_group)

        self.csv_mapping_group = QGroupBox("GPS CSV Column Mapping")
        csv_map_layout = QGridLayout()
        csv_map_layout.addWidget(QLabel("Lat Col:"), 0, 0)
        self.lat_col_combo = QComboBox()
        csv_map_layout.addWidget(self.lat_col_combo, 0, 1)
        csv_map_layout.addWidget(QLabel("Lon Col:"), 1, 0)
        self.lon_col_combo = QComboBox()
        csv_map_layout.addWidget(self.lon_col_combo, 1, 1)
        csv_map_layout.addWidget(QLabel("Timestamp:"), 2, 0, 1, 2)
        self.ts_separate_radio = QRadioButton("Separate Date & Time")
        self.ts_separate_radio.setChecked(True)
        self.ts_separate_radio.toggled.connect(self.update_timestamp_column_ui)
        csv_map_layout.addWidget(self.ts_separate_radio, 3, 0, 1, 2)
        self.date_col_label = QLabel("  Date Col:")
        csv_map_layout.addWidget(self.date_col_label, 4, 0)
        self.date_col_combo = QComboBox()
        csv_map_layout.addWidget(self.date_col_combo, 4, 1)
        self.time_col_label = QLabel("  Time Col:")
        csv_map_layout.addWidget(self.time_col_label, 5, 0)
        self.time_col_combo = QComboBox()
        csv_map_layout.addWidget(self.time_col_combo, 5, 1)
        self.ts_single_radio = QRadioButton("Single Timestamp")
        self.ts_single_radio.toggled.connect(self.update_timestamp_column_ui)
        csv_map_layout.addWidget(self.ts_single_radio, 6, 0, 1, 2)
        self.datetime_col_label = QLabel("  Timestamp Col:")
        csv_map_layout.addWidget(self.datetime_col_label, 7, 0)
        self.datetime_col_combo = QComboBox()
        csv_map_layout.addWidget(self.datetime_col_combo, 7, 1)
        csv_map_layout.addWidget(QLabel("Format Str:"), 8, 0)
        self.datetime_format_input = QLineEdit("%d/%m/%Y %H:%M:%S")
        self.datetime_format_input.setToolTip("Python strptime format (e.g., %d/%m/%Y %H:%M:%S or %Y-%m-%dT%H:%M:%S.%fZ).")
        csv_map_layout.addWidget(self.datetime_format_input, 8, 1)
        csv_map_layout.setColumnStretch(1, 1)
        self.csv_mapping_group.setLayout(csv_map_layout)
        self.csv_mapping_group.setVisible(False)
        self.update_timestamp_column_ui()
        self.input_layout.addWidget(self.csv_mapping_group)

        self.transect_csv_load_group = QGroupBox("Bulk Transect Definition (from CSV)")
        transect_csv_load_layout = QVBoxLayout()
        self.transect_csv_path_display = QLineEdit()
        self.transect_csv_path_display.setPlaceholderText("Load Transect CSV file...")
        self.transect_csv_path_display.setReadOnly(True)
        self.load_transects_csv_button = QPushButton("Load Transect CSV...")
        self.load_transects_csv_button.setIcon(QIcon.fromTheme("text-csv"))
        self.load_transects_csv_button.setToolTip("Load multiple transect definitions from a CSV file.")
        self.load_transects_csv_button.clicked.connect(self._load_transects_csv_file)
        self.load_transects_csv_button.setEnabled(False)
        load_transect_csv_hbox = QHBoxLayout()
        load_transect_csv_hbox.addWidget(self.transect_csv_path_display, 1)
        load_transect_csv_hbox.addWidget(self.load_transects_csv_button)
        transect_csv_load_layout.addLayout(load_transect_csv_hbox)
        self.transect_csv_load_group.setLayout(transect_csv_load_layout)
        self.input_layout.addWidget(self.transect_csv_load_group)

        self.transect_csv_mapping_group = QGroupBox("Transect CSV Column Mapping")
        transect_csv_map_layout = QGridLayout()
        transect_csv_map_layout.addWidget(QLabel("Name Col:"), 0, 0)
        self.transect_csv_name_col_combo = QComboBox()
        self.transect_csv_name_col_combo.setToolTip("Column in CSV containing the transect name/ID.")
        transect_csv_map_layout.addWidget(self.transect_csv_name_col_combo, 0, 1)
        transect_csv_map_layout.addWidget(QLabel("Start Time Col:"), 1, 0)
        self.transect_csv_start_time_col_combo = QComboBox()
        self.transect_csv_start_time_col_combo.setToolTip("Column for transect start time.")
        transect_csv_map_layout.addWidget(self.transect_csv_start_time_col_combo, 1, 1)
        transect_csv_map_layout.addWidget(QLabel("End Time Col:"), 2, 0)
        self.transect_csv_end_time_col_combo = QComboBox()
        self.transect_csv_end_time_col_combo.setToolTip("Column for transect end time.")
        transect_csv_map_layout.addWidget(self.transect_csv_end_time_col_combo, 2, 1)

        transect_csv_map_layout.addWidget(QLabel("Length Col (Optional):"), 3, 0)
        self.transect_csv_length_col_combo = QComboBox()
        self.transect_csv_length_col_combo.setToolTip("Optional: Column for pre-calculated transect length in meters.")
        transect_csv_map_layout.addWidget(self.transect_csv_length_col_combo, 3, 1)

        transect_csv_map_layout.addWidget(QLabel("Timestamp Format:"), 4, 0)
        self.transect_csv_datetime_format_input = QLineEdit("%Y-%m-%d %H:%M:%S")
        self.transect_csv_datetime_format_input.setToolTip("Python strptime format for start/end time columns in the CSV (e.g., %Y-%m-%d %H:%M:%S).")
        transect_csv_map_layout.addWidget(self.transect_csv_datetime_format_input, 4, 1)

        transect_csv_map_layout.addWidget(QLabel("CSV Timestamps are in 'Data Timezone' selected above."), 5, 0, 1, 2)
        self.add_csv_transects_to_list_button = QPushButton("Add Mapped CSV Transects to Batch List")
        self.add_csv_transects_to_list_button.setIcon(QIcon.fromTheme("list-add"))
        self.add_csv_transects_to_list_button.clicked.connect(self._add_mapped_csv_transects_to_list)
        transect_csv_map_layout.addWidget(self.add_csv_transects_to_list_button, 6,0,1,2)
        transect_csv_map_layout.setColumnStretch(1,1)
        self.transect_csv_mapping_group.setLayout(transect_csv_map_layout)
        self.transect_csv_mapping_group.setVisible(False)
        self.input_layout.addWidget(self.transect_csv_mapping_group)
        self.input_layout.addStretch()

        # --- Config Group and Layout (self.config_layout) ---
        self.config_group = QGroupBox("Configuration")
        config_layout_main = QVBoxLayout()
        self.time_sync_group = QGroupBox("Time Synchronization")
        time_sync_layout = QVBoxLayout()
        self.sync_manual_radio = QRadioButton("Manual Offset")
        self.sync_manual_radio.setChecked(True)
        man_lay = QHBoxLayout()
        self.offset_spinbox = QDoubleSpinBox()
        self.offset_spinbox.setDecimals(3)
        self.offset_spinbox.setRange(-172800, 172800)
        self.offset_spinbox.setValue(0.0)
        self.offset_spinbox.setSuffix(" s")
        self.offset_spinbox.setToolTip("GPS Time (UTC) = Media Time + Offset")
        man_lay.addWidget(self.offset_spinbox)
        man_lay.addWidget(QLabel("(+ if Media Time lags GPS/UTC)"))
        man_lay.addStretch()
        time_sync_layout.addWidget(self.sync_manual_radio)
        time_sync_layout.addLayout(man_lay)
        self.sync_point_radio = QRadioButton("Sync Point Calibration")
        sync_lay = QGridLayout()
        sync_lay.addWidget(QLabel("Media Time (Local):"), 0, 0)
        self.gopro_time_edit = QDateTimeEdit(QDateTime.currentDateTime())
        self.gopro_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss.zzz")
        self.gopro_time_edit.setCalendarPopup(True)
        self.gopro_time_edit.setToolTip("Time from Media source in the selected Data Timezone.")
        sync_lay.addWidget(self.gopro_time_edit, 0, 1)
        sync_lay.addWidget(QLabel("GPS Time (UTC):"), 1, 0)
        self.gps_time_edit = QDateTimeEdit(QDateTime.currentDateTimeUtc())
        self.gps_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss.zzz")
        self.gps_time_edit.setTimeSpec(Qt.TimeSpec.UTC)
        self.gps_time_edit.setCalendarPopup(True)
        self.gps_time_edit.setToolTip("Corresponding time from GPS source (must be UTC).")
        sync_lay.addWidget(self.gps_time_edit, 1, 1)
        sync_lay.setColumnStretch(1, 1)
        time_sync_layout.addWidget(self.sync_point_radio)
        time_sync_layout.addLayout(sync_lay)

        # --- Visual Sync UI ---
        # --- Visual Sync Panel (Image → Map Point) ---
        self.sync_visual_radio = QRadioButton("Visual Sync (Image to Map Point)")
        time_sync_layout.addWidget(self.sync_visual_radio)
        self.sync_visual_radio.toggled.connect(self._on_visual_sync_toggled)

        visual_sync_layout = QGridLayout()
        visual_sync_layout.setContentsMargins(20, 5, 5, 5)

        # 1) Select‐sync button
        self.select_sync_image_button = QPushButton("1. Select Sync Image…")
        self.select_sync_image_button.setFixedSize(120, 30)
        self.select_sync_image_button.clicked.connect(self.select_image_for_sync)
        visual_sync_layout.addWidget(self.select_sync_image_button, 0, 0, 1, 1)

        # 2) Thumbnail preview
        self.sync_image_preview_label = QLabel("No Image Selected")
        self.sync_image_preview_label.setFixedSize(150, 100)
        self.sync_image_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sync_image_preview_label.setStyleSheet(
            "border: 1px solid #c0c0c0; background-color: #f0f0f0;"
        )
        visual_sync_layout.addWidget(self.sync_image_preview_label, 0, 1, 1, 1)

        # 3) Status message
        self.visual_sync_status_label = QLabel("Status: Load media & GPS first.")
        self.visual_sync_status_label.setWordWrap(True)
        visual_sync_layout.addWidget(self.visual_sync_status_label, 1, 0, 1, 2)

        # 4) Image‐time display
        visual_sync_layout.addWidget(QLabel("Image Time:"),                   2, 0)
        self.sync_image_time_display = QLineEdit("N/A")
        self.sync_image_time_display.setReadOnly(True)
        visual_sync_layout.addWidget(self.sync_image_time_display,             2, 1)

        # 5) GPS‐time display
        visual_sync_layout.addWidget(QLabel("GPS Time (UTC):"),               3, 0)
        self.sync_gps_point_time_display = QLineEdit("N/A")
        self.sync_gps_point_time_display.setReadOnly(True)
        visual_sync_layout.addWidget(self.sync_gps_point_time_display,         3, 1)

        # 6) GPS‐coords display
        visual_sync_layout.addWidget(QLabel("GPS Coords:"),                   4, 0)
        self.sync_gps_point_coords_display = QLineEdit("N/A")
        self.sync_gps_point_coords_display.setReadOnly(True)
        visual_sync_layout.addWidget(self.sync_gps_point_coords_display,       4, 1)

        # Stretch rules so the left column expands, but the preview stays fixed
        visual_sync_layout.setColumnStretch(0, 1)
        visual_sync_layout.setColumnStretch(1, 0)

        # Wrap it all in a group box and add it back into your time_sync_group
        self.visual_sync_group = QGroupBox("Visual Sync (Image → Map Point)")
        self.visual_sync_group.setLayout(visual_sync_layout)
        time_sync_layout.addWidget(self.visual_sync_group)

        time_sync_layout.addLayout(visual_sync_layout)

        self.sync_manual_radio.toggled.connect(self.update_sync_method_ui)
        self.sync_visual_radio.toggled.connect(self.update_sync_method_ui)
        self.update_sync_method_ui()
        self.time_sync_group.setLayout(time_sync_layout)
        config_layout_main.addWidget(self.time_sync_group)

        self.transect_group = QGroupBox("Define Single Transect (via Map Clicks)")
        self.transect_layout = QGridLayout()
        self.define_transect_button = QPushButton("Define Transect Start/End")
        self.define_transect_button.setToolTip("Click this, then click start point on map, then click end point on map.")
        self.define_transect_button.setCheckable(True)
        self.define_transect_button.clicked.connect(self.toggle_transect_definition_mode)
        self.transect_layout.addWidget(self.define_transect_button, 0, 0, 1, 2)
        self.transect_status_label = QLabel("Status: Load data and process first.")
        self.transect_status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.transect_status_label.setWordWrap(True)
        self.transect_layout.addWidget(self.transect_status_label, 1, 0, 1, 2)
        self.transect_layout.addWidget(QLabel("Start Time:"), 2, 0)
        self.transect_start_display = QLineEdit("Not Set")
        self.transect_start_display.setReadOnly(True)
        self.transect_start_display.setToolTip("Timestamp of the first map point clicked.")
        self.transect_layout.addWidget(self.transect_start_display, 2, 1)
        self.transect_layout.addWidget(QLabel("End Time:"), 3, 0)
        self.transect_end_display = QLineEdit("Not Set")
        self.transect_end_display.setReadOnly(True)
        self.transect_end_display.setToolTip("Timestamp of the second map point clicked.")
        self.transect_layout.addWidget(self.transect_end_display, 3, 1)
        self.transect_length_label = QLabel("Length: N/A")
        self.transect_length_label.setToolTip("Calculated great-circle distance of the transect.")
        self.transect_layout.addWidget(self.transect_length_label, 4, 0, 1, 2)
        preset_length_layout = QHBoxLayout()
        self.enforce_length_checkbox = QCheckBox("Enforce Preset Length:")
        self.enforce_length_checkbox.setToolTip("If checked, transect end point selection will be constrained by the preset length.")
        self.enforce_length_checkbox.setEnabled(GEOPY_AVAILABLE)
        preset_length_layout.addWidget(self.enforce_length_checkbox)
        self.preset_length_spinbox = QDoubleSpinBox()
        self.preset_length_spinbox.setDecimals(2)
        self.preset_length_spinbox.setRange(0.1, 10000.0)
        self.preset_length_spinbox.setValue(10.0)
        self.preset_length_spinbox.setSuffix(" m")
        self.preset_length_spinbox.setEnabled(GEOPY_AVAILABLE)
        preset_length_layout.addWidget(self.preset_length_spinbox)
        preset_length_layout.addStretch()
        self.transect_layout.addLayout(preset_length_layout, 5, 0, 1, 2)
        
        # This button is now removed from view, its functionality is automated.
        self.add_transect_to_batch_button = QPushButton("Add Defined Transect to Batch")
        self.add_transect_to_batch_button.setEnabled(False)
        self.add_transect_to_batch_button.setToolTip("Adds the currently defined transect (from map clicks) to a queue for labeling.")
        self.add_transect_to_batch_button.clicked.connect(self.add_defined_transect_to_unlabeled_list)
        # self.transect_layout.addWidget(self.add_transect_to_batch_button, 6, 0) # REMOVED FROM VIEW

        self.clear_transect_button = QPushButton("Clear Single Selection")
        self.clear_transect_button.clicked.connect(self.clear_transect_definition)
        # We now use the full row 6 for the clear button
        self.transect_layout.addWidget(self.clear_transect_button, 6, 0, 1, 2)
        
        transect_io_layout = QHBoxLayout()
        self.load_transect_button = QPushButton("Load Single Transect Def.")
        self.load_transect_button.setIcon(QIcon.fromTheme("document-open"))
        self.load_transect_button.setToolTip("Load a single transect definition from a JSON file into this UI.")
        self.load_transect_button.clicked.connect(self.load_transect)
        self.load_transect_button.setEnabled(GEOPY_AVAILABLE)
        transect_io_layout.addWidget(self.load_transect_button)
        self.transect_layout.addLayout(transect_io_layout, 7, 0, 1, 2)
        self.transect_layout.setColumnStretch(1, 1)
        self.transect_group.setLayout(self.transect_layout)
        config_layout_main.addWidget(self.transect_group)

        self.batch_ops_group = QGroupBox("Batch Transect Operations")
        batch_ops_layout = QVBoxLayout()
        self.load_all_transects_button = QPushButton("Load Transect Set (from JSON)")
        self.load_all_transects_button.setIcon(QIcon.fromTheme("document-open"))
        self.load_all_transects_button.setToolTip(
            "Load a previously saved set of transect definitions from a JSON file.\n"
            "This populates the list below for batch processing."
        )
        self.load_all_transects_button.clicked.connect(self.load_transect_set_for_batch)
        self.load_all_transects_button.setEnabled(False)
        batch_ops_layout.addWidget(self.load_all_transects_button)

        self.save_all_transects_button = QPushButton("Save All Listed Transects (to JSON)")
        self.save_all_transects_button.setIcon(QIcon.fromTheme("document-save-as"))
        self.save_all_transects_button.setToolTip("Save all transects currently in the list below to a single JSON file.")
        self.save_all_transects_button.clicked.connect(self.save_all_transects)
        self.save_all_transects_button.setEnabled(False)
        batch_ops_layout.addWidget(self.save_all_transects_button)

        batch_ops_layout.addWidget(QLabel("Transects for Batch Processing:"))
        self.loaded_transects_list_widget = QListWidget()
        self.loaded_transects_list_widget.setFixedHeight(100)
        self.loaded_transects_list_widget.setToolTip("Displays transects added manually, from CSV, or from JSON for batch processing.")
        batch_ops_layout.addWidget(self.loaded_transects_list_widget)

        buffer_layout = QHBoxLayout()
        buffer_layout.addWidget(QLabel("Transect buffer:"))
        self.transect_buffer_spinbox = QDoubleSpinBox()
        self.transect_buffer_spinbox.setDecimals(2)
        self.transect_buffer_spinbox.setRange(0.10, 50.00)
        self.transect_buffer_spinbox.setValue(2.00)
        self.transect_buffer_spinbox.setSuffix(" m")
        self.transect_buffer_spinbox.setToolTip(
            "Images within this distance of each transect line will be copied.\n"
            "Increase this if the image GPS points do not fall exactly on the transect line."
        )
        buffer_layout.addWidget(self.transect_buffer_spinbox)
        buffer_layout.addStretch()
        batch_ops_layout.addLayout(buffer_layout)

        self.batch_process_button = QPushButton("Extract Images for All Listed Transects")
        self.batch_process_button.setIcon(QIcon.fromTheme("view-list-tree"))
        self.batch_process_button.setToolTip(
             "Extract and save images for ALL transects currently in the list above.\n"
             "Output will be organized into subfolders."
        )
        self.batch_process_button.clicked.connect(self.batch_process_loaded_transects)
        self.batch_process_button.setEnabled(False)
        batch_ops_layout.addWidget(self.batch_process_button)
                # New button to process the queue of unlabeled transects
        self.batch_ops_group.setLayout(batch_ops_layout)
        config_layout_main.addWidget(self.batch_ops_group)
        config_layout_main.addStretch()
        self.config_group.setLayout(config_layout_main)
        self.config_layout.addWidget(self.config_group)

        # Add input_layout and config_layout to top_layout (for left panel)
        top_layout.addLayout(self.input_layout, 1)
        top_layout.addLayout(self.config_layout, 1)

        # --- Process Layout (for left panel) ---
        process_layout = QVBoxLayout()
        self.process_button = QPushButton(" Process Data")
        self.process_button.setIcon(QIcon.fromTheme("system-run"))
        self.process_button.setStyleSheet("QPushButton { background-color: #c8e6c9; height: 40px; font-size: 16px; font-weight: bold; border: 1px solid #a5d6a7; border-radius: 4px; padding: 5px;} QPushButton:hover { background-color: #a5d6a7; } QPushButton:pressed { background-color: #81c784; } QPushButton:disabled { background-color: #e0e0e0; color: #a0a0a0; border-color: #c0c0c0; }")
        self.process_button.clicked.connect(self.start_processing)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        process_layout.addWidget(self.process_button)
        process_layout.addWidget(self.progress_bar)

        # --- Output Tabs (for right panel of splitter) ---
        self.output_tabs = QTabWidget()
        self.map_view = QWebEngineView()
        settings = self.map_view.settings()
        try:
            settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.PluginsEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.ScrollAnimatorEnabled, True)
            if hasattr(QWebEngineSettings.WebAttribute, 'DnsPrefetchEnabled'):
                settings.setAttribute(QWebEngineSettings.WebAttribute.DnsPrefetchEnabled, True)
            if hasattr(QWebEngineSettings.WebAttribute, 'FullScreenSupportEnabled'):
                settings.setAttribute(QWebEngineSettings.WebAttribute.FullScreenSupportEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.DeveloperExtrasEnabled, True)
        except Exception as e:
            self.log_message(f"Error setting some WebEngine attributes: {e}", logging.WARNING)

        self.channel = QWebChannel()
        self.map_handler = MapInteractionHandler(self)
        self.map_handler.gpsPointClicked.connect(self.handle_gps_point_click)
        self.channel.registerObject("pyHandler", self.map_handler)
        self.map_view.page().setWebChannel(self.channel)
        self.map_view.setUrl(QUrl("about:blank"))
        self.map_view.setToolTip("Map: Used for selecting Transect Start/End points when 'Define Transect' button is active.")
        self.output_tabs.addTab(self.map_view, "Map View")

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(["Identifier", "File Path", "Original Time", "UTC Time", "Lat", "Lon"])
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setSortingEnabled(True)
        self.results_table.verticalHeader().setVisible(False)
        self.output_tabs.addTab(self.results_table, "Results Table")

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        log_handler = QTextEditLogger(self.log_output)
        log_handler.setFormatter(logging.Formatter('%(asctime)s,%(msecs)03d - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logging.getLogger().addHandler(log_handler)
        logging.getLogger().setLevel(logging.INFO)
        self.output_tabs.addTab(self.log_output, "Log")
        self.output_tabs.setCurrentIndex(2) # Default to Log tab

        # --- Export Layout (for left panel) ---
        export_layout = QHBoxLayout()
        self.export_csv_button = QPushButton("Export Table CSV")
        self.export_csv_button.setIcon(QIcon.fromTheme("document-save"))
        self.export_csv_button.clicked.connect(self.export_csv)
        self.export_csv_button.setEnabled(False)
        self.save_map_button = QPushButton("Save Map HTML")
        self.save_map_button.setIcon(QIcon.fromTheme("document-save"))
        self.save_map_button.clicked.connect(self.save_map)
        self.save_map_button.setEnabled(False)
        self.geotag_images_button = QPushButton("Save Geotagged")
        self.geotag_images_button.setIcon(QIcon.fromTheme("document-save-as"))
        self.geotag_images_button.clicked.connect(self.geotag_images)
        self.geotag_images_button.setEnabled(False)
        self.extract_button = QPushButton("Extract Selected Images...")
        self.extract_button.setIcon(QIcon.fromTheme("edit-copy"))
        self.extract_button.setToolTip("Copy image files corresponding to the rows currently SELECTED in the Results Table.")
        self.extract_button.clicked.connect(self.extract_selected_images)
        self.extract_button.setEnabled(False)
        export_layout.addStretch()
        export_layout.addWidget(self.export_csv_button)
        export_layout.addWidget(self.save_map_button)
        export_layout.addWidget(self.geotag_images_button)
        export_layout.addWidget(self.extract_button)

        # --- Create Left Panel Widget ---
        left_panel_widget = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_widget)
        left_panel_layout.addLayout(top_layout) # Contains input_layout and config_layout
        left_panel_layout.addLayout(process_layout)
        left_panel_layout.addLayout(export_layout)
        left_panel_layout.addStretch(1) # Add stretch to push controls up

        # --- Create Splitter ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel_widget)
        splitter.addWidget(self.output_tabs)

        # Set initial sizes and stretch factors for the splitter
        # Give more relative size and stretch to the map/output panel
        splitter.setSizes([450, 850]) # Adjust these values as needed for a good default
        splitter.setStretchFactor(0, 0) # Left panel less stretchy
        splitter.setStretchFactor(1, 1) # Right panel (map/output) more stretchy

        # Add splitter to the main layout
        self.main_layout.addWidget(splitter)

        self.log_message("GeoTagger Application started.", logging.INFO)
        if not GEOPY_AVAILABLE:
            self.log_message("WARNING: 'geopy' library not found. Transect length calculation, preset length enforcement, and transect save/load (with length) are disabled.", logging.WARNING)
        self.update_transect_ui()
        self._update_batch_processing_ui_states()
        self.enforce_length_checkbox.stateChanged.connect(self.update_preset_length_circle_on_map)
        self.preset_length_spinbox.valueChanged.connect(self.update_preset_length_circle_on_map)

    def _extract_exif_gps(self, image_path):
        """Return (lat, lon, gps_utc_datetime). Any/all may be None on failure.
        GPS date/time tags are UTC per EXIF spec, so we prefer them over DateTimeOriginal."""
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            img = Image.open(image_path)
            exif = img._getexif()
            if not exif:
                return None, None, None
            gps_info = None
            for tag_id, value in exif.items():
                if TAGS.get(tag_id) == 'GPSInfo':
                    gps_info = value
                    break
            if not gps_info or 2 not in gps_info or 4 not in gps_info:
                return None, None, None

            def dms_to_decimal(dms, ref):
                deg = float(dms[0]) + float(dms[1]) / 60.0 + float(dms[2]) / 3600.0
                if ref in ('S', 'W'):
                    deg = -deg
                return deg

            lat = dms_to_decimal(gps_info[2], gps_info.get(1, 'N'))
            lon = dms_to_decimal(gps_info[4], gps_info.get(3, 'E'))

            gps_dt = None
            gps_date = gps_info.get(29)   # GPSDateStamp 'YYYY:MM:DD'
            gps_time = gps_info.get(7)    # GPSTimeStamp (h, m, s)
            if gps_date and gps_time:
                try:
                    y, mo, d = (int(p) for p in str(gps_date).split(':'))
                    gps_dt = datetime(y, mo, d,
                                      int(gps_time[0]), int(gps_time[1]), int(float(gps_time[2])),
                                      tzinfo=pytz.utc)
                except Exception:
                    gps_dt = None
            return lat, lon, gps_dt
        except Exception as e:
            self.log_message(f"EXIF read failed for '{image_path}': {e}", logging.DEBUG)
            return None, None, None


    def load_images_with_embedded_gps(self):
        """Load a folder of images with EXIF GPS. Builds the GPS track and results
        dataframe directly from EXIF, bypassing the GPS-file and time-sync workflow."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.show_error("Cannot load new data while processing is active.")
            return

        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Folder of GPS-Tagged Images", self.media_path or "")
        if not dir_path:
            return

        self.log_message(f"Loading EXIF-GPS images from: {dir_path}", logging.INFO)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            media_items = engine.get_image_files_and_times(dir_path)
            if not media_items:
                self.show_error("No compatible images found in the selected folder.")
                return

            media_tz = pytz.timezone(self.current_timezone_str)
            gps_records, results_records = [], []
            skipped_no_gps = 0

            for item in media_items:
                file_path = item.get('file_path')
                identifier = item.get('identifier') or (os.path.basename(file_path) if file_path else None)
                naive_time = item.get('image_time')
                if not file_path or not identifier:
                    continue

                lat, lon, gps_dt = self._extract_exif_gps(file_path)
                if lat is None or lon is None:
                    skipped_no_gps += 1
                    continue

                # Prefer EXIF GPS time (always UTC). Fall back to DateTimeOriginal + Data Timezone.
                if gps_dt is not None:
                    utc_dt = gps_dt
                    orig_display = utc_dt.strftime('%Y-%m-%d %H:%M:%S') + " (EXIF GPS UTC)"
                elif naive_time is not None:
                    try:
                        local_dt = media_tz.localize(naive_time, is_dst=None)
                    except Exception:
                        local_dt = pytz.utc.localize(naive_time)
                    utc_dt = local_dt.astimezone(pytz.utc)
                    orig_display = naive_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + f" ({self.current_timezone_str})"
                else:
                    # No timestamp at all — fabricate one from file order so the workflow still functions,
                    # but this image won't be usefully time-filterable.
                    utc_dt = pytz.utc.localize(datetime(1970, 1, 1)) + timedelta(seconds=len(gps_records))
                    orig_display = "N/A (no timestamp)"

                gps_records.append({'timestamp': utc_dt, 'latitude': lat, 'longitude': lon})
                results_records.append({
                    'Identifier': identifier,
                    'File Path': file_path,
                    'Original Timestamp': orig_display,
                    'Corrected Timestamp (UTC)': utc_dt,
                    'Latitude': lat,
                    'Longitude': lon,
                })

            if not gps_records:
                self.show_error(f"None of the {len(media_items)} images contained EXIF GPS data.")
                return

            # Build the GPS DataFrame in the same shape as processed_gps_df
            gps_df = pd.DataFrame(gps_records).set_index('timestamp').sort_index()
            # De-duplicate identical timestamps (rare but possible with burst-mode shots)
            gps_df = gps_df[~gps_df.index.duplicated(keep='first')]

            results_df = pd.DataFrame(results_records)
            results_df['Corrected Timestamp (UTC)'] = pd.to_datetime(
                results_df['Corrected Timestamp (UTC)'], utc=True)

            # Set application state for the rest of the workflow
            self.gps_file_path = None
            self.gps_path_display.setText(f"<from EXIF: {len(gps_df)} points>")
            self.media_path = dir_path
            self.media_type = 'images'
            self.media_path_display.setText(os.path.basename(dir_path))
            self.original_media_list = media_items
            self.preliminary_media_list = media_items
            self.preliminary_gps_df = gps_df.copy()
            self.csv_mapping_group.setVisible(False)
            self.saved_transects = []
            self._populate_loaded_transects_list()
            self.geotag_output_dir_base = f"geotagged_{os.path.basename(dir_path)}"

            # Re-use the normal finalisation path — builds k-d tree, populates date filter,
            # results table, map, and enables transect / batch operations.
            self.selected_filter_date = None
            self.filtered_gps_by_date_df = gps_df.copy()
            self.processing_finished(results_df, media_items, gps_df)

            # Direct EXIF loading bypasses WorkerThread.map_ready, so explicitly force
            # a map refresh after the UI event loop has caught up. This prevents a
            # blank QWebEngine map after loading GPS-tagged images.
            self.regenerate_map_display()
            QTimer.singleShot(250, self.regenerate_map_display)

            msg = f"Loaded {len(gps_df)} GPS-tagged images."
            if skipped_no_gps:
                msg += f"\n{skipped_no_gps} image(s) without EXIF GPS were skipped."
            self.log_message(msg, logging.INFO)
            QMessageBox.information(self, "Images Loaded",
                                    msg + "\n\nYou can now define transects, filter by date, "
                                          "and use all batch processing features.")

        except Exception as e:
            self.log_message(f"Error loading EXIF-GPS images: {e}", logging.ERROR)
            self.log_message(traceback.format_exc(), logging.DEBUG)
            self.show_error(f"Failed to load EXIF-GPS images:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

    @property
    def transects_json_dir(self):
        """
        Dynamically determines the correct directory path for saving/loading transect JSON files.
        Prioritizes a location relative to the media path, then the GPS path, then the current working directory.
        """
        base_path = os.getcwd() # Default
        if self.media_path and os.path.isdir(self.media_path):
            base_path = self.media_path
        elif self.gps_file_path and os.path.isfile(self.gps_file_path):
            base_path = os.path.dirname(self.gps_file_path)
        
        # self.transects_json_dir_name is defined as "transects_json" in __init__
        return os.path.join(base_path, self.transects_json_dir_name)

    def _finalize_transect_save(self, transect_name):
        """
        This method is called via a QTimer to finalize saving a transect.
        It handles file I/O and UI updates in a stable state after the input dialog has closed.
        """
        self.log_message(f"Finalizing save for transect: '{transect_name}'")

        if self.media_path and os.path.isdir(self.media_path):
            base_output_dir_for_transects = self.media_path
        elif self.gps_file_path and os.path.isfile(self.gps_file_path):
            base_output_dir_for_transects = os.path.dirname(self.gps_file_path)
        else:
            base_output_dir_for_transects = os.getcwd()
        transects_output_parent_dir = os.path.join(base_output_dir_for_transects, "transects_output")

        specific_transect_output_dir = os.path.join(transects_output_parent_dir, transect_name)
        os.makedirs(specific_transect_output_dir, exist_ok=True)

        start_utc = min(self.transect_start_time.astimezone(pytz.utc), self.transect_end_time.astimezone(pytz.utc))
        end_utc = max(self.transect_start_time.astimezone(pytz.utc), self.transect_end_time.astimezone(pytz.utc))
        length_m = self.transect_length_meters

        mask = (self.results_df['Corrected Timestamp (UTC)'].notna() &
                (self.results_df['Corrected Timestamp (UTC)'] >= start_utc) &
                (self.results_df['Corrected Timestamp (UTC)'] <= end_utc))
        images_in_transect_df = self.results_df[mask]

        copy_ok_count = 0
        if not images_in_transect_df.empty:
            for _, row in images_in_transect_df.iterrows():
                source_path = self.identifier_to_path_map.get(row.get('Identifier'))
                if source_path and os.path.exists(source_path):
                    try:
                        shutil.copy2(source_path, specific_transect_output_dir)
                        copy_ok_count += 1
                    except Exception as copy_e:
                        self.log_message(f"Error copying image for transect '{transect_name}': {copy_e}", logging.ERROR)

        self.log_message(f"Saved {copy_ok_count} images for transect '{transect_name}'.")

        active_gps_df = self.get_active_gps_df()
        try:
            start_point = active_gps_df.loc[self.transect_start_time]
            end_point = active_gps_df.loc[self.transect_end_time]
            final_transect_def = {
                "name": transect_name,
                "start_lat": start_point['latitude'], "start_lon": start_point['longitude'],
                "end_lat": end_point['latitude'], "end_lon": end_point['longitude'],
                "length": length_m,
            }
            self.saved_transects.append(final_transect_def)
            self._populate_loaded_transects_list()
            self._update_batch_processing_ui_states()

            if self.map_view and self.map_view.page():
                # **THIS IS THE CRITICAL FIX for the JS call**
                # Use json.dumps to safely escape the name for JavaScript
                safe_name_json = json.dumps(transect_name)
                js_call = f"drawSavedTransectLine({start_point['latitude']}, {start_point['longitude']}, {end_point['latitude']}, {end_point['longitude']}, {safe_name_json});"
                self.map_view.page().runJavaScript(js_call)

        except KeyError:
            self.log_message(f"Could not find start/end times in GPS data for transect '{transect_name}'. Could not save its definition.", logging.ERROR)

        # Always clear the UI for the next one
        self.clear_transect_definition()
        self.update_transect_ui()

    def enter_transect_mode(self):
        self.log_message("Switching to Transect-only mode.", logging.INFO)
        # hide data-loading panels
        if hasattr(self, 'input_group'): self.input_group.hide()
        if hasattr(self, 'csv_mapping_group'): self.csv_mapping_group.hide()
        if hasattr(self, 'transect_csv_load_group'): self.transect_csv_load_group.hide()
        if hasattr(self, 'transect_csv_mapping_group'): self.transect_csv_mapping_group.hide()

        # These individual controls are inside input_group but good to hide explicitly if input_group might not always exist
        if hasattr(self, 'media_load_button'): self.media_load_button.hide()
        if hasattr(self, 'gps_load_button'): self.gps_load_button.hide()
        if hasattr(self, 'media_timezone_input'): self.media_timezone_input.hide()
        if hasattr(self, 'media_timezone_label'): self.media_timezone_label.hide()
        if hasattr(self, 'date_filter_label'): self.date_filter_label.hide()
        if hasattr(self, 'date_filter_combo'): self.date_filter_combo.hide()

        # hide sync
        widgets_to_hide = []
        if hasattr(self, 'sync_point_radio'): widgets_to_hide.append(self.sync_point_radio)
        if hasattr(self, 'sync_manual_radio'): widgets_to_hide.append(self.sync_manual_radio)
        if hasattr(self, 'time_sync_group'): widgets_to_hide.append(self.time_sync_group)
        # MODIFICATION: DO NOT hide self.batch_ops_group
        # if hasattr(self, 'batch_ops_group'): widgets_to_hide.append(self.batch_ops_group)

        for w in widgets_to_hide:
            w.hide()

        # Show batch_ops_group if it exists
        if hasattr(self, 'batch_ops_group'):
            self.batch_ops_group.show()

        # hide the process button and progress bar
        if hasattr(self, 'process_button'): self.process_button.hide()
        if hasattr(self, 'progress_bar'): self.progress_bar.hide()

        # hide the export buttons
        if hasattr(self, 'export_csv_button'): self.export_csv_button.hide()
        if hasattr(self, 'save_map_button'): self.save_map_button.hide()
        if hasattr(self, 'geotag_images_button'): self.geotag_images_button.hide()
        if hasattr(self, 'extract_button'): self.extract_button.hide()

        if hasattr(self, 'output_tabs'):
            for i in range(self.output_tabs.count()):
                if self.output_tabs.tabText(i) != "Map View":
                    if self.output_tabs.widget(i):
                        self.output_tabs.widget(i).hide()
                    self.output_tabs.setTabEnabled(i, False)
                else:
                    if self.output_tabs.widget(i):
                        self.output_tabs.widget(i).show()
                    self.output_tabs.setTabEnabled(i, True)
                    self.output_tabs.setCurrentIndex(i)

        if hasattr(self, 'config_group'): self.config_group.show()
        if hasattr(self, 'transect_group'): self.transect_group.show()

        self.adjustSize()

    @pyqtSlot(object, list, object)
    def processing_finished(self, results_df, original_media_list, processed_gps_df):
        self.log_message("Processing finished successfully. Updating UI.", logging.INFO)
        self.results_df = results_df
        self.original_media_list = original_media_list
        self.processed_gps_df = processed_gps_df
        # --- ADD THIS BLOCK TO BUILD THE SPATIAL INDEX ---
        self.gps_kdtree = None # Reset any previous index
        if SCIPY_AVAILABLE and self.processed_gps_df is not None and not self.processed_gps_df.empty:
            self.log_message("Building spatial index (k-d tree) for GPS track for fast lookups...", logging.INFO)
            try:
                # Store a clean DataFrame with only valid lat/lon for indexing
                self.gps_kdtree_df_map = self.processed_gps_df[['latitude', 'longitude']].dropna().copy()
                if not self.gps_kdtree_df_map.empty:
                    self.gps_coordinates_for_kdtree = self.gps_kdtree_df_map.values
                    self.gps_kdtree = cKDTree(self.gps_coordinates_for_kdtree)
                    self.log_message(f"Spatial index built successfully with {len(self.gps_kdtree_df_map)} points.", logging.INFO)
                else:
                    self.log_message("No valid coordinates found in GPS data to build spatial index.", logging.WARNING)
            except Exception as e:
                self.log_message(f"Failed to build spatial index: {e}", logging.ERROR)
                self.gps_kdtree = None
        # --- END OF ADDED BLOCK ---

        self.preliminary_media_list = original_media_list

        self.identifier_to_path_map = {}
        if self.original_media_list:
            try:
                self.identifier_to_path_map = {item['identifier']: item['file_path'] for item in self.original_media_list if 'identifier' in item and 'file_path' in item and item.get('file_path')}
            except Exception as e:
                self.log_message(f"Error creating identifier-to-path map: {e}", logging.ERROR)

        self.date_filter_combo.blockSignals(True)
        current_date_filter_text = self.date_filter_combo.currentText()
        self.date_filter_combo.clear()
        self.date_filter_combo.addItem("Show All Dates")
        self.available_dates = []

        base_gps_valid_for_date_filter = False
        if self.processed_gps_df is not None and \
           isinstance(self.processed_gps_df, pd.DataFrame) and \
           not self.processed_gps_df.empty and \
           isinstance(self.processed_gps_df.index, pd.DatetimeIndex) and \
           self.processed_gps_df.index.tz is not None and \
           self.processed_gps_df.index.tz.utcoffset(self.processed_gps_df.index.min()) == timedelta(0):
            base_gps_valid_for_date_filter = True
            self.log_message(f"Stored processed GPS data ({len(self.processed_gps_df)} points). Populating date filter.", logging.INFO)
            try:
                target_tz = pytz.timezone(self.current_timezone_str)
                local_times = self.processed_gps_df.index.tz_convert(target_tz)
                unique_dates = sorted(list(set(dt.date() for dt in local_times)))
                self.available_dates = unique_dates
                for date_obj in self.available_dates:
                    self.date_filter_combo.addItem(date_obj.strftime("%Y-%m-%d"))
                self.date_filter_combo.setEnabled(True)
                # Enable transect CSV loading now that GPS is processed
                if hasattr(self, 'load_transects_csv_button'):
                    self.load_transects_csv_button.setEnabled(True)


                idx = self.date_filter_combo.findText(current_date_filter_text)
                if idx != -1:
                    self.date_filter_combo.setCurrentIndex(idx)
                    if idx > 0:
                         self.on_date_filter_changed(current_date_filter_text)
                else:
                    self.date_filter_combo.setCurrentIndex(0)
                    self.selected_filter_date = None
                    self.filtered_gps_by_date_df = self.processed_gps_df.copy() if self.processed_gps_df is not None else None
                    self.regenerate_map_display()
            except pytz.UnknownTimeZoneError:
                self.log_message(f"Cannot populate date filter: Unknown timezone '{self.current_timezone_str}'.", logging.ERROR)
                self.date_filter_combo.setEnabled(False)
                self.date_filter_combo.setCurrentIndex(0)
                if hasattr(self, 'load_transects_csv_button'): self.load_transects_csv_button.setEnabled(False)
                self.show_error(f"Date filter cannot be populated due to invalid timezone: {self.current_timezone_str}. Please select a valid one.")
                self.regenerate_map_display()
            except Exception as e:
                self.log_message(f"Error populating date filter: {e}", logging.ERROR)
                self.log_message(traceback.format_exc(), logging.DEBUG)
                self.date_filter_combo.setEnabled(False)
                self.date_filter_combo.setCurrentIndex(0)
                if hasattr(self, 'load_transects_csv_button'): self.load_transects_csv_button.setEnabled(False)
                self.regenerate_map_display()
        else:
            self.log_message("Processed GPS data not suitable for date filtering.", logging.WARNING)
            self.date_filter_combo.setEnabled(False)
            self.date_filter_combo.setCurrentIndex(0)
            if hasattr(self, 'load_transects_csv_button'): self.load_transects_csv_button.setEnabled(False)
            self.selected_filter_date = None
            self.filtered_gps_by_date_df = None
            self.regenerate_map_display()

        self.date_filter_combo.blockSignals(False)
        self.populate_results_table(self.results_df)
        self.export_csv_button.setEnabled(self.results_df is not None and not self.results_df.empty)
        self.clear_transect_definition()

        can_geotag = False
        if ENGINE_AVAILABLE and self.results_df is not None and not self.results_df.empty and self.original_media_list:
            req = ['Identifier', 'File Path', 'Latitude', 'Longitude']
            if all(c in self.results_df.columns for c in req):
                try:
                    lat_valid = pd.to_numeric(self.results_df['Latitude'], errors='coerce').notna()
                    lon_valid = pd.to_numeric(self.results_df['Longitude'], errors='coerce').notna()
                    path_valid = self.results_df['File Path'].notna() & \
                                 (self.results_df['File Path'].astype(str).str.strip() != 'N/A') & \
                                 (self.results_df['File Path'].astype(str).str.strip() != '')
                    can_geotag = not self.results_df[path_valid & lat_valid & lon_valid].empty
                except Exception as e:
                    self.log_message(f"Error checking geotag eligibility: {e}", logging.WARNING)
                    can_geotag = False
        self.geotag_images_button.setEnabled(can_geotag)
        self.reset_progress_bar_color()
        self.progress_bar.setValue(100)
        self._update_batch_processing_ui_states()
        self.update_sync_method_ui()

    @pyqtSlot(bool)
    def _on_visual_sync_toggled(self, checked):
        """When the user clicks the Visual‐Sync radio, show the map:
        - if we've already processed, re‐show the processed map,
        - otherwise fall back to preliminary parse/map."""
        if not checked:
            return

        # 1) If we've processed the track, just re‐display it
        if getattr(self, 'processed_gps_df', None) is not None and not self.processed_gps_df.empty:
            # this will regenerate using both GPS and any image markers, etc.
            self.regenerate_map_display()
            return

        # 2) Otherwise (no processed data yet), show the preliminary GPS‐only map
        if getattr(self, 'preliminary_gps_df', None) is not None and not self.preliminary_gps_df.empty:
            self._run_preliminary_gps_parse_and_map()



    @pyqtSlot(str, bool, bool)
    def handle_gps_point_click(self, timestamp_str_iso, ctrl_pressed, shift_pressed):
        # --- Visual Sync Handling (This part is unchanged) ---
        if self.sync_visual_radio.isChecked() and self.sync_image_info is not None:
            self.log_message(f"Map GPS Click Received for Visual Sync: TS='{timestamp_str_iso}'", logging.DEBUG)
            active_gps_df = self.preliminary_gps_df
            if active_gps_df is None or active_gps_df.empty:
                self.log_message("Ignoring map click for sync: No preliminary GPS data.", logging.WARNING)
                QMessageBox.warning(self, "No GPS Data", "Cannot select a sync point. GPS data has not been loaded and mapped yet.")
                return
            try:
                clicked_dt_utc = datetime.fromisoformat(timestamp_str_iso.replace('Z', '+00:00')).astimezone(pytz.utc)
                nearest_idx = active_gps_df.index.get_indexer([clicked_dt_utc], method='nearest', tolerance=timedelta(seconds=1))[0]
                if nearest_idx != -1:
                    actual_timestamp = active_gps_df.index[nearest_idx]
                    point_data = active_gps_df.iloc[nearest_idx]
                    self.sync_gps_point_info = {
                        'timestamp': actual_timestamp,
                        'latitude': point_data['latitude'],
                        'longitude': point_data['longitude']
                    }
                    self.sync_gps_point_time_display.setText(actual_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + " Z")
                    self.sync_gps_point_coords_display.setText(f"Lat: {point_data['latitude']:.6f}, Lon: {point_data['longitude']:.6f}")
                    self.visual_sync_status_label.setText("Status: Sync point selected. Ready to process.")
                else:
                    self.show_error("Could not find the clicked GPS point in the data. Please try clicking closer to a point.")
            except Exception as e:
                self.show_error(f"An error occurred while selecting the GPS sync point: {e}")
            return # End of visual sync handling

        # --- Transect Definition Handling ---
        self.log_message(f"Map GPS Click Received: TS='{timestamp_str_iso}', Mode='{self.transect_definition_mode}'", logging.DEBUG)
        active_gps_df = self.get_active_gps_df()
        if active_gps_df.empty:
            self.log_message("Ignoring map click: No active GPS data for transect selection.", logging.WARNING)
            QMessageBox.warning(self, "No GPS Data", "No GPS data currently displayed on map to select transect points from.")
            self.update_preset_length_circle_on_map()
            return
        if self.transect_definition_mode not in ['selecting_start', 'selecting_end']:
            self.log_message("Ignoring map click: Not defining transect.", logging.DEBUG)
            return

        try:
            dt_utc_clicked = datetime.fromisoformat(timestamp_str_iso.replace('Z', '+00:00')).astimezone(pytz.utc)
        except ValueError:
            self.log_message(f"Error parsing clicked time: '{timestamp_str_iso}'", logging.ERROR)
            self.update_preset_length_circle_on_map()
            return

        try:
            nearest_idx = active_gps_df.index.get_indexer([dt_utc_clicked], method='nearest', tolerance=timedelta(seconds=1))[0]
            if nearest_idx == -1:
                self.log_message(f"Clicked time {dt_utc_clicked} too far from active track.", logging.WARNING)
                QMessageBox.warning(self, "Point Not Found", "Click closer to a blue GPS point on the currently displayed track.")
                self.update_preset_length_circle_on_map()
                return
            actual_timestamp_from_click = active_gps_df.index[nearest_idx]
            self.log_message(f"Matched map click to actual GPS time: {actual_timestamp_from_click}", logging.DEBUG)
        except Exception as e:
            self.log_message(f"Error finding nearest GPS time: {e}", logging.ERROR)
            self.log_message(traceback.format_exc(), logging.DEBUG)
            self.show_error("Error matching clicked point to GPS track.")
            self.update_preset_length_circle_on_map()
            return


        if self.transect_definition_mode == 'selecting_start':
            self.transect_start_time = actual_timestamp_from_click
            self.transect_end_time = None
            self.transect_length_meters = None
            self.transect_definition_mode = 'selecting_end'
            self.log_message(f"Transect START set: {self.transect_start_time}")
            self.highlight_map_range(self.transect_start_time, self.transect_start_time)

        # ======================================================================
        # ========= START OF THE MODIFIED CODE BLOCK FROM PREVIOUS REPLY =========
        # ======================================================================
        elif self.transect_definition_mode == 'selecting_end':
            final_end_timestamp = actual_timestamp_from_click

            if self.enforce_length_checkbox.isChecked() and GEOPY_AVAILABLE:
                # --- This is where your preset length logic goes ---
                # (I've collapsed it for clarity, but it should be here)
                preset_len_m = self.preset_length_spinbox.value()
                self.log_message(f"Enforcing preset length: {preset_len_m}m", logging.DEBUG)
                try:
                    start_point_data = active_gps_df.loc[self.transect_start_time]
                    start_lat, start_lon = start_point_data['latitude'], start_point_data['longitude']
                    clicked_track_point_data = active_gps_df.loc[actual_timestamp_from_click]
                    click_dir_lat, click_dir_lon = clicked_track_point_data['latitude'], clicked_track_point_data['longitude']

                    if geodesic((start_lat, start_lon), (click_dir_lat, click_dir_lon)).meters < 0.1:
                        QMessageBox.information(self, "Select Direction", "Please click further away from the start point to indicate the transect direction.")
                        self.highlight_map_range(self.transect_start_time, self.transect_start_time)
                        self.update_transect_ui()
                        return

                    from geographiclib.geodesic import Geodesic as GeographicLibGeodesic
                    geod_calc = GeographicLibGeodesic.WGS84
                    inverse_result = geod_calc.Inverse(start_lat, start_lon, click_dir_lat, click_dir_lon)
                    bearing = inverse_result['azi1']
                    direct_result = geod_calc.Direct(start_lat, start_lon, bearing, preset_len_m)
                    ideal_end_lat, ideal_end_lon = direct_result['lat2'], direct_result['lon2']

                    min_dist_to_ideal = float('inf')
                    snapped_end_time = None
                    lat_col_idx = active_gps_df.columns.get_loc('latitude')
                    lon_col_idx = active_gps_df.columns.get_loc('longitude')

                    for i in range(len(active_gps_df)):
                        track_point_time = active_gps_df.index[i]
                        track_lat = active_gps_df.iat[i, lat_col_idx]
                        track_lon = active_gps_df.iat[i, lon_col_idx]
                        if pd.isna(track_lat) or pd.isna(track_lon):
                            continue
                        dist = geodesic((ideal_end_lat, ideal_end_lon), (track_lat, track_lon)).meters
                        if dist < min_dist_to_ideal:
                            min_dist_to_ideal = dist
                            snapped_end_time = track_point_time
                    
                    if snapped_end_time is not None:
                        final_end_timestamp = snapped_end_time

                except Exception as e:
                    self.log_message(f"Error during preset length calculation: {e}", logging.ERROR)
                    self.show_error(f"Error during preset length calculation: {e}\nUsing direct click.")
            
            if self.transect_start_time and final_end_timestamp == self.transect_start_time:
                QMessageBox.information(self, "Select Different Point", "End point cannot be the same as start point.")
                self.update_transect_ui()
                return

            # Capture the current values.
            start_time = self.transect_start_time
            end_time = final_end_timestamp

            # Immediately reset the definition mode.
            self.transect_definition_mode = None
            if self.define_transect_button.isChecked():
                self.define_transect_button.setChecked(False)

            # Create a NEW nested function to contain all the follow-up work.
            def process_and_save_transect():
                # This code will run AFTER handle_gps_point_click has finished.
                self.transect_start_time = start_time
                self.transect_end_time = end_time

                # Now calculate length
                self.transect_length_meters = self.calculate_transect_length(start_time, end_time)
                length_m = self.transect_length_meters
                self.log_message(f"Transect END set: {end_time}. Length: {length_m:.2f} m" if length_m is not None else "N/A")

                # Provide immediate visual feedback with a temporary line
                try:
                    start_point = active_gps_df.loc[start_time]
                    end_point = active_gps_df.loc[end_time]
                    js = f"drawTransectLine({start_point['latitude']}, {start_point['longitude']}, {end_point['latitude']}, {end_point['longitude']});"
                    self.map_view.page().runJavaScript(js)
                except Exception as e:
                    self.log_message(f"Could not draw temporary transect line: {e}", logging.WARNING)
                    self.clear_transect_definition()
                    return

                # --- START OF MODIFICATION ---

                # 1. Count the images within the transect times
                image_count = 0
                if self.results_df is not None and not self.results_df.empty:
                    # Ensure timestamps are timezone-aware UTC for comparison
                    start_utc = start_time.astimezone(pytz.utc)
                    end_utc = end_time.astimezone(pytz.utc)
                    
                    # Make sure the start is always before the end for filtering
                    filter_start = min(start_utc, end_utc)
                    filter_end = max(start_utc, end_utc)
                    
                    try:
                        mask = (
                            self.results_df['Corrected Timestamp (UTC)'].notna() &
                            (self.results_df['Corrected Timestamp (UTC)'] >= filter_start) &
                            (self.results_df['Corrected Timestamp (UTC)'] <= filter_end)
                        )
                        image_count = mask.sum()
                        self.log_message(f"Found {image_count} images in the defined transect.", logging.INFO)
                    except Exception as e:
                        self.log_message(f"Error while counting images in transect: {e}", logging.WARNING)
                        image_count = "Error"

                # 2. Prepare a suggested name and the dialog prompt text
                s_time_str = start_time.astimezone(pytz.utc).strftime('%Y%m%d_%H%M%S')
                length_str = f"{length_m:.0f}m" if length_m is not None else "lenNA"
                suggested_name = f"Transect_{s_time_str}_{length_str}"
                
                prompt_text = (f"Found {image_count} images for this transect.\n\n"
                               "Enter a name for the transect:")
                
                # 3. Prompt the user with the updated text
                transect_name, ok = QInputDialog.getText(self, "Label Transect",
                                                         prompt_text,
                                                         QLineEdit.EchoMode.Normal,
                                                         suggested_name)

                # 4. Check if the user clicked OK and provided a name
                if ok and transect_name.strip():
                    safe_transect_name = "".join(c for c in transect_name if c.isalnum() or c in (' ', '_', '-')).strip()
                    if not safe_transect_name:
                        safe_transect_name = suggested_name
                    
                    self._autosave_defined_transect(start_time, end_time, length_m, safe_transect_name)
                else:
                    self.log_message("User cancelled transect naming. Transect not saved.", logging.INFO)
                    self.clear_transect_definition()

                # --- END OF MODIFICATION ---

            # Schedule the nested function to run as soon as this event handler is finished.
            QTimer.singleShot(0, process_and_save_transect)

                # --- END OF MODIFICATION ---

        # ======================================================================
        # ========= END OF THE MODIFIED CODE BLOCK FROM PREVIOUS REPLY ===========
        # ======================================================================

        # This line is executed *after* the if/elif block, for example,
        # when the mode is 'selecting_start'.
        self.update_preset_length_circle_on_map()

    def _label_and_process_defined_transect(self):
        if self.transect_start_time is None or self.transect_end_time is None:
            self.show_error("Internal error: No transect is defined to be labeled and saved.")
            return

        if self.results_df is None or self.results_df.empty:
            self.show_error("No georeferenced image data is available to filter for the transect.")
            self.clear_transect_definition()
            return

        # Prepare details for the dialog
        start_utc = min(self.transect_start_time.astimezone(pytz.utc), self.transect_end_time.astimezone(pytz.utc))
        length_m = self.transect_length_meters
        mask = (self.results_df['Corrected Timestamp (UTC)'].notna() &
                (self.results_df['Corrected Timestamp (UTC)'] >= start_utc) &
                (self.results_df['Corrected Timestamp (UTC)'] <= max(self.transect_start_time.astimezone(pytz.utc), self.transect_end_time.astimezone(pytz.utc))))
        image_count = len(self.results_df[mask])

        if self.media_path and os.path.isdir(self.media_path):
            base_output_dir_for_transects = self.media_path
        elif self.gps_file_path and os.path.isfile(self.gps_file_path):
            base_output_dir_for_transects = os.path.dirname(self.gps_file_path)
        else:
            base_output_dir_for_transects = os.getcwd()
        transects_output_parent_dir = os.path.join(base_output_dir_for_transects, "transects_output")

        s_time_str = start_utc.strftime('%Y%m%d_%H%M%S')
        length_str = f"{length_m:.0f}m" if length_m is not None else "lenNA"
        suggested_transect_name = f"Transect_{s_time_str}_{length_str}"

        # Get name from user
        transect_name_input, ok = QInputDialog.getText(self, "Label Transect",
                                                     f"Found {image_count} images for this transect.\n"
                                                     f"Enter a name to save the images and definition.\n\n"
                                                     f"Output subfolder will be inside:\n'{transects_output_parent_dir}'",
                                                     QLineEdit.EchoMode.Normal, suggested_transect_name)

        if ok and transect_name_input.strip():
            safe_transect_name = "".join(c for c in transect_name_input if c.isalnum() or c in (' ', '_', '-')).strip()
            if not safe_transect_name:
                safe_transect_name = suggested_transect_name

            # **THIS IS THE CRITICAL FIX for UI timing**
            # Defer the rest of the processing to avoid a crash
            QTimer.singleShot(0, lambda: self._finalize_transect_save(safe_transect_name))
        else:
            self.log_message("User cancelled or provided no name. Transect was not saved.")
            # If cancelled, still clear the yellow line from the map
            self.clear_transect_definition()
            self.update_transect_ui()

    def _update_batch_processing_ui_states(self):
        main_data_ready = (self.processed_gps_df is not None and not self.processed_gps_df.empty and
                           self.results_df is not None)
        media_loaded = bool(self.media_path)
        identifier_map_populated = bool(self.identifier_to_path_map)
        transects_available = bool(self.saved_transects)

        can_load_set = main_data_ready and media_loaded
        self.load_all_transects_button.setEnabled(can_load_set)
        can_batch_process = main_data_ready and media_loaded and identifier_map_populated and transects_available
        self.batch_process_button.setEnabled(can_batch_process)
        self.save_all_transects_button.setEnabled(transects_available)

        # Enable transect CSV loading if main GPS data is processed
        if hasattr(self, 'load_transects_csv_button'):
            self.load_transects_csv_button.setEnabled(self.processed_gps_df is not None and not self.processed_gps_df.empty)

        self.log_message(f"Batch UI State: LoadSetJSON={self.load_all_transects_button.isEnabled()}, "
                         f"BatchProcess={self.batch_process_button.isEnabled()}, "
                         f"SaveAllJSON={self.save_all_transects_button.isEnabled()}, "
                         f"LoadTransectCSV_BtnEnabled={getattr(self, 'load_transects_csv_button', None) is not None and self.load_transects_csv_button.isEnabled()}",
                         logging.DEBUG)

    def _populate_loaded_transects_list(self):
        self.loaded_transects_list_widget.clear()
        if not self.saved_transects:
            self.loaded_transects_list_widget.addItem("No transects defined or loaded for this session.")
            self.loaded_transects_list_widget.setEnabled(False)
            return

        self.loaded_transects_list_widget.setEnabled(True)
        for transect_info in self.saved_transects:
            name = transect_info.get("name", "Unnamed Transect")
            try:
                # Now displays coordinates instead of times
                start_lat = transect_info.get("start_lat")
                start_lon = transect_info.get("start_lon")
                length_m = transect_info.get("length")

                if start_lat is None or start_lon is None:
                    display_text = f"{name} (Coordinates Missing)"
                else:
                    length_str = f"{length_m:.1f}m" if length_m is not None else "N/A"
                    display_text = f"{name} ({length_str}) @ {start_lat:.4f}, {start_lon:.4f}"

                item = QListWidgetItem(display_text)
                item.setData(Qt.ItemDataRole.UserRole, transect_info)
                self.loaded_transects_list_widget.addItem(item)
            except Exception as e:
                self.log_message(f"Error formatting transect '{name}' for list display: {e}", logging.WARNING)
                self.loaded_transects_list_widget.addItem(f"Error displaying: {name}")
        self.loaded_transects_list_widget.scrollToBottom()

    def load_transect_set_for_batch(self):
        self.log_message("Attempting to load transect set for batch processing...")
        if not (self.processed_gps_df is not None and self.results_df is not None and self.media_path):
            self.show_error("Cannot load transect set: Main data not processed or image directory not set.")
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Transect Set (JSON)", self.transects_json_dir, "JSON files (*.json)"
        )
        if not filepath:
            self.log_message("Transect set loading cancelled by user.", logging.INFO)
            return

        try:
            with open(filepath, 'r') as f:
                loaded_data = json.load(f)

            if not isinstance(loaded_data, list):
                self.show_error("Invalid transect set file: Expected a JSON list of transect definitions.")
                return

            loaded_count = 0
            invalid_count = 0
            newly_added_transects = []

            for transect_dict in loaded_data:
                if not isinstance(transect_dict, dict):
                    invalid_count += 1
                    continue

                # Check for new coordinate format
                name = transect_dict.get("name")
                start_lat = transect_dict.get("start_lat")
                start_lon = transect_dict.get("start_lon")
                end_lat = transect_dict.get("end_lat")
                end_lon = transect_dict.get("end_lon")
                length = transect_dict.get("length")

                if not (name and start_lat is not None and start_lon is not None and end_lat is not None and end_lon is not None):
                    self.log_message(f"Skipping invalid transect entry (missing name or coordinates): {transect_dict}", logging.WARNING)
                    invalid_count +=1
                    continue
                try:
                    # Validate that coordinates are numbers
                    float(start_lat), float(start_lon), float(end_lat), float(end_lon)
                except (ValueError, TypeError):
                    self.log_message(f"Skipping invalid transect entry (invalid coordinate format): {transect_dict}", logging.WARNING)
                    invalid_count +=1
                    continue

                newly_added_transects.append({
                    "name": name,
                    "start_lat": float(start_lat),
                    "start_lon": float(start_lon),
                    "end_lat": float(end_lat),
                    "end_lon": float(end_lon),
                    "length": float(length) if length is not None else None
                })
                loaded_count += 1

            if newly_added_transects:
                self.saved_transects.extend(newly_added_transects)
                self._populate_loaded_transects_list()
                self._update_batch_processing_ui_states()
                QMessageBox.information(self, "Transects Loaded",
                                        f"Successfully loaded {loaded_count} transect definitions.\n"
                                        f"{invalid_count} invalid entries were skipped.\n"
                                        "They have been added to the 'Transects for Batch Processing' list.")
                self.log_message(f"Loaded {loaded_count} transects from {filepath}. Invalid/skipped: {invalid_count}.", logging.INFO)
            elif invalid_count > 0 and loaded_count == 0:
                self.show_error(f"No valid transect definitions found in the selected file. All {invalid_count} entries were invalid.")
            else:
                 QMessageBox.information(self, "No New Transects", "No new transect definitions were added from the file.")

        except FileNotFoundError:
            self.show_error(f"Transect set file not found: {filepath}")
        except json.JSONDecodeError:
            self.show_error(f"Error decoding JSON from transect set file: {filepath}. Ensure it's a valid JSON.")
        except Exception as e:
            self.log_message(f"Error loading transect set: {e}", logging.ERROR)
            self.log_message(traceback.format_exc(), logging.DEBUG)
            self.show_error(f"An unexpected error occurred while loading the transect set:\n{e}")
        finally:
            self._update_batch_processing_ui_states()
            self.regenerate_map_display()

    # Add this new method to the GeoTaggerApp class

    # In the GeoTaggerApp class
# Replace your entire _autosave_defined_transect method with this one.

    def _autosave_defined_transect(self, start_time, end_time, length_m, transect_name):
        """
        Automatically saves the transect definition and extracts the corresponding images.
        """
        self.log_message(f"Processing and saving transect '{transect_name}'...", logging.INFO)
        active_gps_df = self.get_active_gps_df()

        # --- (Section 1: Determine autosave file path - This part is correct and remains) ---
        if self.autosave_json_path is None:
            base_output_dir = None
            if self.media_path and os.path.isdir(self.media_path):
                base_output_dir = self.media_path
            elif self.gps_file_path and os.path.isfile(self.gps_file_path):
                base_output_dir = os.path.dirname(self.gps_file_path)
            else:
                base_output_dir = os.getcwd()

            full_transect_dir_path = os.path.join(base_output_dir, self.transects_json_dir_name)
            
            if not os.path.exists(full_transect_dir_path):
                try:
                    os.makedirs(full_transect_dir_path)
                except OSError as e:
                    self.show_error(f"Could not create transect directory '{full_transect_dir_path}': {e}")
                    return

            gps_file_base = os.path.splitext(os.path.basename(self.gps_file_path))[0] if self.gps_file_path else "NoGPS"
            session_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"autosave_transects_{gps_file_base}_{session_time_str}.json"
            self.autosave_json_path = os.path.join(full_transect_dir_path, filename)
            self.log_message(f"New autosave session file created: {self.autosave_json_path}", logging.INFO)

        # --- (Section 2: Get transect coordinates - This part is correct and remains) ---
        try:
            start_point = active_gps_df.loc[start_time]
            end_point = active_gps_df.loc[end_time]
        except KeyError:
            self.show_error("Autosave failed: Could not find transect start/end points in the GPS data.")
            return

        # --- START OF NEW/RESTORED LOGIC ---

        # 3. Create a dedicated folder for this transect's images
        if self.media_path and os.path.isdir(self.media_path):
            # Create a main "transects_output" folder inside the media directory
            transects_output_parent_dir = os.path.join(self.media_path, "transects_output")
            specific_transect_output_dir = os.path.join(transects_output_parent_dir, transect_name)
            try:
                os.makedirs(specific_transect_output_dir, exist_ok=True)
                self.log_message(f"Created output directory for images: {specific_transect_output_dir}", logging.DEBUG)
            except OSError as e:
                self.show_error(f"Could not create image output directory for transect '{transect_name}': {e}")
                return
        else:
            self.log_message("Cannot extract images: No media path is set.", logging.WARNING)
            specific_transect_output_dir = None

        # 4. Filter and copy the images for this transect
        if specific_transect_output_dir and self.results_df is not None and not self.results_df.empty:
            start_utc, end_utc = start_time.astimezone(pytz.utc), end_time.astimezone(pytz.utc)
            filter_start, filter_end = min(start_utc, end_utc), max(start_utc, end_utc)
            
            mask = (self.results_df['Corrected Timestamp (UTC)'].notna() &
                    (self.results_df['Corrected Timestamp (UTC)'] >= filter_start) &
                    (self.results_df['Corrected Timestamp (UTC)'] <= filter_end))
            images_in_transect_df = self.results_df[mask]

            copy_ok_count = 0
            if not images_in_transect_df.empty:
                for _, row in images_in_transect_df.iterrows():
                    source_path = self.identifier_to_path_map.get(row.get('Identifier'))
                    if source_path and os.path.exists(source_path):
                        try:
                            shutil.copy2(source_path, specific_transect_output_dir)
                            copy_ok_count += 1
                        except Exception as copy_e:
                            self.log_message(f"Error copying image for transect '{transect_name}': {copy_e}", logging.ERROR)
            
            self.log_message(f"Extracted and saved {copy_ok_count} images for transect '{transect_name}'.", logging.INFO)

        # --- END OF NEW/RESTORED LOGIC ---

        # 5. Create the transect data dictionary for the JSON file
        new_transect_data = {
            "name": transect_name,
            "start_lat": start_point['latitude'],
            "start_lon": start_point['longitude'],
            "end_lat": end_point['latitude'],
            "end_lon": end_point['longitude'],
            "length": length_m,
            "source_gps_file": os.path.basename(self.gps_file_path) if self.gps_file_path else "N/A",
            "saved_at_utc_iso": datetime.now(timezone.utc).isoformat(),
        }

        # 6. Add to session list and save the JSON file (This part is correct and remains)
        self.saved_transects.append(new_transect_data)
        self.log_message(f"Added '{transect_name}' to session list. Total transects: {len(self.saved_transects)}.")

        try:
            with open(self.autosave_json_path, 'w') as f_json:
                json.dump(self.saved_transects, f_json, indent=4)
            self.log_message(f"Successfully autosaved {len(self.saved_transects)} transects to {self.autosave_json_path}", logging.INFO)
            self.transect_status_label.setText(f"Status: Transect '{transect_name}' saved. Define next transect.")
        except Exception as e:
            self.show_error(f"Autosave failed while writing to file: {e}")
            return

        # 7. Update UI (This part is correct and remains)
        self._populate_loaded_transects_list()
        self._update_batch_processing_ui_states()
        if self.map_view and self.map_view.page():
            safe_name_json = json.dumps(transect_name)
            js_call = f"drawSavedTransectLine({start_point['latitude']}, {start_point['longitude']}, {end_point['latitude']}, {end_point['longitude']}, {safe_name_json});"
            self.map_view.page().runJavaScript(js_call)
        self.clear_transect_definition()

    def _normalise_column_name(self, col_name):
        return str(col_name).strip().lower().replace(" ", "").replace("_", "").replace("-", "")

    def _find_column_by_alias(self, columns, aliases):
        normalised = {self._normalise_column_name(c): c for c in columns}
        for alias in aliases:
            key = self._normalise_column_name(alias)
            if key in normalised:
                return normalised[key]
        return None

    def _distance_to_transect_line_m(self, point_lats, point_lons, start_lat, start_lon, end_lat, end_lon):
        """Vectorised approximate point-to-line distance in metres for short transects."""
        lat0 = np.radians((float(start_lat) + float(end_lat)) / 2.0)
        metres_per_deg_lat = 111_320.0
        metres_per_deg_lon = 111_320.0 * np.cos(lat0)

        sx, sy = float(start_lon) * metres_per_deg_lon, float(start_lat) * metres_per_deg_lat
        ex, ey = float(end_lon) * metres_per_deg_lon, float(end_lat) * metres_per_deg_lat
        px = pd.to_numeric(point_lons, errors='coerce').astype(float).to_numpy() * metres_per_deg_lon
        py = pd.to_numeric(point_lats, errors='coerce').astype(float).to_numpy() * metres_per_deg_lat

        vx, vy = ex - sx, ey - sy
        seg_len2 = vx * vx + vy * vy
        if seg_len2 == 0:
            return np.sqrt((px - sx) ** 2 + (py - sy) ** 2), np.zeros_like(px, dtype=float)

        t = ((px - sx) * vx + (py - sy) * vy) / seg_len2
        t_clamped = np.clip(t, 0.0, 1.0)
        nearest_x = sx + t_clamped * vx
        nearest_y = sy + t_clamped * vy
        distances = np.sqrt((px - nearest_x) ** 2 + (py - nearest_y) ** 2)
        return distances, t

    def _get_images_near_transect(self, transect_info, buffer_m):
        """Return images whose georeferenced point falls within buffer_m of the transect segment."""
        required = ["start_lat", "start_lon", "end_lat", "end_lon"]
        if any(transect_info.get(k) is None for k in required):
            return pd.DataFrame()
        if self.results_df is None or self.results_df.empty:
            return pd.DataFrame()

        df = self.results_df.copy()
        if not all(c in df.columns for c in ["Latitude", "Longitude"]):
            return pd.DataFrame()
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
        df = df.dropna(subset=["Latitude", "Longitude"]).copy()
        if df.empty:
            return df

        distances, along_segment = self._distance_to_transect_line_m(
            df["Latitude"], df["Longitude"],
            transect_info["start_lat"], transect_info["start_lon"],
            transect_info["end_lat"], transect_info["end_lon"]
        )
        df["Distance_to_transect_m"] = distances
        df["Transect_position"] = along_segment

        # Keep points within the segment, plus a small end-cap tolerance controlled by the same buffer.
        length_m = transect_info.get("length")
        endcap_fraction = 0.05
        try:
            if length_m and float(length_m) > 0:
                endcap_fraction = min(0.25, float(buffer_m) / float(length_m))
        except Exception:
            pass

        mask = (
            (df["Distance_to_transect_m"] <= float(buffer_m)) &
            (df["Transect_position"] >= -endcap_fraction) &
            (df["Transect_position"] <= 1.0 + endcap_fraction)
        )
        return df.loc[mask].sort_values(["Transect_position", "Distance_to_transect_m"])

    def batch_process_loaded_transects(self):
        self.log_message("Batch Process Loaded Transects button clicked.")
        if not self.saved_transects:
            self.show_error("No transects loaded or defined to process.")
            return
        if not (self.processed_gps_df is not None and not self.processed_gps_df.empty and
                self.results_df is not None and
                self.media_path and
                self.identifier_to_path_map):
            self.show_error("Prerequisites for batch processing are not met (GPS/Results/Media Path/Identifier Map). Please re-process data if needed.")
            self._update_batch_processing_ui_states()
            return

        reply = QMessageBox.question(self, "Confirm Batch Process",
                                     f"This will process {len(self.saved_transects)} transect(s).\n"
                                     "Images will be filtered and copied into subfolders based on transect names.\n"
                                     "This may take some time. Do you want to proceed?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.Yes)
        if reply == QMessageBox.StandardButton.No:
            self.log_message("Batch processing cancelled by user.", logging.INFO)
            return

        self.log_message(f"--- Starting Batch Transect Processing ({len(self.saved_transects)} transects) ---", logging.INFO)
        self.progress_bar.setValue(0)
        self.reset_progress_bar_color()
        self.batch_process_button.setEnabled(False)
        self.load_all_transects_button.setEnabled(False)
        QApplication.processEvents()

        if self.media_path and os.path.isdir(self.media_path):
            base_output_dir_for_transects = self.media_path
        elif self.gps_file_path and os.path.isfile(self.gps_file_path):
            base_output_dir_for_transects = os.path.dirname(self.gps_file_path)
        else:
            base_output_dir_for_transects = os.getcwd()
        transects_output_parent_dir = os.path.join(base_output_dir_for_transects, "transects_output_batch")
        os.makedirs(transects_output_parent_dir, exist_ok=True)

        total_transects_processed = 0
        total_images_copied = 0
        failed_transects = []

        active_gps_df = self.get_active_gps_df()

        for i, transect_info in enumerate(self.saved_transects):
            transect_name = transect_info.get("name", f"Unnamed_Transect_{i+1}")
            self.log_message(f"Batch processing transect {i+1}/{len(self.saved_transects)}: '{transect_name}'", logging.INFO)
            QApplication.processEvents()

            try:
                buffer_m = self.transect_buffer_spinbox.value() if hasattr(self, 'transect_buffer_spinbox') else 2.0
                images_in_transect_df = self._get_images_near_transect(transect_info, buffer_m)

                if images_in_transect_df.empty:
                    self.log_message(f"No images found within {buffer_m:.2f} m of transect '{transect_name}'. Skipping.", logging.INFO)
                    total_transects_processed +=1
                    self.progress_bar.setValue(int(((i + 1) / len(self.saved_transects)) * 100))
                    continue
                safe_transect_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in transect_name).strip()
                if not safe_transect_name: safe_transect_name = f"Transect_{i+1}"
                specific_transect_output_dir = os.path.join(transects_output_parent_dir, safe_transect_name)
                os.makedirs(specific_transect_output_dir, exist_ok=True)

                copy_ok_count_this_transect = 0
                copy_fail_count_this_transect = 0

                for _, row in images_in_transect_df.iterrows():
                    identifier = row.get('Identifier')
                    source_path = self.identifier_to_path_map.get(identifier)

                    if not source_path or not os.path.exists(source_path):
                        self.log_message(f"Source file for ID '{identifier}' (transect '{transect_name}') not found at '{source_path}'. Skipping.", logging.WARNING)
                        copy_fail_count_this_transect += 1
                        continue
                    try:
                        new_filename = os.path.basename(source_path)
                        destination_path = os.path.join(specific_transect_output_dir, new_filename)
                        if os.path.exists(destination_path):
                             self.log_message(f"Destination file '{destination_path}' (transect '{transect_name}') already exists. Overwriting.", logging.WARNING)
                        shutil.copy2(source_path, destination_path)
                        copy_ok_count_this_transect += 1
                        total_images_copied += 1
                    except Exception as copy_e:
                        copy_fail_count_this_transect += 1
                        self.log_message(f"Error copying image ID '{identifier}' for transect '{transect_name}': {copy_e}", logging.ERROR)

                self.log_message(f"Transect '{transect_name}' processing complete. Copied: {copy_ok_count_this_transect}, Failed: {copy_fail_count_this_transect}", logging.INFO)
                total_transects_processed +=1

            except Exception as e_transect_proc:
                self.log_message(f"Error processing transect '{transect_name}': {e_transect_proc}", logging.ERROR)
                self.log_message(traceback.format_exc(), logging.DEBUG)
                failed_transects.append(transect_name)

            self.progress_bar.setValue(int(((i + 1) / len(self.saved_transects)) * 100))
            QApplication.processEvents()

        self.progress_bar.setValue(100)
        self._update_batch_processing_ui_states()

        summary_msg = f"Batch processing finished.\n\nProcessed {total_transects_processed}/{len(self.saved_transects)} transects.\n" \
                      f"Total images copied: {total_images_copied}.\n"
        if failed_transects:
            summary_msg += f"\nFailed to process {len(failed_transects)} transect(s):\n- " + "\n- ".join(failed_transects)
            summary_msg += "\n\nCheck log for details."

        QMessageBox.information(self, "Batch Processing Complete", summary_msg)
        self.log_message(f"--- Batch Transect Processing Finished --- {summary_msg.replace(os.linesep, ' ')}", logging.INFO)

    def update_preset_length_circle_on_map(self):
        if not GEOPY_AVAILABLE:
            return
        if self._map_file_path is None or not os.path.exists(self._map_file_path) or not (self.map_view and self.map_view.page()):
            return

        active_gps_df = self.get_active_gps_df()
        if self.transect_definition_mode == 'selecting_end' and \
           self.enforce_length_checkbox.isChecked() and \
           self.transect_start_time is not None and \
           not active_gps_df.empty:
            try:
                start_time_utc_aware = self.transect_start_time
                if start_time_utc_aware.tzinfo is None:
                    start_time_utc_aware = pytz.utc.localize(start_time_utc_aware)
                start_point_data = active_gps_df.loc[start_time_utc_aware]
                lat = start_point_data['latitude']
                lon = start_point_data['longitude']
                radius_m = self.preset_length_spinbox.value()
                if pd.notna(lat) and pd.notna(lon) and radius_m > 0:
                    js_call = f"drawPresetLengthCircle({lat}, {lon}, {radius_m}, 'draw');"
                    self.map_view.page().runJavaScript(js_call)
                    return
            except KeyError:
                self.log_message(f"Preset circle: Start time {self.transect_start_time} not found in active GPS for circle.", logging.WARNING)
            except Exception as e:
                self.log_message(f"Error preparing to draw preset length circle: {e}", logging.ERROR)
        js_call_remove = "drawPresetLengthCircle(null, null, 0, 'remove');"
        self.map_view.page().runJavaScript(js_call_remove)

    @pyqtSlot()
    def clear_transect_definition(self):
        self.log_message("Clearing current single transect definition and resetting its results view.")
        self.transect_start_time = None
        self.transect_end_time = None
        self.transect_length_meters = None
        if self.transect_definition_mode is not None:
            self.transect_definition_mode = None
            if self.define_transect_button.isChecked():
                self.define_transect_button.setChecked(False)

        self.clear_map_highlight()
        self.update_preset_length_circle_on_map()
        if self.map_view and self.map_view.page():
            self.map_view.page().runJavaScript("clearTransectLine();")

        if self.results_df is not None:
            self.populate_results_table(self.results_df)
        else:
            self.populate_results_table(None)
        self.results_table.clearSelection()
        self.extract_button.setEnabled(False)
        self.update_transect_ui()
        if self.map_view and self.map_view.page():
            self.map_view.page().runJavaScript("drawPresetLengthCircle(null, null, 0, 'remove');")

    @pyqtSlot(bool)
    def toggle_transect_definition_mode(self, checked):
        active_gps_df = self.get_active_gps_df()
        if active_gps_df.empty and (self.processed_gps_df is None or self.processed_gps_df.empty):
            self.log_message("Cannot define transect: GPS data not loaded/processed.", logging.WARNING)
            self.define_transect_button.setChecked(False)
            self.transect_definition_mode = None
            self.update_transect_ui()
            self.update_preset_length_circle_on_map()
            return
        if active_gps_df.empty and self.selected_filter_date:
            self.log_message(f"Cannot define transect: No GPS points for selected date {self.selected_filter_date}. Clear filter or choose another date.", logging.WARNING)
            QMessageBox.warning(self, "No GPS Points", f"No GPS points available for {self.selected_filter_date.strftime('%Y-%m-%d')} to define a transect. Try 'Show All Dates'.")
            self.define_transect_button.setChecked(False)
            self.transect_definition_mode = None
            self.update_transect_ui()
            self.update_preset_length_circle_on_map()
            return

        if checked:
            self.transect_definition_mode = 'selecting_start'
            self.transect_start_time = None
            self.transect_end_time = None
            self.transect_length_meters = None
            self.clear_map_highlight()
            self.log_message("Starting transect definition: Click map for start.")
            self.populate_results_table(self.results_df)
            self.results_table.clearSelection()
            self.extract_button.setEnabled(False)
        else:
            if self.transect_definition_mode in ['selecting_start', 'selecting_end']:
                self.log_message("Transect definition cancelled by user.", logging.INFO)
            self.transect_definition_mode = None
        self.update_transect_ui()
        self.update_preset_length_circle_on_map()

    @pyqtSlot(str)
    def on_date_filter_changed(self, date_str):
        self.log_message(f"Date filter changed to: '{date_str}' (Context Timezone: {self.current_timezone_str})", logging.INFO)
        self.clear_transect_definition()

        if date_str == "Show All Dates" or not date_str:
            self.selected_filter_date = None
            self.filtered_gps_by_date_df = self.processed_gps_df.copy() if self.processed_gps_df is not None else None
            self.log_message("Date filter cleared. Showing all GPS data.", logging.DEBUG)
        else:
            if self.processed_gps_df is None or self.processed_gps_df.empty:
                self.log_message("Cannot filter by date: Main processed_gps_df is empty or None.", logging.WARNING)
                self.selected_filter_date = None
                self.filtered_gps_by_date_df = None
                self.date_filter_combo.blockSignals(True)
                self.date_filter_combo.setCurrentIndex(0)
                self.date_filter_combo.blockSignals(False)
                self.regenerate_map_display()
                self.update_transect_ui()
                return

            try:
                self.selected_filter_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                target_tz = pytz.timezone(self.current_timezone_str)
                local_day_start_naive = datetime.combine(self.selected_filter_date, datetime.min.time())
                local_day_end_naive = datetime.combine(self.selected_filter_date, datetime.max.time().replace(microsecond=999999))
                utc_start = target_tz.localize(local_day_start_naive, is_dst=None).astimezone(pytz.utc)
                utc_end = target_tz.localize(local_day_end_naive, is_dst=None).astimezone(pytz.utc)
                self.log_message(f"Filtering GPS data for local date {date_str} ({self.current_timezone_str}): UTC range from {utc_start.isoformat()} to {utc_end.isoformat()}", logging.DEBUG)
                mask = (self.processed_gps_df.index >= utc_start) & (self.processed_gps_df.index <= utc_end)
                self.filtered_gps_by_date_df = self.processed_gps_df[mask].copy()
                if self.filtered_gps_by_date_df.empty:
                    self.log_message(f"No GPS data found for selected date {date_str} within the calculated UTC range.", logging.INFO)
                    QMessageBox.information(self, "No GPS Data", f"No GPS data points found for {date_str} in timezone {self.current_timezone_str}.")
            except ValueError:
                self.log_message(f"Error parsing date string: '{date_str}'. Reverting to 'Show All Dates'.", logging.ERROR)
                self.selected_filter_date = None
                self.filtered_gps_by_date_df = self.processed_gps_df.copy() if self.processed_gps_df is not None else None
                self.date_filter_combo.blockSignals(True)
                self.date_filter_combo.setCurrentIndex(0)
                self.date_filter_combo.blockSignals(False)
            except pytz.UnknownTimeZoneError:
                self.log_message(f"Cannot filter by date: Unknown timezone '{self.current_timezone_str}'.", logging.ERROR)
                self.show_error(f"Invalid timezone '{self.current_timezone_str}' for date filtering. Please select a valid one.")
                self.selected_filter_date = None
                self.filtered_gps_by_date_df = self.processed_gps_df.copy() if self.processed_gps_df is not None else None
                self.date_filter_combo.blockSignals(True)
                self.date_filter_combo.setCurrentIndex(0)
                self.date_filter_combo.blockSignals(False)
                self.update_transect_ui()
                return
            except Exception as e:
                self.log_message(f"Unexpected error during date filtering for '{date_str}': {e}", logging.ERROR)
                self.log_message(traceback.format_exc(), logging.DEBUG)
                self.selected_filter_date = None
                self.filtered_gps_by_date_df = self.processed_gps_df.copy() if self.processed_gps_df is not None else None
                self.date_filter_combo.blockSignals(True)
                self.date_filter_combo.setCurrentIndex(0)
                self.date_filter_combo.blockSignals(False)

        self.regenerate_map_display()
        self.update_transect_ui()

    def regenerate_map_display(self):
        self.log_message("Regenerating map display due to filter change or initial load...", logging.DEBUG)
        active_gps_for_map = self.get_active_gps_df()
        results_for_map = self.results_df if self.results_df is not None else pd.DataFrame()

        if active_gps_for_map.empty and (self.processed_gps_df is None or self.processed_gps_df.empty) and results_for_map.empty:
            self.log_message("No GPS data (full or filtered) and no results to display on map.", logging.INFO)
            if self.map_view and self.map_view.page():
                self.map_view.setUrl(QUrl("about:blank"))
            if self._map_file_path and os.path.exists(self._map_file_path):
                try:
                    os.remove(self._map_file_path)
                except OSError as e:
                    self.log_message(f"Error removing old temp map file: {e}", logging.WARNING)
            self._map_file_path = None
            self.save_map_button.setEnabled(False)
        else:
            new_map_path = self.generate_interactive_map_for_app(active_gps_for_map, results_for_map)
            if new_map_path and os.path.exists(new_map_path):
                self.display_map(new_map_path)
            else:
                self.log_message("Map regeneration failed to produce a valid map file.", logging.ERROR)
                if self.map_view and self.map_view.page():
                    self.map_view.setUrl(QUrl("about:blank"))
                if self._map_file_path and os.path.exists(self._map_file_path):
                    try:
                        os.remove(self._map_file_path)
                    except OSError as e:
                        self.log_message(f"Error removing old temp map file: {e}", logging.WARNING)
                self._map_file_path = None
                self.save_map_button.setEnabled(False)

        self.update_transect_ui()
        self.update_preset_length_circle_on_map()

    # REPLACE your entire old find_closest_point_in_df method with this:

    # In GeoTaggerApp class
# REPLACE your entire old find_closest_point_in_df method with this:

    def find_closest_point_in_df(self, target_coords, gps_df):
        """
        Finds the closest point in a GPS dataframe to a given coordinate.
        This version prioritizes the fast k-d tree lookup.
        """
        # --- PRIMARY OPTIMIZED PATH ---
        # Use the k-d tree if it exists AND we are working with the full, unfiltered GPS track.
        # The (gps_df is self.processed_gps_df) check is the most reliable way to know this.
        if self.gps_kdtree is not None and gps_df is self.processed_gps_df:
            try:
                # The query() method is incredibly fast. It returns the distance and the index.
                distance, index = self.gps_kdtree.query(target_coords)
                
                # Use the returned index to get the original timestamp from our mapping DataFrame.
                closest_timestamp = self.gps_kdtree_df_map.index[index]
                
                # Get the coordinates from the same row.
                closest_point_series = self.gps_kdtree_df_map.iloc[index]
                closest_coords = (closest_point_series['latitude'], closest_point_series['longitude'])
                
                return closest_timestamp, closest_coords
            except Exception as e:
                self.log_message(f"Optimized point lookup failed: {e}. Falling back to slow method.", logging.WARNING)

        # --- SLOW FALLBACK PATH (Original Method) ---
        # This will now only be used for filtered DataFrames (e.g., when a date filter is active)
        # or if the k-d tree failed to build for some reason.
        if not GEOPY_AVAILABLE or gps_df.empty:
            return None, None

        # self.log_message("Using slow linear scan for point lookup (expected if date filter is active).", logging.DEBUG)
        min_dist = float('inf')
        closest_timestamp = None
        closest_coords = None

        for ts, row in gps_df.iterrows():
            point_coords = (row['latitude'], row['longitude'])
            if pd.isna(point_coords[0]) or pd.isna(point_coords[1]):
                continue
            dist = geodesic(target_coords, point_coords).meters
            if dist < min_dist:
                min_dist = dist
                closest_timestamp = ts
                closest_coords = point_coords

        return closest_timestamp, closest_coords


    def load_transect(self):
        if not GEOPY_AVAILABLE:
            self.show_error("Cannot load transect: 'geopy' library is required.")
            return
        if self.processed_gps_df is None or self.processed_gps_df.empty:
            self.show_error("Please process GPS data before loading a transect definition.")
            return

        if not os.path.exists(self.transects_json_dir):
             self.log_message(f"Transect JSON directory '{self.transects_json_dir}' does not exist. Nothing to load.", logging.INFO)
             QMessageBox.information(self, "No Transects Saved", f"The directory for saved transects ('{self.transects_json_dir}') does not exist.")
             return

        filepath, _ = QFileDialog.getOpenFileName(self, "Load Transect Definition",
                                                  self.transects_json_dir,
                                                  "JSON files (*.json)")
        if not filepath:
            return
        if self.map_view and self.map_view.page():
            self.map_view.page().runJavaScript("drawPresetLengthCircle(null, null, 0, 'remove');")
            self.map_view.page().runJavaScript("clearTransectLine();")

        try:
            with open(filepath, 'r') as f:
                transect_data_or_list = json.load(f)
            transect_data_to_load = None
            if isinstance(transect_data_or_list, list):
                if not transect_data_or_list:
                    self.show_error("Loaded transect file is empty (list of transects).")
                    return
                transect_data_to_load = transect_data_or_list[0]
                if len(transect_data_or_list) > 1:
                    QMessageBox.information(self, "Multiple Transects in File",
                                            "The loaded file contains multiple transect definitions.\n"
                                            "Only the first transect has been loaded into the single-transect UI for now.\n"
                                            "To process all, use 'Load Transect Set (JSON)' in Batch Operations.")
                self.log_message(f"Loaded first transect from a list in file: {filepath}", logging.INFO)
            elif isinstance(transect_data_or_list, dict):
                transect_data_to_load = transect_data_or_list
            else:
                self.show_error("Invalid transect file format: Not a JSON object or list.")
                return

            # --- Check for new coordinate-based format ---
            if "start_lat" in transect_data_to_load and "start_lon" in transect_data_to_load:
                start_lat = transect_data_to_load['start_lat']
                start_lon = transect_data_to_load['start_lon']
                end_lat = transect_data_to_load['end_lat']
                end_lon = transect_data_to_load['end_lon']
                length_meters_from_file = transect_data_to_load.get("length")
                transect_name_from_file = transect_data_to_load.get("name", "Loaded Transect")

                active_gps_df = self.get_active_gps_df()
                if active_gps_df.empty:
                    self.show_error("Cannot map loaded transect to track: No active GPS data.")
                    return

                start_time_utc, _ = self.find_closest_point_in_df((start_lat, start_lon), active_gps_df)
                end_time_utc, _ = self.find_closest_point_in_df((end_lat, end_lon), active_gps_df)

                if start_time_utc is None or end_time_utc is None:
                    self.show_error("Could not find points on the current GPS track that match the saved transect coordinates.")
                    return

            # --- Fallback for old timestamp-based format ---
            elif "start" in transect_data_to_load and "end" in transect_data_to_load:
                self.log_message("Loading legacy timestamp-based transect file.", logging.INFO)
                start_time_utc = datetime.fromisoformat(transect_data_to_load["start"].replace('Z', '+00:00')).astimezone(pytz.utc)
                end_time_utc = datetime.fromisoformat(transect_data_to_load["end"].replace('Z', '+00:00')).astimezone(pytz.utc)
                length_meters_from_file = transect_data_to_load.get("length")
                transect_name_from_file = transect_data_to_load.get("name", "Loaded Transect")
            else:
                self.show_error("Invalid transect file: Missing required keys for coordinates or timestamps.")
                return

            # Check if times are out of bounds of current view
            active_gps_df = self.get_active_gps_df()
            if not active_gps_df.empty:
                min_gps_time, max_gps_time = active_gps_df.index.min(), active_gps_df.index.max()
                if not (min_gps_time <= start_time_utc <= max_gps_time and \
                        min_gps_time <= end_time_utc <= max_gps_time):
                    QMessageBox.warning(self, "Transect Time Mismatch",
                                        "The loaded transect times are outside the range of the currently active GPS data "
                                        "(possibly due to a date filter).\n\n"
                                        "The transect will be loaded, but may not be fully visible or usable until the "
                                        "date filter is adjusted or cleared.")

            self.clear_transect_definition()
            self.transect_start_time = start_time_utc
            self.transect_end_time = end_time_utc
            recalculated_length = self.calculate_transect_length(start_time_utc, end_time_utc)
            if recalculated_length is not None:
                self.transect_length_meters = recalculated_length
            elif length_meters_from_file is not None:
                 self.transect_length_meters = float(length_meters_from_file)
                 self.log_message("Used transect length from file as recalculation was not possible.", logging.INFO)
            else:
                self.transect_length_meters = None

            self.transect_definition_mode = None
            if self.define_transect_button.isChecked():
                 self.define_transect_button.setChecked(False)

            if not active_gps_df.empty and \
               'latitude' in active_gps_df.columns and 'longitude' in active_gps_df.columns:
                try:
                    start_coords = active_gps_df.loc[start_time_utc]
                    end_coords = active_gps_df.loc[end_time_utc]
                    js_draw_line = f"drawTransectLine({start_coords['latitude']}, {start_coords['longitude']}, {end_coords['latitude']}, {end_coords['longitude']}, false);"
                    self.map_view.page().runJavaScript(js_draw_line)
                except KeyError:
                     self.log_message("Could not find loaded transect start/end points in current GPS track to draw line.", logging.WARNING)
                except Exception as e_draw:
                    self.log_message(f"Error drawing loaded transect line on map: {e_draw}", logging.ERROR)

            self.log_message(f"Single transect definition loaded from: {filepath}", logging.INFO)
            self.update_transect_ui()
            self.highlight_map_range(self.transect_start_time, self.transect_end_time)
            self.find_images_in_defined_range()

            source_gps_from_file = transect_data_to_load.get("source_gps_file", "N/A")
            current_gps_filename = os.path.basename(self.gps_file_path) if self.gps_file_path else "Unknown"
            if source_gps_from_file != "N/A" and source_gps_from_file != current_gps_filename:
                QMessageBox.information(self, "Transect Loaded",
                                        f"Transect '{transect_name_from_file}' loaded successfully.\n"
                                        f"Note: This transect was originally saved with GPS file '{source_gps_from_file}'.\n"
                                        f"Current GPS file is '{current_gps_filename}'.")
            else:
                QMessageBox.information(self, "Transect Loaded", f"Transect '{transect_name_from_file}' loaded successfully.")
            self.update_preset_length_circle_on_map()

        except FileNotFoundError:
            self.log_message(f"Transect file not found: {filepath}", logging.ERROR)
            self.show_error("Selected transect file does not exist.")
        except json.JSONDecodeError:
            self.log_message(f"Error decoding JSON from transect file: {filepath}", logging.ERROR)
            self.show_error("Could not read transect data: Invalid JSON format.")
        except Exception as e:
            self.log_message(f"Error loading transect definition: {e}", logging.ERROR)
            self.log_message(traceback.format_exc(), logging.DEBUG)
            self.show_error(f"Could not load transect definition:\n{e}")

    def log_message(self, message, level=logging.INFO):
        logging.log(level, str(message))

    def show_error(self, message):
        QMessageBox.critical(self, "Error", str(message))
        self.log_message(f"ERROR DIALOG SHOWN: {message}", logging.ERROR)
        p = self.progress_bar.palette()
        p.setColor(QPalette.ColorRole.Highlight, QColor('red'))
        self.progress_bar.setPalette(p)
        QTimer.singleShot(3000, self.reset_progress_bar_color)
        if self.worker_thread is None or not self.worker_thread.isRunning():
            self.process_button.setEnabled(True)
            self._update_batch_processing_ui_states()

    def reset_progress_bar_color(self):
        self.progress_bar.setPalette(QApplication.instance().palette())

    def get_active_gps_df(self):
        if self.selected_filter_date and self.filtered_gps_by_date_df is not None and not self.filtered_gps_by_date_df.empty:
            return self.filtered_gps_by_date_df
        if self.processed_gps_df is not None and not self.processed_gps_df.empty:
            return self.processed_gps_df
        return pd.DataFrame()

    @pyqtSlot(str)
    def update_current_timezone(self, tz_string):
        if not tz_string:
            return
        try:
            pytz.timezone(tz_string)
            old_tz = self.current_timezone_str
            self.current_timezone_str = tz_string
            self.log_message(f"Data timezone changed from '{old_tz}' to: {self.current_timezone_str}", logging.INFO)
            self.date_filter_label.setText(f"Filter GPS by Date ({self.current_timezone_str}):")

            if self.processed_gps_df is not None and not self.processed_gps_df.empty:
                self.date_filter_combo.blockSignals(True)
                current_date_filter_text = self.date_filter_combo.currentText()
                self.date_filter_combo.clear()
                self.date_filter_combo.addItem("Show All Dates")
                self.available_dates = []
                try:
                    target_tz_obj = pytz.timezone(self.current_timezone_str)
                    local_times = self.processed_gps_df.index.tz_convert(target_tz_obj)
                    unique_dates = sorted(list(set(dt.date() for dt in local_times)))
                    self.available_dates = unique_dates
                    for date_obj in self.available_dates:
                        self.date_filter_combo.addItem(date_obj.strftime("%Y-%m-%d"))
                    idx = self.date_filter_combo.findText(current_date_filter_text)
                    if idx != -1:
                        self.date_filter_combo.setCurrentIndex(idx)
                    else:
                        self.date_filter_combo.setCurrentIndex(0)
                except Exception as e:
                    self.log_message(f"Error re-populating date filter for new timezone: {e}", logging.ERROR)
                    self.date_filter_combo.setCurrentIndex(0)
                self.date_filter_combo.setEnabled(True)
                self.date_filter_combo.blockSignals(False)
                if self.date_filter_combo.currentIndex() > 0 :
                    self.on_date_filter_changed(self.date_filter_combo.currentText())
                else:
                    self.selected_filter_date = None
                    self.filtered_gps_by_date_df = self.processed_gps_df.copy() if self.processed_gps_df is not None else None
                    self.regenerate_map_display()
            self.update_transect_ui()
            self._populate_loaded_transects_list()
        except pytz.UnknownTimeZoneError:
            self.log_message(f"Invalid timezone entered: {tz_string}. Reverting.", logging.WARNING)
            idx = self.media_timezone_input.findText(self.current_timezone_str, Qt.MatchFlag.MatchFixedString) # old_tz is now current_timezone_str
            if idx >=0 :
                self.media_timezone_input.blockSignals(True)
                self.media_timezone_input.setCurrentIndex(idx)
                # self.current_timezone_str = old_tz # no need to change it back
                self.media_timezone_input.blockSignals(False)
            else: # Fallback if old_tz somehow not in list
                utc_idx = self.media_timezone_input.findText("UTC", Qt.MatchFlag.MatchFixedString)
                if utc_idx >= 0:
                    self.media_timezone_input.setCurrentIndex(utc_idx)
                self.current_timezone_str = "UTC"
            self.date_filter_label.setText(f"Filter GPS by Date ({self.current_timezone_str}):")
        except Exception as e:
            self.log_message(f"Error updating timezone: {e}", logging.ERROR)

    def format_datetime_for_display(self, dt_utc):
        if dt_utc is None or not isinstance(dt_utc, (datetime, pd.Timestamp)):
            return "Not Set"
        try:
            target_tz = pytz.timezone(self.current_timezone_str)
            if dt_utc.tzinfo is None:
                dt_utc = pytz.utc.localize(dt_utc)
            elif dt_utc.tzinfo.utcoffset(dt_utc) != timedelta(0): # Ensure it's truly UTC before converting
                dt_utc = dt_utc.astimezone(pytz.utc)
            local_dt = dt_utc.astimezone(target_tz)
            return local_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + f" ({self.current_timezone_str})"
        except pytz.UnknownTimeZoneError:
            self.log_message(f"Unknown timezone '{self.current_timezone_str}' for display. Using UTC.", logging.WARNING)
            if dt_utc.tzinfo is None: dt_utc = pytz.utc.localize(dt_utc) # Ensure UTC if naive
            return dt_utc.astimezone(pytz.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " (UTC)"
        except Exception as e:
            self.log_message(f"Error formatting time {dt_utc} to tz {self.current_timezone_str}: {e}", logging.ERROR)
            if dt_utc.tzinfo is None: dt_utc = pytz.utc.localize(dt_utc) # Ensure UTC if naive
            return dt_utc.astimezone(pytz.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " (UTC)"

    def update_transect_ui(self):
        start_text = self.format_datetime_for_display(self.transect_start_time)
        end_text = self.format_datetime_for_display(self.transect_end_time)
        self.transect_start_display.setText(start_text)
        self.transect_end_display.setText(end_text)

        length_text = "Length: N/A"
        if self.transect_length_meters is not None and GEOPY_AVAILABLE:
            length_text = f"Length: {self.transect_length_meters:.2f} m"
        self.transect_length_label.setText(length_text)

                # ... inside update_transect_ui ...
        is_transect_defined = self.transect_start_time is not None and self.transect_end_time is not None

        # Update button states
        # The add_transect_to_batch_button is no longer used, so we don't need to set its state.
        

        active_gps_for_ui_check = self.get_active_gps_df()
        base_gps_available = self.processed_gps_df is not None and not self.processed_gps_df.empty

        self.define_transect_button.setEnabled(base_gps_available)
        self.load_transect_button.setEnabled(base_gps_available and GEOPY_AVAILABLE)
        self.enforce_length_checkbox.setEnabled(GEOPY_AVAILABLE and base_gps_available)
        self.preset_length_spinbox.setEnabled(GEOPY_AVAILABLE and base_gps_available and self.enforce_length_checkbox.isChecked())
        self.save_all_transects_button.setEnabled(bool(self.saved_transects))
        self.clear_transect_button.setEnabled(is_transect_defined or base_gps_available)

        status_prefix = "Status: "
        if self.selected_filter_date:
            status_prefix += f"GPS filtered to {self.selected_filter_date.strftime('%Y-%m-%d')} ({self.current_timezone_str}). "
            if active_gps_for_ui_check.empty and base_gps_available :
                status_prefix += "No GPS points for this date. "

        if not base_gps_available:
            self.transect_status_label.setText("Status: Load and Process GPS data first.")
            if self.define_transect_button.isChecked():
                self.define_transect_button.setChecked(False)
            self.define_transect_button.setText("Define Transect Start/End")
        elif self.transect_definition_mode == 'selecting_start':
            self.transect_status_label.setText(status_prefix + "Click the desired START point on the map.")
            self.define_transect_button.setText("Defining Transect (Click Start...)")
        elif self.transect_definition_mode == 'selecting_end':
            msg = status_prefix + "Click the desired END point on the map."
            if self.enforce_length_checkbox.isChecked() and GEOPY_AVAILABLE:
                msg += f" Target length: {self.preset_length_spinbox.value():.2f} m."
            self.transect_status_label.setText(msg)
            self.define_transect_button.setText("Defining Transect (Click End...)")
        else: # Not actively defining
            if self.define_transect_button.isChecked():
                self.define_transect_button.setChecked(False) # Ensure button state is correct
            self.define_transect_button.setText("Define Transect Start/End")

            if is_transect_defined:
                current_length_info = ""
                if self.transect_length_meters is not None and GEOPY_AVAILABLE:
                    current_length_info = f" Current transect length: {self.transect_length_meters:.2f} m."
                self.transect_status_label.setText(status_prefix + "Transect defined." + current_length_info + " Click 'Clear' to reset.")
            else:
                self.transect_status_label.setText(status_prefix + "Click 'Define Transect' to select points.")

    def _get_gps_parsing_params_from_ui(self):
        """Helper to get GPS parsing parameters from the UI widgets."""
        gps_params = {}
        if not self.gps_file_path:
            return None

        is_csv = self.gps_file_path.lower().endswith(('.csv', '.txt'))
        if is_csv:
            if not self.csv_mapping_group.isVisible():
                self.show_error("CSV mapping details are not configured. Please reload the CSV file.")
                return None
            gps_params['type'] = 'csv'
            gps_params['lat_col'] = self.lat_col_combo.currentText()
            gps_params['lon_col'] = self.lon_col_combo.currentText()
            gps_params['datetime_format'] = self.datetime_format_input.text().strip()
            gps_params['boat_timezone'] = self.current_timezone_str
            if not gps_params['lat_col'] or not gps_params['lon_col']:
                self.show_error("Latitude and Longitude columns must be selected for CSV.")
                return None
            if self.ts_single_radio.isChecked():
                gps_params['datetime_col'] = self.datetime_col_combo.currentText()
                gps_params['date_col'] = None
                gps_params['time_col'] = None
                if not gps_params['datetime_col']:
                    self.show_error("Single Timestamp column must be selected.")
                    return None
            else:
                gps_params['datetime_col'] = None
                gps_params['date_col'] = self.date_col_combo.currentText()
                gps_params['time_col'] = self.time_col_combo.currentText()
                if not gps_params['date_col'] or not gps_params['time_col']:
                    self.show_error("Separate Date and Time columns must be selected.")
                    return None
            if not gps_params['datetime_format']:
                self.show_error("Timestamp Format String is required for CSV parsing.")
                return None
        else: # GPX
            gps_params['type'] = 'gpx'
            gps_params['boat_timezone'] = 'UTC'
        return gps_params

    def _run_preliminary_gps_parse_and_map(self):
        """Parses GPS data quickly on the main thread for UI setup."""
        if not self.gps_file_path or not ENGINE_AVAILABLE:
            return

        self.log_message("Starting preliminary GPS parse for UI setup...", logging.INFO)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            gps_parsing_params = self._get_gps_parsing_params_from_ui()
            if not gps_parsing_params:
                self.log_message("Could not get GPS parsing params for preliminary parse.", logging.WARNING)
                QApplication.restoreOverrideCursor()
                return

            parse_type = gps_parsing_params.get('type', 'gpx')
            temp_gps_df = None
            if parse_type == 'gpx':
                temp_gps_df = engine.parse_gpx_file(self.gps_file_path)
            elif parse_type == 'csv':
                delimiter = ','
                try:
                    with open(self.gps_file_path, 'r', errors='ignore') as f:
                        dialect = csv.Sniffer().sniff(f.read(2048)); f.seek(0)
                        delimiter = dialect.delimiter
                except Exception: pass # Use default

                temp_gps_df = engine.parse_boat_log_csv(
    csv_file_path=self.gps_file_path,
    date_col=gps_parsing_params['date_col'],
    time_col=gps_parsing_params['time_col'],
    lat_col=gps_parsing_params['lat_col'],
    lon_col=gps_parsing_params['lon_col'],
    alt_col=None,                                       
    boat_timezone_str=gps_parsing_params['boat_timezone'],
    datetime_format=gps_parsing_params['datetime_format'],
    datetime_col=gps_parsing_params.get('datetime_col'),
    delimiter=delimiter
)


            if temp_gps_df is not None and not temp_gps_df.empty:
                # Basic validation and timezone conversion
                if temp_gps_df.index.tz is None:
                    temp_gps_df.index = temp_gps_df.index.tz_localize('UTC')
                elif temp_gps_df.index.tz.utcoffset(temp_gps_df.index.min()) != timedelta(0):
                    temp_gps_df.index = temp_gps_df.index.tz_convert('UTC')
                if not temp_gps_df.index.is_monotonic_increasing:
                    temp_gps_df.sort_index(inplace=True)

                self.preliminary_gps_df = temp_gps_df.dropna(subset=['latitude', 'longitude'])
                self.log_message(f"Preliminary GPS parse successful: {len(self.preliminary_gps_df)} points.", logging.INFO)

                # Generate a preliminary map
                prelim_map_path = self.generate_interactive_map_for_app(self.preliminary_gps_df, pd.DataFrame())
                self.display_map(prelim_map_path)
            else:
                self.log_message("Preliminary GPS parse resulted in empty data.", logging.WARNING)
                self.preliminary_gps_df = None
                self.map_view.setUrl(QUrl("about:blank"))

        except Exception as e:
            self.log_message(f"Preliminary GPS parse failed: {e}", logging.ERROR)
            self.show_error(f"Failed to pre-process GPS data for map display: {e}")
            self.preliminary_gps_df = None
        finally:
            QApplication.restoreOverrideCursor()
            self.update_sync_method_ui()


    def load_gps_data(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.show_error("Cannot load new data while processing is active.")
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "Select GPS Data File", "", "GPS Logs (*.gpx *.csv *.txt);;All Files (*)")
        if file_path:
            # --- Reset application state for new file ---
            self.gps_file_path = file_path
            self.gps_path_display.setText(os.path.basename(file_path))
            self.log_message(f"GPS file selected: {file_path}")
            self.csv_mapping_group.setVisible(False)
            self.results_df = self.processed_gps_df = self.preliminary_gps_df = None
            self.original_media_list = self.saved_transects = []
            self.identifier_to_path_map = {}
            self.sync_image_info = self.sync_gps_point_info = None

            self._populate_loaded_transects_list()
            self.clear_transect_definition()
            self.populate_results_table(None)
            self.map_view.setUrl(QUrl("about:blank"))
            self._map_file_path = None

            # Reset buttons and UI states
            for btn in [self.export_csv_button, self.save_map_button, self.geotag_images_button, self.extract_button]:
                btn.setEnabled(False)
            self.date_filter_combo.clear()
            self.date_filter_combo.addItem("Show All Dates")
            self.date_filter_combo.setEnabled(False)

            if file_path.lower().endswith(('.csv', '.txt')):
                self.log_message("CSV/TXT file detected. Reading headers...")
                try:
                    delimiter=','
                    try:
                        with open(file_path, 'r', errors='ignore') as f:
                            sample_lines = [line for line in (f.readline() for _ in range(20)) if line and line.strip()]
                            sample = "".join(sample_lines)
                            if not sample:
                                self.show_error("CSV file appears to be empty or contains no valid data for sniffing.")
                                self.gps_file_path = None
                                self.gps_path_display.clear()
                                return
                            dialect = csv.Sniffer().sniff(sample)
                            delimiter = dialect.delimiter
                            self.log_message(f"Detected delimiter: '{repr(delimiter)}'")
                    except csv.Error as sniff_err:
                        self.log_message(f"Delimiter sniffing failed ({sniff_err}), using default ','.", logging.WARNING)
                    except Exception as e:
                        self.log_message(f"Error during delimiter sniffing setup ({e}), using default ','.", logging.WARNING)
                    try:
                        headers = pd.read_csv(file_path, sep=delimiter, nrows=0, engine='python', skipinitialspace=True, encoding_errors='ignore').columns.tolist()
                    except UnicodeDecodeError:
                        self.log_message("UnicodeDecodeError reading CSV headers with utf-8, trying latin-1", logging.WARNING)
                        headers = pd.read_csv(file_path, sep=delimiter, nrows=0, engine='python', skipinitialspace=True, encoding='latin-1', encoding_errors='ignore').columns.tolist()
                    except pd.errors.EmptyDataError:
                        self.show_error("CSV file is empty or contains no data after headers.")
                        self.gps_file_path = None
                        self.gps_path_display.clear()
                        return
                    if not headers:
                        self.show_error("Could not read headers from CSV file.")
                        self.gps_file_path = None
                        self.gps_path_display.clear()
                        return
                    self.log_message(f"CSV Headers found: {headers}")
                    for combo in [self.lat_col_combo, self.lon_col_combo, self.date_col_combo, self.time_col_combo, self.datetime_col_combo]:
                        combo.clear()
                        combo.addItems(headers)
                    for widget in [self.lat_col_combo, self.lon_col_combo, self.datetime_format_input]:
                        widget.setEnabled(True)
                    self.guess_common_columns(headers)
                    self.csv_mapping_group.setVisible(True)
                except Exception as e:
                    self.show_error(f"Error processing CSV headers: {e}")
                    self.log_message(f"CSV header processing error: {e}", logging.ERROR)
                    self.log_message(traceback.format_exc(), logging.DEBUG)
                    self.gps_file_path = None
                    self.gps_path_display.clear()
            elif not file_path.lower().endswith('.gpx'):
                self.log_message(f"Unknown GPS file type: {os.path.splitext(file_path)[1]}", logging.WARNING)
                self.show_error(f"Unsupported GPS file type: {os.path.splitext(file_path)[1]}. Please use GPX or CSV/TXT.")
                self.gps_file_path = None
                self.gps_path_display.clear()

            self._run_preliminary_gps_parse_and_map()
            self.update_transect_ui()
            self.update_sync_method_ui()

    def load_media(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.show_error("Cannot load new data while processing is active.")
            return
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Directory", self.media_path or "")
        if dir_path:
            self.media_path = dir_path
            self.media_type = 'images'
            self.media_path_display.setText(os.path.basename(dir_path))
            self.log_message(f"Image directory selected: {dir_path}")
            self.geotag_output_dir_base = f"geotagged_{os.path.basename(dir_path)}"
            self.geotag_images_button.setEnabled(False)

            self.log_message("Performing preliminary scan of image directory for sync purposes...")
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.preliminary_media_list = engine.get_image_files_and_times(self.media_path)
                self.sync_image_info = None
                self.sync_gps_point_info = None
                self.log_message(f"Preliminary scan found {len(self.preliminary_media_list)} images.")
            except Exception as e:
                self.log_message(f"Preliminary media scan failed: {e}", logging.ERROR)
                self.show_error(f"Failed to scan image directory for times: {e}")
                self.preliminary_media_list = []
            finally:
                QApplication.restoreOverrideCursor()
                self.update_sync_method_ui()
        else:
            self.log_message("Image directory selection cancelled.")
        self._update_batch_processing_ui_states()

    def update_timestamp_column_ui(self):
        if not hasattr(self, 'ts_single_radio') or not hasattr(self, 'csv_mapping_group'):
            return
        is_single = self.ts_single_radio.isChecked()
        csv_is_visible = self.csv_mapping_group.isVisible()

        self.datetime_col_label.setVisible(is_single and csv_is_visible)
        self.datetime_col_combo.setVisible(is_single and csv_is_visible)
        self.datetime_col_label.setEnabled(is_single and csv_is_visible)
        self.datetime_col_combo.setEnabled(is_single and csv_is_visible)

        self.date_col_label.setVisible(not is_single and csv_is_visible)
        self.date_col_combo.setVisible(not is_single and csv_is_visible)
        self.time_col_label.setVisible(not is_single and csv_is_visible)
        self.time_col_combo.setVisible(not is_single and csv_is_visible)
        self.date_col_label.setEnabled(not is_single and csv_is_visible)
        self.date_col_combo.setEnabled(not is_single and csv_is_visible)
        self.time_col_label.setEnabled(not is_single and csv_is_visible)
        self.time_col_combo.setEnabled(not is_single and csv_is_visible)
        self.datetime_format_input.setEnabled(csv_is_visible)

    def update_sync_method_ui(self):
        if not hasattr(self, 'sync_manual_radio'):
            return
        is_manual = self.sync_manual_radio.isChecked()
        is_point  = self.sync_point_radio .isChecked()
        is_visual = self.sync_visual_radio .isChecked()

        # Manual vs Point sync widgets
        self.offset_spinbox           .setEnabled(is_manual)
        self.gopro_time_edit          .setEnabled(is_point)
        self.gps_time_edit            .setEnabled(is_point)

        # Visual sync widgets
        for w in (
            self.select_sync_image_button,
            self.visual_sync_status_label,
            self.sync_image_time_display,
            self.sync_gps_point_time_display,
            self.sync_gps_point_coords_display,
            self.sync_image_preview_label
        ):
            w.setEnabled(is_visual)

        if is_visual:

            # 2) now your normal Visual Sync logic
            can_use_visual_sync = bool(self.preliminary_media_list) and \
                                (self.preliminary_gps_df is not None and not self.preliminary_gps_df.empty)
            self.select_sync_image_button.setEnabled(can_use_visual_sync)

            if not can_use_visual_sync:
                self.visual_sync_status_label.setText(
                    "Status: Load both GPS file and Image Directory first."
                )
            elif not self.sync_image_info:
                self.visual_sync_status_label.setText(
                    "Status: Ready. Please select a sync image."
                )
            elif not self.sync_gps_point_info:
                self.visual_sync_status_label.setText(
                    "Status: Image selected. Now click the corresponding point on the map."
                )
            else:
                self.visual_sync_status_label.setText(
                    "Status: Sync point selected. Ready to process."
                )
                # print (and log) the offset so you can type it in manually
                import pytz
                # localize the image timestamp to UI timezone:
                img_local = pytz.timezone(self.current_timezone_str) \
                                 .localize(self.sync_image_info['timestamp'])
                gps_utc  = self.sync_gps_point_info['timestamp']
                offset_s = (gps_utc - img_local).total_seconds()
                # log to your GUI & console:
                self.log_message(f"Calculated offset: {offset_s:.3f} s (UTC - local)", logging.INFO)
                # (you can now copy that number into the Manual Offset spinbox by hand)

                                # auto-calculate manual‐offset from the two timestamps,
                # fill the spin‐box and switch to Manual mode
                import pytz
                # 1) localize the image’s naive timestamp
                img_dt_local = pytz.timezone(self.current_timezone_str) \
                                  .localize(self.sync_image_info['timestamp'])
                # 2) grab the GPS UTC timestamp
                gps_dt_utc   = self.sync_gps_point_info['timestamp']
                # 3) compute delta in seconds
                delta = (gps_dt_utc - img_dt_local).total_seconds()
                # 4) populate the manual‐offset widget and flip the radio
                self.offset_spinbox.setValue(delta)
                self.sync_manual_radio.setChecked(True)
                # 5) update status to reflect the change
                self.visual_sync_status_label.setText(
                    f"Computed offset: {delta:.3f} s → switched to Manual Offset"
                )



        else:
            # Reset state when switching away
            self.sync_image_info = None
            self.sync_gps_point_info = None
            self.sync_image_preview_label.setText("No Image Selected")
            self.sync_image_time_display  .setText("N/A")
            self.sync_gps_point_time_display.setText("N/A")
            self.sync_gps_point_coords_display.setText("N/A")

        # finally, show/hide the whole Visual‐Sync group box
        self.visual_sync_group.setVisible(is_visual)


    def select_image_for_sync(self):
        if not self.preliminary_media_list:
            self.show_error("No images have been loaded. Please load an image directory first.")
            return

        image_filenames = [os.path.basename(item['file_path']) for item in self.preliminary_media_list if item.get('file_path')]
        chosen_filename, ok = QInputDialog.getItem(self, "Select Sync Image",
                                                   "Choose an image from the loaded directory:",
                                                   image_filenames, 0, False)

        if ok and chosen_filename:
            selected_item_info = next((item for item in self.preliminary_media_list if os.path.basename(item.get('file_path', '')) == chosen_filename), None)

            if selected_item_info and 'image_time' in selected_item_info and selected_item_info['image_time'] is not None:
                self.sync_image_info = {
                    'path': selected_item_info['file_path'],
                    'timestamp': selected_item_info['image_time'] # Naive datetime
                }
                local_tz_time = self.format_datetime_for_display(pytz.timezone(self.current_timezone_str).localize(self.sync_image_info['timestamp']))
                self.sync_image_time_display.setText(local_tz_time)

                pixmap = QPixmap(self.sync_image_info['path'])
                if not pixmap.isNull():
                    self.sync_image_preview_label.setPixmap(pixmap.scaled(self.sync_image_preview_label.size(),
                                                                           Qt.AspectRatioMode.KeepAspectRatio,
                                                                           Qt.TransformationMode.SmoothTransformation))
                else:
                    self.sync_image_preview_label.setText("Preview\nNot\nAvailable")

                self.sync_gps_point_info = None
                self.sync_gps_point_time_display.setText("N/A - Click on map")
                self.sync_gps_point_coords_display.setText("N/A - Click on map")
                self.update_sync_method_ui()
            else:
                self.show_error(f"The selected image '{chosen_filename}' is missing a valid timestamp. Please choose another.")
                self.sync_image_info = None

    def guess_common_columns(self, headers):
        self.log_message("Attempting to guess common CSV columns...")
        headers_lower_map = {h.lower().strip().replace(" ", "").replace("_", ""): h for h in headers}

        def find_set(patterns, combo):
            normalized_patterns = [p.lower().replace(" ", "").replace("_", "") for p in patterns]
            for norm_p in normalized_patterns:
                if norm_p in headers_lower_map:
                    target_header = headers_lower_map[norm_p]
                    idx = combo.findText(target_header, Qt.MatchFlag.MatchFixedString | Qt.MatchFlag.MatchCaseSensitive)
                    if idx < 0:
                        idx = combo.findText(target_header, Qt.MatchFlag.MatchFixedString)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                        self.log_message(f"Guessed '{target_header}' for {combo.objectName() if combo.objectName() else 'a combo box'}", logging.DEBUG)
                        return True
            if combo.count() > 0: # If no match, default to first item if combo is not empty
                 combo.setCurrentIndex(0)
            return False
        # Set object names for better logging if find_set is modified
        self.lat_col_combo.setObjectName("lat_col_combo")
        self.lon_col_combo.setObjectName("lon_col_combo")
        self.datetime_col_combo.setObjectName("datetime_col_combo")
        self.date_col_combo.setObjectName("date_col_combo")
        self.time_col_combo.setObjectName("time_col_combo")

        find_set(['latitude', 'lat', 'gps.lat', 'latitude dd', 'y', 'northing'], self.lat_col_combo)
        find_set(['longitude', 'lon', 'lng', 'gps.lon', 'longitude dd', 'x', 'easting'], self.lon_col_combo)
        guessed_datetime = find_set(['timestamp', 'datetime', 'datetimeutc', 'isodatetime', 'gpsdatetime', 'date time', 'time', 'datetimegmt', 'gpstimestamp', 'eventdatetime'], self.datetime_col_combo)
        guessed_date = find_set(['date', 'gpsdate', 'clock.currentdate', 'eventdate'], self.date_col_combo)
        guessed_time_col = find_set(['time', 'gpstime', 'utctime', 'clock.currentutctime', 'timeutc', 'eventtime'], self.time_col_combo)

        if guessed_datetime:
            self.ts_single_radio.setChecked(True)
            dt_col_text_raw = self.datetime_col_combo.currentText()
            dt_col_text_lower = dt_col_text_raw.lower() if dt_col_text_raw else ""
            # Try to guess format based on column name or typical formats
            if 'iso' in dt_col_text_lower or 'z' in dt_col_text_lower or ('t' in dt_col_text_raw and ':' in dt_col_text_raw and '-' in dt_col_text_raw) :
                self.datetime_format_input.setText("%Y-%m-%dT%H:%M:%S.%fZ") # Common ISO8601 with milliseconds and Z
            elif '.' in dt_col_text_raw and ':' in dt_col_text_raw and ('/' in dt_col_text_raw or '-' in dt_col_text_raw) : # Date, time, and milliseconds
                 self.datetime_format_input.setText("%d/%m/%Y %H:%M:%S.%f") # Default to d/m/Y, user can change
            elif '/' in dt_col_text_raw and ':' in dt_col_text_raw: # Date and time, no milliseconds
                 self.datetime_format_input.setText("%d/%m/%Y %H:%M:%S")
            elif '-' in dt_col_text_raw and ':' in dt_col_text_raw: # Date and time, Y-m-d
                 self.datetime_format_input.setText("%Y-%m-%d %H:%M:%S")
            else: # Fallback
                self.datetime_format_input.setText("%Y-%m-%d %H:%M:%S.%f")
        elif guessed_date and guessed_time_col: # Separate date and time found
            self.ts_separate_radio.setChecked(True)
            date_text_raw = self.date_col_combo.currentText()
            date_fmt = "%d/%m/%Y" # Default
            if date_text_raw and '-' in date_text_raw: # Y-m-d or m-d-Y etc.
                date_fmt = "%Y-%m-%d" # Assume Y-m-d, user can change
            time_fmt = "%H:%M:%S" # Assume HH:MM:SS, user can change if ms needed
            self.datetime_format_input.setText(f"{date_fmt} {time_fmt}")
        else: # No good guess for datetime or separate date/time
            self.ts_separate_radio.setChecked(True) # Default to separate
            self.datetime_format_input.setText("%d/%m/%Y %H:%M:%S") # Common default
        self.update_timestamp_column_ui()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def on_worker_error(self, message):
        self.log_message(f"Worker thread encountered an error: {message}", logging.ERROR)
        self.show_error(f"Processing Error Occurred:\n{message}")
        self.process_button.setEnabled(True)
        if self.worker_thread:
            self.worker_thread.quit()
            if not self.worker_thread.wait(500):
                self.worker_thread.terminate()
                self.worker_thread.wait()
            self.worker_thread.deleteLater()
            self.worker_thread = None
        self.update_transect_ui()
        self._update_batch_processing_ui_states()

    def on_worker_finished(self):
        self.log_message("Worker thread has finished signal received.", logging.INFO)
        if self.worker_thread:
            self.worker_thread.deleteLater()
            self.worker_thread = None
        # Only re-enable process button if progress is not 100 (implies error or early finish)
        # Or if it was disabled for other reasons. Processing_finished will handle success.
        if self.progress_bar.value() < 100 or not self.process_button.isEnabled():
            self.process_button.setEnabled(True)
        self.update_transect_ui()
        self._update_batch_processing_ui_states()

    def display_map(self, map_html_path):
        self.log_message(f"Attempting to display map: {map_html_path}")
        self._map_file_path = map_html_path
        if map_html_path and os.path.exists(map_html_path):
            try:
                abs_path = os.path.abspath(map_html_path)
                url = QUrl.fromLocalFile(abs_path)
                if url.isValid() and url.scheme() == 'file':
                    try:
                        QWebEngineProfile.defaultProfile().clearHttpCache()
                        self.log_message("Cleared WebEngine HTTP cache.", logging.DEBUG)
                    except Exception as e:
                        self.log_message(f"Could not clear WebEngine cache: {e}", logging.WARNING)
                    self.map_view.setUrl(url)
                    self.save_map_button.setEnabled(True)
                    self.log_message("Map loaded successfully.")
                    self.output_tabs.setCurrentWidget(self.map_view)
                else:
                    err_msg = f"Invalid map URL generated: {url.errorString() if not url.isValid() else 'Scheme not file'}"
                    self.log_message(err_msg, logging.ERROR)
                    if self.map_view and self.map_view.page():
                        self.map_view.setHtml(f"<h2>Map Error</h2><p>{err_msg}</p>")
                    self.save_map_button.setEnabled(False)
            except Exception as e:
                self.log_message(f"Error displaying map: {e}", logging.ERROR)
                self.log_message(traceback.format_exc(), logging.DEBUG)
                if self.map_view and self.map_view.page():
                    self.map_view.setHtml(f"<h2>Map Error</h2><p>Failed to load map view: {e}</p>")
                self.save_map_button.setEnabled(False)
        else:
            msg = f"Map display failed: Path invalid or file not found ('{map_html_path}')."
            self.log_message(msg, logging.WARNING)
            if self.map_view and self.map_view.page():
                 self.map_view.setHtml(f"<h2>Map Error</h2><p>{msg}</p>")
            self.save_map_button.setEnabled(False)

    def start_processing(self):
        self.log_message("Start Processing button clicked.")
        if not ENGINE_AVAILABLE:
            self.show_error("Georeference engine (georeference_engine.py) not loaded.")
            return
        if not self.gps_file_path or not os.path.exists(self.gps_file_path):
            self.show_error("Load GPS file first.")
            return

        # --- Determine Sync Method and Params ---
        sync_method = "manual"
        if self.sync_point_radio.isChecked():
            sync_method = "sync_point"
        elif self.sync_visual_radio.isChecked():
            sync_method = "sync_point"

        sync_params = {}
        media_data_tz_str = self.current_timezone_str
        try:
            pytz.timezone(media_data_tz_str)
        except pytz.UnknownTimeZoneError:
            self.show_error(f"Invalid Data Timezone selected: '{media_data_tz_str}'. Please choose a valid timezone.")
            return
        sync_params['boat_timezone_str'] = media_data_tz_str

        if self.sync_manual_radio.isChecked():
            sync_params["offset_seconds"] = self.offset_spinbox.value()
        elif self.sync_point_radio.isChecked():
            if not self.media_path:
                self.show_error("Sync Point method requires an Image Directory to be loaded.")
                return
            try:
                media_dt_naive = self.gopro_time_edit.dateTime().toPyDateTime().replace(tzinfo=None)
                gps_dt_utc = self.gps_time_edit.dateTime().toPyDateTime()
                if gps_dt_utc.tzinfo is None or gps_dt_utc.tzinfo.utcoffset(gps_dt_utc) != timedelta(0):
                    gps_dt_utc = pytz.utc.localize(gps_dt_utc.replace(tzinfo=None))
                sync_params["gopro_sync_time"] = media_dt_naive
                sync_params["gps_sync_time"] = gps_dt_utc
            except Exception as e:
                self.show_error(f"Invalid Sync Point time entered: {e}")
                return
        elif self.sync_visual_radio.isChecked():
            if not self.sync_image_info or not self.sync_gps_point_info:
                self.show_error("For Visual Sync, you must select an image AND a corresponding point on the map before processing.")
                return
            sync_params["gopro_sync_time"] = self.sync_image_info['timestamp'] # Naive datetime
            sync_params["gps_sync_time"] = self.sync_gps_point_info['timestamp'] # UTC datetime

        # --- Get GPS Parsing Params ---
        gps_params = self._get_gps_parsing_params_from_ui()
        if not gps_params:
            return # Error was already shown in helper function

        # --- Reset state and start worker ---
        self.media_type = 'images' if self.media_path and os.path.isdir(self.media_path) else None
        self.results_df = None
        self.processed_gps_df = None
        self.results_table.setRowCount(0)
        self.saved_transects = []
        self._populate_loaded_transects_list()
        self.export_csv_button.setEnabled(False)
        self.save_map_button.setEnabled(False)
        self.geotag_images_button.setEnabled(False)
        self.extract_button.setEnabled(False)
        self.clear_transect_definition()
        self.progress_bar.setValue(0)
        self.reset_progress_bar_color()
        self.output_tabs.setCurrentWidget(self.log_output)

        self.log_message("--- Starting Data Processing ---")
        if self.worker_thread and self.worker_thread.isRunning():
            self.show_error("Processing is already running. Please wait.")
            return

        self.process_button.setEnabled(False)
        QApplication.processEvents()

        self.worker_thread = WorkerThread(
            gps_path=self.gps_file_path,
            gps_parsing_params=gps_params,
            media_path=self.media_path,
            media_type=self.media_type,
            frame_interval=0,
            sync_method=sync_method,
            sync_params=sync_params
        )
        self.worker_thread.progress.connect(self.update_progress)
        self.worker_thread.log_message.connect(lambda msg: self.log_message(msg, logging.INFO))
        self.worker_thread.results_ready.connect(self.processing_finished)
        self.worker_thread.error_occurred.connect(self.on_worker_error)
        self.worker_thread.map_ready.connect(self.display_map)
        self.worker_thread.finished.connect(self.on_worker_finished)
        self.worker_thread.start()

    def calculate_transect_length(self, start_time_utc, end_time_utc):
        if not GEOPY_AVAILABLE:
            self.log_message("Geopy not available, cannot calculate transect length.", logging.WARNING)
            return None
        if start_time_utc is None or end_time_utc is None:
            self.log_message("Start or end time for transect length calculation is None.", logging.DEBUG)
            return None

        active_gps_df = self.get_active_gps_df()
        if active_gps_df.empty:
            self.log_message("Cannot calculate length: Active GPS data is empty.", logging.WARNING)
            return None
        if 'latitude' not in active_gps_df.columns or 'longitude' not in active_gps_df.columns:
            self.log_message("Cannot calculate length: Missing lat/lon columns in active GPS.", logging.ERROR)
            return None
        try:
            # Ensure times are UTC and datetime objects
            if isinstance(start_time_utc, str): start_time_utc = datetime.fromisoformat(start_time_utc.replace("Z", "+00:00"))
            if isinstance(end_time_utc, str): end_time_utc = datetime.fromisoformat(end_time_utc.replace("Z", "+00:00"))

            if start_time_utc.tzinfo is None or start_time_utc.tzinfo.utcoffset(start_time_utc) is None:
                 start_time_utc = pytz.utc.localize(start_time_utc)
            elif start_time_utc.tzinfo.utcoffset(start_time_utc) != timedelta(0):
                 start_time_utc = start_time_utc.astimezone(pytz.utc)
            if end_time_utc.tzinfo is None or end_time_utc.tzinfo.utcoffset(end_time_utc) is None:
                 end_time_utc = pytz.utc.localize(end_time_utc)
            elif end_time_utc.tzinfo.utcoffset(end_time_utc) != timedelta(0):
                 end_time_utc = end_time_utc.astimezone(pytz.utc)
            try:
                start_point_data = active_gps_df.loc[start_time_utc]
                end_point_data = active_gps_df.loc[end_time_utc]
            except KeyError as e:
                self.log_message(f"Could not find one or both transect points ({start_time_utc}, {end_time_utc}) "
                                 f"directly in active GPS data index for length calculation: {e}. Trying nearest.", logging.WARNING)
                start_idx = active_gps_df.index.get_indexer([start_time_utc], method='nearest')[0]
                end_idx = active_gps_df.index.get_indexer([end_time_utc], method='nearest')[0]
                if start_idx == -1 or end_idx == -1 :
                    self.log_message("Fallback to nearest also failed for length calculation.", logging.ERROR)
                    return None
                start_point_data = active_gps_df.iloc[start_idx]
                end_point_data = active_gps_df.iloc[end_idx]
            point1_coords = (start_point_data['latitude'], start_point_data['longitude'])
            point2_coords = (end_point_data['latitude'], end_point_data['longitude'])
            if pd.isna(point1_coords[0]) or pd.isna(point1_coords[1]) or \
               pd.isna(point2_coords[0]) or pd.isna(point2_coords[1]):
                self.log_message("Invalid (NaN) coordinates for one or both transect points for length calculation.", logging.WARNING)
                return None
            length = geodesic(point1_coords, point2_coords).meters
            self.log_message(f"Calculated transect length: {length:.2f} m between {start_time_utc} and {end_time_utc}", logging.INFO)
            return length
        except Exception as e:
            self.log_message(f"Error calculating transect length: {e}", logging.ERROR)
            self.log_message(traceback.format_exc(), logging.DEBUG)
            return None
    def add_defined_transect_to_unlabeled_list(self):
        """
        Takes the currently defined transect, adds it to the unlabeled queue,
        and resets the UI for the next definition.
        """
        self.log_message("Adding defined transect to the labeling queue.")
        if self.transect_start_time is None or self.transect_end_time is None:
            self.show_error("Internal Error: Tried to add an incomplete transect to the queue.")
            return

        start_utc = min(self.transect_start_time.astimezone(pytz.utc), self.transect_end_time.astimezone(pytz.utc))
        end_utc = max(self.transect_start_time.astimezone(pytz.utc), self.transect_end_time.astimezone(pytz.utc))

        # Add to the queue for later processing
        self.unlabeled_transects.append({
            "start": start_utc,
            "end": end_utc,
            "length": self.transect_length_meters,
        })
        self.log_message(f"Transect from {start_utc.isoformat()} to {end_utc.isoformat()} added to queue. Queue size: {len(self.unlabeled_transects)}.")

        # --- User Feedback & Reset ---
        # Update the status label to show it was added
        self.transect_status_label.setText(f"Status: Transect added to queue ({len(self.unlabeled_transects)} pending). Define next transect.")
        QTimer.singleShot(2500, self.update_transect_ui) # Reset status message after 2.5s

        # Clear the single transect definition UI to allow the user to immediately define the next one
        self.clear_transect_definition()
        # Update UI states (e.g., enable the "Label Queued Transects" button)
        self.update_transect_ui()


    def process_unlabeled_transects(self):
        """
        Iterates through the queue of unlabeled transects, prompting the user
        to name and save images for each one.
        """
        self.log_message(f"Starting to process queue of {len(self.unlabeled_transects)} unlabeled transects.")
        if not self.unlabeled_transects:
            self.show_error("There are no queued transects to label.")
            return

        if self.results_df is None or self.results_df.empty:
            self.show_error("No georeferenced image data available to filter for the transects.")
            return
        
        active_gps_df = self.get_active_gps_df()
        if active_gps_df.empty:
            self.show_error("No active GPS data to get coordinates from for saving.")
            return

        # Determine a common base output directory
        if self.media_path and os.path.isdir(self.media_path):
            base_output_dir_for_transects = self.media_path
        elif self.gps_file_path and os.path.isfile(self.gps_file_path):
            base_output_dir_for_transects = os.path.dirname(self.gps_file_path)
        else:
            base_output_dir_for_transects = os.getcwd()
        transects_output_parent_dir = os.path.join(base_output_dir_for_transects, "transects_output")

        processed_transects_this_session = []
        cancelled = False

        # Process a copy and clear the original later
        queue_to_process = list(self.unlabeled_transects)
        self.unlabeled_transects.clear()

        for i, transect_to_label in enumerate(queue_to_process):
            start_utc = transect_to_label["start"]
            end_utc = transect_to_label["end"]
            length_m = transect_to_label["length"]

            # Filter images for this specific transect
            mask = (
                self.results_df['Corrected Timestamp (UTC)'].notna() &
                (self.results_df['Corrected Timestamp (UTC)'] >= start_utc) &
                (self.results_df['Corrected Timestamp (UTC)'] <= end_utc)
            )
            images_in_transect_df = self.results_df[mask]
            image_count = len(images_in_transect_df)

            self.log_message(f"Processing queued transect {i+1}/{len(queue_to_process)}: Found {image_count} images.")

            # Prompt for name
            s_time_str = start_utc.strftime('%Y%m%d_%H%M%S')
            length_str = f"{length_m:.0f}m" if length_m is not None else "lenNA"
            suggested_transect_name = f"Transect_{s_time_str}_{length_str}"

            transect_name_input, ok = QInputDialog.getText(self, f"Labeling Transect {i+1} of {len(queue_to_process)}",
                                                         f"Found {image_count} images for this transect.\n"
                                                         f"Enter a name to save the images and definition.\n\n"
                                                         f"Output subfolder will be inside:\n'{transects_output_parent_dir}'",
                                                         QLineEdit.EchoMode.Normal, suggested_transect_name)

            if not ok:
                self.log_message("User cancelled the labeling process.")
                # Add the remaining unprocessed transects back to the queue
                self.unlabeled_transects.extend(queue_to_process[i:])
                cancelled = True
                break # Exit the loop

            if not transect_name_input.strip():
                self.log_message(f"Skipping transect {i+1} as no name was provided.")
                continue # Skip to the next transect in the queue

            # --- Save the images for this named transect ---
            safe_transect_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in transect_name_input).strip()
            specific_transect_output_dir = os.path.join(transects_output_parent_dir, safe_transect_name)
            os.makedirs(specific_transect_output_dir, exist_ok=True)

            copy_ok_count = 0
            if image_count > 0:
                for _, row in images_in_transect_df.iterrows():
                    identifier = row.get('Identifier')
                    source_path = self.identifier_to_path_map.get(identifier)
                    if source_path and os.path.exists(source_path):
                        try:
                            shutil.copy2(source_path, specific_transect_output_dir)
                            copy_ok_count += 1
                        except Exception as copy_e:
                            self.log_message(f"Error copying image ID '{identifier}' for transect '{safe_transect_name}': {copy_e}", logging.ERROR)

            self.log_message(f"Saved {copy_ok_count} images for transect '{safe_transect_name}'.")

            # Get coordinates for saving
            try:
                start_point = active_gps_df.loc[start_utc]
                end_point = active_gps_df.loc[end_utc]
            except KeyError:
                self.log_message(f"Could not find start/end times in GPS data for transect '{safe_transect_name}'. Cannot save its definition.", logging.ERROR)
                continue

            # Add the now-named transect to the main 'saved_transects' list with coordinates
            final_transect_def = {
                "name": safe_transect_name,
                "start_lat": start_point['latitude'],
                "start_lon": start_point['longitude'],
                "end_lat": end_point['latitude'],
                "end_lon": end_point['longitude'],
                "length": length_m,
            }
            self.saved_transects.append(final_transect_def)
            processed_transects_this_session.append(safe_transect_name)

        # --- Final UI Updates and Summary ---
        self.log_message("Finished processing labeling queue.")
        summary_title = "Labeling Cancelled" if cancelled else "Labeling Complete"
        summary_message = (f"Processed {len(processed_transects_this_session)} transect(s).\n\n"
                           f"The definitions and their images have been saved.\n"
                           f"Remaining in queue: {len(self.unlabeled_transects)}")
        QMessageBox.information(self, summary_title, summary_message)

        # Update the batch list and other UI elements
        self._populate_loaded_transects_list()
        self._update_batch_processing_ui_states()
        self.update_transect_ui()
        # Finally, regenerate the map to show the newly saved (orange) transect lines
        self.regenerate_map_display()


    def highlight_map_range(self, start_utc, end_utc):
        active_gps_df = self.get_active_gps_df()
        if active_gps_df.empty:
            self.log_message("Cannot highlight map: Active GPS data not available or empty.", logging.DEBUG)
            self.clear_map_highlight()
            return
        if self._map_file_path is None or not os.path.exists(self._map_file_path) or not (self.map_view and self.map_view.page()):
            self.log_message("Cannot highlight map: Map not generated, loaded, or page not available.", logging.DEBUG)
            return

        t_highlight_start, t_highlight_end = None, None
        if start_utc and end_utc:
            t_highlight_start = min(start_utc.astimezone(pytz.utc), end_utc.astimezone(pytz.utc))
            t_highlight_end = max(start_utc.astimezone(pytz.utc), end_utc.astimezone(pytz.utc))
        elif start_utc:
            t_highlight_start = start_utc.astimezone(pytz.utc)
            t_highlight_end = t_highlight_start
        elif end_utc: # Should not happen if start_utc is None but end_utc is not, due to logic flow
            t_highlight_start = end_utc.astimezone(pytz.utc)
            t_highlight_end = t_highlight_start

        if t_highlight_start is None: # No valid times to highlight
            self.clear_map_highlight()
            return

        self.log_message(f"Requesting map highlight for range: {t_highlight_start.isoformat()} to {t_highlight_end.isoformat()}", logging.DEBUG)
        try:
            mask = (active_gps_df.index >= t_highlight_start) & (active_gps_df.index <= t_highlight_end)
            timestamps_in_range_df = active_gps_df[mask]

            if timestamps_in_range_df.empty:
                if t_highlight_start == t_highlight_end: # Single point selection, try nearest if exact match fails
                    idx = active_gps_df.index.get_indexer([t_highlight_start], method='nearest', tolerance=timedelta(seconds=1))[0]
                    if idx != -1:
                        ts_to_highlight = active_gps_df.index[idx]
                        timestamps_iso_list = [ts_to_highlight.isoformat(timespec='milliseconds').replace('+00:00', 'Z')]
                    else:
                        self.log_message("No GPS points found in the specified range for highlighting (single point not found).", logging.DEBUG)
                        self.clear_map_highlight()
                        return
                else:
                    self.log_message("No GPS points found in the specified range for highlighting (range yielded no points).", logging.DEBUG)
                    self.clear_map_highlight()
                    return
            else:
                 timestamps_iso_list = [ts.isoformat(timespec='milliseconds').replace('+00:00', 'Z') for ts in timestamps_in_range_df.index]

            js_array_string = json.dumps(timestamps_iso_list)
            js_call = f"highlightTransectRange({js_array_string});"
            self.log_message(f"Sending {len(timestamps_iso_list)} timestamps to JS for highlighting.", logging.DEBUG)
            self.map_view.page().runJavaScript(js_call)
        except Exception as e:
            self.log_message(f"Error preparing or sending map highlight command: {e}", logging.ERROR)
            self.log_message(traceback.format_exc(), logging.DEBUG)
            self.clear_map_highlight() # Attempt to reset highlight on error

    def clear_map_highlight(self):
        if self._map_file_path is None or not os.path.exists(self._map_file_path) or not (self.map_view and self.map_view.page()):
            self.log_message("Cannot clear map highlight: Map not generated, loaded, or page not available.", logging.DEBUG)
            return
        try:
            self.log_message("Clearing map highlight via JS.", logging.DEBUG)
            js_call = "highlightTransectRange([]);" # Call with empty array to clear
            self.map_view.page().runJavaScript(js_call)
        except Exception as e:
            self.log_message(f"Error sending map highlight clear command: {e}", logging.ERROR)

    def generate_interactive_map_for_app(self, gps_df_to_map, results_df_to_map):
            self.log_message("[App] --- Generating interactive map ---")
            valid_gps_df_map = pd.DataFrame()
            has_valid_gps = False
            if gps_df_to_map is not None and not gps_df_to_map.empty:
                temp = gps_df_to_map.copy()
                if all(col in temp.columns for col in ['latitude','longitude']):
                    temp['latitude'] = pd.to_numeric(temp['latitude'], errors='coerce')
                    temp['longitude'] = pd.to_numeric(temp['longitude'], errors='coerce')
                    valid_gps_df_map = temp.dropna(subset=['latitude','longitude']).copy()
                    if isinstance(valid_gps_df_map.index, pd.DatetimeIndex) and \
                       valid_gps_df_map.index.tz is not None and \
                       valid_gps_df_map.index.tz.utcoffset(valid_gps_df_map.index.min()) == timedelta(0): # Ensure UTC
                        if not valid_gps_df_map.index.is_monotonic_increasing:
                             valid_gps_df_map.sort_index(inplace=True)
                        if not valid_gps_df_map.empty:
                            has_valid_gps = True
                            self.log_message(f"[App] Using {len(valid_gps_df_map)} valid GPS points for map.")
                    else:
                        self.log_message("[App] GPS DF for map has invalid or non-UTC index.", logging.WARNING)

            valid_results_df_map = pd.DataFrame()
            has_valid_results = False
            if results_df_to_map is not None and not results_df_to_map.empty:
                temp_res = results_df_to_map.copy()
                req_cols = ['Latitude','Longitude','Corrected Timestamp (UTC)','Identifier']
                if all(c in temp_res.columns for c in req_cols):
                    if not pd.api.types.is_datetime64_any_dtype(temp_res['Corrected Timestamp (UTC)']) or \
                       temp_res['Corrected Timestamp (UTC)'].dt.tz is None:
                        temp_res['TS_dt'] = pd.to_datetime(
                            temp_res['Corrected Timestamp (UTC)'].astype(str).str.replace('Z','+00:00', regex=False),
                            utc=True, errors='coerce'
                        )
                    else: # Already datetime and hopefully UTC
                        temp_res['TS_dt'] = temp_res['Corrected Timestamp (UTC)'].dt.tz_convert('UTC') if temp_res['Corrected Timestamp (UTC)'].dt.tz is not None else temp_res['Corrected Timestamp (UTC)']


                    temp_res['Latitude']=pd.to_numeric(temp_res['Latitude'], errors='coerce')
                    temp_res['Longitude']=pd.to_numeric(temp_res['Longitude'], errors='coerce')
                    valid_results_df_map = temp_res.dropna(subset=['Latitude','Longitude','TS_dt']).copy()
                    if not valid_results_df_map.empty:
                        has_valid_results = True
                        self.log_message(f"[App] Using {len(valid_results_df_map)} valid results for map.")
                else:
                    self.log_message(f"[App] Results DF for map missing one of required cols: {req_cols}", logging.WARNING)

            if not has_valid_gps and not has_valid_results:
                self.log_message("[App] No valid data to plot on map.", logging.WARNING)
                return None

            center, zoom = [0,0], 2
            try:
                if has_valid_gps:
                    mean_lat, mean_lon = valid_gps_df_map['latitude'].mean(), valid_gps_df_map['longitude'].mean()
                    if pd.notna(mean_lat) and pd.notna(mean_lon): center, zoom = [mean_lat, mean_lon], 15
                elif has_valid_results: # Fallback to results if no GPS
                    mean_lat, mean_lon = valid_results_df_map['Latitude'].mean(), valid_results_df_map['Longitude'].mean()
                    if pd.notna(mean_lat) and pd.notna(mean_lon): center, zoom = [mean_lat, mean_lon], 15
            except Exception as e:
                self.log_message(f"[App] Error calculating map center: {e}", logging.WARNING)

            m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Tiles © Esri', name='Esri Satellite', overlay=False, control=True
            ).add_to(m)

            js_code_for_map = """
    <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script>
window.savedTransectLines = window.savedTransectLines || []; // Ensure it's initialized
window.addEventListener("DOMContentLoaded", function() {
    if (window.qt && qt.webChannelTransport) {
        new QWebChannel(qt.webChannelTransport, function(channel) {
            window.pyHandler = channel.objects.pyHandler;
            console.log("JS: pyHandler connected.");
        });
    }
});
function sendClickedGpsTimestampToPython(ts, ctrl, shift) {
    if (window.pyHandler && window.pyHandler.handleGpsPointClick) {
        window.pyHandler.handleGpsPointClick(ts, ctrl, shift);
    }
}
function highlightTransectRange(timestamps) {
    var map = Object.values(window).find(x => x instanceof L.Map);
    if (!map) return;
    map.eachLayer(function(layer) {
        if (layer.options && layer.options.customTimestamp) { // Check if it's a GPS point marker
            var isHl = timestamps.indexOf(layer.options.customTimestamp) !== -1;
            layer.setStyle({
                color: isHl? 'lime':'blue', // Highlighted vs default GPS point color
                fillColor: isHl? 'lime':'blue',
                fillOpacity: isHl? 0.9:0.6,
                weight: isHl? 3:1 // Thicker if highlighted
            });
            if (layer.setRadius) layer.setRadius(isHl? 6:4); // Larger if highlighted
        }
    });
}
// This now only handles the temporary YELLOW line during definition
function drawTransectLine(lat1, lon1, lat2, lon2) {
    var map = Object.values(window).find(x => x instanceof L.Map);
    if (!map) return;
    if (window.transectLine) { // Always clear the previous temp line
        map.removeLayer(window.transectLine);
    }
    window.transectLine = L.polyline([[lat1, lon1], [lat2, lon2]], { color: 'yellow', weight: 2, interactive: false }).addTo(map);
}
// NEW FUNCTION: Draws a permanent ORANGE line for a saved transect
function drawSavedTransectLine(lat1, lon1, lat2, lon2, name) {
    var map = Object.values(window).find(x => x instanceof L.Map);
    if (!map) return;
    const newLine = L.polyline([[lat1, lon1], [lat2, lon2]], { color: 'orange', weight: 3, opacity: 0.8 })
        .bindTooltip("Saved: " + name)
        .addTo(map);
    window.savedTransectLines.push(newLine); // Add to our list to keep track
}
function clearTransectLine() { // Clears the single, active (yellow) transect line
    var map = Object.values(window).find(x => x instanceof L.Map);
    if (map && window.transectLine) {
        map.removeLayer(window.transectLine);
        window.transectLine = null;
    }
}
function drawPresetLengthCircle(lat, lon, r, action) {
    var map = Object.values(window).find(x => x instanceof L.Map);
    if (!map) return;
    if (window.presetRadiusCircle) { // Clear existing circle first
        map.removeLayer(window.presetRadiusCircle);
        window.presetRadiusCircle = null;
    }
    if (action==='draw' && lat!=null && lon!=null && r>0) {
        window.presetRadiusCircle = L.circle([lat, lon], {
            radius: r, // in meters
            color: 'orange', dashArray: '5,5', weight:2,
            fillColor: 'orange', fillOpacity:0.1,
            interactive: false
        }).addTo(map);
    }
}
function clearAllSavedTransectLines() {
    var map = Object.values(window).find(x => x instanceof L.Map);
    if (!map) return;
    if (window.savedTransectLines && window.savedTransectLines.length > 0) {
        for (var i = 0; i < window.savedTransectLines.length; i++) {
            map.removeLayer(window.savedTransectLines[i]);
        }
        window.savedTransectLines = []; // Clear the array
    }
    if (window.transectLine) {
        map.removeLayer(window.transectLine);
        window.transectLine = null;
    }
}
</script>
        """
            m.get_root().html.add_child(folium.Element(js_code_for_map))

            extra_click_binding = """
                <script>
                // once the channel is ready, walk all layers and bind marker clicks
                function registerMarkerClicks() {
                // if pyHandler not ready, try again shortly
                if (!window.pyHandler) {
                    setTimeout(registerMarkerClicks, 50);
                    return;
                }
                // find the Leaflet map instance
                var map = Object.values(window).find(function(o){ return o instanceof L.Map; });
                if (!map) return;
                map.eachLayer(function(layer){
                    // our GPS point markers all have a customTimestamp option
                    if (layer.options && layer.options.customTimestamp) {
                    // remove any old handler just in case
                    layer.off('click');
                    // when the user clicks the dot, send the timestamp straight to Python
                    layer.on('click', function(e){
                        window.pyHandler.handleGpsPointClick(
                        layer.options.customTimestamp,
                        e.originalEvent.ctrlKey,
                        e.originalEvent.shiftKey
                        );
                    });
                    }
                });
                }
                // wait for the DOM (and web channel) to be ready
                window.addEventListener("DOMContentLoaded", registerMarkerClicks);
                </script>
                """
            m.get_root().html.add_child(folium.Element(extra_click_binding))

            if has_valid_gps:
                gps_group = folium.FeatureGroup(name="GPS Track (Click for Transect)", show=True, overlay=True).add_to(m)
                for ts, row in valid_gps_df_map.iterrows():
                    ts_iso = ts.isoformat(timespec='milliseconds').replace('+00:00','Z')
                    ts_tip = ts.strftime('%H:%M:%S.%f')[:-3] + 'Z'
                    js_call = f"sendClickedGpsTimestampToPython('{ts_iso}',event.ctrlKey,event.shiftKey); var m=null;for(var k in window){{if(window[k] instanceof L.Map){{m=window[k];break;}}}} if(m){{m.closePopup();}};"
                    escaped_js = js_call.replace('"', '&quot;')
                    popup_html = (f'<b>GPS Point</b><br>'
                                  f'Time: {ts_tip}<br>'
                                  f'Lat: {row["latitude"]:.6f}, Lon: {row["longitude"]:.6f}<br>'
                                  f'<button type="button" onclick="{escaped_js}">Select This Point</button>')
                    folium.CircleMarker([row['latitude'],row['longitude']], radius=4, color='blue', weight=1, fill=True, fill_color='blue', fill_opacity=0.6,
                                        popup=folium.Popup(popup_html, max_width=250), tooltip=f"Click to open: {ts_tip}",
                                        options={'customTimestamp': ts_iso}).add_to(gps_group)
            if has_valid_results:
                img_group = folium.plugins.MarkerCluster(name="Images", show=True, overlay=True).add_to(m)
                for _, row in valid_results_df_map.iterrows():
                    ts_dt = row['TS_dt'] # Already converted to datetime, should be UTC
                    popup = f"ID:{row['Identifier']}<br>Time:{ts_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ' Z' if pd.notna(ts_dt) else 'N/A'}<br>Lat:{row['Latitude']:.7f},Lon:{row['Longitude']:.7f}"
                    folium.CircleMarker([row['Latitude'], row['Longitude']], radius=4, color='red', weight=1, fill=True, fill_color='red', fill_opacity=0.7,
                                        popup=folium.Popup(popup, max_width=300), tooltip=f"Img: {row['Identifier']}").add_to(img_group)

            if hasattr(self, 'saved_transects') and self.saved_transects:
                saved_lines_group = folium.FeatureGroup(name="Saved Session Transects", show=True, overlay=True).add_to(m)
                for tx_info in self.saved_transects:
                    try:
                        start_lat = float(tx_info["start_lat"])
                        start_lon = float(tx_info["start_lon"])
                        end_lat = float(tx_info["end_lat"])
                        end_lon = float(tx_info["end_lon"])
                        tx_name_display = tx_info.get('name', 'Unnamed Transect')
                        tx_length_display = f"{float(tx_info.get('length')):.1f}m" if tx_info.get('length') not in (None, '', 'N/A') else "N/A"

                        # Draw coordinate CSV transects using their actual coordinates.
                        # Do not require snapping to a GPS point just to display them.
                        folium.PolyLine(
                            [(start_lat, start_lon), (end_lat, end_lon)],
                            color='orange', weight=3, opacity=0.8,
                            tooltip=f"Saved: {tx_name_display}",
                            popup=folium.Popup(
                                f"<b>{tx_name_display}</b><br>"
                                f"Start: {start_lat:.7f}, {start_lon:.7f}<br>"
                                f"End: {end_lat:.7f}, {end_lon:.7f}<br>"
                                f"Length: {tx_length_display}",
                                max_width=300
                            )
                        ).add_to(saved_lines_group)
                    except Exception as e_draw_saved:
                        self.log_message(f"Error drawing saved transect '{tx_info.get('name')}' on map: {e_draw_saved}", logging.WARNING)

            folium.LayerControl().add_to(m)
            fd, temp_map = tempfile.mkstemp(suffix='.html', prefix='geotagger_map_app_')
            os.close(fd)
            m.save(temp_map)
            self.log_message(f"[App] Interactive map saved to: {temp_map}")
            return temp_map

    def populate_results_table(self, df_to_display):
        if not hasattr(self, 'results_table'):
            self.log_message("Populate table error: Table widget missing.", logging.ERROR)
            return
        self.results_table.blockSignals(True)
        self.results_table.setSortingEnabled(False)
        self.results_table.clearSelection()
        self.results_table.clearContents()
        self.results_table.setRowCount(0)
        self.extract_button.setEnabled(False)
        target_hdrs = ["Identifier", "File Path", "Original Timestamp", "Corrected Timestamp (UTC)", "Latitude", "Longitude"]
        self.results_table.setColumnCount(len(target_hdrs))
        self.results_table.setHorizontalHeaderLabels(target_hdrs)

        if df_to_display is None or df_to_display.empty:
            self.log_message("Populating results table with empty or None dataset.", logging.DEBUG)
            self.results_table.setSortingEnabled(True)
            self.results_table.blockSignals(False)
            return
        try:
            # Ensure all target headers are present or use available ones, logging discrepancies
            actual_df_cols = df_to_display.columns.tolist()
            display_headers = []
            for th in target_hdrs:
                if th in actual_df_cols:
                    display_headers.append(th)
                else:
                    self.log_message(f"Target header '{th}' not found in DataFrame for table display.", logging.DEBUG)

            if not display_headers:
                self.log_message("No target headers found in DataFrame. Cannot populate table.", logging.ERROR)
                self.results_table.setSortingEnabled(True)
                self.results_table.blockSignals(False)
                return

            self.results_table.setColumnCount(len(display_headers))
            self.results_table.setHorizontalHeaderLabels(display_headers)

            self.results_table.setRowCount(len(df_to_display))
            for r_idx, (_, row_data) in enumerate(df_to_display.iterrows()):
                for c_idx, header_name in enumerate(display_headers): # Use the filtered display_headers
                    value = row_data.get(header_name, pd.NA) # Default to pd.NA if somehow still missing
                    display_string = ""
                    try:
                        if pd.isna(value):
                            display_string = ""
                        elif header_name in ['Latitude', 'Longitude']:
                            display_string = f"{float(value):.7f}" if pd.notna(value) else ""
                        elif header_name == 'Corrected Timestamp (UTC)' and isinstance(value, (datetime, pd.Timestamp)):
                            if value.tzinfo is None:
                                value = pytz.utc.localize(value) # Assume UTC if naive
                            else: # Ensure it's converted to UTC for display consistency
                                value = value.astimezone(pytz.utc)
                            display_string = value.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " Z" if pd.notna(value) else ""
                        else:
                            display_string = str(value)
                    except (ValueError, TypeError) as format_err:
                        self.log_message(f"Error formatting value '{value}' for column '{header_name}' at row {r_idx}: {format_err}", logging.DEBUG)
                        display_string = str(value) if pd.notna(value) else "" # Fallback to string or empty

                    item = QTableWidgetItem(display_string)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.results_table.setItem(r_idx, c_idx, item)
            self.results_table.resizeColumnsToContents()
        except Exception as e:
            self.log_message(f"Error during results table population: {e}", logging.ERROR)
            self.log_message(traceback.format_exc(), logging.DEBUG)
            self.results_table.setRowCount(0) # Clear on error
        finally:
            self.results_table.setSortingEnabled(True)
            self.results_table.blockSignals(False)

    def export_csv(self):
        if self.results_table.rowCount() == 0:
            self.show_error("No data in table to export.")
            return
        bn="geotagger_results"
        if self.transect_start_time and self.transect_end_time:
            start_str = min(self.transect_start_time, self.transect_end_time).strftime('%Y%m%d_%H%M%S')
            end_str = max(self.transect_start_time, self.transect_end_time).strftime('%H%M%S') # Short form for end
            length_str = f"{self.transect_length_meters:.0f}m" if self.transect_length_meters is not None else "lenNA"
            bn = f"transect_data_{start_str}_to_{end_str}_{length_str}"
        elif self.selected_filter_date:
            bn = f"geotagger_results_{self.selected_filter_date.strftime('%Y%m%d')}"

        save_dir = os.getcwd() # Default save directory
        if self.media_path and os.path.isdir(os.path.dirname(self.media_path)): # Prefer media dir's parent
             save_dir = os.path.dirname(self.media_path)
        elif self.gps_file_path and os.path.isfile(self.gps_file_path): # Or GPS file's dir
             save_dir = os.path.dirname(self.gps_file_path)

        path, _ = QFileDialog.getSaveFileName(self, "Save Table Data as CSV",
                                              os.path.join(save_dir, f"{bn}.csv"),
                                              "CSV Files (*.csv)")
        if path:
            try:
                if not path.lower().endswith(".csv"):
                    path += ".csv"

                # Get data directly from the currently displayed table to ensure what-you-see-is-what-you-get
                headers = [self.results_table.horizontalHeaderItem(c).text() for c in range(self.results_table.columnCount())]
                data_to_export = []
                for row_idx in range(self.results_table.rowCount()):
                    row_items = []
                    for col_idx in range(self.results_table.columnCount()):
                        item = self.results_table.item(row_idx, col_idx)
                        row_items.append(item.text() if item else "")
                    data_to_export.append(row_items)

                df_export = pd.DataFrame(data_to_export, columns=headers)
                df_export.to_csv(path, index=False, encoding='utf-8-sig') # utf-8-sig for Excel compatibility
                QMessageBox.information(self, "Export Successful", f"Table data saved to:\n{path}")
                self.log_message(f"Results table exported to CSV: {path}", logging.INFO)
            except Exception as e:
                self.show_error(f"CSV export failed: {e}")
                self.log_message(f"Error exporting results table to CSV: {e}", logging.ERROR)

    def save_map(self):
        if not self._map_file_path or not os.path.exists(self._map_file_path):
            self.show_error("No map has been generated or the temporary map file is missing.")
            return

        bn = "geotagger_map"
        if self.media_path and os.path.basename(self.media_path): # Use media folder name if available
            bn=f"{os.path.splitext(os.path.basename(self.media_path))[0]}_map"
        elif self.gps_file_path and os.path.basename(self.gps_file_path): # Or GPS file name
            bn=f"{os.path.splitext(os.path.basename(self.gps_file_path))[0]}_map"

        save_dir = os.getcwd()
        if self.media_path and os.path.isdir(os.path.dirname(self.media_path)):
             save_dir = os.path.dirname(self.media_path)
        elif self.gps_file_path and os.path.isfile(self.gps_file_path):
             save_dir = os.path.dirname(self.gps_file_path)

        path, _ = QFileDialog.getSaveFileName(self, "Save Map as HTML File",
                                              os.path.join(save_dir, f"{bn}.html"),
                                              "HTML Files (*.html)")
        if path:
            try:
                if not path.lower().endswith(".html"):
                    path += ".html"
                shutil.copyfile(self._map_file_path, path)
                QMessageBox.information(self, "Map Saved", f"Map file saved as:\n{path}")
                self.log_message(f"Map HTML file saved to: {path}", logging.INFO)
            except Exception as e:
                self.show_error(f"Map save failed: {e}")
                self.log_message(f"Error saving map HTML: {e}", logging.ERROR)

    def geotag_images(self):
        self.log_message("ACTION: Save Geotagged Images button clicked.")
        if not ENGINE_AVAILABLE:
            self.show_error("Geotagging engine (georeference_engine.py) is not loaded. Cannot geotag images.")
            return
        if self.results_df is None or self.results_df.empty:
            self.show_error("No processed results data available to use for geotagging.")
            return

        # Ensure identifier_to_path_map is up-to-date if original_media_list exists
        if not self.identifier_to_path_map and self.original_media_list:
            self.identifier_to_path_map = {item['identifier']: item['file_path']
                                           for item in self.original_media_list
                                           if 'identifier' in item and 'file_path' in item and item.get('file_path')}
        if not self.identifier_to_path_map:
            self.show_error("Internal Error: Original media file paths map is missing. Cannot locate source images.")
            return

        required_cols_geotag = ['Identifier', 'File Path', 'Latitude', 'Longitude']
        if not all(col in self.results_df.columns for col in required_cols_geotag):
            missing = [col for col in required_cols_geotag if col not in self.results_df.columns]
            self.show_error(f"Results data is missing required columns for geotagging: {', '.join(missing)}.")
            return

        try:
            # Check if any rows have valid numeric coordinates
            valid_coords_exist = (pd.to_numeric(self.results_df['Latitude'], errors='coerce').notna() & \
                                  pd.to_numeric(self.results_df['Longitude'], errors='coerce').notna()).any()
            if not valid_coords_exist:
                self.show_error("No valid GPS coordinates (Latitude/Longitude) found in the results data. Cannot geotag.")
                return
        except Exception as e:
            self.show_error(f"Error checking coordinates for geotagging: {e}")
            return

        default_output_dir_name = self.geotag_output_dir_base
        base_path_for_output = os.getcwd()
        if self.media_path and os.path.isdir(os.path.dirname(self.media_path)):
            base_path_for_output = os.path.dirname(self.media_path)
        default_full_output_dir = os.path.join(base_path_for_output, default_output_dir_name)

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory for Geotagged Image Copies", default_full_output_dir)
        if not output_dir:
            self.log_message("Geotagging operation cancelled by user (no output directory selected).")
            return

        reply = QMessageBox.StandardButton.Yes
        if os.path.exists(output_dir) and os.listdir(output_dir): # Check if directory exists AND is not empty
            reply = QMessageBox.question(self, 'Confirm Overwrite',
                                         f"The selected output directory already exists and is not empty:\n{output_dir}\n"
                                         "Existing files with the same names might be overwritten.\n\nDo you want to proceed?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No) # Default to No for safety

        if reply == QMessageBox.StandardButton.No:
            self.log_message("Geotagging cancelled by user (overwrite confirmation declined).")
            return

        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            self.show_error(f"Could not create or access the output directory:\n{output_dir}\nError: {e}")
            return

        ok_count, fail_count, skip_coord_count, skip_file_issue_count, skip_unsupported_type = 0, 0, 0, 0, 0

        self.geotag_images_button.setEnabled(False) # Disable button during operation
        self.progress_bar.setValue(0)
        self.reset_progress_bar_color()
        QApplication.processEvents() # Ensure UI updates

        rows_to_process_df = self.results_df.copy() # Work on a copy
        total_rows = len(rows_to_process_df)

        if total_rows == 0:
            self.log_message("No rows in results to process for geotagging.", logging.INFO)
            self.geotag_images_button.setEnabled(True) # Re-enable button
            return

        for index, row in rows_to_process_df.iterrows():
            identifier = row.get('Identifier')
            # Prefer path from identifier_to_path_map as it's from original scan
            actual_source_path = self.identifier_to_path_map.get(identifier)
            if not actual_source_path:
                # Fallback to 'File Path' column if map is somehow out of sync or ID not found
                source_path_from_df = row.get('File Path')
                if source_path_from_df and source_path_from_df.lower() != 'n/a' and source_path_from_df.strip() != "":
                    actual_source_path = source_path_from_df
                else:
                    self.log_message(f"Skipping ID '{identifier}': No valid source file path found.", logging.DEBUG)
                    skip_file_issue_count += 1
                    continue

            if not os.path.exists(actual_source_path):
                self.log_message(f"Skipping ID '{identifier}': Source file does not exist at '{actual_source_path}'.", logging.WARNING)
                skip_file_issue_count += 1
                continue

            latitude = row.get('Latitude')
            longitude = row.get('Longitude')

            if pd.isna(latitude) or pd.isna(longitude):
                self.log_message(f"Skipping ID '{identifier}': Missing valid Latitude/Longitude.", logging.DEBUG)
                skip_coord_count += 1
                continue

            try:
                lat_float = float(latitude)
                lon_float = float(longitude)
                if not (-90 <= lat_float <= 90 and -180 <= lon_float <= 180): # Basic coord validation
                     self.log_message(f"Skipping ID '{identifier}': Invalid coordinate values (Lat: {lat_float}, Lon: {lon_float}).", logging.WARNING)
                     skip_coord_count +=1
                     continue

                base_identifier, _ = os.path.splitext(identifier)
                _, source_ext = os.path.splitext(actual_source_path)
                output_base_name = f"{base_identifier}{source_ext}"
                output_path = os.path.join(output_dir, output_base_name)

                _, source_ext = os.path.splitext(actual_source_path)
                supported_extensions = ['.jpg', '.jpeg', '.tif', '.tiff'] # Check engine for full list
                if source_ext.lower() not in supported_extensions:
                    self.log_message(f"Skipping ID '{identifier}': Unsupported file type '{source_ext}'. Supported: {supported_extensions}", logging.WARNING)
                    skip_unsupported_type +=1
                    continue

                altitude = row.get('Altitude', 0.0) # Default altitude if not present
                try:
                    alt_float = float(altitude) if pd.notna(altitude) else 0.0
                except ValueError: # If altitude is non-numeric string
                    alt_float = 0.0

                # Parse the corrected UTC timestamp from the results so the engine can write it
                # into EXIF GPSDateStamp/GPSTimeStamp. Without this the engine has no choice but
                # to skip those tags (the previous behaviour wrote datetime.now(), which silently
                # corrupted any track-from-photos workflow downstream).
                gps_time_utc = None
                corrected_ts_value = row.get('Corrected Timestamp (UTC)')
                if corrected_ts_value is not None and pd.notna(corrected_ts_value):
                    try:
                        # Worker stores ISO-8601 with a trailing 'Z'; fromisoformat needs '+00:00'
                        # before Python 3.11.
                        gps_time_utc = datetime.fromisoformat(str(corrected_ts_value).replace('Z', '+00:00'))
                    except (ValueError, TypeError) as ts_err:
                        self.log_message(
                            f"Could not parse corrected timestamp ('{corrected_ts_value}') for ID '{identifier}': {ts_err}. "
                            "GPSDateStamp/GPSTimeStamp will not be written for this image.",
                            logging.DEBUG,
                        )
                        gps_time_utc = None

                if engine.set_gps_location(actual_source_path, lat_float, lon_float, alt_float, output_path, gps_time_utc=gps_time_utc):
                    ok_count += 1
                else:
                    fail_count += 1
                    self.log_message(f"Failed to geotag ID '{identifier}' using engine.set_gps_location.", logging.WARNING)

            except ValueError: # For float conversion of lat/lon
                self.log_message(f"Skipping ID '{identifier}': Invalid non-numeric Latitude/Longitude values.", logging.WARNING)
                skip_coord_count +=1
            except Exception as e_geotag:
                fail_count += 1
                self.log_message(f"Error geotagging file for ID '{identifier}' (source: '{actual_source_path}'): {e_geotag}", logging.ERROR)
                self.log_message(traceback.format_exc(), logging.DEBUG)

            progress = int(((index + 1) / total_rows) * 100) if total_rows > 0 else 0
            self.progress_bar.setValue(progress)
            if (index + 1) % 20 == 0: # Process events periodically to keep UI responsive
                QApplication.processEvents()

        self.progress_bar.setValue(100) # Ensure it reaches 100%
        self.geotag_images_button.setEnabled(True) # Re-enable button

        summary_message = (f"Geotagging process complete.\n\n"
                           f"Successfully geotagged: {ok_count}\n"
                           f"Failed to geotag (engine/write errors): {fail_count}\n"
                           f"Skipped (missing/invalid coords): {skip_coord_count}\n"
                           f"Skipped (file path issues): {skip_file_issue_count}\n"
                           f"Skipped (unsupported file type): {skip_unsupported_type}\n\n"
                           f"Output directory:\n{output_dir}")
        QMessageBox.information(self, "Geotagging Done", summary_message)
        self.log_message(f"Geotagging finished. OK: {ok_count}, Failed: {fail_count}, Skipped Coords: {skip_coord_count}, Skipped File: {skip_file_issue_count}, Skipped Type: {skip_unsupported_type}. Output: {output_dir}", logging.INFO)

    def extract_selected_images(self):
        self.log_message("ACTION: Extract Selected Images button clicked.")
        if self.results_df is None or self.results_df.empty:
            self.show_error("No results data loaded from which to extract images.")
            return

        if not self.identifier_to_path_map:
            self.show_error("Internal Error: Original media file paths map is missing. Cannot locate source images.")
            return

        selected_model_indices = self.results_table.selectionModel().selectedRows()
        if not selected_model_indices:
            self.show_error("No rows selected in the Results Table to extract images from.")
            return

        headers = [self.results_table.horizontalHeaderItem(c).text() for c in range(self.results_table.columnCount()) if self.results_table.horizontalHeaderItem(c)]
        try:
            id_col_idx = headers.index('Identifier')
        except ValueError:
            self.show_error("Internal Error: 'Identifier' column not found in results table headers.")
            return

        selected_ids_from_table = set()
        for model_index in selected_model_indices:
            item = self.results_table.item(model_index.row(), id_col_idx)
            if item and item.text():
                selected_ids_from_table.add(item.text())

        if not selected_ids_from_table:
            self.show_error("Could not retrieve valid Identifiers from the selected rows.")
            return

        default_dir_name = f"Extracted_Images_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        base_path_for_suggestion = os.getcwd()
        if self.media_path and os.path.isdir(os.path.dirname(self.media_path)):
            base_path_for_suggestion = os.path.dirname(self.media_path)

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder for Image Copies",
                                                      os.path.join(base_path_for_suggestion, default_dir_name))
        if not output_dir:
            self.log_message("Image extraction cancelled by user (no output directory selected).")
            return

        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            self.show_error(f"Could not create or access the output directory:\n{output_dir}\nError: {e}")
            return

        self.extract_button.setEnabled(False) # Disable during operation
        self.progress_bar.setValue(0)
        QApplication.processEvents() # UI update

        files_to_copy_info = []
        missing_source_ids = []
        for sid in selected_ids_from_table:
            source_path = self.identifier_to_path_map.get(sid)
            if source_path and os.path.exists(source_path):
                files_to_copy_info.append({'identifier': sid, 'source_path': source_path})
            else:
                missing_source_ids.append(sid)
                self.log_message(f"Source file for selected ID '{sid}' not found at '{source_path if source_path else 'path not in map'}'. Skipping.", logging.WARNING)

        if not files_to_copy_info:
            self.show_error("Could not find valid source files for any of the selected image Identifiers.")
            self.extract_button.setEnabled(bool(self.results_table.selectionModel().hasSelection())) # Re-enable based on selection state
            return

        if missing_source_ids:
            QMessageBox.warning(self, "Missing Source Files",
                                f"Could not find source files for {len(missing_source_ids)} selected ID(s).\n"
                                "Only images with found source files will be copied.")

        ok_copy_count, fail_copy_count = 0, 0
        total_to_copy = len(files_to_copy_info)
        for i, item_info in enumerate(files_to_copy_info):
            try:
                destination_filename = os.path.basename(item_info['source_path'])
                destination_path = os.path.join(output_dir, destination_filename)
                if os.path.exists(destination_path):
                    # Consider adding a counter or unique name if overwrite is not desired
                    self.log_message(f"Warning: Destination file '{destination_path}' already exists. Overwriting.", logging.WARNING)
                shutil.copy2(item_info['source_path'], destination_path)
                ok_copy_count += 1
            except Exception as e_copy:
                fail_copy_count += 1
                self.log_message(f"Error copying image for ID '{item_info['identifier']}' (from '{item_info['source_path']}'): {e_copy}", logging.ERROR)

            prog = int(((i + 1) / total_to_copy) * 100) if total_to_copy > 0 else 0
            self.progress_bar.setValue(prog)
            if (i + 1) % 10 == 0: # Update UI periodically
                QApplication.processEvents()

        self.progress_bar.setValue(100)
        self.extract_button.setEnabled(bool(self.results_table.selectionModel().hasSelection())) # Re-enable based on selection state

        summary_message = (f"Image extraction complete.\n\n"
                           f"Successfully copied: {ok_copy_count}\n"
                           f"Failed copies: {fail_copy_count}\n\n"
                           f"Output directory:\n{output_dir}")
        QMessageBox.information(self, "Extraction Complete", summary_message)
        self.log_message(f"Selected image extraction finished. OK: {ok_copy_count}, Failed: {fail_copy_count}. Output: {output_dir}", logging.INFO)

    def save_transect(self):
        if not GEOPY_AVAILABLE:
            self.show_error("Cannot save transect: 'geopy' library is required for transect length data.")
            return
        if self.transect_start_time is None or self.transect_end_time is None:
            self.show_error("Transect is not fully defined. Start and end points are required to save.")
            return

        active_gps_df = self.get_active_gps_df()
        if active_gps_df.empty:
            self.show_error("Cannot save transect: No active GPS data to get coordinates from.")
            return

        if not os.path.exists(self.transects_json_dir):
            try:
                os.makedirs(self.transects_json_dir)
                self.log_message(f"Created transect JSON directory: {self.transects_json_dir}", logging.INFO)
            except OSError as e:
                self.show_error(f"Could not create directory for saving transects:\n{self.transects_json_dir}\nError: {e}")
                return

        start_time_utc = min(self.transect_start_time, self.transect_end_time)
        s_time_str = start_time_utc.strftime('%Y%m%d_%H%M%S')
        length_str = f"{self.transect_length_meters:.0f}m" if self.transect_length_meters is not None else "lenNA"
        gps_file_base = os.path.splitext(os.path.basename(self.gps_file_path))[0] if self.gps_file_path else "NoGPS"
        default_filename = f"transect_{gps_file_base}_{s_time_str}_{length_str}.json"

        filepath, _ = QFileDialog.getSaveFileName(self, "Save Transect Definition",
                                                  os.path.join(self.transects_json_dir, default_filename),
                                                  "JSON files (*.json)")
        if not filepath:
            self.log_message("Transect save cancelled by user.", logging.INFO)
            return

        if not filepath.lower().endswith(".json"):
            filepath += ".json"

        try:
            start_point = active_gps_df.loc[self.transect_start_time]
            end_point = active_gps_df.loc[self.transect_end_time]
        except KeyError:
            self.show_error("Could not find start/end points in the current GPS data. Cannot save coordinates.")
            return

        transect_data_to_save = {
            "name": os.path.splitext(os.path.basename(filepath))[0],
            "start_lat": start_point['latitude'],
            "start_lon": start_point['longitude'],
            "end_lat": end_point['latitude'],
            "end_lon": end_point['longitude'],
            "length": self.transect_length_meters,
            "source_gps_file": os.path.basename(self.gps_file_path) if self.gps_file_path else "N/A",
            "saved_at_utc_iso": datetime.now(timezone.utc).isoformat(),
        }

        try:
            with open(filepath, 'w') as f_json:
                json.dump(transect_data_to_save, f_json, indent=4)
            self.log_message(f"Transect definition saved successfully to: {filepath}", logging.INFO)
            QMessageBox.information(self, "Transect Saved", f"Transect definition saved to:\n{filepath}")
        except Exception as e_save_json:
            self.log_message(f"Error saving transect definition to JSON file '{filepath}': {e_save_json}", logging.ERROR)
            self.show_error(f"Could not save transect definition:\n{e_save_json}")

    def save_all_transects(self):
        if not self.saved_transects:
            QMessageBox.information(self, "Nothing to Save", "No transects in the 'Transects for Batch Processing' list to save.")
            return

        if not os.path.exists(self.transects_json_dir):
            try:
                os.makedirs(self.transects_json_dir)
                self.log_message(f"Created transect JSON directory for 'Save All': {self.transects_json_dir}", logging.INFO)
            except OSError as e:
                self.show_error(f"Could not create directory for saving all transects:\n{self.transects_json_dir}\nError: {e}")
                return

        gps_file_base = os.path.splitext(os.path.basename(self.gps_file_path))[0] if self.gps_file_path else "NoGPS"
        current_datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_filename = f"all_transects_SET_{gps_file_base}_{current_datetime_str}.json"

        filepath, _ = QFileDialog.getSaveFileName(self, "Save All Transects As…",
                                                 os.path.join(self.transects_json_dir, default_filename),
                                                 "JSON files (*.json)")
        if not filepath:
            self.log_message("Save all transects cancelled by user.", logging.INFO)
            return

        if not filepath.lower().endswith(".json"):
            filepath += ".json"

        try:
            with open(filepath, "w") as f:
                json.dump(self.saved_transects, f, indent=2) # Use indent 2 for readability of list
            self.log_message(f"All {len(self.saved_transects)} transects saved to: {filepath}", logging.INFO)
            QMessageBox.information(self, "All Transects Saved", f"{len(self.saved_transects)} transect definitions written to:\n{filepath}")
        except Exception as e_save_all:
            self.log_message(f"Error saving all transects to JSON file '{filepath}': {e_save_all}", logging.ERROR)
            self.show_error(f"Could not save all transects:\n{e_save_all}")

    def _load_transects_csv_file(self):
        self.log_message("Load Transects from CSV button clicked.")
        if self.processed_gps_df is None or self.processed_gps_df.empty:
            self.show_error("Please load/process GPS-tagged images or main GPS data before loading transect definitions from a CSV.")
            return

        filepath, _ = QFileDialog.getOpenFileName(self, "Select Transect CSV File", "", "CSV files (*.csv *.txt)")
        if not filepath:
            self.log_message("Transect CSV loading cancelled.", logging.INFO)
            return

        self.transect_csv_file_path = filepath
        if hasattr(self, 'transect_csv_path_display'):
            self.transect_csv_path_display.setText(os.path.basename(filepath))
        self.log_message(f"Transect CSV file selected: {filepath}")

        try:
            # Attempt to sniff delimiter
            delimiter = ','
            try:
                with open(filepath, 'r', errors='ignore') as f:
                    sample = "".join(line for line in (f.readline() for _ in range(5)) if line and line.strip())
                    if sample:
                        dialect = csv.Sniffer().sniff(sample)
                        delimiter = dialect.delimiter
                        self.log_message(f"Detected delimiter for Transect CSV: '{repr(delimiter)}'")
            except Exception as sniff_err:
                self.log_message(f"Delimiter sniffing for Transect CSV failed ({sniff_err}), using default ','.", logging.WARNING)

            # Read headers
            try:
                headers = pd.read_csv(filepath, sep=delimiter, nrows=0, engine='python', skipinitialspace=True, encoding_errors='ignore').columns.tolist()
            except UnicodeDecodeError:
                self.log_message("UnicodeDecodeError reading Transect CSV headers with utf-8, trying latin-1", logging.WARNING)
                headers = pd.read_csv(filepath, sep=delimiter, nrows=0, engine='python', skipinitialspace=True, encoding='latin-1', encoding_errors='ignore').columns.tolist()
            except pd.errors.EmptyDataError:
                self.show_error("Transect CSV file is empty or contains no data after headers.")
                self.transect_csv_file_path = None
                if hasattr(self, 'transect_csv_path_display'): self.transect_csv_path_display.clear()
                return

            if not headers:
                self.show_error("Could not read headers from Transect CSV file.")
                self.transect_csv_file_path = None
                if hasattr(self, 'transect_csv_path_display'): self.transect_csv_path_display.clear()
                return

            self.log_message(f"Transect CSV Headers: {headers}")

            # Populate mapping combos
            for combo in [self.transect_csv_name_col_combo, self.transect_csv_start_time_col_combo,
                          self.transect_csv_end_time_col_combo, self.transect_csv_length_col_combo]:
                combo.clear()
                if combo == self.transect_csv_length_col_combo: # Add "Not Specified" for optional length
                    combo.addItem("<Not Specified - Calculate>")
                combo.addItems(headers)

            # Auto-guess common column names for transect CSV
            self._guess_transect_csv_columns(headers)

            self.transect_csv_mapping_group.setVisible(True)

        except Exception as e:
            self.show_error(f"Error processing Transect CSV headers: {e}")
            self.log_message(f"Transect CSV header processing error: {e}", logging.ERROR)
            self.log_message(traceback.format_exc(), logging.DEBUG)
            self.transect_csv_file_path = None
            if hasattr(self, 'transect_csv_path_display'): self.transect_csv_path_display.clear()
            self.transect_csv_mapping_group.setVisible(False)

    def _guess_transect_csv_columns(self, headers):
        self.log_message("Attempting to guess Transect CSV columns...")
        headers_lower_map = {h.lower().strip().replace(" ", "").replace("_", ""): h for h in headers}

        def find_set_transect_csv(patterns, combo, default_to_first_valid=True):
            normalized_patterns = [p.lower().replace(" ", "").replace("_", "") for p in patterns]
            for norm_p in normalized_patterns:
                if norm_p in headers_lower_map:
                    target_header = headers_lower_map[norm_p]
                    idx = combo.findText(target_header, Qt.MatchFlag.MatchFixedString | Qt.MatchFlag.MatchCaseSensitive)
                    if idx < 0: idx = combo.findText(target_header, Qt.MatchFlag.MatchFixedString) # Case-insensitive fallback
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                        self.log_message(f"Guessed '{target_header}' for Transect CSV {combo.objectName() if combo.objectName() else 'combo'}", logging.DEBUG)
                        return True
            # If no match, behavior depends on default_to_first_valid and combo type
            if combo == self.transect_csv_length_col_combo: # Length is optional, default to "Not Specified"
                 combo.setCurrentIndex(0) # Index 0 is "<Not Specified - Calculate>"
            elif default_to_first_valid and combo.count() > 0:
                 combo.setCurrentIndex(0) # Default to first actual column for others
            return False

        # Set object names if not set (useful for logging)
        self.transect_csv_name_col_combo.setObjectName("TransectNameCol")
        self.transect_csv_start_time_col_combo.setObjectName("TransectStartTimeCol")
        self.transect_csv_end_time_col_combo.setObjectName("TransectEndTimeCol")
        self.transect_csv_length_col_combo.setObjectName("TransectLengthCol")

        find_set_transect_csv(['name', 'transect_name', 'transectid', 'id', 'transect name', 'label'], self.transect_csv_name_col_combo)
        find_set_transect_csv(['start_time', 'starttime', 'begin_time', 'begintime', 'start', 'start time'], self.transect_csv_start_time_col_combo)
        find_set_transect_csv(['end_time', 'endtime', 'stop_time', 'stoptime', 'end', 'end time'], self.transect_csv_end_time_col_combo)
        find_set_transect_csv(['length', 'transect_length', 'length_m', 'distance', 'transect length'], self.transect_csv_length_col_combo, default_to_first_valid=False)


    def _add_mapped_csv_transects_to_list(self):
        self.log_message("Add Mapped CSV Transects button clicked.")
        if not self.transect_csv_file_path or not os.path.exists(self.transect_csv_file_path):
            self.show_error("No Transect CSV file loaded to process.")
            return

        name_col = self.transect_csv_name_col_combo.currentText()
        start_time_col = self.transect_csv_start_time_col_combo.currentText()
        end_time_col = self.transect_csv_end_time_col_combo.currentText()
        length_col_text = self.transect_csv_length_col_combo.currentText()
        length_col = length_col_text if length_col_text != "<Not Specified - Calculate>" else None

        dt_format = self.transect_csv_datetime_format_input.text().strip()

        if not (name_col and start_time_col and end_time_col and dt_format):
            self.show_error("Name, Start Time, End Time columns, and Timestamp Format must be specified.")
            return

        try:
            delimiter = ',' # Default, try to reuse sniffed one if available
            try:
                with open(self.transect_csv_file_path, 'r', errors='ignore') as f:
                    sample = "".join(line for line in (f.readline() for _ in range(5)) if line and line.strip())
                    if sample:
                        dialect = csv.Sniffer().sniff(sample)
                        delimiter = dialect.delimiter
            except: pass # Ignore if sniffing fails again, use default

            df_transects = pd.read_csv(self.transect_csv_file_path, sep=delimiter, skipinitialspace=True, encoding_errors='ignore', keep_default_na=False, na_values=['', 'NA', 'N/A', '#N/A'])
        except Exception as e:
            self.show_error(f"Error reading Transect CSV file: {e}")
            self.log_message(f"Error reading Transect CSV: {e}", logging.ERROR)
            return

        added_count = 0
        skipped_count = 0
        newly_added_transects_from_csv = []
        target_tz = pytz.timezone(self.current_timezone_str) # Timezone for naive CSV times
        active_gps_df = self.get_active_gps_df()
        coord_cols = {
            "name": self._find_column_by_alias(df_transects.columns, ["name", "transect_name", "transectid", "id", "label"]),
            "start_lat": self._find_column_by_alias(df_transects.columns, ["start_lat", "start latitude", "startlat", "lat1", "y1"]),
            "start_lon": self._find_column_by_alias(df_transects.columns, ["start_lon", "start longitude", "startlon", "lon1", "long1", "x1"]),
            "end_lat": self._find_column_by_alias(df_transects.columns, ["end_lat", "end latitude", "endlat", "lat2", "y2"]),
            "end_lon": self._find_column_by_alias(df_transects.columns, ["end_lon", "end longitude", "endlon", "lon2", "long2", "x2"]),
            "length": self._find_column_by_alias(df_transects.columns, ["length", "length_m", "transect_length", "distance"]),
        }

        if all(coord_cols[k] for k in ["name", "start_lat", "start_lon", "end_lat", "end_lon"]):
            for index, row in df_transects.iterrows():
                try:
                    name = str(row[coord_cols["name"]]).strip()
                    start_lat = float(row[coord_cols["start_lat"]])
                    start_lon = float(row[coord_cols["start_lon"]])
                    end_lat = float(row[coord_cols["end_lat"]])
                    end_lon = float(row[coord_cols["end_lon"]])
                    length_m = None
                    if coord_cols["length"] and pd.notna(row[coord_cols["length"]]):
                        try:
                            length_m = float(row[coord_cols["length"]])
                        except Exception:
                            length_m = None
                    if not name:
                        name = f"Transect_{index + 1}"
                    newly_added_transects_from_csv.append({
                        "name": name,
                        "start_lat": start_lat,
                        "start_lon": start_lon,
                        "end_lat": end_lat,
                        "end_lon": end_lon,
                        "length": length_m,
                        "source": "Coordinate CSV"
                    })
                    added_count += 1
                except Exception as e_row:
                    self.log_message(f"Skipping coordinate transect CSV row {index+2}: {e_row}", logging.WARNING)
                    skipped_count += 1

            if newly_added_transects_from_csv:
                self.saved_transects.extend(newly_added_transects_from_csv)
                self._populate_loaded_transects_list()
                self._update_batch_processing_ui_states()
                self.regenerate_map_display()
                QMessageBox.information(self, "Coordinate Transects Added",
                                        f"Added {added_count} coordinate transects from CSV to the batch list.\n"
                                        f"Skipped {skipped_count} rows due to errors (see log).\n\n"
                                        f"Batch extraction will copy images within the selected transect buffer distance.")
                self.log_message(f"Added {added_count} coordinate transects from CSV. Skipped: {skipped_count}.", logging.INFO)
            elif skipped_count > 0:
                self.show_error(f"No coordinate transects added from CSV. All {skipped_count} rows had errors. Check log.")
            return


        for index, row in df_transects.iterrows():
            try:
                name = str(row[name_col]).strip() if pd.notna(row[name_col]) else ""
                start_time_str = str(row[start_time_col]).strip() if pd.notna(row[start_time_col]) else ""
                end_time_str = str(row[end_time_col]).strip() if pd.notna(row[end_time_col]) else ""


                if not (name and start_time_str and end_time_str):
                    self.log_message(f"Skipping row {index+2} in Transect CSV: Missing name, start, or end time.", logging.WARNING)
                    skipped_count +=1
                    continue

                start_naive = datetime.strptime(start_time_str, dt_format)
                end_naive = datetime.strptime(end_time_str, dt_format)
                start_utc = target_tz.localize(start_naive, is_dst=None).astimezone(pytz.utc)
                end_utc = target_tz.localize(end_naive, is_dst=None).astimezone(pytz.utc)

                if start_utc > end_utc:
                    start_utc, end_utc = end_utc, start_utc

                # Get coordinates from the current GPS track based on the parsed times
                try:
                    start_point = active_gps_df.loc[active_gps_df.index.get_indexer([start_utc], method='nearest')[0]]
                    end_point = active_gps_df.loc[active_gps_df.index.get_indexer([end_utc], method='nearest')[0]]
                except (IndexError, KeyError):
                    self.log_message(f"Skipping row {index+2} ('{name}'): Could not map start/end times from CSV to the current GPS track.", logging.WARNING)
                    skipped_count += 1
                    continue

                length_m = None
                if length_col and length_col in row and pd.notna(row[length_col]):
                    try:
                        length_m = float(row[length_col])
                    except (ValueError, TypeError):
                        length_m = None

                if length_m is None:
                    length_m = self.calculate_transect_length(start_point.name, end_point.name)

                newly_added_transects_from_csv.append({
                    "name": name,
                    "start_lat": start_point['latitude'],
                    "start_lon": start_point['longitude'],
                    "end_lat": end_point['latitude'],
                    "end_lon": end_point['longitude'],
                    "length": length_m,
                    "source": "CSV"
                })
                added_count +=1
            except ValueError as ve:
                self.log_message(f"Skipping row {index+2} in Transect CSV due to datetime parse error: {ve}", logging.WARNING)
                skipped_count += 1
            except Exception as e_row:
                self.log_message(f"Error processing row {index+2} from Transect CSV: {e_row}", logging.ERROR)
                skipped_count += 1

        if newly_added_transects_from_csv:
            self.saved_transects.extend(newly_added_transects_from_csv)
            self._populate_loaded_transects_list()
            self._update_batch_processing_ui_states()
            self.regenerate_map_display()
            QMessageBox.information(self, "Transects Added",
                                    f"Added {added_count} transects from CSV to the batch list.\n"
                                    f"Skipped {skipped_count} rows due to errors (see log).")
            self.log_message(f"Added {added_count} transects from CSV. Skipped: {skipped_count}.", logging.INFO)
        elif skipped_count > 0 and added_count == 0:
            self.show_error(f"No transects added from CSV. All {skipped_count} rows had errors. Check log.")
        else:
            QMessageBox.information(self, "No Transects Added", "No new transects found or added from the CSV file.")

    def closeEvent(self, event):
        self.log_message("Closing application...", logging.INFO)
        if self._map_file_path and os.path.exists(self._map_file_path):
            if "geotagger_map_app_" in os.path.basename(self._map_file_path) or \
               "geotagger_map_worker_" in os.path.basename(self._map_file_path):
                try:
                    os.remove(self._map_file_path)
                    self.log_message(f"Removed temporary map file: {self._map_file_path}", logging.DEBUG)
                except OSError as e_remove_map:
                    self.log_message(f"Error removing temporary map file '{self._map_file_path}': {e_remove_map}", logging.WARNING)
            else:
                self.log_message(f"Did not remove map file '{self._map_file_path}' as it doesn't match temp naming convention.", logging.DEBUG)

        if self.worker_thread and self.worker_thread.isRunning():
            self.log_message("Terminating worker thread before closing...", logging.WARNING)
            self.worker_thread.quit()
            if not self.worker_thread.wait(1000):
                self.log_message("Worker thread did not quit gracefully, forcing termination.", logging.WARNING)
                self.worker_thread.terminate()
                self.worker_thread.wait()

        super().closeEvent(event)


# --- Main Execution ---
if __name__ == '__main__':
    log_file_path = "geotagger_app.log"
    log_format = '%(asctime)s,%(msecs)03d - %(levelname)-8s - [%(filename)s:%(lineno)d] - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    try:
        logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt=date_format, filename=log_file_path, filemode='a', encoding='utf-8')
    except Exception as log_e:
        print(f"FATAL: Could not configure file logging to '{log_file_path}': {log_e}")

    # Console handler for INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.getLogger().addHandler(console_handler)

    app = QApplication(sys.argv)
    app.setStyle("Fusion") # A modern, platform-agnostic style

    # Dependency checks
    missing_deps = []
    try:
        import PyQt6.QtWebEngineCore
    except ImportError:
        missing_deps.append("PyQt6-WebEngine (pip install PyQt6-WebEngine)")
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas (pip install pandas)")
    try:
        import folium
    except ImportError:
        missing_deps.append("folium (pip install folium)")
    try:
        import pytz
    except ImportError:
        missing_deps.append("pytz (pip install pytz)")

    if not GEOPY_AVAILABLE: # geopy is optional
        missing_deps.append("geopy (pip install geopy) - Optional, but enables transect length features.")
    if not ENGINE_AVAILABLE: # engine is critical
        missing_deps.append("georeference_engine.py (must be in the same folder or Python path)")

    critical_missing_for_dialog = [dep for dep in missing_deps if "geopy" not in dep.lower() and "engine.py" not in dep.lower()]

    if critical_missing_for_dialog:
        error_message = "Required libraries missing:\n\n- " + "\n- ".join(critical_missing_for_dialog) + \
                        "\n\nPlease install missing libraries.\nApplication exiting."
        QMessageBox.critical(None, "Missing Dependencies", error_message)
        logging.critical(f"FATAL ERROR: Missing critical dependencies: {', '.join(critical_missing_for_dialog)}")
        sys.exit(1)

    if not ENGINE_AVAILABLE:
        engine_missing_msg = "Core component 'georeference_engine.py' not found. Please ensure it is in the same directory as the application or in your Python path. Application cannot run without it."
        QMessageBox.critical(None, "Missing Core Component", engine_missing_msg)
        logging.critical("FATAL ERROR: georeference_engine.py not found.")
        sys.exit(1)

    if not GEOPY_AVAILABLE and not critical_missing_for_dialog and ENGINE_AVAILABLE:
         warn_message = "Optional component 'geopy' is missing.\n\n- geopy (pip install geopy)\n\nSome features (like transect length calculation and preset length enforcement) will be disabled."
         QMessageBox.warning(None, "Optional Component Missing", warn_message)
         logging.warning("Optional component 'geopy' missing.")


    try:
        profile = QWebEngineProfile.defaultProfile()
        logging.info("Default WebEngine profile obtained.")
    except Exception as e:
        logging.error(f"Could not get default WebEngine profile: {e}. Map functionality might be affected if this persists across runs/systems.")

    try:
        window = GeoTaggerApp()
        window.show()
        logging.info("==================== Application Start ====================")
        print(f"NOTE: Detailed logs (DEBUG level) available in: {os.path.abspath(log_file_path)}")
        if not GEOPY_AVAILABLE:
            print("WARNING: 'geopy' not found. Transect length features are disabled.")
        exit_code = app.exec()
        logging.info(f"==================== Application End (Exit Code: {exit_code}) ====================")
        sys.exit(exit_code)
    except Exception as main_err:
        logging.critical(f"Unhandled top-level exception: {main_err}", exc_info=True)
        try:
            QMessageBox.critical(None, "Fatal Application Error", f"A critical error occurred:\n{main_err}\n\nCheck log file ('{log_file_path}') for details.\nApplication exiting.")
        except Exception:
            print(f"FATAL APPLICATION ERROR: {main_err}\nCheck log file for details: {os.path.abspath(log_file_path)}")
        sys.exit(1)
