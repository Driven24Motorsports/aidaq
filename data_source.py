# =============================================================================
# data_source.py — aiDAQ Data Ingestion Layer
# =============================================================================
# This module owns everything related to GETTING data into the app.
# It knows nothing about figures, Dash callbacks, or physics equations.
#
# TWO MODES:
#
#   Live mode  (DEPLOYMENT_MODE = 'local')
#   ─────────────────────────────────────
#   Wraps irsdk.IRSDK().  A background thread polls the SDK at 60 Hz and
#   pushes each frame (as a plain Python dict) into a thread-safe deque.
#   The Dash dcc.Interval callback reads from that deque every 100 ms.
#   NOTE: irsdk only runs on the same Windows PC as iRacing.
#
#   File mode  (both 'local' and 'cloud')
#   ──────────────────────────────────────
#   .ibt  upload: decode base64 → save temp file → iterate all frames with
#                 irsdk test_file mode → build Polars DataFrame → save as
#                 .parquet.  (Local only — irsdk not available on server.)
#
#   .parquet upload: decode base64 → read with Polars directly.
#                    Works on both local and cloud.
#
# KEY DESIGN DECISIONS:
#   • Polars is used for DataFrames (4–10× faster than pandas for large files,
#     and LazyFrame supports memory-efficient column-only reads).
#   • irsdk is imported inside a try/except so the module loads cleanly on the
#     cloud server where irsdk is not installed.
#   • All derived physics columns (TC, USR_TC, balance%, X/Y position) are
#     computed here once at load time, not repeatedly in Dash callbacks.
#   • The public API is three methods: get_live_frame(), get_session_df(),
#     and mode().  The Dash app calls only these.
#
# =============================================================================

import io
import base64
import json
import math
import threading
import collections
import tempfile
import os
import time

import numpy as np
import polars as pl

from config import (
    DEPLOYMENT_MODE,
    TRACK_F_INIT, TRACK_R_INIT,
    AX_MAX_DEFAULT, AY_MAX_DEFAULT,
    EPSILON,
)

# VAR, REQUIRED_VARS, OPTIONAL_VARS live in physics.py (not config.py).
# They are iRacing channel name mappings — physics knowledge, not config.
from physics import VAR, REQUIRED_VARS, OPTIONAL_VARS

# physics module provides all the maths
import physics as phys

# ---------------------------------------------------------------------------
# Attempt to import irsdk (only available locally on Windows with iRacing)
# ---------------------------------------------------------------------------
try:
    import irsdk
    IRSDK_AVAILABLE = True
except ImportError:
    IRSDK_AVAILABLE = False
    if DEPLOYMENT_MODE == 'local':
        print('WARNING: irsdk not found — live telemetry will be unavailable.')

# ---------------------------------------------------------------------------
# Columns to read from iRacing per frame (order matters for CSV-like structs)
# ---------------------------------------------------------------------------
# These are the raw iRacing channel names we attempt to read each frame.
# Columns marked [R] are required; [O] are optional (NaN if missing).
FRAME_COLUMNS = [
    'SessionTime',          # [R] session clock (s)
    'Lap',                  # [R] lap counter
    'LapDistPct',           # [R] 0-1 fraction of lap completed
    'LapDist',              # [O] actual metres from start/finish
    'Speed',                # [R] vehicle speed (m/s)
    'YawNorth',             # [O] heading from north (rad)
    'YawRate',              # [R] yaw rate (rad/s)
    'VelocityY',            # [R] lateral body velocity (m/s)
    'SteeringWheelAngle',   # [R] steering wheel angle (rad) — not road angle
    'Throttle',             # [O] 0-1 fraction
    'Brake',                # [O] 0-1 fraction
    'LatAccel',             # [O] lateral accel (m/s²)
    'LongAccel',            # [O] longitudinal accel (m/s²)
    'LFwheelSpeed',         # [O] wheel speed (rad/s)
    'RFwheelSpeed',
    'LRwheelSpeed',
    'RRwheelSpeed',
]


# =============================================================================
# _IBTFrameAccessor — thin wrapper so IBT frame looks like IRSDK subscript
# =============================================================================

class _IBTFrameAccessor:
    """Makes a single irsdk.IBT frame at index `i` behave like IRSDK[key].

    This lets the existing per-frame code (ir['Speed'], etc.) and physics
    helpers (phys.update_track_widths(ir, ...)) work unchanged when reading
    from an IBT file rather than the live IRSDK shared-memory buffer.

    ibt.get(i, key) returns None when the channel is absent; we convert that
    to float('nan') so downstream float() casts don't raise TypeError.
    """
    __slots__ = ('_ibt', '_i')

    def __init__(self, ibt, i):
        self._ibt = ibt
        self._i   = i

    def __getitem__(self, key):
        v = self._ibt.get(self._i, key)
        return float('nan') if v is None else v

    def __contains__(self, key):
        return key in self._ibt._var_headers_dict


# =============================================================================
# DataSource — main public class
# =============================================================================

class DataSource:
    """Unified data source for live iRacing telemetry or .ibt/.parquet files.

    Attributes
    ----------
    _mode         : str             'live' | 'file'
    _live_deque   : deque           ring buffer for live frames (dicts)
    _session_df   : pl.DataFrame    full session data (file mode)
    _ir           : irsdk.IRSDK     SDK handle (live mode)
    _thread       : Thread          background poller (live mode)
    _running      : bool            True while background thread should run
    """

    def __init__(self):
        self._mode        = 'file'        # default; set_live() or load_file()
        self._live_deque  = collections.deque(maxlen=10)
        self._session_df  = None
        self._ir          = None
        self._thread      = None
        self._running     = False

        # Live-mode position dead-reckoning state.
        # Reset each time set_live() / stop_live() is called so the track map
        # always starts from (0, 0) at the beginning of a new live session.
        self._live_pos_x    = 0.0
        self._live_pos_y    = 0.0
        self._live_last_time = None

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def mode(self):
        """Return 'live' or 'file'."""
        return self._mode

    def get_live_frame(self):
        """Return the most recent telemetry dict, or None if nothing yet.

        Called by the Dash dcc.Interval callback every 100 ms in live mode.
        Thread-safe: deque.append / deque[-1] are atomic in CPython.
        """
        if not self._live_deque:
            return None
        return self._live_deque[-1]

    def get_session_df(self):
        """Return the full session as a Polars DataFrame (file mode).

        Returns None if no file has been loaded yet.
        """
        return self._session_df

    def get_session_parquet(self):
        """Serialise the current session DataFrame to Parquet bytes.

        Used by the local app to offer a 'Download .parquet' button so the
        user can upload the pre-processed file to the cloud dashboard.

        Returns
        -------
        bytes  — Parquet file contents, or None if no session loaded.
        """
        if self._session_df is None:
            return None
        buf = io.BytesIO()
        self._session_df.write_parquet(buf)
        return buf.getvalue()

    # -------------------------------------------------------------------------
    # LIVE MODE
    # -------------------------------------------------------------------------

    def set_live(self):
        """Switch to live mode and start the iRacing polling thread.

        Raises
        ------
        RuntimeError  if irsdk is not installed (cloud environment)
        """
        if not IRSDK_AVAILABLE:
            raise RuntimeError(
                'irsdk is not installed — live mode unavailable. '
                'Install it: pip install pyirsdk')

        self._mode = 'live'
        self._running = True
        self._ir = irsdk.IRSDK()

        # Start background thread — does NOT block the Dash UI loop
        self._thread = threading.Thread(
            target=self._live_poll_loop,
            daemon=True,       # thread dies when main process exits
            name='aidaq-live-poller',
        )
        self._thread.start()
        print('aiDAQ live poller thread started.', flush=True)

    def stop_live(self):
        """Stop the background polling thread (called on app shutdown)."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        # Reset dead-reckoning so the next live session starts from (0, 0).
        self._live_pos_x     = 0.0
        self._live_pos_y     = 0.0
        self._live_last_time = None

    def _live_poll_loop(self):
        """Background thread: connect to iRacing and push frames to deque.

        Runs at up to 60 Hz (iRacing's native update rate).
        On disconnect: keeps retrying until self._running is False.
        """
        ir = self._ir
        telemetry_vars = set()
        connected_once = False

        while self._running:
            # ---- Try to connect ----
            try:
                ir.startup()
            except Exception as e:
                print(f'iRSdk startup error: {e}', flush=True)
                time.sleep(1.0)
                continue

            if not ir.is_connected:
                time.sleep(0.5)
                continue

            # ---- First connection: run diagnostics ----
            if not connected_once:
                telemetry_vars = phys.inventory_vars(ir)
                try:
                    phys.startup_diagnostics(ir, telemetry_vars, var_map=phys.VAR)
                except RuntimeError as e:
                    print(f'Startup validation error: {e}', flush=True)
                    # Still continue — missing optional channels are OK
                connected_once = True

            # ---- Polling loop ----
            while self._running and ir.is_connected:
                try:
                    ir.freeze_var_buffer_latest()
                except Exception:
                    time.sleep(0.05)
                    continue

                frame = self._read_live_frame(ir, telemetry_vars)
                if frame:
                    self._live_deque.append(frame)

                # ~60 Hz polling; sleep briefly to yield CPU
                time.sleep(0.016)

            print('iRacing disconnected — retrying…', flush=True)

    def _read_live_frame(self, ir, telemetry_vars):
        """Read one telemetry frame from irsdk into a dict.

        Returns None if the frame is malformed or session time is bad.
        All optional channels that are absent are stored as float('nan').
        """
        try:
            t = float(ir['SessionTime'])
            if not math.isfinite(t):
                return None

            frame = {'SessionTime': t}
            for col in FRAME_COLUMNS[1:]:   # skip SessionTime already read
                if col in telemetry_vars:
                    try:
                        frame[col] = float(ir[col])
                    except Exception:
                        frame[col] = float('nan')
                else:
                    frame[col] = float('nan')

            # ---- Dead-reckoning position (mirrors _parse_ibt_file logic) ----
            # Integrate speed × heading to grow the live track map the same way
            # file-mode sessions do.  Starts at (0, 0) when live mode begins.
            yaw_n = frame.get('YawNorth', float('nan'))
            spd   = frame.get('Speed',    float('nan'))
            if (self._live_last_time is not None
                    and math.isfinite(yaw_n)
                    and math.isfinite(spd)):
                dt = t - self._live_last_time
                if 0 < dt < 0.5:   # ignore gaps (pauses, session reloads)
                    self._live_pos_x += spd * math.sin(yaw_n) * dt
                    self._live_pos_y += spd * math.cos(yaw_n) * dt
            self._live_last_time = t
            frame['PosX'] = self._live_pos_x
            frame['PosY'] = self._live_pos_y

            return frame
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # FILE MODE — .ibt
    # -------------------------------------------------------------------------

    def load_ibt_b64(self, b64_content):
        """Load a .ibt file from a base64-encoded string (from dcc.Upload).

        Flow:
            1. Decode base64 → raw bytes
            2. Write to a temp file (irsdk requires a real file path)
            3. Iterate all frames with irsdk test_file mode
            4. Build Polars DataFrame + compute derived physics columns
            5. Store in self._session_df

        Parameters
        ----------
        b64_content : str  — base64 string, possibly with data-URI prefix
                             e.g. 'data:application/octet-stream;base64,AAAA…'

        Returns
        -------
        pl.DataFrame  — the loaded session, or None on error
        """
        if not IRSDK_AVAILABLE:
            raise RuntimeError(
                'irsdk is not available — cannot read .ibt files on this server. '
                'Upload a .parquet file instead.')

        # Strip data-URI prefix if present (dcc.Upload adds it)
        if ',' in b64_content:
            b64_content = b64_content.split(',', 1)[1]

        raw_bytes = base64.b64decode(b64_content)

        # Write to a named temp file that irsdk can open
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ibt') as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        try:
            df = self._parse_ibt_file(tmp_path)
        finally:
            # Always clean up the temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if df is not None:
            self._session_df = df
            self._mode = 'file'

        return df

    def _parse_ibt_file(self, file_path):
        """Read every frame from a .ibt file and build a Polars DataFrame.

        Uses irsdk in test_file mode which replays the file frame-by-frame.
        After reading the raw channels, derived physics columns are added:
            - PosX, PosY  (dead-reckoning from YawNorth + Speed)
            - TC          (normalised traction circle usage)
            - USR_TC      (steering responsiveness)
            - Balance%    (over/understeer score −100 to +100)

        Parameters
        ----------
        file_path : str  — absolute path to the .ibt file

        Returns
        -------
        pl.DataFrame  or  None on error
        """
        # Use irsdk.IBT — the correct pyirsdk 1.3.5 API for .ibt file iteration.
        # (irsdk.IRSDK.parse_to() now requires a to_file path arg and dumps a
        # text snapshot; it no longer advances frames one-by-one.)
        ir = irsdk.IBT()
        try:
            ir.open(file_path)
        except Exception as e:
            print(f'irsdk IBT failed to open {file_path}: {e}', flush=True)
            return None

        # IBT exposes channel names through _var_headers_dict (dict keyed by name)
        telemetry_vars = set(ir._var_headers_dict.keys())
        n_frames = ir._disk_header.session_record_count
        rows = []

        # Create per-session median filters (isolated from live-mode filters)
        f_filt = phys.MedianFilter(400)
        r_filt = phys.MedianFilter(400)
        tf = TRACK_F_INIT
        tr = TRACK_R_INIT

        # Position integration state
        pos_x = 0.0
        pos_y = 0.0
        last_time = None

        try:
            # Iterate frame-by-frame using IBT.get(index, key).
            # _IBTFrameAccessor wraps IBT so existing ir[key] code is unchanged.
            for i in range(n_frames):
                accessor = _IBTFrameAccessor(ir, i)
                row = {}
                try:
                    t = float(accessor['SessionTime'])
                except Exception:
                    continue

                if not math.isfinite(t):
                    continue

                row['SessionTime'] = t

                # Read all configured channels (NaN if unavailable)
                for col in FRAME_COLUMNS[1:]:
                    if col in telemetry_vars:
                        try:
                            row[col] = float(accessor[col])
                        except Exception:
                            row[col] = float('nan')
                    else:
                        row[col] = float('nan')

                # ---- Dead-reckoning position ----
                # Integrate speed × heading to get X/Y track coordinates.
                # This drifts over multiple laps (no GPS correction) but is
                # sufficient for the track-outline figure in v1.
                yaw_n = row.get('YawNorth', float('nan'))
                spd   = row.get('Speed',    float('nan'))
                if last_time is not None and math.isfinite(yaw_n) and math.isfinite(spd):
                    dt = t - last_time
                    if 0 < dt < 0.5:    # ignore gaps (pauses, reloads)
                        pos_x += spd * math.sin(yaw_n) * dt
                        pos_y += spd * math.cos(yaw_n) * dt

                row['PosX'] = pos_x
                row['PosY'] = pos_y
                last_time = t

                # ---- Track-width update ----
                u = row.get('Speed', 0.0)
                r = row.get('YawRate', 0.0)
                tf_new, tr_new = phys.update_track_widths(
                    accessor, u, r, telemetry_vars, f_filt, r_filt)
                if tf_new is not None:
                    tf, tr = tf_new, tr_new
                row['TrackWidth_F'] = tf
                row['TrackWidth_R'] = tr

                rows.append(row)

        except Exception as e:
            print(f'Error during .ibt parsing: {e}', flush=True)
        finally:
            try:
                ir.close()
            except Exception:
                pass

        if not rows:
            print('No frames extracted from .ibt file.', flush=True)
            return None

        # ---- Build Polars DataFrame from list of row dicts ----
        df = pl.DataFrame(rows)

        # ---- Add derived physics columns ----
        df = self._add_derived_columns(df)

        print(f'Parsed {len(df)} frames from {os.path.basename(file_path)}.',
              flush=True)
        return df

    # -------------------------------------------------------------------------
    # FILE MODE — .parquet
    # -------------------------------------------------------------------------

    def load_parquet_b64(self, b64_content):
        """Load a .parquet file from a base64-encoded dcc.Upload string.

        .parquet files are pre-processed (all derived columns already present).
        Works on both local and cloud (no irsdk needed).

        Parameters
        ----------
        b64_content : str  — base64 string, possibly with data-URI prefix

        Returns
        -------
        pl.DataFrame  or  None on error
        """
        if ',' in b64_content:
            b64_content = b64_content.split(',', 1)[1]

        raw_bytes = base64.b64decode(b64_content)

        try:
            df = pl.read_parquet(io.BytesIO(raw_bytes))
        except Exception as e:
            print(f'Failed to read .parquet: {e}', flush=True)
            return None

        # If the parquet was created by an older version without derived cols,
        # recompute them now.
        if 'TC' not in df.columns:
            df = self._add_derived_columns(df)

        self._session_df = df
        self._mode = 'file'

        print(f'Loaded {len(df)} frames from .parquet.', flush=True)
        return df

    # -------------------------------------------------------------------------
    # DERIVED PHYSICS COLUMNS
    # -------------------------------------------------------------------------

    def _add_derived_columns(self, df):
        """Compute and append all derived physics columns to a raw DataFrame.

        Added columns:
            TC           — normalised traction circle usage (0–1+)
            TC_color     — hex string for Plotly trace colouring (Fig 3A)
            USR_TC       — steering responsiveness ratio (0=good, 1=wasteful)
            USR_color    — hex string for Plotly trace colouring (Fig 3B)
            Balance_pct  — over/understeer score −100 to +100
            SlipFL/FR/RL/RR — per-wheel slip angles (deg)
            LapDist_m    — metres from S/F (computed if LapDist channel absent)

        Parameters
        ----------
        df : pl.DataFrame  — raw frame data

        Returns
        -------
        pl.DataFrame  — same data with added columns
        """
        # Convert to numpy for vectorised physics operations
        n = len(df)

        def col_or_nan(name):
            """Extract column as numpy array, or NaN array if absent."""
            if name in df.columns:
                return df[name].to_numpy().astype(float)
            return np.full(n, np.nan)

        speed   = col_or_nan('Speed')
        yaw_r   = col_or_nan('YawRate')
        lat_vel = col_or_nan('VelocityY')
        steer   = col_or_nan('SteeringWheelAngle')
        lat_acc = col_or_nan('LatAccel')
        lon_acc = col_or_nan('LongAccel')
        throttle= col_or_nan('Throttle')
        brake   = col_or_nan('Brake')

        # ---- LapDist in metres ----
        if 'LapDist' in df.columns and df['LapDist'].is_not_nan().sum() > 0:
            lap_dist_m = col_or_nan('LapDist')
        elif 'LapDistPct' in df.columns:
            # Estimate: find the max LapDist or use a config default
            # iRacing's LapDistPct is 0-1; multiply by track length.
            # Track length is not always available; use 3 km as a safe default.
            # Calibrate TRACK_LENGTH_M in config.py if needed.
            lap_dist_pct = col_or_nan('LapDistPct')
            # We'll compute a rough estimate — accurate enough for X-axis
            lap_dist_m = lap_dist_pct * 3000.0  # placeholder; update per track
        else:
            lap_dist_m = np.arange(n, dtype=float)  # frame index as fallback

        # ---- Traction circle ----
        # Use session max for normalisation (better than fixed default)
        ax_max = float(np.nanpercentile(np.abs(lon_acc), 99)) if np.any(np.isfinite(lon_acc)) else AX_MAX_DEFAULT
        ay_max = float(np.nanpercentile(np.abs(lat_acc), 99)) if np.any(np.isfinite(lat_acc)) else AY_MAX_DEFAULT
        ax_max = max(ax_max, 1.0)   # prevent divide-by-zero from bad sessions
        ay_max = max(ay_max, 1.0)

        tc_arr = phys.compute_traction_circle_array(lon_acc, lat_acc, ax_max, ay_max)

        # ---- Normalised lateral accel for USR_TC ----
        Y_TC = lat_acc / (ay_max + EPSILON)

        # ---- USR_TC (steering responsiveness) ----
        usr_arr = phys.compute_usr_tc_array(steer, Y_TC)

        # ---- Balance score ----
        # Compute per-wheel slip angles if wheel speed data is available
        has_ws = all(c in df.columns for c in
                     ['LFwheelSpeed', 'RFwheelSpeed', 'LRwheelSpeed', 'RRwheelSpeed'])

        track_f = col_or_nan('TrackWidth_F') if 'TrackWidth_F' in df.columns else np.full(n, TRACK_F_INIT)
        track_r = col_or_nan('TrackWidth_R') if 'TrackWidth_R' in df.columns else np.full(n, TRACK_R_INIT)

        if has_ws:
            # Compute slip angles for front (average FL/FR) and rear (average RL/RR)
            # We can't call the per-frame function vectorised easily, so loop for now.
            # TODO: vectorise wheel_slip_angles in a future optimisation pass.
            alpha_f_arr = np.full(n, np.nan)
            alpha_r_arr = np.full(n, np.nan)
            lf_ws = col_or_nan('LFwheelSpeed')
            rf_ws = col_or_nan('RFwheelSpeed')
            lr_ws = col_or_nan('LRwheelSpeed')
            rr_ws = col_or_nan('RRwheelSpeed')
            for i in range(n):
                if all(np.isfinite(x) for x in [speed[i], lat_vel[i], yaw_r[i], steer[i]]):
                    try:
                        alphas = phys.wheel_slip_angles(
                            speed[i], lat_vel[i], yaw_r[i], steer[i],
                            track_f[i] if np.isfinite(track_f[i]) else TRACK_F_INIT,
                            track_r[i] if np.isfinite(track_r[i]) else TRACK_R_INIT,
                        )
                        alpha_f_arr[i] = (alphas['FL'] + alphas['FR']) / 2.0
                        alpha_r_arr[i] = (alphas['RL'] + alphas['RR']) / 2.0
                    except Exception:
                        pass
        else:
            alpha_f_arr = None
            alpha_r_arr = None

        balance = phys.compute_balance_score_array(
            speed, yaw_r, steer, lat_acc, alpha_f_arr, alpha_r_arr)

        # ---- Colour arrays (computed once here, not in every callback) ----
        tc_colors  = phys.color_from_tc_array(tc_arr)
        usr_colors = phys.color_from_usr_array(usr_arr)

        # ---- Throttle / brake as 0-100% ----
        throttle_pct = np.where(np.isfinite(throttle), throttle * 100.0, np.nan)
        brake_pct    = np.where(np.isfinite(brake),    brake    * 100.0, np.nan)

        # ---- Append columns to DataFrame ----
        df = df.with_columns([
            pl.Series('LapDist_m',    lap_dist_m),
            pl.Series('TC',           tc_arr.astype(float)),
            pl.Series('TC_color',     tc_colors),
            pl.Series('USR_TC',       usr_arr.astype(float)),
            pl.Series('USR_color',    usr_colors),
            pl.Series('Balance_pct',  balance['balance_pct'].astype(float)),
            pl.Series('Throttle_pct', throttle_pct),
            pl.Series('Brake_pct',    brake_pct),
            pl.Series('ax_max_used',  np.full(n, ax_max)),  # stored for info
            pl.Series('ay_max_used',  np.full(n, ay_max)),
        ])

        # Add per-wheel slip angles if computed
        if alpha_f_arr is not None:
            df = df.with_columns([
                pl.Series('SlipAngle_F_deg', np.rad2deg(alpha_f_arr)),
                pl.Series('SlipAngle_R_deg', np.rad2deg(alpha_r_arr)),
            ])

        return df

    # -------------------------------------------------------------------------
    # SERIALISATION / DESERIALISATION FOR dcc.Store
    # -------------------------------------------------------------------------

    def df_to_json(self, df=None):
        """Serialise a Polars DataFrame to a JSON string for dcc.Store.

        Parameters
        ----------
        df : pl.DataFrame, optional  — defaults to self._session_df

        Returns
        -------
        str  — JSON (row-oriented list of dicts), or None if no data
        """
        if df is None:
            df = self._session_df
        if df is None:
            return None
        # Polars 1.x removed the `row_oriented` kwarg from write_json().
        # to_dicts() produces a list of row dicts; json.dumps gives a JSON string.
        # fill_nan(None) converts float NaN → JSON null (JSON has no NaN literal).
        return json.dumps(df.fill_nan(None).to_dicts())

    @staticmethod
    def df_from_json(json_str):
        """Deserialise a JSON string back to a Polars DataFrame.

        Parameters
        ----------
        json_str : str  — JSON produced by df_to_json()

        Returns
        -------
        pl.DataFrame  or  None on error
        """
        if not json_str:
            return None
        try:
            # Companion to df_to_json(): list-of-dicts JSON → Polars DataFrame.
            return pl.from_dicts(json.loads(json_str))
        except Exception as e:
            print(f'df_from_json error: {e}', flush=True)
            return None

    # -------------------------------------------------------------------------
    # CONVENIENCE: Get unique lap list from a DataFrame
    # -------------------------------------------------------------------------

    @staticmethod
    def get_lap_list(df):
        """Return a sorted list of unique lap numbers in the DataFrame.

        Parameters
        ----------
        df : pl.DataFrame

        Returns
        -------
        list[int]  e.g. [1, 2, 3, 4]
        """
        if df is None or 'Lap' not in df.columns:
            return []
        return sorted(df['Lap'].cast(pl.Int32).unique().to_list())

    @staticmethod
    def filter_laps(df, lap_list):
        """Return rows belonging to any of the specified lap numbers.

        Parameters
        ----------
        df       : pl.DataFrame
        lap_list : list[int]  — e.g. [1, 3] to show only laps 1 and 3

        Returns
        -------
        pl.DataFrame  (filtered)
        """
        if df is None or not lap_list:
            return df
        return df.filter(pl.col('Lap').cast(pl.Int32).is_in(lap_list))
