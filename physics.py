# =============================================================================
# physics.py — aiDAQ Vehicle Dynamics & Signal Processing
# =============================================================================
# This module contains ALL physics, maths, and signal-processing functions.
# It has NO knowledge of the UI, Dash, Plotly, or iRacing SDK.
# That separation means:
#   - A new developer can read and test the maths here without running the app.
#   - The same functions run identically on the local .exe and the cloud server.
#   - Unit tests can import this file directly without starting Dash.
#
# SECTIONS:
#   1. Utility helpers
#   2. Moving median filter (track width estimation)
#   3. Car geometry + iRacing variable map (ported from op_fixteleme_figs0330.py)
#   4. Track-width estimator        (ported)
#   5. Per-wheel slip angles        (ported)
#   6. Over/under/neutral state     (ported)
#   7. iRacing telemetry inventory  (ported)
#   8. Startup diagnostics          (ported, matplotlib refs removed)
#   9. Traction circle              (NEW)
#  10. Traction Circle Steering Responsiveness (TCSR / USR_TC)  (NEW)
#  11. Car balance score (yaw error + lateral accel agreement)   (NEW)
#  12. Colour helpers for Plotly traces                          (NEW)
#  13. Data-display helpers (downsampling)                       (NEW)
# =============================================================================

import math
import sys
import platform
import collections

import numpy as np

from config import (
    WHEELBASE_M, FRONT_WEIGHT_FRAC, A, B,
    TRACK_F_INIT, TRACK_R_INIT,
    EPSILON,
    AX_MAX_DEFAULT, AY_MAX_DEFAULT,
    TC_GREEN_LOW, TC_GREEN_HIGH, TC_YELLOW_LOW,
    USR_GREEN_MAX, USR_RED_MIN,
    USR_K_DELTA,
    BALANCE_W_YAW, BALANCE_W_AY,
    BALANCE_MAX_OVERSTEER, BALANCE_MAX_UNDERSTEER,
    MAX_PLOTLY_POINTS,
)


# =============================================================================
# 1. UTILITY HELPERS
# =============================================================================

def deg(x):
    """Convert radians to degrees.  Works on scalars and numpy arrays."""
    return np.rad2deg(x)


def clip(x, a, b):
    """Clamp scalar x into the closed interval [a, b]."""
    return max(a, min(b, x))


def clip_arr(arr, a, b):
    """Clamp a numpy array into [a, b] element-wise."""
    return np.clip(arr, a, b)


# =============================================================================
# 2. MOVING MEDIAN FILTER
# =============================================================================
# Used to smooth the track-width estimates computed from wheel speed differences.
# Outliers (wheel-lock, kerb strike) are rejected because the median is
# insensitive to a small fraction of bad values.

class MedianFilter:
    """Fixed-length circular buffer that returns the median of its contents.

    Parameters
    ----------
    N : int
        Window length (number of samples to keep).  Default 200.

    Usage
    -----
    f = MedianFilter(N=400)
    f.push(estimated_value)
    smoothed = f.median(default=1.52)   # falls back to `default` when empty
    """

    def __init__(self, N=200):
        self.buf = collections.deque(maxlen=N)

    def push(self, x):
        """Append x only if it is a real finite number (ignore NaN/Inf)."""
        if np.isfinite(x):
            self.buf.append(float(x))

    def median(self, default=np.nan):
        """Return median of current window, or `default` if no data yet."""
        return float(np.median(self.buf)) if self.buf else default


# Module-level filter instances (shared state for live session)
# These are re-created inside DataSource for file-mode per-session isolation.
f_track_med = MedianFilter(400)
r_track_med = MedianFilter(400)


# =============================================================================
# 3. CAR GEOMETRY + iRACING VARIABLE MAP
# =============================================================================
# VAR maps short internal keys → iRacing telemetry channel names.
# If you switch to a different channel (e.g. a different steering var),
# only change the value in this dict — not every place it's used.

VAR = {
    'speed_mps':   'Speed',               # vehicle speed  (m/s)
    'yaw_rate_rps':'YawRate',             # yaw rate       (rad/s, + = CCW)
    'lat_vel_mps': 'VelocityY',           # lateral body velocity (m/s)
    'steer_rad':   'SteeringWheelAngle',  # steering wheel angle  (rad)
    'lf_ws':       'LFwheelSpeed',        # wheel speed, left-front  (rad/s)
    'rf_ws':       'RFwheelSpeed',        # wheel speed, right-front (rad/s)
    'lr_ws':       'LRwheelSpeed',        # wheel speed, left-rear   (rad/s)
    'rr_ws':       'RRwheelSpeed',        # wheel speed, right-rear  (rad/s)
    'lat_accel':   'LatAccel',            # lateral acceleration     (m/s²)
    'long_accel':  'LongAccel',           # longitudinal acceleration(m/s²)
    'lap_dist':    'LapDist',             # distance from S/F this lap (m)
    'lap_dist_pct':'LapDistPct',          # same, as 0-1 fraction
    'throttle':    'Throttle',            # throttle pedal 0-1
    'brake':       'Brake',               # brake pedal    0-1
    'lap':         'Lap',                 # lap counter
    'session_time':'SessionTime',         # session clock  (s)
    'yaw_north':   'YawNorth',            # heading from north (rad)
}

# REQUIRED: the app will fail on startup if any of these are missing.
# These are the minimum needed for slip-angle and balance calculations.
REQUIRED_VARS = [
    'Speed',
    'YawRate',
    'VelocityY',
    'SteeringWheelAngle',
    'SessionTime',
    'Lap',
    'LapDistPct',
]

# OPTIONAL: nice-to-have but the app degrades gracefully without them.
OPTIONAL_VARS = [
    'LFwheelSpeed', 'RFwheelSpeed', 'LRwheelSpeed', 'RRwheelSpeed',
    'Throttle', 'Brake',
    'LapDist',
    'LatAccel', 'LongAccel',
    'YawNorth',
]


# =============================================================================
# 4. TRACK-WIDTH ESTIMATOR
# =============================================================================
# When the car is cornering, the inside and outside wheels travel different
# arcs. The width of that arc difference divided by the yaw rate gives an
# estimate of the track width.  A median filter rejects transient noise.

def _get_ir(ir, key_name, telemetry_vars):
    """Safe iRSdk read: returns float or NaN if channel is absent."""
    var_name = VAR.get(key_name)
    if var_name and var_name in telemetry_vars:
        return float(ir[var_name])
    return np.nan


def update_track_widths(ir, u, r, telemetry_vars,
                        f_filter=None, r_filter=None):
    """Estimate front and rear track widths from wheel-speed differential.

    Parameters
    ----------
    ir              : irsdk.IRSDK  — live SDK handle (or None in file mode)
    u               : float        — vehicle speed  (m/s)
    r               : float        — yaw rate       (rad/s)
    telemetry_vars  : set[str]     — available iRacing channel names
    f_filter        : MedianFilter — front width filter (module-level default)
    r_filter        : MedianFilter — rear  width filter

    Returns
    -------
    (tf, tr) : (float, float)  track widths in metres
               Falls back to TRACK_F/R_INIT when data is insufficient.
    """
    if f_filter is None:
        f_filter = f_track_med
    if r_filter is None:
        r_filter = r_track_med

    # Need meaningful yaw rate and speed to compute arc difference
    if abs(r) < 1e-3 or u < 5.0:
        return None, None

    lf = _get_ir(ir, 'lf_ws', telemetry_vars)
    rf = _get_ir(ir, 'rf_ws', telemetry_vars)
    lr = _get_ir(ir, 'lr_ws', telemetry_vars)
    rr = _get_ir(ir, 'rr_ws', telemetry_vars)

    if not all(np.isfinite(x) for x in [lf, rf, lr, rr]):
        return TRACK_F_INIT, TRACK_R_INIT

    tf = TRACK_F_INIT
    tr = TRACK_R_INIT

    if np.isfinite(lf) and np.isfinite(rf):
        dv = max(lf, rf) - min(lf, rf)          # speed difference front axle
        tf_est = dv / abs(r)                      # arc width = dv / yaw_rate
        if 0.9 < tf_est < 2.2:                    # sanity bounds (metres)
            f_filter.push(tf_est)
            tf = f_filter.median(default=TRACK_F_INIT)

    if np.isfinite(lr) and np.isfinite(rr):
        dv = max(lr, rr) - min(lr, rr)
        tr_est = dv / abs(r)
        if 0.9 < tr_est < 2.2:
            r_filter.push(tr_est)
            tr = r_filter.median(default=TRACK_R_INIT)

    return tf, tr


# =============================================================================
# 5. PER-WHEEL SLIP ANGLES
# =============================================================================
# Slip angle α for each wheel = angle between the wheel's heading and its
# actual velocity vector.  Positive α = tyre is pushing outward.
#
# Bicycle model corner positions:
#   Front wheels sit at  +A  ahead of CG (x), ±tf/2 lateral (y)
#   Rear  wheels sit at  -B  behind CG  (x), ±tr/2 lateral (y)
#
# Body velocities at each corner:
#   vx_i = u  - r * y_i     (longitudinal, reduced by yaw at lateral offset)
#   vy_i = v  + r * x_i     (lateral,      increased by yaw at longitudinal offset)

def wheel_slip_angles(u, v, r, delta, tf, tr):
    """Compute the four-wheel slip angles.

    Parameters
    ----------
    u     : float  — longitudinal speed   (m/s)
    v     : float  — lateral body speed   (m/s)
    r     : float  — yaw rate             (rad/s)
    delta : float  — road-wheel steer angle (rad). Positive = left turn.
                     NOTE: iRacing SteeringWheelAngle needs dividing by the
                     steering ratio to get road-wheel angle.  The current
                     code passes the raw wheel angle as an approximation.
    tf    : float  — front track width    (m)
    tr    : float  — rear  track width    (m)

    Returns
    -------
    dict : {'FL': α, 'FR': α, 'RL': α, 'RR': α}  angles in radians
    """
    pos = {
        'FL': (+A, +tf / 2.0),
        'FR': (+A, -tf / 2.0),
        'RL': (-B, +tr / 2.0),
        'RR': (-B, -tr / 2.0),
    }
    # Front wheels steer; rear wheels are fixed straight ahead
    headings = {'FL': delta, 'FR': delta, 'RL': 0.0, 'RR': 0.0}

    alphas = {}
    for k, (x_i, y_i) in pos.items():
        vx = u - r * y_i
        vy = v + r * x_i
        theta = headings[k]
        # atan2(vy, |vx|) gives the velocity direction; subtract heading
        alpha = math.atan2(vy, abs(vx)) - theta
        alphas[k] = alpha
    return alphas


# =============================================================================
# 6. OVER / UNDER / NEUTRAL STATE
# =============================================================================
# Simple yaw-error method: compare actual yaw rate to the "neutral steer" rate
# that a bicycle model predicts for the current speed and steer angle.

def yaw_tol(u):
    """Speed-dependent yaw tolerance (rad/s) below which we call it 'Neutral'.
    Prevents noise at low speeds from flipping the state constantly.
    """
    return 0.02 + 0.001 * u


def ou_state(u, r, delta):
    """Return 'Oversteer', 'Understeer', or 'Neutral'.

    Parameters
    ----------
    u     : float  — speed      (m/s)
    r     : float  — yaw rate   (rad/s)
    delta : float  — steer angle (rad)
    """
    r_ref = (u * delta) / WHEELBASE_M   # neutral-steer yaw rate prediction
    e = r - r_ref                        # positive → car rotating more → oversteer
    t = yaw_tol(u)
    if e > t:
        return 'Oversteer'
    if e < -t:
        return 'Understeer'
    return 'Neutral'


# =============================================================================
# 7. iRACING TELEMETRY INVENTORY
# =============================================================================
# pyirsdk exposes available channel names through various internal attributes
# depending on the SDK version.  This function tries all known patterns.

def get_telemetry_var_names(ir):
    """Return a set of available telemetry channel name strings from irsdk.

    Tries multiple internal attributes because different pyirsdk versions
    expose the header list differently.

    Returns
    -------
    set[str]  — empty set if none found (caller should warn loudly)
    """
    for attr in ('var_headers_names', '_var_headers_names', '_var_headers'):
        if hasattr(ir, attr):
            obj = getattr(ir, attr)
            # Case A: already a list/set of name strings
            if (isinstance(obj, (list, set, tuple))
                    and obj
                    and isinstance(next(iter(obj)), str)):
                return set(obj)
            # Case B: list of dicts or objects with a .name / ['name'] field
            names = set()
            try:
                for h in obj:
                    if isinstance(h, dict) and 'name' in h:
                        names.add(h['name'])
                    elif hasattr(h, 'name'):
                        names.add(h.name)
                if names:
                    return names
            except Exception:
                pass
    return set()


def inventory_vars(ir):
    """Convenience wrapper: get telemetry vars and warn if empty."""
    telemetry_vars = get_telemetry_var_names(ir)
    if not telemetry_vars:
        print('WARNING: telemetry_vars inventory returned 0. '
              'Validation will be limited.', flush=True)
    return telemetry_vars


# =============================================================================
# 8. STARTUP DIAGNOSTICS
# =============================================================================
# Prints a structured report when first connecting to iRacing.
# Helps quickly spot missing channels and configuration issues.
# (matplotlib references removed from original — this version is UI-agnostic)

def startup_diagnostics(ir, telemetry_vars, var_map=None):
    """Print a structured startup report and raise if required vars are absent.

    Parameters
    ----------
    ir              : irsdk.IRSDK   — connected SDK handle
    telemetry_vars  : set[str]      — from get_telemetry_var_names()
    var_map         : dict, optional— VAR dict to cross-check channel names
    """
    SEP = '=' * 70
    print('\n' + SEP)
    print('aiDAQ Startup Diagnostics')
    print(SEP)

    # Environment
    print(f'Python  : {sys.version.split()[0]}  |  Platform: {platform.platform()}')

    # Connection
    print(f'iRacing connected : {getattr(ir, "is_connected", None)}')

    # Quick safe reads of key sim-state vars
    def _safe(name, default=None):
        try:
            return ir[name]
        except Exception:
            return default

    print(f'SessionTime  : {_safe("SessionTime")}')
    print(f'IsOnTrack    : {_safe("IsOnTrack")}  |  '
          f'IsOnTrackCar: {_safe("IsOnTrackCar")}')
    print(f'Telemetry channel count: {len(telemetry_vars)}')

    # Required vars
    missing_req  = [v for v in REQUIRED_VARS if v not in telemetry_vars]
    present_req  = [v for v in REQUIRED_VARS if v in telemetry_vars]
    print('\nRequired channels:')
    for v in present_req:
        print(f'  ✓  {v}')
    for v in missing_req:
        print(f'  ✗  {v}  ← MISSING')

    # Optional vars
    present_opt = [v for v in OPTIONAL_VARS if v in telemetry_vars]
    missing_opt = [v for v in OPTIONAL_VARS if v not in telemetry_vars]
    print('\nOptional channels (OK if missing):')
    for v in present_opt[:20]:
        print(f'     {v}')
    if len(present_opt) > 20:
        print(f'  … +{len(present_opt) - 20} more optional channels present')
    if missing_opt:
        print(f'  (missing {len(missing_opt)} optional channels — some figure '
              'features may be limited)')

    # VAR-map cross-check
    if var_map:
        print('\nVAR mapping validation:')
        bad = [(k, n) for k, n in var_map.items() if n not in telemetry_vars]
        if not bad:
            print('  All VAR-mapped channel names exist.')
        else:
            print('  Missing VAR-mapped channel names:')
            for k, n in bad[:30]:
                print(f'    {k} → {n}')
            if len(bad) > 30:
                print(f'  … +{len(bad) - 30} more')

    # Hard fail if required vars are absent
    if missing_req:
        print(f'\n✗  Startup validation FAILED — missing required channels: '
              f'{missing_req}')
        print('   Fix: drive on-track in a live session OR adjust REQUIRED_VARS.')
        print(SEP + '\n')
        raise RuntimeError(f'Missing required telemetry channels: {missing_req}')

    print('\n✓  Startup validation PASSED.')
    print(SEP + '\n')


# =============================================================================
# 9. TRACTION CIRCLE
# =============================================================================
# The traction circle plots normalised lateral vs. longitudinal acceleration.
# TC = sqrt( (ax/ax_max)² + (ay/ay_max)² )
#
#   TC < 1  → inside the grip limit   (available grip unused)
#   TC ≈ 1  → at the grip limit       (ideal — maximising tyres)
#   TC > 1  → beyond the grip limit   (likely sliding)
#
# ax_max / ay_max should be the session-maximum observed values OR
# a known physical estimate for the car+tyre combination.

def compute_traction_circle(ax, ay, ax_max=None, ay_max=None):
    """Compute normalised traction-circle usage TC for a single sample.

    Parameters
    ----------
    ax      : float  — longitudinal accel (m/s²). Positive = accelerating.
    ay      : float  — lateral accel      (m/s²). Positive = left.
    ax_max  : float  — max |ax| for normalisation (defaults to AX_MAX_DEFAULT)
    ay_max  : float  — max |ay| for normalisation (defaults to AY_MAX_DEFAULT)

    Returns
    -------
    float  — TC value (0+).  1.0 = exactly on the grip-limit circle.
    """
    if ax_max is None:
        ax_max = AX_MAX_DEFAULT
    if ay_max is None:
        ay_max = AY_MAX_DEFAULT

    ax_norm = ax / (ax_max + EPSILON)
    ay_norm = ay / (ay_max + EPSILON)
    return math.sqrt(ax_norm ** 2 + ay_norm ** 2)


def compute_traction_circle_array(ax_arr, ay_arr, ax_max=None, ay_max=None):
    """Vectorised version of compute_traction_circle for numpy arrays.

    Parameters
    ----------
    ax_arr  : np.ndarray  — longitudinal accel array (m/s²)
    ay_arr  : np.ndarray  — lateral accel array      (m/s²)
    ax_max  : float       — normalisation max (scalar)
    ay_max  : float       — normalisation max (scalar)

    Returns
    -------
    np.ndarray of TC values (same length as input)
    """
    if ax_max is None:
        ax_max = AX_MAX_DEFAULT
    if ay_max is None:
        ay_max = AY_MAX_DEFAULT

    ax_norm = np.asarray(ax_arr, dtype=float) / (ax_max + EPSILON)
    ay_norm = np.asarray(ay_arr, dtype=float) / (ay_max + EPSILON)
    return np.sqrt(ax_norm ** 2 + ay_norm ** 2)


# =============================================================================
# 10. TRACTION CIRCLE STEERING RESPONSIVENESS  (TCSR / USR_TC)
# =============================================================================
# On the traction circle, the Y-axis is normalised lateral acceleration.
# When the driver adds steering, we expect Y_TC to increase (more cornering).
# If steering increases but Y_TC does NOT increase, that steering is wasteful.
#
# TCSR  = ΔY_TC / (Δδ+ + ε)
#   High TCSR → added steering produced more lateral utilisation  (good)
#   Low  TCSR → added steering produced little extra cornering    (neutral)
#   Near zero / negative → steering is likely wasteful or mistimed
#
# USR_TC = 1 - min(1, ΔY_TC / (k_δ * Δδ+ + ε))
#   USR_TC ≈ 0 → responsive steering (green)
#   USR_TC ≈ 1 → unresponsive / unnecessary steering (red)

def compute_tcsr(delta_t, delta_t_prev, Y_TC_t, Y_TC_t_prev):
    """Traction Circle Steering Responsiveness for a single time-step.

    Parameters
    ----------
    delta_t      : float  — steering wheel angle at time t  (rad)
    delta_t_prev : float  — steering wheel angle at t-1     (rad)
    Y_TC_t       : float  — normalised lateral accel at t   (ay/ay_max)
    Y_TC_t_prev  : float  — normalised lateral accel at t-1

    Returns
    -------
    float : TCSR value  (high = responsive, near 0 = wasteful)
    """
    # Only count steering increments (adding lock), not unwinding
    delta_steer_plus = max(0.0, abs(delta_t) - abs(delta_t_prev))
    delta_Y_TC = Y_TC_t - Y_TC_t_prev
    return delta_Y_TC / (delta_steer_plus + EPSILON)


def compute_usr_tc(delta_steer_plus, delta_Y_TC, k_delta=None):
    """Unnecessary Steering Ratio on Traction Circle (USR_TC).

    Parameters
    ----------
    delta_steer_plus : float  — positive steering increment (rad, ≥ 0)
    delta_Y_TC       : float  — change in normalised lateral accel
    k_delta          : float  — expected gain per unit steering (default USR_K_DELTA)

    Returns
    -------
    float in [0, 1]:  0 = fully responsive,  1 = fully unresponsive
    """
    if k_delta is None:
        k_delta = USR_K_DELTA

    numerator   = delta_Y_TC
    denominator = k_delta * delta_steer_plus + EPSILON
    return 1.0 - min(1.0, numerator / denominator)


def compute_usr_tc_array(delta_arr, Y_TC_arr, k_delta=None):
    """Vectorised USR_TC for a full lap's worth of data.

    Takes the steering-angle and normalised-lateral-accel arrays,
    computes per-sample increments, then returns the USR_TC array.

    Parameters
    ----------
    delta_arr  : np.ndarray  — steering wheel angle (rad), shape (N,)
    Y_TC_arr   : np.ndarray  — normalised lateral accel,   shape (N,)
    k_delta    : float

    Returns
    -------
    np.ndarray of USR_TC values, shape (N,).  First element is 0 (no prev).
    """
    if k_delta is None:
        k_delta = USR_K_DELTA

    delta_arr = np.asarray(delta_arr, dtype=float)
    Y_TC_arr  = np.asarray(Y_TC_arr,  dtype=float)

    # Steering increment: only positive (adding lock) counts
    d_steer = np.diff(np.abs(delta_arr), prepend=np.abs(delta_arr[0]))
    d_steer_plus = np.maximum(0.0, d_steer)

    # Change in lateral utilisation
    d_Y_TC = np.diff(Y_TC_arr, prepend=Y_TC_arr[0])

    usr = 1.0 - np.minimum(1.0, d_Y_TC / (k_delta * d_steer_plus + EPSILON))
    return np.clip(usr, 0.0, 1.0)


# =============================================================================
# 11. CAR BALANCE SCORE
# =============================================================================
# Answers the question: "Is the car rotating as much as the steering predicts?"
#
# Step 1 — Expected yaw rate from bicycle model:
#   r_exp = (U / L) * tan(δ)
#
# Step 2 — Three normalised balance signals:
#   B_yaw  = (r - r_exp) / (|r_exp| + ε)
#             negative = understeer, positive = oversteer
#   B_ay   = (ay - ay_exp) / (|ay_exp| + ε)
#             cross-checks yaw signal with lateral g
#   B_α    = (α_r - α_f) / (|α_f| + |α_r| + ε)
#             needs estimated front/rear slip angles; most accurate signal
#
# Step 3 — Combined raw balance:
#   B_raw = 0.65 * B_yaw + 0.35 * B_ay   (when slip angles not trusted)
#
# Step 4 — Normalised to ±100%:
#   Balance% ∈ [−100, +100]   (0 = neutral, −100 = max US, +100 = max OS)

def compute_balance_score(U, r, delta, ay,
                           alpha_f=None, alpha_r=None,
                           ax_max=None, ay_max=None):
    """Compute the full balance score for a single sample.

    Parameters
    ----------
    U       : float  — vehicle speed  (m/s)
    r       : float  — actual yaw rate (rad/s)
    delta   : float  — road-wheel steer angle (rad)
    ay      : float  — measured lateral accel (m/s²)
    alpha_f : float, optional  — front slip angle (rad)
    alpha_r : float, optional  — rear  slip angle (rad)
    ax_max  : float  — not used here (kept for future extension)
    ay_max  : float  — used if computing ay_exp

    Returns
    -------
    dict with keys: r_exp, B_yaw, B_ay, B_alpha (or NaN), B_raw, balance_pct
    """
    # Expected yaw rate from neutral-steer bicycle model
    r_exp = (U / WHEELBASE_M) * math.tan(delta)

    # ---- B_yaw: yaw-rate error normalised by expected yaw rate ----
    B_yaw = (r - r_exp) / (abs(r_exp) + EPSILON)

    # ---- B_ay: lateral accel agreement ----
    # Expected lateral accel = U * r_exp  (centripetal formula)
    ay_exp = U * r_exp
    B_ay = (ay - ay_exp) / (abs(ay_exp) + EPSILON)

    # ---- B_alpha: slip angle balance (most accurate if available) ----
    if alpha_f is not None and alpha_r is not None:
        B_alpha = (alpha_r - alpha_f) / (abs(alpha_f) + abs(alpha_r) + EPSILON)
    else:
        B_alpha = float('nan')

    # ---- Combined raw balance ----
    if not math.isnan(B_alpha):
        # Trust slip angles: use all three with research weights
        B_raw = 0.50 * B_yaw + 0.30 * B_ay + 0.20 * B_alpha
    else:
        # Slip angles not available: rely on yaw + ay only
        B_raw = BALANCE_W_YAW * B_yaw + BALANCE_W_AY * B_ay

    # ---- Normalise to ±100% ----
    if B_raw >= 0:
        balance_pct = 100.0 * B_raw / (BALANCE_MAX_OVERSTEER + EPSILON)
    else:
        balance_pct = 100.0 * B_raw / (BALANCE_MAX_UNDERSTEER + EPSILON)

    balance_pct = float(clip(balance_pct, -100.0, 100.0))

    return {
        'r_exp':       r_exp,
        'B_yaw':       B_yaw,
        'B_ay':        B_ay,
        'B_alpha':     B_alpha,
        'B_raw':       B_raw,
        'balance_pct': balance_pct,
    }


def compute_balance_score_array(U_arr, r_arr, delta_arr, ay_arr,
                                 alpha_f_arr=None, alpha_r_arr=None):
    """Vectorised balance score for a full lap.

    All input arrays must have the same length N.

    Parameters
    ----------
    U_arr     : np.ndarray  speed (m/s)
    r_arr     : np.ndarray  actual yaw rate (rad/s)
    delta_arr : np.ndarray  road-wheel steer angle (rad)
    ay_arr    : np.ndarray  lateral accel (m/s²)
    alpha_f_arr : np.ndarray or None  front slip angle (rad)
    alpha_r_arr : np.ndarray or None  rear  slip angle (rad)

    Returns
    -------
    dict with array values for: r_exp, B_yaw, B_ay, B_alpha, B_raw, balance_pct
    """
    U     = np.asarray(U_arr,     dtype=float)
    r     = np.asarray(r_arr,     dtype=float)
    delta = np.asarray(delta_arr, dtype=float)
    ay    = np.asarray(ay_arr,    dtype=float)

    r_exp  = (U / WHEELBASE_M) * np.tan(delta)
    B_yaw  = (r - r_exp) / (np.abs(r_exp) + EPSILON)
    ay_exp = U * r_exp
    B_ay   = (ay - ay_exp) / (np.abs(ay_exp) + EPSILON)

    if alpha_f_arr is not None and alpha_r_arr is not None:
        af = np.asarray(alpha_f_arr, dtype=float)
        ar = np.asarray(alpha_r_arr, dtype=float)
        B_alpha = (ar - af) / (np.abs(af) + np.abs(ar) + EPSILON)
        B_raw   = 0.50 * B_yaw + 0.30 * B_ay + 0.20 * B_alpha
    else:
        B_alpha = np.full_like(U, np.nan)
        B_raw   = BALANCE_W_YAW * B_yaw + BALANCE_W_AY * B_ay

    # Piecewise normalisation to ±100%
    balance_pct = np.where(
        B_raw >= 0,
        100.0 * B_raw / (BALANCE_MAX_OVERSTEER  + EPSILON),
        100.0 * B_raw / (BALANCE_MAX_UNDERSTEER + EPSILON),
    )
    balance_pct = np.clip(balance_pct, -100.0, 100.0)

    return {
        'r_exp':       r_exp,
        'B_yaw':       B_yaw,
        'B_ay':        B_ay,
        'B_alpha':     B_alpha,
        'B_raw':       B_raw,
        'balance_pct': balance_pct,
    }


# =============================================================================
# 12. COLOUR HELPERS FOR PLOTLY TRACES
# =============================================================================
# These convert physics metrics into hex colour strings that Plotly accepts.
# The colour scheme is defined by thresholds in config.py so it can be tuned
# without touching this file.

def color_from_tc(tc):
    """Return hex colour for a traction-circle TC value.

    Colour scheme (bidirectional — penalises BOTH over AND under the limit):
      Green  → 0.85 – 1.25  (at or near the grip limit — ideal)
      Yellow → 0.65 – 0.85  (below limit — grip unused)
      Red    → < 0.65 OR > 1.25  (well away from limit in either direction)

    Parameters
    ----------
    tc : float  — normalised traction circle usage

    Returns
    -------
    str  — hex colour e.g. '#00cc44'
    """
    if TC_GREEN_LOW <= tc <= TC_GREEN_HIGH:
        return '#00cc44'   # green  — at/near limit
    elif TC_YELLOW_LOW <= tc < TC_GREEN_LOW:
        return '#ffaa00'   # amber  — below limit (underusing grip)
    elif TC_GREEN_HIGH < tc <= 1.40:
        return '#ffaa00'   # amber  — slightly over limit
    else:
        return '#ff3333'   # red    — well outside the limit zone


def color_from_tc_array(tc_arr):
    """Vectorised color_from_tc — returns a list of hex strings."""
    return [color_from_tc(float(tc)) for tc in tc_arr]


def color_from_usr(usr_tc):
    """Return hex colour for a steering-responsiveness USR_TC value.

    Colour scheme (measures STEERING EFFICIENCY, not oversteer/understeer):
      Green  → USR_TC < 0.25  steering efficiently producing cornering
      Yellow → 0.25 – 0.60    becoming unresponsive
      Red    → > 0.60         wasteful steering — adding lock, losing grip

    Parameters
    ----------
    usr_tc : float in [0, 1]

    Returns
    -------
    str  — hex colour
    """
    if usr_tc < USR_GREEN_MAX:
        return '#00cc44'   # green  — responsive
    elif usr_tc < USR_RED_MIN:
        return '#ffaa00'   # amber  — moderate waste
    else:
        return '#ff3333'   # red    — wasteful / unresponsive


def color_from_usr_array(usr_arr):
    """Vectorised color_from_usr — returns a list of hex strings."""
    return [color_from_usr(float(u)) for u in usr_arr]


# =============================================================================
# 13. DATA-DISPLAY HELPERS
# =============================================================================
# Plotly can handle ~10 k points per trace before rendering slows noticeably.
# For a 20-minute session at 60 Hz that's 72 k rows per lap — too many.
# Downsample before sending to the browser.

def downsample_indices(n_total, n_target=None):
    """Return an index array that uniformly thins n_total to n_target rows.

    Parameters
    ----------
    n_total  : int  — total number of rows
    n_target : int  — desired number of rows (default MAX_PLOTLY_POINTS)

    Returns
    -------
    np.ndarray of int indices (sorted, unique)
    """
    if n_target is None:
        n_target = MAX_PLOTLY_POINTS
    if n_total <= n_target:
        return np.arange(n_total)
    step = n_total / n_target
    idx  = np.unique(np.round(np.arange(n_target) * step).astype(int))
    idx  = idx[idx < n_total]
    return idx


def downsample_arrays(arrays, n_target=None):
    """Downsample a dict of equal-length arrays to n_target rows.

    Parameters
    ----------
    arrays   : dict[str, np.ndarray]  — e.g. {'x': ..., 'y': ..., 'c': ...}
    n_target : int

    Returns
    -------
    dict[str, np.ndarray]  — same keys, shorter arrays
    """
    if not arrays:
        return arrays
    lengths = {len(v) for v in arrays.values()}
    assert len(lengths) == 1, 'All arrays must have the same length'
    n_total = lengths.pop()
    idx = downsample_indices(n_total, n_target)
    return {k: np.asarray(v)[idx] for k, v in arrays.items()}
