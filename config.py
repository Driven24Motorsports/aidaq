# =============================================================================
# config.py — aiDAQ Global Configuration
# =============================================================================
# This file is the single source of truth for:
#   1. Deployment mode (local app vs. Railway cloud)
#   2. Car geometry constants (update these per car/series)
#   3. Physics tuning constants (traction circle thresholds, balance weights)
#   4. UI/data constants
#
# HOW TO USE:
#   - Local app:  run  python aidaq_app.py  (AIDAQ_MODE defaults to 'local')
#   - Cloud app:  set environment variable  AIDAQ_MODE=cloud  on Railway,
#                 then push to GitHub → Railway auto-deploys.
#   - Per-car:    change WHEELBASE_M, FRONT_WEIGHT_FRAC, TRACK_F_INIT, TRACK_R_INIT
#                 in this file (or override via env vars later).
# =============================================================================

import os

# ---------------------------------------------------------------------------
# DEPLOYMENT MODE
# ---------------------------------------------------------------------------
# 'local' → live iRacing telemetry tab enabled, irsdk can be imported,
#            .ibt file upload shows parquet export button.
# 'cloud' → live tab hidden, irsdk import skipped (not available on server),
#            only .parquet upload is shown.
# Set the environment variable AIDAQ_MODE=cloud on Railway.
DEPLOYMENT_MODE = os.environ.get('AIDAQ_MODE', 'local')

# ---------------------------------------------------------------------------
# CAR GEOMETRY  (update per car — these are Lotus 49 estimates)
# ---------------------------------------------------------------------------

# Distance from front axle to centre of gravity (metres)
# Formula: WHEELBASE * (1 - FRONT_WEIGHT_FRAC)
WHEELBASE_M         = 2.60   # total wheelbase in metres

FRONT_WEIGHT_FRAC   = 0.53   # fraction of weight on front axle (0–1)
#   > 0.5 → front-heavy  |  < 0.5 → rear-heavy

# Derived CG-to-axle distances (used in slip-angle equations)
A = WHEELBASE_M * (1.0 - FRONT_WEIGHT_FRAC)   # CG → front axle  (m)
B = WHEELBASE_M * FRONT_WEIGHT_FRAC            # CG → rear axle   (m)

# Initial track width guesses — updated live by MedianFilter
TRACK_F_INIT = 1.524   # front track width (m)
TRACK_R_INIT = 1.549   # rear  track width (m)

# ---------------------------------------------------------------------------
# PHYSICS CONSTANTS
# ---------------------------------------------------------------------------

# Small value added to denominators to prevent divide-by-zero errors.
EPSILON = 1e-6

# Default max accelerations used before session maximums are computed.
# These set the traction circle radius at startup.
# Units: m/s²  (1 g ≈ 9.81 m/s²)
AX_MAX_DEFAULT = 2.0 * 9.81   # ~2 g longitudinal (braking/accel)
AY_MAX_DEFAULT = 2.0 * 9.81   # ~2 g lateral (cornering)

# ---------------------------------------------------------------------------
# TRACTION CIRCLE COLOR THRESHOLDS
# ---------------------------------------------------------------------------
# TC = sqrt( (ax/ax_max)² + (ay/ay_max)² )   — normalised grip usage
# TC = 1.0  →  exactly at the grip limit  (green zone below)
#
#  TC range          Colour     Meaning
#  ──────────────    ──────     ──────────────────────────────────────────
#  > TC_RED_HIGH     RED        well over limit — sliding / loss of control
#  TC_GREEN_LOW ..   GREEN      at or near limit — ideal grip usage
#    TC_RED_HIGH
#  TC_YELLOW_LOW ..  YELLOW     below limit, some grip left — near zone
#    TC_GREEN_LOW
#  < TC_YELLOW_LOW   RED        well under limit — significant grip unused

TC_RED_HIGH    = 1.25   # above this → red (over limit)
TC_GREEN_HIGH  = 1.25   # same upper bound for green zone start
TC_GREEN_LOW   = 0.85   # green zone: 0.85 – 1.25
TC_YELLOW_LOW  = 0.65   # yellow zone: 0.65 – 0.85 (or 1.25 – ?)
# below TC_YELLOW_LOW → red again (well under limit)

# ---------------------------------------------------------------------------
# STEERING RESPONSIVENESS (USR_TC) COLOR THRESHOLDS
# ---------------------------------------------------------------------------
# USR_TC = 0  → steering is responsive (every degree of lock = more cornering)
# USR_TC = 1  → steering is wasteful   (adding lock produces no extra lateral g)
#
#  USR_TC range      Colour     Meaning
#  ─────────────     ──────     ──────────────────────────────────────────
#  < USR_GREEN_MAX   GREEN      steering efficient — lock is producing cornering
#  USR_GREEN_MAX ..  YELLOW     moderate waste — becoming unresponsive
#    USR_RED_MIN
#  > USR_RED_MIN     RED        steering wasteful — adding lock, not gaining grip

USR_GREEN_MAX = 0.25   # below this → green
USR_RED_MIN   = 0.60   # above this → red

# ---------------------------------------------------------------------------
# BALANCE SCORE WEIGHTS
# ---------------------------------------------------------------------------
# B_raw = w_yaw * B_yaw + w_ay * B_ay   (when slip angles not trusted)
# Start with these; update once slip-angle estimates are validated.
BALANCE_W_YAW   = 0.65
BALANCE_W_AY    = 0.35

# Max expected |Balance%| for piecewise normalisation to [-100, +100].
# These are calibrated from a driver deliberately inducing max understeer/oversteer.
# Before calibration, a conservative estimate is used.
BALANCE_MAX_OVERSTEER  = 1.5   # raw B_raw value mapped to +100%
BALANCE_MAX_UNDERSTEER = 1.5   # raw |B_raw| value mapped to -100%

# ---------------------------------------------------------------------------
# STEERING RESPONSIVENESS GAIN
# ---------------------------------------------------------------------------
# k_delta in: USR_TC = 1 - min(1, delta_Y_TC / (k_delta * delta_steer+ + ε))
# Empirical — increase if USR shows too many red segments on balanced corners.
USR_K_DELTA = 1.0

# ---------------------------------------------------------------------------
# DISPLAY / UI CONSTANTS
# ---------------------------------------------------------------------------

# Traction circle window: ± metres around cursor shown in Fig 3A
TC_WINDOW_M = 150   # metres each side of cursor position

# Max data points per trace sent to Plotly (downsampled if needed for speed)
MAX_PLOTLY_POINTS = 5_000

# Live telemetry poll rate (milliseconds between Dash dcc.Interval ticks)
LIVE_INTERVAL_MS = 100   # 10 Hz UI refresh

# Play-back speed (metres of track distance advanced per play tick)
PLAY_STEP_M      = 20    # metres per tick
PLAY_INTERVAL_MS = 200   # tick rate (ms) → 20/200ms = 100 m/s playback
