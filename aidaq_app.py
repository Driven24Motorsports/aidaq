# =============================================================================
# aidaq_app.py — Main Dash Application
# =============================================================================
# This is the entry point.  Run with:
#
#   python aidaq_app.py               ← local mode (default)
#   AIDAQ_MODE=cloud python aidaq_app.py  ← cloud mode (Railway sets this)
#
# ARCHITECTURE OVERVIEW:
#
#   ┌────────────┐    ┌──────────────────────────────────────────────────┐
#   │  Sidebar   │    │  Main Panel                                       │
#   │            │    │                                                    │
#   │ Mode       │    │  [Fig 1]  Track Map  (all laps, cursor dot)       │
#   │ Upload     │    │  [Fig 2]  Throttle & Brake vs Lap Distance        │
#   │ Lap select │    │  [Fig 3A] Traction Circle  │  [Fig 3B] Balance    │
#   └────────────┘    └──────────────────────────────────────────────────┘
#                     [====|====== scrubber ======] [▶ Play]
#
# CROSS-FIGURE LINKING:
#   A single dcc.Store('cursor-position') holds the current lap-distance
#   cursor value (metres).  Every figure callback reads from it.
#   Clicking any figure writes to it, propagating to all others.
#
# CALLBACKS (summary):
#   upload → session-data store (parse .ibt or .parquet)
#   session-data → lap-checklist options, scrubber range
#   fig1/fig2 clickData + scrubber + play-interval → cursor-position
#   [cursor-position, selected-laps, session-data] → fig1, fig2, fig3a, fig3b
#   play-button → play-state toggle
#
# =============================================================================

import json
import math
import os

import numpy as np
import polars as pl

import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from config import (
    DEPLOYMENT_MODE,
    TC_WINDOW_M,
    PLAY_STEP_M, PLAY_INTERVAL_MS,
    LIVE_INTERVAL_MS,
    MAX_PLOTLY_POINTS,
    AX_MAX_DEFAULT, AY_MAX_DEFAULT,
    EPSILON,
)
import physics as phys
from data_source import DataSource

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],   # dark theme from bootstrap
    title='aiDAQ',
    # suppress_callback_exceptions: needed because some components are created
    # dynamically (lap checklist options), so Dash can't validate them at startup.
    suppress_callback_exceptions=True,
)

# Global DataSource instance.
# In Dash, this module is imported once and the global object persists for
# the lifetime of the server process.  In production (multi-worker), use
# a shared cache (Redis) instead.
_ds = DataSource()


# =============================================================================
# LAYOUT HELPERS
# =============================================================================

def _sidebar():
    """Build the left sidebar with mode controls, upload, and lap selector."""

    # Mode selector (hidden in cloud mode — no iRacing available on server)
    mode_selector = dbc.Card([
        dbc.CardHeader('Mode'),
        dbc.CardBody([
            dcc.RadioItems(
                id='mode-selector',
                options=[
                    {'label': '  Live iRacing',  'value': 'live'},
                    {'label': '  Post-Session',   'value': 'file'},
                ],
                value='file',
                inputStyle={'margin-right': '6px'},
                labelStyle={'display': 'block', 'margin-bottom': '6px'},
            ),
        ]),
    ], className='mb-2',
       style={'display': 'none'} if DEPLOYMENT_MODE == 'cloud' else {})

    # Upload zone — label changes based on deployment mode
    upload_label = 'Upload .parquet' if DEPLOYMENT_MODE == 'cloud' else 'Upload .ibt or .parquet'
    upload_accept = '.parquet' if DEPLOYMENT_MODE == 'cloud' else '.ibt,.parquet'

    upload_card = dbc.Card([
        dbc.CardHeader('Session File'),
        dbc.CardBody([
            dcc.Upload(
                id='file-upload',
                children=html.Div([
                    html.Span('Drag & Drop or '),
                    html.A('Select File', style={'color': '#aef'}),
                    html.Br(),
                    html.Small(upload_label, style={'color': '#888'}),
                ]),
                style={
                    'width': '100%', 'height': '70px',
                    'lineHeight': '18px', 'borderWidth': '1px',
                    'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'padding': '10px',
                    'cursor': 'pointer',
                },
                accept=upload_accept,
                multiple=False,
            ),
            html.Div(id='upload-status',
                     style={'color': '#aaa', 'fontSize': '12px', 'marginTop': '4px'}),

            # Download .parquet button (local mode only)
            html.Div(
                dcc.Download(id='download-parquet'),
                style={'display': 'none'} if DEPLOYMENT_MODE == 'cloud' else {}
            ),
            html.Div(
                html.Button('⬇ Export .parquet',
                            id='export-parquet-btn',
                            className='btn btn-sm btn-outline-secondary mt-2 w-100'),
                style={'display': 'none'} if DEPLOYMENT_MODE == 'cloud' else {}
            ),
        ]),
    ], className='mb-2')

    # X-axis toggle for Figure 2
    xaxis_card = dbc.Card([
        dbc.CardHeader('Fig 2 X-Axis'),
        dbc.CardBody([
            dcc.RadioItems(
                id='xaxis-toggle',
                options=[
                    {'label': '  Lap Distance (m)', 'value': 'dist'},
                    {'label': '  Session Time (s)',  'value': 'time'},
                ],
                value='dist',
                inputStyle={'margin-right': '6px'},
                labelStyle={'display': 'block', 'margin-bottom': '4px'},
            ),
        ]),
    ], className='mb-2')

    # Lap selector checklist (options populated by callback after file load)
    laps_card = dbc.Card([
        dbc.CardHeader('Select Laps'),
        dbc.CardBody([
            # "Select All" master toggle
            dcc.Checklist(
                id='select-all-checkbox',
                options=[{'label': '  Select All', 'value': 'all'}],
                value=['all'],
                inputStyle={'margin-right': '6px'},
            ),
            html.Hr(style={'margin': '6px 0'}),
            # Per-lap checkboxes — populated by callback
            dcc.Checklist(
                id='lap-checklist',
                options=[],      # filled in after upload
                value=[],
                inputStyle={'margin-right': '6px'},
                labelStyle={'display': 'block', 'margin-bottom': '4px'},
            ),
        ]),
    ], className='mb-2')

    return html.Div([
        mode_selector,
        upload_card,
        xaxis_card,
        laps_card,
    ], style={'width': '240px', 'minWidth': '240px', 'padding': '8px'})


def _empty_figure(message='No data — upload a session file'):
    """Return a minimal placeholder Plotly figure with a centred message."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        font_color='#cccccc',
        annotations=[dict(
            text=message, showarrow=False,
            xref='paper', yref='paper', x=0.5, y=0.5,
            font=dict(size=14, color='#777'),
        )],
        margin=dict(l=40, r=20, t=30, b=30),
    )
    return fig


def _dark_layout(**kwargs):
    """Return common dark-theme layout kwargs for all figures."""
    base = dict(
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#111122',
        font=dict(color='#cccccc', size=11),
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#444'),
        xaxis=dict(gridcolor='#333', zerolinecolor='#555'),
        yaxis=dict(gridcolor='#333', zerolinecolor='#555'),
    )
    base.update(kwargs)
    return base


# =============================================================================
# APP LAYOUT
# =============================================================================

app.layout = dbc.Container([

    # ---- Header ----
    dbc.Row([
        dbc.Col([
            html.H2('aiDAQ', style={'color': '#7ec8e3', 'margin': '8px 0 4px'}),
            html.Small('Sim Racing Telemetry Analysis',
                       style={'color': '#888', 'marginBottom': '8px', 'display': 'block'}),
        ], width=12),
    ]),

    # ---- Main body: sidebar + figures ----
    dbc.Row([

        # Sidebar
        dbc.Col(_sidebar(), width='auto', style={'padding': '0'}),

        # Figures
        dbc.Col([

            # Loading spinner wraps all figures so the user sees feedback
            # while a large .ibt file is being parsed.
            dcc.Loading(id='figures-loading', type='circle', children=[

                # Fig 1 — Track Map
                dbc.Card([
                    dbc.CardHeader('Fig 1 — Track Map'),
                    dbc.CardBody([
                        dcc.Graph(id='fig-track',
                                  figure=_empty_figure(),
                                  config={'scrollZoom': True,
                                          'displayModeBar': True},
                                  style={'height': '380px'}),
                    ], style={'padding': '4px'}),
                ], className='mb-2'),

                # Fig 2 — Throttle & Brake
                dbc.Card([
                    dbc.CardHeader('Fig 2 — Throttle & Brake vs Lap Distance'),
                    dbc.CardBody([
                        dcc.Graph(id='fig-waveform',
                                  figure=_empty_figure(),
                                  config={'scrollZoom': True,
                                          'displayModeBar': True},
                                  style={'height': '280px'}),
                    ], style={'padding': '4px'}),
                ], className='mb-2'),

                # Fig 3A + 3B side-by-side
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('Fig 3A — Traction Circle'),
                            dbc.CardBody([
                                dcc.Graph(id='fig-traction',
                                          figure=_empty_figure(),
                                          config={'displayModeBar': True},
                                          style={'height': '320px'}),
                            ], style={'padding': '4px'}),
                        ]),
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('Fig 3B — Balance Validator'),
                            dbc.CardBody([
                                dcc.Graph(id='fig-balance',
                                          figure=_empty_figure(),
                                          config={'scrollZoom': True,
                                                  'displayModeBar': True},
                                          style={'height': '320px'}),
                            ], style={'padding': '4px'}),
                        ]),
                    ], width=6),
                ], className='mb-2'),

            ]),  # end Loading

            # ---- Scrubber + Play controls ----
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dcc.Slider(
                                id='scrubber',
                                min=0, max=5000, step=10, value=0,
                                marks={},     # populated by callback
                                tooltip={'placement': 'bottom',
                                         'always_visible': False},
                                disabled=True,
                            ),
                        ], width=10),
                        dbc.Col([
                            html.Button('▶ Play', id='play-button',
                                        className='btn btn-sm btn-outline-info w-100',
                                        disabled=True),
                        ], width=2),
                    ], align='center'),
                ], style={'padding': '8px'}),
            ]),

        ], style={'flex': '1', 'minWidth': '0', 'padding': '0 8px'}),

    ], style={'display': 'flex', 'flexWrap': 'nowrap'}),

    # ---- Hidden state stores (client-side, no server memory cost) ----
    dcc.Store(id='session-data',     storage_type='memory'),  # serialised DataFrame
    dcc.Store(id='selected-laps',    storage_type='memory', data=[]),
    dcc.Store(id='cursor-position',  storage_type='memory', data={'lap_dist': 0.0}),
    dcc.Store(id='play-state',       storage_type='memory', data={'playing': False}),

    # ---- Interval timers ----
    # Live telemetry: polls DataSource at LIVE_INTERVAL_MS
    dcc.Interval(id='live-interval',
                 interval=LIVE_INTERVAL_MS,
                 disabled=True,       # enabled by mode-selector callback
                 n_intervals=0),

    # Play scrubber advance
    dcc.Interval(id='play-interval',
                 interval=PLAY_INTERVAL_MS,
                 disabled=True,       # enabled by play-button callback
                 n_intervals=0),

], fluid=True, style={'minHeight': '100vh', 'backgroundColor': '#0d0d1a'})


# =============================================================================
# CALLBACKS
# =============================================================================
# Callbacks are listed in dependency order (earlier callbacks produce outputs
# that later callbacks consume).
# Each callback starts with a comment explaining WHAT triggers it and WHY.

# ---------------------------------------------------------------------------
# CB-1: File upload → parse → store session data
# ---------------------------------------------------------------------------
# Triggered when the user drops or selects a file in the dcc.Upload zone.
# Parses .ibt (local only) or .parquet (local + cloud) and stores the
# resulting DataFrame as JSON in the 'session-data' store.

@app.callback(
    Output('session-data',   'data'),
    Output('upload-status',  'children'),
    Input('file-upload',     'contents'),
    State('file-upload',     'filename'),
    prevent_initial_call=True,
)
def cb_upload_file(contents, filename):
    """Parse uploaded file and populate session-data store."""
    if contents is None or filename is None:
        return dash.no_update, 'No file received.'

    fname_lower = filename.lower()

    try:
        if fname_lower.endswith('.ibt'):
            df = _ds.load_ibt_b64(contents)
        elif fname_lower.endswith('.parquet'):
            df = _ds.load_parquet_b64(contents)
        else:
            return None, f'Unsupported file type: {filename}'

        if df is None:
            return None, f'Failed to parse {filename}.'

        n_laps   = len(DataSource.get_lap_list(df))
        n_frames = len(df)
        status   = f'✓ {filename}  |  {n_frames:,} frames  |  {n_laps} laps'
        return _ds.df_to_json(df), status

    except Exception as e:
        return None, f'Error: {str(e)}'


# ---------------------------------------------------------------------------
# CB-2: Session data loaded → update lap checklist + scrubber range
# ---------------------------------------------------------------------------
# Fired whenever new session data arrives (after file upload).
# Builds the per-lap checkbox options and sets the scrubber's range.

@app.callback(
    Output('lap-checklist',      'options'),
    Output('lap-checklist',      'value'),
    Output('select-all-checkbox','value'),
    Output('scrubber',           'max'),
    Output('scrubber',           'marks'),
    Output('scrubber',           'disabled'),
    Output('play-button',        'disabled'),
    Input('session-data',        'data'),
    prevent_initial_call=True,
)
def cb_session_loaded(session_json):
    """Populate lap selector and enable scrubber when session data arrives."""
    if not session_json:
        return [], [], [], 5000, {}, True, True

    df = DataSource.df_from_json(session_json)
    if df is None:
        return [], [], [], 5000, {}, True, True

    laps = DataSource.get_lap_list(df)
    options = [{'label': f'  Lap {lap}', 'value': lap} for lap in laps]
    values  = laps   # all laps selected by default

    # Scrubber range: 0 → max LapDist_m across all laps
    if 'LapDist_m' in df.columns:
        max_dist = float(df['LapDist_m'].max())
    elif 'LapDistPct' in df.columns:
        max_dist = float(df['LapDistPct'].max()) * 3000.0
    else:
        max_dist = 5000.0

    # Marks at every 500 m
    step_m = 500
    marks = {int(m): f'{m}m' for m in range(0, int(max_dist) + step_m, step_m)}

    return options, values, ['all'], int(max_dist), marks, False, False


# ---------------------------------------------------------------------------
# CB-3: Lap checklist ↔ "Select All" synchronisation
# ---------------------------------------------------------------------------
# "Select All" unchecks/rechecks all laps; per-lap changes update the master.

@app.callback(
    Output('lap-checklist',       'value',  allow_duplicate=True),
    Output('select-all-checkbox', 'value',  allow_duplicate=True),
    Output('selected-laps',       'data'),
    Input('lap-checklist',        'value'),
    Input('select-all-checkbox',  'value'),
    State('lap-checklist',        'options'),
    prevent_initial_call=True,
)
def cb_lap_selection(lap_values, all_value, lap_options):
    """Keep lap checklist and Select-All in sync; write to selected-laps store."""
    triggered = ctx.triggered_id
    all_lap_values = [o['value'] for o in lap_options]

    if triggered == 'select-all-checkbox':
        if 'all' in (all_value or []):
            # Checked → select every lap
            return all_lap_values, ['all'], all_lap_values
        else:
            # Unchecked → deselect all
            return [], [], []

    elif triggered == 'lap-checklist':
        # If user manually selects all laps, tick the master too
        master = ['all'] if set(lap_values) == set(all_lap_values) else []
        return dash.no_update, master, lap_values

    return dash.no_update, dash.no_update, lap_values or []


# ---------------------------------------------------------------------------
# CB-4: Cursor position — unified writer
# ---------------------------------------------------------------------------
# ONE callback owns writing to cursor-position (Dash rule: one output = one
# writer).  Multiple inputs (figure clicks, scrubber, play interval) can
# all trigger it.  ctx.triggered_id identifies which one fired.

@app.callback(
    Output('cursor-position', 'data'),
    Input('fig-track',        'clickData'),
    Input('fig-waveform',     'clickData'),
    Input('fig-balance',      'clickData'),
    Input('scrubber',         'value'),
    Input('play-interval',    'n_intervals'),
    State('play-state',       'data'),
    State('cursor-position',  'data'),
    State('session-data',     'data'),
    prevent_initial_call=True,
)
def cb_update_cursor(track_click, wave_click, balance_click,
                     scrubber_val, play_n,
                     play_state, current_cursor, session_json):
    """Update the shared cursor position from any interaction source."""
    triggered = ctx.triggered_id

    # ---- Play interval advance ----
    if triggered == 'play-interval':
        if not (play_state or {}).get('playing', False):
            return dash.no_update
        # Advance by PLAY_STEP_M metres
        current = (current_cursor or {}).get('lap_dist', 0.0)
        # Wrap around at max distance
        if session_json:
            df = DataSource.df_from_json(session_json)
            max_d = float(df['LapDist_m'].max()) if (df is not None and 'LapDist_m' in df.columns) else 5000.0
        else:
            max_d = 5000.0
        new_dist = (current + PLAY_STEP_M) % max_d
        return {'lap_dist': new_dist}

    # ---- Scrubber drag ----
    if triggered == 'scrubber':
        return {'lap_dist': float(scrubber_val or 0)}

    # ---- Figure click: Fig 1 (track map) ----
    if triggered == 'fig-track' and track_click:
        # Track map stores LapDist_m in customdata[0]
        try:
            pt = track_click['points'][0]
            lap_dist = float(pt.get('customdata', [0])[0])
            return {'lap_dist': lap_dist}
        except Exception:
            pass

    # ---- Figure click: Fig 2 (waveform) ----
    if triggered == 'fig-waveform' and wave_click:
        # X value IS the lap distance (or time, but we store dist too)
        try:
            pt = wave_click['points'][0]
            lap_dist = float(pt.get('x', 0))
            return {'lap_dist': lap_dist}
        except Exception:
            pass

    # ---- Figure click: Fig 3B (balance waveform) ----
    if triggered == 'fig-balance' and balance_click:
        try:
            pt = balance_click['points'][0]
            lap_dist = float(pt.get('x', 0))
            return {'lap_dist': lap_dist}
        except Exception:
            pass

    return dash.no_update


# ---------------------------------------------------------------------------
# CB-5: Play button toggle
# ---------------------------------------------------------------------------

@app.callback(
    Output('play-state',    'data'),
    Output('play-button',   'children'),
    Output('play-interval', 'disabled'),
    Input('play-button',    'n_clicks'),
    State('play-state',     'data'),
    prevent_initial_call=True,
)
def cb_play_toggle(n_clicks, play_state):
    """Toggle play/pause state and enable/disable the play interval."""
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update
    is_playing = (play_state or {}).get('playing', False)
    new_playing = not is_playing
    label    = '⏸ Pause' if new_playing else '▶ Play'
    disabled = not new_playing
    return {'playing': new_playing}, label, disabled


# ---------------------------------------------------------------------------
# CB-6: Figure 1 — Track Map
# ---------------------------------------------------------------------------
# Rebuilds the track map when selected laps or the cursor changes.
# One line trace per lap + one marker dot at the cursor position.

@app.callback(
    Output('fig-track', 'figure'),
    Input('selected-laps',   'data'),
    Input('cursor-position', 'data'),
    State('session-data',    'data'),
    prevent_initial_call=True,
)
def cb_fig_track(selected_laps, cursor, session_json):
    """Render the track map: one trace per selected lap + cursor marker."""
    if not session_json:
        return _empty_figure('Upload a session file to see the track map.')

    df = DataSource.df_from_json(session_json)
    if df is None:
        return _empty_figure('Failed to read session data.')

    if not selected_laps:
        return _empty_figure('Select at least one lap.')

    # Guard: PosX/PosY are only present after file-mode parsing or once live
    # dead-reckoning has produced at least one frame with position data.
    if 'PosX' not in df.columns or 'PosY' not in df.columns:
        return _empty_figure(
            'Track position data (PosX / PosY) not available yet.\n'
            'Upload a session file for an instant map, or wait a moment '
            'for live position to initialise.'
        )

    # Palette for lap traces (cycles if more laps than colours)
    COLORS = ['#4cc9f0', '#f72585', '#7209b7', '#3a0ca3',
              '#4361ee', '#4cc9f0', '#560bad', '#480ca8']

    cursor_dist = (cursor or {}).get('lap_dist', 0.0)
    traces = []

    for i, lap_num in enumerate(sorted(selected_laps)):
        lap_df = df.filter(pl.col('Lap').cast(pl.Int32) == lap_num)
        if len(lap_df) == 0:
            continue

        x_raw = lap_df['PosX'].to_numpy()
        y_raw = lap_df['PosY'].to_numpy()
        d_raw = lap_df['LapDist_m'].to_numpy() if 'LapDist_m' in lap_df.columns else np.arange(len(lap_df))

        # Downsample for Plotly performance
        idx = phys.downsample_indices(len(x_raw), MAX_PLOTLY_POINTS)
        x_ds = x_raw[idx]
        y_ds = y_raw[idx]
        d_ds = d_raw[idx]

        color = COLORS[i % len(COLORS)]

        traces.append(go.Scatter(
            x=x_ds,
            y=y_ds,
            mode='lines',
            name=f'Lap {lap_num}',
            line=dict(color=color, width=1.5),
            # customdata carries LapDist_m so CB-4 can read it on click
            customdata=np.column_stack([d_ds]),
            hovertemplate=(
                f'Lap {lap_num}<br>'
                'X: %{x:.1f} m<br>'
                'Y: %{y:.1f} m<br>'
                'Dist: %{customdata[0]:.0f} m<extra></extra>'
            ),
        ))

    # ---- Cursor marker ----
    # Find the row closest to cursor_dist in the first selected lap
    if selected_laps:
        ref_lap = sorted(selected_laps)[0]
        lap_df  = df.filter(pl.col('Lap').cast(pl.Int32) == ref_lap)
        if len(lap_df) > 0 and 'LapDist_m' in lap_df.columns:
            d_arr = lap_df['LapDist_m'].to_numpy()
            idx_c = int(np.argmin(np.abs(d_arr - cursor_dist)))
            cx = float(lap_df['PosX'][idx_c])
            cy = float(lap_df['PosY'][idx_c])
            traces.append(go.Scatter(
                x=[cx], y=[cy],
                mode='markers',
                name='Cursor',
                marker=dict(color='#ffffff', size=10, symbol='circle',
                            line=dict(color='#ff3333', width=2)),
                showlegend=False,
                hovertemplate=(
                    f'Cursor<br>Dist: {cursor_dist:.0f} m<extra></extra>'
                ),
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        **_dark_layout(
            title=dict(text='Track Map', font=dict(size=13)),
            xaxis=dict(title='X (m)', scaleanchor='y', scaleratio=1,
                       gridcolor='#333', zerolinecolor='#555'),
            yaxis=dict(title='Y (m)', gridcolor='#333', zerolinecolor='#555'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='left', x=0),
            hovermode='closest',
        )
    )
    return fig


# ---------------------------------------------------------------------------
# CB-7: Figure 2 — Throttle & Brake Waveforms
# ---------------------------------------------------------------------------
# One throttle trace + one brake trace per selected lap.
# Brake is plotted as negative so it mirrors throttle.
# Vertical cursor line shows the current position.

@app.callback(
    Output('fig-waveform',   'figure'),
    Input('selected-laps',   'data'),
    Input('cursor-position', 'data'),
    Input('xaxis-toggle',    'value'),
    State('session-data',    'data'),
    prevent_initial_call=True,
)
def cb_fig_waveform(selected_laps, cursor, xaxis_mode, session_json):
    """Render throttle & brake waveforms vs lap distance (or time)."""
    if not session_json:
        return _empty_figure('Upload a session file to see waveforms.')

    df = DataSource.df_from_json(session_json)
    if df is None or not selected_laps:
        return _empty_figure('Select at least one lap.')

    COLORS = ['#4cc9f0', '#f72585', '#7209b7', '#3a0ca3',
              '#4361ee', '#56cfe1', '#560bad', '#480ca8']

    cursor_dist = (cursor or {}).get('lap_dist', 0.0)
    use_time    = (xaxis_mode == 'time')
    x_label     = 'Session Time (s)' if use_time else 'Lap Distance (m)'
    traces      = []

    for i, lap_num in enumerate(sorted(selected_laps)):
        lap_df = df.filter(pl.col('Lap').cast(pl.Int32) == lap_num)
        if len(lap_df) == 0:
            continue

        # X axis
        if use_time and 'SessionTime' in lap_df.columns:
            x_raw = lap_df['SessionTime'].to_numpy()
        elif 'LapDist_m' in lap_df.columns:
            x_raw = lap_df['LapDist_m'].to_numpy()
        else:
            x_raw = lap_df['LapDistPct'].to_numpy() * 100.0

        # Throttle & brake
        if 'Throttle_pct' in lap_df.columns:
            throttle = lap_df['Throttle_pct'].to_numpy()
            brake    = lap_df['Brake_pct'].to_numpy() * -1.0   # negative
        elif 'Throttle' in lap_df.columns:
            throttle = lap_df['Throttle'].to_numpy() * 100.0
            brake    = lap_df['Brake'].to_numpy()    * -100.0
        else:
            continue

        # Downsample
        idx = phys.downsample_indices(len(x_raw), MAX_PLOTLY_POINTS)
        x_ds = x_raw[idx]
        t_ds = throttle[idx]
        b_ds = brake[idx]
        color = COLORS[i % len(COLORS)]

        # Throttle trace (positive)
        traces.append(go.Scatter(
            x=x_ds, y=t_ds,
            mode='lines',
            name=f'Lap {lap_num} Throttle',
            line=dict(color=color, width=1.2),
            hovertemplate=f'Lap {lap_num} Throttle<br>%{{x:.0f}} m : %{{y:.1f}}%<extra></extra>',
        ))

        # Brake trace (negative — mirrors throttle)
        # Use a slightly different shade so traces are distinguishable
        traces.append(go.Scatter(
            x=x_ds, y=b_ds,
            mode='lines',
            name=f'Lap {lap_num} Brake',
            line=dict(color=color, width=1.2, dash='dot'),
            hovertemplate=f'Lap {lap_num} Brake<br>%{{x:.0f}} m : %{{y:.1f}}%<extra></extra>',
        ))

    # ---- Vertical cursor line ----
    if not use_time:
        traces.append(go.Scatter(
            x=[cursor_dist, cursor_dist],
            y=[-105, 105],
            mode='lines',
            line=dict(color='#ffffff', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        **_dark_layout(
            title=dict(text='Throttle & Brake', font=dict(size=13)),
            xaxis=dict(title=x_label, gridcolor='#333', zerolinecolor='#555'),
            yaxis=dict(title='%', range=[-110, 110],
                       gridcolor='#333', zeroline=True, zerolinecolor='#555'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            hovermode='x unified',
        )
    )
    return fig


# ---------------------------------------------------------------------------
# CB-8: Figure 3A — Traction Circle
# ---------------------------------------------------------------------------
# Shows a ±TC_WINDOW_M window of data around the cursor position.
# Points are colour-coded by TC value (green = at limit, red = far from limit).
# The unit circle at TC=1 is drawn as the reference grip limit.

@app.callback(
    Output('fig-traction',   'figure'),
    Input('cursor-position', 'data'),
    Input('selected-laps',   'data'),
    State('session-data',    'data'),
    prevent_initial_call=True,
)
def cb_fig_traction(cursor, selected_laps, session_json):
    """Render the traction circle for the corner near the cursor position."""
    if not session_json:
        return _empty_figure('Upload a session file to see the traction circle.')

    df = DataSource.df_from_json(session_json)
    if df is None or not selected_laps:
        return _empty_figure('Select at least one lap.')

    # Check that LatAccel / LongAccel columns exist
    has_accel = ('LatAccel' in df.columns and 'LongAccel' in df.columns)
    if not has_accel:
        return _empty_figure('LatAccel / LongAccel channels not available in this session.')

    cursor_dist = (cursor or {}).get('lap_dist', 0.0)
    lo = cursor_dist - TC_WINDOW_M
    hi = cursor_dist + TC_WINDOW_M

    # --- Reference circle at TC = 1 ---
    theta = np.linspace(0, 2 * math.pi, 200)
    traces = [go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode='lines',
        name='Grip limit (TC=1)',
        line=dict(color='#445566', width=1.5, dash='dot'),
        showlegend=True,
        hoverinfo='skip',
    )]

    MARKER_SIZES = [4, 5, 6]

    for i, lap_num in enumerate(sorted(selected_laps)):
        lap_df = df.filter(pl.col('Lap').cast(pl.Int32) == lap_num)
        if len(lap_df) == 0:
            continue
        if 'LapDist_m' not in lap_df.columns:
            continue

        # Window: only data within ±TC_WINDOW_M of the cursor
        win_df = lap_df.filter(
            (pl.col('LapDist_m') >= lo) & (pl.col('LapDist_m') <= hi)
        )
        if len(win_df) == 0:
            continue

        # Get the ax_max / ay_max used when the file was parsed (consistent normalisation)
        if 'ax_max_used' in win_df.columns:
            ax_max = float(win_df['ax_max_used'][0])
            ay_max = float(win_df['ay_max_used'][0])
        else:
            ax_max = AX_MAX_DEFAULT
            ay_max = AY_MAX_DEFAULT

        lat_acc = win_df['LatAccel'].to_numpy()
        lon_acc = win_df['LongAccel'].to_numpy()

        x_norm = lat_acc / (ay_max + EPSILON)   # X = normalised lateral
        y_norm = lon_acc / (ax_max + EPSILON)   # Y = normalised longitudinal

        # Use pre-computed TC colours if available, else compute now
        if 'TC_color' in win_df.columns:
            colors = win_df['TC_color'].to_list()
        elif 'TC' in win_df.columns:
            colors = phys.color_from_tc_array(win_df['TC'].to_numpy())
        else:
            tc_arr = phys.compute_traction_circle_array(lon_acc, lat_acc, ax_max, ay_max)
            colors = phys.color_from_tc_array(tc_arr)

        lap_dist = win_df['LapDist_m'].to_numpy()

        traces.append(go.Scatter(
            x=x_norm, y=y_norm,
            mode='markers',
            name=f'Lap {lap_num}',
            marker=dict(
                color=colors,
                size=MARKER_SIZES[i % len(MARKER_SIZES)],
                opacity=0.85,
            ),
            customdata=np.column_stack([lap_dist]),
            hovertemplate=(
                f'Lap {lap_num}<br>'
                'Lat: %{x:.3f} g<br>'
                'Lon: %{y:.3f} g<br>'
                'Dist: %{customdata[0]:.0f} m<extra></extra>'
            ),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        **_dark_layout(
            title=dict(
                text=f'Traction Circle  |  Cursor ±{TC_WINDOW_M}m  '
                     f'({cursor_dist:.0f}m)',
                font=dict(size=12),
            ),
            xaxis=dict(
                title='Lat Accel (normalised)',
                range=[-1.5, 1.5],
                scaleanchor='y', scaleratio=1,
                gridcolor='#333', zerolinecolor='#555',
            ),
            yaxis=dict(
                title='Lon Accel (normalised)',
                range=[-1.5, 1.5],
                gridcolor='#333', zerolinecolor='#555',
            ),
            hovermode='closest',
        )
    )

    # Colour legend annotation
    fig.add_annotation(
        text='🟢 at limit  🟡 near  🔴 far',
        xref='paper', yref='paper',
        x=0.01, y=0.01,
        showarrow=False,
        font=dict(size=10, color='#aaa'),
        align='left',
    )
    return fig


# ---------------------------------------------------------------------------
# CB-9: Figure 3B — Car Balance / Steering Responsiveness
# ---------------------------------------------------------------------------
# X = LapDist (m),  Y = Balance% (−100 to +100)
# Colour = USR_TC (steering responsiveness), NOT the direction of balance.
#   Green = steering is efficient
#   Yellow = becoming unresponsive
#   Red = wasteful steering input

@app.callback(
    Output('fig-balance',    'figure'),
    Input('selected-laps',   'data'),
    Input('cursor-position', 'data'),
    State('session-data',    'data'),
    prevent_initial_call=True,
)
def cb_fig_balance(selected_laps, cursor, session_json):
    """Render balance validator: balance% vs distance, coloured by steering responsiveness."""
    if not session_json:
        return _empty_figure('Upload a session file to see the balance validator.')

    df = DataSource.df_from_json(session_json)
    if df is None or not selected_laps:
        return _empty_figure('Select at least one lap.')

    if 'Balance_pct' not in df.columns:
        return _empty_figure('Balance_pct column not computed. Check LatAccel / YawRate availability.')

    cursor_dist = (cursor or {}).get('lap_dist', 0.0)
    traces = []

    for lap_num in sorted(selected_laps):
        lap_df = df.filter(pl.col('Lap').cast(pl.Int32) == lap_num)
        if len(lap_df) == 0:
            continue

        x_raw = lap_df['LapDist_m'].to_numpy() if 'LapDist_m' in lap_df.columns else np.arange(len(lap_df))
        y_raw = lap_df['Balance_pct'].to_numpy()

        # Per-point colour from USR_TC (steering responsiveness)
        if 'USR_color' in lap_df.columns:
            colors = lap_df['USR_color'].to_list()
        elif 'USR_TC' in lap_df.columns:
            colors = phys.color_from_usr_array(lap_df['USR_TC'].to_numpy())
        else:
            colors = ['#aaaaaa'] * len(x_raw)

        # Downsample for Plotly performance
        idx = phys.downsample_indices(len(x_raw), MAX_PLOTLY_POINTS)
        x_ds = x_raw[idx]
        y_ds = y_raw[idx]
        c_ds = [colors[ii] for ii in idx]

        traces.append(go.Scatter(
            x=x_ds, y=y_ds,
            mode='markers',
            name=f'Lap {lap_num}',
            marker=dict(color=c_ds, size=3, opacity=0.9),
            hovertemplate=(
                f'Lap {lap_num}<br>'
                'Dist: %{x:.0f} m<br>'
                'Balance: %{y:.1f}%<extra></extra>'
            ),
        ))

    # ---- Zero line (neutral balance) ----
    traces.append(go.Scatter(
        x=[0, df['LapDist_m'].max() if 'LapDist_m' in df.columns else 5000],
        y=[0, 0],
        mode='lines',
        line=dict(color='#555', width=1, dash='dash'),
        showlegend=False,
        hoverinfo='skip',
    ))

    # ---- Vertical cursor line ----
    traces.append(go.Scatter(
        x=[cursor_dist, cursor_dist],
        y=[-105, 105],
        mode='lines',
        line=dict(color='#ffffff', width=1, dash='dash'),
        showlegend=False,
        hoverinfo='skip',
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        **_dark_layout(
            title=dict(text='Balance Validator (colour = steering responsiveness)',
                       font=dict(size=12)),
            xaxis=dict(title='Lap Distance (m)', gridcolor='#333', zerolinecolor='#555'),
            yaxis=dict(
                title='Balance %',
                range=[-110, 110],
                gridcolor='#333',
                zerolinecolor='#888',
                tickvals=[-100, -50, 0, 50, 100],
                ticktext=['-100 (US)', '-50', '0 Neutral', '+50', '+100 (OS)'],
            ),
            hovermode='x unified',
        )
    )

    # Colour legend
    fig.add_annotation(
        text='🟢 responsive steering  🟡 moderate waste  🔴 unresponsive',
        xref='paper', yref='paper',
        x=0.01, y=0.01,
        showarrow=False,
        font=dict(size=10, color='#aaa'),
        align='left',
    )
    return fig


# ---------------------------------------------------------------------------
# CB-10: Live mode control (local only)
# ---------------------------------------------------------------------------
# When the user switches to 'live' mode, start the irsdk poller.
# When they switch away, stop it.

@app.callback(
    Output('live-interval', 'disabled'),
    Input('mode-selector',  'value'),
    prevent_initial_call=True,
)
def cb_mode_selector(mode):
    """Start/stop live iRacing poller based on mode selection."""
    if mode == 'live':
        try:
            if _ds.mode() != 'live':
                _ds.set_live()
            return False   # enable live-interval
        except RuntimeError as e:
            print(f'Cannot start live mode: {e}', flush=True)
            return True
    else:
        # Stop the live poller if running
        if _ds.mode() == 'live':
            _ds.stop_live()
        return True   # disable live-interval


# ---------------------------------------------------------------------------
# CB-11: Live telemetry — append frame to session-data store
# ---------------------------------------------------------------------------
# Runs every LIVE_INTERVAL_MS when in live mode.
# Reads the latest frame from DataSource, appends it to the in-memory
# session DataFrame, then re-serialises to JSON for the store.
# NOTE: This fires at 10 Hz — figures update at the same rate.

@app.callback(
    Output('session-data',  'data',   allow_duplicate=True),
    Input('live-interval',  'n_intervals'),
    State('session-data',   'data'),
    prevent_initial_call=True,
)
def cb_live_update(n, session_json):
    """Append the latest live frame to the session store."""
    frame = _ds.get_live_frame()
    if frame is None:
        return dash.no_update

    # Build single-row DataFrame from the frame dict
    new_row = pl.DataFrame([frame])
    # Compute derived columns for this single row
    new_row = _ds._add_derived_columns(new_row)

    if session_json:
        existing = DataSource.df_from_json(session_json)
        if existing is not None:
            # Append new row (keep last N rows to avoid unlimited growth)
            combined = pl.concat([existing, new_row], how='diagonal_relaxed')
            MAX_LIVE_ROWS = 100_000
            if len(combined) > MAX_LIVE_ROWS:
                combined = combined.tail(MAX_LIVE_ROWS)
            return _ds.df_to_json(combined)

    return _ds.df_to_json(new_row)


# ---------------------------------------------------------------------------
# CB-12: Export .parquet download (local mode only)
# ---------------------------------------------------------------------------

@app.callback(
    Output('download-parquet', 'data'),
    Input('export-parquet-btn','n_clicks'),
    prevent_initial_call=True,
)
def cb_export_parquet(n_clicks):
    """Serialise session to .parquet and trigger browser download."""
    parquet_bytes = _ds.get_session_parquet()
    if parquet_bytes is None:
        return dash.no_update
    return dcc.send_bytes(parquet_bytes, 'aidaq_session.parquet')


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # debug=True enables hot-reload (auto restarts on file save) and the
    # Dash DevTools panel in the browser.  Set to False for production.
    is_cloud = (DEPLOYMENT_MODE == 'cloud')

    print(f'Starting aiDAQ in [{DEPLOYMENT_MODE}] mode…')
    print(f'Open your browser at http://127.0.0.1:8050')

    app.run(
        debug=not is_cloud,    # DevTools off in production
        host='0.0.0.0',        # '0.0.0.0' lets Railway expose the port
        port=int(os.environ.get('PORT', 8050)),  # Railway injects PORT env var
    )
