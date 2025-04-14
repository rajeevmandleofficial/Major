import dash
from dash import html, dcc, callback, Output, Input, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import logging
from collections import deque
import os

# Import project modules
import config
from simulation import TBFState
from data_processing import (generate_historical_data_csv,
                             load_and_preprocess_data,
                             calculate_features_from_history)
from ml_models import (train_or_load_model, predict_value, predict_anomaly)
# Import shared state and logger
from shared_state import GLOBAL_STATE, logger

# --- Initialize Dash App with Pages ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX],
    use_pages=True,
    suppress_callback_exceptions=True,
    title=config.PROJECT_NAME,
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}]
    )
server = app.server

# --- Initialize State Objects ---
if not GLOBAL_STATE["tbf_state_objects"]:
     GLOBAL_STATE["tbf_state_objects"] = [TBFState(i+1) for i in range(config.NUM_TBFs)]
if not GLOBAL_STATE["plot_rul_values"]:
     GLOBAL_STATE["plot_rul_values"] = [deque(maxlen=config.PLOT_HISTORY_LENGTH) for _ in range(config.NUM_TBFs)]

# --- Resource Loading Function ---
def load_initial_resources():
    state_key = 'resources_loaded_flag'
    if GLOBAL_STATE.get(state_key): return
    try:
        logger.info("Initializing Dash App Resources...", extra={'tbf_id': 'System'})
        historical_df = load_and_preprocess_data(config.HISTORICAL_DATA_FILE)
        if historical_df is None:
            logger.info("Historical data not found. Generating new data...", extra={'tbf_id': 'System'})
            historical_df = generate_historical_data_csv()
            if historical_df is None: log_critical("Fatal: Could not load/generate data.", extra={'tbf_id': 'System'}); GLOBAL_STATE["initial_load_error"] = "Fatal: Could not load/generate data."; return
        missing_hist = [f for f in config.RUL_FAULT_FEATURES if f not in historical_df.columns]
        missing_anomaly = [f for f in config.ANOMALY_FEATURES if f not in historical_df.columns]
        if missing_hist or missing_anomaly: error_msg = f"Data missing features. RUL/Fault: {missing_hist}. Anomaly: {missing_anomaly}."; log_critical(error_msg, extra={'tbf_id': 'System'}); GLOBAL_STATE["initial_load_error"] = error_msg; return
        GLOBAL_STATE["rul_model"], _ = train_or_load_model("RUL", historical_df, config.RUL_FAULT_FEATURES, 'RUL', config.RUL_MODEL_FILE, model_params={'n_estimators': 150, 'random_state': 42, 'n_jobs': -1, 'max_depth': 20})
        GLOBAL_STATE["fault_model"], _ = train_or_load_model("Fault", historical_df, config.RUL_FAULT_FEATURES, 'FaultMode', config.FAULT_MODEL_FILE, model_params={'n_estimators': 100, 'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced'}, classifier=True)
        GLOBAL_STATE["anomaly_model"], GLOBAL_STATE["anomaly_scaler"] = train_or_load_model("Anomaly", historical_df, config.ANOMALY_FEATURES, None, config.ANOMALY_MODEL_FILE, scaler_path=config.ANOMALY_SCALER_FILE, model_params={'n_estimators': 100, 'random_state': 42, 'contamination': config.ANOMALY_CONTAMINATION}, is_anomaly=True)
        if GLOBAL_STATE["rul_model"] is None: GLOBAL_STATE["initial_load_error"] = "RUL Model failed."; log_critical(GLOBAL_STATE["initial_load_error"], extra={'tbf_id': 'System'}); return
        if GLOBAL_STATE["fault_model"] is None: logger.warning("Fault Model unavailable.", extra={'tbf_id': 'System'})
        if GLOBAL_STATE["anomaly_model"] is None or GLOBAL_STATE["anomaly_scaler"] is None: logger.warning("Anomaly Detection unavailable.", extra={'tbf_id': 'System'}); GLOBAL_STATE["anomaly_model"] = None
        GLOBAL_STATE["models_loaded"] = True; GLOBAL_STATE["status_message"] = "Models loaded. Ready."; logger.info("Models loaded successfully.", extra={'tbf_id': 'System'})
        GLOBAL_STATE[state_key] = True
    except Exception as e: error_msg = f"Critical error during initialization: {e}"; logger.critical(error_msg, exc_info=True, extra={'tbf_id': 'System'}); GLOBAL_STATE["initial_load_error"] = error_msg

# --- Load resources when the script starts ---
load_initial_resources()

# --- Main App Layout ---
app.layout = dbc.Container([
    # -- Header (Common to all pages) --
    dbc.Row([
        dbc.Col(html.Img(src=app.get_asset_url('fan-icon.png') if os.path.exists('assets/fan-icon.png') else config.LANDING_IMAGE_URL, height="50px"), width="auto"),
        dbc.Col(html.H2(config.PROJECT_NAME, className="text-primary my-auto"), width=True),
        dbc.Col(dbc.Nav([ # Navigation links
            # --- MODIFIED LINES: Use hardcoded paths ---
            dbc.NavLink("Home", href="/", active="exact"),
            dbc.NavLink("Dashboard", href="/dashboard", active="exact"),
            # --- END MODIFIED LINES ---
            ], pills=True, className="justify-content-end"), width="auto")
    ], align="center", className="mb-4 mt-2 p-3 bg-body-tertiary border rounded shadow-sm"),

    # -- Page Content Area --
    # dash.page_container renders the layout from pages/landing.py or pages/dashboard.py
    dash.page_container,

    # -- Footer (Common to all pages) --
    html.Footer(dbc.Container(html.Small(["Developed by: ", html.Strong(", ".join(config.TEAM_MEMBERS)), f" - {config.AFFILIATION}"]), className="text-muted text-center py-4"), className="mt-5")

], fluid=True, className="dbc p-4")


# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True) # Use debug=True for development

