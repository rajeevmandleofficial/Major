import pandas as pd
import random
import streamlit as st
import config # Import settings from config.py
from simulation import TBFState # Import the TBFState class

def generate_historical_data_csv(file_path=config.HISTORICAL_DATA_FILE, num_rows=config.NUM_HISTORICAL_ROWS):
    """Generates historical sensor data with different fault modes and saves to CSV."""
    # This function now uses the TBFState class from simulation.py
    st.info(f"Generating {num_rows} rows of historical data for training...")
    temp_tbf = TBFState(tbf_id=0) # Use the class
    temp_tbf.simulation_mode = 'Random' # Ensure it progresses through faults

    all_data = []
    progress_bar = st.progress(0, text="Generating historical data...")
    for i in range(num_rows):
        simulated_data = temp_tbf.advance_simulation() # Call method on the object
        true_fault_mode = temp_tbf.current_fault_mode # Access attribute

        row_data = simulated_data.copy()
        row_data['FaultMode'] = true_fault_mode

        # Calculate pseudo-RUL based on fault mode and time in fault
        rul = 100.0
        if true_fault_mode == "Bearing Wear":
            rul = max(5, 80 - temp_tbf.time_in_fault * 1.2)
        elif true_fault_mode == "Imbalance":
            rul = max(5, 90 - temp_tbf.time_in_fault * 0.8)
        elif true_fault_mode == "Blade Damage":
            rul = max(5, 85 - temp_tbf.time_in_fault * 1.0)
        else: # Normal
            rul = max(50, 100 - temp_tbf.runtime_steps * 0.1)
        row_data['RUL'] = max(1, rul + random.normalvariate(0, 5))

        all_data.append(row_data)
        progress_bar.progress((i + 1) / num_rows, text=f"Generating historical data... {i+1}/{num_rows}")
    progress_bar.empty() # Remove progress bar after completion

    df = pd.DataFrame(all_data)
    st.info("Calculating features for historical data...")

    # Calculate rolling features for the generated data
    window_size = config.ROLLING_WINDOW_SIZE
    features_to_calc = ['Vibration', 'Temperature', 'Power'] # Base features for rolling/diff
    for feature in features_to_calc:
        df[f'{feature}_Rolling_Mean'] = df[feature].rolling(window=window_size).mean()
        df[f'{feature}_Diff'] = df[feature].diff()

    # Ensure all features expected by models are present, even if just 0
    # This includes the base features and the engineered ones
    all_expected_features = set(config.RUL_FAULT_FEATURES) | set(config.ANOMALY_FEATURES)
    for feat in all_expected_features:
         if feat not in df.columns and feat not in ['FaultMode', 'RUL']: # Don't add target columns here
             df[feat] = 0.0 # Add missing feature columns with 0

    # Fill initial NaN values created by rolling/diff operations
    df = df.bfill() # Use bfill() directly
    df.fillna(0, inplace=True) # Fill any remaining NaNs

    df = df.round(4)

    try:
        df.to_csv(file_path, index=False)
        st.success(f"Successfully generated historical data and saved to {file_path}")
        # log_info is not defined here, logging happens in main app
        return df
    except Exception as e:
        st.error(f"Error saving historical data: {e}")
        # log_error is not defined here
        return None

#@st.cache_data # Consider caching if loading is slow
def load_and_preprocess_data(file_path=config.HISTORICAL_DATA_FILE):
    """Loads data from CSV."""
    try:
        df = pd.read_csv(file_path)
        # log_info(f"Loaded historical data from {file_path}") # Logging in main app
        return df
    except FileNotFoundError:
        # log_error(f"Historical data file not found: {file_path}") # Logging in main app
        return None
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        # log_error(f"Error reading CSV {file_path}: {e}") # Logging in main app
        return None

def calculate_features_from_history(data_history_deque):
    """
    Calculates rolling/diff features from a TBF's history buffer (deque).
    Takes the deque directly as input.
    """
    history_list = list(data_history_deque)
    if len(history_list) < 1: # Need at least one point
        return None # Return None if no history

    # Convert history to DataFrame for easier calculation
    history_df = pd.DataFrame(history_list)
    # Create a DataFrame for the latest features based on the last row of history
    latest_data = history_df.iloc[-1].to_dict()
    features_df = pd.DataFrame([latest_data])

    window_size = min(config.ROLLING_WINDOW_SIZE, len(history_df)) # Adjust window if history is short

    # Calculate Rolling Mean and Diff
    for feature in ['Vibration', 'Temperature', 'Power']:
        if feature in history_df.columns:
            # Calculate rolling mean using the relevant history slice
            features_df[f'{feature}_Rolling_Mean'] = history_df[feature].rolling(window=window_size, min_periods=1).mean().iloc[-1]
            # Calculate difference using the relevant history slice (needs at least 2 points)
            if len(history_df) >= 2:
                features_df[f'{feature}_Diff'] = history_df[feature].diff().iloc[-1]
            else:
                features_df[f'{feature}_Diff'] = 0.0 # Assign 0 if only one point
        else:
             # If base feature is missing in history (shouldn't happen ideally), set rolling/diff to 0
             features_df[f'{feature}_Rolling_Mean'] = 0.0
             features_df[f'{feature}_Diff'] = 0.0


    # Ensure all expected base features are present in the output df
    base_features = ['Speed', 'Vibration', 'Temperature', 'Gas', 'Pressure', 'Power']
    for bf in base_features:
        if bf not in features_df.columns:
            features_df[bf] = 0.0 # Add missing base feature columns with 0

    # Fill any NaNs that might have occurred (e.g., diff for the first element after history reset)
    features_df.fillna(0.0, inplace=True)

    # Ensure all columns needed for prediction are present
    all_needed_features = set(config.RUL_FAULT_FEATURES) | set(config.ANOMALY_FEATURES)
    for feat in all_needed_features:
        if feat not in features_df.columns:
            features_df[feat] = 0.0

    return features_df

