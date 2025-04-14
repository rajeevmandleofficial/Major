# Configuration constants for the TBF Monitoring App

# File Paths
HISTORICAL_DATA_FILE = "tbf_historical_data.csv"
RUL_MODEL_FILE = "tbf_rul_model.pkl"
FAULT_MODEL_FILE = "tbf_fault_model.pkl"
ANOMALY_MODEL_FILE = "tbf_anomaly_model.pkl"
ANOMALY_SCALER_FILE = "tbf_anomaly_scaler.pkl"
LOG_FILE = 'tbf_monitor_streamlit.log'

# Simulation Parameters
NUM_TBFs = 4
NUM_HISTORICAL_ROWS = 600
ROLLING_WINDOW_SIZE = 5
SIM_BASE_SPEED_MEAN = 2550
SIM_BASE_SPEED_STD = 50
SIM_BASE_TEMP_MEAN = 55
SIM_BASE_TEMP_STD = 5
SIM_BASE_VIBRATION_MEAN = 2.0
SIM_BASE_VIBRATION_STD = 0.1
SIM_VIBRATION_NOISE_STD = 0.15
SIM_TEMP_NOISE_STD = 0.6
SIM_SENSOR_DRIFT_FACTOR = 0.0001
SIM_POWER_BASE_KW = 50.0
SIM_POWER_LOAD_FACTOR = 0.05
SIM_MIN_NORMAL_RUNTIME = 100
SIM_FAULT_CHANCE_PER_STEP = 0.008

# Constraints
MAX_VIBRATION = 15.0
MAX_TEMPERATURE = 120.0
MAX_POWER = 150.0

# Anomaly Detection
ANOMALY_CONTAMINATION = 0.05

# ML Model Features
RUL_FAULT_FEATURES = [
    'Speed', 'Vibration', 'Temperature', 'Gas', 'Pressure', 'Power',
    'Vibration_Rolling_Mean', 'Vibration_Diff', 'Temperature_Rolling_Mean',
    'Temperature_Diff', 'Power_Rolling_Mean', 'Power_Diff'
]
ANOMALY_FEATURES = ['Speed', 'Vibration', 'Temperature', 'Gas', 'Pressure', 'Power']

# Failure Modes Definition
FAILURE_MODES = {
    "Normal": {"vib_factor": 1.0, "temp_factor": 1.0, "power_factor": 1.0, "vib_pattern_amp": 0.0},
    "Bearing Wear": {"vib_factor": 1.8, "temp_factor": 1.5, "power_factor": 1.2, "vib_pattern_amp": 0.1},
    "Imbalance": {"vib_factor": 2.5, "temp_factor": 1.1, "power_factor": 1.1, "vib_pattern_amp": 0.8},
    "Blade Damage": {"vib_factor": 1.5, "temp_factor": 1.0, "power_factor": 1.05, "vib_pattern_amp": 0.4},
}
FAULT_MODE_LIST = list(FAILURE_MODES.keys())
ACTUAL_FAULT_MODES = list(FAILURE_MODES.keys())[1:]
SIMULATION_MODES = ['Random', 'Good', 'Worst']

# --- UI Settings ---
UPDATE_INTERVAL_SEC = 2.0 # Compromise interval for auto-refresh
PLOT_HISTORY_LENGTH = 100
RUL_CRITICAL_THRESHOLD = 15
RUL_WARNING_THRESHOLD = 40

# --- UI Text / Project Details ---
# Using slightly shorter names and clearer descriptions
PROJECT_NAME = "TBF Condition Monitoring"
PROJECT_SUBTITLE = "AI-Powered Predictive Maintenance Dashboard"
PROJECT_DETAILS = "This AI system monitors Tunnel Booster Fans using simulated sensor data. It predicts Remaining Useful Life (RUL) and diagnoses potential faults like Bearing Wear, Imbalance, or Blade Damage, enabling proactive maintenance."
TEAM_MEMBERS = ["Rajeev Mandle", "Ayush Mark Ekka", "Janmay Jai Singh"]
AFFILIATION = "Bhilai Institute of Technology Durg"
LANDING_IMAGE_URL = "https://img.icons8.com/external-flaticons-lineal-color-flat-icons/128/external-fan-air-conditioning-flaticons-lineal-color-flat-icons-3.png"

# --- UI Colors (Approximate Material-like) ---
# These are just examples, actual rendering depends on Streamlit theme
COLOR_PRIMARY = "#1E88E5" # Blue
COLOR_SUCCESS = "#4CAF50" # Green
COLOR_WARNING = "#FB8C00" # Orange
COLOR_ERROR = "#E53935" # Red
COLOR_INFO = "#546E7A" # Blue Grey
COLOR_BACKGROUND_LIGHT = "#f5f5f5" # Light grey for card backgrounds (via markdown)
COLOR_BORDER = "#e0e0e0" # Grey border

