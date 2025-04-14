import logging
from collections import deque
import config # Assuming config.py is in the same directory

# --- Custom Logging Filter ---
class TbfIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'tbf_id'):
            record.tbf_id = 'System' # Default value if tbf_id missing
        return True

# --- Logging Setup ---
# Configure logger here so it's shared
log_formatter = logging.Formatter('%(asctime)s - TBF %(tbf_id)s - %(levelname)s - %(message)s')
log_file_handler = logging.FileHandler(config.LOG_FILE)
log_file_handler.setFormatter(log_formatter)

# Add the custom filter to the handler
tbf_filter = TbfIdFilter()
log_file_handler.addFilter(tbf_filter)

logger = logging.getLogger("TBF_App_Logger") # Give it a specific name maybe
# Prevent duplicate handlers if this module is somehow imported multiple times
if not logger.hasHandlers():
    logger.addHandler(log_file_handler)
    logger.setLevel(logging.INFO)

# --- Global State Dictionary ---
# Initialize with default values
GLOBAL_STATE = {
    "models_loaded": False,
    "rul_model": None,
    "fault_model": None,
    "anomaly_model": None,
    "anomaly_scaler": None,
    # Initialize tbf_state_objects later in dash_app after TBFState class is known
    "tbf_state_objects": [],
    "plot_time_steps": deque(maxlen=config.PLOT_HISTORY_LENGTH),
    "plot_rul_values": [], # Initialize later based on NUM_TBFs
    "time_step_counter": 0,
    "monitoring_running": False,
    "status_message": "Initializing...",
    "status_level": "info",
    "initial_load_error": None,
    'resources_loaded_flag': False # Flag to prevent reloading resources
}
