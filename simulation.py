import random
import numpy as np
from collections import deque
import config # Import settings from config.py

class TBFState:
    """Manages the state and simulation logic for a single TBF."""
    def __init__(self, tbf_id):
        self.id = tbf_id
        self.data_history = deque(maxlen=config.ROLLING_WINDOW_SIZE)
        self.current_data = {}
        self.predicted_rul = 100.0
        self.predicted_fault = "Initializing"
        self.is_anomaly = False
        self.status = "Normal"
        # Simulation state
        self.runtime_steps = 0
        self.current_fault_mode = "Normal"
        self.time_in_fault = 0
        self.base_speed = random.normalvariate(config.SIM_BASE_SPEED_MEAN, config.SIM_BASE_SPEED_STD)
        self.base_temp = random.normalvariate(config.SIM_BASE_TEMP_MEAN, config.SIM_BASE_TEMP_STD)
        self.base_vibration = random.normalvariate(config.SIM_BASE_VIBRATION_MEAN, config.SIM_BASE_VIBRATION_STD)
        self.temp_drift_offset = 0.0
        self.vibration_drift_offset = 0.0
        # Control
        self.simulation_mode = 'Random' # Default simulation mode
        self.worst_case_fault_type = random.choice(config.ACTUAL_FAULT_MODES)

    def _generate_base_readings(self, noise_factor=1.0):
        """Generates base readings with optional noise reduction."""
        speed = self.base_speed + random.normalvariate(0, config.SIM_BASE_SPEED_STD / 2 * noise_factor)
        current_base_temp = self.base_temp + self.temp_drift_offset
        current_base_vib = self.base_vibration + self.vibration_drift_offset
        vibration = current_base_vib + random.normalvariate(0, config.SIM_VIBRATION_NOISE_STD * noise_factor)
        temperature = current_base_temp + random.normalvariate(0, config.SIM_TEMP_NOISE_STD * noise_factor)
        gas = 0.01 + random.uniform(0, 0.01 * noise_factor)
        pressure = 101300 + random.normalvariate(0, 50 * noise_factor)
        power = config.SIM_POWER_BASE_KW + random.normalvariate(0, 0.2 * noise_factor)
        return speed, vibration, temperature, gas, pressure, power

    def advance_simulation(self):
        """
        Advances the simulation state for one time step based on self.simulation_mode.
        Returns the generated data for this step.
        """
        generated_data = {}
        fault_mode_for_data_gen = self.current_fault_mode
        sim_mode = self.simulation_mode

        if sim_mode == 'Good':
            speed, vibration, temperature, gas, pressure, power = self._generate_base_readings(noise_factor=0.1)
            fault_mode_for_data_gen = "Normal"

        elif sim_mode == 'Worst':
            fault_mode_for_data_gen = self.worst_case_fault_type
            target_vib = config.MAX_VIBRATION * random.uniform(0.9, 0.98)
            target_temp = config.MAX_TEMPERATURE * random.uniform(0.9, 0.98)
            target_power = config.MAX_POWER * random.uniform(0.9, 0.98)
            fault_params = config.FAILURE_MODES[self.worst_case_fault_type]
            if fault_params["vib_pattern_amp"] > 0:
                 vib_pattern = fault_params["vib_pattern_amp"] * 2.5 * np.sin(self.runtime_steps * 0.5 + random.uniform(0, np.pi))
                 target_vib += vib_pattern
            speed = self.base_speed + random.normalvariate(0, config.SIM_BASE_SPEED_STD * 1.2)
            vibration = target_vib + random.normalvariate(0, config.SIM_VIBRATION_NOISE_STD * 2.5)
            temperature = target_temp + random.normalvariate(0, config.SIM_TEMP_NOISE_STD * 2.5)
            gas = 0.01 + random.uniform(0, 0.08)
            pressure = 101300 + random.normalvariate(0, 300)
            power = target_power + random.normalvariate(0, 3.0)

        else: # 'Random' mode
            self.runtime_steps += 1
            if self.current_fault_mode == "Normal" and self.runtime_steps > config.SIM_MIN_NORMAL_RUNTIME:
                if random.random() < config.SIM_FAULT_CHANCE_PER_STEP:
                    self.current_fault_mode = random.choice(config.ACTUAL_FAULT_MODES)
                    self.time_in_fault = 0
                    # Logging should happen in the main app based on state change
            if self.current_fault_mode != "Normal":
                self.time_in_fault += 1

            fault_params = config.FAILURE_MODES[self.current_fault_mode]
            progression_factor = min(1.0, self.time_in_fault / 50.0)
            self.temp_drift_offset += config.SIM_SENSOR_DRIFT_FACTOR * self.base_temp
            self.vibration_drift_offset += config.SIM_SENSOR_DRIFT_FACTOR * self.base_vibration
            current_base_temp = self.base_temp + self.temp_drift_offset
            current_base_vib = self.base_vibration + self.vibration_drift_offset
            target_temp = current_base_temp * (1 + (fault_params["temp_factor"] - 1) * progression_factor)
            target_vib = current_base_vib * (1 + (fault_params["vib_factor"] - 1) * progression_factor)
            if fault_params["vib_pattern_amp"] > 0:
                 vib_pattern = fault_params["vib_pattern_amp"] * progression_factor * np.sin(self.runtime_steps * 0.5)
                 target_vib += vib_pattern
            speed, base_vib, base_temp, gas, pressure, base_power = self._generate_base_readings(noise_factor=1.0)
            vibration = target_vib + random.normalvariate(0, config.SIM_VIBRATION_NOISE_STD)
            temperature = target_temp + random.normalvariate(0, config.SIM_TEMP_NOISE_STD)
            temp_deviation = max(0, temperature - config.SIM_BASE_TEMP_MEAN)
            vib_deviation = max(0, vibration - config.SIM_BASE_VIBRATION_MEAN)
            power = config.SIM_POWER_BASE_KW * fault_params["power_factor"] + \
                    (temp_deviation + vib_deviation * 2) * config.SIM_POWER_LOAD_FACTOR
            power += random.normalvariate(0, 0.5)

        # Apply Constraints and Store
        self.current_data = {
            'Speed': max(2000, min(speed, 3000)),
            'Vibration': max(0.1, min(vibration, config.MAX_VIBRATION)),
            'Temperature': max(20, min(temperature, config.MAX_TEMPERATURE)),
            'Gas': max(0.001, min(gas, 0.1)),
            'Pressure': max(100000, min(pressure, 102000)),
            'Power': max(10, min(power, config.MAX_POWER))
        }
        self.data_history.append(self.current_data.copy())
        return self.current_data

    def get_state_dict(self):
         """Returns the current state as a dictionary for session state."""
         # Convert deque to list for JSON compatibility if needed, but Streamlit handles deques okay
         return {
             "id": self.id,
             "data_history": self.data_history, # Keep as deque
             "current_data": self.current_data,
             "predicted_rul": self.predicted_rul,
             "predicted_fault": self.predicted_fault,
             "is_anomaly": self.is_anomaly,
             "status": self.status,
             "runtime_steps": self.runtime_steps,
             "current_fault_mode": self.current_fault_mode,
             "time_in_fault": self.time_in_fault,
             "base_speed": self.base_speed,
             "base_temp": self.base_temp,
             "base_vibration": self.base_vibration,
             "temp_drift_offset": self.temp_drift_offset,
             "vibration_drift_offset": self.vibration_drift_offset,
             "simulation_mode": self.simulation_mode,
             "worst_case_fault_type": self.worst_case_fault_type
         }

    def update_predictions(self, rul, fault, anomaly_status, overall_status):
        """Updates prediction results in the state."""
        self.predicted_rul = rul
        self.predicted_fault = fault
        self.is_anomaly = anomaly_status
        self.status = overall_status

    def set_simulation_mode(self, mode):
        """Sets the simulation mode."""
        if mode in config.SIMULATION_MODES:
            self.simulation_mode = mode

    def reset_runtime_state(self):
        """Resets runtime counters and fault state for a fresh monitoring run."""
        self.runtime_steps = 0
        self.current_fault_mode = "Normal"
        self.time_in_fault = 0
        self.temp_drift_offset = 0.0
        self.vibration_drift_offset = 0.0
        self.data_history.clear()
        self.current_data = {}
        self.predicted_rul = 100.0
        self.predicted_fault = "Initializing"
        self.is_anomaly = False
        self.status = "Normal"

    def set_stopped_state(self):
        """Sets the state to stopped."""
        self.predicted_fault = "Stopped"
        self.status = "Stopped"
        self.is_anomaly = False

