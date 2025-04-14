import joblib
import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import config # Import settings from config.py

#@st.cache_resource # Cache models and scaler to avoid reloading/retraining
def train_or_load_model(model_type, df, features, target, model_path, scaler_path=None, model_params={}, classifier=False, is_anomaly=False):
    """Loads a model/scaler if it exists, otherwise trains and saves it."""
    model = None
    scaler = None
    model_loaded = False
    scaler_loaded = False

    # --- Try Loading ---
    if is_anomaly and scaler_path:
        if os.path.exists(scaler_path):
            try: scaler = joblib.load(scaler_path); scaler_loaded = True; # log_info(f"Loaded scaler from {scaler_path}") # Logging in main app
            except Exception as e: pass # log_error(f"Error loading scaler {scaler_path}: {e}. Will retrain.") # Logging in main app
        # else: log_warning(f"Scaler file not found at {scaler_path}. Need to train.") # Logging in main app

    if os.path.exists(model_path):
        try: model = joblib.load(model_path); model_loaded = True; # log_info(f"Loaded {model_type} model from {model_path}") # Logging in main app
        except Exception as e: pass # log_error(f"Error loading model {model_path}: {e}. Will retrain.") # Logging in main app
    # else: log_warning(f"Model file not found at {model_path}. Need to train.") # Logging in main app

    # --- Train if Loading Failed ---
    if not model_loaded or (is_anomaly and not scaler_loaded):
        if df is None or df.empty:
            st.error(f"Cannot train {model_type} model: No historical data provided.")
            # log_error(f"Cannot train {model_type}: Historical data is None or empty.") # Logging in main app
            return None, None

        st.info(f"Training new {model_type} model...")
        # log_info(f"Training new {model_type} model...") # Logging in main app

        # Validate features and target
        missing_features = [f for f in features if f not in df.columns]
        if missing_features: st.error(f"Error: Missing features for {model_type} training: {missing_features}"); return None, None
        if target is not None and target not in df.columns: st.error(f"Error: Missing target column '{target}' for {model_type} training."); return None, None

        X = df[features]
        y = df[target] if target is not None else None

        try:
            # --- Anomaly Detection Training ---
            if is_anomaly:
                st.info("Fitting Anomaly Detection model (Isolation Forest)...")
                scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
                model = IsolationForest(**model_params); model.fit(X_scaled)
                st.success(f"{model_type} Model Training Complete.")
                # Save scaler and model
                if scaler_path:
                    try: joblib.dump(scaler, scaler_path); # log_info(f"Saved anomaly scaler to {scaler_path}")
                    except Exception as e: st.error(f"Error saving scaler: {e}") # log_error(f"Error saving anomaly scaler {scaler_path}: {e}")
                try: joblib.dump(model, model_path); # log_info(f"Trained {model_type} model saved to {model_path}")
                except Exception as e: st.error(f"Error saving anomaly model: {e}") # log_error(f"Error saving anomaly model {model_path}: {e}")

            # --- Classifier Training ---
            elif classifier:
                split_params = {"test_size": 0.2, "random_state": 42}
                if y is not None and len(pd.unique(y)) > 1 :
                     try: split_params["stratify"] = y
                     except ValueError: pass # log_warning(f"Could not stratify split for {model_type}...")
                X_train, X_test, y_train, y_test = train_test_split(X, y, **split_params)
                model = RandomForestClassifier(**model_params); model.fit(X_train, y_train)
                y_pred = model.predict(X_test); accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
                st.success(f"{model_type} Model Training Complete. Test Accuracy: {accuracy:.2f}")
                with st.expander("Show Classification Report (Test Set)"):
                     st.dataframe(pd.DataFrame(report).transpose())
                # log_info(f"{model_type} Model trained. Accuracy: {accuracy:.2f}")
                joblib.dump(model, model_path) # log_info(f"Trained {model_type} model saved to {model_path}")

            # --- Regressor Training ---
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(**model_params); model.fit(X_train, y_train)
                y_pred = model.predict(X_test); mse = mean_squared_error(y_test, y_pred)
                st.success(f"{model_type} Model Training Complete. Test MSE: {mse:.2f}")
                # log_info(f"{model_type} Model trained. MSE: {mse:.2f}")
                joblib.dump(model, model_path) # log_info(f"Trained {model_type} model saved to {model_path}")

        except Exception as e:
            st.error(f"Error training {model_type} model: {e}")
            # log_error(f"Error training {model_type} model: {e}") # Logging in main app
            return None, None # Return None if training failed

    # Return the loaded or newly trained model and scaler
    return model, scaler


def predict_value(model, feature_df, required_features, tbf_id, prediction_type="Value"):
    """Generic prediction function for RUL or Fault."""
    default_return = 0.0 if prediction_type == "RUL" else "Unknown"
    if model is None: return default_return
    if feature_df is None or feature_df.empty: return default_return

    try:
        # Ensure DataFrame has all required columns in the correct order, fill missing with 0
        required_df = pd.DataFrame(0.0, index=feature_df.index, columns=required_features) # Use 0.0 for float dtype
        common_cols = list(set(feature_df.columns) & set(required_features))
        # Ensure dtypes match where possible before assigning
        for col in common_cols:
            if required_df[col].dtype == feature_df[col].dtype:
                 required_df[col] = feature_df[col]
            else:
                 try:
                     required_df[col] = feature_df[col].astype(required_df[col].dtype)
                 except Exception: # Fallback if conversion fails
                     required_df[col] = 0.0 # Or handle more gracefully

        # Make the prediction
        prediction = model.predict(required_df)[0]

        # Post-process prediction
        if prediction_type == "RUL":
            return max(0.0, float(prediction)) # Ensure RUL is not negative
        else: # Fault prediction
            return str(prediction)
    except KeyError as e:
         # log_error(f"Missing feature during {prediction_type} prediction for TBF {tbf_id}: {e}", tbf_id=tbf_id) # Logging in main app
         return "Error" if prediction_type == "Fault" else 0.0
    except Exception as e:
        # log_error(f"Error during {prediction_type} prediction for TBF {tbf_id}: {e}", tbf_id=tbf_id) # Logging in main app
        st.toast(f"Prediction Error TBF {tbf_id}: {e}", icon="🚨") # Show brief error to user
        return "Error" if prediction_type == "Fault" else 0.0


def predict_anomaly(model, scaler, current_data_dict, features, tbf_id):
    """Predicts if the current data point is an anomaly."""
    if model is None or scaler is None: return False
    if not current_data_dict: return False

    try:
        # Create DataFrame from the single current data point
        current_df = pd.DataFrame([current_data_dict])
        # Select and reorder features used by anomaly detector, fill missing with 0
        required_df = pd.DataFrame(0.0, index=current_df.index, columns=features)
        common_cols = list(set(current_df.columns) & set(features))
        # Ensure dtypes match where possible
        for col in common_cols:
             if required_df[col].dtype == current_df[col].dtype:
                 required_df[col] = current_df[col]
             else:
                 try:
                     required_df[col] = current_df[col].astype(required_df[col].dtype)
                 except Exception:
                     required_df[col] = 0.0

        # Scale the data using the loaded scaler
        scaled_data = scaler.transform(required_df)
        # Predict anomaly (-1 for anomaly, 1 for inlier)
        prediction = model.predict(scaled_data)[0]
        is_anomaly = (prediction == -1)

        # if is_anomaly: log_warning(f"Anomaly detected! Sensors: {current_data_dict}", tbf_id=tbf_id) # Logging in main app

        return is_anomaly
    except KeyError as e:
         # log_error(f"Missing feature during anomaly prediction for TBF {tbf_id}: {e}", tbf_id=tbf_id) # Logging in main app
         return False # Default to false on error
    except Exception as e:
        # log_error(f"Error during anomaly prediction for TBF {tbf_id}: {e}", tbf_id=tbf_id) # Logging in main app
        st.toast(f"Anomaly Prediction Error TBF {tbf_id}: {e}", icon=" B") # Show brief error
        return False # Default to false on error

