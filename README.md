# TBF Condition Monitoring - AI-Powered Predictive Maintenance Dashboard

> An intelligent predictive maintenance system for Tunnel Booster Fans (TBFs) using machine learning and real-time dashboard visualization.

---

## Overview

This project is an AI-based predictive maintenance prototype for **Tunnel Booster Fans (TBFs)**. It uses simulated sensor data to perform three critical monitoring tasks:

- **RUL Prediction** - Predicts Remaining Useful Life of TBF components
- **Fault Diagnosis** - Identifies specific fault types from sensor patterns  
- **Anomaly Detection** - Detects unusual behavior deviating from normal operation

**Developed By:** Rajeev Mandle, Ayush Mark Ekka, Janmay Jai Singh  
**Institution:** Bhilai Institute of Technology, Durg

---

## Demo

Launch the dashboard locally and navigate to `http://localhost:8050` to view the interactive monitoring interface.

---

## Features

- 📊 **Real-time Dashboard** - Interactive Dash-based web interface with live data visualization
- 🔮 **RUL Prediction** - Machine learning model estimates remaining component lifetime
- ⚠️ **Fault Diagnosis** - Multi-class classification identifies specific fault conditions
- 🔍 **Anomaly Detection** - Isolation Forest algorithm spots unusual sensor readings
- 📈 **Interactive Charts** - Plotly-powered visualizations for trend analysis
- 🎛️ **Simulation Mode** - Generates realistic synthetic sensor data for demonstration

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **ML/DL** | scikit-learn, joblib |
| **Dashboard** | Dash, Plotly, Dash Bootstrap Components |
| **Data Processing** | Pandas, NumPy |
| **UI Styling** | CSS3, Bootstrap |

---

## Project Structure

```
├── assets/
│   └── style.css              # Dashboard styling
├── pages/
│   ├── landing.py             # Landing page
│   └── dashboard.py           # Main monitoring dashboard
├── config.py                  # Configuration settings
├── simulation.py              # TBF simulation engine (TBFState class)
├── data_processing.py         # Data generation & feature engineering
├── ml_models.py               # Model training, loading & prediction
├── shared_state.py            # Shared variables & logger (GLOBAL_STATE)
├── dash_app.py                # Main Dash application entry point
├── requirements.txt           # Python dependencies
├── tbf_historical_data.csv    # Historical sensor data (auto-generated)
├── tbf_rul_model.pkl          # Trained RUL prediction model
├── tbf_fault_model.pkl        # Trained fault classification model
├── tbf_anomaly_model.pkl      # Trained anomaly detection model
├── tbf_anomaly_scaler.pkl     # Feature scaler for anomaly detection
└── tbf_monitor_streamlit.log  # Application logs
```

---

## Prerequisites

- **Python** 3.8 or higher ([Download](https://www.python.org/downloads/))
- **pip** (Python package installer)

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rajeevmandleofficial/Major.git
   cd Major
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Start the application:

```bash
python dash_app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:8050
```

Click **"Launch Dashboard"** on the landing page to access the monitoring interface.

> **Note:** If model files (`.pkl`) or historical data (`.csv`) are missing, the application will automatically generate and train them on first launch.

---

## Dependencies

```
pandas
numpy
scikit-learn
joblib
plotly
dash >= 2.10
dash-bootstrap-components >= 1.4
```

---

## Architecture

```
Sensor Simulation → Data Processing → Feature Engineering 
                                            ↓
                    Dashboard ← Predictions ← ML Models
                        ↑
                   User Interface
```

1. **simulation.py** generates realistic TBF sensor data  
2. **data_processing.py** cleans and engineers features  
3. **ml_models.py** loads/trains models and generates predictions  
4. **dash_app.py** serves the interactive web dashboard  

---

## Contributors

| Name | Role |
|------|------|
| **Rajeev Mandle** | ML Engineer & Dashboard Developer |
| **Ayush Mark Ekka** | Backend & Data Processing |
| **Janmay Jai Singh** | Testing & Documentation |

---

## License

This project was developed as part of the academic curriculum at Bhilai Institute of Technology, Durg.

---

<div align="center">

⭐ Star this repo if you found it useful!

</div>
