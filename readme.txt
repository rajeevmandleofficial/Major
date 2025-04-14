# TBF Condition Monitoring - AI-Powered Predictive Maintenance Dashboard

Yeh project Tunnel Booster Fans (TBFs) ke liye ek AI-based predictive maintenance system ka prototype hai. Yeh simulated sensor data use karke RUL (Remaining Useful Life), Fault Diagnosis, aur Anomaly Detection karta hai.

**Developed By:** Rajeev Mandle, Ayush Mark Ekka, Janmay Jai Singh (Bhilai Institute of Technology Durg)

## File Structure (Folder Layout)

Project ko run karne ke liye, aapke folder ka structure aisa hona chahiye:

Your_Project_Folder_Name/│├── assets/              # CSS aur images ke liye folder│   └── style.css        # UI styling ke liye file│├── pages/               # Dash multi-page feature ke liye folder│   ├── landing.py       # Landing page ka code│   └── dashboard.py     # Main dashboard ka code│├── config.py            # Saari configuration settings├── simulation.py        # TBF simulation logic (TBFState class)├── data_processing.py   # Data generation aur feature engineering functions├── ml_models.py         # Model training/loading aur prediction functions├── shared_state.py      # Shared variables/logger (GLOBAL_STATE)├── dash_app.py          # Main Dash application file (isko run karna hai)├── requirements.txt     # Zaroori Python libraries ki list│├── tbf_historical_data.csv  (Optional - agar nahi hai toh generate ho jayegi)├── tbf_rul_model.pkl        (Optional - agar nahi hai toh train ho jayega)├── tbf_fault_model.pkl      (Optional - agar nahi hai toh train ho jayega)├── tbf_anomaly_model.pkl    (Optional - agar nahi hai toh train ho jayega)├── tbf_anomaly_scaler.pkl   (Optional - agar nahi hai toh train ho jayega)└── tbf_monitor_streamlit.log (Log file - automatically banegi)
## Prerequisites (Pehle Se Kya Chahiye)

* **Python:** Version 3.8 ya usse naya install hona chahiye. (Humne 3.13 use kiya tha, par 3.8+ chalna chahiye). Aap [python.org](https://www.python.org/downloads/) se download kar sakte hain.
* **pip:** Python ka package installer (usually Python ke saath aata hai).

## Setup Instructions (Kaise Install Karein)

1.  **Get the Code:** Project ki saari files (upar diye gaye structure ke hisaab se) apne dost ke laptop pe copy karo. Ya agar Git use kar rahe ho toh clone karo.

2.  **Open Terminal:** Us folder mein jao jahan aapne project copy kiya hai (Command Prompt, PowerShell, Git Bash, etc. use karke).
    ```bash
    cd path/to/Your_Project_Folder_Name
    ```

3.  **Create Virtual Environment (Recommended):** Ek virtual environment banana acchi practice hai taaki project ki libraries system ki libraries se mix na hon.
    ```bash
    # Create virtual environment (venv naam ka folder banega)
    python -m venv venv
    # Activate virtual environment
    # Windows pe:
    .\venv\Scripts\activate
    # MacOS/Linux pe:
    source venv/bin/activate
    ```
    *(Jab kaam ho jaaye, deactivate karne ke liye sirf `deactivate` type karna hota hai).*

4.  **Install Dependencies:** Ab zaroori libraries install karo `requirements.txt` file use karke.
    ```bash
    pip install -r requirements.txt
    ```
    *(Yeh command neeche di gayi saari libraries install kar degi).*

## Dependencies (Kya Install Hoga)

`requirements.txt` file mein yeh libraries honi chahiye:

```text
pandas
numpy
scikit-learn
joblib
plotly
dash >= 2.10
dash-bootstrap-components >= 1.4
Running the Application (Kaise Chalayein)Ensure karo ki aapka virtual environment activated hai (agar banaya tha toh) aur aap project folder mein ho terminal ke andar.Neeche di gayi command run karo:python dash_app.py
Terminal mein output aayega, kuch aisa:Dash is running on [http://127.0.0.1:8050/](http://127.0.0.1:8050/)
 * Serving Flask app 'dash_app'
 * Debug mode: on
... (baaki logs) ...
Apne web browser (Chrome, Firefox, etc.) mein yeh URL open karo: http://127.0.0.1:8050/App ka landing page khul jaayega. Wahan se "Launch Dashboard" button click karke monitoring dashboard pe jaa sakte ho.NotesAgar .pkl model files ya tbf_historical_data.csv file project folder mein nahi hongi, toh app pehli baar start hone par unhe automatically generate/train karne ki koshish karega. Is
  @media print {
    .ms-editor-squiggler {
        display:none !important;
    }
  }
  .ms-editor-squiggler {
    all: initial;
    display: block !important;
    height: 0px !important;
    width: 0px !important;
  }