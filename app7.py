import streamlit as st
import json
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import requests
import geocoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --------------------------------------------
# Section 1: Train and Load Improved AG3.0 Model
# --------------------------------------------
def generate_synthetic_data(n=1000):
    """
    Generate synthetic data for the AG3.0 model.
    Features: PM2.5, PM10, Temperature (°C), Humidity (%)
    Labels (AQI_Category) are derived based on thresholds:
      0: Good, 1: Satisfactory, 2: Moderate,
      3: Poor, 4: Very Poor, 5: Severe.
    """
    np.random.seed(42)
    pm25 = np.random.uniform(10, 300, n)
    pm10 = np.random.uniform(20, 500, n)
    temp = np.random.uniform(10, 40, n)
    humidity = np.random.uniform(20, 80, n)

    def assign_category(p25, p10):
        # Determine category for PM2.5
        def cat_pm25(x):
            if x <= 30:
                return 0
            elif x <= 60:
                return 1
            elif x <= 90:
                return 2
            elif x <= 120:
                return 3
            elif x <= 250:
                return 4
            else:
                return 5
        # Determine category for PM10
        def cat_pm10(x):
            if x <= 50:
                return 0
            elif x <= 100:
                return 1
            elif x <= 250:
                return 2
            elif x <= 350:
                return 3
            elif x <= 430:
                return 4
            else:
                return 5
        # Take the worst category (higher value) from PM2.5 and PM10
        return max(cat_pm25(p25), cat_pm10(p10))

    labels = [assign_category(p, q) for p, q in zip(pm25, pm10)]
    df = pd.DataFrame({
        "PM2.5": pm25,
        "PM10": pm10,
        "Temperature": temp,
        "Humidity": humidity,
        "AQI_Category": labels
    })
    return df


def calculate_category(p25, p10):
    """
    Determine AQI category index (0-5) based on PM2.5 and PM10 using CPCB thresholds.
    """
    # PM2.5 breakpoints
    def cat_pm25(x):
        if x is None:
            return None
        if x <= 30:
            return 0
        elif x <= 60:
            return 1
        elif x <= 90:
            return 2
        elif x <= 120:
            return 3
        elif x <= 250:
            return 4
        else:
            return 5
    # PM10 breakpoints
    def cat_pm10(x):
        if x is None:
            return None
        if x <= 50:
            return 0
        elif x <= 100:
            return 1
        elif x <= 250:
            return 2
        elif x <= 350:
            return 3
        elif x <= 430:
            return 4
        else:
            return 5
    c25 = cat_pm25(p25)
    c10 = cat_pm10(p10)
    if c25 is None or c10 is None:
        return None
    return max(c25, c10)


def train_ag3_model():
    data = generate_synthetic_data(n=1000)
    X = data[["PM2.5", "PM10", "Temperature", "Humidity"]]
    y = data["AQI_Category"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"AG3.0 model trained on synthetic data with accuracy: {acc:.2f}")
    
    with open("ag3_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return model


def load_ag3_model():
    if os.path.exists("ag3_model.pkl"):
        with open("ag3_model.pkl", "rb") as f:
            return pickle.load(f)
    return train_ag3_model()


def map_category_to_label(category):
    mapping = {0: "Good", 1: "Satisfactory", 2: "Moderate", 3: "Poor", 4: "Very Poor", 5: "Severe"}
    return mapping.get(category, "Unknown")


def adjust_for_vulnerability(predicted_category, age, health_condition):
    if predicted_category is None:
        return None
    if age < 12 or age > 65 or health_condition.lower() in ["lung", "heart", "both"]:
        return min(predicted_category + 1, 5)
    return predicted_category


def compute_health_risk(predicted_category, age, health_condition):
    if predicted_category is None:
        return None
    base_risk = predicted_category * 20
    age_risk = 10 if (age < 12 or age > 65) else 0
    if health_condition.lower() == "both":
        health_risk = 15
    elif health_condition.lower() in ["lung", "heart"]:
        health_risk = 10
    else:
        health_risk = 0
    return min(base_risk + age_risk + health_risk, 100)

# --------------------------------------------
# Section 2 : Time Series Forecasting for AQI Predictions
# --------------------------------------------
def generate_historical_data(days=30):
    dates = [datetime.now() - timedelta(days=i) for i in range(days)][::-1]
    np.random.seed(42)
    base = 2
    values = []
    current = base
    for _ in range(days):
        step = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        current = np.clip(current + step, 0, 5)
        values.append(current)
    return pd.DataFrame({"Date": dates, "AQI_Category": values}).set_index("Date")


def forecast_aqi(historical_data, forecast_days=7):
    last = historical_data["AQI_Category"].iloc[-1]
    dates = [historical_data.index[-1] + timedelta(days=i) for i in range(1, forecast_days+1)]
    vals = []
    current = last
    for _ in range(forecast_days):
        step = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        current = np.clip(current + step, 0, 5)
        vals.append(current)
    return pd.DataFrame({"Date": dates, "Forecast_AQI": vals}).set_index("Date")


def plot_forecast_matplotlib(historical_data, forecast_data):
    hist = historical_data.rename(columns={"AQI_Category": "AQI"})
    fc = forecast_data.rename(columns={"Forecast_AQI": "AQI"})
    combined = pd.concat([hist, fc]).sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(combined.index, combined["AQI"], marker='o', linestyle='-')
    ax.set_title("Historical AQI Trends and Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI Category")
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# -----------------------------------------------
# Section 3: Helper functions for auto fetching weather data
# -----------------------------------------------
def get_location():
    g = geocoder.ip('me')
    return g.city, g.latlng


def get_weather_data(lat, lon, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    data = requests.get(url).json()
    return data['main']['temp'], data['main']['humidity']

# --------------------------------------------
# --------------------------------------------
# Section 4: Real-Time Sensor Integration & AQI Calculation
# --------------------------------------------
import requests
import streamlit as st

def fetch_sensor_data():
    """
    Fetches the latest PM sensor data from the AirGuard API.
    Returns the raw PMS_DATA dict.
    """
    try:
        resp = requests.get(
            "https://airguardv1-production.up.railway.app/api/getData",
            timeout=5
        )
        resp.raise_for_status()
        return resp.json().get("PMS_DATA", {})
    except Exception as e:
        st.error(f"Error fetching sensor data: {e}")
        return {}

def calculate_aqi(pm25, pm10):
    """
    Calculate the numeric AQI based on Indian CPCB breakpoints for PM2.5 and PM10.
    Returns the rounded max of the two sub-indices, clamped between 0 and 500.
    """
    # Cast and clamp
    try:
        p25 = max(float(pm25), 0.0)
        p10 = max(float(pm10), 0.0)
    except (TypeError, ValueError):
        return None

    bp_pm25 = [
        (0, 30,   0,  50),
        (30, 60,  51, 100),
        (60, 90, 101, 200),
        (90,120, 201, 300),
        (120,250,301, 400),
        (250,350,401, 500),
    ]
    bp_pm10 = [
        (0, 50,   0,  50),
        (50,100,  51, 100),
        (100,250,101,200),
        (250,350,201,300),
        (350,430,301,400),
        (430,600,401,500),
    ]

    def interp(conc, bps):
        for cl, ch, il, ih in bps:
            if cl <= conc <= ch:
                return ((ih - il)/(ch - cl))*(conc - cl) + il
        # above max range
        return bps[-1][3]

    aqi25 = interp(p25, bp_pm25)
    aqi10 = interp(p10, bp_pm10)
    aqi   = max(aqi25, aqi10)
    return int(round(min(max(aqi, 0), 500)))

# --------------------------------------------
# Section 5: Integrate Google AI API (Gemini)
# -------------------------------------------- (Gemini)
# --------------------------------------------
from google import genai
client = genai.Client(api_key="AIzaSyD40P9XPXBncy0LqILu7MIgW-LhcalclOU")

def call_google_ai_api(user_profile, ag3_output):
    prompt = f"User Profile: {user_profile}\nPredicted Air Quality: {ag3_output}"
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text


def google_ai_chat(prompt_text):
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt_text)
    return response.text


# --------------------------------------------
# --------------------------------------------
# --------------------------------------------
# Section 6: Streamlit App Interface (with Voice Assistant)
# --------------------------------------------
import streamlit as st
import streamlit.components.v1 as components
import json
import numpy as np
import pandas as pd

st.title("AirGuard - AG3.0 with Personalized Recommendations")

# Sidebar—added only the Voice Assistant line
st.sidebar.markdown("""
## Welcome to AirGuard!
AirGuard is your **AI-powered air quality predictor** that now includes advanced features:
- **Time Series Forecasting:** See historical AQI trends and forecast future levels.
- **Dynamic Health Risk Scoring:** Get a numeric risk score tailored to your profile.

### How to Use AirGuard:
1. **Enter Your Details:**  
   Provide your name, age, and health condition.
2. **AirGuard will fetch Sensor Data and Weather details:**  
   The sensor readings for PM2.5, PM10, Temperature, and Humidity.
3. **Get Your Prediction:**  
   Click **"Get Recommendation"** to see your personalized air quality prediction, dynamic risk score, and historical trend with a forecast.
4. **Chat with Our AI:**  
   Use the chat tab or Voice Assitant to ask any follow-up questions about your air quality or health.

*Stay informed and safe with AirGuard!*

                                                      - Initiative by Divyansh and Akshat.
""")


tabs = st.tabs([
    "Air Quality Prediction",
    "Still Got Questions?",
    "Voice Assistant"
])

# --- Tab 0: Inputs + Sensor Data + Recommendation ---
with tabs[0]:
    st.header("Air Quality Prediction & Personalized Recommendation")
    st.markdown("### Enter Your Details:")

    name             = st.text_input("Name", value="John Doe", key="name")
    age              = st.number_input("Age", min_value=1, max_value=100, value=25, key="age")
    health_condition = st.selectbox("Health Condition", ["None","Lung","Heart","Both"], key="health")

    city, coords = get_location()
    if coords:
        st.success(f"Location detected: {city}")
        lat, lon = coords
        api_key = "ce7796b8d88918005874ba8bb157c59c"
        temperature, humidity = get_weather_data(lat, lon, api_key)
        st.write(f"🌡️ Temperature: {temperature} °C")
        st.write(f"💧 Humidity: {humidity} %")
    else:
        st.error("Could not detect location.")

    st.markdown("### Sensor Data & AQI (updates on rerun):")
    raw = fetch_sensor_data() or {}

    if raw:
        pm1  = raw.get("PM1")
        pm25 = raw.get("PM2_5")
        pm10 = raw.get("PM10")
        try:
            pm1_f  = float(pm1)
            pm25_f = float(pm25)
            pm10_f = float(pm10)
        except:
            pm1_f = pm25_f = pm10_f = None

        c1, c2, c3 = st.columns(3)
        c1.metric("PM1 (µg/m³)",  pm1 or "N/A")
        c2.metric("PM2.5 (µg/m³)",pm25 or "N/A")
        c3.metric("PM10 (µg/m³)", pm10 or "N/A")

        if pm25_f is not None and pm10_f is not None:
            aqi_value = calculate_aqi(pm25_f, pm10_f)
            cat_idx   = calculate_category(pm25_f, pm10_f)
            cat_adj   = adjust_for_vulnerability(cat_idx, age, health_condition)
            category  = map_category_to_label(cat_adj)
        else:
            aqi_value = None
            category  = "Unknown"

        st.markdown(f"**AQI (Numeric):** {aqi_value if aqi_value is not None else 'N/A'}")
        st.markdown(f"**Category:** {category}")

        if aqi_value is not None:
            if aqi_value <= 50:
                st.success("✅ AQI is Good — no mask needed.")
            elif aqi_value <= 100:
                st.info("ℹ️ AQI is Satisfactory — mask optional.")
            elif aqi_value <= 200:
                st.warning("⚠️ AQI is Moderate — take precautions.")
            elif aqi_value <= 300:
                st.warning("🚧 AQI is Poor — wear a mask & limit exertion.")
            elif aqi_value <= 400:
                st.error("🚨 AQI is Very Poor — stay indoors & use purifier.")
            else:
                st.error("🔥 AQI is Severe — avoid outdoors; seek medical advice.")
        else:
            st.warning("Could not compute AQI—check sensor data.")

        tips = {
            "Good":"Enjoy outdoor activities safely!",
            "Satisfactory":"Air quality is acceptable.",
            "Moderate":"Consider reducing prolonged outdoor exertion.",
            "Poor":"Limit outdoor activities and wear a mask.",
            "Very Poor":"Stay indoors and use air purifiers.",
            "Severe":"Avoid outdoors; seek medical advice."
        }
        st.markdown(f"**Tip:** {tips.get(category, '')}")
    else:
        st.warning("Sensor data unavailable; please rerun to retry.")

    if st.button("Get Recommendation", key="get_recommend"):
        user_profile = {
            "Name": name, "Age": age, "Health Condition": health_condition,
            "PM2.5": pm25, "PM10": pm10,
            "Temperature": temperature, "Humidity": humidity
        }
        profile_str = json.dumps(user_profile)
        model       = load_ag3_model()
        features    = np.array([[pm25_f, pm10_f, temperature, humidity]])
        pred_cat    = model.predict(features)[0]
        adj_cat     = adjust_for_vulnerability(pred_cat, age, health_condition)
        ag3_label   = map_category_to_label(adj_cat)

        st.markdown("#### AG3.0 Prediction")
        st.write(f"The predicted Air Quality is: **{ag3_label}**")

        risk = compute_health_risk(pred_cat, age, health_condition)
        st.markdown("#### Your Health Risk Score")
        st.write(f"Your dynamic health risk score is: **{risk}/100**")

        rec = call_google_ai_api(profile_str, ag3_label)
        st.markdown("#### Personalized Recommendations for you")
        st.write(rec)

        st.session_state['user_profile'] = user_profile
        st.session_state['ag3_label']    = ag3_label
        st.session_state['risk_score']   = risk

        st.markdown("#### Historical AQI Trends and Forecast")
        hist = generate_historical_data(30)
        fc   = forecast_aqi(hist, 7)
        vis_h = hist.rename(columns={"AQI_Category":"AQI"})
        vis_f = fc.rename(columns={"Forecast_AQI":"AQI"})
        st.line_chart(pd.concat([vis_h,vis_f]).sort_index()["AQI"])
        plot_forecast_matplotlib(hist, fc)

# --- Tab 1: Chat (unchanged) ---
with tabs[1]:
    st.header("Chat with AirGuard – Your Personal AI Assistant")
    st.markdown("Enter your query regarding air quality, health recommendations, or safety:")
    prompt = st.text_area("Your Prompt", height=150, placeholder="Type your question here...", key="chat_prompt")
    if st.button("Send", key="send_chat"):
        if prompt:
            base = ""
            if 'user_profile' in st.session_state:
                base = (
                    f"User Profile: {json.dumps(st.session_state['user_profile'])}\n"
                    f"Predicted Air Quality: {st.session_state['ag3_label']}\n"
                )
            resp = google_ai_chat(base + prompt if base else prompt)
            st.markdown("#### AirGuard Response")
            st.write(resp)
        else:
            st.error("Please enter a prompt.")

# --- Tab 2: Voice Assistant (aesthetic embed & launch) ---
with tabs[2]:
    st.header("🎙️ AirGuard Voice Assistant")
    st.markdown("Chat with AirGuard using your voice or open in a new window for full experience.")

    # card-style container
    components.html("""
    <div style="
        background: #f0f4f8;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
    ">
        <iframe
          src="https://vapi.ai?demo=true&shareKey=afdc16cc-da80-480d-a509-77ea754e6cd3&assistantId=2bd0b76d-9d7e-4bef-8193-91812e523bb4"
          width="100%" height="350" style="border:none; border-radius:8px;">
        </iframe>
        <br><br>
        <a href="https://vapi.ai?demo=true&shareKey=afdc16cc-da80-480d-a509-77ea754e6cd3&assistantId=2bd0b76d-9d7e-4bef-8193-91812e523bb4" target="_blank"
           style="
               display: inline-block;
               background: linear-gradient(135deg, #4a90e2, #50e3c2);
               color: white;
               font-weight: bold;
               padding: 12px 24px;
               border-radius: 24px;
               text-decoration: none;
               box-shadow: 0 6px 16px rgba(0,0,0,0.15);
           ">
          Launch Full Voice Assistant
        </a>
    </div>
    """, height=460)

# Footer (unchanged)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 0.8em;'>"
    "AirGuard – a project by <strong>Divyansh Pathak</strong> and <strong>Akshat Kareer</strong>"
    "</p>", unsafe_allow_html=True
)
