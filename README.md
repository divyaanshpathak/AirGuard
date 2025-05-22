# AirGuard
AirGuard AG3.0 is an AI &amp; IoT-based system that predicts AQI using sensor data (PM2.5/PM10), assesses health risks, and offers personalized recommendations via a Streamlit app. Built with Python, scikit-learn, and Google Gemini API, it combines real-time monitoring with intelligent, user-focused insights.

# 🌬️ AirGuard AG3.0 – AI & IoT-Based Real-Time Air Quality Monitoring System

**AirGuard AG3.0** is a smart air quality monitoring system that integrates IoT sensors with AI models to provide real-time **Air Quality Index (AQI)** predictions, **health risk assessments**, and **personalized recommendations**. Built using Python and Streamlit, it uses PM2.5 and PM10 data, a trained Random Forest model, and the Google Gemini API for intelligent suggestions.

---

## 🚀 Features

- 📡 Real-time AQI prediction from PM2.5 and PM10 sensor inputs  
- 🧠 AI-driven health risk scoring  
- 💬 Personalized recommendations using Google Gemini API  
- 📊 Interactive Streamlit dashboard  
- 🔌 Ready for integration with real hardware (e.g., sensors, microcontrollers)

---

## 🛠️ Technologies Used

- Python  
- pandas, NumPy  
- scikit-learn  
- matplotlib  
- Streamlit  
- Google Gemini API  
- GitHub

---

## 📦 Project Structure

```
airguard-ag3.0/
│
├── app.py                 # Main Streamlit app
├── model/
│   └── rf_model.pkl       # Trained Random Forest model
├── utils/
│   └── aqi_calculator.py  # AQI calculation & health risk logic
├── data/
│   └── sample_data.csv    # Sample input data
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/airguard-ag3.0.git
cd airguard-ag3.0
```

### 2. Create a Virtual Environment (optional but recommended)

```bash
python -m venv venv

# Activate:
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:

```bash
pip install pandas numpy scikit-learn matplotlib streamlit openai python-dotenv
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory and add your Google Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```

Alternatively, set it as an environment variable in your terminal.

---

## ▶️ Run the App

Launch the Streamlit app with:

```bash
streamlit run app.py
```

The app will open in your browser. Enter PM2.5 and PM10 values to get real-time AQI predictions, health scores, and AI recommendations.

---

## 📈 Example Use Case

- User inputs: PM2.5 = 110, PM10 = 200  
- Output: AQI = "Very Unhealthy", Health Risk = "High", AI Assistant: “Avoid outdoor activities, wear an N95 mask.”

---

## 📱 Future Enhancements

- Android app with push notifications  
- Integration with weather APIs  
- Voice assistant support  
- Full hardware-software IoT deployment (ESP32/Arduino)

---

## 🤝 Contributing

Feel free to fork this repo and submit pull requests. Suggestions, bug reports, and improvements are always welcome!

---

## 📄 License

MIT License – feel free to use and adapt with attribution.

