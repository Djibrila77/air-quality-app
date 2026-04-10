import streamlit as st
import joblib
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌍",
    layout="wide"
)

st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# Chargement des fichiers
import os

BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "xgboost_model.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))

# HEADER
st.title("🌍 Prédiction intelligente de la pollution d'une Smart City")
st.markdown("""
Bienvenue 👋  
Cette application permet de **prédire la qualité de l'air du lendemain**  
à partir de données récentes et météorologiques.
""")


# SIDEBAR
st.sidebar.header("⚙️ Paramètres utilisateur")

st.sidebar.markdown("### 📈 Pollution récente")
pm25_t_1 = st.sidebar.number_input("Hier (t-1)", 0.0, 500.0, 80.0)
pm25_t_2 = st.sidebar.number_input("Avant-hier (t-2)", 0.0, 500.0, 70.0)
pm25_t_3 = st.sidebar.number_input("Il y a 3 jours (t-3)", 0.0, 500.0, 60.0)

st.sidebar.markdown("### 🌡️ Conditions météo")
TEMP = st.sidebar.number_input("Température (°C)", -19.0, 43.0, 4.0)
DEWP = st.sidebar.number_input("Humidité (DEWP)", -40.0, 28.0, 2.0)
Iws = st.sidebar.slider("Vent", 0.45, 585.6, 1.0)
PRES = st.sidebar.number_input("Pression", 900.0, 1100.0, 1010.0)
Ir = st.sidebar.slider("🌧️ Pluie (mm)", 0.0, 50.0, 0.0)
cbwd = st.sidebar.selectbox("Direction du vent", ["NW", "SE", "NE", "cv"])


# ENCODAGE
cbwd_NW = 1 if cbwd == "NW" else 0
cbwd_SE = 1 if cbwd == "SE" else 0
cbwd_NE = 1 if cbwd == "NE" else 0
cbwd_cv = 1 if cbwd == "cv" else 0


# FEATURES AUTOmatique

pm25_mean_3 = (pm25_t_1 + pm25_t_2 + pm25_t_3) / 3
rolling_6 = pm25_mean_3
rolling_12 = pm25_mean_3
temp_wind = TEMP * Iws


now = datetime.datetime.now()
hour, day, month, year, day_of_week = now.hour, now.day, now.month, now.year, now.weekday()

# DATA
data = {
    "pm25_mean_3": pm25_mean_3,
    "pm25_t-2": pm25_t_2,
    "cbwd_NW": cbwd_NW,
    "pm25_t-1": pm25_t_1,
    "DEWP": DEWP,
    "temp_wind": temp_wind,
    "Ir": Ir,
    "cbwd_SE": cbwd_SE,
    "Iws": Iws,
    "day_of_week": day_of_week,
    "PRES": PRES,
    "hour": hour,
    "TEMP": TEMP,
    "cbwd_NE": cbwd_NE,
    "pm25_t-3": pm25_t_3,
    "rolling_12": rolling_12,
    "rolling_6": rolling_6,
    "day": day,
    "cbwd_cv": cbwd_cv,
    "month": month,
    "year": year,
    
}

X = pd.DataFrame([data])[features]
st.info("⚠️ La prédiction est principalement basée sur l’historique de pollution.")

# BOUTON
if st.button("🔮 Lancer la prédiction"):

    prediction = model.predict(X)[0]
    col1, col2 = st.columns(2)

    with col1:
        st.metric("📊 PM2.5 prévu", f"{prediction:.2f}")

    with col2:
        if prediction < 50:
            st.success("🟢 Air bon")
        elif prediction < 100:
            st.warning("🟡 Air moyen")
        else:
            st.error("🔴 Air mauvais")

    st.markdown("### 📘 Interprétation")
    st.info("Plus la valeur est élevée, plus la pollution est importante.")
   
    # Visualisation
    st.markdown("### 📈 Évolution temporelle")

    df_plot = pd.DataFrame({
        "Temps": ["t-3", "t-2", "t-1", "Demain"],
        "PM2.5": [pm25_t_3, pm25_t_2, pm25_t_1, prediction]
    })

    fig, ax = plt.subplots()
    ax.plot(df_plot["Temps"], df_plot["PM2.5"], marker='o')
    st.pyplot(fig)

  
    # GRAPH 2
   
    st.markdown("### 📊 Comparaison")

    fig2, ax2 = plt.subplots()
    ax2.bar(df_plot["Temps"], df_plot["PM2.5"])
    st.pyplot(fig2)
st.info("Conseil: Protegez bien ainsi que votre famille,en suivant les consignes fournis par les structures de la sante")

# FOOTER
st.markdown("---")
st.caption("Projet Data Science | Prédiction de pollution urbaine")