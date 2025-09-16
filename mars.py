import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import google.generativeai as genai
from io import StringIO

# --------------------------
# CONFIG & PAGE SETTINGS
# --------------------------
st.set_page_config(
    page_title="HabitatX – Mars Sustainability Simulator",
    layout="wide",
    page_icon="🪐"
)

st.title("🪐 HabitatX: Sustainable Living Simulator for Mars")
st.markdown("Model Mars habitat survival with **AI-powered analysis** (Gemini 2.0 Flash).")

# --------------------------
# SIDEBAR INPUTS
# --------------------------
st.sidebar.header("🔧 Habitat Parameters")
crew_size = st.sidebar.slider("Crew Size", 2, 10, 4)
greenhouse_area = st.sidebar.slider("Greenhouse Area (m²)", 50, 500, 200, step=10)
solar_capacity = st.sidebar.slider("Solar Power (kW)", 10, 200, 80, step=5)
recycle_efficiency = st.sidebar.slider("Water/O2 Recycling Efficiency (%)", 50, 99, 85)
initial_oxygen = st.sidebar.number_input("Initial Oxygen (kg)", 1000, 10000, 5000, step=500)
initial_water = st.sidebar.number_input("Initial Water (liters)", 5000, 50000, 20000, step=1000)
initial_food = st.sidebar.number_input("Initial Food (kg)", 1000, 20000, 8000, step=500)
backup_power = st.sidebar.checkbox("Include Backup Nuclear Power", value=False)
sim_days = st.sidebar.number_input("Simulation Length (days)", 100, 5000, 1000, step=100)
use_ai_opt = st.sidebar.checkbox("🤖 Auto-Optimize Habitat", value=False)

# --------------------------
# SIMULATION MODEL
# --------------------------
days = sim_days
consumption_oxygen_per_day = 0.84 * crew_size
consumption_water_per_day = 3 * crew_size
consumption_food_per_day = 0.62 * crew_size
power_need_per_day = 30 * crew_size

oxygen_production = greenhouse_area * 0.02
water_recycle_gain = (recycle_efficiency/100) * consumption_water_per_day
solar_gain = solar_capacity * 5
if backup_power:
    solar_gain += 150

oxygen, water, food, power = [initial_oxygen], [initial_water], [initial_food], [solar_capacity*5]

for _ in range(1, days):
    new_o = oxygen[-1] + oxygen_production - consumption_oxygen_per_day
    new_w = water[-1] + water_recycle_gain - consumption_water_per_day
    new_f = food[-1] - consumption_food_per_day
    new_p = power[-1] + solar_gain - power_need_per_day
    if min(new_o, new_w, new_f, new_p) <= 0:
        break
    oxygen.append(new_o); water.append(new_w); food.append(new_f); power.append(new_p)

survival_days = len(oxygen)

# --------------------------
# METRICS & ALERT
# --------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Survival Days", survival_days)
col2.metric("Final O₂ (kg)", f"{oxygen[-1]:.1f}")
col3.metric("Final Water (L)", f"{water[-1]:.1f}")
col4.metric("Final Power (kWh)", f"{power[-1]:.1f}")

# Detect first depleted resource
final_values = {'Oxygen': oxygen[-1], 'Water': water[-1], 'Food': food[-1], 'Power': power[-1]}
first_out = min(final_values, key=final_values.get)
st.warning(f"⚠️ **First critical resource likely to run out:** {first_out}")

# --------------------------
# PLOTS
# --------------------------
st.subheader("📊 Resource Levels Over Time")
df = pd.DataFrame({
    "Day": range(survival_days),
    "Oxygen (kg)": oxygen,
    "Water (L)": water,
    "Food (kg)": food,
    "Power (kWh)": power
})
fig = px.line(df, x="Day", y=df.columns[1:], title="Resource Depletion Timeline")
st.plotly_chart(fig, use_container_width=True)

# --------------------------
# AI OPTIMIZATION
# --------------------------
if use_ai_opt:
    st.subheader("🤖 AI Habitat Optimization")
    data = []
    for _ in range(800):
        cs = np.random.randint(2, 10)
        gh = np.random.randint(50, 500)
        sc = np.random.randint(10, 200)
        re = np.random.randint(50, 99)
        initO = np.random.randint(1000, 10000)
        initW = np.random.randint(5000, 50000)
        initF = np.random.randint(1000, 20000)
        days_est = (initO/(0.84*cs) + initW/(3*cs) + initF/(0.62*cs))/3 + gh*0.2 + sc*0.3 + re
        data.append([cs, gh, sc, re, initO, initW, initF, days_est])
    df_train = pd.DataFrame(data, columns=["cs","gh","sc","re","o","w","f","days"])
    X, y = df_train.iloc[:,:-1], df_train["days"]
    model = RandomForestRegressor().fit(X, y)
    base_pred = model.predict([[crew_size, greenhouse_area, solar_capacity,
                                recycle_efficiency, initial_oxygen,
                                initial_water, initial_food]])[0]
    st.info(f"💡 AI-predicted optimal survival near **{int(base_pred)} days**.")
    # Suggest parameter improvements
    suggestions = []
    for factor, bump in [('greenhouse_area', 50), ('solar_capacity', 20), ('recycle_efficiency', 5)]:
        new_vals = dict(crew_size=crew_size, greenhouse_area=greenhouse_area,
                        solar_capacity=solar_capacity, recycle_efficiency=recycle_efficiency,
                        initial_oxygen=initial_oxygen, initial_water=initial_water,
                        initial_food=initial_food)
        new_vals[factor] += bump
        s = model.predict([[new_vals['crew_size'], new_vals['greenhouse_area'],
                            new_vals['solar_capacity'], new_vals['recycle_efficiency'],
                            new_vals['initial_oxygen'], new_vals['initial_water'],
                            new_vals['initial_food']]])[0]
        suggestions.append([factor.replace('_',' ').title(), bump, int(s)])
    st.table(pd.DataFrame(suggestions, columns=["Parameter +Increase", "Amount", "Predicted Days"]))

# --------------------------
# GEMINI 2.0 FLASH INSIGHTS
# --------------------------
st.subheader("🧠 Mission Insight (Gemini 2.0 Flash)")
api_key = st.secrets.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = (
        f"Our Mars habitat lasted {survival_days} days with crew {crew_size}, "
        f"greenhouse {greenhouse_area} m², solar {solar_capacity} kW, "
        f"recycling {recycle_efficiency}%. Provide a concise mission report "
        "with key risks and suggest 2 improvements."
    )
    try:
        response = model.generate_content(prompt)
        mission_report = response.text
        st.write(mission_report)

        # Allow download
        buf = StringIO(mission_report)
        st.download_button(
            label="📥 Download Mission Report",
            data=buf.getvalue(),
            file_name="HabitatX_Mission_Report.txt",
            mime="text/plain"
        )
    except Exception as e:
        st.warning(f"Gemini request failed: {e}")
else:
    st.info("Add your Gemini API key in `.streamlit/secrets.toml` as GEMINI_API_KEY to enable mission reports.")

st.caption("🚀 HabitatX – Enhanced Open Source Living Simulator for Mars")
