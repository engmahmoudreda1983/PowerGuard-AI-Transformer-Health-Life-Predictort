import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
from datetime import datetime

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="PowerGuard AI | Transformer Monitoring",
    page_icon="⚡",
    layout="wide"
)

# --- التنسيق الجمالي (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .status-card { padding: 20px; border-radius: 10px; color: white; text-align: center; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- تحميل الموديلات ---
@st.cache_resource
def load_all_models():
    try:
        m_health = joblib.load('rf_health_model.pkl')
        m_life = joblib.load('rf_life_model.pkl')
        m_thermal = joblib.load('rf_thermal_model.pkl')
        return m_health, m_life, m_thermal
    except Exception as e:
        st.error(f"⚠️ خطأ في تحميل ملفات الموديل: {e}")
        return None, None, None

model_health, model_life, model_thermal = load_all_models()

# --- الهيدر ---
st.markdown("""
<div style='background-color: #0b2e59; padding: 20px; border-radius: 10px; margin-bottom: 25px;'>
    <h1 style='color: #ffffff; text-align: center; margin: 0;'>AI Transformer Health Guard ⚡</h1>
    <p style='color: #4da6ff; text-align: center; margin: 5px 0 0 0;'>Advanced Predictive Maintenance Dashboard</p>
</div>
""", unsafe_allow_html=True)

# --- القائمة الجانبية (Sidebar) ---
st.sidebar.header("📊 Input Parameters")
analysis_type = st.sidebar.radio("Select Analysis:", ["DGA Health Analysis", "Thermal Prediction"])

# --- الجزء الأول: تحليل الغازات (DGA) ---
if analysis_type == "DGA Health Analysis":
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🧪 DGA Gas Levels (ppm)")
        h2 = st.number_input("Hydrogen (H2)", 0.0, 10000.0, 50.0)
        o2 = st.number_input("Oxygen (O2)", 0.0, 50000.0, 1500.0)
        n2 = st.number_input("Nitrogen (N2)", 0.0, 100000.0, 50000.0)
        ch4 = st.number_input("Methane (CH4)", 0.0, 10000.0, 20.0)
        co = st.number_input("Carbon Monoxide (CO)", 0.0, 2000.0, 300.0)
        co2 = st.number_input("Carbon Dioxide (CO2)", 0.0, 15000.0, 2000.0)
        c2h4 = st.number_input("Ethylene (C2H4)", 0.0, 5000.0, 10.0)
        c2h6 = st.number_input("Ethane (C2H6)", 0.0, 5000.0, 15.0)
        c2h2 = st.number_input("Acetylene (C2H2)", 0.0, 5000.0, 0.5)
        
        st.subheader("⚙️ Physical Properties")
        dbds = st.number_input("DBDS", 0.0, 500.0, 5.0)
        pf = st.number_input("Power Factor", 0.0, 10.0, 0.5)
        iv = st.number_input("Interfacial Tension", 0.0, 50.0, 35.0)
        dr = st.number_input("Dielectric Rigidity", 0.0, 100.0, 60.0)
        wc = st.number_input("Water Content", 0.0, 100.0, 15.0)

    with col2:
        if st.button("🔄 Run Diagnostics", use_container_width=True):
            if model_health and model_life:
                # تجهيز البيانات للتنبؤ
                input_data = pd.DataFrame([[h2, o2, n2, ch4, co, co2, c2h4, c2h6, c2h2, dbds, pf, iv, dr, wc]], 
                                        columns=['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 
                                                 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 
                                                 'Interfacial V', 'Dielectric rigidity', 'Water content'])
                
                health_score = model_health.predict(input_data)[0]
                life_exp = model_life.predict(input_data)[0]
                
                # عرض النتائج
                res_c1, res_c2 = st.columns(2)
                
                # Gauge Chart for Health
                fig_h = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = health_score,
                    title = {'text': "Health Index (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#0b2e59"},
                        'steps': [
                            {'range': [0, 50], 'color': "#ff4b4b"},
                            {'range': [50, 80], 'color': "#ffa500"},
                            {'range': [80, 100], 'color': "#00cc96"}]}))
                res_c1.plotly_chart(fig_h, use_container_width=True)
                
                # Estimated Life
                res_c2.metric("Estimated Remaining Life", f"{life_exp:.1f} Years")
                
                # تفسير الحالة
                if health_score > 80:
                    st.success("✅ Transformer condition is EXCELLENT. Continue routine maintenance.")
                elif health_score > 50:
                    st.warning("⚠️ Transformer condition is FAIR. Increase monitoring frequency.")
                else:
                    st.error("🚨 CRITICAL CONDITION. Immediate inspection and oil filtration recommended.")

# --- الجزء الثاني: التنبؤ الحراري (Thermal) ---
elif analysis_type == "Thermal Prediction":
    st.subheader("🌡️ Transformer Thermal Load Prediction")
    
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        hufl = st.slider("High Utilization Factor (HUFL)", 0.0, 1.0, 0.5)
        hull = st.slider("High Load Limit (HULL)", 0.0, 1.0, 0.4)
        mufl = st.slider("Medium Utilization Factor (MUFL)", 0.0, 1.0, 0.3)
        mull = st.slider("Medium Load Limit (MULL)", 0.0, 1.0, 0.2)
        
    with t_col2:
        lufl = st.slider("Low Utilization Factor (LUFL)", 0.0, 1.0, 0.1)
        lull = st.slider("Low Load Limit (LULL)", 0.0, 1.0, 0.1)
        current_hour = datetime.now().hour
        current_month = datetime.now().month
        hour = st.number_input("Hour of Day", 0, 23, current_hour)
        month = st.number_input("Month", 1, 12, current_month)

    if st.button("🌡️ Predict Oil Temp"):
        if model_thermal:
            t_input = pd.DataFrame([[hufl, hull, mufl, mull, lufl, lull, hour, month]], 
                                 columns=['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'Hour', 'Month'])
            pred_ot = model_thermal.predict(t_input)[0]
            
            st.metric("Predicted Oil Temperature", f"{pred_ot:.2f} °C")
            
            # رسم توضيحي للحرارة
            fig_t = go.Figure(go.Indicator(
                mode = "number+gauge",
                value = pred_ot,
                gauge = {'axis': {'range': [0, 120]},
                         'steps': [
                             {'range': [0, 60], 'color': "green"},
                             {'range': [60, 90], 'color': "yellow"},
                             {'range': [90, 120], 'color': "red"}]}))
            st.plotly_chart(fig_t)

# --- تذييل الصفحة ---
st.markdown("---")
st.caption(f"PowerGuard AI v1.0 | Developed by Eng. Mahmoud | Last Synced: {datetime.now().strftime('%Y-%m-%d %H:%M')}")