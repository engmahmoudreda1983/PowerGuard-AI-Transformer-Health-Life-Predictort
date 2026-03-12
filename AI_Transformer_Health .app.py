import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

# Page Configuration
st.set_page_config(page_title="PowerGuard AI", page_icon="⚡", layout="wide")

# Header Design
st.markdown("""
<div style='background-color: #0b2e59; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h1 style='color: #ffffff; text-align: center; margin: 0;'>AI Predictive Maintenance ⚡</h1>
    <h3 style='color: #4da6ff; text-align: center; margin: 5px 0 0 0;'>Power Transformers DGA & Health Assessment</h3>
</div>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    model_health = joblib.load('rf_health_model.pkl')
    model_life = joblib.load('rf_life_model.pkl')
    return model_health, model_life

try:
    model_health, model_life = load_models()
    models_loaded = True
except FileNotFoundError:
    st.error("⚠️ Model files not found. Ensure .pkl files are in the repository.")
    models_loaded = False

# Duval Triangle Logic
def get_duval_diagnosis(ch4, c2h4, c2h2):
    total = ch4 + c2h4 + c2h2
    if total == 0: return "Normal (No significant fault gases)"
    p_ch4, p_c2h4, p_c2h2 = (ch4/total)*100, (c2h4/total)*100, (c2h2/total)*100
    if p_ch4 >= 98: return "PD (Partial Discharge)"
    if p_c2h2 < 4 and p_c2h4 < 20: return "T1 (Thermal Fault < 300°C)"
    if p_c2h2 < 4 and 20 <= p_c2h4 < 50: return "T2 (Thermal Fault 300-700°C)"
    if p_c2h2 < 15 and p_c2h4 >= 50: return "T3 (Thermal Fault > 700°C)"
    if 4 <= p_c2h2 <= 13 and p_c2h4 < 50: return "D1 (Low Energy Discharge)"
    if p_c2h2 > 13 or (p_c2h2 > 4 and p_c2h4 >= 50): return "D2 (High Energy Discharge)"
    return "DT (Mixed Faults)"

if models_loaded:
    tab1, tab2 = st.tabs(["✍️ Manual Entry (Single Test)", "📁 Batch Analysis (Upload File)"])

    # --- TAB 1: MANUAL ENTRY ---
    with tab1:
        col_input, col_output = st.columns([1, 2])
        with col_input:
            st.markdown("### 📊 Parameters (Inputs)")
            with st.form("input_form"):
                st.markdown("**1. Dissolved Gases (ppm):**")
                hydrogen = st.number_input("Hydrogen (H2)", value=10.0, help="Partial discharge indicator.")
                oxigen = st.number_input("Oxygen (O2)", value=500.0, help="Leakage indicator.")
                nitrogen = st.number_input("Nitrogen (N2)", value=10000.0)
                methane = st.number_input("Methane (CH4)", value=10.0, help="Low-temp thermal fault.")
                co = st.number_input("Carbon Monoxide (CO)", value=50.0, help="Paper degradation.")
                co2 = st.number_input("Carbon Dioxide (CO2)", value=500.0)
                ethylene = st.number_input("Ethylene (C2H4)", value=2.0, help="High-temp thermal fault.")
                ethane = st.number_input("Ethane (C2H6)", value=5.0)
                acethylene = st.number_input("Acetylene (C2H2)", value=0.0, help="CRITICAL: Arcing indicator.")
                
                st.markdown("**2. Oil Physical Properties:**")
                dbds = st.number_input("DBDS (ppm)", value=0.5, help="Corrosive sulfur.")
                power_factor = st.number_input("Power Factor (%)", value=0.1, help="Dielectric loss.")
                interfacial_v = st.number_input("Interfacial Tension (mN/m)", value=45.0, help="Oil purity.")
                dielectric_rigidity = st.number_input("Dielectric Rigidity (kV)", value=60.0, help="Insulation strength.")
                water_content = st.number_input("Water Content (ppm)", value=2.0, help="Moisture level.")
                
                submitted = st.form_submit_button("🔍 Run AI Analysis", use_container_width=True)

        with col_output:
            if submitted:
                # Prepare data
                input_df = pd.DataFrame([[hydrogen, oxigen, nitrogen, methane, co, co2, ethylene, ethane, acethylene, dbds, power_factor, interfacial_v, dielectric_rigidity, water_content]], 
                                          columns=['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V', 'Dielectric rigidity', 'Water content'])
                
                health_score = model_health.predict(input_df)[0]
                life_yrs = model_life.predict(input_df)[0]
                diagnosis = get_duval_diagnosis(methane, ethylene, acethylene)

                if health_score >= 70: status, color = "SAFE", "#00cc66"
                elif health_score >= 40: status, color = "WARNING", "#ffcc00"
                else: status, color = "RISKY", "#ff3333"

                # Results Display
                st.markdown("### 📈 AI Analysis Results")
                k1, k2, k3 = st.columns(3)
                k1.metric("Health Score", f"{health_score:.2f}")
                k2.metric("Status", status)
                k3.metric("Life Expectancy", f"{life_yrs:.1f} Yrs")

                col_g, col_d = st.columns([1, 1.2])
                with col_g:
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge", value=health_score,
                        gauge={'axis': {'range': [100, 0]}, 'bar': {'color': "white"},
                               'steps': [{'range': [70, 100], 'color': "#00cc66"},
                                         {'range': [40, 70], 'color': "#ffcc00"},
                                         {'range': [0, 40], 'color': "#ff3333"}]}))
                    fig_g.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                    st.plotly_chart(fig_g, use_container_width=True)
                    st.markdown(f"<h2 style='text-align: center; color: {color};'>{status}</h2>", unsafe_allow_html=True)

                with col_d:
                    df_tri = pd.DataFrame({'CH4': [methane], 'C2H4': [ethylene], 'C2H2': [acethylene]})
                    fig_tri = px.scatter_ternary(df_tri, a="CH4", b="C2H4", c="C2H2")
                    fig_tri.update_traces(marker=dict(size=12, color='red', symbol='cross'))
                    fig_tri.update_layout(title="Duval Triangle Projection", height=280, paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                    st.plotly_chart(fig_tri, use_container_width=True)
                    st.info(f"**Duval Diagnosis:** {diagnosis}")

                st.markdown("---")
                # Bar Chart (The one that was missing)
                st.markdown("#### 📊 Impact of Variables on Health Score")
                imps = model_health.feature_importances_ * 100
                df_imp = pd.DataFrame({'Feature': input_df.columns, 'Impact (%)': imps}).sort_values('Impact (%)', ascending=True).tail(7)
                fig_bar = px.bar(df_imp, x='Impact (%)', y='Feature', orientation='h', color='Impact (%)', color_continuous_scale='Blues')
                fig_bar.update_layout(height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font={'color': 'white'}, coloraxis_showscale=False)
                st.plotly_chart(fig_bar, use_container_width=True)

    # --- TAB 2: BATCH ANALYSIS ---
    with tab2:
        st.markdown("### 📁 Batch Processing")
        up = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
        if up:
            df_b = pd.read_csv(up) if up.name.endswith('.csv') else pd.read_excel(up)
            cols = ['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V', 'Dielectric rigidity', 'Water content']
            df_b['Predicted_Health'] = model_health.predict(df_b[cols])
            st.line_chart(df_b['Predicted_Health'])
            st.dataframe(df_b)
            st.download_button("📥 Download Results", df_b.to_csv(index=False), "transformer_results.csv", "text/csv")