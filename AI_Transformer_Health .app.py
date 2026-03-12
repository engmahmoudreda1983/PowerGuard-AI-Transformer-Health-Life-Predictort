import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

# إعدادات الصفحة
st.set_page_config(page_title="PowerGuard AI", page_icon="⚡", layout="wide")

# تصميم الهيدر باللغة الإنجليزية كما طلبت
st.markdown("""
<div style='background-color: #0b2e59; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h1 style='color: #ffffff; text-align: center; margin: 0;'>AI Predictive Maintenance ⚡</h1>
    <h3 style='color: #4da6ff; text-align: center; margin: 5px 0 0 0;'>Power Transformers DGA & Health Assessment</h3>
</div>
""", unsafe_allow_html=True)

# تحميل النماذج
@st.cache_resource
def load_models():
    try:
        model_health = joblib.load('rf_health_model.pkl')
        model_life = joblib.load('rf_life_model.pkl')
        return model_health, model_life
    except:
        return None, None

model_health, model_life = load_models()

# منطق تشخيص مثلث دوفال
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

if model_health is not None:
    tab1, tab2 = st.tabs(["✍️ Manual Entry (Single Test)", "📁 Batch Analysis (Upload File)"])

    # --- TAB 1: MANUAL ENTRY ---
    with tab1:
        col_input, col_output = st.columns([1, 2])
        with col_input:
            st.markdown("### 📊 Parameters (Inputs)")
            with st.form("input_form"):
                st.markdown("**1. Dissolved Gases (ppm):**")
                h2 = st.number_input("Hydrogen (H2)", value=10.0, help="Partial discharge indicator.")
                o2 = st.number_input("Oxygen (O2)", value=500.0, help="Leakage indicator.")
                n2 = st.number_input("Nitrogen (N2)", value=10000.0)
                ch4 = st.number_input("Methane (CH4)", value=5.0, help="Low-temp thermal fault.")
                co = st.number_input("Carbon Monoxide (CO)", value=100.0, help="Paper degradation.")
                co2 = st.number_input("Carbon Dioxide (CO2)", value=500.0)
                c2h4 = st.number_input("Ethylene (C2H4)", value=1.0, help="High-temp thermal fault.")
                c2h6 = st.number_input("Ethane (C2H6)", value=2.0)
                c2h2 = st.number_input("Acetylene (C2H2)", value=0.0, help="CRITICAL: Arcing indicator.")
                
                st.markdown("**2. Physical Properties:**")
                dbds = st.number_input("DBDS (ppm)", value=0.1, help="Corrosive sulfur.")
                pf = st.number_input("Power Factor (%)", value=0.05, help="Dielectric loss.")
                ift = st.number_input("Interfacial Tension (mN/m)", value=48.0, help="Oil purity.")
                dr = st.number_input("Dielectric Rigidity (kV)", value=65.0, help="Insulation strength.")
                wc = st.number_input("Water Content (ppm)", value=1.0, help="Moisture level.")
                
                submitted = st.form_submit_button("🔍 Run AI Analysis", use_container_width=True)

        with col_output:
            if submitted:
                # Prediction
                data = pd.DataFrame([[h2, o2, n2, ch4, co, co2, c2h4, c2h6, c2h2, dbds, pf, ift, dr, wc]], 
                                    columns=['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V', 'Dielectric rigidity', 'Water content'])
                
                h_score = model_health.predict(data)[0]
                l_yrs = model_life.predict(data)[0]
                diagnosis = get_duval_diagnosis(ch4, c2h4, c2h2)

                if h_score >= 70: status, color = "SAFE", "#00cc66"
                elif h_score >= 40: status, color = "WARNING", "#ffcc00"
                else: status, color = "RISKY", "#ff3333"

                # 1. Key Metrics
                st.markdown("### 📈 AI Analysis Results")
                m1, m2, m3 = st.columns(3)
                m1.metric("Health Score", f"{h_score:.2f}")
                m2.metric("Status", status)
                m3.metric("Life Expectancy", f"{l_yrs:.1f} Yrs")

                # 2. Gauge & Duval
                col_g, col_d = st.columns([1, 1.2])
                with col_g:
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number", value=h_score,
                        gauge={'axis': {'range': [100, 0]}, 'bar': {'color': "white"},
                               'steps': [{'range': [70, 100], 'color': "#00cc66"},
                                         {'range': [40, 70], 'color': "#ffcc00"},
                                         {'range': [0, 40], 'color': "#ff3333"}]}))
                    fig_g.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                    st.plotly_chart(fig_g, use_container_width=True)
                    st.markdown(f"<h2 style='text-align: center; color: {color};'>{status}</h2>", unsafe_allow_html=True)

                with col_d:
                    fig_tri = px.scatter_ternary(pd.DataFrame({'CH4':[ch4], 'C2H4':[c2h4], 'C2H2':[c2h2]}), a="CH4", b="C2H4", c="C2H2")
                    fig_tri.update_traces(marker=dict(size=12, color='red', symbol='cross'))
                    fig_tri.update_layout(title="Duval Triangle Analysis", height=300, paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                    st.plotly_chart(fig_tri, use_container_width=True)
                    st.info(f"**Diagnosis:** {diagnosis}")

                st.markdown("---")
                
                # 3. Dynamic Bar Chart (Gases Composition)
                st.markdown("#### 📊 Current Sample Gases Composition (%)")
                g_vals = [h2, ch4, c2h4, c2h6, c2h2, co, co2]
                g_names = ['H2', 'CH4', 'C2H4', 'C2H6', 'C2H2', 'CO', 'CO2']
                if sum(g_vals) > 0:
                    df_bar = pd.DataFrame({'Gas': g_names, 'Value (%)': [(v/sum(g_vals))*100 for v in g_vals]}).sort_values('Value (%)')
                    fig_b = px.bar(df_bar, x='Value (%)', y='Gas', orientation='h', text='Value (%)', color='Value (%)', color_continuous_scale='Blues')
                    fig_b.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig_b.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font={'color': 'white'}, coloraxis_showscale=False)
                    st.plotly_chart(fig_b, use_container_width=True)

    # --- TAB 2: BATCH ANALYSIS ---
    with tab2:
        st.markdown("### 📁 Batch Processing")
        up = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
        if up:
            df = pd.read_csv(up) if up.name.endswith('.csv') else pd.read_excel(up)
            req = ['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V', 'Dielectric rigidity', 'Water content']
            df['Predicted_Health'] = model_health.predict(df[req])
            st.markdown("#### Health Trend")
            st.line_chart(df['Predicted_Health'])
            st.dataframe(df)
            st.download_button("📥 Download Results", df.to_csv(index=False), "results.csv", "text/csv")
else:
    st.error("Please ensure model files (.pkl) are present in the directory.")