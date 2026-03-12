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
    try:
        model_health = joblib.load('rf_health_model.pkl')
        model_life = joblib.load('rf_life_model.pkl')
        model_thermal = joblib.load('rf_thermal_model.pkl')
        return model_health, model_life, model_thermal
    except Exception as e:
        st.error(f"⚠️ Model Error: {e}")
        return None, None, None

model_health, model_life, model_thermal = load_models()

# Duval Triangle Diagnosis Logic
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

if model_health is not None and model_thermal is not None:
    tab1, tab2, tab3 = st.tabs(["🧪 DGA & Oil Quality", "🌡️ Real-Time SCADA (Thermal)", "📁 Batch Analysis"])

    # --- TAB 1: DGA & OIL QUALITY ---
    with tab1:
        col_input, col_output = st.columns([1, 2.2])
        with col_input:
            st.markdown("### 📊 DGA Parameters")
            with st.form("input_form"):
                st.markdown("**1. Dissolved Gases (ppm):**")
                h2 = st.number_input("Hydrogen (H2)", value=10.0)
                o2 = st.number_input("Oxygen (O2)", value=500.0)
                n2 = st.number_input("Nitrogen (N2)", value=10000.0)
                ch4 = st.number_input("Methane (CH4)", value=10.0)
                co = st.number_input("Carbon Monoxide (CO)", value=100.0)
                co2 = st.number_input("Carbon Dioxide (CO2)", value=500.0)
                c2h4 = st.number_input("Ethylene (C2H4)", value=2.0)
                c2h6 = st.number_input("Ethane (C2H6)", value=5.0)
                c2h2 = st.number_input("Acetylene (C2H2)", value=0.0)
                
                st.markdown("**2. Physical Properties:**")
                dbds = st.number_input("DBDS (ppm)", value=0.1)
                pf = st.number_input("Power Factor (%)", value=0.1)
                ift = st.number_input("Interfacial Tension (mN/m)", value=45.0)
                dr = st.number_input("Dielectric Rigidity (kV)", value=60.0)
                wc = st.number_input("Water Content (ppm)", value=2.0)
                
                submitted = st.form_submit_button("🔍 Analyze Oil Health", use_container_width=True)

        with col_output:
            if submitted:
                input_df = pd.DataFrame([[h2, o2, n2, ch4, co, co2, c2h4, c2h6, c2h2, dbds, pf, ift, dr, wc]], 
                                        columns=['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V', 'Dielectric rigidity', 'Water content'])
                
                h_score = model_health.predict(input_df)[0]
                l_yrs = model_life.predict(input_df)[0]
                diagnosis = get_duval_diagnosis(ch4, c2h4, c2h2)

                if h_score >= 70: status, color = "SAFE", "#00cc66"
                elif h_score >= 40: status, color = "WARNING", "#ffcc00"
                else: status, color = "RISKY", "#ff3333"

                m1, m2, m3 = st.columns(3)
                m1.metric("Health Score", f"{h_score:.2f}")
                m2.metric("Status", status)
                m3.metric("Life Expectancy", f"{l_yrs:.1f} Yrs")

                col_g, col_d = st.columns([1, 1.2])
                with col_g:
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number", value=h_score,
                        gauge={'axis': {'range': [100, 0], 'tickwidth': 1},
                               'bar': {'color': "white", 'thickness': 0.15},
                               'steps': [{'range': [70, 100], 'color': "#00cc66"},
                                         {'range': [40, 70], 'color': "#ffcc00"},
                                         {'range': [0, 40], 'color': "#ff3333"}]}))
                    fig_g.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                    st.plotly_chart(fig_g, use_container_width=True)
                    st.markdown(f"<h2 style='text-align: center; color: {color}; margin-top:-30px;'>{status}</h2>", unsafe_allow_html=True)

                with col_d:
                    df_tri = pd.DataFrame({'CH4':[ch4], 'C2H4':[c2h4], 'C2H2':[c2h2]})
                    fig_tri = px.scatter_ternary(df_tri, a="CH4", b="C2H4", c="C2H2")
                    fig_tri.update_traces(marker=dict(size=14, color='red', symbol='cross', line=dict(width=2, color='white')))
                    fig_tri.update_layout(title="Duval Triangle Diagnostic", height=320, paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                    st.plotly_chart(fig_tri, use_container_width=True)
                    st.info(f"**Diagnosis:** {diagnosis}")

                # --- رسمة تأثير المتغيرات (Feature Importance) للغازات ---
                st.markdown("---")
                st.markdown("### 📊 Variables Impact (Feature Importance)")
                features_list = ['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V', 'Dielectric rigidity', 'Water content']
                importances = model_health.feature_importances_
                df_imp = pd.DataFrame({'Feature': features_list, 'Importance': importances}).sort_values(by='Importance', ascending=True)
                fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
                fig_imp.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, coloraxis_showscale=False)
                st.plotly_chart(fig_imp, use_container_width=True)

    # --- TAB 2: SCADA & THERMAL ---
    with tab2:
        st.markdown("### 🌡️ Thermal Load Predictive Analysis (Virtual Sensor)")
        st.info("Predicts Transformer Top Oil Temperature (OT) based on active/reactive loads and time.")
        
        col_t_in, col_t_out = st.columns([1, 1.5])
        with col_t_in:
            with st.form("thermal_form"):
                st.markdown("**1. Time Features:**")
                c1, c2 = st.columns(2)
                hour = c1.number_input("Hour (0-23)", min_value=0, max_value=23, value=14)
                month = c2.number_input("Month (1-12)", min_value=1, max_value=12, value=7)
                
                st.markdown("**2. High Voltage Load:**")
                hufl = st.number_input("HUFL (Active Load kW)", value=12.5)
                hull = st.number_input("HULL (Reactive Load kVAR)", value=3.2)
                
                st.markdown("**3. Medium Voltage Load:**")
                mufl = st.number_input("MUFL (Active Load kW)", value=8.1)
                mull = st.number_input("MULL (Reactive Load kVAR)", value=1.5)
                
                st.markdown("**4. Low Voltage Load:**")
                lufl = st.number_input("LUFL (Active Load kW)", value=4.5)
                lull = st.number_input("LULL (Reactive Load kVAR)", value=0.8)
                
                sub_thermal = st.form_submit_button("🌡️ Predict Oil Temp", use_container_width=True)
                
        with col_t_out:
            if sub_thermal:
                t_input = pd.DataFrame([[hufl, hull, mufl, mull, lufl, lull, hour, month]], 
                                       columns=['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'Hour', 'Month'])
                pred_ot = model_thermal.predict(t_input)[0]
                
                if pred_ot < 50: t_status, t_color, msg = "NORMAL", "#00cc66", "Temperature is within safe operating limits."
                elif pred_ot < 70: t_status, t_color, msg = "WARNING", "#ffcc00", "High load detected. Monitor cooling fans."
                else: t_status, t_color, msg = "CRITICAL", "#ff3333", "OVERHEATING RISK! Reduce load immediately."
                
                st.markdown("### 📈 Real-Time AI Thermal Prediction")
                tm1, tm2 = st.columns(2)
                tm1.metric("Predicted Top Oil Temp", f"{pred_ot:.1f} °C")
                tm2.metric("Thermal Status", t_status)
                
                fig_t = go.Figure(go.Indicator(
                    mode = "gauge+number", value = pred_ot, title = {'text': "Oil Temp °C"},
                    gauge = {'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                             'bar': {'color': t_color},
                             'steps': [{'range': [0, 50], 'color': "rgba(0, 204, 102, 0.4)"},
                                       {'range': [50, 70], 'color': "rgba(255, 204, 0, 0.4)"},
                                       {'range': [70, 100], 'color': "rgba(255, 51, 51, 0.4)"}],
                             'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': pred_ot}}))
                fig_t.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                st.plotly_chart(fig_t, use_container_width=True)
                st.info(f"💡 **AI Recommendation:** {msg}")

                # --- رسمة تأثير المتغيرات (Feature Importance) للحرارة ---
                st.markdown("---")
                st.markdown("### 📊 Variables Impact (Thermal Features)")
                t_features_list = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'Hour', 'Month']
                t_importances = model_thermal.feature_importances_
                df_t_imp = pd.DataFrame({'Feature': t_features_list, 'Importance': t_importances}).sort_values(by='Importance', ascending=True)
                
                # استخدمنا لون أحمر ليتناسب مع فكرة الحرارة
                fig_t_imp = px.bar(df_t_imp, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Reds')
                fig_t_imp.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, coloraxis_showscale=False)
                st.plotly_chart(fig_t_imp, use_container_width=True)

    # --- TAB 3: BATCH ANALYSIS (تحديث لدعم الموديلين) ---
    with tab3:
        st.markdown("### 📁 Batch Historical Analysis")
        st.write("Upload a CSV/Excel file to process multiple samples at once.")
        
        # إضافة زرار اختيار لنوع التحليل
        analysis_type = st.radio("Select Analysis Type:", ["🧪 DGA & Oil Quality (Health Index)", "🌡️ SCADA & Thermal (Oil Temp)"])
        
        up = st.file_uploader("Upload File", type=['csv', 'xlsx'])
        if up:
            df_b = pd.read_csv(up) if up.name.endswith('.csv') else pd.read_excel(up)
            
            if "DGA" in analysis_type:
                req_cols = ['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V', 'Dielectric rigidity', 'Water content']
                if all(c in df_b.columns for c in req_cols):
                    df_b['Predicted_Health'] = model_health.predict(df_b[req_cols])
                    fig_l = px.line(df_b, y='Predicted_Health', markers=True, title="Transformer Health Index over Samples")
                    st.plotly_chart(fig_l, use_container_width=True)
                    st.dataframe(df_b[['Predicted_Health'] + req_cols].head()) # عرض جزء من الجدول
                else:
                    st.error(f"⚠️ Missing required columns for DGA Analysis. Expected: {', '.join(req_cols)}")
                    
            elif "SCADA" in analysis_type:
                req_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'Hour', 'Month']
                if all(c in df_b.columns for c in req_cols):
                    df_b['Predicted_Oil_Temp'] = model_thermal.predict(df_b[req_cols])
                    # رسم بياني باللون الأحمر عشان الحرارة
                    fig_l = px.line(df_b, y='Predicted_Oil_Temp', markers=True, title="Predicted Oil Temperature (°C) over Time", color_discrete_sequence=['#ff3333'])
                    st.plotly_chart(fig_l, use_container_width=True)
                    st.dataframe(df_b[['Predicted_Oil_Temp'] + req_cols].head()) # عرض جزء من الجدول
                else:
                    st.error(f"⚠️ Missing required columns for Thermal Analysis. Expected: {', '.join(req_cols)}")

else:
    st.error("⚠️ Model files not detected! Please ensure 'rf_health_model.pkl', 'rf_life_model.pkl', and 'rf_thermal_model.pkl' are uploaded to GitHub.")