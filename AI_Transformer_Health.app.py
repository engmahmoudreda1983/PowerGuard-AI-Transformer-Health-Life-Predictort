import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime
import time 

# Page Configuration
st.set_page_config(page_title="PowerGuard AI", page_icon="⚡", layout="wide")

# ==========================================
# --- 1. نظام تسجيل الدخول (Login System) ---
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# تهيئة قاعدة بيانات سجل الـ FMEA في ذاكرة الجلسة
if 'fmea_log' not in st.session_state:
    st.session_state['fmea_log'] = pd.DataFrame(columns=['Timestamp', 'Asset ID', 'DGA Index', 'Pred Temp', 'Overall Risk (%)', 'Status', 'RPN Level', 'Action Required'])

if not st.session_state['logged_in']:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #4da6ff;'>⚡ PowerGuard AI Login</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.warning("🔒 Restricted Access. Authorized Personnel Only.")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Secure Login", use_container_width=True)
            
            if submit:
                if username == "admin" and password == "DBA2026": 
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("❌ Invalid Username or Password")
    
    st.stop()


# ==========================================
# --- 2. لوحة التحكم الرئيسية (Dashboard) ---
# ==========================================

col_l, col_r = st.columns([10, 1])
with col_r:
    if st.button("Logout 🚪"):
        st.session_state['logged_in'] = False
        st.rerun()

st.markdown("""
<div style='background-color: #0b2e59; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h1 style='color: #ffffff; text-align: center; margin: 0;'>AI Predictive Maintenance ⚡</h1>
    <h3 style='color: #4da6ff; text-align: center; margin: 5px 0 0 0;'>Power Transformers Fleet Management</h3>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        m_health = joblib.load('rf_health_model.pkl')
        m_life = joblib.load('rf_life_model.pkl')
        m_thermal = joblib.load('rf_thermal_model.pkl')
        return m_health, m_life, m_thermal
    except Exception as e:
        return None, None, None

model_health, model_life, model_thermal = load_models()

if model_health is None or model_thermal is None:
    st.error("⚠️ Model files not detected! Please ensure 'rf_health_model.pkl', 'rf_life_model.pkl', and 'rf_thermal_model.pkl' are uploaded to GitHub.")
    st.stop()

def get_model_confidence(model, input_df):
    try:
        preds = np.array([tree.predict(input_df.values)[0] for tree in model.estimators_])
        std = np.std(preds)
        conf = 98.5 - (std * 2.5)
        return max(65.0, min(99.0, conf))
    except:
        return 92.5  

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

# --- ستايل الأرقام الجديد (أبيض نقي وعريض بدون المربع المزعج) ---
lbl_style = dict(showarrow=False, font=dict(color="white", size=16))

tab1, tab2, tab3, tab4 = st.tabs(["🧪 DGA & Oil Quality", "🌡️ Real-Time SCADA", "📁 Batch Analysis", "📑 Executive Report & Logs"])

# --- TAB 1: DGA & OIL QUALITY ---
with tab1:
    col_input, col_output = st.columns([1, 2.2])
    with col_input:
        st.markdown("### 📊 DGA Parameters")
        with st.form("input_form"):
            st.markdown("**1. Dissolved Gases (ppm):**")
            h2 = st.number_input("Hydrogen (H2)", value=5.0, help="High levels indicate partial discharge or low energy faults.")
            o2 = st.number_input("Oxygen (O2)", value=500.0, help="Indicates leak in sealing or excessive paper aging.")
            n2 = st.number_input("Nitrogen (N2)", value=10000.0, help="Inert gas, usually part of atmospheric air or nitrogen blanket.")
            ch4 = st.number_input("Methane (CH4)", value=2.0, help="Indicates low temperature thermal faults (< 300°C).")
            co = st.number_input("Carbon Monoxide (CO)", value=100.0, help="Primary indicator of severe paper insulation degradation.")
            co2 = st.number_input("Carbon Dioxide (CO2)", value=300.0, help="Indicates normal paper aging, but high ratio to CO implies localized overheating.")
            c2h4 = st.number_input("Ethylene (C2H4)", value=1.0, help="Indicates high temperature thermal faults (> 700°C).")
            c2h6 = st.number_input("Ethane (C2H6)", value=5.0, help="Indicates moderate temperature thermal faults (300°C - 700°C).")
            c2h2 = st.number_input("Acetylene (C2H2)", value=0.0, help="CRITICAL: Indicates arcing (high energy discharge) and extreme localized heating.")
            
            st.markdown("**2. Physical Properties:**")
            dbds = st.number_input("DBDS (ppm)", value=0.1, help="Corrosive sulfur compound, harmful to copper winding.")
            pf = st.number_input("Power Factor (%)", value=0.05, help="Measure of dielectric loss in the oil. High value means contamination.")
            ift = st.number_input("Interfacial Tension (mN/m)", value=45.0, help="Indicates presence of polar contaminants or sludge. Low value is bad.")
            dr = st.number_input("Dielectric Rigidity (kV)", value=70.0, help="Breakdown voltage. The ability of oil to withstand electrical stress.")
            wc = st.number_input("Water Content (ppm)", value=2.0, help="Moisture in oil reduces dielectric strength and accelerates paper aging.")
            
            submitted = st.form_submit_button("🔍 Analyze Oil Health", use_container_width=True)

    with col_output:
        if submitted:
            input_df = pd.DataFrame([[h2, o2, n2, ch4, co, co2, c2h4, c2h6, c2h2, dbds, pf, ift, dr, wc]], 
                                    columns=['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V', 'Dielectric rigidity', 'Water content'])
            
            h_score = model_health.predict(input_df)[0]
            l_yrs = model_life.predict(input_df)[0]
            diagnosis = get_duval_diagnosis(ch4, c2h4, c2h2)

            if h_score <= 30: status, color = "SAFE", "#00cc66"
            elif h_score <= 45: status, color = "WARNING", "#ffcc00"
            else: status, color = "RISKY", "#ff3333"

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Degradation Index", f"{h_score:.2f}")
            m2.metric("Status", status)
            m3.metric("Life Expectancy", f"{l_yrs:.1f} Yrs")
            
            conf_h = get_model_confidence(model_health, input_df)
            m4.metric("AI Confidence", f"{conf_h:.1f} %")

            col_g, col_d = st.columns([1, 1.2])
            with col_g:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=h_score,
                    gauge={'axis': {'range': [0, 60], 'tickwidth': 1, 'showticklabels': False},
                           'bar': {'color': "white", 'thickness': 0.15},
                           'steps': [{'range': [0, 30], 'color': "#00cc66"},
                                     {'range': [30, 45], 'color': "#ffcc00"},
                                     {'range': [45, 60], 'color': "#ff3333"}]}))
                
                # إضافة الأرقام بتصميم جذاب وبدون مربعات
                fig_g.add_annotation(x=0.12, y=0.15, text="<b>0</b>", **lbl_style)
                fig_g.add_annotation(x=0.50, y=0.88, text="<b>30</b>", **lbl_style)
                fig_g.add_annotation(x=0.85, y=0.50, text="<b>45</b>", **lbl_style)
                fig_g.add_annotation(x=0.88, y=0.15, text="<b>60</b>", **lbl_style)

                fig_g.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                st.plotly_chart(fig_g, use_container_width=True)
                st.markdown(f"<h2 style='text-align: center; color: {color}; margin-top:-30px;'>{status}</h2>", unsafe_allow_html=True)

            with col_d:
                # --- حل مشكلة مثلث دوفال الثابت ---
                tot_gas = ch4 + c2h4 + c2h2
                if tot_gas > 0:
                    p_ch4, p_c2h4, p_c2h2 = (ch4/tot_gas)*100, (c2h4/tot_gas)*100, (c2h2/tot_gas)*100
                else:
                    p_ch4, p_c2h4, p_c2h2 = 33.33, 33.33, 33.33 # النقطة في المنتصف لو مفيش غازات
                
                df_tri = pd.DataFrame({'CH4 (%)':[p_ch4], 'C2H4 (%)':[p_c2h4], 'C2H2 (%)':[p_c2h2]})
                fig_tri = px.scatter_ternary(df_tri, a="CH4 (%)", b="C2H4 (%)", c="C2H2 (%)")
                fig_tri.update_traces(marker=dict(size=14, color='red', symbol='cross', line=dict(width=2, color='white')))
                
                # إجبار أبعاد المثلث من 0 إلى 100% ليتحرك المؤشر بشكل صحيح
                fig_tri.update_layout(
                    title="Duval Triangle Diagnostic", 
                    height=320, margin=dict(b=0), 
                    paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"},
                    ternary=dict(
                        sum=100,
                        aaxis=dict(min=0, title="CH4 %"),
                        baxis=dict(min=0, title="C2H4 %"),
                        caxis=dict(min=0, title="C2H2 %")
                    )
                )
                st.plotly_chart(fig_tri, use_container_width=True)
                st.info(f"**Duval Fault Diagnosis:** {diagnosis}")
            
            st.markdown("---")
            st.markdown("### 📊 Variables Impact (Feature Importance)")
            features_list = ['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V', 'Dielectric rigidity', 'Water content']
            importances = model_health.feature_importances_
            df_imp = pd.DataFrame({'Feature': features_list, 'Importance': importances}).sort_values(by='Importance', ascending=True)
            fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
            fig_imp.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)

# --- TAB 2: SCADA & THERMAL ---
with tab2:
    st.markdown("### 🌡️ Thermal Load Predictive Analysis")
    
    col_t_in, col_t_out = st.columns([1, 1.5])
    with col_t_in:
        with st.form("thermal_form"):
            st.markdown("**1. Time Features:**")
            c1, c2 = st.columns(2)
            hour = c1.number_input("Hour (0-23)", min_value=0, max_value=23, value=14, help="Hour of the day. Affects ambient temperature.")
            month = c2.number_input("Month (1-12)", min_value=1, max_value=12, value=7, help="Month of the year. Affects seasonal base temperature.")
            
            st.markdown("**2. High Voltage Load:**")
            hufl = st.number_input("HUFL (Active Load kW)", value=12.5, help="High Use Full Load: Active power demand on the high voltage side.")
            hull = st.number_input("HULL (Reactive Load kVAR)", value=3.2, help="High Use Low Load: Reactive power demand on the high voltage side.")
            
            st.markdown("**3. Medium Voltage Load:**")
            mufl = st.number_input("MUFL (Active Load kW)", value=8.1, help="Medium Use Full Load: Active power on the medium voltage side.")
            mull = st.number_input("MULL (Reactive Load kVAR)", value=1.5, help="Medium Use Low Load: Reactive power on the medium voltage side.")
            
            st.markdown("**4. Low Voltage Load:**")
            lufl = st.number_input("LUFL (Active Load kW)", value=4.5, help="Low Use Full Load: Active power on the low voltage side.")
            lull = st.number_input("LULL (Reactive Load kVAR)", value=0.8, help="Low Use Low Load: Reactive power on the low voltage side.")
            
            sub_thermal = st.form_submit_button("🌡️ Predict Oil Temp", use_container_width=True)
            
    with col_t_out:
        if sub_thermal:
            t_input = pd.DataFrame([[hufl, hull, mufl, mull, lufl, lull, hour, month]], 
                                   columns=['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'Hour', 'Month'])
            pred_ot = model_thermal.predict(t_input)[0]
            
            if pred_ot < 60: t_status, t_color, msg = "NORMAL", "#00cc66", "Temperature is within safe operating limits."
            elif pred_ot < 80: t_status, t_color, msg = "WARNING", "#ffcc00", "High load detected. Ensure cooling fans are active."
            else: t_status, t_color, msg = "RISKY", "#ff3333", "CRITICAL OVERHEATING! Reduce load immediately."
            
            tm1, tm2, tm3 = st.columns(3)
            tm1.metric("Predicted Top Oil Temp", f"{pred_ot:.1f} °C")
            tm2.metric("Thermal Status", t_status)
            
            conf_t = get_model_confidence(model_thermal, t_input)
            tm3.metric("AI Confidence", f"{conf_t:.1f} %")
            
            fig_t = go.Figure(go.Indicator(
                mode = "gauge+number", value = pred_ot, title = {'text': "Oil Temp °C"},
                gauge = {'axis': {'range': [20, 120], 'tickwidth': 1, 'showticklabels': False},
                         'bar': {'color': t_color},
                         'steps': [{'range': [20, 60], 'color': "rgba(0, 204, 102, 0.4)"},
                                   {'range': [60, 80], 'color': "rgba(255, 204, 0, 0.4)"},
                                   {'range': [80, 120], 'color': "rgba(255, 51, 51, 0.4)"}],
                         'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': pred_ot}}))
            
            # إضافة الأرقام بتصميم جذاب بدون مربعات
            fig_t.add_annotation(x=0.12, y=0.15, text="<b>20</b>", **lbl_style)
            fig_t.add_annotation(x=0.35, y=0.75, text="<b>60</b>", **lbl_style)
            fig_t.add_annotation(x=0.65, y=0.75, text="<b>80</b>", **lbl_style)
            fig_t.add_annotation(x=0.88, y=0.15, text="<b>120</b>", **lbl_style)

            fig_t.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_t, use_container_width=True)
            
            st.markdown(f"<h2 style='text-align: center; color: {t_color}; margin-top:-30px;'>{t_status}</h2>", unsafe_allow_html=True)
            st.info(f"💡 **AI Recommendation:** {msg}")
            
            st.markdown("---")
            st.markdown("### 📊 Variables Impact (Thermal Features)")
            t_features_list = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'Hour', 'Month']
            t_importances = model_thermal.feature_importances_
            df_t_imp = pd.DataFrame({'Feature': t_features_list, 'Importance': t_importances}).sort_values(by='Importance', ascending=True)
            fig_t_imp = px.bar(df_t_imp, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Reds')
            fig_t_imp.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, coloraxis_showscale=False)
            st.plotly_chart(fig_t_imp, use_container_width=True)

# --- TAB 3: BATCH ANALYSIS ---
with tab3:
    st.markdown("### 📁 Batch Historical Analysis")
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
            else:
                st.error(f"⚠️ Missing required columns.")
        elif "SCADA" in analysis_type:
            req_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'Hour', 'Month']
            if all(c in df_b.columns for c in req_cols):
                df_b['Predicted_Oil_Temp'] = model_thermal.predict(df_b[req_cols])
                fig_l = px.line(df_b, y='Predicted_Oil_Temp', markers=True, title="Predicted Oil Temperature (°C)", color_discrete_sequence=['#ff3333'])
                st.plotly_chart(fig_l, use_container_width=True)
            else:
                st.error(f"⚠️ Missing required columns.")

# --- TAB 4: EXECUTIVE REPORT & FMEA LOG ---
with tab4:
    st.markdown("### 📑 Overall Transformer Risk Assessment (AHI)")
    
    col_ex_in, col_ex_out = st.columns([1, 1.5])
    
    with col_ex_in:
        with st.form("executive_form"):
            st.markdown("**🔍 Asset Identification:**")
            tx_list = [f"T{i:02d} (Substation Unit)" for i in range(1, 11)]
            asset_selection = st.selectbox("Select Transformer:", tx_list)
            st.markdown("---")
            
            st.markdown("**1. Key Fault Gases (ppm):**")
            c_g1, c_g2 = st.columns(2)
            e_h2 = c_g1.number_input("Hydrogen (H2)", value=5.0)
            e_ch4 = c_g2.number_input("Methane (CH4)", value=2.0)
            e_c2h4 = c_g1.number_input("Ethylene (C2H4)", value=1.0)
            e_c2h2 = c_g2.number_input("Acetylene (C2H2)", value=0.0)
            
            st.markdown("**2. Paper Degradation Gases (ppm):**")
            c_p1, c_p2 = st.columns(2)
            e_co = c_p1.number_input("Carbon Monoxide (CO)", value=100.0)
            e_co2 = c_p2.number_input("Carbon Dioxide (CO2)", value=300.0)
            
            st.markdown("**3. Physical Properties (CRITICAL):**")
            c_ph1, c_ph2 = st.columns(2)
            e_dr = c_ph1.number_input("Dielectric Rigidity (kV)", value=70.0)
            e_wc = c_ph2.number_input("Water Content (ppm)", value=2.0)
            e_pf = st.number_input("Power Factor (%)", value=0.05)
            
            st.markdown("**4. Operating & Ambient Conditions:**")
            c_o1, c_o2 = st.columns(2)
            e_hufl = c_o1.number_input("HUFL (Load kW)", value=12.0)
            e_hour = c_o2.number_input("Hour (0-23)", value=5)
            e_month = st.number_input("Month (1-12)", value=1)
            
            gen_report = st.form_submit_button("📊 Evaluate & Push to CMMS", use_container_width=True)
            
    with col_ex_out:
        if gen_report:
            asset_id = asset_selection.split(' ')[0] 
            
            dga_input = pd.DataFrame([[e_h2, 500, 10000, e_ch4, e_co, e_co2, e_c2h4, 5, e_c2h2, 0.1, e_pf, 45, e_dr, e_wc]], 
                                    columns=['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 'Interfacial V', 'Dielectric rigidity', 'Water content'])
            health_score = model_health.predict(dga_input)[0]
            dga_risk = min(100, max(0, ((health_score - 20) / 30) * 100)) 
            
            scada_input = pd.DataFrame([[e_hufl, e_hufl*0.2, e_hufl*0.6, e_hufl*0.1, e_hufl*0.3, e_hufl*0.05, e_hour, e_month]], 
                                   columns=['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'Hour', 'Month'])
            ot_temp = model_thermal.predict(scada_input)[0]
            thermal_risk = min(100, max(0, ((ot_temp - 40) / 45) * 100))
            overall_risk = (dga_risk * 0.6) + (thermal_risk * 0.4)
            
            if overall_risk <= 35: 
                final_status, final_color = "🟢 EXCELLENT (Low Risk)", "#00cc66"
                final_action = "Continue normal operation. Next DGA in 6 months."
                rpn_level = "Low (16-48)"
            elif overall_risk <= 70: 
                final_status, final_color = "🟡 WATCH (Medium Risk)", "#ffcc00"
                final_action = "Schedule PM. Reduce load and check cooling."
                rpn_level = "Medium (72-126)"
            else: 
                final_status, final_color = "🔴 ACTION REQUIRED (High Risk)", "#ff3333"
                final_action = "URGENT: Isolate transformer. Initiate emergency FMEA."
                rpn_level = "Critical (160-200)"
            
            new_entry = pd.DataFrame([{
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Asset ID': asset_id, 
                'DGA Index': round(health_score, 1),
                'Pred Temp': round(ot_temp, 1),
                'Overall Risk (%)': round(overall_risk, 1),
                'Status': final_status.split(' ')[1],
                'RPN Level': rpn_level,
                'Action Required': final_action
            }])
            st.session_state['fmea_log'] = pd.concat([st.session_state['fmea_log'], new_entry], ignore_index=True).tail(100)
            
            st.markdown(f"### 📋 Executive Summary: {asset_selection}")
            st.markdown(f"<h2 style='text-align: center; color: {final_color}; border: 2px solid {final_color}; padding: 10px; border-radius: 5px;'>{final_status}</h2>", unsafe_allow_html=True)
            st.markdown("---")
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Degradation Index", f"{health_score:.1f}/60")
            rc2.metric("Predicted Temp", f"{ot_temp:.1f} °C")
            rc3.metric("OVERALL RISK", f"{overall_risk:.1f} %")
            
            conf_h = get_model_confidence(model_health, dga_input)
            conf_t = get_model_confidence(model_thermal, scada_input)
            overall_conf = (conf_h + conf_t) / 2
            rc4.metric("AI Confidence", f"{overall_conf:.1f} %")
            
            fig_risk = go.Figure(go.Indicator(
                mode = "gauge+number", value = overall_risk, title = {'text': "Asset Risk Level %"},
                gauge = {'axis': {'range': [0, 100], 'tickwidth': 1, 'showticklabels': False},
                         'bar': {'color': final_color},
                         'steps': [{'range': [0, 35], 'color': "rgba(0, 204, 102, 0.4)"},
                                   {'range': [35, 70], 'color': "rgba(255, 204, 0, 0.4)"},
                                   {'range': [70, 100], 'color': "rgba(255, 51, 51, 0.4)"}],
                         'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': overall_risk}}))
            
            # إضافة الأرقام بتصميم جذاب بدون مربعات
            fig_risk.add_annotation(x=0.12, y=0.15, text="<b>0</b>", **lbl_style)
            fig_risk.add_annotation(x=0.32, y=0.75, text="<b>35</b>", **lbl_style)
            fig_risk.add_annotation(x=0.68, y=0.75, text="<b>70</b>", **lbl_style)
            fig_risk.add_annotation(x=0.88, y=0.15, text="<b>100</b>", **lbl_style)

            fig_risk.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_risk, use_container_width=True)
            
            st.warning(f"**Asset Manager Action:** {final_action}")

            # --- API Simulation ---
            with st.spinner("Pushing Data to Central Database & CMMS..."):
                time.sleep(1.5)
                
            if overall_risk > 70:
                st.error("🚨 **CRITICAL ALARM TRIGGERED:** API Request sent to SCADA to prepare for automated load shedding!")
                st.error(f"📧 **EMAIL ALERT SENT:** Notification dispatched to Plant Manager for {asset_id}.")
                st.success("✅ Log successfully saved to Cloud Database.")
            elif overall_risk > 35:
                st.warning("⚠️ **PREVENTIVE WORK ORDER CREATED:** API Request sent to SAP CMMS to schedule inspection.")
                st.success("✅ Log successfully saved to Cloud Database.")
            else:
                st.success("✅ Data synchronized with Cloud Database. No immediate action required.")

    # --- FMEA Log ---
    st.markdown("---")
    st.markdown("### 📋 Dynamic Risk Logs & FMEA Tracker")
    st.info("This log is automatically updated with every evaluation. It acts as a feed for the CMMS system.")
    
    if not st.session_state['fmea_log'].empty:
        def color_risk(val):
            color = '#00cc66' if val == 'EXCELLENT' else '#ffcc00' if val == 'WATCH' else '#ff3333'
            return f'color: {color}; font-weight: bold'
        
        try:
            styled_df = st.session_state['fmea_log'].style.map(color_risk, subset=['Status'])
        except:
            styled_df = st.session_state['fmea_log'].style.applymap(color_risk, subset=['Status'])
            
        st.dataframe(styled_df, use_container_width=True)
        
        csv = st.session_state['fmea_log'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Fleet FMEA Log (CSV)",
            data=csv,
            file_name='Dynamic_FMEA_Risk_Logs.csv',
            mime='text/csv',
        )
    else:
        st.warning("No records yet. Please generate a report to create an automatic FMEA entry.")