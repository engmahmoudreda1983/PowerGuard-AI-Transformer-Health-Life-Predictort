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
    st.error("⚠️ Model files not found. Please ensure `rf_health_model.pkl` and `rf_life_model.pkl` are in the main directory.")
    models_loaded = False

# Duval Triangle Logic
def get_duval_diagnosis(ch4, c2h4, c2h2):
    total = ch4 + c2h4 + c2h2
    if total == 0:
        return "Normal (No significant fault gases)"
    
    p_ch4 = (ch4 / total) * 100
    p_c2h4 = (c2h4 / total) * 100
    p_c2h2 = (c2h2 / total) * 100
    
    if p_ch4 >= 98: return "PD (Partial Discharge)"
    if p_c2h2 < 4 and p_c2h4 < 20: return "T1 (Thermal Fault < 300°C)"
    if p_c2h2 < 4 and 20 <= p_c2h4 < 50: return "T2 (Thermal Fault 300-700°C)"
    if p_c2h2 < 15 and p_c2h4 >= 50: return "T3 (Thermal Fault > 700°C)"
    if 4 <= p_c2h2 <= 13 and p_c2h4 < 50: return "D1 (Low Energy Discharge / Arcing)"
    if p_c2h2 > 13 or (p_c2h2 > 4 and p_c2h4 >= 50): return "D2 (High Energy Discharge / Arcing)"
    return "DT (Mix of Thermal and Electrical Faults)"

if models_loaded:
    # --- TABS FOR INPUT METHODS ---
    tab1, tab2 = st.tabs(["✍️ Manual Entry (Single Test)", "📁 Batch Analysis (CSV/Excel Upload)"])

    # ==========================================
    # TAB 1: MANUAL ENTRY
    # ==========================================
    with tab1:
        col_input, col_output = st.columns([1, 2])

        with col_input:
            st.markdown("### 📊 Parameters (Inputs)")
            with st.form("input_form"):
                st.markdown("**1. Dissolved Gases (DGA - ppm):**")
                hydrogen = st.number_input("Hydrogen (H2)", value=100.0, step=10.0)
                oxigen = st.number_input("Oxygen (O2)", value=5000.0, step=100.0)
                nitrogen = st.number_input("Nitrogen (N2)", value=40000.0, step=1000.0)
                methane = st.number_input("Methane (CH4)", value=50.0, step=5.0)
                co = st.number_input("Carbon Monoxide (CO)", value=200.0, step=10.0)
                co2 = st.number_input("Carbon Dioxide (CO2)", value=1500.0, step=50.0)
                ethylene = st.number_input("Ethylene (C2H4)", value=10.0, step=2.0)
                ethane = st.number_input("Ethane (C2H6)", value=20.0, step=5.0)
                acethylene = st.number_input("Acetylene (C2H2)", value=0.0, step=1.0)

                st.markdown("---")
                st.markdown("**2. Oil Physical Properties:**")
                dbds = st.number_input("DBDS (ppm)", value=10.0, step=1.0)
                power_factor = st.number_input("Power Factor (%)", value=1.5, step=0.1)
                interfacial_v = st.number_input("Interfacial Tension (mN/m)", value=40.0, step=1.0)
                dielectric_rigidity = st.number_input("Dielectric Rigidity (kV)", value=50.0, step=1.0)
                water_content = st.number_input("Water Content (ppm)", value=15.0, step=1.0)
                
                submitted = st.form_submit_button("🔍 Run AI Analysis", use_container_width=True)

        with col_output:
            if submitted:
                # 1. AI Predictions
                input_data = pd.DataFrame({
                    'Hydrogen': [hydrogen], 'Oxigen': [oxigen], 'Nitrogen': [nitrogen], 
                    'Methane': [methane], 'CO': [co], 'CO2': [co2], 
                    'Ethylene': [ethylene], 'Ethane': [ethane], 'Acethylene': [acethylene], 
                    'DBDS': [dbds], 'Power factor': [power_factor], 
                    'Interfacial V': [interfacial_v], 'Dielectric rigidity': [dielectric_rigidity], 
                    'Water content': [water_content]
                })

                health_index_pred = model_health.predict(input_data)[0]
                life_expectation_pred = model_life.predict(input_data)[0]
                duval_fault = get_duval_diagnosis(methane, ethylene, acethylene)

                if health_index_pred >= 70:
                    status_text, color, rec = "SAFE", "#00cc66", "Transformer is in good condition."
                elif health_index_pred >= 40:
                    status_text, color, rec = "WARNING", "#ffcc00", "Oil degradation detected. Schedule inspection."
                else:
                    status_text, color, rec = "RISKY", "#ff3333", "Severe degradation! Immediate intervention required."

                # 2. Key Metrics
                st.markdown("### 📈 AI Analysis Results")
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric(label="Health Score", value=f"{health_index_pred:.2f}")
                kpi2.metric(label="Status", value=status_text)
                kpi3.metric(label="Life Expectancy", value=f"{life_expectation_pred:.1f} Years")
                st.markdown("---")

                # 3. Gauges & Duval Triangle
                col_gauge, col_duval = st.columns([1, 1.2])

                with col_gauge:
                    # Health Index Gauge
                    fig_health = go.Figure(go.Indicator(
                        mode = "gauge",
                        value = health_index_pred,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                            'bar': {'color': "white", 'thickness': 0.1},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 2, 'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 40], 'color': "#ff3333"},
                                {'range': [40, 70], 'color': "#ffcc00"},
                                {'range': [70, 100], 'color': "#00cc66"}
                            ],
                            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': health_index_pred}
                        }
                    ))
                    fig_health.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=220, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_health, use_container_width=True)
                    st.markdown(f"<h3 style='text-align: center; color: {color}; margin-top: 0px;'>{status_text}</h3>", unsafe_allow_html=True)

                with col_duval:
                    # Duval Triangle Plot
                    df_duval = pd.DataFrame({'CH4': [methane], 'C2H4': [ethylene], 'C2H2': [acethylene]})
                    fig_duval = px.scatter_ternary(df_duval, a="CH4", b="C2H4", c="C2H2")
                    fig_duval.update_traces(marker=dict(size=12, color='red', symbol='cross'))
                    fig_duval.update_layout(
                        title="Duval Triangle 1 Projection",
                        ternary=dict(sum=100, aaxis_title="CH4 %", baxis_title="C2H4 %", caxis_title="C2H2 %"),
                        paper_bgcolor="rgba(0,0,0,0)", font={'color': 'white'}, height=300, margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig_duval, use_container_width=True)
                    st.info(f"**Duval Diagnosis:** {duval_fault}")

            else:
                st.info("👈 Enter the readings in the left panel and click 'Run AI Analysis'.")

    # ==========================================
    # TAB 2: BATCH ANALYSIS (FILE UPLOAD)
    # ==========================================
    with tab2:
        st.markdown("### 📁 Upload Historical Data (CSV or Excel)")
        st.markdown("Upload a file containing historical readings to analyze trends and predict future health.")
        
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_batch = pd.read_csv(uploaded_file)
                else:
                    df_batch = pd.read_excel(uploaded_file)
                
                # Verify columns
                expected_cols = ['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 
                                 'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor', 
                                 'Interfacial V', 'Dielectric rigidity', 'Water content']
                
                missing_cols = [col for col in expected_cols if col not in df_batch.columns]
                
                if len(missing_cols) > 0:
                    st.error(f"⚠️ Missing columns in uploaded file: {', '.join(missing_cols)}")
                else:
                    st.success("✅ File loaded successfully. Running AI Batch Analysis...")
                    
                    # Predictions
                    df_batch['Predicted_Health_Index'] = model_health.predict(df_batch[expected_cols])
                    df_batch['Predicted_Life_Years'] = model_life.predict(df_batch[expected_cols])
                    
                    # Display Trend Chart
                    st.markdown("#### 📉 Health Index Trend Over Time")
                    # Assuming index represents time/samples
                    fig_trend = px.line(df_batch, y='Predicted_Health_Index', markers=True, 
                                        title="Transformer Health Degradation Trend")
                    
                    # Add Risk Zones to background
                    fig_trend.add_hrect(y0=0, y1=40, fillcolor="red", opacity=0.2, layer="below", line_width=0)
                    fig_trend.add_hrect(y0=40, y1=70, fillcolor="orange", opacity=0.2, layer="below", line_width=0)
                    fig_trend.add_hrect(y0=70, y1=100, fillcolor="green", opacity=0.2, layer="below", line_width=0)
                    
                    fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                    st.plotly_chart(fig_trend, use_container_width=True)

                    # Display Dataframe
                    st.markdown("#### 📄 Detailed Predictions Table")
                    st.dataframe(df_batch[expected_cols + ['Predicted_Health_Index', 'Predicted_Life_Years']].style.highlight_min(subset=['Predicted_Health_Index'], color='red'))
                    
            except Exception as e:
                st.error(f"Error reading file: {e}")