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
<div style='background-color: #0b2e59; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
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

if models_loaded:
    # Screen Layout
    col_input, col_output = st.columns([1, 1.2])

    with col_input:
        st.markdown("### 📊 Transformer Parameters (Inputs)")
        st.markdown("Enter the chemical and physical test results of the oil:")
        
        with st.form("input_form"):
            col1, col2 = st.columns(2)
            
            # Key Gases
            with col1:
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

            # Physical Properties
            with col2:
                st.markdown("**2. Oil Physical Properties:**")
                dbds = st.number_input("DBDS (ppm)", value=10.0, step=1.0)
                power_factor = st.number_input("Power Factor (%)", value=1.5, step=0.1)
                interfacial_v = st.number_input("Interfacial Tension (mN/m)", value=40.0, step=1.0)
                dielectric_rigidity = st.number_input("Dielectric Rigidity (kV)", value=50.0, step=1.0)
                water_content = st.number_input("Water Content (ppm)", value=15.0, step=1.0)
            
            submitted = st.form_submit_button("🔍 Run AI Analysis", use_container_width=True)

    with col_output:
        st.markdown("### 📈 AI Analysis Results")
        
        if submitted:
            # Prepare Inputs
            input_data = pd.DataFrame({
                'Hydrogen': [hydrogen], 'Oxigen': [oxigen], 'Nitrogen': [nitrogen], 
                'Methane': [methane], 'CO': [co], 'CO2': [co2], 
                'Ethylene': [ethylene], 'Ethane': [ethane], 'Acethylene': [acethylene], 
                'DBDS': [dbds], 'Power factor': [power_factor], 
                'Interfacial V': [interfacial_v], 'Dielectric rigidity': [dielectric_rigidity], 
                'Water content': [water_content]
            })

            # Predictions
            health_index_pred = model_health.predict(input_data)[0]
            life_expectation_pred = model_life.predict(input_data)[0]

            # Define Status
            if health_index_pred >= 70:
                status_text, color, rec = "SAFE", "#00cc66", "Transformer is in good condition. Routine maintenance recommended."
            elif health_index_pred >= 40:
                status_text, color, rec = "WARNING", "#ffcc00", "Oil degradation detected. Schedule a comprehensive inspection."
            else:
                status_text, color, rec = "RISKY", "#ff3333", "Severe degradation! High risk of failure. Immediate intervention required."

            # --- 1. Key Numbers (Metrics) ---
            st.markdown("#### 🔑 Key Summary")
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric(label="Health Score", value=f"{health_index_pred:.2f}")
            kpi2.metric(label="Status", value=status_text)
            kpi3.metric(label="Life Expectancy", value=f"{life_expectation_pred:.1f} Years")
            st.markdown("---")

            # --- 2. Health Index Gauge ---
            fig_health = go.Figure(go.Indicator(
                mode = "gauge",
                value = health_index_pred,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "white", 'thickness': 0.1},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': "#ff3333"},
                        {'range': [40, 70], 'color': "#ffcc00"},
                        {'range': [70, 100], 'color': "#00cc66"}
                    ],
                    'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': health_index_pred}
                }
            ))
            
            fig_health.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=200, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_health, use_container_width=True)

            # --- 3. Explicit Text Under Gauge ---
            st.markdown(f"<h3 style='text-align: center; margin-bottom: 0px;'>Health Index: {health_index_pred:.2f}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; color: {color}; margin-top: 0px; font-weight: bold;'>{status_text}</h1>", unsafe_allow_html=True)
            st.info(f"**Recommendation:** {rec}")
            st.markdown("---")

            # --- 4. Feature Importance Bar Chart ---
            st.markdown("#### 📊 Variables Impact on Transformer Health")
            
            # Extract feature importances from the model
            importances = model_health.feature_importances_ * 100
            df_imp = pd.DataFrame({'Feature': input_data.columns, 'Impact (%)': importances})
            df_imp = df_imp.sort_values(by='Impact (%)', ascending=True)

            # Create Plotly Bar Chart
            fig_bar = px.bar(df_imp, x='Impact (%)', y='Feature', orientation='h',
                             text='Impact (%)', color='Impact (%)', color_continuous_scale='Blues')
            
            fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_bar.update_layout(height=400, margin=dict(l=0, r=20, t=30, b=20), 
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
                                  font={'color': 'white'})
            
            st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.info("👈 Enter the readings in the table and click 'Run AI Analysis'.")