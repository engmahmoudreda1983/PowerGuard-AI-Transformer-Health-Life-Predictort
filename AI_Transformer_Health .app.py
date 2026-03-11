import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# إعداد الصفحة
st.set_page_config(page_title="Transformer Predictive Maintenance", page_icon="⚡", layout="wide")

# تصميم الهيدر
st.markdown("""
<div style='background-color: #0b2e59; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
    <h1 style='color: #ffffff; text-align: center; margin: 0;'>Deep Learning Predictive Maintenance ⚡</h1>
    <h3 style='color: #4da6ff; text-align: center; margin: 5px 0 0 0;'>Power Transformers DGA & Health Index Assessment</h3>
</div>
""", unsafe_allow_html=True)

# تحميل النماذج
@st.cache_resource
def load_models():
    # تأكد من أن الملفات في نفس المجلد
    model_health = joblib.load('rf_health_model.pkl')
    model_life = joblib.load('rf_life_model.pkl')
    return model_health, model_life

try:
    model_health, model_life = load_models()
    models_loaded = True
except FileNotFoundError:
    st.error("⚠️ ملفات الموديل غير موجودة. تأكد من تشغيل كود التدريب وحفظ `rf_health_model.pkl` و `rf_life_model.pkl` في نفس المجلد.")
    models_loaded = False

if models_loaded:
    # تقسيم الشاشة
    col_input, col_output = st.columns([1, 1.2])

    with col_input:
        st.markdown("### 📊 إدخال قراءات المحول (Inputs)")
        st.markdown("أدخل نتائج الفحص الكيميائي والفيزيائي للزيت:")
        
        with st.form("input_form"):
            col1, col2 = st.columns(2)
            
            # الغازات الأساسية
            with col1:
                st.markdown("**1. الغازات المذابة (DGA - ppm):**")
                hydrogen = st.number_input("Hydrogen (H2)", value=100.0, step=10.0, help="غاز الهيدروجين")
                oxigen = st.number_input("Oxigen (O2)", value=5000.0, step=100.0, help="غاز الأكسجين")
                nitrogen = st.number_input("Nitrogen (N2)", value=40000.0, step=1000.0, help="غاز النيتروجين")
                methane = st.number_input("Methane (CH4)", value=50.0, step=5.0, help="غاز الميثان")
                co = st.number_input("Carbon Monoxide (CO)", value=200.0, step=10.0, help="أول أكسيد الكربون")
                co2 = st.number_input("Carbon Dioxide (CO2)", value=1500.0, step=50.0, help="ثاني أكسيد الكربون")
                ethylene = st.number_input("Ethylene (C2H4)", value=10.0, step=2.0, help="غاز الإيثيلين")
                ethane = st.number_input("Ethane (C2H6)", value=20.0, step=5.0, help="غاز الإيثان")
                acethylene = st.number_input("Acethylene (C2H2)", value=0.0, step=1.0, help="غاز الأسيتيلين (مؤشر التفريغ)")

            # الخواص الفيزيائية
            with col2:
                st.markdown("**2. خواص الزيت الفيزيائية:**")
                dbds = st.number_input("DBDS (ppm)", value=10.0, step=1.0, help="مضاد أكسدة (كثرته تسبب كبريتيد النحاس)")
                power_factor = st.number_input("Power Factor (%)", value=1.5, step=0.1, help="معامل القدرة للزيت (التسريب)")
                interfacial_v = st.number_input("Interfacial Tension (mN/m)", value=40.0, step=1.0, help="التوتر السطحي (الأعلى أفضل)")
                dielectric_rigidity = st.number_input("Dielectric Rigidity (kV)", value=50.0, step=1.0, help="جهد الانهيار (الأعلى أفضل)")
                water_content = st.number_input("Water Content (ppm)", value=15.0, step=1.0, help="المحتوى المائي (الرطوبة)")
            
            submitted = st.form_submit_button("🔍 تحليل حالة المحول (Run Analysis)", use_container_width=True)

    with col_output:
        st.markdown("### 📈 نتيجة التحليل والذكاء الاصطناعي (Output)")
        
        if submitted:
            # تجهيز المدخلات
            input_data = pd.DataFrame({
                'Hydrogen': [hydrogen], 'Oxigen': [oxigen], 'Nitrogen': [nitrogen], 
                'Methane': [methane], 'CO': [co], 'CO2': [co2], 
                'Ethylene': [ethylene], 'Ethane': [ethane], 'Acethylene': [acethylene], 
                'DBDS': [dbds], 'Power factor': [power_factor], 
                'Interfacial V': [interfacial_v], 'Dielectric rigidity': [dielectric_rigidity], 
                'Water content': [water_content]
            })

            # التوقع
            health_index_pred = model_health.predict(input_data)[0]
            life_expectation_pred = model_life.predict(input_data)[0]

            # دالة لتحديد الحالة والألوان بناءً على مؤشر الصحة
            def get_status_info(health_score):
                if health_score >= 70:
                    return "Safe (آمن)", "#00cc66", "المحول في حالة جيدة جداً، لا يتطلب سوى الصيانة الروتينية."
                elif health_score >= 40:
                    return "Warning (تحذير)", "#ffcc00", "حالة الزيت متدهورة، يرجى جدولة فحص شامل قريباً."
                else:
                    return "Risky (خطر)", "#ff3333", "تدهور شديد! خطر عطل مفاجئ، يتطلب تدخل فوري أو استبدال الزيت."

            status_text, color, recommendation = get_status_info(health_index_pred)

            # --- رسم مؤشر الصحة (Health Index Gauge) ---
            fig_health = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = health_index_pred,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "مؤشر صحة المحول (Health Index)", 'font': {'size': 20, 'color': 'white'}},
                number = {'font': {'size': 40, 'color': 'white'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "white", 'thickness': 0.1},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': "#ff3333"},  # أحمر
                        {'range': [40, 70], 'color': "#ffcc00"},  # أصفر
                        {'range': [70, 100], 'color': "#00cc66"}   # أخضر
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': health_index_pred
                    }
                }
            ))
            
            fig_health.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                font={'color': "white"}, 
                height=350, 
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig_health, use_container_width=True)

            # --- عرض العمر المتوقع والتوصية ---
            st.markdown("---")
            col_life, col_status = st.columns(2)
            
            with col_life:
                st.markdown(f"""
                <div style='background-color: #1e3d59; padding: 20px; border-radius: 10px; border-left: 5px solid #4da6ff;'>
                    <h4 style='color: #4da6ff; margin-top: 0;'>⏳ العمر المتبقي المتوقع</h4>
                    <h2 style='color: white; margin-bottom: 0;'>{life_expectation_pred:.1f} سنة</h2>
                </div>
                """, unsafe_allow_html=True)

            with col_status:
                st.markdown(f"""
                <div style='background-color: #1e3d59; padding: 20px; border-radius: 10px; border-left: 5px solid {color};'>
                    <h4 style='color: {color}; margin-top: 0;'>⚠️ حالة الخطر: {status_text}</h4>
                    <p style='color: white; margin-bottom: 0;'>{recommendation}</p>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.info("👈 قم بإدخال القراءات في الجدول المجاور واضغط على الزر الأزرق لبدء التحليل.")