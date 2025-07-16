from db import crear_base, guardar_evento, cargar_eventos, get_connection 
from ranking import generar_ranking_fallas
import streamlit as st
import pandas as pd
from datetime import date
import plotly.express as px
from PIL import Image
import base64
import modelo 
import io # Asegúrate de que modelo.py está en la misma carpeta

# --- CONEXIÓN Y DATOS ---
conn = get_connection()
data = cargar_eventos()

# Si no hay datos en la base, cargar los del Excel original
if data.empty:
    try:
        data = pd.read_excel("datos1.xlsx")
        st.warning("⚠️ No se han registrado eventos aún. Se están mostrando datos de ejemplo desde Excel.")
    except FileNotFoundError:
        st.error("❌ No se encontró datos1.xlsx. Por favor, registra al menos un evento.")

# Asegurarse de que las columnas necesarias existen
for col in ["Nombre del equipo", "Marca del equipo", "Diagnóstico técnico"]:
    if col not in data.columns:
        data[col] = ""

# --- FUNCIONES VISUALES ---
def get_base64_image(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# --- CONFIGURACIÓN ---
banner_path = "C:/Users/ana cristina/Downloads/Estandarizar/foto1.jpg"
encoded_banner = get_base64_image(banner_path)

st.set_page_config(
    page_title="Modelo Predictivo de Fallas en Equipos Biomédicos",
    layout="wide",
    page_icon="🩺"
)

# Entrenar modelo si existe función
if hasattr(modelo, "entrenar_modelo"):
    modelo.entrenar_modelo()

crear_base()

# --- ESTILOS ---
st.markdown(f"""
    <style>
    .stApp {{ background-color: #F6FFF5; }}
    .main-container {{ background-color: rgba(255, 255, 255, 0.90); padding: 2rem; border-radius: 15px; }}
    h1, h2, h3 {{ color: #247A55; }}
    .stButton > button {{ background-color: #FFF9DB; color: #000; border-radius: 8px; }}
    [data-testid="stSidebar"] {{ background-color: #FFFFFF; }}
    .title-container {{
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #EEE;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }}
    .custom-headline {{
        width: 100%;
        height: 200px;
        background-image: url("data:image/jpeg;base64,{encoded_banner}");
        background-size: cover;
        background-position: center;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 36px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }}
    </style>
    <div class="custom-headline">Modelo Predictivo de fallas en equipos biomédicos</div>
""", unsafe_allow_html=True)

# --- TÍTULO ---
st.markdown("""<div class="title-container">
    Visualización, monitoreo y predicción de eventos críticos para equipos UCI
</div>""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("C:/Users/ana cristina/Downloads/Estandarizar/images.jpeg", width=150)
    st.title("🔍 Navegación")
    opcion = st.radio("Ir a:", ["📊 Dashboard", "➕ Nuevo evento", "📋 Predicción"])

# --- DASHBOARD ---
if opcion == "📊 Dashboard":
    st.markdown("<h2>Visualización General de Equipos</h2>", unsafe_allow_html=True)
    st.markdown("#### 📋 Historial de eventos")
    st.dataframe(data)

    if "predicciones" in st.session_state:
        st.markdown("#### 📈 Predicciones recientes de falla")
        df_pred = pd.DataFrame(st.session_state.predicciones)
        st.dataframe(df_pred)

        fig = px.bar(df_pred.sort_values("Probabilidad de falla", ascending=False),
                     x="Equipo", y="Probabilidad de falla", color="Marca",
                     title="Equipos con mayor probabilidad de falla")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📈 Ver ranking de fallas por equipo"):
        if st.button("Actualizar ranking"):
            figuras, mensaje = generar_ranking_fallas()
            if mensaje:
                st.warning(mensaje)
            else:
                fig_eq, fig_marca, fig_garantia = figuras

                tab1, tab2, tab3 = st.tabs(["📊 Por equipo", "🏷️ Por marca", "🛡️ Por garantía"])
                with tab1:
                    st.plotly_chart(fig_eq, use_container_width=True)
                with tab2:
                    st.plotly_chart(fig_marca, use_container_width=True)
                with tab3:
                    st.plotly_chart(fig_garantia, use_container_width=True)

# --- NUEVO EVENTO ---
elif opcion == "➕ Nuevo evento":
    st.markdown("<h2>Registrar nuevo evento</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        nombre_opciones = list(data["Nombre del equipo"].dropna().unique()) + ["Otro"]
        nombre = st.selectbox("Nombre del equipo", nombre_opciones, key="new_eq")
        if nombre == "Otro":
            nombre = st.text_input("Especificar nuevo nombre del equipo")

        marca_opciones = list(data["Marca del equipo"].dropna().unique()) + ["Otro"]
        marca = st.selectbox("Marca del equipo", marca_opciones, key="new_marca")
        if marca == "Otro":
            marca = st.text_input("Especificar nueva marca del equipo")

        numero_serie = st.text_input("Número de serie del equipo")
        tipo = st.radio("Tipo de evento", ["Falla", "Preventivo", "Correctivo"])
    with col2:
        fecha = st.date_input("Fecha del incidente", value=date.today())
        garantia = st.radio("¿Estaba en garantía?", ["SI", "NO"])

        diag_opciones = list(data["Diagnóstico técnico"].dropna().unique()) + ["Otro"]
        diagnostico = st.selectbox("Diagnóstico técnico", diag_opciones)
        if diagnostico == "Otro":
            diagnostico = st.text_input("Especificar nuevo diagnóstico")

    if st.button("Guardar evento"):
        guardar_evento(nombre, marca, numero_serie, tipo, fecha, garantia, diagnostico)
        st.success("✅ Evento guardado correctamente en la base de datos.")

# --- PREDICCIÓN ---
elif opcion == "📋 Predicción":
    st.markdown("<h2>Predicción de falla</h2>", unsafe_allow_html=True)

    nombre = st.selectbox("Nombre del equipo", data["Nombre del equipo"].unique())
    marca = st.selectbox("Marca del equipo", data["Marca del equipo"].unique())
    garantia = st.radio("¿Está en garantía?", ["SI", "NO"], horizontal=True)
    horizonte_dias = st.slider("Horizonte de predicción (días)", min_value=3, max_value=60, value=30)

    if st.button("Predecir"):
        dias_estimados, riesgo = modelo.predecir_falla_auto(nombre, marca, garantia, horizonte_dias)

        if dias_estimados is not None:
            # Armar diccionario embellecido
            resultado = {
             "Equipo": nombre,
             "Marca": marca,
             "Garantía": "Sí" if garantia == "SI" else "No",
              "Horizonte de predicción": f"{horizonte_dias} días",
              "Días estimados hasta la falla": f"{round(dias_estimados, 1)} días",
              "Probabilidad de falla": f"{round(riesgo, 1)}%",  # <- Solo agrega el símbolo
              "Fecha de predicción": date.today().strftime("%d/%m/%Y")
                }


            # Guardar en session_state
            if "predicciones" not in st.session_state:
                st.session_state.predicciones = []
            st.session_state.predicciones.append(resultado)

            st.success("✅ Predicción realizada con éxito")

            # Mostrar como tabla bonita
            df_resultado = pd.DataFrame(resultado.items(), columns=["Parámetro", "Valor"])
            st.markdown("### 🧠 Resultados de la predicción")
            st.dataframe(df_resultado, use_container_width=True, hide_index=True)

            # Descargar como Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df_resultado.to_excel(writer, index=False, sheet_name="Predicción")
               # writer.save()
                buffer.seek(0)

            st.download_button(
                label="📥 Descargar resultado en Excel",
                data=buffer,
                file_name=f"prediccion_{nombre.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("⚠️ No hay historial suficiente para ese equipo y marca.")
