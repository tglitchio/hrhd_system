import pandas as pd
import plotly.express as px

def generar_ranking_fallas(path_excel="datos1.xlsx"):
    try:
        df = pd.read_excel(path_excel)
    except FileNotFoundError:
        return None, "❌ Archivo Excel no encontrado."

    # --- Renombrar columnas si es necesario ---
    df.columns = df.columns.str.strip()
    if "Garantía de servicio en esa fecha" in df.columns:
        df.rename(columns={"Garantía de servicio en esa fecha": "¿Estaba en garantía?"}, inplace=True)

    # --- Verificación de columnas necesarias ---
    required_cols = ["Tipo", "Nombre del equipo", "Marca del equipo", "¿Estaba en garantía?"]
    for col in required_cols:
        if col not in df.columns:
            return None, f"❌ Falta la columna: '{col}' en el archivo Excel."

    # --- Limpieza de datos ---
    df["Tipo"] = df["Tipo"].astype(str).str.strip().str.lower()
    df["¿Estaba en garantía?"] = df["¿Estaba en garantía?"].astype(str).str.strip().str.upper()

    # --- Filtrar solo fallas ---
    fallas = df[df["Tipo"] == "falla"]
    if fallas.empty:
        return None, "⚠️ No hay suficientes fallas registradas aún."

    # --- Gráfico 1: Por equipo ---
    ranking_eq = fallas.groupby(["Nombre del equipo", "Marca del equipo"]).size().reset_index(name="Cantidad de fallas")

    def color_riesgo(f):
        if f >= 5:
            return "🔴 Alta"
        elif f >= 3:
            return "🟡 Media"
        else:
            return "🟢 Baja"

    ranking_eq["Riesgo"] = ranking_eq["Cantidad de fallas"].apply(color_riesgo)

    fig_eq = px.bar(
        ranking_eq.sort_values("Cantidad de fallas", ascending=False),
        x="Nombre del equipo",
        y="Cantidad de fallas",
        color="Riesgo",
        text="Cantidad de fallas",
        title="🔧 Ranking de Equipos con Más Fallas",
        color_discrete_map={"🔴 Alta": "red", "🟡 Media": "orange", "🟢 Baja": "green"}
    )
    fig_eq.update_traces(textposition="outside")

    # --- Gráfico 2: Por marca ---
    ranking_marca = fallas.groupby("Marca del equipo").size().reset_index(name="Cantidad de fallas")
    fig_marca = px.bar(
        ranking_marca.sort_values("Cantidad de fallas", ascending=False),
        x="Marca del equipo",
        y="Cantidad de fallas",
        text="Cantidad de fallas",
        title="🏷️ Marcas con Mayor Número de Fallas",
        color="Cantidad de fallas",
        color_continuous_scale="Reds"
    )
    fig_marca.update_traces(textposition="outside")

    # --- Gráfico 3: Por garantía ---
    garantia = fallas.groupby("¿Estaba en garantía?").size().reset_index(name="Cantidad de fallas")
    fig_garantia = px.pie(
        garantia,
        names="¿Estaba en garantía?",
        values="Cantidad de fallas",
        title="📉 Proporción de Fallas por Estado de Garantía",
        color="¿Estaba en garantía?",
        color_discrete_map={"SI": "green", "NO": "red"}
    )

    return (fig_eq, fig_marca, fig_garantia), None
