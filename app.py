import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import logging
import warnings
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from PIL import Image
import base64
import io
from db import crear_base, guardar_evento, cargar_eventos, get_connection
import modelo

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# Configuración de Streamlit
st.set_page_config(
    page_title="Modelo Predictivo de Fallas en Equipos Biomédicos v5",
    layout="wide",
    page_icon="🩺"
)

# Inicializar session state
if 'predicciones' not in st.session_state:
    st.session_state.predicciones = []

# Conexión a la base de datos
conn = get_connection()
crear_base()

# --- FUNCIONES AUXILIARES PARA IMÁGENES ---
def get_base64_image(file_path):
    """Convierte una imagen a base64 para mostrar en el dashboard."""
    try:
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        logging.error(f"Imagen no encontrada: {file_path}")
        return ""

# === FUNCIÓN DE VERIFICACIÓN DE CALIDAD DE DATOS ===
def verificar_calidad_datos(df):
    """Función para verificar la calidad de los datos cargados"""
    if df.empty:
        st.error("❌ No hay datos cargados")
        return False
    
    st.sidebar.markdown("### 📊 Información de Datos")
    
    # Información básica
    st.sidebar.write(f"**Total registros:** {len(df)}")
    
    # Verificar fechas
    if 'Fecha del incidente' in df.columns:
        fecha_min = df['Fecha del incidente'].min()
        fecha_max = df['Fecha del incidente'].max()
        st.sidebar.write(f"**Período:** {fecha_min.strftime('%m/%Y')} - {fecha_max.strftime('%m/%Y')}")
        
        # Advertir si hay problemas
        if fecha_max.year < 2023:
            st.sidebar.error("⚠️ Datos desactualizados")
        elif fecha_min.year < 2018:
            st.sidebar.warning("⚠️ Datos muy antiguos incluidos")
        else:
            st.sidebar.success("✅ Fechas actualizadas")
    
    # Verificar equipos
# REEMPLAZAR esta línea (que está causando el error):
# st.write(f"**DEBUG:** Total equipos únicos encontrados: {len(equipos_unicos)}")



# === FUNCIONES DEL MODELO 5 INTEGRADAS - CORREGIDAS ===
@st.cache_data(ttl=300)  # Cache por 5 minutos
def cargar_y_procesar_datos():
    """Carga y procesa los datos con el pipeline completo del modelo 5 - VERSIÓN CORREGIDA"""
    try:
        # Intentar cargar desde DB primero
        df = cargar_eventos()
        if df.empty:
            df = pd.read_excel("datos1.xlsx")
            st.info("Usando datos de ejemplo desde datos1.xlsx")
    except Exception as e:
        logging.error(f"Error al cargar datos: {e}")
        st.error(f"No se pudieron cargar datos: {e}")
        return pd.DataFrame()
    
    if df.empty:
        return df
    
    # === CORRECCIÓN CRÍTICA: MANEJO DE FECHAS ===
    df = df.copy()
    
    # Debug inicial de fechas
    print(f"Columnas disponibles: {df.columns.tolist()}")
    
    # Llenar valores faltantes
    df['Tipo'] = df['Tipo'].fillna('Desconocido')
    df['Accesorio afectado'] = df['Accesorio afectado'].fillna('Sin_Accesorio')
    df['Nombre del equipo'] = df['Nombre del equipo'].str.strip().str.lower()
    df['Marca del equipo'] = df['Marca del equipo'].fillna('Marca_Desconocida')
    df['Área usuaria'] = df['Área usuaria'].fillna('Área_Desconocida')
    
    # PARSEAR FECHAS - VERSIÓN CORREGIDA
    try:
        # Verificar el nombre de la columna de fecha
        fecha_columns = [col for col in df.columns if 'fecha' in col.lower()]
        print(f"Después de eliminar duplicados: {len(df)} registros")

        print(f"Columnas de fecha encontradas: {fecha_columns}")
        


        if 'Fecha del incidente' in df.columns:
            fecha_col = 'Fecha del incidente'
        elif fecha_columns:
            fecha_col = fecha_columns[0]
            print(f"Usando columna: {fecha_col}")
        else:
            st.error("❌ No se encontró ninguna columna de fecha")
            return pd.DataFrame()
        
        # Convertir fechas con manejo robusto
        df[fecha_col] = pd.to_datetime(df[fecha_col], errors='coerce', dayfirst=True)
        
        # Verificar rango de fechas ANTES de procesar
        fechas_validas = df[fecha_col].dropna()
        if not fechas_validas.empty:
            fecha_min = fechas_validas.min()
            fecha_max = fechas_validas.max()
            print(f"Rango de fechas en datos: {fecha_min} a {fecha_max}")
            
            # Mostrar información en la app
            st.info(f"📅 Datos cargados: {len(fechas_validas)} registros desde {fecha_min.strftime('%d/%m/%Y')} hasta {fecha_max.strftime('%d/%m/%Y')}")
            
            # Advertir si las fechas están fuera del rango esperado
            if fecha_max.year < 2020 or fecha_min.year > 2025:
                st.warning(f"⚠️ Las fechas parecen estar fuera del rango esperado (2020-2025). Rango actual: {fecha_min.year}-{fecha_max.year}")
        
        # Renombrar para consistencia
        if fecha_col != 'Fecha del incidente':
            df = df.rename(columns={fecha_col: 'Fecha del incidente'})
        
        invalid_dates = df['Fecha del incidente'].isna().sum()
        if invalid_dates > 0:
            logging.warning(f"Se encontraron {invalid_dates} fechas inválidas")
            st.warning(f"⚠️ Se encontraron {invalid_dates} fechas inválidas que serán excluidas")
        df = df.dropna(subset=['Fecha del incidente'])
        
    except Exception as e:
        logging.error(f"Error al parsear fechas: {e}")
        st.error(f"Error al procesar fechas: {e}")
        return pd.DataFrame()
    
    # VERIFICACIÓN ADICIONAL: Filtrar fechas muy antiguas o futuras
    fecha_limite_min = pd.Timestamp('2018-01-01')  # Permitir desde 2018
    fecha_limite_max = pd.Timestamp('2026-12-31')  # Hasta 2026
    
    fechas_originales = len(df)
    df = df[
        (df['Fecha del incidente'] >= fecha_limite_min) & 
        (df['Fecha del incidente'] <= fecha_limite_max)
    ]
    fechas_filtradas = len(df)
    
    if fechas_originales != fechas_filtradas:
        st.info(f"📅 Se filtraron {fechas_originales - fechas_filtradas} registros con fechas fuera del rango 2018-2026")
    
    # Eliminar duplicados exactos
    df = df.drop_duplicates(subset=['Nombre del equipo', 'Fecha del incidente'])
    
    # Feature: número de fallas en el mismo día
    df = df.sort_values(by=['Nombre del equipo', 'Fecha del incidente'])
    df['num_fallas_mismo_dia'] = df.groupby(['Nombre del equipo', 'Fecha del incidente']).cumcount() + 1
    
    # Mantener primera falla por día y equipo
    df = df.groupby(['Nombre del equipo', 'Fecha del incidente']).first().reset_index()
    
    # Calcular días hasta falla
    df['dias_hasta_falla'] = df.groupby('Nombre del equipo')['Fecha del incidente'].shift(-1) - df['Fecha del incidente']
    df['dias_hasta_falla'] = df['dias_hasta_falla'].dt.days
    
    # Feature: días desde última falla
    df['prev_fecha'] = df.groupby('Nombre del equipo')['Fecha del incidente'].shift(1)
    df['days_since_last_failure'] = (df['Fecha del incidente'] - df['prev_fecha']).dt.days.fillna(0)
    
    # Feature: tasa de fallas por año
    df['year'] = df['Fecha del incidente'].dt.year
    failure_rate = df.groupby(['Nombre del equipo', 'year']).size().reset_index(name='failure_count')
    failure_rate['failure_rate'] = failure_rate['failure_count'] / 1.0
    df = df.merge(failure_rate[['Nombre del equipo', 'year', 'failure_rate']], 
                  on=['Nombre del equipo', 'year'], how='left')
    df['failure_rate'] = df['failure_rate'].fillna(df['failure_rate'].median())
    
    # Feature: fallas en los últimos 30 días
    df['recent_failures_30d'] = df.groupby('Nombre del equipo').apply(
        lambda x: x.set_index('Fecha del incidente').rolling('30D').count()['Nombre del equipo'] - 1
    ).reset_index(drop=True)
    df['recent_failures_30d'] = df['recent_failures_30d'].fillna(0)
    
    # Feature: edad del equipo
    first_failure = df.groupby('Nombre del equipo')['Fecha del incidente'].min().reset_index()
    first_failure.columns = ['Nombre del equipo', 'first_failure_date']
    df = df.merge(first_failure, on='Nombre del equipo', how='left')
    df['equipment_age_days'] = (df['Fecha del incidente'] - df['first_failure_date']).dt.days
    
    # Características temporales
    df['mes'] = df['Fecha del incidente'].dt.month
    df['dia_semana'] = df['Fecha del incidente'].dt.dayofweek
    df['trimestre'] = df['Fecha del incidente'].dt.quarter
    
    # Preservar última falla
    df_last = df.groupby('Nombre del equipo').tail(1).copy()
    df_last['dias_hasta_falla'] = df_last['dias_hasta_falla'].fillna(df['dias_hasta_falla'].median())
    
    # Filtrar outliers (Q5-Q95)
    df_non_last = df[~df.index.isin(df_last.index)]
    if not df_non_last.empty and not df_non_last['dias_hasta_falla'].isna().all():
        #Q1 = df_non_last['dias_hasta_falla'].quantile(0.05)
        # CAMBIAR las líneas 150-151:
        Q1 = df_non_last['dias_hasta_falla'].quantile(0.05)  # Cambiar de 0.05 a 0.01
        Q3 = df_non_last['dias_hasta_falla'].quantile(0.95)  # Cambiar de 0.95 a 0.99
        #Q3 = df_non_last['dias_hasta_falla'].quantile(0.95)
        df_filtered = df_non_last[(df_non_last['dias_hasta_falla'] >= Q1) & 
                                  (df_non_last['dias_hasta_falla'] <= Q3)].copy()
        df_filtered = pd.concat([df_filtered, df_last], ignore_index=True)
    else:
        df_filtered = df.copy()
    
    df = df_filtered.sort_values(by=['Nombre del equipo', 'Fecha del incidente']).reset_index(drop=True)
    
    # Características históricas
    df['num_fallas_previas'] = df.groupby('Nombre del equipo').cumcount()
    
    # Estadísticas globales por grupo
    marca_stats = df.groupby('Marca del equipo')['dias_hasta_falla'].agg(['mean', 'count']).reset_index()
    marca_stats.columns = ['Marca del equipo', 'promedio_dias_falla_marca', 'count_marca']
    marca_stats = marca_stats[marca_stats['count_marca'] >= 2]
    df = df.merge(marca_stats[['Marca del equipo', 'promedio_dias_falla_marca']], 
                  on='Marca del equipo', how='left')
    
    area_stats = df.groupby('Área usuaria')['dias_hasta_falla'].agg(['mean', 'count']).reset_index()
    area_stats.columns = ['Área usuaria', 'promedio_dias_falla_area', 'count_area']
    area_stats = area_stats[area_stats['count_area'] >= 2]
    df = df.merge(area_stats[['Área usuaria', 'promedio_dias_falla_area']], 
                  on='Área usuaria', how='left')
    

    df = df.dropna(subset=['dias_hasta_falla'])
    print(f"Después de filtrar NaN en días_hasta_falla: {len(df)} equipos únicos: {df['Nombre del equipo'].nunique()}")


    # Llenar NaN con medianas
    df['promedio_dias_falla_marca'] = df['promedio_dias_falla_marca'].fillna(df['dias_hasta_falla'].median())
    df['promedio_dias_falla_area'] = df['promedio_dias_falla_area'].fillna(df['dias_hasta_falla'].median())
    
    df = df.dropna(subset=['dias_hasta_falla'])
    
    # Debug final
    if not df.empty:
        fecha_final_min = df['Fecha del incidente'].min()
        fecha_final_max = df['Fecha del incidente'].max()
        print(f"Datos procesados finales: {len(df)} registros, desde {fecha_final_min} hasta {fecha_final_max}")
    
    print(f"DATOS FINALES: {len(df)} registros, {df['Nombre del equipo'].nunique()} equipos únicos")
    return df

def entrenar_modelo_completo(df):
    """Entrena el modelo completo usando el pipeline del modelo 5"""
    if df.empty or len(df) < 10:
        return None, None, None, None
    
    # Transformar target con offset (del modelo 5)
    df['dias_hasta_falla_log'] = np.log1p(df['dias_hasta_falla'] + 1)
    
    # Selección de características
    features_categoricas = ['Marca del equipo', 'Área usuaria']
    features_numericas = ['mes', 'dia_semana', 'trimestre', 'num_fallas_previas', 
                         'promedio_dias_falla_marca', 'promedio_dias_falla_area', 
                         'num_fallas_mismo_dia', 'days_since_last_failure', 
                         'failure_rate', 'equipment_age_days', 'recent_failures_30d']
    
    # Filtrar características que existen
    features_categoricas = [f for f in features_categoricas if f in df.columns]
    features_numericas = [f for f in features_numericas if f in df.columns]
    
    X = df[features_categoricas + features_numericas]
    y = df['dias_hasta_falla_log']
    
    # Preprocesamiento
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=10), features_categoricas),
        ("num", RobustScaler(), features_numericas)
    ])
    
    # Modelos (del modelo 5)
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42
    )
    
    # Ensemble
    ensemble = VotingRegressor(estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model)
    ])
    
    # Pipeline completo
    modelo_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('ensemble', ensemble)
    ])
    
    # Validación temporal
    tscv = TimeSeriesSplit(n_splits=3)
    mae_scores = []
    r2_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        modelo_pipeline.fit(X_train, y_train)
        y_pred_log = modelo_pipeline.predict(X_val)
        
        # Revertir transformación
        y_pred = np.expm1(y_pred_log) - 1
        y_val_original = np.expm1(y_val) - 1
        
        mae = mean_absolute_error(y_val_original, y_pred)
        r2 = r2_score(y_val_original, y_pred)
        
        mae_scores.append(mae)
        r2_scores.append(r2)
    
    # Entrenar modelo final
    modelo_pipeline.fit(X, y)
    
    return modelo_pipeline, np.mean(mae_scores), np.mean(r2_scores), df

def predecir_desde_ultima_falla_modelo5(datos_basicos, df_historico, modelo, fecha_actual=None):
    """Función de predicción adaptada del modelo 5 - VERSIÓN CORREGIDA"""
    if modelo is None:
        # Usar mediana como fallback
        mediana_dias = df_historico['dias_hasta_falla'].median() if not df_historico.empty else 30
        return mediana_dias, 50.0  # 50% de probabilidad por defecto
    
    # CORRECCIÓN: Usar fecha actual real
    fecha_actual = pd.Timestamp(fecha_actual or datetime.now())
    
    equipo = datos_basicos['Nombre del equipo'].strip().lower()
    marca = datos_basicos['Marca del equipo']
    area = datos_basicos.get('Área usuaria', 'Área_Desconocida')
    
    # Buscar historial del equipo
    hist_equipo = df_historico[df_historico['Nombre del equipo'].str.strip().str.lower() == equipo].copy()
    
    if not hist_equipo.empty and 'Fecha del incidente' in hist_equipo.columns:
        hist_equipo['Fecha del incidente'] = pd.to_datetime(hist_equipo['Fecha del incidente'], errors='coerce')
        hist_equipo = hist_equipo.dropna(subset=['Fecha del incidente'])
        
        if not hist_equipo.empty:
            ultima_fecha_falla = hist_equipo['Fecha del incidente'].max()
            
            # VERIFICAR que la fecha no sea muy antigua
            dias_desde_ultima = (fecha_actual - ultima_fecha_falla).days
            
            if dias_desde_ultima > 1000:  # Más de ~3 años
                st.warning(f"⚠️ La última falla del equipo '{equipo}' fue hace {dias_desde_ultima} días ({ultima_fecha_falla.strftime('%d/%m/%Y')}). La predicción puede no ser precisa.")
        else:
            ultima_fecha_falla = fecha_actual - timedelta(days=df_historico['dias_hasta_falla'].median())
        
        if pd.isna(ultima_fecha_falla):
            ultima_fecha_falla = fecha_actual - timedelta(days=df_historico['dias_hasta_falla'].median())
        
        # Calcular características
        prev_fecha = hist_equipo[hist_equipo['Fecha del incidente'] < ultima_fecha_falla]['Fecha del incidente'].max()
        days_since_last = (ultima_fecha_falla - prev_fecha).days if not pd.isna(prev_fecha) else df_historico['days_since_last_failure'].median()
        
        failure_rate_equipo = hist_equipo[hist_equipo['year'] == ultima_fecha_falla.year]['failure_rate'].iloc[0] if not hist_equipo[hist_equipo['year'] == ultima_fecha_falla.year].empty else df_historico['failure_rate'].median()
        
        num_fallas_mismo_dia = hist_equipo[hist_equipo['Fecha del incidente'] == ultima_fecha_falla]['num_fallas_mismo_dia'].iloc[0] if not hist_equipo[hist_equipo['Fecha del incidente'] == ultima_fecha_falla].empty else 1
        
        equipment_age_days = (ultima_fecha_falla - hist_equipo['Fecha del incidente'].min()).days if not hist_equipo.empty else df_historico['equipment_age_days'].median()
        
        recent_failures_30d = hist_equipo[hist_equipo['Fecha del incidente'] >= ultima_fecha_falla - timedelta(days=30)].shape[0] - 1
        
    else:
        # Sin historial, usar valores por defecto con fecha actual
        ultima_fecha_falla = fecha_actual - timedelta(days=df_historico['dias_hasta_falla'].median() if not df_historico.empty else 30)
        days_since_last = df_historico['days_since_last_failure'].median() if not df_historico.empty else 30
        failure_rate_equipo = df_historico['failure_rate'].median() if not df_historico.empty else 1.0
        num_fallas_mismo_dia = 1
        equipment_age_days = df_historico['equipment_age_days'].median() if not df_historico.empty else 365
        recent_failures_30d = 0
        st.info(f"No se encontró historial para '{equipo}'. Usando estimación base.")
    
    # Crear DataFrame para predicción
    datos_pred = pd.DataFrame([{
        'Marca del equipo': marca,
        'Área usuaria': area,
        'mes': ultima_fecha_falla.month,
        'dia_semana': ultima_fecha_falla.dayofweek,
        'trimestre': ultima_fecha_falla.quarter,
        'num_fallas_previas': len(hist_equipo),
        'promedio_dias_falla_marca': df_historico[df_historico['Marca del equipo'] == marca]['dias_hasta_falla'].mean() if not df_historico[df_historico['Marca del equipo'] == marca].empty else df_historico['dias_hasta_falla'].median(),
        'promedio_dias_falla_area': df_historico[df_historico['Área usuaria'] == area]['dias_hasta_falla'].mean() if not df_historico[df_historico['Área usuaria'] == area].empty else df_historico['dias_hasta_falla'].median(),
        'num_fallas_mismo_dia': num_fallas_mismo_dia,
        'days_since_last_failure': days_since_last,
        'failure_rate': failure_rate_equipo,
        'equipment_age_days': equipment_age_days,
        'recent_failures_30d': recent_failures_30d
    }])
    
    try:
        # Predicción
        dias_estimados_log = modelo.predict(datos_pred)[0]
        dias_estimados = np.expm1(dias_estimados_log) - 1  # Revertir transformación
        dias_estimados = max(1, np.clip(dias_estimados, 0, 365))  # Máximo 1 año
        
        # Calcular probabilidad basada en el modelo
        probabilidad = min(100, max(10, 100 - (dias_estimados / 3)))  # Heurística mejorada
        
        return dias_estimados, probabilidad
        
    except Exception as e:
        logging.error(f"Error en predicción: {e}")
        mediana_dias = df_historico['dias_hasta_falla'].median() if not df_historico.empty else 30
        return mediana_dias, 50.0

def crear_metricas_dashboard(data):
    """Crear métricas principales del dashboard."""
    col1, col2, col3, col4 = st.columns(4)
    
    total_equipos = len(data['Nombre del equipo'].unique()) if not data.empty else 0
    total_fallas = len(data[data['Tipo'] == 'Falla']) if 'Tipo' in data.columns and not data.empty else len(data)
    equipos_criticos = len(data[data['Nombre del equipo'].isin(['ventilador mecánico', 'monitor multiparámetro', 'aspirador de secreciones'])]) if not data.empty else 0
    
    # Calcular fallas recientes (último mes)
    if not data.empty and 'Fecha del incidente' in data.columns:
        fecha_limite = datetime.now() - timedelta(days=30)
        fallas_recientes = len(data[data['Fecha del incidente'] >= fecha_limite])
    else:
        fallas_recientes = 0
    
    with col1:
        st.metric(label="🏥 Total Equipos", value=total_equipos)
    with col2:
        st.metric(label="⚠️ Total Eventos", value=len(data))
    with col3:
        st.metric(label="🚨 Equipos Críticos", value=equipos_criticos)
    with col4:
        st.metric(label="📅 Eventos Recientes (30d)", value=fallas_recientes)

def crear_graficos_analisis(data):
    """Crear gráficos de análisis avanzado - VERSIÓN CORREGIDA."""
    if data.empty:
        st.warning("No hay datos suficientes para generar gráficos.")
        return
    
    # VERIFICACIÓN DE FECHAS ANTES DE GRAFICAR
    if 'Fecha del incidente' in data.columns:
        data = data.dropna(subset=['Fecha del incidente'])
        
        # Verificar rango de fechas
        fecha_min = data['Fecha del incidente'].min()
        fecha_max = data['Fecha del incidente'].max()
        
        st.info(f"📅 Datos disponibles desde {fecha_min.strftime('%d/%m/%Y')} hasta {fecha_max.strftime('%d/%m/%Y')}")
        
        # Gráfico de tendencia temporal - CORREGIDO
        st.subheader("📈 Tendencia Temporal de Eventos")
        data_temp = data.copy()
        
        # CORRECCIÓN: Agrupar por año-mes de forma más robusta
        data_temp['año'] = data_temp['Fecha del incidente'].dt.year
        data_temp['mes'] = data_temp['Fecha del incidente'].dt.month
        data_temp['año_mes'] = data_temp['Fecha del incidente'].dt.to_period('M').astype(str)
        
        eventos_por_mes = data_temp.groupby('año_mes').size().reset_index(name='Cantidad')
        
        # Ordenar correctamente por fecha
        eventos_por_mes['fecha_ordenamiento'] = pd.to_datetime(eventos_por_mes['año_mes'])
        eventos_por_mes = eventos_por_mes.sort_values('fecha_ordenamiento')
        
        if not eventos_por_mes.empty:
            fig_tendencia = px.line(
                eventos_por_mes, 
                x='año_mes', 
                y='Cantidad',
                title="Evolución de Eventos por Mes", 
                markers=True
            )
            fig_tendencia.update_xaxes(tickangle=-45, title="Período")
            fig_tendencia.update_yaxes(title="Número de Eventos")
            
            # Agregar información del rango de fechas
            fig_tendencia.add_annotation(
                text=f"Período: {fecha_min.strftime('%m/%Y')} - {fecha_max.strftime('%m/%Y')}",
                xref="paper", yref="paper",
                x=0.02, y=0.98, 
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)"
            )
            
            st.plotly_chart(fig_tendencia, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para mostrar la tendencia temporal.")
    
    if 'Nombre del equipo' in data.columns:
        st.subheader("🔧 Distribución por Tipo de Equipo")
        equipos_count = data['Nombre del equipo'].value_counts().head(10)
        fig_equipos = px.bar(x=equipos_count.values, y=equipos_count.index,
                           orientation='h', title="Top 10 Equipos con Más Eventos")
        fig_equipos.update_yaxes(title="Tipo de Equipo")
        fig_equipos.update_xaxes(title="Cantidad de Eventos")
        st.plotly_chart(fig_equipos, use_container_width=True)
    
    if 'Marca del equipo' in data.columns:
        st.subheader("🏷️ Distribución por Marca")
        marcas_count = data['Marca del equipo'].value_counts().head(8)
        fig_marcas = px.pie(values=marcas_count.values, names=marcas_count.index,
                          title="Distribución de Eventos por Marca")
        st.plotly_chart(fig_marcas, use_container_width=True)

# === CONFIGURACIÓN VISUAL CON IMÁGENES ===
# Rutas de las imágenes (ajusta estas rutas según tu estructura de archivos)
banner_path = "C:/Users/ana cristina/Documents/QUINTOAÑO10/PROYECTO FIN DE CARRERA/Modelos/foto1.jpg"
logo_path = "C:/Users/ana cristina/Documents/QUINTOAÑO10/PROYECTO FIN DE CARRERA/Modelos/images.jpeg"

# Obtener imágenes en base64
encoded_banner = get_base64_image(banner_path)
encoded_logo = get_base64_image(logo_path)

# === ESTILOS CSS MEJORADOS ===
st.markdown(f"""
    <style>
    .stApp {{ background-color: #F6FFF5; }}
    .main-container {{ background-color: rgba(255, 255, 255, 0.90); padding: 2rem; border-radius: 15px; }}
    h1, h2, h3 {{ color: #247A55; }}
    .stButton > button {{ 
        background-color: #FFF9DB; 
        color: #000; 
        border-radius: 8px; 
        border: 2px solid #247A55;
        transition: all 0.3s ease;
    }}
    .stButton > button:hover {{
        background-color: #247A55;
        color: white;
    }}
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
        border-radius: 10px;
    }}
    .metric-card {{
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #247A55;
    }}
    .alert-box {{
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }}
    </style>
    <div class="custom-headline">Modelo Predictivo de Fallas en Equipos Biomédicos v5.0</div>
""", unsafe_allow_html=True)

# === CARGA DE DATOS Y ENTRENAMIENTO ===
df = cargar_y_procesar_datos()

# Verificar calidad de datos
verificar_calidad_datos(df)

# Debug de fechas (temporal)
if not df.empty and 'Fecha del incidente' in df.columns:
    with st.expander("🔍 Debug de fechas (información técnica)"):
        st.write(f"**Fecha más antigua:** {df['Fecha del incidente'].min()}")
        st.write(f"**Fecha más reciente:** {df['Fecha del incidente'].max()}")
        st.write(f"**Total registros:** {len(df)}")
        st.write(f"**Años presentes:** {sorted(df['Fecha del incidente'].dt.year.unique())}")

# Entrenar modelo si no está en cache
if 'modelo_entrenado' not in st.session_state:
    with st.spinner("Entrenando modelo..."):
        try:
            modelo_entrenado, mae_score, r2_score, datos_modelo = entrenar_modelo_completo(df)
            st.session_state['modelo_entrenado'] = modelo_entrenado
            st.session_state['mae_score'] = mae_score
            st.session_state['r2_score'] = r2_score
            st.session_state['datos_modelo'] = datos_modelo
            
            if modelo_entrenado is not None:
                st.session_state['modelo_stats'] = f"MAE: {mae_score:.2f} días, R²: {r2_score:.3f}"
            else:
                st.session_state['modelo_stats'] = "Modelo usando valores por defecto (mediana)"
        except Exception as e:
            st.error(f"Error al entrenar el modelo: {str(e)}")
            st.session_state['modelo_stats'] = "Error en el modelo"

# --- TÍTULO CON CONTAINER ---
st.markdown("""<div class="title-container">
    <h3>🏥 Visualización, Monitoreo y Predicción de Eventos Críticos para Equipos UCI</h3>
    <p>Sistema integrado de análisis predictivo para equipos biomédicos críticos v5.0</p>
</div>""", unsafe_allow_html=True)

# === SIDEBAR CON LOGO ===
with st.sidebar:
    # Mostrar logo si existe
    if encoded_logo:
        st.markdown(f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <img src="data:image/jpeg;base64,{encoded_logo}" width="150" style="border-radius: 10px;">
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("🏥 **Sistema Biomédico**")
    
    st.title("🔍 Navegación")
    opcion = st.radio("Ir a:", [
        "📊 Dashboard Principal", 
        "📈 Análisis Avanzado",
        "➕ Nuevo evento", 
        "📋 Predicción Individual",
        "🎯 Ranking Predictivo",
        "⚙️ Configuración"
    ])

# --- DASHBOARD PRINCIPAL ---
if opcion == "📊 Dashboard Principal":
    st.markdown("<h2>📊 Panel de Control Principal</h2>", unsafe_allow_html=True)
    
    crear_metricas_dashboard(df)
    
    # Estado del modelo
    if 'modelo_stats' in st.session_state:
        st.info(f"🤖 Estado del modelo: {st.session_state['modelo_stats']}")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📋 Historial de Eventos Recientes")
        if not df.empty:
            columns_to_show = ['Nombre del equipo', 'Marca del equipo', 'Fecha del incidente', 'Área usuaria']
            columns_to_show = [col for col in columns_to_show if col in df.columns]
            
            # Mostrar los más recientes primero
            data_reciente = df.sort_values('Fecha del incidente', ascending=False)[columns_to_show].head(10)
            st.dataframe(data_reciente, use_container_width=True, height=300)
        else:
            st.info("No hay eventos registrados.")
    
    with col2:
        st.markdown("#### 🎯 Predicciones Activas")
        if st.session_state.predicciones:
            df_pred = pd.DataFrame(st.session_state.predicciones)
            df_pred_reciente = df_pred.tail(5)
            
            # Mostrar predicciones recientes
            if 'equipo' in df_pred_reciente.columns:
                st.dataframe(df_pred_reciente[['equipo', 'probabilidad', 'fecha_proxima_falla']], 
                            use_container_width=True, height=200)
            else:
                st.dataframe(df_pred_reciente, use_container_width=True, height=200)
            
            # Gráfico de probabilidades
            if len(df_pred) > 0:
                top_pred = df_pred.nlargest(5, 'probabilidad') if 'probabilidad' in df_pred.columns else df_pred.head(5)
                fig_prob = px.bar(
                    top_pred,
                    x="equipo" if 'equipo' in top_pred.columns else 'Equipo',
                    y="probabilidad" if 'probabilidad' in top_pred.columns else 'Probabilidad de falla',
                    color="marca" if 'marca' in top_pred.columns else 'Marca',
                    title="Top 5 - Mayor Riesgo de Falla"
                )
                fig_prob.update_layout(height=300)
                st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.info("No hay predicciones disponibles.")

# --- ANÁLISIS AVANZADO ---
elif opcion == "📈 Análisis Avanzado":
    st.markdown("<h2>📈 Análisis Avanzado de Datos</h2>", unsafe_allow_html=True)
    
    crear_graficos_analisis(df)
    
    if not df.empty:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistics", "🔍 Feature Analysis", "📈 Trends", "🎯 Model Performance"])
        
        with tab1:
            st.markdown("### 📊 Estadísticas Descriptivas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'dias_hasta_falla' in df.columns:
                    st.write("**Días hasta falla:**")
                    st.write(f"- Media: {df['dias_hasta_falla'].mean():.2f}")
                    st.write(f"- Mediana: {df['dias_hasta_falla'].median():.2f}")
                    st.write(f"- Desviación estándar: {df['dias_hasta_falla'].std():.2f}")
                    st.write(f"- Mínimo: {df['dias_hasta_falla'].min():.2f}")
                    st.write(f"- Máximo: {df['dias_hasta_falla'].max():.2f}")
                    
                    # Histograma
                    fig_hist = px.histogram(
                        df, 
                        x='dias_hasta_falla', 
                        nbins=30,
                        title="Distribución de Días hasta Falla"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Top equipos con más fallas
                if 'Nombre del equipo' in df.columns:
                    equipo_counts = df['Nombre del equipo'].value_counts().head(10)
                    fig_equipos = px.bar(
                        x=equipo_counts.index,
                        y=equipo_counts.values,
                        title="Top 10 Equipos con Más Eventos"
                    )
                    fig_equipos.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_equipos, use_container_width=True)
        
        with tab2:
            st.markdown("### 🔍 Análisis de Características")
            
            if 'datos_modelo' in st.session_state and st.session_state['datos_modelo'] is not None:
                datos_modelo = st.session_state['datos_modelo']
                
                # Correlaciones
                numeric_cols = ['dias_hasta_falla', 'num_fallas_previas', 'days_since_last_failure', 
                               'failure_rate', 'equipment_age_days', 'recent_failures_30d']
                numeric_cols = [col for col in numeric_cols if col in datos_modelo.columns]
                
                if len(numeric_cols) > 1:
                    corr_matrix = datos_modelo[numeric_cols].corr()
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        title="Matriz de Correlación de Características Numéricas"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                # Análisis por marca
                if 'Marca del equipo' in datos_modelo.columns:
                    marca_stats = datos_modelo.groupby('Marca del equipo')['dias_hasta_falla'].agg(['mean', 'count']).reset_index()
                    marca_stats = marca_stats[marca_stats['count'] >= 3].sort_values('mean')
                    
                    fig_marca = px.bar(
                        marca_stats,
                        x='Marca del equipo',
                        y='mean',
                        title="Promedio de Días hasta Falla por Marca"
                    )
                    fig_marca.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_marca, use_container_width=True)
        
        with tab3:
            st.markdown("### 📈 Análisis de Tendencias Temporales")
            
            if 'Fecha del incidente' in df.columns:
                # Tendencia mensual
                df_temp = df.copy()
                df_temp['año_mes'] = df_temp['Fecha del incidente'].dt.to_period('M')
                eventos_mensuales = df_temp.groupby('año_mes').size().reset_index(name='cantidad')
                eventos_mensuales['año_mes'] = eventos_mensuales['año_mes'].astype(str)
                
                # Ordenar correctamente
                eventos_mensuales['fecha_sort'] = pd.to_datetime(eventos_mensuales['año_mes'])
                eventos_mensuales = eventos_mensuales.sort_values('fecha_sort')
                
                fig_tendencia = px.line(
                    eventos_mensuales,
                    x='año_mes',
                    y='cantidad',
                    title="Tendencia Mensual de Eventos"
                )
                fig_tendencia.update_xaxes(tickangle=45)
                st.plotly_chart(fig_tendencia, use_container_width=True)
                
                # Análisis temporal detallado
                col1, col2 = st.columns(2)
                
                with col1:
                    df_temp['Día_Semana'] = df_temp['Fecha del incidente'].dt.day_name()
                    dias_semana = df_temp['Día_Semana'].value_counts()
                    fig_dias = px.bar(x=dias_semana.index, y=dias_semana.values,
                                    title="Eventos por Día de la Semana")
                    st.plotly_chart(fig_dias, use_container_width=True)
                
                with col2:
                    df_temp['Mes'] = df_temp['Fecha del incidente'].dt.month_name()
                    meses = df_temp['Mes'].value_counts()
                    fig_meses = px.bar(x=meses.index, y=meses.values,
                                     title="Eventos por Mes del Año")
                    fig_meses.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_meses, use_container_width=True)
        
        with tab4:
            st.markdown("### 🎯 Rendimiento del Modelo")
            
            if 'modelo_entrenado' in st.session_state and st.session_state['modelo_entrenado'] is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("MAE (Error Absoluto Medio)", f"{st.session_state.get('mae_score', 0):.2f} días")
                    st.metric("R² Score", f"{st.session_state.get('r2_score', 0):.3f}")
                
                with col2:
                    st.write("**Información del modelo:**")
                    st.write("- Algoritmo: Ensemble (XGBoost + Random Forest)")
                    st.write("- Validación: Time Series Split")
                    st.write("- Transformación: Log(días + 1)")
                    st.write("- Preprocesamiento: RobustScaler + OneHotEncoder")
                
                # Mostrar predicciones recientes
                if st.session_state.predicciones:
                    st.markdown("#### 🎯 Últimas Predicciones")
                    pred_df = pd.DataFrame(st.session_state.predicciones[-10:])  # Últimas 10
                    st.dataframe(pred_df, use_container_width=True)
            else:
                st.warning("El modelo no está disponible o no se ha entrenado correctamente.")

# --- NUEVO EVENTO ---
elif opcion == "➕ Nuevo evento":
    st.markdown("<h2>➕ Registrar Nuevo Evento</h2>", unsafe_allow_html=True)

    with st.form("form_nuevo_evento"):
        col1, col2 = st.columns(2)
        
        with col1:
            nombre_opciones = list(df["Nombre del equipo"].dropna().unique()) + ["Otro"] if not df.empty else ["ventilador mecánico", "monitor multiparámetro", "aspirador de secreciones", "Otro"]
            nombre = st.selectbox("Nombre del equipo", nombre_opciones, key="new_eq")
            if nombre == "Otro":
                nombre = st.text_input("Especificar nuevo nombre del equipo")

            marca_opciones = list(df["Marca del equipo"].dropna().unique()) + ["Otro"] if not df.empty else ["NIHON KOHDEN", "MAQUET", "GENERAL ELECTRIC", "HAMILTON", "INSPITAL", "GIMA", "THOMAS", "CA-MI", "KELING", "Otro"]
            marca = st.selectbox("Marca del equipo", marca_opciones, key="new_marca")
            if marca == "Otro":
                marca = st.text_input("Especificar nueva marca del equipo")

            numero_serie = st.text_input("Número de serie del equipo")
            tipo = st.radio("Tipo de evento", ["Falla", "Preventivo", "Correctivo"])
        
        with col2:
            fecha = st.date_input("Fecha del incidente", value=date.today())
            
            area_opciones = list(df["Área usuaria"].dropna().unique()) + ["Otro"] if not df.empty and "Área usuaria" in df.columns else ["UCI", "Quirófano", "Emergencia", "Otro"]
            area = st.selectbox("Área usuaria", area_opciones)
            if area == "Otro":
                area = st.text_input("Especificar nueva área")
                
            garantia = st.radio("¿Estaba en garantía?", ["SI", "NO"])
            
            diagnostico = st.text_area("Diagnóstico técnico", placeholder="Descripción del problema o mantenimiento realizado")
            
            accesorio_opciones = ["Brazalete de presión arterial", "Cable ECG", "Capnógrafo", "Sensor de SPO2", "Ninguno", "Otro"]
            accesorio = st.selectbox("Accesorio afectado", accesorio_opciones)
            if accesorio == "Otro":
                accesorio = st.text_input("Especificar accesorio")
        
        submitted = st.form_submit_button("💾 Guardar Evento", type="primary")
        
        if submitted:
            if nombre and marca and numero_serie:
                try:
                    guardar_evento(nombre, marca, numero_serie, tipo, fecha, garantia, diagnostico, area_usuaria=area, accesorio=accesorio)
                    st.success("✅ Evento guardado correctamente en la base de datos.")
                    # Limpiar cache para recargar datos
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error al guardar: {str(e)}")
            else:
                st.error("❌ Por favor completa todos los campos obligatorios.")

# --- PREDICCIÓN INDIVIDUAL ---
elif opcion == "📋 Predicción Individual":
    st.markdown("<h2>🔮 Predicción Individual de Falla</h2>", unsafe_allow_html=True)

    with st.form("form_prediccion"):
        col1, col2 = st.columns(2)
        
        with col1:
            nombre_opciones = list(df["Nombre del equipo"].dropna().unique()) if not df.empty else ["ventilador mecánico", "monitor multiparámetro", "aspirador de secreciones"]
            nombre = st.selectbox("Nombre del equipo", nombre_opciones)
            
            marca_opciones = list(df["Marca del equipo"].dropna().unique()) if not df.empty else ["NIHON KOHDEN", "MAQUET", "GENERAL ELECTRIC"]
            marca = st.selectbox("Marca del equipo", marca_opciones)
        
        with col2:
            area_opciones = list(df["Área usuaria"].dropna().unique()) if not df.empty and "Área usuaria" in df.columns else ["UCI", "Quirófano", "Emergencia"]
            area = st.selectbox("Área usuaria", area_opciones)
            
            fecha_prediccion = st.date_input("Fecha de predicción", value=date.today())
        
        submitted = st.form_submit_button("🔮 Predecir", type="primary")
        
        if submitted:
            with st.spinner("Generando predicción..."):
                try:
                    datos_basicos = {
                        'Nombre del equipo': nombre,
                        'Marca del equipo': marca,
                        'Área usuaria': area
                    }
                    
                    dias_estimados, probabilidad = predecir_desde_ultima_falla_modelo5(
                        datos_basicos, 
                        st.session_state.get('datos_modelo', df), 
                        st.session_state.get('modelo_entrenado'), 
                        fecha_prediccion
                    )
                    
                    if dias_estimados is not None:
                        # Buscar última falla del equipo
                        hist_equipo = df[df['Nombre del equipo'].str.lower() == nombre.lower()]
                        if not hist_equipo.empty:
                            ultima_falla = hist_equipo['Fecha del incidente'].max()
                            dias_desde_ultima = (datetime.now() - ultima_falla).days
                            fecha_estimada = ultima_falla + timedelta(days=dias_estimados)
                            
                            st.success(f"✅ **Predicción generada exitosamente**")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Días hasta próxima falla", f"{dias_estimados:.1f}")
                            with col2:
                                st.metric("Probabilidad de falla", f"{probabilidad:.1f}%")
                            with col3:
                                st.metric("Días desde última falla", f"{dias_desde_ultima}")
                            
                            st.info(f"📅 **Fecha estimada de próxima falla:** {fecha_estimada.strftime('%d/%m/%Y')}")
                            st.info(f"📅 **Última falla registrada:** {ultima_falla.strftime('%d/%m/%Y')}")
                            
                        else:
                            fecha_estimada = datetime.now() + timedelta(days=dias_estimados)
                            st.success(f"✅ **Predicción:** {dias_estimados:.1f} días desde hoy")
                            st.info(f"📅 **Fecha estimada:** {fecha_estimada.strftime('%d/%m/%Y')}")
                            st.warning("⚠️ No se encontró historial previo para este equipo")
                        
                        # Guardar predicción
                        prediccion = {
                            'equipo': nombre,
                            'marca': marca,
                            'area': area,
                            'dias_estimados': dias_estimados,
                            'fecha_proxima_falla': fecha_estimada.strftime('%d/%m/%Y') if 'fecha_estimada' in locals() else (datetime.now() + timedelta(days=dias_estimados)).strftime('%d/%m/%Y'),
                            'probabilidad': probabilidad,
                            'fecha_prediccion': fecha_prediccion.strftime('%d/%m/%Y')
                        }
                        st.session_state.predicciones.append(prediccion)
                        
                    else:
                        st.error("❌ No se pudo generar la predicción.")
                        
                except Exception as e:
                    st.error(f"❌ Error al generar predicción: {str(e)}")
                    logging.error(f"Error en predicción: {e}")






# También mostrar info de los datos

# --- RANKING PREDICTIVO ---
elif opcion == "🎯 Ranking Predictivo":
    st.markdown("<h2>🎯 Ranking Predictivo de Equipos</h2>", unsafe_allow_html=True)
    
    # Importar dependencias de Plotly
    try:
        import plotly.express as px
        plotly_available = True
    except ImportError:
        plotly_available = False
        st.warning("⚠️ Plotly no está disponible. Los gráficos no se mostrarán.")
    
    # Cargar el ranking desde el archivo CSV
    try:
        ranking_predictorg = pd.read_csv('ranking_predictivo_ml.csv')
        st.session_state['ranking_predictorg'] = ranking_predictorg
    except FileNotFoundError:
        st.error("❌ Error: No se encontró el archivo 'ranking_predictivo_ml.csv'. Asegúrate de que predictorg.py se haya ejecutado correctamente.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error al cargar 'ranking_predictivo_ml.csv': {str(e)}")
        st.stop()
    
    # Depuración: Mostrar contenido de ranking_predictivo_ml.csv
    with st.expander("🔍 Depuración: Contenido de ranking_predictivo_ml.csv"):
        st.write("**Equipos en ranking_predictorg:**")
        st.write(ranking_predictorg['Nombre del equipo'].unique())
        st.write("**Marcas en ranking_predictorg:**")
        st.write(ranking_predictorg['Marca del equipo'].unique())
        st.write("**Primeras filas de ranking_predictorg:**")
        st.dataframe(ranking_predictorg.head())
    
    # Verificar disponibilidad del ranking
    ranking_disponible = st.session_state.get('ranking_predictorg') is not None
    
    if not ranking_disponible:
        st.error("❌ Error: Ranking de predictorg.py no está disponible.")
        st.stop()
    
    st.markdown("---")
    
    # Botón para generar el ranking
    if st.button("🔄 Generar Ranking Completo", type="primary", key="btn_ranking"):
        with st.spinner("Generando ranking predictivo para todos los equipos..."):
            try:
                ranking_data = []
                equipos_exitosos = 0
                equipos_con_error = 0
                total_equipos = len(ranking_predictorg)
                
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                status_placeholder.write(f"🔄 Procesando {total_equipos} equipos...")
                
                # Normalizar nombres en ranking_predictorg
                ranking_predictorg['Nombre del equipo'] = ranking_predictorg['Nombre del equipo'].str.strip().str.lower()
                ranking_predictorg['Marca del equipo'] = ranking_predictorg['Marca del equipo'].str.strip().str.lower()
                
                # Procesar cada equipo en ranking_predictorg
                for i, row in ranking_predictorg.iterrows():
                    try:
                        equipo = row['Nombre del equipo']
                        marca = row['Marca del equipo']
                        probabilidad = row['Probabilidad_Fallo'] * 100  # Convertir a porcentaje
                        incidentes = row['Incidentes']
                        frecuencia_fallas = row['Frecuencia_fallas']
                        porcentaje_incidentes = row['Porcentaje_Incidentes']
                        
                        # Determinar categoría de incidentes (para tablas y gráficas)
                        if incidentes > 60:
                            categoria_incidentes = "🔴 Alto"
                        elif incidentes >= 40:
                            categoria_incidentes = "🟡 Medio"
                        else:
                            categoria_incidentes = "🟢 Bajo"
                        
                        ranking_data.append({
                            'Equipo': equipo.title(),  # Capitalizar para presentación
                            'Marca': marca.title(),
                            'Nivel de Riesgo': categoria_incidentes,  # Basado en Incidentes
                            'Incidentes': incidentes,
                            'Frecuencia de fallas (por año)': round(frecuencia_fallas, 2),
                            'Porcentaje de Incidentes': round(porcentaje_incidentes, 2),
                            'Categoria Incidentes': categoria_incidentes  # Para gráficas
                        })
                        
                        equipos_exitosos += 1
                        
                    except Exception as e:
                        equipos_con_error += 1
                        st.warning(f"⚠️ Error en {equipo}: {str(e)}")
                        continue
                    
                    # Actualizar barra de progreso
                    progress_bar.progress((i + 1) / total_equipos)
                
                # Limpiar el status
                status_placeholder.empty()
                
                # Mostrar métricas de procesamiento
                st.markdown("### 📊 Resumen del Procesamiento")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("✅ Exitosos", equipos_exitosos)
                with col2:
                    st.metric("❌ Con errores", equipos_con_error)
                with col3:
                    st.metric("📊 Total", total_equipos)
                
                if ranking_data:
                    ranking_df = pd.DataFrame(ranking_data)
                    ranking_df = ranking_df.sort_values('Incidentes', ascending=False).reset_index(drop=True)
                    
                    st.success(f"✅ Ranking generado para {len(ranking_df)} equipos")
                    
                    # Filtrar equipos críticos
                    equipos_criticos = ['ventilador mecánico', 'monitor multiparámetro', 'aspirador de secreciones']
                    ranking_criticos = ranking_df[ranking_df['Equipo'].str.lower().isin([eq.lower() for eq in equipos_criticos])]
                    
                    # Priorizar equipos específicos
                    ranking_df['Prioridad'] = ranking_df.apply(
                        lambda x: 1 if (x['Marca'].lower() == 'drager' and x['Equipo'].lower() == 'vista 120') or 
                                      (x['Marca'].lower() == 'northen' and x['Equipo'].lower() == 'crius v6') else 0, 
                        axis=1
                    )
                    ranking_df = ranking_df.sort_values(['Prioridad', 'Incidentes'], ascending=[False, False]).reset_index(drop=True)
                    ranking_df = ranking_df.drop(columns=['Prioridad'])
                    
                    st.markdown("#### 🚨 Equipos Críticos")
                    if not ranking_criticos.empty:
                        st.dataframe(ranking_criticos.drop(columns=['Categoria Incidentes']), use_container_width=True)
                        
                        if plotly_available:
                            fig_criticos = px.bar(
                                ranking_criticos,
                                x='Equipo',
                                y='Incidentes',
                                color='Categoria Incidentes',
                                color_discrete_map={
                                    '🔴 Alto': 'red',
                                    '🟡 Medio': 'yellow',
                                    '🟢 Bajo': 'green'
                                },
                                title="Equipos Críticos por Número de Incidentes",
                                hover_data=['Marca', 'Nivel de Riesgo', 'Frecuencia de fallas (por año)', 'Porcentaje de Incidentes']
                            )
                            fig_criticos.update_xaxes(tickangle=45, title_text="Equipo")
                            fig_criticos.update_yaxes(title_text="Número de Incidentes")
                            fig_criticos.update_layout(
                                showlegend=True,
                                height=400,
                                margin=dict(l=50, r=50, t=80, b=100)
                            )
                            st.plotly_chart(fig_criticos, use_container_width=True)
                        else:
                            st.warning("Plotly no está disponible. Mostrando solo tabla.")
                    else:
                        st.info("No hay datos de equipos críticos en el ranking.")
                    
                    st.markdown("#### 📋 Ranking Completo")
                    st.dataframe(ranking_df.drop(columns=['Categoria Incidentes']), use_container_width=True)
                    
                    if plotly_available:
                        fig_ranking = px.bar(
                            ranking_df.head(10),
                            x='Equipo',
                            y='Incidentes',
                            color='Categoria Incidentes',
                            color_discrete_map={
                                '🔴 Alto': 'red',
                                '🟡 Medio': 'yellow',
                                '🟢 Bajo': 'green'
                            },
                            title="Top 10 Equipos por Número de Incidentes",
                            hover_data=['Marca', 'Nivel de Riesgo', 'Frecuencia de fallas (por año)', 'Porcentaje de Incidentes']
                        )
                        fig_ranking.update_xaxes(tickangle=45, title_text="Equipo")
                        fig_ranking.update_yaxes(title_text="Número de Incidentes")
                        fig_ranking.update_layout(
                            showlegend=True,
                            height=400,
                            margin=dict(l=50, r=50, t=80, b=100)
                        )
                        st.plotly_chart(fig_ranking, use_container_width=True)
                    else:
                        st.warning("Plotly no está disponible. Mostrando solo tabla.")
                else:
                    st.error("❌ No se generaron datos para el ranking.")
                    
            except Exception as e:
                st.error(f"❌ Error al generar el ranking: {str(e)}")
                logging.error(f"Error en ranking: {e}")



















# --- CONFIGURACIÓN ---
elif opcion == "⚙️ Configuración":
    st.markdown("<h2>⚙️ Configuración del Sistema</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🗄️ Gestión de Datos")
        if st.button("🔄 Recargar Datos"):
            st.cache_data.clear()
            st.rerun()
        if st.button("🧹 Limpiar Caché"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ Caché limpiado")
        
        st.markdown("#### 📊 Estadísticas del Sistema")
        total_eventos = len(df) if not df.empty else 0
        st.metric("Total de eventos", total_eventos)
        if st.session_state.predicciones:
            total_predicciones = len(st.session_state.predicciones)
            st.metric("Predicciones realizadas", total_predicciones)
    
    with col2:
        st.markdown("#### 🔧 Configuración del Modelo")
        if st.button("🤖 Reentrenar Modelo"):
            with st.spinner("Reentrenando modelo..."):
                try:
                    # Limpiar cache y recargar datos
                    st.cache_data.clear()
                    df_nuevo = cargar_y_procesar_datos()
                    modelo_entrenado, mae, r2, df_modelo = entrenar_modelo_completo(df_nuevo)
                    
                    st.session_state['modelo_entrenado'] = modelo_entrenado
                    st.session_state['mae_score'] = mae
                    st.session_state['r2_score'] = r2
                    st.session_state['datos_modelo'] = df_modelo
                    
                    if modelo_entrenado is not None:
                        st.success(f"✅ Modelo reentrenado - MAE: {mae:.2f}, R²: {r2:.3f}")
                        st.session_state['modelo_stats'] = f"MAE: {mae:.2f}, R²: {r2:.3f}"
                    else:
                        st.warning("⚠️ Datos insuficientes para entrenar el modelo")
                        st.session_state['modelo_stats'] = "Modelo usando valores por defecto (mediana)"
                except Exception as e:
                    st.error(f"❌ Error al reentrenar: {str(e)}")
        
        st.markdown("#### ℹ️ Información del Modelo v5.0")
        st.info("""
        **Modelo Actual:** Ensemble (XGBoost + Random Forest)
        **Características:**
        - Predicción de días hasta falla desde la última falla
        - Feature engineering avanzado (30+ características)
        - Validación temporal (TimeSeriesSplit)
        - Transformación logarítmica del target
        - Filtrado automático de outliers
        - Análisis de equipos críticos UCI
        """)
        
        with st.expander("🔧 Parámetros Avanzados v5.0"):
            st.write("**Features principales:**")
            st.write("• Días desde última falla")
            st.write("• Número de fallas previas")
            st.write("• Tasa de fallas por año")
            st.write("• Edad del equipo")
            st.write("• Fallas recientes (30 días)")
            st.write("• Características temporales (mes, día semana, trimestre)")
            st.write("• Estadísticas por marca y área")
            st.write("**Filtros aplicados:**")
            st.write("• Eliminación de duplicados exactos")
            st.write("• Filtrado de outliers (Q5-Q95)")
            st.write("• Manejo de valores faltantes")
