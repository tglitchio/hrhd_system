import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime

# Fecha actual para cálculos
CURRENT_DATE = pd.to_datetime('2025-08-01')

# Cargar los datos
try:
    df = pd.read_excel('datos1.xlsx')
    print("Datos cargados exitosamente. Número de registros:", len(df))
except FileNotFoundError:
    print("Error: No se encontró el archivo 'datos1.xlsx'.")
    exit()
except Exception as e:
    print(f"Error al cargar el archivo: {e}")
    exit()

# Verificar valores únicos en 'Tipo'
print("Valores únicos en 'Tipo':", df['Tipo'].unique())
print("Conteo de valores en 'Tipo':", df['Tipo'].value_counts(dropna=False))

# Normalizar y filtrar por 'Tipo'
df['Tipo'] = df['Tipo'].fillna('Desconocido').str.strip().str.lower()
df = df[df['Tipo'].isin(['falla', 'correctivo'])]
print("Número de registros tras filtrar por Tipo ('falla' o 'correctivo'):", len(df))

if len(df) == 0:
    print("Error: No hay registros con Tipo 'falla' o 'correctivo'.")
    exit()

# Normalizar y filtrar por 'Retirado del servicio'
print("Valores únicos en 'Retirado del servicio':", df['Retirado del servicio'].unique())
df['Retirado del servicio'] = df['Retirado del servicio'].fillna('no').str.strip().str.lower()
df = df[df['Retirado del servicio'] != 'sí']
print("Número de registros tras filtrar por equipos no retirados:", len(df))

if len(df) == 0:
    print("Error: No hay registros de equipos no retirados.")
    exit()

# Manejar valores nulos
df['Marca del equipo'] = df['Marca del equipo'].fillna('Desconocido').astype(str)
df['Nombre del equipo'] = df['Nombre del equipo'].fillna('Desconocido').astype(str)

# Verificar si 'Nombre del equipo' existe
if 'Nombre del equipo' not in df.columns:
    print("Error: La columna 'Nombre del equipo' no existe en el dataset. Por favor, verifica el nombre de la columna.")
    exit()

# Procesar Fecha del incidente
try:
    df['Fecha del incidente'] = pd.to_datetime(df['Fecha del incidente'], dayfirst=True)
    df['Antiguedad_incidente'] = (CURRENT_DATE - df['Fecha del incidente']).dt.days
    df['Incidentes_ultimo_año'] = df['Fecha del incidente'].apply(
        lambda x: 1 if (CURRENT_DATE - x).days <= 365 else 0
    )
except Exception as e:
    print(f"Error al procesar 'Fecha del incidente': {e}")
    exit()

# Crear variable objetivo
df['Fallo'] = 1  # Todos los registros reales son fallos

# Calcular número de incidentes y frecuencia de fallas
incidentes = df.groupby(['Marca del equipo', 'Nombre del equipo']).agg({
    'Fallo': 'count',  # Número total de incidentes
    'Fecha del incidente': ['min', 'max']  # Rango de fechas
}).reset_index()
incidentes.columns = ['Marca del equipo', 'Nombre del equipo', 'Incidentes', 'Fecha_min', 'Fecha_max']

# Calcular el período en años para la frecuencia
incidentes['Periodo_años'] = (incidentes['Fecha_max'] - incidentes['Fecha_min']).dt.days / 365.25
incidentes['Periodo_años'] = incidentes['Periodo_años'].replace(0, 1)  # Evitar división por 0
incidentes['Frecuencia_fallas'] = incidentes['Incidentes'] / incidentes['Periodo_años']
incidentes = incidentes[['Marca del equipo', 'Nombre del equipo', 'Incidentes', 'Frecuencia_fallas']]

# Guardar valores originales para el ranking
incidentes_original = incidentes.copy()

# Unir frecuencia al DataFrame principal
df = df.merge(incidentes, on=['Marca del equipo', 'Nombre del equipo'], how='left')

# Normalizar características numéricas
scaler = MinMaxScaler()
df[['Antiguedad_incidente', 'Incidentes', 'Incidentes_ultimo_año', 'Frecuencia_fallas']] = scaler.fit_transform(
    df[['Antiguedad_incidente', 'Incidentes', 'Incidentes_ultimo_año', 'Frecuencia_fallas']]
)

# Codificar variables categóricas
le_marca = LabelEncoder()
le_nombre = LabelEncoder()
df['Marca del equipo_encoded'] = le_marca.fit_transform(df['Marca del equipo'])
df['Nombre del equipo_encoded'] = le_nombre.fit_transform(df['Nombre del equipo'])

# Generar datos sintéticos
unique_combinations = df[['Marca del equipo', 'Nombre del equipo', 'Marca del equipo_encoded', 'Nombre del equipo_encoded']].drop_duplicates()
synthetic_data = []
for _, row in unique_combinations.iterrows():
    marca = row['Marca del equipo_encoded']
    nombre = row['Nombre del equipo_encoded']
    num_incidentes = len(df[(df['Marca del equipo_encoded'] == marca) & (df['Nombre del equipo_encoded'] == nombre)])
    for _ in range(3 * max(1, num_incidentes)):  # Tres no-incidentes por incidente
        sample = df[(df['Marca del equipo_encoded'] == marca) & (df['Nombre del equipo_encoded'] == nombre)].iloc[0].copy()
        sample['Fallo'] = 0
        min_date = df['Fecha del incidente'].min()
        max_date = df['Fecha del incidente'].max()
        random_days = np.random.randint(0, (max_date - min_date).days, dtype=np.int64)
        sample['Fecha del incidente'] = min_date + pd.Timedelta(days=random_days)
        sample['Antiguedad_incidente'] = (CURRENT_DATE - sample['Fecha del incidente']).days
        # Crear DataFrame para transformar con MinMaxScaler
        temp_df = pd.DataFrame({
            'Antiguedad_incidente': [sample['Antiguedad_incidente']],
            'Incidentes': [0],
            'Incidentes_ultimo_año': [0],
            'Frecuencia_fallas': [0]
        })
        temp_scaled = scaler.transform(temp_df)
        sample['Antiguedad_incidente'] = temp_scaled[0][0]
        sample['Incidentes'] = temp_scaled[0][1]
        sample['Incidentes_ultimo_año'] = temp_scaled[0][2]
        sample['Frecuencia_fallas'] = temp_scaled[0][3]
        synthetic_data.append(sample)

# Combinar datos reales y sintéticos
df_synthetic = pd.DataFrame(synthetic_data)
df = pd.concat([df, df_synthetic], ignore_index=True)

# Características y variable objetivo
X = df[['Marca del equipo_encoded', 'Nombre del equipo_encoded', 'Año del evento', 'Antiguedad_incidente', 'Incidentes', 'Incidentes_ultimo_año', 'Frecuencia_fallas']]
y = df['Fallo']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar y entrenar el modelo
model = CatBoostClassifier(iterations=500, depth=8, learning_rate=0.05, cat_features=[0, 1], scale_pos_weight=3, verbose=0)
model.fit(X_train, y_train)

# Predecir probabilidades
predictions = []
for _, row in incidentes_original.iterrows():
    marca = le_marca.transform([row['Marca del equipo']])[0]
    nombre = le_nombre.transform([row['Nombre del equipo']])[0]
    sample = df[(df['Marca del equipo_encoded'] == marca) & (df['Nombre del equipo_encoded'] == nombre)].iloc[-1][X.columns].copy()
    prob = model.predict_proba([sample])[0][1]
    predictions.append({
        'Marca del equipo': row['Marca del equipo'],
        'Nombre del equipo': row['Nombre del equipo'],
        'Probabilidad_Fallo': prob,
        'Incidentes': row['Incidentes'],
        'Frecuencia_fallas': row['Frecuencia_fallas']
    })

# Crear ranking
ranking = pd.DataFrame(predictions)
ranking['Porcentaje_Incidentes'] = (ranking['Incidentes'] / df[df['Fallo'] == 1].shape[0] * 100).round(2)
# Combinar probabilidad y frecuencia de fallas
ranking['Puntaje_Combinado'] = ranking['Probabilidad_Fallo'] * ranking['Frecuencia_fallas'] * ranking['Incidentes']
ranking = ranking.sort_values(by='Puntaje_Combinado', ascending=False).reset_index(drop=True)

# Mostrar ranking
print("\nRanking predictivo de equipos por probabilidad de fallo:")
print(ranking[['Marca del equipo', 'Nombre del equipo', 'Probabilidad_Fallo', 'Incidentes', 'Frecuencia_fallas', 'Porcentaje_Incidentes']].head(10))

# Guardar ranking
ranking.to_csv('ranking_predictivo_ml.csv', index=False)
print("Ranking guardado en 'ranking_predictivo_ml.csv'.")

# Gráfico de barras
plt.figure(figsize=(12, 6))
colors = ['#FF5555' if inc > 3 else '#36A2EB' for inc in ranking['Incidentes'][:10]]
bars = plt.bar(ranking['Marca del equipo'][:10] + ' - ' + ranking['Nombre del equipo'][:10], ranking['Probabilidad_Fallo'][:10], color=colors)
plt.xlabel('Marca y Nombre del Equipo')
plt.ylabel('Probabilidad de Fallo')
plt.title('Top 10 Equipos con Mayor Riesgo de Fallo')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
for bar, inc, freq in zip(bars, ranking['Incidentes'][:10], ranking['Frecuencia_fallas'][:10]):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2%}\n({inc} inc, {freq:.2f}/año)', ha='center', va='bottom')
plt.show()

# Gráfico de dispersión
plt.figure(figsize=(10, 6))
plt.scatter(ranking['Incidentes'], ranking['Probabilidad_Fallo'], s=ranking['Frecuencia_fallas']*100, alpha=0.5)
for i, (marca, nombre) in enumerate(zip(ranking['Marca del equipo'][:10], ranking['Nombre del equipo'][:10])):
    plt.annotate(f"{marca} - {nombre}", (ranking['Incidentes'][i], ranking['Probabilidad_Fallo'][i]))
plt.xlabel('Número de Incidentes')
plt.ylabel('Probabilidad de Fallo')
plt.title('Probabilidad de Fallo vs. Número de Incidentes')
plt.grid(True)
plt.show()
