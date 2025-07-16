import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- FUNCIÓN PARA CALCULAR CARACTERÍSTICAS TEMPORALES ---
def get_previous_features(group):
    group = group.sort_values('Fecha del incidente')
    group['Num_Mantenimientos_Previos'] = group['Tipo'].shift(1).apply(lambda x: 1 if x in ['Preventivo', 'Correctivo'] else 0).cumsum()

    mantenimientos = group[group['Tipo'].isin(['Preventivo', 'Correctivo'])]
    group['Días_Desde_Ultimo_Mantenimiento'] = np.nan
    for idx, row in group.iterrows():
        prev_mant = mantenimientos[mantenimientos['Fecha del incidente'] < row['Fecha del incidente']]
        if not prev_mant.empty:
            group.loc[idx, 'Días_Desde_Ultimo_Mantenimiento'] = (row['Fecha del incidente'] - prev_mant['Fecha del incidente'].iloc[-1]).days

    fallas = group[group['Tipo'] == 'Falla']
    group['Días_Entre_Falla_1'] = np.nan
    group['Días_Entre_Falla_2'] = np.nan
    group['Días_Entre_Falla_3'] = np.nan
    for idx, row in group[group['Tipo'] == 'Falla'].iterrows():
        prev_fallas = fallas[fallas['Fecha del incidente'] < row['Fecha del incidente']]
        if len(prev_fallas) >= 1:
            group.loc[idx, 'Días_Entre_Falla_1'] = (row['Fecha del incidente'] - prev_fallas['Fecha del incidente'].iloc[-1]).days
        if len(prev_fallas) >= 2:
            group.loc[idx, 'Días_Entre_Falla_2'] = (row['Fecha del incidente'] - prev_fallas['Fecha del incidente'].iloc[-2]).days
        if len(prev_fallas) >= 3:
            group.loc[idx, 'Días_Entre_Falla_3'] = (row['Fecha del incidente'] - prev_fallas['Fecha del incidente'].iloc[-3]).days
    return group

# --- FUNCIÓN PRINCIPAL PARA ENTRENAR MODELO ---
def entrenar_modelo():
    global df, reg, label_encoders

    try:
        df = pd.read_excel("datos1.xlsx")
    except FileNotFoundError:
        print("❌ Archivo datos1.xlsx no encontrado.")
        return

    if not np.issubdtype(df['Fecha del incidente'].dtype, np.datetime64):
        df['Fecha del incidente'] = pd.to_datetime(df['Fecha del incidente'])

    columnas_agrupacion = ['Nombre del equipo', 'Marca del equipo', 'Serie del equipo'] if 'Serie del equipo' in df.columns else ['Nombre del equipo', 'Marca del equipo']
    df.sort_values(by=columnas_agrupacion + ['Fecha del incidente'], inplace=True)
    df = df.groupby(columnas_agrupacion).apply(get_previous_features, include_groups=False).reset_index()

    features = ['Nombre del equipo', 'Marca del equipo', 'Garantía de servicio en esa fecha',
                'Días_Entre_Falla_1', 'Días_Entre_Falla_2', 'Días_Entre_Falla_3',
                'Num_Mantenimientos_Previos', 'Días_Desde_Ultimo_Mantenimiento']
    df_fallas = df[df['Tipo'] == 'Falla'].copy()

    X = df_fallas[features].copy()
    y = df_fallas['Días_Entre_Falla_1'].fillna(0)

    label_encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    X.fillna(0, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
    model.fit(X_train, y_train)
    reg = model.best_estimator_

    y_pred = reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Entrenamiento actualizado:")
    print(f"➡️ RMSE: {rmse:.2f} días")
    print(f"➡️ MAE: {mae:.2f} días")
    print(f"➡️ R²: {r2:.2f}")

# --- FUNCIÓN PARA HACER PREDICCIÓN ---
def predecir_falla_auto(equipo, marca, garantia, horizonte=30):
    equipo_hist = df[(df['Nombre del equipo'] == equipo) & (df['Marca del equipo'] == marca)]

    if equipo_hist.empty:
        print("❌ No hay historial suficiente para ese equipo/marca.")
        return None, None

    dias_f1 = equipo_hist['Días_Entre_Falla_1'].dropna().iloc[-1] if not equipo_hist['Días_Entre_Falla_1'].dropna().empty else 0
    dias_f2 = equipo_hist['Días_Entre_Falla_2'].dropna().iloc[-1] if not equipo_hist['Días_Entre_Falla_2'].dropna().empty else 0
    dias_f3 = equipo_hist['Días_Entre_Falla_3'].dropna().iloc[-1] if not equipo_hist['Días_Entre_Falla_3'].dropna().empty else 0
    dias_mant = equipo_hist['Días_Desde_Ultimo_Mantenimiento'].dropna().iloc[-1] if not equipo_hist['Días_Desde_Ultimo_Mantenimiento'].dropna().empty else 0
    n_mants = equipo_hist['Num_Mantenimientos_Previos'].dropna().iloc[-1] if not equipo_hist['Num_Mantenimientos_Previos'].dropna().empty else 0

    input_data = pd.DataFrame([{
        'Nombre del equipo': equipo,
        'Marca del equipo': marca,
        'Garantía de servicio en esa fecha': garantia,
        'Días_Entre_Falla_1': dias_f1,
        'Días_Entre_Falla_2': dias_f2,
        'Días_Entre_Falla_3': dias_f3,
        'Num_Mantenimientos_Previos': n_mants,
        'Días_Desde_Ultimo_Mantenimiento': dias_mant
    }])

    for col in input_data.select_dtypes(include='object').columns:
        if col in label_encoders:
            input_data[col] = input_data[col].apply(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)
        else:
            input_data[col] = 0

    dias_estimados = reg.predict(input_data)[0]
    prob = 100 * np.exp(-dias_estimados / horizonte)
    prob = min(prob, 100)

    return dias_estimados, prob

# --- EJECUTAR ENTRENAMIENTO INICIAL AUTOMÁTICAMENTE ---
entrenar_modelo()

# --- EJEMPLO DE USO LOCAL ---
if __name__ == "__main__":
    print("\n--- EJEMPLO ---")
    eq = "Aspirador de secreciones"
    mar = "THOMAS"
    gar = "NO"
    dias, prob = predecir_falla_auto(eq, mar, gar, horizonte=30)
    if dias is not None:
        print(f"➡️ Estimación: {dias:.1f} días hasta la próxima falla")
        print(f"➡️ Probabilidad de falla en 30 días: {prob:.2f}%")
