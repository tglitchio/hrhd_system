# --- Bloque para actualizar el modelo con nuevos eventos de eventos.db ---

import pandas as pd
import sqlite3
import numpy as np

# Reutiliza tu función get_previous_features
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

# Función para fusionar el Excel original con los nuevos eventos de la base de datos
def actualizar_con_nuevos_eventos(excel_path, db_path="eventos.db"):
    # Cargar Excel original
    df_excel = pd.read_excel(excel_path)
    df_excel['Fuente'] = 'Excel'

    # Cargar base de datos
    conn = sqlite3.connect(db_path)
    df_nuevos = pd.read_sql_query("SELECT * FROM eventos", conn)
    conn.close()
    df_nuevos.rename(columns={
        'nombre_equipo': 'Nombre del equipo',
        'marca': 'Marca del equipo',
        'numero_serie': 'Serie del equipo',
        'tipo_evento': 'Tipo',
        'fecha_incidente': 'Fecha del incidente',
        'garantia': 'Garantía de servicio en esa fecha',
        'diagnostico': 'Diagnóstico técnico'
    }, inplace=True)
    df_nuevos['Fuente'] = 'Base de datos'
    df_nuevos['Fecha del incidente'] = pd.to_datetime(df_nuevos['Fecha del incidente'])

    # Unir ambos
    df_combinado = pd.concat([df_excel, df_nuevos], ignore_index=True)
    df_combinado = df_combinado.sort_values(by=['Nombre del equipo', 'Marca del equipo', 'Serie del equipo', 'Fecha del incidente'])

    # Recalcular variables temporales
    df_final = df_combinado.groupby(['Nombre del equipo', 'Marca del equipo', 'Serie del equipo']).apply(get_previous_features, include_groups=False).reset_index()
    return df_final

# --- Uso:
# df_actualizado = actualizar_con_nuevos_eventos("datos1.xlsx")
# Ahora puedes reentrenar con df_actualizado como hiciste antes
