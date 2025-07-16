from actualizador import actualizar_con_nuevos_eventos

# Ruta al archivo original de Excel
ruta_excel = "datos1.xlsx"

# Ejecutar la actualización
df_actualizado = actualizar_con_nuevos_eventos(ruta_excel)

# Guardar resultado en un nuevo Excel
df_actualizado.to_excel("datos_actualizados.xlsx", index=False)
print("✅ Archivo 'datos_actualizados.xlsx' generado exitosamente.")
