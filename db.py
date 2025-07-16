import sqlite3
import pandas as pd

def get_connection():
    return sqlite3.connect("eventos.db")

# Crear la base de datos y la tabla si no existen
def crear_base():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS eventos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre_equipo TEXT,
            marca TEXT,
            numero_serie TEXT,
            tipo_evento TEXT,
            fecha_incidente DATE,
            garantia TEXT,
            diagnostico TEXT
        )
    """)
    conn.commit()
    conn.close()

# Insertar nuevo evento
def guardar_evento(nombre, marca, numero_serie, tipo, fecha, garantia, diagnostico):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO eventos (
            nombre_equipo, marca, numero_serie,
            tipo_evento, fecha_incidente,
            garantia, diagnostico
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (nombre, marca, numero_serie, tipo, fecha, garantia, diagnostico))
    conn.commit()
    conn.close()

# Cargar todos los eventos como DataFrame con columnas renombradas
def cargar_eventos():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM eventos", conn)
    conn.close()

    df.rename(columns={
        "nombre_equipo": "Nombre del equipo",
        "marca": "Marca del equipo",
        "numero_serie": "Número de serie",
        "tipo_evento": "Tipo",
        "fecha_incidente": "Fecha del incidente",
        "garantia": "Garantía",
        "diagnostico": "Diagnóstico técnico"
    }, inplace=True)

    return df
