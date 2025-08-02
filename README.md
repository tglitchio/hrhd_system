# Predictor de Fallas de Equipos Médicos

Modelo de Machine Learning para predecir fallas en equipos médicos usando CatBoost.

## Características
- **MAE:** 6.8 días (mejora del 70% vs baseline)
- **Predicción desde última falla real**
- **Validación temporal**
- **Features avanzadas** (15 características)

## Uso Rápido
```python
# Predecir desde última falla
datos, fecha = predecir_desde_ultima_fecha_real(
    "Ventilador mecánico", 
    "NIHON KOHDEN", 
    "Ninguno"
)
