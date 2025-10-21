# 🏃 Predicción de Ritmo y Tiempo Estimado en Running con XGBoost

Este proyecto utiliza modelos de machine learning para **predecir el ritmo medio (pace)** y estimar el **tiempo total de carrera** a partir de características de la ruta como distancia y elevación. El objetivo es brindar una estimación realista del rendimiento para diferentes tipos de terrenos y esfuerzos.

## 📋 Contenido
- [1. Descripción](#1-descripción)
- [2. Estructura del proyecto](#2-estructura-del-proyecto)
- [3. Feature Engineering](#3-feature-engineering)
- [4. Modelo](#4-modelo)
- [5. Entrenamiento y evaluación](#5-entrenamiento-y-evaluación)
- [6. Predicción de tiempo estimado](#6-predicción-de-tiempo-estimado)
- [7. Requisitos](#7-requisitos)
- [8. Cómo usar](#8-cómo-usar)

## 1. Descripción
Este notebook entrena un modelo de regresión para predecir el **ritmo (min/km)** de una actividad de running o ciclismo en función de variables derivadas de la distancia y la dificultad del terreno. Con el ritmo predicho, se estima el **tiempo total** que tardaría en completarse una ruta.

## 2. Estructura del proyecto
```
.
├── data/                  # datasets de entrenamiento
├── models/                # modelos entrenados (XGBoost)
├── Notebook.ipynb         # notebook principal
├── README.md
└── requirements.txt
```

## 3. Feature Engineering
Se generan variables derivadas para capturar la dificultad de la ruta y el esfuerzo físico estimado:

| Variable              | Descripción                                               |
|------------------------|----------------------------------------------------------|
| `elev_per_km`          | Ganancia de elevación por kilómetro                      |
| `speed_kmh`            | Velocidad promedio estimada                              |
| `effort_index`         | Índice relativo de esfuerzo (ajusta velocidad por elev.) |
| `effort_total`         | Carga total de esfuerzo (distancia × elevación relativa) |
| `elev_per_km_sq`       | Elevación al cuadrado (captura efectos no lineales)      |

```python
df['elev_per_km'] = df['total_elevation_gain'] / (df['distance'] / 1000)
df['speed_kmh'] = (df['distance'] / 1000) / (df['moving_time_min'] / 60)
df['effort_index'] = df['speed_kmh'] / (1 + df['elev_per_km'])
df['effort_total'] = df['distance'] * df['elev_per_km']
df['elev_per_km_sq'] = df['elev_per_km'] ** 2
```

## 4. Modelo
Se utiliza XGBoost para predecir `pace_min_per_km` (ritmo medio):

```python
from xgboost import XGBRegressor
xgb = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    booster="gbtree"
)
```

La búsqueda de hiperparámetros se realiza con `GridSearchCV` y optimiza **MAE (Mean Absolute Error)**.

## 5. Entrenamiento y evaluación

```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=3,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)
best_xgb = XGBRegressor(**grid.best_params_, early_stopping_rounds=50)
best_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred_best = best_xgb.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_best)
r2 = r2_score(y_test, y_pred_best)
```

📊 Métricas:
- `MAE` → error absoluto medio en min/km  
- `R²` → poder explicativo del modelo

## 6. Predicción de tiempo estimado
Una vez predicho el ritmo:
\`\`\`
Tiempo (min) = Pace (min/km) × Distancia (km)
\`\`\`

Ejemplo:
```python
predicted_pace = best_xgb.predict(nueva_ruta_X)[0]  # min/km
distance_km = 5
predicted_time_min = predicted_pace * distance_km
```

Salida:
```
Pace predicho: 5.30 min/km
Tiempo estimado 5K: 26:30 min
```

## 7. Requisitos
- Python 3.9+
- pandas  
- scikit-learn  
- xgboost  
- joblib  
- matplotlib / seaborn (para visualización)

Instalación rápida:
```bash
pip install -r requirements.txt
```

## 8. Cómo usar

1. Prepara tu dataset con columnas mínimas:
   - `distance` (en m)
   - `moving_time_min` (en min)
   - `total_elevation_gain` (en m)

2. Genera las features con el bloque de ingeniería de variables.

3. Entrena el modelo XGBoost con `GridSearchCV`.

4. Guarda el modelo entrenado:
```python
import joblib, os
os.makedirs("models", exist_ok=True)
joblib.dump(best_xgb, "models/best_xgb_model.joblib")
```

5. Para predecir tiempo en una nueva ruta:
```python
model = joblib.load("models/best_xgb_model.joblib")
y_pred = model.predict(nueva_ruta_X)[0]
time_min = y_pred * 5  # si son 5 km
```

---

📌 **Autor:** Eder  
📆 **Proyecto:** Predicción de ritmo y tiempo estimado en running 🏃  
🧠 Modelo: XGBoost Regressor | 🎯 Target: pace_min_per_km
