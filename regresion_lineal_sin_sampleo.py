import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----------------------------------------------------
# 1. CARGA Y PREPARACIÓN GENERAL DE DATOS
# ----------------------------------------------------
file_path = "deptos_caba_2020.csv"
df = pd.read_csv(file_path)

target_col = 'precio_usd'

# ----------------------------------------------------
# PARTE A: MODELO DE 1 SOLA VARIABLE (`m2`)
# ----------------------------------------------------
print("=" * 60)
print(" 1. REGRESIÓN LINEAL UNIVARIABLE (Solo 'm2') ")
print("=" * 60)

# Filtrar nulos para el análisis de 1 variable
data_uni = df[['m2', target_col]].dropna()

X_uni = data_uni[['m2']]  # Debe ser DataFrame (matriz 2D)
y_uni = data_uni[target_col]

# Entrenamiento
model_uni = LinearRegression(fit_intercept=True)
model_uni.fit(X_uni, y_uni)

# Coeficientes
coef_uni = pd.DataFrame({
    'Variable': ['Intercepto (Constante)', 'm2'],
    'Coeficiente': [model_uni.intercept_, model_uni.coef_[0]]
})
print("\n--- Coeficientes (Modelo 1 Variable) ---")
print(coef_uni.to_string(index=False))

# Evaluación
y_pred_uni = model_uni.predict(X_uni)

mae_uni = mean_absolute_error(y_uni, y_pred_uni)
mse_uni = mean_squared_error(y_uni, y_pred_uni)
rmse_uni = np.sqrt(mse_uni)
r2_uni = r2_score(y_uni, y_pred_uni)

print("\n--- Métricas de Evaluación (Solo 'm2') ---")
print(f"MAE : {mae_uni:.4f}")
print(f"MSE : {mse_uni:.4f}")
print(f"RMSE: {rmse_uni:.4f}")
print(f"R²  : {r2_uni:.4f}")


# ----------------------------------------------------
# PARTE B: MODELO MULTIVARIABLE (`m2`, `ambientes`, `barrio`)
# ----------------------------------------------------
print("\n" + "=" * 60)
print(" 2. REGRESIÓN LINEAL MULTIVARIABLE ('m2', 'ambientes', 'barrio') ")
print("=" * 60)

feature_cols = ['m2', 'ambientes', 'barrio']
data_multi = df[feature_cols + [target_col]].dropna()

# One-hot encoding para la variable discreta
X_multi = pd.get_dummies(data_multi[feature_cols], columns=['barrio'], drop_first=False)
y_multi = data_multi[target_col]

# Entrenamiento
model_multi = LinearRegression(fit_intercept=True)
model_multi.fit(X_multi, y_multi)

# Coeficientes
intercept_multi = pd.DataFrame({'Variable': ['Intercepto (Constante)'], 'Coeficiente': [model_multi.intercept_]})
coef_multi = pd.DataFrame({'Variable': X_multi.columns, 'Coeficiente': model_multi.coef_})
tabla_coef_multi = pd.concat([intercept_multi, coef_multi], ignore_index=True)

print("\n--- Coeficientes (Modelo Multivariable) ---")
print(tabla_coef_multi.to_string(index=False))

# Evaluación
y_pred_multi = model_multi.predict(X_multi)

mae_multi = mean_absolute_error(y_multi, y_pred_multi)
mse_multi = mean_squared_error(y_multi, y_pred_multi)
rmse_multi = np.sqrt(mse_multi)
r2_multi = r2_score(y_multi, y_pred_multi)

print("\n--- Métricas de Evaluación (Multivariable) ---")
print(f"MAE : {mae_multi:.4f}")
print(f"MSE : {mse_multi:.4f}")
print(f"RMSE: {rmse_multi:.4f}")
print(f"R²  : {r2_multi:.4f}")


# ----------------------------------------------------
# 3. VISUALIZACIONES COMPARATIVAS
# ----------------------------------------------------
sns.set_theme(style="whitegrid")

# Gráfico 1: Recta de Ajuste de la Regresión Univariable
plt.figure(figsize=(8, 5))
sns.scatterplot(x=data_uni['m2'], y=y_uni, alpha=0.5, label='Datos Reales')
plt.plot(data_uni['m2'], y_pred_uni, color='red', linewidth=2, label='Línea de Regresión')
plt.title("Modelo Univariable: m² vs Precio USD")
plt.xlabel("m²")
plt.ylabel("Precio USD")
plt.legend()
plt.tight_layout()
plt.show()

# Gráfico 2: Comparación de Valores Reales vs Predichos de ambos modelos
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot Modelo 1 Variable
sns.scatterplot(ax=axes[0], x=y_uni, y=y_pred_uni, alpha=0.5, color='orange')
axes[0].plot([y_uni.min(), y_uni.max()], [y_uni.min(), y_uni.max()], 'r--')
axes[0].set_title(f"Univariable (Solo m²)\nR² = {r2_uni:.4f}")
axes[0].set_xlabel("Precio Real (USD)")
axes[0].set_ylabel("Precio Predicho (USD)")

# Plot Modelo Multivariable
sns.scatterplot(ax=axes[1], x=y_multi, y=y_pred_multi, alpha=0.5, color='green')
axes[1].plot([y_multi.min(), y_multi.max()], [y_multi.min(), y_multi.max()], 'r--')
axes[1].set_title(f"Multivariable (m², ambientes, barrio)\nR² = {r2_multi:.4f}")
axes[1].set_xlabel("Precio Real (USD)")

plt.tight_layout()
plt.show()