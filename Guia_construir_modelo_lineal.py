"""
Guía de regresión lineal con Python.

Materia: Introducción al Aprendizaje Automático.
Dataset: departamentos en venta en CABA durante 2020.

Este archivo está preparado para ejecutarse por celdas en Spyder, VS Code
o un entorno compatible con la marca ``# %%``.
"""

# =============================================================================
# BLOQUE 0. PREPARACIÓN DE LA CLASE
# =============================================================================
# %% 0. Importar las bibliotecas
"""
Objetivo
--------
Presentar las bibliotecas:

- NumPy aporta operaciones numéricas con vectores.
- pandas permite cargar, inspeccionar y transformar tablas.
- Matplotlib permite construir gráficos.
- pathlib permite expresar la ubicación del archivo como una ruta.

Ubicar este script y ``deptos_caba_2020.csv`` en la
misma carpeta. Si el archivo tiene otro nombre, modificar RUTA_DATOS.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




# - El 'if' evalúa si existe la variable global '__file__'. Esta variable solo 
# está disponible cuando ejecutas el código como un script (.py).
#* Path(__file__).resolve().parent: Encuentra la ruta absoluta y se queda 
# con la carpeta contenedora exacta donde está guardado el script.
# - El 'else' se activa en entornos como Jupyter Notebook o Google Colab (donde 
# '__file__' no existe).
#* Path.cwd(): Obtiene el directorio de trabajo actual desde donde se 
# lanzó el entorno.

if "__file__" in globals():
    CARPETA_CLASE = Path(__file__).resolve().parent
else:
    CARPETA_CLASE = Path.cwd()

# Esta variable señala el archivo que pandas debe leer.
RUTA_DATOS = CARPETA_CLASE / "/workspaces/IAA/departamentos-en-venta-2020.csv"


# =============================================================================
# BLOQUE 1. CARGA Y RECONOCIMIENTO DE LOS DATOS
# =============================================================================
# %% 1. Cargar el archivo CSV
"""
Pregunta inicial
----------------
¿Qué representa una fila y qué representa una columna?
"""

# ``read_csv`` lee el archivo y devuelve un DataFrame, es decir, una tabla.
datos_crudos = pd.read_csv(RUTA_DATOS)

# ``head`` muestra las primeras cinco filas sin recorrer toda la tabla.
print("Primeras filas:")
print(datos_crudos.head())

# ``shape`` es una tupla: cantidad de filas y cantidad de columnas.
print("\nDimensiones:", datos_crudos.shape)

# ``columns`` permite conocer los nombres de todas las variables.
print("\nColumnas:", datos_crudos.columns.tolist())


# %% 2. Inspeccionar estructura, tipos y resumen
"""
El resultado esperado es una tabla de 600 filas y 5 columnas:

- m2: superficie del departamento.
- ambientes: cantidad de ambientes.
- barrio: ubicación dentro de CABA.
- usd_m2: precio informado por metro cuadrado.
- precio_usd: precio total, que será nuestra variable objetivo.
"""

# ``info`` imprime cantidad de filas, valores no nulos y tipos de datos.
datos_crudos.info()

# ``describe`` calcula medidas como media, desvío y cuartiles.
resumen_numerico = datos_crudos.describe()
print("\nResumen numérico:")
print(resumen_numerico)

# ``value_counts`` cuenta cuántas filas corresponden a cada barrio.
cantidad_por_barrio = datos_crudos["barrio"].value_counts()
print("\nCantidad de departamentos por barrio:")
print(cantidad_por_barrio)


# =============================================================================
# BLOQUE 2. CONTROL Y LIMPIEZA
# =============================================================================
# %% 3. Buscar problemas antes de modificar los datos
"""
Principio de trabajo
--------------------
Limpiar no significa borrar filas automáticamente. Primero se detectan
posibles problemas; luego se decide qué hacer y se deja registro.

Resultados esperados
--------------------
- No hay valores faltantes.
- Hay una fila duplicada.
- No hay superficies, ambientes ni precios menores o iguales que cero.

Un valor extremo no es necesariamente un error. En esta clase se conserva
para observar cómo el ECM da más peso a errores grandes.
"""

# ``isna`` marca faltantes y ``sum`` los cuenta columna por columna.
faltantes_por_columna = datos_crudos.isna().sum()
print("Faltantes por columna:")
print(faltantes_por_columna)

# ``duplicated`` marca filas idénticas a una fila anterior.
filas_duplicadas = datos_crudos.duplicated()
cantidad_duplicadas = int(filas_duplicadas.sum())
print("\nCantidad de filas duplicadas:", cantidad_duplicadas)

# ``keep=False`` permite ver las dos apariciones de cada duplicado.
duplicados_completos = datos_crudos[
    datos_crudos.duplicated(keep=False)
]
print("\nFilas involucradas en duplicados:")
print(duplicados_completos)

# Estas comparaciones controlan valores imposibles para este problema.
superficie_invalida = datos_crudos["m2"] <= 0
ambientes_invalidos = datos_crudos["ambientes"] <= 0
precio_invalido = datos_crudos["precio_usd"] <= 0

print("\nSuperficies inválidas:", int(superficie_invalida.sum()))
print("Ambientes inválidos:", int(ambientes_invalidos.sum()))
print("Precios inválidos:", int(precio_invalido.sum()))


# %% 4. Aplicar las decisiones de limpieza
"""
Decisiones
----------
1. Crear una copia para preservar los datos crudos.
2. Uniformar los textos de ``barrio``.
3. Eliminar la segunda aparición de una fila idéntica.
4. Conservar los valores extremos porque pueden ser casos reales.

Después de quitar el duplicado deben quedar 599 observaciones.
"""

# ``copy`` evita modificar accidentalmente la tabla original.
datos = datos_crudos.copy()

# ``astype`` usa el tipo de texto de pandas.
datos["barrio"] = datos["barrio"].astype("string")

# ``str.strip`` quita espacios externos y ``str.upper`` unifica mayúsculas.
datos["barrio"] = datos["barrio"].str.strip().str.upper()

# ``drop_duplicates`` conserva una sola copia de cada fila idéntica.
datos = datos.drop_duplicates().copy()

# ``reset_index`` crea índices consecutivos luego de eliminar una fila.
datos = datos.reset_index(drop=True)

print("Filas antes de limpiar:", len(datos_crudos))
print("Filas después de limpiar:", len(datos))


# =============================================================================
# BLOQUE 3. DEFINIR EL PROBLEMA Y VISUALIZAR
# =============================================================================
# %% 5. Separar la variable explicativa y el objetivo
"""
Problema de aprendizaje
-----------------------
Queremos estimar el precio total de un departamento usando su superficie.

X = m2             variable conocida al realizar la predicción.
Y = precio_usd     variable que queremos predecir.

Usamos una sola variable para que el modelo pueda verse como una recta.
No usamos ``usd_m2`` como entrada porque contiene información derivada del
precio. Utilizarla haría circular parte de Y dentro de X.
"""

# ``x`` contiene la superficie de cada departamento.
x = datos["m2"]

# ``y`` contiene el precio real correspondiente a cada departamento.
y = datos["precio_usd"]


# %% 6. Construir el diagrama de dispersión
"""
Preguntas para hacer antes de calcular
--------------------------------------
1. ¿La relación parece creciente o decreciente?
2. ¿Todos los puntos forman una recta perfecta?
3. ¿Hay puntos alejados de la nube principal?
4. ¿Alcanza con la superficie para conocer exactamente el precio?
"""

# ``subplots`` crea una figura y un eje sobre el cual dibujar.
figura_datos, eje_datos = plt.subplots(figsize=(8, 5))

# ``scatter`` dibuja un punto por cada par formado por x e y.
eje_datos.scatter(x, y, alpha=0.55, color="steelblue")

# ``set`` reúne el título y los nombres de ambos ejes.
eje_datos.set(
    title="Precio de departamentos según superficie",
    xlabel="Superficie (m²)",
    ylabel="Precio (USD)",
)

# ``tight_layout`` ajusta los márgenes para evitar texto cortado.
figura_datos.tight_layout()
plt.show()


# =============================================================================
# BLOQUE 4. REPASO ESTADÍSTICO SOBRE DATOS REALES
# =============================================================================
# %% 7. Definir funciones estadísticas sencillas
"""
Convención usada
----------------
Tratamos las 599 filas como una distribución empírica. Por eso dividimos
por N en la varianza y la covarianza. Lo importante es usar la misma
convención en ambas medidas.

Las funciones hacen explícitas las fórmulas. Después se puede mostrar que
pandas ofrece operaciones equivalentes ya implementadas.
"""


def esperanza(valores):
    """Devuelve el promedio de una colección de valores."""
    arreglo = np.asarray(valores, dtype=float)
    return float(np.mean(arreglo))


def varianza(valores):
    """Devuelve la distancia cuadrática media respecto del promedio."""
    arreglo = np.asarray(valores, dtype=float)
    centro = esperanza(arreglo)
    return float(np.mean((arreglo - centro) ** 2))


def covarianza(valores_x, valores_y):
    """Devuelve cuánto varían juntas dos colecciones de igual largo."""
    arreglo_x = np.asarray(valores_x, dtype=float)
    arreglo_y = np.asarray(valores_y, dtype=float)
    desvio_x = arreglo_x - esperanza(arreglo_x)
    desvio_y = arreglo_y - esperanza(arreglo_y)
    return float(np.mean(desvio_x * desvio_y))


def correlacion(valores_x, valores_y):
    """Devuelve la covarianza normalizada entre -1 y 1."""
    cov_xy = covarianza(valores_x, valores_y)
    escala = np.sqrt(varianza(valores_x) * varianza(valores_y))
    return float(cov_xy / escala)


# %% 8. Calcular e interpretar las medidas
"""
Lectura esperada
----------------
- E[X] ubica la superficie media.
- E[Y] ubica el precio medio.
- Var(X) y Var(Y) miden dispersión, pero conservan unidades al cuadrado.
- Cov(X,Y) positiva indica que suelen crecer juntas.
- Corr(X,Y) ronda 0.80: existe una asociación lineal positiva fuerte.

La correlación no dice que todos los puntos estén sobre la misma recta y
no demuestra que aumentar la superficie cause por sí solo todo el precio.
"""

# Cada variable guarda un resultado para poder reutilizarlo más adelante.
media_x = esperanza(x)
media_y = esperanza(y)
varianza_x = varianza(x)
varianza_y = varianza(y)
covarianza_xy = covarianza(x, y)
correlacion_xy = correlacion(x, y)

print("Esperanza de X:", media_x)
print("Esperanza de Y:", media_y)
print("Varianza de X:", varianza_x)
print("Varianza de Y:", varianza_y)
print("Covarianza entre X e Y:", covarianza_xy)
print("Correlación entre X e Y:", correlacion_xy)


# %% 9. Recuperar propiedades de la clase anterior
"""
Propiedades verificadas con los datos
-------------------------------------
E[aX + c] = aE[X] + c
Var(aX) = a²Var(X)
Var(X + c) = Var(X)
Cov(X,Y) = E[XY] - E[X]E[Y]
Corr(X,Y) = Cov(X,Y) / sqrt(Var(X)Var(Y))

``np.isclose`` compara números decimales admitiendo pequeños errores de
representación de la computadora.
"""

# Estas constantes permiten construir transformaciones simples de X.
a = 2.0
c = 10.0

# ``to_numpy`` convierte las columnas de pandas en arreglos de NumPy.
x_vector = x.to_numpy()
y_vector = y.to_numpy()

# Cada variable booleana indica si una igualdad se cumple numéricamente.
esperanza_es_lineal = np.isclose(
    esperanza(a * x_vector + c),
    a * esperanza(x_vector) + c,
)
varianza_escala = np.isclose(
    varianza(a * x_vector),
    a**2 * varianza(x_vector),
)
varianza_no_cambia = np.isclose(
    varianza(x_vector + c),
    varianza(x_vector),
)
covarianza_alternativa = np.isclose(
    covarianza(x_vector, y_vector),
    esperanza(x_vector * y_vector)
    - esperanza(x_vector) * esperanza(y_vector),
)
correlacion_normalizada = np.isclose(
    correlacion(x_vector, y_vector),
    covarianza(x_vector, y_vector)
    / np.sqrt(varianza(x_vector) * varianza(y_vector)),
)

print("E[aX + c] = aE[X] + c:", esperanza_es_lineal)
print("Var(aX) = a²Var(X):", varianza_escala)
print("Var(X + c) = Var(X):", varianza_no_cambia)
print("Fórmula alternativa de Cov(X,Y):", covarianza_alternativa)
print("Correlación normalizada:", correlacion_normalizada)


# =============================================================================
# BLOQUE 5. EL MODELO CONSTANTE Y EL ECM
# =============================================================================
# %% 10. Definir el Error Cuadrático Medio
"""
El ECM responde una pregunta concreta:

¿Qué tan alejadas están, en conjunto, las predicciones de los valores reales?

ECM = promedio de (valor real - predicción)²

Elevar al cuadrado evita que errores positivos y negativos se cancelen y
penaliza con más fuerza los errores grandes.
"""


def ecm(valores_reales, valores_predichos):
    """Calcula el promedio de los residuos elevados al cuadrado."""
    reales = np.asarray(valores_reales, dtype=float)
    predichos = np.asarray(valores_predichos, dtype=float)
    residuos = reales - predichos
    return float(np.mean(residuos**2))


# %% 11. Construir el mejor modelo constante
"""
Un modelo constante ignora X y siempre devuelve el mismo número:

    Y_hat = c

Si se usa ECM, el mejor valor de c es E[Y]. Puede demostrarse derivando la
función de error respecto de c. Acá lo comprobamos con los datos y un gráfico.
"""

# La constante óptima es el promedio de los precios observados.
constante_optima = media_y

# ``full`` repite la constante una vez por cada observación de Y.
prediccion_constante = np.full(len(y), constante_optima)

# Este número resume el error del modelo constante sobre todos los datos.
ecm_constante = ecm(y, prediccion_constante)

print("Constante óptima:", constante_optima)
print("ECM del modelo constante:", ecm_constante)


# %% 12. Visualizar por qué el promedio minimiza el ECM
"""
La curva muestra el ECM obtenido al probar distintas constantes. El punto
mínimo debe coincidir con el promedio de Y.
"""

# ``linspace`` crea 200 constantes alrededor del precio medio.
constantes_candidatas = np.linspace(
    media_y - 150000,
    media_y + 150000,
    200,
)

# La comprensión calcula el ECM para cada constante candidata.
errores_constantes = [
    ecm(y, np.full(len(y), candidata))
    for candidata in constantes_candidatas
]

figura_constante, ejes_constante = plt.subplots(
    1,
    2,
    figsize=(12, 4),
)

# El primer gráfico muestra la predicción constante sobre los datos.
ejes_constante[0].scatter(x, y, alpha=0.35, color="steelblue")
ejes_constante[0].axhline(
    constante_optima,
    color="crimson",
    label="Predicción promedio",
)
ejes_constante[0].set(
    title="Modelo constante",
    xlabel="Superficie (m²)",
    ylabel="Precio (USD)",
)
ejes_constante[0].legend()

# El segundo gráfico presenta el ECM como función de la constante.
ejes_constante[1].plot(
    constantes_candidatas,
    errores_constantes,
    color="darkorange",
)
ejes_constante[1].axvline(
    constante_optima,
    color="crimson",
    linestyle="--",
    label="E[Y]",
)
ejes_constante[1].set(
    title="ECM según la constante elegida",
    xlabel="Constante c (USD)",
    ylabel="ECM",
)
ejes_constante[1].legend()
figura_constante.tight_layout()
plt.show()


# =============================================================================
# BLOQUE 6. EL MODELO LINEAL QUE MINIMIZA EL ECM
# =============================================================================
# %% 13. Calcular la pendiente y el intercepto
"""
Ahora el modelo puede utilizar X:

    Y_hat = wX + b

Los valores que minimizan el ECM son:

    w = Cov(X,Y) / Var(X)
    b = E[Y] - wE[X]

La pendiente w se expresa en USD por m². El intercepto b es la predicción
teórica para X = 0; no siempre tiene una interpretación realista.
"""

# La pendiente relaciona la variación conjunta con la variación de X.
pendiente = covarianza_xy / varianza_x

# El intercepto fuerza a que la recta pase por el punto de medias.
intercepto = media_y - pendiente * media_x

# La fórmula se aplica a cada superficie para generar una predicción.
prediccion_lineal = pendiente * x + intercepto

# Se evalúan las predicciones con la misma métrica usada antes.
ecm_lineal = ecm(y, prediccion_lineal)

print("Pendiente:", pendiente)
print("Intercepto:", intercepto)
print("ECM del modelo lineal:", ecm_lineal)


# %% 14. Dibujar la recta sobre el diagrama de dispersión
"""
Lectura esperada
----------------
La recta sigue la tendencia general, pero no pasa por todos los puntos.
El modelo lineal no memoriza cada precio: resume la relación entre X e Y.
"""

# Ordenar X evita que la línea se dibuje saltando entre valores.
orden_x = np.argsort(x.to_numpy())
x_ordenado = x.to_numpy()[orden_x]
y_lineal_ordenado = np.asarray(prediccion_lineal)[orden_x]

figura_lineal, eje_lineal = plt.subplots(figsize=(8, 5))
eje_lineal.scatter(x, y, alpha=0.40, color="steelblue", label="Datos")
eje_lineal.plot(
    x_ordenado,
    y_lineal_ordenado,
    color="crimson",
    linewidth=2,
    label="Modelo lineal",
)
eje_lineal.scatter(
    media_x,
    media_y,
    color="black",
    s=70,
    label="Punto de medias",
)
eje_lineal.set(
    title="Recta que minimiza el ECM",
    xlabel="Superficie (m²)",
    ylabel="Precio (USD)",
)
eje_lineal.legend()
figura_lineal.tight_layout()
plt.show()


# =============================================================================
# BLOQUE 7. RESIDUOS Y COMPARACIÓN DE MODELOS
# =============================================================================
# %% 15. Calcular y observar los residuos
"""
El residuo de una observación es:

    residuo = valor real - valor predicho

Residuo positivo: el modelo subestimó el precio.
Residuo negativo: el modelo sobreestimó el precio.
"""

# La resta se realiza fila por fila porque ambas series están alineadas.
residuos_lineales = y - prediccion_lineal

# Esta tabla reúne casos concretos para interpretar en clase.
tabla_residuos = pd.DataFrame(
    {
        "m2": x,
        "precio_real": y,
        "precio_predicho": prediccion_lineal,
        "residuo": residuos_lineales,
    }
)

print("Cinco residuos:")
print(tabla_residuos.head())

figura_residuos, eje_residuos = plt.subplots(figsize=(8, 4))
eje_residuos.scatter(
    prediccion_lineal,
    residuos_lineales,
    alpha=0.45,
    color="slateblue",
)
eje_residuos.axhline(0, color="black", linestyle="--")
eje_residuos.set(
    title="Residuos del modelo lineal",
    xlabel="Precio predicho (USD)",
    ylabel="Residuo (USD)",
)
figura_residuos.tight_layout()
plt.show()


# %% 16. Comparar el modelo constante y el modelo lineal
"""
Ambos modelos se evalúan sobre las mismas observaciones y con el mismo ECM.
Por eso la comparación es válida.

R² puede leerse como la proporción del error del modelo constante que el
modelo lineal logró reducir:

    R² = 1 - ECM_lineal / ECM_constante

Con estos datos se espera un valor cercano a 0.64.
"""

# La mejora relativa usa al modelo constante como referencia.
r_cuadrado = 1 - ecm_lineal / ecm_constante

# Esta tabla resume los resultados sin perder los valores originales.
comparacion = pd.DataFrame(
    {
        "modelo": ["Constante", "Lineal con m2"],
        "ecm": [ecm_constante, ecm_lineal],
    }
)

print("Comparación final:")
print(comparacion)
print("R² del modelo lineal:", r_cuadrado)


# =============================================================================
# BLOQUE 8. CIERRE Y PREGUNTAS DE DISCUSIÓN
# =============================================================================
# %% 17. Síntesis
"""
Preguntas de cierre
-------------------
1. ¿Qué información usa el modelo constante?
2. ¿Qué información adicional usa el modelo lineal?
3. ¿Por qué la recta pasa por (E[X], E[Y])?
4. ¿Qué significa la pendiente en este problema?
5. ¿Por qué el ECM lineal es menor que el ECM constante?
6. ¿Un ECM menor demuestra que el modelo funcionará igual con datos nuevos?
7. ¿Qué variables podrían explicar parte de los residuos?
8. ¿Por qué ``usd_m2`` sería una entrada problemática?

Respuestas orientativas
-----------------------
1. Solo conoce los precios observados y siempre predice su promedio.
2. Usa la superficie para cambiar la predicción entre departamentos.
3. Porque b = E[Y] - wE[X]. Al reemplazar X por E[X], queda E[Y].
4. Es el aumento estimado del precio por cada m² adicional.
5. La superficie aporta información asociada con el precio.
6. No. Todavía falta evaluar la generalización con datos no usados al ajustar.
7. Barrio, estado, antigüedad, expensas, luminosidad y otras características.
8. Se calcula usando el precio y produciría fuga de información del objetivo.

Resultado conceptual de la clase
--------------------------------
El recorrido completo fue:

datos -> control -> limpieza -> gráfico -> medidas -> modelo -> ECM
"""