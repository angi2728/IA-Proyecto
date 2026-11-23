import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Catálogo de ejemplo con características de cada producto.
productos = pd.DataFrame(
    [
        {"id": 1, "nombre": "Auriculares inalámbricos", "categoria": "tecnologia", "precio": 65, "calificacion": 4.7, "popularidad": 90},
        {"id": 2, "nombre": "Altavoz Bluetooth", "categoria": "tecnologia", "precio": 48, "calificacion": 4.5, "popularidad": 82},
        {"id": 3, "nombre": "Teclado mecánico", "categoria": "tecnologia", "precio": 80, "calificacion": 4.6, "popularidad": 75},
        {"id": 4, "nombre": "Ratón inalámbrico", "categoria": "tecnologia", "precio": 30, "calificacion": 4.3, "popularidad": 88},
        {"id": 5, "nombre": "Botella reutilizable", "categoria": "hogar", "precio": 18, "calificacion": 4.4, "popularidad": 70},
        {"id": 6, "nombre": "Lámpara de escritorio", "categoria": "hogar", "precio": 35, "calificacion": 4.2, "popularidad": 65},
        {"id": 7, "nombre": "Cafetera italiana", "categoria": "hogar", "precio": 28, "calificacion": 4.8, "popularidad": 84},
        {"id": 8, "nombre": "Mochila urbana", "categoria": "accesorios", "precio": 55, "calificacion": 4.1, "popularidad": 60},
        {"id": 9, "nombre": "Cargador portátil", "categoria": "tecnologia", "precio": 40, "calificacion": 4.5, "popularidad": 78},
        {"id": 10, "nombre": "Soporte para portátil", "categoria": "accesorios", "precio": 25, "calificacion": 4.4, "popularidad": 73},
        {"id": 11, "nombre": "Taza térmica", "categoria": "hogar", "precio": 22, "calificacion": 4.6, "popularidad": 68},
        {"id": 12, "nombre": "Funda para auriculares", "categoria": "accesorios", "precio": 15, "calificacion": 4.0, "popularidad": 55},
    ]
)

# Interacciones de ejemplo: 1 indica que al usuario le gustó el producto y 0 que no.
interacciones = pd.DataFrame(
    [
        {"producto_id": 1, "le_gusta": 1},
        {"producto_id": 2, "le_gusta": 1},
        {"producto_id": 3, "le_gusta": 1},
        {"producto_id": 4, "le_gusta": 1},
        {"producto_id": 5, "le_gusta": 0},
        {"producto_id": 6, "le_gusta": 0},
        {"producto_id": 7, "le_gusta": 0},
        {"producto_id": 8, "le_gusta": 0},
    ]
)

# Separamos las características numéricas y categóricas para prepararlas correctamente.
columnas_numericas = ["precio", "calificacion", "popularidad"]
columnas_categoricas = ["categoria"]
preprocesador = ColumnTransformer(
    transformers=[
        ("numericas", StandardScaler(), columnas_numericas),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
    ]
)

# El modelo aprende a distinguir productos que gustan de los que no gustan.
modelo = Pipeline(
    steps=[
        ("preprocesamiento", preprocesador),
        ("clasificador", KNeighborsClassifier(n_neighbors=3, weights="distance")),
    ]
)

# Unimos las interacciones con la información de sus productos para entrenar el modelo.
datos_entrenamiento = interacciones.merge(
    productos, left_on="producto_id", right_on="id"
)
caracteristicas_entrenamiento = datos_entrenamiento[
    columnas_numericas + columnas_categoricas
]
etiquetas = datos_entrenamiento["le_gusta"]
modelo.fit(caracteristicas_entrenamiento, etiquetas)


def recomendar_productos(catalogo, historial, clasificador, cantidad=5):
    """Devuelve productos no evaluados ordenados por probabilidad de gustar."""
    ids_evaluados = historial["producto_id"]
    candidatos = catalogo[~catalogo["id"].isin(ids_evaluados)].copy()

    if candidatos.empty:
        return candidatos.assign(probabilidad_gusto=pd.Series(dtype=float))

    caracteristicas = candidatos[columnas_numericas + columnas_categoricas]
    indice_gusto = list(
        clasificador.named_steps["clasificador"].classes_
    ).index(1)
    candidatos["probabilidad_gusto"] = clasificador.predict_proba(caracteristicas)[
        :, indice_gusto
    ]

    return candidatos.sort_values(
        "probabilidad_gusto", ascending=False
    ).head(cantidad)[["nombre", "categoria", "precio", "probabilidad_gusto"]]


# Mostramos las recomendaciones para el usuario de ejemplo.
if __name__ == "__main__":
    recomendaciones = recomendar_productos(productos, interacciones, modelo)
    print("Productos recomendados:")
    print(recomendaciones.to_string(index=False))