import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf





# Definir función para predecir una imagen
def predecir_imagen(ruta_imagen, modelo):
    """
    Carga una imagen, la procesa y la predice usando un modelo entrenado.
    Luego, muestra la imagen con la predicción.

    Parámetros:
    - ruta_imagen: str, ruta de la imagen a predecir.
    - modelo: modelo de deep learning ya entrenado.
    """
    # Cargar la imagen
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen en {ruta_imagen}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (198, 198))  # Redimensionar al tamaño esperado por el modelo
    img_array = img / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensiones para la predicción

    # Hacer la predicción
    prediccion = modelo.predict(img_array, verbose = 0)
    clase_predicha = np.argmax(prediccion)  # Obtener la clase con mayor probabilidad

    # Ruta del archivo labels.txt
    ruta_labels = "/content/drive/MyDrive/tokio/deep_learning/proyecto_final_foodAi/food-101/meta/labels.txt"

    # Verificar si el diccionario de clases ya existe, si no, crearlo
    ruta_clases = "/content/drive/MyDrive/tokio/deep_learning/proyecto_final_foodAi/clases_food101.npy"
    if not os.path.exists(ruta_clases):
        with open(ruta_labels, "r") as f:
            clases_lista = [line.strip() for line in f.readlines()]  # Leer y limpiar nombres de clase

        diccionario_clases = {i: clase for i, clase in enumerate(clases_lista)}
        np.save(ruta_clases, diccionario_clases)  # Guardar el diccionario
        print(f"Diccionario de clases creado y guardado en {ruta_clases}")
    else:
        diccionario_clases = np.load(ruta_clases, allow_pickle=True).item()  # Cargar el diccionario

    # Obtener el nombre de la clase predicha
    nombre_clase = diccionario_clases.get(clase_predicha, f"Clase {clase_predicha}")

    # Mostrar la imagen con la predicción
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicción: {nombre_clase}")
    plt.show()

    return nombre_clase


def predecir_plato_y_nutricion(ruta_imagen, modelo, clases, df_nutricion):
    # Cargar la imagen
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen en {ruta_imagen}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (198, 198))  # Ajustar al tamaño del modelo
    img_array = img / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)

    # Hacer la predicción
    prediccion = modelo.predict(img_array, verbose = 0)
    clase_predicha = np.argmax(prediccion)
    nombre_clase = clases.get(clase_predicha, f"Clase {clase_predicha}")

    # Mostrar imagen con el nombre del plato predicho
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicción: {nombre_clase}")
    plt.show()

    # Buscar en el dataset de nutrición
    info_nutricion = df_nutricion[df_nutricion["Plato"].str.lower() == nombre_clase.lower()]

    if not info_nutricion.empty:
        print(f"📌 Información nutricional de {nombre_clase}:")
        print(info_nutricion.to_string(index=False))
    else:
        print(f"⚠ No se encontró información nutricional exacta para '{nombre_clase}'.")

        # Intentar encontrar el más parecido (búsqueda flexible)
        coincidencias = df_nutricion[df_nutricion["Plato"].str.contains(nombre_clase.split()[0], case=False, na=False)]

        if not coincidencias.empty:
            print("🔎 Plato más similar encontrado en la base de datos:")
            print(coincidencias.head(1).to_string(index=False))
        else:
            print("❌ No se encontraron platos similares en la base de datos.")



def obtener_info_completa(ruta_imagen, modelo, df_nutricion):
    """
    Función que recibe una imagen, predice el plato, obtiene su información nutricional y su receta.

    Parámetros:
        - ruta_imagen: str, ruta de la imagen en Drive.
        - modelo: modelo de deep learning para la predicción.
        - df_nutricion: DataFrame con la información nutricional y recetas.
    
    Retorna:
        - dict: Datos de la predicción, calorías y receta.
    """
    # Predicción del plato
    plato_predicho = predecir_imagen(ruta_imagen, modelo)
    
    # Buscar la información nutricional y la receta
    info_plato = df_nutricion[df_nutricion["Plato"].str.lower() == plato_predicho.lower()]
    
    if info_plato.empty:
        resultado = {
            "plato": plato_predicho.capitalize(),
            "calorias": "No disponible",
            "grasas": "No disponible",
            "carbohidratos": "No disponible",
            "proteinas": "No disponible",
            "url_receta": "No disponible",
            "ingredientes": "No disponible"
        }
    else:
        # Extraer información
        info_plato = info_plato.iloc[0]
        resultado = {
            "plato": plato_predicho.capitalize(),
            "calorias": info_plato['Calorías (kcal)'],
            "grasas": info_plato['Grasas (g)'],
            "carbohidratos": info_plato['Carbohidratos (g)'],
            "proteinas": info_plato['Proteínas (g)'],
            "ingredientes": info_plato.get('Ingredientes', "No disponible").strip(),
            "url_receta": info_plato.get('URL_Receta', "No disponible").strip()
        }

    # Imprimir la información de manera bonita
    print("\n" + "="*40)
    print(f"🍽️ Plato: {resultado['plato']}")
    print("")
    print(f"🔥 Calorías: {resultado['calorias']} kcal")
    print(f"🛢️ Grasas: {resultado['grasas']} g")
    print(f"🍞 Carbohidratos: {resultado['carbohidratos']} g")
    print(f"💪 Proteínas: {resultado['proteinas']} g")
    print("")
    print(f"📜 Ingredientes: {resultado['ingredientes']}") 
    print("")
    print(f"📌 URL Receta: {resultado['url_receta']}")
     

    return resultado
