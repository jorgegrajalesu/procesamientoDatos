# ===========================================
# Procesamiento de datos de emociones por edad
# ===========================================

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# --- Generación de dataset sintético ---
n = 50  # cantidad de personas
nombres = ["Ana", "Luis", "Marta", "Carlos", "Lucía", "Pedro", "Jorge", "Laura", "Andrés", "Sofía"]
emociones = ["Feliz", "Triste", "Serio", "Amargo", "Ansioso", "Relajado"]

# Crear DataFrame
data = {
    "ID": range(1, n + 1),
    "Nombre": [random.choice(nombres) for _ in range(n)],
    "Edad": [random.randint(5, 90) for _ in range(n)],
    "Emoción": [random.choice(emociones) for _ in range(n)]
}

df = pd.DataFrame(data)

# --- Clasificación por grupo de edad ---
def clasificar_edad(edad):
    if edad <= 12:
        return "Niño"
    elif edad <= 19:
        return "Adolescente"
    elif edad <= 35:
        return "Joven"
    elif edad <= 59:
        return "Adulto"
    else:
        return "Adulto Mayor"

df["ClasificaciónEdad"] = df["Edad"].apply(clasificar_edad)

# --- Mostrar primeras filas ---
print("📊 Dataset generado:")
print(df.head())

# --- Procesamiento estadístico ---
resumen = df.groupby("ClasificaciónEdad")["Edad"].agg(["count", "mean", "min", "max"])
print("\n📈 Resumen por grupo de edad:")
print(resumen)

# --- Visualización de distribución ---
plt.figure(figsize=(8,5))
df["ClasificaciónEdad"].value_counts().plot(kind='bar', color='skyblue')
plt.title("Distribución de personas por grupo de edad")
plt.xlabel("Clasificación de Edad")
plt.ylabel("Cantidad de Personas")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# --- Análisis de emociones ---
plt.figure(figsize=(8,5))
df["Emoción"].value_counts().plot(kind='bar', color='orange')
plt.title("Frecuencia de emociones en el dataset")
plt.xlabel("Emoción")
plt.ylabel("Cantidad de Personas")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
