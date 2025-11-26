# =======================================================
# PROCESAMIENTO DE DATOS Y CLASIFICACIÓN AUTOMÁTICA
# =======================================================

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===========================================
#  GENERACIÓN DEL DATASET
# ===========================================

n = 100  # número de registros

nombres = ["Ana", "Luis", "Marta", "Carlos", "Lucía", "Pedro", "Jorge", "Laura", "Andrés", "Sofía"]
emociones = ["Feliz", "Triste", "Serio", "Amargo", "Ansioso", "Relajado"]

data = {
    "ID": range(1, n + 1),
    "Nombre": [random.choice(nombres) for _ in range(n)],
    "Edad": [random.randint(5, 90) for _ in range(n)],
    "Emoción": [random.choice(emociones) for _ in range(n)]
}

df = pd.DataFrame(data)

# ===========================================
#  CLASIFICACIÓN DE EDADES
# ===========================================

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

print("Dataset generado correctamente\n")
print(df.head())

# ===========================================
#  GUARDAR ARCHIVOS (CSV y Excel)
# ===========================================

df.to_csv("dataset_emociones.csv", index=False, encoding="utf-8")
df.to_excel("dataset_emociones.xlsx", index=False)

print("Archivos guardados:")
print("- dataset_emociones.csv")
print("- dataset_emociones.xlsx\n")

# ===========================================
# ANÁLISIS ESTADÍSTICO
# ===========================================

resumen = df.groupby("ClasificaciónEdad")["Edad"].agg(["count", "mean", "min", "max"])
print("Resumen por grupo de edad:")
print(resumen)
print("\n")

# ===========================================
# VISUALIZACIÓN DE DATOS
# ===========================================

plt.figure(figsize=(8,5))
df["ClasificaciónEdad"].value_counts().plot(kind='bar', color='skyblue')
plt.title("Distribución de personas por grupo de edad")
plt.xlabel("Clasificación de Edad")
plt.ylabel("Cantidad de Personas")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(8,5))
df["Emoción"].value_counts().plot(kind='bar', color='orange')
plt.title("Frecuencia de emociones en el dataset")
plt.xlabel("Emoción")
plt.ylabel("Cantidad de Personas")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ===========================================
# MODELO DE CLASIFICACIÓN AUTOMÁTICA
# ===========================================

# Codificar etiquetas
le = LabelEncoder()
df["ClaseCodificada"] = le.fit_transform(df["ClasificaciónEdad"])

# Variables independientes (X) y dependientes (y)
X = df[["Edad"]]  # solo la edad
y = df["ClaseCodificada"]

# Separar datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear modelo (árbol de decisión)
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

# Predicciones
y_pred = modelo.predict(X_test)

# Evaluación
accuracy = accuracy_score(y_test, y_pred)
print(f" Precisión del modelo: {accuracy*100:.2f}%\n")
print(" Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Mostrar matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("\n Matriz de confusión:")
print(cm)

# ===========================================
#  PRUEBA DE PREDICCIÓN
# ===========================================

nueva_edad = int(input("\n Ingresa una edad para predecir su clasificación: "))
prediccion = modelo.predict([[nueva_edad]])
print(f"👉 Resultado: {le.inverse_transform(prediccion)[0]}")
