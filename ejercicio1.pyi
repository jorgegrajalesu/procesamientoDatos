import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Datos iniciales
data = {
    'Nombre': ['Ana', 'Luis', 'Marta', 'Carlos', 'Sofía', 'Pedro', 'Elena', 'Juan', 'Lucía', 'Andrés'],
    'Edad': [25, 32, 29, 40, 22, 35, 28, 31, 27, 30],
    'Estado': ['Feliz', 'Triste', 'Serio', 'Amargo', 'Feliz', 'Triste', 'Feliz', 'Serio', 'Amargo', 'Feliz']
}

# Crear un DataFrame
df = pd.DataFrame(data)

# Mostrar los datos
print("Datos originales:\n", df)

# Estadísticas básicas
print("\n Promedio de edad:", df['Edad'].mean())

# Contar personas por estado emocional
conteo_estados = df['Estado'].value_counts()
print("\n Conteo por estado emocional:\n", conteo_estados)

# Visualización
sns.countplot(x='Estado', data=df, palette='pastel')
plt.title("Distribución de estados emocionales")
plt.xlabel("Estado emocional")
plt.ylabel("Cantidad de personas")
plt.show()
