import numpy as np
import matplotlib.pyplot as plt

def bsas_clustering(puntos_de_datos, umbral, max_clusters, orden_procesamiento):
    """
    Implementación del algoritmo de agrupamiento BSAS (Basic Sequential Algorithmic Scheme).

    Args:
        puntos_de_datos (np.array): Matriz de datos donde las columnas son muestras y las filas son características.
        umbral (float): La distancia máxima (theta) para que un punto se una a un clúster existente.
        max_clusters (int): El número máximo de clústeres (q) que se pueden crear.
        orden_procesamiento (list or np.array): El orden en el que se procesarán los puntos de datos.

    Returns:
        tuple: Una tupla que contiene las etiquetas de los clústeres asignados y los centroides finales.
    """
    num_caracteristicas, num_muestras = puntos_de_datos.shape

    # Si se proporciona un orden, reordena los datos para su procesamiento
    if len(orden_procesamiento) == num_muestras:
        puntos_de_datos = puntos_de_datos[:, orden_procesamiento]

    # Inicialización
    num_clusters_actual = 1
    asignaciones = np.zeros(num_muestras, dtype=int)

    # El primer punto de datos forma el primer clúster
    asignaciones[0] = num_clusters_actual
    centroides = puntos_de_datos[:, [0]]  # El primer centroide es el primer punto

    # Itera a través de los puntos de datos restantes
    for i in range(1, num_muestras):
        punto_actual = puntos_de_datos[:, i]

        # Calcula la distancia euclidiana del punto actual a todos los centroides existentes
        distancias = np.linalg.norm(centroides - punto_actual.reshape(-1, 1), axis=0)

        distancia_minima = np.min(distancias)
        indice_centroide_cercano = np.argmin(distancias)

        # Si el punto está "lejos" y no hemos alcanzado el máximo de clústeres
        if distancia_minima > umbral and num_clusters_actual < max_clusters:
            num_clusters_actual += 1
            asignaciones[i] = num_clusters_actual
            # El nuevo punto se convierte en un nuevo centroide
            centroides = np.hstack((centroides, punto_actual.reshape(-1, 1)))
       # Si el punto está "cerca" de un clúster existente
        else:
            # Asigna el punto al clúster más cercano (índice + 1 para etiquetas base 1)
            asignaciones[i] = indice_centroide_cercano + 1

            # Actualiza el centroide del clúster asignado usando un promedio móvil
            etiqueta_asignada = asignaciones[i]
            puntos_en_cluster = np.sum(asignaciones[:i + 1] == etiqueta_asignada)

            centroide_a_actualizar = centroides[:, indice_centroide_cercano]
            nuevo_centroide = ((puntos_en_cluster - 1) * centroide_a_actualizar + punto_actual) / puntos_en_cluster
            centroides[:, indice_centroide_cercano] = nuevo_centroide

    return asignaciones, centroides


# Ejecución

np.random.seed(0)
cluster1 = np.random.randn(2, 20) + np.array([[5], [5]])
cluster2 = np.random.randn(2, 20) + np.array([[0], [0]])
cluster3 = np.random.randn(2, 20) + np.array([[5], [-5]])

X = np.hstack((cluster1, cluster2, cluster3))

# Parámetros del algoritmo
orden_aleatorio = np.random.permutation(X.shape[1])
umbral_distancia = 4.0
clusters_maximos = 5

etiquetas, centroides_finales = bsas_clustering(X, umbral_distancia, clusters_maximos, orden_aleatorio)

# Visualización de los resultados
plt.figure(figsize=(8, 6))
for k in np.unique(etiquetas):
    indices = np.where(etiquetas == k)
    plt.scatter(X[0, indices], X[1, indices], label=f'Clúster {k}')

plt.scatter(centroides_finales[0, :], centroides_finales[1, :], c='black', marker='x', s=100, label='Centroides')
plt.title('Agrupamiento BSAS')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.grid(True)
plt.show()