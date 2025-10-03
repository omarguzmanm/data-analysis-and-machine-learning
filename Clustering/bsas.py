import numpy as np
import matplotlib.pyplot as plt

def bsas_clustering(X, theta, max_clusters):
    """
    Algoritmo Básico de Agrupamiento Secuencial (BSAS).

    :param X: Matriz de datos (numpy array, donde cada fila es un punto).
    :param theta: Umbral de disimilitud (distancia máxima para crear un nuevo clúster).
    :param max_clusters: Número máximo de clústeres permitidos.
    :return: Lista de etiquetas de clúster para cada punto.
    """
    # 1. Inicialización

    # Asigna el primer punto al clúster 0. Las etiquetas son las que devolveremos.
    labels = np.full(len(X), -1)
    labels[0] = 0

    # Lista para almacenar los centroides de los clústeres.
    centroids = [X[0].copy()]

    # Contador de clústeres creados.
    num_clusters = 1

    # Función para calcular la distancia euclidiana al cuadrado (más rápido).
    def euclidean_distance_sq(a, b):
        return np.sum((a - b) ** 2)

    # Convertimos theta al cuadrado para evitar calcular la raíz cuadrada en cada iteración
    # (ya que solo comparamos distancias y no necesitamos el valor real).
    theta_sq = theta ** 2

    # 2. Procesamiento Secuencial
    for i in range(1, len(X)):
        point = X[i]

        # A) Buscar el clúster más cercano
        min_distance_sq = float('inf')
        closest_cluster_index = -1

        for k in range(num_clusters):
            centroid = centroids[k]
            # Medir la distancia del punto al centroide (prototipo)
            dist_sq = euclidean_distance_sq(point, centroid)

            if dist_sq < min_distance_sq:
                min_distance_sq = dist_sq
                closest_cluster_index = k

        # B) Criterio de Asignación/Creación

        # Si la distancia es mayor al umbral Y no hemos alcanzado el límite de clústeres
        if min_distance_sq > theta_sq and num_clusters < max_clusters:
            # Crear un nuevo clúster (Cluster $m+1$)
            labels[i] = num_clusters
            centroids.append(point.copy())
            num_clusters += 1

        else:
            # Asignar a Ck y actualizar su centroide
            labels[i] = closest_cluster_index

            # Recalcular el centroide del clúster Ck de forma incremental
            current_cluster_indices = np.where(labels == closest_cluster_index)[0]
            count = len(current_cluster_indices)

            # Fórmula de actualización del centroide:
            # nuevo_centroide = (viejo_centroide * (count - 1) + nuevo_punto) / count

            # En la práctica, con el punto ya añadido, simplemente recalculamos el promedio
            # de todos los puntos que ahora pertenecen al clúster.
            # En el esquema puramente secuencial y online, la actualización es incremental
            # para no necesitar todos los puntos anteriores, pero para este ejemplo simple
            # y por robustez, recalcularemos el centroide como el promedio.
            centroids[closest_cluster_index] = np.mean(X[current_cluster_indices], axis=0)

    return labels, centroids, num_clusters


# -----------------------------------
## 🚀 Ejemplo de Uso
# -----------------------------------

# 1. Generar datos de ejemplo (3 clústeres en 2D)
np.random.seed(42)
N = 300
X1 = np.random.randn(N // 3, 2) + np.array([5, 5])
X2 = np.random.randn(N // 3, 2) + np.array([-5, -5])
X3 = np.random.randn(N // 3, 2) + np.array([5, -5])
X = np.vstack([X1, X2, X3])
np.random.shuffle(X)  # Es crucial para un algoritmo secuencial

# 2. Definir parámetros
THETA = 4.0  # Umbral de distancia. Un valor más bajo crea más clústeres.
MAX_CLUSTERS = 10  # Límite máximo de clústeres

# 3. Ejecutar el algoritmo BSAS
labels, centroids, final_clusters = bsas_clustering(X, THETA, MAX_CLUSTERS)

print(f"Parámetros: Θ = {THETA}, q = {MAX_CLUSTERS}")
print(f"Número final de clústeres encontrados: {final_clusters}")

# 4. Visualizar los resultados
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)

# Dibujar los centroides
centroids = np.array(centroids)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroides')

# Crear leyenda para los clústeres
legend1 = plt.legend(*scatter.legend_elements(), title="Clústeres")
plt.gca().add_artist(legend1)

plt.title(f'Agrupamiento Secuencial Básico (BSAS) con Θ={THETA}')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.grid(True, alpha=0.3)
plt.show()