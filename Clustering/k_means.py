import random
import math
import matplotlib.pyplot as plt

def calcular_distancia_euclidiana(punto_a, punto_b):
    """Calcula la distancia euclidiana entre dos puntos (listas)."""
    sum_sq_diff = 0
    for i in range(len(punto_a)):
        sum_sq_diff += (punto_a[i] - punto_b[i]) ** 2
    return math.sqrt(sum_sq_diff)


def calcular_centroide(grupo_puntos):
    """Calcula el nuevo centroide como la media de un grupo de puntos."""
    if not grupo_puntos:
        return None

    dimension = len(grupo_puntos[0])
    nuevo_centroide = [0] * dimension

    for punto in grupo_puntos:
        for i in range(dimension):
            nuevo_centroide[i] += punto[i]

    num_puntos = len(grupo_puntos)
    for i in range(dimension):
        nuevo_centroide[i] /= num_puntos

    return nuevo_centroide


def k_means(datos, k, max_iter=100):
    """
    Implementa el algoritmo k-means.}
    """

    if k > len(datos):
        raise ValueError("K no puede ser mayor que el número de datos.")

    centroides = random.sample(datos, k)
    asignaciones = [-1] * len(datos)

    for iteracion in range(max_iter):

        centroides_ant = [list(c) for c in centroides]

        # FASE 1: ASIGNACIÓN (E-step)
        for i, punto in enumerate(datos):
            min_distancia = float('inf')
            mejor_centroide_indice = -1

            for j, centroide in enumerate(centroides):
                distancia = calcular_distancia_euclidiana(punto, centroide)

                if distancia < min_distancia:
                    min_distancia = distancia
                    mejor_centroide_indice = j

            asignaciones[i] = mejor_centroide_indice

        # FASE 2: ACTUALIZACIÓN (M-step)
        nuevos_grupos = [[] for _ in range(k)]

        for i, punto in enumerate(datos):
            cluster_idx = asignaciones[i]
            nuevos_grupos[cluster_idx].append(punto)

        centroides_actualizados = []
        for i, grupo in enumerate(nuevos_grupos):
            if grupo:
                centroides_actualizados.append(calcular_centroide(grupo))
            else:
                centroides_actualizados.append(random.choice(datos))

        centroides = centroides_actualizados

        # VERIFICAR CONVERGENCIA
        convergencia = True
        for i in range(k):
            if calcular_distancia_euclidiana(centroides[i], centroides_ant[i]) > 1e-4:
                convergencia = False
                break

        if convergencia:
            break

    return centroides, asignaciones


# Uso y ploteo del algoritmo k-means

# Datos de ejemplo (puntos en 2D)
datos_ejemplo = [
    [1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0],
    [3.5, 5.0], [4.5, 5.0], [3.5, 4.5], [10.0, 10.0],
    [9.0, 8.0], [8.0, 9.0], [11.0, 11.0], [12.0, 10.0],
    [0.5, 0.8], [2.0, 1.8], [4.0, 6.0], [4.8, 4.2]
]

K_CLUSTERS = 3

if __name__ == "__main__":
    centroides_finales, asignaciones_finales = k_means(datos_ejemplo, K_CLUSTERS)

    print(f"--- Algoritmo k-means (K={K_CLUSTERS}) ---")
    print("Centroides Finales:")
    for i, c in enumerate(centroides_finales):
        c_formateado = [round(coord, 4) for coord in c]
        print(f"Cluster {i}: {c_formateado}")

    print("\nAsignaciones de Datos (Punto -> Cluster):")
    for i, (punto, cluster) in enumerate(zip(datos_ejemplo, asignaciones_finales)):
        print(f"{punto} -> Cluster {cluster}")

    # PLOTEO DE RESULTADOS -
    if len(datos_ejemplo[0]) == 2: # Solo si los datos son 2D
        plt.figure(figsize=(8, 6))

        # Generar una lista de colores para los clusters
        # Se asegura que haya suficientes colores para K_CLUSTERS
        colores = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        colores_clusters = colores[:K_CLUSTERS] if K_CLUSTERS <= len(colores) else plt.cm.get_cmap('hsv', K_CLUSTERS)(
            range(K_CLUSTERS))

        # Plotear cada punto de dato con el color de su cluster asignado
        for i, punto in enumerate(datos_ejemplo):
            cluster_idx = asignaciones_finales[i]
            plt.scatter(punto[0], punto[1], color=colores_clusters[cluster_idx], alpha=0.7,
                        s=100)  # s es el tamaño del punto

        # Plotear los centroides finales
        for i, centroide in enumerate(centroides_finales):
            plt.scatter(centroide[0], centroide[1], color='black', marker='X', s=300, edgecolor='white', linewidth=1,
                        label=f'Centroide {i}')

        plt.title(f'Resultados de k-means con K={K_CLUSTERS}')
        plt.xlabel('Dimensión 1')
        plt.ylabel('Dimensión 2')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("\nNo se puede plotear porque los datos no son bidimensionales.")