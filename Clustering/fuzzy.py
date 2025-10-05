import random
import math
import matplotlib.pyplot as plt


def calcular_distancia_euclidiana(punto_a, punto_b):
    sum_sq_diff = 0
    for i in range(len(punto_a)):
        sum_sq_diff += (punto_a[i] - punto_b[i]) ** 2
    return math.sqrt(sum_sq_diff)


def inicializar_matriz_membresia(num_datos, k):
    U = []
    for _ in range(num_datos):
        # Generar K valores aleatorios
        row = [random.random() for _ in range(k)]
        # Normalizar para que sumen 1
        suma = sum(row)
        normalized_row = [x / suma for x in row]
        U.append(normalized_row)
    return U


def fuzzy_c_means(datos, k, m, max_iter=100, umbral=1e-4):
    """
    Implementa el algoritmo Fuzzy C-Means (FCM).
    """
    num_datos = len(datos)
    dimension = len(datos[0])

    # Inicialización de la Matriz de Pertenencia U
    U = inicializar_matriz_membresia(num_datos, k)

    # Inicialización de Centroides V (calculados a partir de U inicial)
    V = [[0.0] * dimension for _ in range(k)]

    for iteracion in range(max_iter):
        U_ant = [row[:] for row in U]  # Copia profunda de U anterior

        # FASE 1: ACTUALIZACIÓN DE CENTROIDES (V)
        for j in range(k):  # Para cada centroide j
            num_v = [0.0] * dimension
            den_v = 0.0

            for i in range(num_datos):  # Para cada dato i
                # Exponente de la membresía: u_ij^m
                membresia_m = U[i][j] ** m

                # Numerador: Sum(u_ij^m * x_i)
                for d in range(dimension):
                    num_v[d] += membresia_m * datos[i][d]

                # Denominador: Sum(u_ij^m)
                den_v += membresia_m

            # Nuevo centroide v_j = Num / Den
            if den_v > 0:
                for d in range(dimension):
                    V[j][d] = num_v[d] / den_v
            # Si el denominador es cero (cluster vacío), el centroide se queda igual
            # o podrías reinicializarlo, pero aquí lo mantenemos.

        # FASE 2: ACTUALIZACIÓN DE LA MATRIZ DE PERTENENCIA (U)
        for i in range(num_datos):  # Para cada dato i
            punto = datos[i]

            for j in range(k):  # Para cada cluster j
                dist_ij = calcular_distancia_euclidiana(punto, V[j])

                if dist_ij == 0:
                    # Si la distancia es cero, la pertenencia es 1 para este cluster, 0 para los demás
                    U[i] = [0.0] * k
                    U[i][j] = 1.0
                    break

                # Cálculo de la nueva membresía U_ij
                suma_denominadores = 0.0
                for l in range(k):
                    dist_il = calcular_distancia_euclidiana(punto, V[l])

                    if dist_il == 0:
                        suma_denominadores = float('inf')  # Esto asegura que el denominador sea el dominante
                        break

                    # Expresión (dist_ij / dist_il) ^ (2 / (m-1))
                    exponente = 2 / (m - 1)
                    suma_denominadores += (dist_ij / dist_il) ** exponente

                # U_ij = 1 / Sumatoria
                if suma_denominadores == float('inf'):
                    U[i][j] = 0.0
                else:
                    U[i][j] = 1.0 / suma_denominadores

            # Normalizar U[i] si fue modificado (para corregir posibles errores de punto flotante)
            if not math.isclose(sum(U[i]), 1.0):
                suma = sum(U[i])
                if suma > 0:
                    U[i] = [x / suma for x in U[i]]

        # VERIFICAR CONVERGENCIA
        max_cambio_U = 0.0
        for i in range(num_datos):
            for j in range(k):
                cambio = abs(U[i][j] - U_ant[i][j])
                if cambio > max_cambio_U:
                    max_cambio_U = cambio

        if max_cambio_U < umbral:
            # print(f"Fuzzy C-Means convergió en la iteración {iteracion + 1}")
            break

    return V, U


# Ejecución y ploteo de resultados

# Datos de ejemplo (puntos en 2D, similares al k-means)
datos_ejemplo = [
    [1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0],
    [3.5, 5.0], [4.5, 5.0], [3.5, 4.5], [10.0, 10.0],
    [9.0, 8.0], [8.0, 9.0], [11.0, 11.0], [12.0, 10.0],
    [0.5, 0.8], [2.0, 1.8], [4.0, 6.0], [4.8, 4.2]
]

K_CLUSTERS = 3
FUZZY_EXPONENT = 2.0  # Valor común, m=2

if __name__ == "__main__":
    centroides_finales, matriz_pertenencia = fuzzy_c_means(
        datos_ejemplo, K_CLUSTERS, FUZZY_EXPONENT
    )

    print(f"--- Algoritmo Fuzzy C-Means (K={K_CLUSTERS}, m={FUZZY_EXPONENT}) ---")
    print("Centroides Finales (V):")
    for i, c in enumerate(centroides_finales):
        c_formateado = [round(coord, 4) for coord in c]
        print(f"Cluster {i}: {c_formateado}")

    print("\nPertenencia de Datos (U): [Max Membresía -> Cluster Asignado]")
    asignaciones_duras = []
    for i, (punto, membresias) in enumerate(zip(datos_ejemplo, matriz_pertenencia)):
        # Asignación 'dura' (hard assignment) basada en la máxima membresía
        max_membresia = max(membresias)
        cluster_asignado = membresias.index(max_membresia)
        asignaciones_duras.append(cluster_asignado)

        m_formateadas = [round(m, 4) for m in membresias]
        print(f"{punto} -> Cluster {cluster_asignado} (U: {m_formateadas})")

    # --- PLOTEO DE RESULTADOS ---
    if len(datos_ejemplo[0]) == 2:
        plt.figure(figsize=(8, 6))

        colores = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        colores_clusters = colores[:K_CLUSTERS] if K_CLUSTERS <= len(colores) else plt.cm.get_cmap('hsv', K_CLUSTERS)(
            range(K_CLUSTERS))

        # Plotear cada punto usando la asignación 'dura'
        for i, punto in enumerate(datos_ejemplo):
            cluster_idx = asignaciones_duras[i]

            # El tamaño del punto se pondera por su pertenencia al cluster asignado
            max_membresia = matriz_pertenencia[i][cluster_idx]
            tamanio_punto = 100 + (max_membresia * 100)

            plt.scatter(punto[0], punto[1], color=colores_clusters[cluster_idx],
                        alpha=0.7, s=tamanio_punto)

        # Plotear los centroides finales
        for i, centroide in enumerate(centroides_finales):
            plt.scatter(centroide[0], centroide[1], color='black', marker='X', s=300,
                        edgecolor='white', linewidth=1, label=f'Centroide {i}')

        plt.title(f'Resultados de Fuzzy C-Means (K={K_CLUSTERS}, m={FUZZY_EXPONENT})')
        plt.xlabel('Dimensión 1')
        plt.ylabel('Dimensión 2')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("\nNo se puede plotear porque los datos no son bidimensionales.")