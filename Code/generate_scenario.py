from functionsSetup import generateSetup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_data(L, K, N, tau_p, ASD_varphi, numActiveAPs, grid, semilla=False, N_max=20,
             interference_scope="all"):

    # Llamamos a la función para generar los parámetros del entorno
    gainOverNoisedB, powgain, active_APs, distances, pilotIndex = generateSetup(
        L=L, K=K, N=N, tau_p=tau_p, ASD_varphi=ASD_varphi, numActiveAPs=numActiveAPs, grid=grid, semilla=semilla
    )
    max_gain = np.max(powgain)

    if interference_scope == "all":
        considered = np.ones((L, K), dtype=bool)
    elif interference_scope == "serving":
        considered = np.zeros((L, K), dtype=bool)
        for k in range(K):
            considered[active_APs[k], k] = True
    else:
        raise ValueError(f"interference_scope must be 'all' or 'serving', got {interference_scope!r}")

    considered_power = considered * powgain                              # (L, K)
    total_power = considered_power.sum(axis=1, keepdims=True)            # (L, 1)
    interference_matrix = total_power - powgain * considered             # (L, K), own contribution subtracted

    # Inicializar la lista para almacenar los datos
    data = []

    # Construir cada fila del DataFrame
    for ue_id in range(K):  # Recorrer cada usuario
        max_active = max(active_APs[ue_id])
        for ap_id in active_APs[ue_id]:
            # Crear un diccionario para cada combinación AP-UE
            data.append({
                "UE_id": ue_id,  # Identificador del usuario
                "AP_id": ap_id,  # Identificador de la AP
                "user_gain": (powgain[ap_id, ue_id])/(max_active),
                "interference_sum": interference_matrix[ap_id, ue_id]/(max_active),
                "distances": distances[ap_id, ue_id],
                "pilot": pilotIndex[ue_id],  # MODIFIED: carried through for pilot-contamination edges
            })

        for ap_id in range(numActiveAPs+1, N_max+1):
            data.append({
                "UE_id": ue_id,
                "AP_id": ap_id + L,  # Indicar que es una AP "falsa" o de relleno
                "user_gain": 0.0,
                "interference_sum": 0.0,
                "distances": 0.0,
                "pilot": pilotIndex[ue_id],
            })

    # Convertir la lista a un DataFrame de pandas
    df_simulation = pd.DataFrame(data)
    # print(df_simulation.head(21))

    return active_APs, df_simulation, gainOverNoisedB, powgain, max_gain, pilotIndex, interference_matrix

# get_data(50, 30, 1, 4, 10 * (3.14159 / 180), 10, grid=True, semilla=1)
