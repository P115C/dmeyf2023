import time
import numpy as np
import numba


np.random.seed(102191)

@numba.njit(parallel=True)
def calc(jugadoras, tiros_libres, repeticiones):
    # ~ np.random.seed(102191) # para reproducir si parallel==False.
    puntajes = np.zeros(len(jugadoras))
    for j in numba.prange(0, len(jugadoras)):
        pj = jugadoras[j]
        puntaje = np.sum(np.random.rand(tiros_libres) < pj)
        puntajes[j] = puntaje
    return puntajes


# defino las jugadoras
# ~ mejor = 0.7
# ~ peloton = np.array(range(501, 600)) / 1000
# ~ jugadoras = np.append(peloton, mejor) # intencionalmente la mejor esta al final
jugadoras = np.repeat(0.75, 100)
repeticiones = 10000


t00 = time.perf_counter()
for tiros_libres in [100]:
    t0 = time.perf_counter()
    avg_1 = 0
    avg_2 = 0
    cant_arr = 50
    for i in range(0, cant_arr):
        puntajes = calc(jugadoras, tiros_libres, repeticiones)
        mejores_10 = np.sort(puntajes)[-1:-10:-1]
        print(tiros_libres, "\t", mejores_10)
        avg_1 += mejores_10[0]
        avg_2 += mejores_10[1]
    print()
    print("avg_1 = ", avg_1 / cant_arr)
    print("avg_2 = ", avg_2 / cant_arr)
    print(time.perf_counter() - t0)
    print()
print(time.perf_counter() - t00)
