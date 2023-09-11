import time
import numpy as np
import numba


np.random.seed(102191)

@numba.njit(parallel=True)
def calc(jugadoras, tiros_libres, repeticiones):
  # ~ np.random.seed(102191) # para reproducir si parallel==False.
  primera_ganadora = 0
  for i in numba.prange(repeticiones): # el ciclo for con prange() es paralelizado si parallel == True.
    # ~ np.random.seed(102191 + i * 102563) # para reproducir si parallel==True (podría mejorarse).
    mejor_p = 0
    mejor_j = None
    for j in range(0, len(jugadoras)):
      pj = jugadoras[j]
      puntaje = np.sum(np.random.rand(tiros_libres) < pj)
      if puntaje > mejor_p:
        mejor_p = puntaje
        mejor_j = j
    if mejor_j == len(jugadoras)-1:
      primera_ganadora += 1 # automáticamente pararelizado correctamente por numba (si parallel==True).
  return primera_ganadora


# defino las jugadoras
mejor = 0.7
peloton = np.array(range(501, 600)) / 1000
jugadoras = np.append(peloton, mejor) # intencionalmente la mejor esta al final
repeticiones = 10000


t00 = time.perf_counter()
for tiros_libres in [100]: # [10, 20, 50, 100, 200, 300, 400, 415, 500, 600, 700, 1000]:
  t0 = time.perf_counter()

  primera_ganadora = calc(jugadoras, tiros_libres, repeticiones)

  print(tiros_libres, "\t", primera_ganadora/repeticiones)
  print(time.perf_counter() - t0)
  print()
print(time.perf_counter() - t00)
