import time
import numpy as np
import numba

np.random.seed(102191)


@numba.njit(parallel=True)
def calc(jugadoras, tiros_libres):
  # ~ np.random.seed(102191) # para reproducir si parallel==False.
  primera_ganadora = 0
  for i in numba.prange(10000): # el ciclo for con prange() es paralelizado si parallel == True.
    # ~ np.random.seed(102191 + i * 102563) # para reproducir si parallel==True (podría mejorarse).
    mejor_p = 0
    mejor_j = None
    for j in range(0, len(jugadoras)):
      pj = jugadoras[j]
      puntaje = np.sum(np.random.rand(tiros_libres) < pj)
      if puntaje > mejor_p:
        mejor_p = puntaje
        mejor_j = j
    if mejor_j == 99:
      primera_ganadora += 1 # automáticamente pararelizado correctamente por numba (si parallel==True).
  return primera_ganadora


# calcula cuantos encestes logra un jugador con indice de enceste prob
#  que hace qyt tiros libres
def vec_ftirar(prob, qty):
  return sum(np.random.rand(qty, len(prob)) < prob)


# defino las jugadoras
mejor = 0.7
peloton = np.array(range(501, 600)) / 1000
jugadoras = np.append(peloton, mejor) # intencionalmente la mejor esta al final


t00 = time.perf_counter()
for tiros_libres in [10, 20, 50, 100, 200, 300, 400, 415, 500, 600, 700, 1000]:
  t0 = time.perf_counter()

  if 1: # usando numba
    primera_ganadora = calc(jugadoras, tiros_libres)
  else:
    primera_ganadora = 0
    for i in range(10000):
      vaciertos = vec_ftirar(jugadoras, tiros_libres) # 10 tiros libres cada jugadora
      mejor_ronda = np.argmax(vaciertos)
      if mejor_ronda == 99:
        primera_ganadora += 1

  print(tiros_libres, "\t", primera_ganadora/10000)
  print(time.perf_counter() - t0)
  print()
print(time.perf_counter() - t00)
