# intencionalmente la mejor jugadora va al final de la lista de jugadoras
# porque la funcion np.argmax() de Python hace trampa
# si hay un empate ( dos máximos) se queda con el que esta primero en el vector
import  numpy as np
import  time

np.random.seed(102191)

# calcula cuantos encestes logra un jugador con indice de enceste prob
#  que hace qyt tiros libres
def vec_ftirar(prob, qty):
  return sum(np.random.rand(qty, len(prob)) < prob)


# defino las jugadoras
mejor = 0.7
peloton = np.array(range(501, 600)) / 1000
jugadoras = np.append(peloton, mejor)




t0 = time.perf_counter()


for tiros_libres in [10, 20, 50, 100, 200, 300, 400, 415, 500, 600, 700, 1000]:
  t00 = time.perf_counter()

  primera_ganadora = 0
  for i in range(10000):
    vaciertos = vec_ftirar(jugadoras, tiros_libres) # 10 tiros libres cada jugadora
    mejor_ronda = np.argmax(vaciertos)
    if mejor_ronda == 99:
      primera_ganadora += 1
  print(tiros_libres, "\t", primera_ganadora/10000)
  print(time.perf_counter() - t00)
  print()

t1 = time.perf_counter()
print(t1-t0)
