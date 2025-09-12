import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate
from filterpy.kalman import KalmanFilter

# ===============================
# 1. Leer señal ECG desde MIT-BIH
# ===============================

# Cambia esta ruta a donde extrajiste el ZIP
path = "C:\\Users\\USUARIO\\Desktop\\SemestreVII\\Repo_A01637721\\Hardware\\DigitalizaciónFiltradoDatos"

# Leer registro
record = wfdb.rdrecord(path)
data_raw = record.p_signal[:,0]   # canal 1 (MLII)
fs = record.fs  # frecuencia de muestreo original (360 Hz)

print(f"Frecuencia original: {fs} Hz")
print(f"Tamaño señal: {len(data_raw)} muestras")

# ===============================
# 2. Submuestreo (discretización en tiempo)
# ===============================
factor = 6  # 360 Hz -> 60 Hz
data_subsampled = decimate(data_raw, factor)
fs_new = fs // factor
print(f"Frecuencia después de submuestreo: {fs_new} Hz")

# ===============================
# 3. Cuantización (discretización en amplitud)
# ===============================
def quantize(signal, levels):
    min_val, max_val = np.min(signal), np.max(signal)
    q = np.linspace(min_val, max_val, levels)
    indices = np.digitize(signal, q) - 1
    return q[indices]

data_digitized = quantize(data_subsampled, levels=16)

# ===============================
# 4. Filtro de Kalman
# ===============================
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.F = np.array([[1, 1], [0, 1]])   # modelo transición
kf.H = np.array([[1, 0]])           # observación
kf.P *= 1000                        # cov inicial
kf.R = 5                            # ruido medición
kf.Q = np.eye(2) * 0.01             # ruido proceso

kalman_estimates = []
for z in data_digitized:
    kf.predict()
    kf.update(z)
    kalman_estimates.append(kf.x[0])
kalman_estimates = np.array(kalman_estimates)

# ===============================
# 5. Observador de Luenberger
# ===============================
A = np.array([[1, 1],[0, 1]])
B = np.array([[0],[0]])
C = np.array([[1, 0]])
L = np.array([[0.5],[0.1]])   # ganancia del observador

x_hat = np.zeros((2,1))
luenberger_estimates = []
u = np.zeros((len(data_digitized),1))  # sin entrada

for k in range(len(data_digitized)):
    y = np.array([[data_digitized[k]]])
    x_hat = A @ x_hat + B @ u[k] + L @ (y - C @ x_hat)
    luenberger_estimates.append(float(C @ x_hat))
luenberger_estimates = np.array(luenberger_estimates)

# ===============================
# 6. Graficar resultados
# ===============================
plt.figure(figsize=(15,6))
plt.plot(data_subsampled[:1000], label="Original Submuestreada")
plt.plot(data_digitized[:1000], label="Cuantizada", alpha=0.7)
plt.plot(kalman_estimates[:1000], label="Kalman")
plt.plot(luenberger_estimates[:1000], label="Luenberger")
plt.legend()
plt.title("Comparación de Señales ECG (MIT-BIH)")
plt.xlabel("Muestras")
plt.ylabel("Amplitud (mV)")
plt.grid(True)
plt.show()
