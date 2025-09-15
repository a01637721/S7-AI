import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate, welch
import zipfile, os, glob
from filterpy.kalman import KalmanFilter

# ====================================================
# 1) Rutas y extracción
# ====================================================
zip_path = "C:\\Users\\PC\\Desktop\\yael\\Hardware\\DigitalizaciónFiltradoDatos\\mit-bih-arrhythmia-database-1.0.0.zip"
extract_dir = "C:\\Users\\PC\\Desktop\\yael\\Hardware\\DigitalizaciónFiltradoDatos\\mitdb"

if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

# Buscar registros disponibles
hea_files = glob.glob(os.path.join(extract_dir, "**", "*.hea"), recursive=True)
if len(hea_files) == 0:
    raise FileNotFoundError("No encontré archivos .hea en la carpeta extraída")

print("Registros disponibles:")
for f in hea_files:
    print(os.path.basename(f))

# Usar el primero
first_record = os.path.splitext(os.path.basename(hea_files[0]))[0]
record_dir = os.path.dirname(hea_files[0])
print(f"\nUsando el rSegistro: {first_record}")

# ====================================================
# 2) Leer señal
# ====================================================
record = wfdb.rdrecord(os.path.join(record_dir, first_record))
data_raw = record.p_signal[:,0]   # canal 1
fs = record.fs
print(f"Frecuencia de muestreo original: {fs} Hz")
print(f"Tamaño de la señal: {len(data_raw)} muestras")

# ====================================================
# 3) Submuestreo
# ====================================================
fs_new = 60
factor = int(fs // fs_new)
data_subsampled = decimate(data_raw, factor)
print(f"Frecuencia de muestreo nueva: {fs_new} Hz")

t = np.arange(len(data_subsampled)) / fs_new  # tiempo en segundos

# ====================================================
# 4) Cuantización
# ====================================================
def quantize(signal, levels):
    min_val, max_val = np.min(signal), np.max(signal)
    q = np.linspace(min_val, max_val, levels)
    indices = np.digitize(signal, q) - 1
    return q[indices]

data_digitized = quantize(data_subsampled, levels=16)

# ====================================================
# 5) Filtro de Kalman (ajustado)
# ====================================================
A = np.array([[1, 1], [0, 1]])
H = np.array([[1, 0]])
Q = np.array([[1e-3, 0], [0, 1e-3]])
R = 0.5

x = np.zeros((2, 1))
P = np.eye(2)
kalman_estimates = []

for z in data_digitized:
    # Predicción
    x = A @ x
    P = A @ P @ A.T + Q
    # Medición
    z = np.array([[z]])
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    # Actualización
    x = x + K @ y
    P = (np.eye(2) - K @ H) @ P
    kalman_estimates.append(float(H @ x))
kalman_estimates = np.array(kalman_estimates)

# ====================================================
# 6) Observador de Luenberger
# ====================================================
L = np.array([[0.5],[0.2]])
x_hat = np.zeros((2,1))
luenberger_estimates = []
for z in data_digitized:
    z = np.array([[z]])
    x_hat = A @ x_hat + L @ (z - H @ x_hat)
    luenberger_estimates.append(float((H @ x_hat)[0,0]))
luenberger_estimates = np.array(luenberger_estimates)

# ====================================================
# 7) Métricas MSE y SNR
# ====================================================
def mse(ref, est):
    return np.mean((ref - est) ** 2)

def snr(ref, est):
    signal_power = np.mean(ref**2)
    noise_power = np.mean((ref - est) ** 2)
    return 10 * np.log10(signal_power / noise_power)

print("\n--- MÉTRICAS ---")
print("MSE Kalman:", mse(data_subsampled, kalman_estimates))
print("MSE Luenberger:", mse(data_subsampled, luenberger_estimates))
print("SNR Kalman:", snr(data_subsampled, kalman_estimates), "dB")
print("SNR Luenberger:", snr(data_subsampled, luenberger_estimates), "dB")

# ====================================================
# 8) Espectro Welch
# ====================================================
f_orig, Pxx_orig = welch(data_raw, fs=fs, nperseg=2048)
f_sub, Pxx_sub = welch(data_subsampled, fs=fs_new, nperseg=1024)

plt.figure(figsize=(10, 5))
plt.semilogy(f_orig, Pxx_orig, label="Original 360 Hz")
plt.semilogy(f_sub, Pxx_sub, label="Submuestreada 60 Hz")
plt.axvline(30, color='r', linestyle='--', label="Nyquist 60 Hz")
plt.title("Espectro ECG (Welch)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Densidad espectral")
plt.legend()
plt.grid()
plt.show()

# ====================================================
# 9) Graficar comparación
# ====================================================
plt.figure(figsize=(12, 6))
plt.plot(t[:1000], data_subsampled[:1000], label="Original Submuestreada", linewidth=1)
plt.plot(t[:1000], data_digitized[:1000], label="Cuantizada", alpha=0.7)
plt.plot(t[:1000], kalman_estimates[:1000], label="Kalman")
plt.plot(t[:1000], luenberger_estimates[:1000], label="Luenberger")
plt.legend()
plt.title(f"Comparación de Señales ECG ({first_record})")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid()
plt.show()
