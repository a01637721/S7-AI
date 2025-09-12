# Proyecto: Random Forest desde cero para predicción de muertes por COVID

## Proceso

- **Carga y preprocesa** un archivo CSV con datos de pacientes.
- **Mapea automáticamente** los nombres de columnas del archivo a los nombres esperados por el modelo.
- **Convierte la variable objetivo** `DATE_DIED` en binaria:  
  - `1` = Muerto  
  - `0` = Vivo
- **Entrena un Random Forest** (bosque de árboles de decisión) desde cero, aplicando técnicas avanzadas:
  - Bootstrap balanceado por árbol (igual cantidad de positivos y negativos en cada muestra).
  - Índice de Gini ponderado por pesos de clase (para tratar el desbalance de clases).
  - Selección aleatoria de características en cada split del árbol.
  - Límite de thresholds por característica para acelerar el entrenamiento.
  - Búsqueda automática del umbral óptimo de clasificación (maximizando F1 o Recall).
- **Evalúa el modelo** mostrando métricas como Accuracy, Precision, Recall, F1-score y matriz de confusión.

---

## Funcionamiento

1. **Carga de datos:**  
   El script lee el archivo CSV, detecta las columnas relevantes y convierte los datos a formato numérico.

2. **Preprocesamiento:**  
   Convierte la columna `DATE_DIED` en binaria (muerto/vivo).  
   Selecciona las características relevantes para el modelo.

3. **División de datos:**  
   Separa los datos en entrenamiento, validación y prueba.

4. **Entrenamiento del Random Forest:**  
   - Cada árbol se entrena con una muestra balanceada de los datos (bootstrap balanceado).
   - En cada nodo del árbol, se selecciona aleatoriamente un subconjunto de características.
   - El mejor split se decide usando el índice de Gini ponderado por los pesos de clase.
   - Se limita el número de thresholds candidatos por característica para acelerar el proceso.

5. **Predicción y búsqueda de umbral óptimo:**  
   - El bosque predice la probabilidad de muerte para cada paciente (por votación de los árboles).
   - Se busca el umbral que maximiza la métrica elegida (F1 o Recall) en el conjunto de validación.

6. **Evaluación:**  
   - Se evalúa el modelo en el conjunto de prueba y se muestran las métricas principales.

---

## Características usadas en el modelo

Las siguientes columnas son utilizadas para predecir la muerte del paciente:

- SEX
- AGE
- PATIENT_TYPE
- PNEUMONIA
- PREGNANT
- DIABETES
- COPD
- ASTHMA
- INMSUPR
- HYPERTENSION
- CARDIOVASCULAR
- RENAL_CHRONIC
- OTHER_DISEASE
- OBESITY
- TOBACCO
- USMER
- MEDICAL_UNIT
- INTUBED
- ICU
- CLASSIFICATION_FINAL

---

## Parámetros principales

Puedes modificar estos parámetros en el código para ajustar el modelo:

- `NUM_FOREST`: Número de árboles en el bosque.
- `TREE_DEPTH`: Profundidad máxima de cada árbol.
- `MIN_LEAF`: Mínimo de muestras por hoja.
- `TREE_SAMPLE`: Proporción de datos usados por árbol.
- `FEATURE_SELECT`: Método para seleccionar características por split (`log2`, `sqrt`, o número fijo).
- `MAX_THRESHOLDS`: Máximo de thresholds candidatos por característica.
- `CLASS_WEIGHT`: Pesos para cada clase (para balancear el desbalance).
- `THRESHOLD_OPT`: Métrica para optimizar el umbral (`f1` o `recall`).

---