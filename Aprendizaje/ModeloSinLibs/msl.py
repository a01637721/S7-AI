#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv, math, random, sys, time
from typing import List, Dict, Any, Tuple
from multiprocessing import Pool

# Ruta al archivo de datos y proporción de datos para prueba
DATA_FILE = "Covid Data.csv"
TEST_SPLIT = 0.2
SAMPLE_LIMIT = None  # Limita el número de filas para pruebas rápidas

# Parámetros del bosque aleatorio
NUM_FOREST = 50
TREE_DEPTH = 10
MIN_LEAF = 15
TREE_SAMPLE = 0.6
FEATURE_SELECT = "log2"
MAX_THRESHOLDS = 16

BALANCED_SAMPLING = True
CLASS_WEIGHT = {0: 1.0, 1: 5.0}
PARALLEL_TREES = False
THRESHOLD_OPT = "f1"

# Diccionario para mapear nombres de columnas
COLUMN_MAP: Dict[str, List[str]] = {
    "DATE_DIED": ["date_died","DATE_DIED"],
    "SEX": ["sex","SEX"],
    "AGE": ["age","AGE"],
    "PATIENT_TYPE": ["patient_type","PATIENT_TYPE"],
    "PNEUMONIA": ["pneumonia","PNEUMONIA"],
    "PREGNANT": ["pregnant","PREGNANT","pregnancy","PREGNANCY"],
    "DIABETES": ["diabetes","DIABETES"],
    "COPD": ["copd","COPD"],
    "ASTHMA": ["asthma","ASTHMA"],
    "INMSUPR": ["inmsupr","INMSUPR","immunosuppressed","IMMUNOSUPR"],
    "HYPERTENSION": ["hypertension","HYPERTENSION","hipertension","HIPERTENSION"],
    "CARDIOVASCULAR": ["cardiovascular","CARDIOVASCULAR"],
    "RENAL_CHRONIC": ["renal_chronic","RENAL_CHRONIC","chronic_renal","CHRONIC_RENAL"],
    "OTHER_DISEASE": ["other_disease","OTHER_DISEASE","other_diseases","OTHER_DISEASES"],
    "OBESITY": ["obesity","OBESITY"],
    "TOBACCO": ["tobacco","TOBACCO"],
    "USMER": ["usmer","USMER","usmr","USMR"],
    "MEDICAL_UNIT": ["medical_unit","MEDICAL_UNIT"],
    "INTUBED": ["intubed","INTUBED"],
    "ICU": ["icu","ICU"],
    "CLASSIFICATION_FINAL": [
        "classification_final","CLASSIFICATION_FINAL",
        "classification","CLASSIFICATION","clasiffication_final","CLASIFFICATION_FINAL"
    ],
}

# Lista de características a usar en el modelo
FEATURES_LIST = [
    "SEX","AGE","PATIENT_TYPE","PNEUMONIA","PREGNANT","DIABETES","COPD","ASTHMA",
    "INMSUPR","HYPERTENSION","CARDIOVASCULAR","RENAL_CHRONIC","OTHER_DISEASE",
    "OBESITY","TOBACCO","USMER","MEDICAL_UNIT","INTUBED","ICU","CLASSIFICATION_FINAL"
]

def open_csv_file(path: str):
    # Intenta abrir el archivo CSV con diferentes codificaciones
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            file = open(path, newline="", encoding=encoding)
            _ = file.readline(); file.seek(0)
            return file
        except Exception:
            continue
    raise RuntimeError("No se pudo abrir el archivo CSV con las codificaciones estándar.")

def map_headers(reader: csv.DictReader) -> Dict[str, str]:
    # Mapea los nombres de las columnas reales a los nombres canónicos
    actual_headers = [h.strip() for h in (reader.fieldnames or [])]
    lower_map = {h.lower(): h for h in actual_headers}
    result = {}
    for canonical, aliases in COLUMN_MAP.items():
        for alias in aliases:
            if alias.lower() in lower_map:
                result[canonical] = lower_map[alias.lower()]
                break
    return result

def to_float(value: str, default: float = 0.0) -> float:
    # Convierte un valor de texto a flotante, usando un valor por defecto si falla
    value = value.strip()
    if value == "" or value.upper() == "NA": return default
    try: return float(int(value))
    except: 
        try: return float(value)
        except: return default

def died_to_binary(val: str) -> int:
    # Convierte la columna de fecha de muerte a binario (1 si murió, 0 si no)
    return 0 if val.strip() == "9999-99-99" else 1

def read_data(path: str, features: List[str]) -> Tuple[List[List[float]], List[int]]:
    # Carga el dataset y lo transforma en X (características) y y (etiquetas)
    file = open_csv_file(path)
    reader = csv.DictReader(file)
    header_map = map_headers(reader)

    print("\n== Columnas detectadas ==")
    for col in sorted(header_map): print(f"{col:22s} <- {header_map[col]}")
    missing = [col for col in features + ["DATE_DIED"] if col not in header_map]
    if missing:
        print("\n[ADVERTENCIA] Faltan columnas:")
        for m in missing: print("  -", m)
        print()

    X, y = [], []
    for row in reader:
        try:
            if "DATE_DIED" not in header_map: continue
            label = died_to_binary(row[header_map["DATE_DIED"]])
            values = []
            valid = True
            for col in features:
                if col not in header_map: valid = False; break
                values.append(to_float(row[header_map[col]], 0.0))
            if valid: X.append(values); y.append(label)
        except Exception:
            continue
    file.close()
    return X, y

def stratified_sample(X, y, n):
    # Realiza un muestreo estratificado para mantener la proporción de clases
    if n is None or n >= len(X): return X, y
    positives = [i for i, t in enumerate(y) if t==1]
    negatives = [i for i, t in enumerate(y) if t==0]
    random.shuffle(positives); random.shuffle(negatives)
    num_pos = min(len(positives), n//5)
    num_neg = min(len(negatives), n - num_pos)
    selected = positives[:num_pos] + negatives[:num_neg]
    random.shuffle(selected)
    return [X[i] for i in selected], [y[i] for i in selected]

def split_data(X, y, ratio=0.2):
    # Divide los datos en conjuntos de entrenamiento y prueba
    indices = list(range(len(X))); random.shuffle(indices)
    test_count = int(len(X) * ratio)
    test_indices = set(indices[:test_count])
    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(len(X)):
        if i in test_indices:
            X_test.append(X[i]); y_test.append(y[i])
        else:
            X_train.append(X[i]); y_train.append(y[i])
    return X_train, y_train, X_test, y_test

def calculate_metrics(y_true, y_pred):
    # Calcula métricas de desempeño del modelo
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt==1 and yp==1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt==0 and yp==0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt==0 and yp==1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt==1 and yp==0)
    accuracy = (tp+tn)/max(1,(tp+tn+fp+fn))
    precision = tp/max(1,(tp+fp))
    recall = tp/max(1,(tp+fn))
    f1_score = 2*precision*recall/max(1e-12,(precision+recall))
    return accuracy, precision, recall, f1_score, (tp, fp, fn, tn)

def select_features(n_features: int) -> int:
    # Determina cuántas características usar en cada división del árbol
    if isinstance(FEATURE_SELECT, int): return max(1, min(n_features, FEATURE_SELECT))
    mode = str(FEATURE_SELECT).lower()
    if mode == "sqrt": return max(1, int(math.sqrt(n_features)))
    if mode == "log2": return max(1, int(math.log2(n_features)))
    return max(1, int(math.sqrt(n_features)))

def split_by_value(index: int, value: float, dataset):
    # Divide el dataset en dos grupos según el valor de una característica
    left, right = [], []
    for row in dataset:
        (left if row[index] < value else right).append(row)
    return left, right

def weighted_gini(groups, classes):
    # Calcula el índice de Gini ponderado por los pesos de clase
    total_weight = sum(sum(CLASS_WEIGHT[row[-1]] for row in g) for g in groups)
    gini = 0.0
    for group in groups:
        group_weight = sum(CLASS_WEIGHT[row[-1]] for row in group)
        if group_weight == 0: continue
        score = 0.0
        for c in classes:
            class_weight = sum(CLASS_WEIGHT[row[-1]] for row in group if row[-1]==c)
            p = class_weight / group_weight
            score += p*p
        gini += (1.0 - score) * (group_weight / max(1e-12,total_weight))
    return gini

def find_best_split(dataset):
    # Busca el mejor punto de división para el nodo actual
    class_labels = list(set(row[-1] for row in dataset))
    num_features = len(dataset[0]) - 1
    selected = random.sample(list(range(num_features)), select_features(num_features))

    best_idx, best_val, best_score, best_groups = None, None, float("inf"), None
    for idx in selected:
        values = sorted(set(row[idx] for row in dataset))
        if len(values) > MAX_THRESHOLDS:
            step = len(values) / (MAX_THRESHOLDS + 1)
            candidates = [values[int((i+1)*step)] for i in range(MAX_THRESHOLDS)]
        else:
            candidates = values
        for val in candidates:
            groups = split_by_value(idx, val, dataset)
            gini = weighted_gini(groups, class_labels)
            if gini < best_score:
                best_idx, best_val, best_score, best_groups = idx, val, gini, groups
    return {"index": best_idx, "value": best_val, "groups": best_groups}

def terminal_node(group):
    # Determina la clase mayoritaria en un grupo
    labels = [row[-1] for row in group]
    return max(set(labels), key=labels.count)

def recursive_split(node, max_depth, min_size, depth):
    # Realiza la división recursiva de los nodos del árbol
    left, right = node["groups"]; node.pop("groups", None)
    if not left or not right:
        t = terminal_node(left + right)
        node["left"] = t; node["right"] = t; return
    if depth >= max_depth:
        node["left"], node["right"] = terminal_node(left), terminal_node(right); return
    if len(left) <= min_size:
        node["left"] = terminal_node(left)
    else:
        node["left"] = find_best_split(left)
        recursive_split(node["left"], max_depth, min_size, depth+1)
    if len(right) <= min_size:
        node["right"] = terminal_node(right)
    else:
        node["right"] = find_best_split(right)
        recursive_split(node["right"], max_depth, min_size, depth+1)

def create_tree(train, max_depth, min_size):
    # Construye un árbol de decisión a partir de los datos de entrenamiento
    root = find_best_split(train)
    recursive_split(root, max_depth, min_size, 1)
    return root

def tree_predict(node, row_features):
    # Predice la clase de una fila usando el árbol
    if row_features[node["index"]] < node["value"]:
        return tree_predict(node["left"], row_features) if isinstance(node["left"], dict) else node["left"]
    else:
        return tree_predict(node["right"], row_features) if isinstance(node["right"], dict) else node["right"]

def random_sample(dataset, ratio):
    # Realiza un muestreo aleatorio simple del dataset
    sample, n = [], max(1, int(round(len(dataset) * ratio)))
    for _ in range(n): sample.append(random.choice(dataset))
    return sample

def balanced_sample(dataset, ratio=1.0, pos_label=1):
    # Realiza un muestreo balanceado entre clases
    positives = [r for r in dataset if r[-1]==pos_label]
    negatives = [r for r in dataset if r[-1]!=pos_label]
    total = max(2, int(round(len(dataset)*ratio)))
    num_pos = min(len(positives), total//2)
    num_neg = min(len(negatives), total - num_pos)
    if num_pos == 0: num_pos = min(1, len(positives))
    if num_neg == 0: num_neg = min(1, len(negatives))
    sample = random.sample(positives, num_pos) + random.sample(negatives, num_neg)
    random.shuffle(sample)
    return sample

def _single_tree(args):
    # Entrena un solo árbol, usado para paralelización
    dataset, max_depth, min_size, sample_size, seed, balanced = args
    random.seed(seed)
    sample = balanced_sample(dataset, sample_size) if balanced else random_sample(dataset, sample_size)
    return create_tree(sample, max_depth, min_size)

def train_forest(X, y, n_trees, max_depth, min_size, sample_size, balanced=True):
    # Entrena un bosque aleatorio de árboles de decisión
    dataset = [x + [int(yv)] for x, yv in zip(X, y)]
    trees = []
    t0 = time.perf_counter()
    for i in range(n_trees):
        t1 = time.perf_counter()
        sample = balanced_sample(dataset, sample_size) if balanced else random_sample(dataset, sample_size)
        tree = create_tree(sample, max_depth, min_size)
        trees.append(tree)
        print(f"[RF] Árbol {i+1}/{n_trees} en {time.perf_counter()-t1:.2f}s "
              f"(acum {time.perf_counter()-t0:.2f}s)")
    return trees

def train_forest_parallel(X, y, n_trees, max_depth, min_size, sample_size, balanced=True):
    # Entrena el bosque aleatorio usando múltiples procesos
    dataset = [x + [int(yv)] for x, yv in zip(X, y)]
    args = [(dataset, max_depth, min_size, sample_size, 1337+i, balanced) for i in range(n_trees)]
    with Pool() as pool:
        trees = pool.map(_single_tree, args)
    return trees

def forest_predict_proba(trees, row):
    # Calcula la probabilidad de la clase positiva según los votos de los árboles
    votes = sum(1 if tree_predict(t, row) == 1 else 0 for t in trees)
    return votes / max(1, len(trees))

def optimize_threshold(trees, Xval, yval, metric="f1"):
    # Busca el umbral óptimo para la clasificación binaria
    best_thr, best_score = 0.5, -1.0
    for k in range(5, 96):
        thr = k / 100.0
        yhat = [1 if forest_predict_proba(trees, x) >= thr else 0 for x in Xval]
        acc, prec, rec, f1, _ = calculate_metrics(yval, yhat)
        score = {"f1": f1, "recall": rec}.get(metric, f1)
        if score > best_score:
            best_score, best_thr = score, thr
    return best_thr, best_score

def main():
    # Función principal: carga datos, entrena el modelo y evalúa resultados
    print("Cargando datos…")
    X, y = read_data(DATA_FILE, FEATURES_LIST)

    if SAMPLE_LIMIT is not None:
        X, y = stratified_sample(X, y, SAMPLE_LIMIT)
        print(f"[INFO] Usando subset de {len(X)} filas para prototipo.")

    pos = sum(y); neg = len(y)-pos
    print(f"\nTotal filas válidas: {len(X)} | Positivos (muertes): {pos} | Negativos (vivos): {neg}")

    if len(X) == 0:
        print("[ERROR] No se cargaron filas. Revisa el mapeo/CSV_PATH."); sys.exit(1)

    # División en conjuntos de entrenamiento, validación y prueba
    X_train, y_train, X_test, y_test = split_data(X, y, ratio=TEST_SPLIT)
    X_subtrain, y_subtrain, X_val, y_val = split_data(X_train, y_train, ratio=0.12)

    print(f"Train: {len(X_subtrain)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print("Entrenando Random Forest…")

    if PARALLEL_TREES:
        trees = train_forest_parallel(X_subtrain, y_subtrain, NUM_FOREST, TREE_DEPTH, MIN_LEAF, TREE_SAMPLE, BALANCED_SAMPLING)
    else:
        trees = train_forest(X_subtrain, y_subtrain, NUM_FOREST, TREE_DEPTH, MIN_LEAF, TREE_SAMPLE, BALANCED_SAMPLING)

    thr, best = optimize_threshold(trees, X_val, y_val, metric=THRESHOLD_OPT)
    print(f"\nUmbral óptimo ({THRESHOLD_OPT.upper()}): {thr:.2f}  (score={best:.4f})")

    # Evaluación en el conjunto de prueba usando el umbral óptimo
    y_pred = [1 if forest_predict_proba(trees, x) >= thr else 0 for x in X_test]
    acc, prec, rec, f1, (tp, fp, fn, tn) = calculate_metrics(y_test, y_pred)

    print("\n=== Resultados en TEST ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nMatriz de confusión:")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    
if __name__ == "__main__":
    main() 