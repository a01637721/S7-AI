#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Optional

# Configuración optimizada para dataset grande
DATA_FILE = "Covid Data.csv"
TEST_SPLIT = 0.2
LEARNING_RATE = 0.1
EPOCHS = 200
BATCH_SIZE = 2048
L2_LAMBDA = 0.01
N_FOLDS = 2
ENSEMBLE_MODELS = 3

# Diccionario para mapear nombres de columnas
COLUMN_MAP = {
    "DATE_DIED": ["date_died", "DATE_DIED"],
    "SEX": ["sex", "SEX"],
    "AGE": ["age", "AGE"],
    "PATIENT_TYPE": ["patient_type", "PATIENT_TYPE"],
    "PNEUMONIA": ["pneumonia", "PNEUMONIA"],
    "PREGNANT": ["pregnant", "PREGNANT", "pregnancy", "PREGNANCY"],
    "DIABETES": ["diabetes", "DIABETES"],
    "COPD": ["copd", "COPD"],
    "ASTHMA": ["asthma", "ASTHMA"],
    "INMSUPR": ["inmsupr", "INMSUPR", "immunosuppressed", "IMMUNOSUPR"],
    "HYPERTENSION": ["hypertension", "HYPERTENSION", "hipertension", "HIPERTENSION"],
    "CARDIOVASCULAR": ["cardiovascular", "CARDIOVASCULAR"],
    "RENAL_CHRONIC": ["renal_chronic", "RENAL_CHRONIC", "chronic_renal", "CHRONIC_RENAL"],
    "OTHER_DISEASE": ["other_disease", "OTHER_DISEASE", "other_diseases", "OTHER_DISEASES"],
    "OBESITY": ["obesity", "OBESITY"],
    "TOBACCO": ["tobacco", "TOBACCO"],
    "USMER": ["usmer", "USMER", "usmr", "USMR"],
    "MEDICAL_UNIT": ["medical_unit", "MEDICAL_UNIT"],
    "INTUBED": ["intubed", "INTUBED"],
    "ICU": ["icu", "ICU"],
    "CLASSIFICATION_FINAL": [
        "classification_final", "CLASSIFICATION_FINAL",
        "classification", "CLASSIFICATION", "clasiffication_final", "CLASIFFICATION_FINAL"
    ],
}

# Lista de características a usar en el modelo
FEATURES_LIST = [
    "SEX", "AGE", "PATIENT_TYPE", "PNEUMONIA", "PREGNANT", "DIABETES", "COPD", "ASTHMA",
    "INMSUPR", "HYPERTENSION", "CARDIOVASCULAR", "RENAL_CHRONIC", "OTHER_DISEASE",
    "OBESITY", "TOBACCO", "USMER", "MEDICAL_UNIT", "INTUBED", "ICU", "CLASSIFICATION_FINAL"
]

def open_csv_file(path: str):
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            file = open(path, newline="", encoding=encoding)
            _ = file.readline()
            file.seek(0)
            return file
        except Exception:
            continue
    raise RuntimeError("No se pudo abrir el archivo CSV con las codificaciones estándar.")

def map_headers(reader: csv.DictReader) -> Dict[str, str]:
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
    value = value.strip()
    if value == "" or value.upper() == "NA":
        return default
    try:
        return float(int(value))
    except:
        try:
            return float(value)
        except:
            return default

def died_to_binary(val: str) -> int:
    return 0 if val.strip() == "9999-99-99" else 1

def read_data(path: str, features: List[str]) -> Tuple[List[List[float]], List[int]]:
    file = open_csv_file(path)
    reader = csv.DictReader(file)
    header_map = map_headers(reader)

    print("\n== Columnas detectadas ==")
    for col in sorted(header_map):
        print(f"{col:22s} <- {header_map[col]}")
    missing = [col for col in features + ["DATE_DIED"] if col not in header_map]
    if missing:
        print("\n[ADVERTENCIA] Faltan columnas:")
        for m in missing:
            print("  -", m)
        print()

    X, y = [], []
    for row in reader:
        try:
            if "DATE_DIED" not in header_map:
                continue
            label = died_to_binary(row[header_map["DATE_DIED"]])
            values = []
            valid = True
            for col in features:
                if col not in header_map:
                    valid = False
                    break
                values.append(to_float(row[header_map[col]], 0.0))
            if valid:
                X.append(values)
                y.append(label)
        except Exception:
            continue
    file.close()
    return X, y

def normalize_features_numpy(X: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    """Normalización vectorizada con NumPy"""
    X_array = np.array(X)
    means = np.mean(X_array, axis=0)
    stds = np.std(X_array, axis=0)
    stds[stds == 0] = 1
    
    X_normalized = (X_array - means) / stds
    return X_normalized.tolist(), means.tolist(), stds.tolist()

def add_bias_term(X: List[List[float]]) -> List[List[float]]:
    return [[1.0] + list(x) for x in X]

def split_data(X, y, ratio=0.2):
    indices = list(range(len(X)))
    random.shuffle(indices)
    test_count = int(len(X) * ratio)
    test_indices = set(indices[:test_count])
    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(len(X)):
        if i in test_indices:
            X_test.append(X[i])
            y_test.append(y[i])
        else:
            X_train.append(X[i])
            y_train.append(y[i])
    return X_train, y_train, X_test, y_test

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def predict(x, weights):
    z = sum(x_i * w_i for x_i, w_i in zip(x, weights))
    return sigmoid(z)

def compute_cost(X, y, weights, l2_lambda=0.0):
    """Calcula la función de costo con regularización L2"""
    m = len(X)
    total_cost = 0.0
    
    for i in range(m):
        prediction = predict(X[i], weights)
        if prediction == 0:
            prediction = 1e-12
        if prediction == 1:
            prediction = 1 - 1e-12
            
        total_cost += -y[i] * math.log(prediction) - (1 - y[i]) * math.log(1 - prediction)
    
    reg_term = 0.0
    if l2_lambda > 0:
        reg_term = (l2_lambda / (2 * m)) * sum(w_i * w_i for w_i in weights[1:])
    
    return total_cost / m + reg_term

def calculate_metrics(y_true, y_pred, y_prob=None):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    
    accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1_score = 2 * precision * recall / max(1e-12, (precision + recall))
    
    auc_roc = 0.0
    if y_prob is not None:
        auc_roc = calculate_auc_roc(y_true, y_prob)
    
    return accuracy, precision, recall, f1_score, auc_roc, (tp, fp, fn, tn)

def calculate_auc_roc(y_true, y_prob):
    data = list(zip(y_prob, y_true))
    data.sort(key=lambda x: x[0], reverse=True)
    
    total_pos = sum(y_true)
    total_neg = len(y_true) - total_pos
    
    if total_pos == 0 or total_neg == 0:
        return 0.5
    
    tpr, fpr = [0], [0]
    current_tp, current_fp = 0, 0
    prev_prob = None
    
    for prob, label in data:
        if prob != prev_prob:
            tpr.append(current_tp / total_pos)
            fpr.append(current_fp / total_neg)
            prev_prob = prob
        
        if label == 1:
            current_tp += 1
        else:
            current_fp += 1
    
    tpr.append(1)
    fpr.append(1)
    
    auc = 0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    
    return auc

def plot_cost_history(cost_history):
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history)
    plt.title('Historial de Costo durante el Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Costo')
    plt.grid(True)
    plt.savefig('costo_entrenamiento.png', dpi=100, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(tp, fp, fn, tn):
    plt.figure(figsize=(8, 6))
    matrix = [[tn, fp], [fn, tp]]
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión')
    plt.colorbar()
    
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['Negativo', 'Positivo'])
    plt.yticks(tick_marks, ['Negativo', 'Positivo'])
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i][j]), 
                    horizontalalignment="center",
                    color="white" if matrix[i][j] > (tn+fp+fn+tp)/4 else "black")
    
    plt.tight_layout()
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.savefig('matriz_confusion.png', dpi=100, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_prob, auc_score):
    data = list(zip(y_prob, y_true))
    data.sort(key=lambda x: x[0], reverse=True)
    
    total_pos = sum(y_true)
    total_neg = len(y_true) - total_pos
    
    tpr, fpr = [0], [0]
    current_tp, current_fp = 0, 0
    prev_prob = None
    
    for prob, label in data:
        if prob != prev_prob:
            tpr.append(current_tp / total_pos)
            fpr.append(current_fp / total_neg)
            prev_prob = prob
        
        if label == 1:
            current_tp += 1
        else:
            current_fp += 1
    
    tpr.append(1)
    fpr.append(1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador aleatorio')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('curva_roc.png', dpi=100, bbox_inches='tight')
    plt.close()

def plot_feature_importance(weights, feature_names):
    feature_weights = weights[1:]
    
    sorted_indices = sorted(range(len(feature_weights)), 
                           key=lambda i: abs(feature_weights[i]), 
                           reverse=True)
    
    sorted_weights = [feature_weights[i] for i in sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 8))
    y_pos = range(len(sorted_weights))
    colors = ['red' if w < 0 else 'blue' for w in sorted_weights]
    plt.barh(y_pos, sorted_weights, align='center', color=colors)
    plt.yticks(y_pos, sorted_names)
    plt.xlabel('Importancia (valor absoluto del peso)')
    plt.title('Importancia de Características')
    plt.tight_layout()
    plt.savefig('importancia_caracteristicas.png', dpi=100, bbox_inches='tight')
    plt.close()

def downsample_data(X, y, target_ratio=0.5):
    X_neg = [x for x, label in zip(X, y) if label == 0]
    X_pos = [x for x, label in zip(X, y) if label == 1]
    y_neg = [0] * len(X_neg)
    y_pos = [1] * len(X_pos)
    
    n_pos = len(X_pos)
    n_neg_to_keep = int(n_pos * (1 - target_ratio) / target_ratio)
    
    if len(X_neg) > n_neg_to_keep:
        indices = random.sample(range(len(X_neg)), n_neg_to_keep)
        X_neg = [X_neg[i] for i in indices]
        y_neg = [0] * len(X_neg)
    
    X_balanced = X_pos + X_neg
    y_balanced = y_pos + y_neg
    
    combined = list(zip(X_balanced, y_balanced))
    random.shuffle(combined)
    X_balanced, y_balanced = zip(*combined)
    
    return list(X_balanced), list(y_balanced)

# ========== FUNCIONES OPTIMIZADAS ==========

def train_logistic_regression_vectorized(X_train, y_train, learning_rate, epochs, l2_lambda=0.0):
    """Entrenamiento vectorizado con NumPy"""
    X = np.array(X_train)
    y = np.array(y_train)
    m, n = X.shape
    
    weights = np.random.randn(n) / np.sqrt(n)
    cost_history = []
    
    for epoch in range(epochs):
        z = np.dot(X, weights)
        predictions = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        
        error = predictions - y
        gradient = np.dot(X.T, error) / m
        
        if l2_lambda > 0:
            gradient[1:] += (l2_lambda / m) * weights[1:]
        
        weights -= learning_rate * gradient
        
        if epoch % 50 == 0:
            cost = -np.mean(y * np.log(predictions + 1e-12) + 
                           (1 - y) * np.log(1 - predictions + 1e-12))
            cost_history.append(cost)
            if epoch % 100 == 0:
                print(f"Época {epoch}: Costo = {cost:.6f}")
    
    return weights.tolist(), cost_history

def train_mini_batch(X_train, y_train, learning_rate, epochs, batch_size=1024, l2_lambda=0.0):
    """Entrenamiento con mini-lotes"""
    X = np.array(X_train)
    y = np.array(y_train)
    m, n = X.shape
    weights = np.random.randn(n) / np.sqrt(n)
    cost_history = []
    
    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            batch_len = len(X_batch)
            
            if batch_len == 0:
                continue
                
            z = np.dot(X_batch, weights)
            predictions = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            error = predictions - y_batch
            gradient = np.dot(X_batch.T, error) / batch_len
            
            if l2_lambda > 0:
                gradient[1:] += (l2_lambda / m) * weights[1:]
            weights -= learning_rate * gradient
        
        if epoch % 25 == 0:
            z_full = np.dot(X, weights)
            predictions_full = 1 / (1 + np.exp(-np.clip(z_full, -500, 500)))
            cost = -np.mean(y * np.log(predictions_full + 1e-12) + 
                           (1 - y) * np.log(1 - predictions_full + 1e-12))
            cost_history.append(cost)
            if epoch % 100 == 0:
                print(f"Época {epoch}: Costo = {cost:.6f}")
    
    return weights.tolist(), cost_history

def feature_selection(X, y, feature_names, threshold=0.01):
    """Selección de características basada en importancia"""
    X_with_bias = add_bias_term(X)
    weights, _ = train_mini_batch(np.array(X_with_bias), np.array(y), 0.1, 100, 2048)
    
    feature_importance = []
    for i, name in enumerate(feature_names):
        importance = abs(weights[i+1])
        feature_importance.append((name, importance))
    
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    max_importance = max(imp for _, imp in feature_importance)
    selected_features = [name for name, imp in feature_importance 
                        if imp >= threshold * max_importance]
    
    print("\nSelección de características:")
    print("=" * 40)
    for name, imp in feature_importance:
        keep = "✓" if name in selected_features else "✗"
        print(f"{keep} {name:20s}: {imp:.6f}")
    
    return selected_features

def train_ensemble(X_train, y_train, n_models=3, learning_rate=0.1, epochs=100):
    """Entrena un ensemble de modelos optimizado"""
    models = []
    n_samples = len(X_train)
    X_array = np.array(X_train)
    y_array = np.array(y_train)
    
    print(f"Entrenando ensemble de {n_models} modelos...")
    
    for i in range(n_models):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_subset = X_array[indices]
        y_subset = y_array[indices]
        
        weights, _ = train_mini_batch(X_subset, y_subset, learning_rate, epochs, 2048)
        models.append(weights)
        
        if (i + 1) % max(1, n_models // 3) == 0:
            print(f"  Modelo {i+1}/{n_models} completado")
    
    return models

def predict_ensemble(x, models):
    """Predicción promediando múltiples modelos"""
    predictions = [predict(x, model) for model in models]
    return sum(predictions) / len(predictions)

def efficient_cross_validation(X, y, n_folds=2, n_epochs=100):
    """Validación cruzada optimizada"""
    fold_size = len(X) // n_folds
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    scores = []
    X_array = np.array(X)
    y_array = np.array(y)
    
    print(f"Realizando validación cruzada ({n_folds} folds)...")
    
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size
        
        X_val = X_array[start:end]
        y_val = y_array[start:end]
        
        X_train = np.concatenate([X_array[:start], X_array[end:]])
        y_train = np.concatenate([y_array[:start], y_array[end:]])
        
        X_train_bias = np.hstack([np.ones((len(X_train), 1)), X_train])
        X_val_bias = np.hstack([np.ones((len(X_val), 1)), X_val])
        
        weights, _ = train_mini_batch(X_train_bias, y_train, LEARNING_RATE, n_epochs, BATCH_SIZE)
        
        z = np.dot(X_val_bias, weights)
        y_prob = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        y_pred = (y_prob >= 0.5).astype(int)
        
        tp = np.sum((y_val == 1) & (y_pred == 1))
        tn = np.sum((y_val == 0) & (y_pred == 0))
        fp = np.sum((y_val == 0) & (y_pred == 1))
        fn = np.sum((y_val == 1) & (y_pred == 0))
        
        accuracy = (tp + tn) / len(y_val)
        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        f1 = 2 * precision * recall / max(1e-12, (precision + recall))
        auc = calculate_auc_roc(y_val.tolist(), y_prob.tolist())
        
        scores.append({'acc': accuracy, 'prec': precision, 'rec': recall, 'f1': f1, 'auc': auc})
        print(f"Fold {i+1}/{n_folds}: AUC={auc:.4f}, F1={f1:.4f}")
    
    avg_scores = {metric: sum(s[metric] for s in scores) / n_folds 
                 for metric in scores[0].keys()}
    
    return avg_scores, scores

def main():
    print("Cargando y procesando datos…")
    X, y = read_data(DATA_FILE, FEATURES_LIST)
    
    pos = sum(y)
    neg = len(y) - pos
    print(f"\nDataset: {len(X):,} muestras")
    print(f"Positivos (muertes): {pos:,} | Negativos (vivos): {neg:,}")
    print(f"Proporción: Positivos: {pos/len(y):.3f}, Negativos: {neg/len(y):.3f}")

    if len(X) == 0:
        print("[ERROR] No se cargaron filas. Revisa el mapeo/CSV_PATH.")
        return

    print("Normalizando características...")
    X_normalized, means, stds = normalize_features_numpy(X)
    
    print("Seleccionando características importantes...")
    selected_features = feature_selection(X_normalized, y, FEATURES_LIST, threshold=0.05)
    
    feature_indices = [i for i, name in enumerate(FEATURES_LIST) if name in selected_features]
    X_filtered = [[x[i] for i in feature_indices] for x in X_normalized]
    
    print(f"\nCaracterísticas seleccionadas: {len(selected_features)}/{len(FEATURES_LIST)}")
    
    print("Balanceando dataset...")
    X_balanced, y_balanced = downsample_data(X_filtered, y, target_ratio=0.4)
    
    print(f"Dataset balanceado: {len(X_balanced):,} muestras")
    
    avg_scores, fold_scores = efficient_cross_validation(
        X_balanced, y_balanced, n_folds=N_FOLDS, n_epochs=100
    )
    
    print("\n=== RESULTADOS VALIDACIÓN CRUZADA ===")
    print("=" * 50)
    print(f"Accuracy promedio : {avg_scores['acc']:.4f}")
    print(f"Precision promedio: {avg_scores['prec']:.4f}")
    print(f"Recall promedio   : {avg_scores['rec']:.4f}")
    print(f"F1-score promedio : {avg_scores['f1']:.4f}")
    print(f"AUC-ROC promedio  : {avg_scores['auc']:.4f}")
    
    print("\nEntrenando modelo final con ensemble...")
    X_with_bias = add_bias_term(X_balanced)
    final_models = train_ensemble(
        X_with_bias, y_balanced, 
        n_models=ENSEMBLE_MODELS,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS
    )
    
    # Obtener el historial de costo del primer modelo para graficar
    first_model_weights = final_models[0]
    cost_history = []
    for epoch in range(0, EPOCHS, 50):
        cost = compute_cost(X_with_bias, y_balanced, first_model_weights)
        cost_history.append(cost)
    
    y_pred = []
    y_prob = []
    for x in X_with_bias:
        prob = predict_ensemble(x, final_models)
        y_prob.append(prob)
        y_pred.append(1 if prob >= 0.5 else 0)
    
    acc, prec, rec, f1, auc, (tp, fp, fn, tn) = calculate_metrics(y_balanced, y_pred, y_prob)
    
    print("\n=== RESULTADOS FINALES ===")
    print("=" * 50)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print(f"\nMatriz de confusión:")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    
    plot_cost_history(cost_history)
    plot_confusion_matrix(tp, fp, fn, tn)
    plot_roc_curve(y_balanced, y_prob, auc)
    
    avg_weights = [0.0] * (len(selected_features) + 1)
    for model in final_models:
        for j in range(len(avg_weights)):
            avg_weights[j] += model[j] / len(final_models)
    
    plot_feature_importance(avg_weights, selected_features)
    

if __name__ == "__main__":
    main()