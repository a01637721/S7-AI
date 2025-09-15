import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import time

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class COVIDPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.optimal_threshold = 0.5
        self.best_params = {}
        
    def load_and_preprocess_data(self):
        """Carga y preprocesa los datos"""
        print("Cargando datos...")
        
        # Cargar datos
        df = pd.read_csv(self.data_path, encoding='latin-1')
        
        # Corregir nombre de columna
        if 'CLASIFFICATION_FINAL' in df.columns and 'CLASSIFICATION_FINAL' not in df.columns:
            df.rename(columns={'CLASIFFICATION_FINAL': 'CLASSIFICATION_FINAL'}, inplace=True)
        
        # Crear variable target
        df['TARGET'] = df['DATE_DIED'].apply(
            lambda x: 0 if str(x).strip() in ['9999-99-99', 'NA', ''] else 1
        )
        
        # Definir características
        self.feature_names = [
            'AGE', 'ASTHMA', 'CARDIOVASCULAR', 'CLASSIFICATION_FINAL', 'COPD',
            'DIABETES', 'ICU', 'INMSUPR', 'INTUBED', 'MEDICAL_UNIT', 'OBESITY',
            'OTHER_DISEASE', 'PATIENT_TYPE', 'PNEUMONIA', 'PREGNANT', 
            'RENAL_CHRONIC', 'SEX', 'TOBACCO', 'USMER'
        ]
        
        # Filtrar solo las columnas que existen
        self.feature_names = [f for f in self.feature_names if f in df.columns]
        
        print(f"Características usadas: {len(self.feature_names)}")
        
        # Eliminar filas con valores missing
        df_clean = df[self.feature_names + ['TARGET']].dropna()
        
        X = df_clean[self.feature_names]
        y = df_clean['TARGET']
        
        print(f"Datos finales: {X.shape[0]} muestras, {X.shape[1]} características")
        print(f"Distribución de clases: {y.value_counts().to_dict()}")
        print(f"Porcentaje positivo: {(y.sum() / len(y) * 100):.2f}%")
        
        return X, y
    
    def find_optimal_threshold(self, y_true, y_pred_proba):
        """Encuentra el umbral óptimo que maximiza el F1-Score"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
        
        optimal_idx = np.argmax(f1_scores[:-1])
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        return optimal_threshold, optimal_f1
    
    def optimize_hyperparameters(self, X_train, y_train):
        """Optimiza los hiperparámetros usando GridSearchCV"""
        print("\nIniciando optimización de hiperparámetros...")
        
        # Definir el espacio de búsqueda MÁS EFICIENTE
        param_grid = {
            'C': [0.01, 0.1, 1, 10],  # Valores más comunes
            'penalty': ['l2'],  # Solo L2 para mayor estabilidad
            'solver': ['lbfgs', 'liblinear'],  # Algoritmos más eficientes
            'class_weight': ['balanced', {0: 1, 1: 3}, {0: 1, 1: 5}],
            'max_iter': [2000]  # Más iteraciones para converger
        }
        
        # Crear el modelo base con más iteraciones
        logreg = LogisticRegression(random_state=42, max_iter=2000)
        
        # Configurar GridSearchCV más rápido
        grid_search = GridSearchCV(
            estimator=logreg,
            param_grid=param_grid,
            scoring='f1',
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        
        # Ejecutar la búsqueda
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"Tiempo de optimización: {(end_time - start_time):.2f} segundos")
        print(f"Mejores parámetros encontrados:")
        for param, value in grid_search.best_params_.items():
            print(f"   {param}: {value}")
        print(f"Mejor F1-Score (CV): {grid_search.best_score_:.4f}")
        
        self.best_params = grid_search.best_params_
        return grid_search.best_estimator_
    
    def train_model(self, X, y):
        """Entrena el modelo con optimización de hiperparámetros"""
        print("\nEntrenando modelo con optimización...")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Balancear datos
        print("Aplicando técnicas de balanceo...")
        smote = SMOTE(random_state=42, sampling_strategy=0.5)
        under = RandomUnderSampler(random_state=42, sampling_strategy=0.8)
        
        X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
        X_resampled, y_resampled = under.fit_resample(X_resampled, y_resampled)
        
        print(f"Datos después de balanceo: {X_resampled.shape}")
        print(f"Nueva distribución: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        # Optimizar hiperparámetros
        self.model = self.optimize_hyperparameters(X_resampled, y_resampled)
        
        # Verificar convergencia y entrenar final si es necesario
        if hasattr(self.model, 'n_iter_') and self.model.n_iter_[0] >= self.model.max_iter:
            print("El modelo no convergió, entrenando con más iteraciones...")
            self.model.set_params(max_iter=5000)
            self.model.fit(X_resampled, y_resampled)
        
        # Encontrar umbral óptimo
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        self.optimal_threshold, optimal_f1 = self.find_optimal_threshold(y_test, y_pred_proba)
        
        print(f"Umbral óptimo encontrado: {self.optimal_threshold:.4f}")
        print(f"F1-Score con umbral óptimo: {optimal_f1:.4f}")
        
        # Evaluar modelo
        self.evaluate_model(X_test_scaled, y_test)
        
        # Generar gráficas
        y_pred_optimal = self.predict_with_threshold(X_test_scaled)
        self.plot_confusion_matrix(confusion_matrix(y_test, y_pred_optimal))
        self.plot_roc_curve(y_test, y_pred_proba)
        self.plot_precision_recall_curve(y_test, y_pred_proba)
        self.plot_feature_importance()
        
        return self.model
    
    def predict_with_threshold(self, X):
        """Predice usando el umbral óptimo"""
        probabilities = self.model.predict_proba(X)[:, 1]
        return (probabilities >= self.optimal_threshold).astype(int)
    
    def evaluate_model(self, X_test, y_test):
        """Evalúa el modelo con el umbral óptimo"""
        y_pred = self.predict_with_threshold(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = auc(*roc_curve(y_test, y_pred_proba)[:2])
        
        print("\n" + "="*65)
        print("RESULTADOS OPTIMIZADOS DEL MODELO")
        print("="*65)
        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Precision:         {precision:.4f}")
        print(f"Recall:            {recall:.4f}")
        print(f"F1-Score:          {f1:.4f}")
        print(f"AUC-ROC:           {roc_auc:.4f}")
        print(f"Umbral usado:      {self.optimal_threshold:.4f}")
        
        # Información de convergencia
        if hasattr(self.model, 'n_iter_'):
            print(f"Iteraciones usadas:   {self.model.n_iter_[0]}")
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        print("\nMatriz de Confusión:")
        print(cm)
        
        # Métricas detalladas
        tn, fp, fn, tp = cm.ravel()
        print(f"\nMétricas detalladas:")
        print(f"Verdaderos Negativos:  {tn}")
        print(f"Falsos Positivos:      {fp}")
        print(f"Falsos Negativos:      {fn}") 
        print(f"Verdaderos Positivos:  {tp}")
        
        # Reporte de clasificación
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred))
        
        # Mostrar mejores parámetros
        print("Mejores hiperparámetros encontrados:")
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
    
    def plot_confusion_matrix(self, cm):
        """Grafica la matriz de confusión"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Murió', 'Murió'],
                   yticklabels=['No Murió', 'Murió'])
        plt.title('Matriz de Confusión (con optimización)')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Predicción')
        plt.tight_layout()
        plt.savefig('confusion_matrix_optimized.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    
    def plot_roc_curve(self, y_test, y_pred_proba):
        """Grafica la curva ROC"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC - Con optimización')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('roc_curve_optimized.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    
    def plot_precision_recall_curve(self, y_test, y_pred_proba):
        """Grafica la curva Precision-Recall"""
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Curva Precision-Recall (Optimizada)')
        plt.grid(True)
        
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
        optimal_idx = np.argmax(f1_scores[:-1])
        plt.scatter(recall[optimal_idx], precision[optimal_idx], color='red', s=100, 
                   label=f'Óptimo (F1={f1_scores[optimal_idx]:.3f})')
        plt.legend()
        
        plt.savefig('precision_recall_optimized.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    
    def plot_feature_importance(self):
        """Grafica la importancia de las características"""
        importance = np.abs(self.model.coef_[0])
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(importance)), importance[indices])
        plt.title("Importancia de Características (Optimizada)", fontsize=16)
        plt.xticks(range(len(importance)), 
                  [self.feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        plt.ylabel('Importancia (valor absoluto del coeficiente)')
        plt.tight_layout()
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        plt.savefig('feature_importance_optimized.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Importancia de características optimizada guardada")

# Ejecución automática
if __name__ == "__main__":
    print("="*75)
    print("ALGORITMO CON OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("="*75)
    
    predictor = COVIDPredictor("Covid Data.csv")
    
    try:
        X, y = predictor.load_and_preprocess_data()
        model = predictor.train_model(X, y)
        
        print("\n" + "="*75)
        print("Modelo TERMINADO")
        print("="*75)
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()