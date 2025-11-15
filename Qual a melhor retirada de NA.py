import pandas as pd
import numpy as np

# --- 0. Importações e Configurações ---
# A importação deve ser a primeira do sklearn para habilitar a funcionalidade experimental
from sklearn.experimental import enable_iterative_imputer

# Importações principais
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, IterativeImputer
# Importamos modelos leves para a imputação (BayesianRidge) e para o classificador (LogisticRegression)
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import LogisticRegression, BayesianRidge 
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score


# --- 1. Carregamento e Pré-processamento dos Dados ---

# Nome do arquivo fornecido
filename = 'base_4.0_sem_outliers_.csv'

# Carregar o dataset, corrigindo o erro de codificação com 'latin-1'
print("Tentando carregar o arquivo com codificação 'latin-1'...")
try:
    df = pd.read_csv(filename, delimiter=';', encoding='latin-1') 
except Exception as e:
    try:
        df = pd.read_csv(filename, delimiter=';', encoding='iso-8859-1')
        print("Carregado com sucesso usando 'iso-8859-1'.")
    except Exception as e:
        print(f"Erro fatal ao ler o arquivo: {e}")
        exit()
else:
    print("Carregado com sucesso usando 'latin-1'.")


# Definir a coluna alvo
target_col = 'Tem AVC?'

# Fazer uma cópia para processamento
df_processed = df.copy()

# Mapear a variável alvo 'Tem AVC?' para 0 e 1. (1=Não, 2=Sim)
# A classe 1 ('Tem AVC') é a classe positiva/minorítaria.
df_processed[target_col] = df_processed[target_col].map({1: 0, 2: 1})

# Remover linhas onde o alvo é NaN após o mapeamento
df_processed = df_processed.dropna(subset=[target_col])

# Converter o tipo do alvo para inteiro
df_processed[target_col] = df_processed[target_col].astype(int)

# Separar features (X) e alvo (y)
y = df_processed[target_col]
X = df_processed.drop(columns=[target_col])

# Tratar colunas de features: Substituir placeholders ('99', ';;') por NaN
X = X.replace([99, '99', 99.0, '99.0'], np.nan)
X = X.apply(pd.to_numeric, errors='coerce')

# Remover colunas com mais de 90% de NaN (Prática de limpeza)
missing_percent = X.isnull().sum() / len(X)
cols_to_drop = missing_percent[missing_percent > 0.90].index
X = X.drop(columns=cols_to_drop)
print(f"Colunas removidas devido a >90% de NA: {list(cols_to_drop)}")


# Verificar quantos valores faltantes temos
total_nan = X.isna().sum().sum()
print(f"Total de valores ausentes (NaN) nas features após a limpeza: {total_nan}")
print(f"Shape dos dados (linhas, colunas) para treino: {X.shape}")
print("---")

# --- 2. Divisão dos Dados ---

# Dividir em conjuntos de treino e teste, mantendo a proporção da classe alvo (stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# --- 3. Criação dos Pipelines ---

# O classificador que será usado após a imputação
classifier = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear') 

# Pipeline 1: KNN Imputer
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', KNNImputer(n_neighbors=3)), 
    ('model', classifier)
])

# Pipeline 2: IterativeImputer (Simulando o MissForest)
# CORREÇÃO: Usando BayesianRidge como estimador para evitar MemoryError
iterative_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', IterativeImputer(
        estimator=BayesianRidge(), # Modelo mais leve para imputação
        max_iter=3, 
        random_state=42,
        initial_strategy='mean'
    )),
    ('model', classifier)
])

# --- 4. Treinamento e Avaliação ---

results = {}

print("Treinando e avaliando o Pipeline com KNN Imputer...")
knn_pipeline.fit(X_train, y_train)
y_pred_knn = knn_pipeline.predict(X_test)
y_proba_knn = knn_pipeline.predict_proba(X_test)[:, 1]

print("\n--- Resultados do KNN Imputer ---")
print(classification_report(y_test, y_pred_knn, target_names=['Não Tem AVC (0)', 'Tem AVC (1)'], zero_division=0))

# Cálculo de Métricas e armazenamento
results['knn'] = {
    'precision': precision_score(y_test, y_pred_knn, pos_label=1, average='binary', zero_division=0),
    'recall': recall_score(y_test, y_pred_knn, pos_label=1, average='binary', zero_division=0),
    'f1_score': classification_report(y_test, y_pred_knn, output_dict=True, zero_division=0)['1']['f1-score'],
    'auc_roc': roc_auc_score(y_test, y_proba_knn)
}

# ---

print("\nTreinando e avaliando o Pipeline com IterativeImputer (MissForest-style)...")
iterative_pipeline.fit(X_train, y_train)
y_pred_iterative = iterative_pipeline.predict(X_test)
y_proba_iterative = iterative_pipeline.predict_proba(X_test)[:, 1]

print("\n--- Resultados do IterativeImputer (MissForest-style) ---")
print(classification_report(y_test, y_pred_iterative, target_names=['Não Tem AVC (0)', 'Tem AVC (1)'], zero_division=0))

# Cálculo de Métricas e armazenamento
results['missforest'] = {
    'precision': precision_score(y_test, y_pred_iterative, pos_label=1, average='binary', zero_division=0),
    'recall': recall_score(y_test, y_pred_iterative, pos_label=1, average='binary', zero_division=0),
    'f1_score': classification_report(y_test, y_pred_iterative, output_dict=True, zero_division=0)['1']['f1-score'],
    'auc_roc': roc_auc_score(y_test, y_proba_iterative)
}


# --- 5. Comparação Final das Métricas ---

print("\n\n=========================================================================")
print("             COMPARAÇÃO FINAL DE MÉTICAS (Foco: Classe 'Tem AVC')        ")
print("=========================================================================")
print(f"Imputação          | Precisão | Revocação | F1-Score | AUC-ROC")
print(f"-------------------|----------|-----------|----------|---------")
print(f"KNN Imputer        | {results['knn']['precision']:8.4f} | {results['knn']['recall']:9.4f} | {results['knn']['f1_score']:8.4f} | {results['knn']['auc_roc']:7.4f}")
print(f"MissForest-style   | {results['missforest']['precision']:8.4f} | {results['missforest']['recall']:9.4f} | {results['missforest']['f1_score']:8.4f} | {results['missforest']['auc_roc']:7.4f}")
print("=========================================================================")