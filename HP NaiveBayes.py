import pandas as pd
# Importação do GaussianNB <<< ALTERADO
from sklearn.naive_bayes import GaussianNB
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ignorar warnings para manter a saída limpa
warnings.filterwarnings('ignore')

# --- 1. Carregamento dos Dados ---
print("Carregando arquivos...")
try:
    file_train = 'base_balanceada_rus.csv'
    file_test = 'Base sem balanceamento.csv'
    
    df_train = pd.read_csv(file_train, sep=';', encoding='latin1')
    df_test = pd.read_csv(file_test, sep=';', encoding='latin1')
    
    print("Arquivos carregados com sucesso.")
except Exception as e:
    print(f"Erro ao carregar arquivos: {e}")
    exit()

# --- 2. Separação de Features (X) e Alvo (y) usando iloc ---
print("Separando features e alvo usando a posição das colunas (iloc)...")

y_train = df_train.iloc[:, 58]
X_train = df_train.iloc[:, 0:58]

y_test = df_test.iloc[:, 0]
X_test_raw = df_test.iloc[:, 1:60]

print(f"Shape X_train: {X_train.shape}")
print(f"Shape y_train: {y_train.shape}")
print(f"Shape original X_test: {X_test_raw.shape}")
print(f"Shape y_test: {y_test.shape}")

# --- 3. Alinhamento de Colunas ---
X_test = X_test_raw.iloc[:, 0:58]

print(f"Shape alinhado X_test: {X_test.shape}")
print("Alinhamento de colunas (features) concluído.")

# --- 4. Pré-processamento (Scaling) ---
# Obs: Gaussian Naive Bayes assume que os dados são normalmente 
# distribuídos, então o StandardScaler é benéfico aqui.
print("Aplicando StandardScaler...")
scaler = StandardScaler()

X_train_np = X_train.values
X_test_np = X_test.values

X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np) 

print("Dados padronizados.")

# --- 5. Otimização de Hiperparâmetros (BayesSearchCV) ---
print("\nIniciando a Otimização Bayesiana (BayesSearchCV) para Naive Bayes...")
print("Isso pode levar alguns minutos.")

# Instanciar o modelo base <<< ALTERADO
nb = GaussianNB()

# Definir o ESPAÇO DE BUSCA para o Naive Bayes <<< ALTERADO
# O único hiperparâmetro principal é o 'var_smoothing' (suavização de variância).
# Usamos 'log-uniform' pois o valor varia em ordens de magnitude.
search_spaces = {
    'var_smoothing': (1e-10, 1e-3, 'log-uniform')
}

# Configurar o BayesSearchCV
bayes_search = BayesSearchCV(
    estimator=nb, # <<< ALTERADO
    search_spaces=search_spaces,
    n_iter=30,  
    cv=5, 
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Treinar o BayesSearchCV nos dados de TREINO
bayes_search.fit(X_train_scaled, y_train) 

print("\nOtimização Concluída.")
print(f"Melhores hiperparâmetros encontrados: {bayes_search.best_params_}")
print(f"Melhor score (f1_weighted) na validação cruzada: {bayes_search.best_score_:.4f}")

# --- 6. Avaliação no Conjunto de Teste ---
print("\n--- Avaliação no Conjunto de TESTE (Desbalanceado) ---")

# Usar o melhor modelo encontrado pelo BayesSearchCV <<< ALTERADO
best_nb = bayes_search.best_estimator_

# Fazer previsões nos dados de TESTE <<< ALTERADO
y_pred_test = best_nb.predict(X_test_scaled)

# Exibir o relatório de classificação
print("\nRelatório de Classificação (Teste):")
print(classification_report(y_test, y_pred_test, zero_division=0))

# Calcular a Matriz de Confusão
print("Matriz de Confusão (Teste):")
conf_matrix = confusion_matrix(y_test, y_pred_test)
print(conf_matrix)


# --- 7. Salvar Matriz de Confusão como PNG ---
print("\nGerando e salvando a Matriz de Confusão como PNG...")

try:
    class_labels = bayes_search.classes_
    
    plt.figure(figsize=(10, 7))
    
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    
    # Título e nome do arquivo atualizados <<< ALTERADO
    plt.title('Matriz de Confusão - Naive Bayes (Conjunto de Teste)')
    plt.ylabel('Classe Verdadeira (Real)')
    plt.xlabel('Classe Prevista')
    
    nome_arquivo_png = 'matriz_confusao_naive_bayes.png'
    
    plt.savefig(nome_arquivo_png)
    
    print(f"Gráfico salvo com sucesso como: {nome_arquivo_png}")
    
    plt.close()

except Exception as e:
    print(f"Ocorreu um erro ao salvar o gráfico PNG: {e}")