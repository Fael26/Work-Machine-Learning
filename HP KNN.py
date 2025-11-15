import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import numpy as np

# --- Importações para visualização ---
import matplotlib.pyplot as plt  # <<< ADICIONADO
import seaborn as sns            # <<< ADICIONADO

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

# Base de Treino (Balanceada): 59 colunas
# Features: Colunas 0 a 57 (58 features)
# Alvo: Coluna 58
y_train = df_train.iloc[:, 58]
X_train = df_train.iloc[:, 0:58]

# Base de Teste (Desbalanceada): 60 colunas
# Alvo: Coluna 0
# Features: Colunas 1 a 59 (59 features)
y_test = df_test.iloc[:, 0]
X_test_raw = df_test.iloc[:, 1:60]

print(f"Shape X_train: {X_train.shape}")
print(f"Shape y_train: {y_train.shape}")
print(f"Shape original X_test: {X_test_raw.shape}")
print(f"Shape y_test: {y_test.shape}")

# --- 3. Alinhamento de Colunas ---
# Seleciona as primeiras 58 features do teste para bater com o treino
X_test = X_test_raw.iloc[:, 0:58]

print(f"Shape alinhado X_test: {X_test.shape}")
print("Alinhamento de colunas (features) concluído.")

# --- 4. Pré-processamento (Scaling) ---
print("Aplicando StandardScaler...")
scaler = StandardScaler()

X_train_np = X_train.values
X_test_np = X_test.values

X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np) 

print("Dados padronizados.")

# --- 5. Otimização de Hiperparâmetros (BayesSearchCV) ---
print("\nIniciando a Otimização Bayesiana (BayesSearchCV)...")
print("Isso pode levar alguns minutos.")

knn = KNeighborsClassifier()

# Definir o ESPAÇO DE BUSCA para o BayesSearchCV
search_spaces = {
    'n_neighbors': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),           # Range de inteiros de 1 a 10
    'weights': ['uniform', 'distance'],  # Categórico
    'metric': ['minkowski', 'manhattan'] # Categórico
}

# Configurar o BayesSearchCV
bayes_search = BayesSearchCV(
    estimator=knn,
    search_spaces=search_spaces,
    n_iter=30,  # Vamos testar 30 combinações "inteligentes"
    cv=5, 
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1,
    random_state=42 # Para reprodutibilidade
)

# Treinar o BayesSearchCV nos dados de TREINO (balanceados)
bayes_search.fit(X_train_scaled, y_train)

print("\nOtimização Concluída.")
print(f"Melhores hiperparâmetros encontrados: {bayes_search.best_params_}")
print(f"Melhor score (f1_weighted) na validação cruzada: {bayes_search.best_score_:.4f}")

# --- 6. Avaliação no Conjunto de Teste ---
print("\n--- Avaliação no Conjunto de TESTE (Desbalanceado) ---")

# Usar o melhor modelo encontrado pelo BayesSearchCV
best_knn = bayes_search.best_estimator_

# Fazer previsões nos dados de TESTE (desbalanceados)
y_pred_test = best_knn.predict(X_test_scaled)

# Exibir o relatório de classificação
print("\nRelatório de Classificação (Teste):")
print(classification_report(y_test, y_pred_test, zero_division=0))

# Calcular a Matriz de Confusão
print("Matriz de Confusão (Teste):")
conf_matrix = confusion_matrix(y_test, y_pred_test)
print(conf_matrix)


# --- 7. Salvar Matriz de Confusão como PNG --- # <<< Bloco ADICIONADO
print("\nGerando e salvando a Matriz de Confusão como PNG...")

try:
    # Define o nome das classes (rótulos)
    # Pega os nomes das classes que o modelo aprendeu (ex: 0 e 1)
    class_labels = bayes_search.classes_
    
    # Define o tamanho da figura
    plt.figure(figsize=(10, 7))
    
    # Cria o heatmap (mapa de calor)
    # annot=True: Mostra os números dentro das células
    # fmt='d': Formata os números como inteiros
    # cmap='Blues': Esquema de cores
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    
    # Adiciona títulos e rótulos aos eixos
    plt.title('Matriz de Confusão - KNN (Conjunto de Teste)')
    plt.ylabel('Classe Verdadeira (Real)')
    plt.xlabel('Classe Prevista')
    
    # Define o nome do arquivo
    nome_arquivo_png = 'matriz_confusao_knn.png'
    
    # Salva a figura no disco
    plt.savefig(nome_arquivo_png)
    
    print(f"Gráfico salvo com sucesso como: {nome_arquivo_png}")
    
    # Fecha a figura para liberar memória
    plt.close()

except Exception as e:
    print(f"Ocorreu um erro ao salvar o gráfico PNG: {e}")