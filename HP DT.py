import pandas as pd
# Importação do DecisionTreeClassifier <<< ALTERADO
from sklearn.tree import DecisionTreeClassifier
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
# Obs: Árvores de decisão não são sensíveis à escala, 
# mas mantemos o passo para seguir a estrutura original.
print("Aplicando StandardScaler...")
scaler = StandardScaler()

X_train_np = X_train.values
X_test_np = X_test.values

X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np) 

print("Dados padronizados.")

# --- 5. Otimização de Hiperparâmetros (BayesSearchCV) ---
print("\nIniciando a Otimização Bayesiana (BayesSearchCV) para Árvore de Decisão...")
print("Isso pode levar alguns minutos.")

# Instanciar o modelo base <<< ALTERADO
tree = DecisionTreeClassifier(random_state=42)

# Definir o ESPAÇO DE BUSCA para a Árvore de Decisão <<< ALTERADO
search_spaces = {
    'criterion': ['gini', 'entropy'],           # Categórico
    'max_depth': (3, 50),                       # Range de inteiros
    'min_samples_split': (2, 20),               # Range de inteiros
    'min_samples_leaf': (1, 20)                 # Range de inteiros
}

# Configurar o BayesSearchCV
bayes_search = BayesSearchCV(
    estimator=tree, # <<< ALTERADO
    search_spaces=search_spaces,
    n_iter=30,  
    cv=5, 
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Treinar o BayesSearchCV nos dados de TREINO
# Usando X_train_scaled para manter consistência com o script anterior
bayes_search.fit(X_train_scaled, y_train) 

print("\nOtimização Concluída.")
print(f"Melhores hiperparâmetros encontrados: {bayes_search.best_params_}")
print(f"Melhor score (f1_weighted) na validação cruzada: {bayes_search.best_score_:.4f}")

# --- 6. Avaliação no Conjunto de Teste ---
print("\n--- Avaliação no Conjunto de TESTE (Desbalanceado) ---")

# Usar o melhor modelo encontrado pelo BayesSearchCV <<< ALTERADO
best_tree = bayes_search.best_estimator_

# Fazer previsões nos dados de TESTE <<< ALTERADO
y_pred_test = best_tree.predict(X_test_scaled)

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
    plt.title('Matriz de Confusão - Árvore de Decisão (Conjunto de Teste)')
    plt.ylabel('Classe Verdadeira (Real)')
    plt.xlabel('Classe Prevista')
    
    nome_arquivo_png = 'matriz_confusao_arvore.png'
    
    plt.savefig(nome_arquivo_png)
    
    print(f"Gráfico salvo com sucesso como: {nome_arquivo_png}")
    
    plt.close()

except Exception as e:
    print(f"Ocorreu um erro ao salvar o gráfico PNG: {e}")