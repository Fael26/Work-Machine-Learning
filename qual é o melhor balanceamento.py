import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, recall_score, f1_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings

warnings.filterwarnings('ignore')

# --- Configuração ---
file_path = 'base sem balanceamento.csv'
target_column_name = 'Tem AVC?' 

# !!! ATENÇÃO: DEFINA A CLASSE POSITIVA !!!
# Verifique no output de 'value_counts()' qual é o rótulo da sua classe
# minoritária (ex: 1 para "Sim", 0 para "Não").
# Assumindo que '1' é "Sim" (a classe que queremos detectar).
POSITIVE_CLASS_LABEL = 1 

# --- Carregar Dados ---
try:
    # === CORREÇÃO APLICADA AQUI ===
    df = pd.read_csv(file_path, sep=';', encoding='latin1')

    print("--- Informações Iniciais dos Dados ---")
    df.info()
    print("\n--- Contagem de Valores na Coluna Alvo ---")
    # Este print é crucial para você definir a POSITIVE_CLASS_LABEL acima!
    print(df[target_column_name].value_counts())

    # --- Preparação dos Dados ---
    X = df.drop(columns=[target_column_name])
    y = df[target_column_name]

    # Converter colunas 'object'
    object_cols = X.select_dtypes(include=['object']).columns
    if not object_cols.empty:
        print(f"\nConvertendo colunas 'object': {list(object_cols)}")
        X = pd.get_dummies(X, columns=object_cols, drop_first=True)
    
    # Preencher NaNs/Infinitos
    X = X.fillna(X.mean())
    X.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    X = X.fillna(X.mean())

    # --- Divisão Treino/Teste ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print("\n--- Dimensões após divisão ---")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")

    # --- Definição dos Modelos e Pipelines ---
    scaler = StandardScaler()
    classifier = RandomForestClassifier(random_state=42)
    
    # Samplers
    tomek = TomekLinks()
    rus = RandomUnderSampler(random_state=42)
    smote = SMOTE(random_state=42)
    smote_tomek = SMOTETomek(random_state=42)

    pipelines = {
        "Baseline": Pipeline([('scaler', scaler), ('model', classifier)]),
        "TomekLinks": Pipeline([('scaler', scaler), ('sampler', tomek), ('model', classifier)]),
        "RUS": Pipeline([('scaler', scaler), ('sampler', rus), ('model', classifier)]),
        "TomekLinks + RUS": Pipeline([('scaler', scaler), ('sampler1', tomek), ('sampler2', rus), ('model', classifier)]),
        "SMOTE": Pipeline([('scaler', scaler), ('sampler', smote), ('model', classifier)]),
        "SMOTE + TomekLinks": Pipeline([('scaler', scaler), ('sampler', smote_tomek), ('model', classifier)]),
        "SMOTE + RUS": Pipeline([('scaler', scaler), ('sampler1', smote), ('sampler2', rus), ('model', classifier)])
    }

    # --- Treinamento e Coleta de Resultados ---
    print("\n\n--- INICIANDO AVALIAÇÃO DOS PIPELINES ---")
    
    # Lista para guardar os resultados
    results_list = []

    for name, pipe in pipelines.items():
        print(f"\n================= {name} ==================")
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            
            # Imprimir relatório completo
            print(classification_report(y_test, y_pred))
            
            # Calcular métricas específicas
            recall_pos = recall_score(y_test, y_pred, pos_label=POSITIVE_CLASS_LABEL)
            f1_pos = f1_score(y_test, y_pred, pos_label=POSITIVE_CLASS_LABEL)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            
            # Guardar resultados
            results_list.append({
                "Pipeline": name,
                f"Recall (Classe {POSITIVE_CLASS_LABEL})": recall_pos,
                f"F1-Score (Classe {POSITIVE_CLASS_LABEL})": f1_pos,
                "Balanced Accuracy": bal_acc
            })

        except Exception as e:
            print(f"Falha ao executar o pipeline {name}: {e}")

    # --- Comparação Final ---
    print("\n\n================= COMPARAÇÃO FINAL ==================")
    
    if results_list:
        # Criar um DataFrame com os resultados
        results_df = pd.DataFrame(results_list)
        
        # Ordenar pelos melhores (ex: maior F1-Score da classe positiva)
        results_df = results_df.sort_values(by=f"F1-Score (Classe {POSITIVE_CLASS_LABEL})", ascending=False)
        
        print(results_df.to_string()) # .to_string() garante que todas as colunas sejam mostradas
    else:
        print("Nenhum resultado foi coletado.")

    print("\n--- Avaliação Concluída ---")

except FileNotFoundError:
    print(f"Erro: O arquivo {file_path} não foi encontrado.")
except Exception as e:
    # Agora a mensagem de erro original de encoding não deve mais aparecer
    print(f"Ocorreu um erro inesperado: {e}")