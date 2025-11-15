import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import warnings

warnings.filterwarnings('ignore')

# --- Configuração ---
file_path = 'base sem balanceamento.csv'
target_column_name = 'Tem AVC?'
output_file_name = 'base_balanceada_rus.csv'

try:
    # --- Carregar Dados ---
    # Usando a correção de encoding que descobrimos
    df = pd.read_csv(file_path, sep=';', encoding='latin1')

    print(f"Arquivo '{file_path}' carregado com sucesso.")
    print("\n--- Contagem Original (Antes do Balanceamento) ---")
    print(df[target_column_name].value_counts())

    # --- Preparação dos Dados ---
    
    # Separar X (features) e y (alvo)
    X = df.drop(columns=[target_column_name])
    y = df[target_column_name]

    # Lidar com colunas não numéricas em X (ex: 'object')
    object_cols = X.select_dtypes(include=['object']).columns
    if not object_cols.empty:
        print(f"\nConvertendo colunas 'object': {list(object_cols)}")
        X = pd.get_dummies(X, columns=object_cols, drop_first=True)
    
    # Preencher NaNs/Infinitos (importante para o RUS funcionar)
    X = X.fillna(X.mean())
    X.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    X = X.fillna(X.mean())

    print("\n--- Aplicando o RandomUnderSampler (RUS) ---")

    # --- Aplicar o Balanceamento RUS ---
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    print("\n--- Contagem Nova (Depois do Balanceamento) ---")
    print(y_resampled.value_counts())

    # --- Salvar o Novo Arquivo ---
    
    # Juntar o X e y balanceados em um único DataFrame
    df_balanced = X_resampled.copy()
    df_balanced[target_column_name] = y_resampled
    
    # Salvar em um novo CSV
    df_balanced.to_csv(output_file_name, sep=';', index=False)

    print(f"\n--- SUCESSO! ---")
    print(f"Base de dados balanceada foi salva em: '{output_file_name}'")

except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")