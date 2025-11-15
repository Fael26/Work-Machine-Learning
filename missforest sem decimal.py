import pandas as pd
import numpy as np

# --- 1. Configuração ---
# Arquivo de entrada (o que você acabou de criar, cheio de floats)
input_file = "base_4.0_Sem_Ausentes.csv"
# Novo arquivo de saída
output_file = "base_4.0_Sem balancear.csv"

print(f"Iniciando conversão para inteiros do arquivo: {input_file}")

try:
    # --- 2. Carregar o arquivo ---
    # OBRIGATÓRIO usar encoding='latin-1' e sep=';' com base nos erros anteriores
    df = pd.read_csv(input_file, sep=';', encoding='latin-1')
    
    print("Arquivo carregado com sucesso.")
    print(f"Shape original: {df.shape}")
    # print("Dados originais (amostra):\n", df.head(3))

    # --- 3. Converter os dados ---
    # O Imputer (MissForest) preenche tudo, então não deve haver NaNs.
    # Ele também pode criar valores como 1.99999 ou 2.00001.
    # Portanto, arredondar primeiro (round(0)) é a etapa mais segura.
    print("Arredondando dados e convertendo para inteiros...")
    
    # Arredonda para 0 casas decimais e depois converte para o tipo 'int'
    df_inteiros = df.round(0).astype(int)
    
    # --- 4. Salvar o novo arquivo ---
    print(f"Salvando arquivo convertido em: {output_file}")
    
    # Salva o novo DataFrame, mantendo o padrão do CSV
    df_inteiros.to_csv(output_file, sep=';', encoding='latin-1', index=False)
    
    print("--- CONCLUÍDO ---")
    print(f"Arquivo '{output_file}' salvo com sucesso.")
    
    # --- 5. Verificação ---
    print("\nVerificação (primeiras 3 linhas do novo arquivo, agora como inteiros):")
    # Usando to_string para garantir uma boa formatação no console
    print(df_inteiros.head(3).to_string(index=False))

except FileNotFoundError:
    print(f"Erro: O arquivo '{input_file}' não foi encontrado.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")
    print("Verifique se o arquivo 'base_4.0_Sem_Ausentes.csv' não está vazio ou corrompido.")