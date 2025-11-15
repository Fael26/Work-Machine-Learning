import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # NOVO: Importar a biblioteca de gráficos

# --- Configuração ---
# Arquivo de entrada
file_name = "Base 4.0 com novas colunas tentando melhorar.csv"
# Novo arquivo de saída
output_file = "base_4.0_sem_outliers_.csv"
# Colunas para verificar outliers
columns_to_check = ['Idade', 'Peso - Final', 'Altura - Final']
# Limite do Z-score
threshold = 2.5

print(f"--- Iniciando remoção de outliers (Z-score={threshold}) ---")
print(f"Colunas a serem analisadas: {', '.join(columns_to_check)}")

try:
    # --- CORREÇÃO AQUI ---
    # Adicionado encoding='latin-1' para ler o arquivo corretamente
    df = pd.read_csv(file_name, sep=';', na_values='NA', encoding='latin-1')
    # ---------------------

    print(f"\nArquivo '{file_name}' carregado com sucesso.")
    print(f"Shape original (linhas, colunas): {df.shape}")

    # Copiar o DataFrame para evitar SettingWithCopyWarning
    df_temp = df.copy()

    # Inicializa uma máscara para "manter" todas as linhas
    final_keep_mask = pd.Series(True, index=df_temp.index)

    # Lista para guardar os nomes das colunas z-score temporárias
    zscore_cols = []

    # 1. Iterar sobre cada coluna para calcular o Z-score
    for col in columns_to_check:
        if col in df_temp.columns and pd.api.types.is_numeric_dtype(df_temp[col]):
            print(f"\nProcessando coluna: '{col}'")
            
            # Calcular Média e Desvio Padrão (ignorando NaNs)
            mean_val = df_temp[col].mean()
            std_val = df_temp[col].std()
            
            print(f"  Média: {mean_val:.2f}")
            print(f"  Desvio Padrão: {std_val:.2f}")

            if std_val > 0:
                # 2. Calcular o Z-score
                zscore_col_name = f'zscore_temp_{col}'
                zscore_cols.append(zscore_col_name)
                df_temp[zscore_col_name] = (df_temp[col] - mean_val) / std_val
                
                # 3. Criar a máscara para esta coluna
                # Mantém linhas DENTRO do limite OU linhas que são NaN
                col_keep_mask = (np.abs(df_temp[zscore_col_name]) <= threshold) | (df_temp[zscore_col_name].isna())
                
                # 4. Atualizar a máscara "final"
                # Uma linha só é mantida se for válida em TODAS as colunas
                final_keep_mask = final_keep_mask & col_keep_mask
            else:
                print(f"  Aviso: Desvio padrão da coluna '{col}' é 0.")
        else:
            print(f"\nErro: Coluna '{col}' não encontrada ou não é numérica.")

    # 5. Aplicar o filtro combinado
    # df_temp[final_keep_mask] seleciona apenas as linhas onde final_keep_mask é True
    df_sem_outliers = df_temp[final_keep_mask].copy() # .copy() para segurança

    # 6. Relatório
    original_count = df.shape[0] # NOVO: Guardar contagem original
    clean_count = df_sem_outliers.shape[0] # NOVO: Guardar contagem limpa
    removed_count = original_count - clean_count
    
    print(f"\n--- Resultados da Filtragem ---")
    print(f"Shape original: {df.shape}")
    print(f"Shape novo: {df_sem_outliers.shape}")
    print(f"Total de {removed_count} linhas (outliers) removidas.")
    
    # --- NOVO: Bloco de Geração do Gráfico ---
    print("\nGerando gráfico de impacto...")
    
    # Dados para o gráfico
    labels = ['Treino Original\n(Pré-limpeza)', 'Treino Limpo\n(Pós-limpeza)', 'Outliers Removidos']
    values = [original_count, clean_count, removed_count]
    
    # Cores similares às do exemplo (Verde, Azul, Laranja/Vermelho)
    colors = ['#4CAF50', '#2196F3', '#F44336']
    
    # Criar a figura e as barras
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors)
    
    # Adicionar título e labels
    plt.title(f'Impacto da Remoção de Outliers (Z-Score={threshold}) no Conjunto de Treino')
    plt.ylabel('Número de Instâncias')
    
    # Adicionar os números no topo das barras
    plt.bar_label(bars, fmt='%d', padding=3)
    
    # Ajustar o limite do eixo Y para dar espaço ao label da barra mais alta
    if original_count > 0:
        plt.ylim(0, original_count * 1.15) 
    
    plt.tight_layout() # Ajusta o layout para evitar cortes
    plt.show() # Exibe o gráfico
    print("Gráfico exibido.")
    # --- Fim do Bloco de Gráfico ---

    # 7. Limpar colunas temporárias de Z-score antes de salvar
    df_sem_outliers = df_sem_outliers.drop(columns=zscore_cols)

except FileNotFoundError:
    print(f"Erro: O arquivo '{file_name}' não foi encontrado.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")