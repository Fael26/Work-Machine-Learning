import pandas as pd
import numpy as np

# --- Configuração ---
# ATENÇÃO: Verifique se o nome deste arquivo bate com o que
# foi salvo pelo script anterior.
# Pode ser:
# 'base_preenchida_arredondada.csv'
# 'base_preenchida_e_limpa_knn.csv'
# 'base_preenchida_knn.csv'
arquivo_para_verificar = 'base_4.0_Sem_Ausentes.csv'

separador_csv = ';'
encoding_csv = 'latin1'
# --------------------

print(f"Iniciando verificação do arquivo: {arquivo_para_verificar}")
print(f"Procurando por 'NA' ou 'NaN' (dados ausentes) em TODAS as colunas...")

try:
    # Carregar o dataset
    df = pd.read_csv(arquivo_para_verificar, sep=separador_csv, encoding=encoding_csv)
    print(f"\nArquivo carregado com sucesso.")
    print(f"Ele possui {len(df)} linhas e {len(df.columns)} colunas.")

    # --- A Verificação Principal ---
    
    # 1. Calcular o total de nulos por coluna
    nulos_por_coluna = df.isnull().sum()

    # 2. Calcular o total geral de nulos
    total_nulos = nulos_por_coluna.sum()

    if total_nulos == 0:
        print("\n----------------------------------")
        print("--- RESULTADO: SUCESSO! ---")
        print("Nenhum dado ausente (NA ou NaN) foi encontrado em todo o arquivo.")
        print("A base de dados está completa.")
        print("----------------------------------")
    else:
        print("\n----------------------------------")
        print("--- RESULTADO: ATENÇÃO! ---")
        print(f"Foram encontrados {total_nulos} dados ausentes no total.")
        print("\nColunas que AINDA contêm dados ausentes e a contagem:")
        
        # Filtra e mostra apenas as colunas que têm nulos
        colunas_com_nulos = nulos_por_coluna[nulos_por_coluna > 0]
        print(colunas_com_nulos)
        
        print("\n(Isso é normal se essas colunas não estavam na lista (AM, AO, etc.)")
        print("que você pediu para preencher com o KNN.)")
        print("----------------------------------")

except FileNotFoundError:
    print(f"\n!!! ERRO: Arquivo não encontrado !!!")
    print(f"O arquivo '{arquivo_para_verificar}' não foi encontrado.")
    print("Verifique o nome do arquivo na linha 11 deste script e tente novamente.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")