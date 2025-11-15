import pandas as pd

# Nome do arquivo de entrada (VOCÊ DEVE GARANTIR QUE ESTE ARQUIVO EXISTA)
file_name = "base_4.0_Sem balancear.csv"
# Nome do arquivo de saída
output_file_name = "Base_limpa_Incos_e_Redund.csv"

# A variável de caminho deve ser o file_name, a menos que o arquivo esteja em outro local
# Assumimos que o arquivo está no mesmo diretório do script.
caminho_do_arquivo = file_name 

try:
    # 1. Carregar os dados (Corrigindo o erro de codificação)
    # Tenta latin-1, se falhar, tenta iso-8859-1
    try:
        df = pd.read_csv(caminho_do_arquivo, delimiter=';', encoding='latin-1')
    except UnicodeDecodeError:
        df = pd.read_csv(caminho_do_arquivo, delimiter=';', encoding='iso-8859-1')
        
    print(f"Arquivo '{file_name}' carregado com sucesso.")
    print(f"Shape original: {df.shape} (linhas, colunas)")
    
    # 2. Lidar com dados redundantes (linhas duplicadas exatas)
    num_duplicates = df.duplicated().sum()
    print(f"\nVerificando dados redundantes (linhas duplicadas)...")
    print(f"Encontradas {num_duplicates} linhas duplicadas.")
    
    if num_duplicates > 0:
        df_cleaned = df.drop_duplicates(keep='first')
        print(f"Linhas duplicadas removidas.")
        print(f"Shape após remoção de duplicatas: {df_cleaned.shape}")
    else:
        df_cleaned = df.copy() # Usar uma cópia para evitar avisos
        print("Nenhuma linha duplicada para remover.")
    
    # 3. Lidar com dados inconsistentes (valores ausentes/NaN)
    print(f"\nVerificando dados inconsistentes (valores ausentes/NaN)...")
    
    # Substituir valores placeholder (99, por exemplo) por NaN para que dropna funcione corretamente
    # Este é um passo essencial que geralmente é feito antes do dropna
    df_cleaned.replace([99, '99', 99.0, '99.0', ';;', ''], pd.NA, inplace=True)
    
    # Converter para numérico, tratando erros como NaN, antes de verificar
    df_cleaned = df_cleaned.apply(pd.to_numeric, errors='coerce')
    
    rows_with_na = df_cleaned.isnull().any(axis=1).sum()
    print(f"Encontradas {rows_with_na} linhas com pelo menos um valor ausente (NaN).")
    
    if rows_with_na > 0:
        # Nota: Remover todas as linhas que têm QUALQUER valor ausente (dropna) é o método mais agressivo.
        # Em datasets grandes, isso pode remover muitos dados.
        df_cleaned = df_cleaned.dropna() 
        print(f"Linhas com valores ausentes removidas.")
        print(f"Shape final após remoção de valores ausentes: {df_cleaned.shape}")
    else:
        print("Nenhuma linha com valores ausentes para remover.")
        
    # 4. Salvar o resultado
    # Usar o mesmo separador ';' para consistência
    df_cleaned.to_csv(output_file_name, index=False, sep=';')
    
    print(f"\n--- Resumo da Limpeza ---")
    print(f"Shape original: {df.shape}")
    print(f"Shape final: {df_cleaned.shape}")
    print(f"Total de linhas removidas: {df.shape[0] - df_cleaned.shape[0]}")
    print(f"Arquivo limpo salvo com sucesso como: '{output_file_name}'")

except FileNotFoundError:
    print(f"Erro: O arquivo '{file_name}' não foi encontrado. Certifique-se de que ele está no mesmo diretório do script.")
except Exception as e:
    print(f"Ocorreu um erro durante o processamento: {e}")