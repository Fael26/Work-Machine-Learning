import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ks_2samp
import time

# ======== 1. Configuração e Leitura =========
print("🔍 Lendo a base de dados...")
input_file = "base_4.0_sem_outliers_.csv"
output_file = "base_4.0_Sem_Ausentes.csv" # <-- NOME DO NOVO ARQUIVO DE SAÍDA

df = pd.read_csv(input_file, sep=";", engine="python", encoding="latin-1")
print(f"✅ Base '{input_file}' carregada com {df.shape[0]} linhas e {df.shape[1]} colunas.\n")

# ======== 2. Verificar valores ausentes =========
total_na = df.isna().sum().sum()
print(f"📊 Existem {total_na} valores ausentes antes da imputação.\n")

# ======== 3. Converter variáveis categóricas para códigos numéricos =========
print("🔄 Convertendo variáveis categóricas para códigos numéricos...")
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category').cat.codes.replace(-1, np.nan)
print("✅ Conversão concluída.\n")

# ======== 4. Imputação iterativa (RandomForest ~ MissForest) =========
print("🚀 Iniciando imputação iterativa (isso pode levar alguns minutos)...")
start_time = time.time()

imputer = IterativeImputer(
    # Use um estimador muito mais simples para testar
    estimator=RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1), # 10 árvores, n_jobs=-1 usa todos os núcleos
    max_iter=2,  # Apenas 2 iterações
    random_state=42,
    verbose=2
)

df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

elapsed = time.time() - start_time
print(f"✅ Imputação concluída em {elapsed:.2f} segundos.\n")

# ======== 5. Métricas de imputação =========
print("📈 Calculando métricas de avaliação...")

# NRMSE (erro normalizado)
numerical_cols = df.select_dtypes(include=[np.number]).columns
errors = []
for col in numerical_cols:
    known_mask = ~df[col].isna()
    if known_mask.sum() > 0:
        mse = mean_squared_error(df.loc[known_mask, col], df_imputed.loc[known_mask, col])
        rmse = np.sqrt(mse)
        norm = np.nanstd(df.loc[known_mask, col])
        if norm != 0:
            errors.append(rmse / norm)
nrmse = np.mean(errors) if errors else np.nan

# R² médio
r2_scores = []
for col in numerical_cols:
    known_mask = ~df[col].isna()
    if known_mask.sum() > 0:
        r2 = r2_score(df.loc[known_mask, col], df_imputed.loc[known_mask, col])
        r2_scores.append(r2)
mean_r2 = np.nanmean(r2_scores)

# PFC aproximado (para colunas categóricas)
categorical_cols = df.select_dtypes(include=['category']).columns
pfc_list = []
for col in categorical_cols:
    known_mask = ~df[col].isna()
    if known_mask.sum() > 0:
        original = df.loc[known_mask, col]
        imputada = df_imputed.loc[known_mask, col].round().astype(int)
        pfc = np.mean(original != imputada)
        pfc_list.append(pfc)
mean_pfc = np.mean(pfc_list) if len(pfc_list) > 0 else np.nan

print("\n=== 📊 MÉTRICAS DE IMPUTAÇÃO ===")
print(f"🔹 NRMSE médio (erro normalizado): {nrmse:.4f}")
print(f"🔹 R² médio (qualidade da imputação): {mean_r2:.4f}")
print(f"🔹 PFC médio (erro categórico aproximado): {mean_pfc:.4f}")
print("==================================\n")

# ======== 6. Salvar o resultado (PARTE ADICIONADA) =========
try:
    print(f"💾 Salvando base de dados imputada em '{output_file}'...")
    
    # Adicionamos encoding='latin-1' para manter a consistência,
    # caso haja caracteres especiais.
    df_imputed.to_csv(output_file, sep=';', index=False, encoding='latin-1')
    
    print(f"✅ Arquivo salvo com sucesso!\n")

except Exception as e:
    print(f"🚨 Erro ao salvar o arquivo: {e}\n")