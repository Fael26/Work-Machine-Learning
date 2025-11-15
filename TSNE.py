import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignorar warnings para manter a saída limpa
warnings.filterwarnings('ignore')

# --- 1. Carregamento dos Dados ---
file_test = 'Base sem balanceamento.csv'
print(f"Carregando arquivo: {file_test}...")

try:
    df_test = pd.read_csv(file_test, sep=';', encoding='latin1')
    print("Arquivo carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar arquivo: {e}")
    exit()

# --- 2. Separação de Features (X) e Alvo (y) ---
print("Separando features e alvo...")
y_test = df_test.iloc[:, 0]
X_test_raw = df_test.iloc[:, 1:60]

# --- 3. Alinhamento de Colunas ---
X_test = X_test_raw.iloc[:, 0:58]
print(f"Shape das features (colunas alinhadas): {X_test.shape}")

# --- 4. Pré-processamento (Scaling) ---
print("Aplicando StandardScaler...")
scaler = StandardScaler()
X_test_np = X_test.values
X_test_scaled = scaler.fit_transform(X_test_np) 
print("Dados padronizados.")

# --- 5. Cálculo do t-SNE ---
print("\nIniciando cálculo do t-SNE...")
print("AVISO: Isso pode levar alguns minutos, dependendo do tamanho da base.")

tsne = TSNE(
    n_components=2, 
    perplexity=30,  
    max_iter=1000,    # <<< CORRIGIDO (era n_iter)
    random_state=42
)

# Aplicar o t-SNE nos dados de TESTE
X_tsne = tsne.fit_transform(X_test_scaled)

print("Cálculo do t-SNE concluído.")

# --- 6. Geração e Salvamento do Gráfico ---
print("Gerando e salvando o gráfico...")

# Criar um DataFrame para o plot
df_tsne = pd.DataFrame()
df_tsne["componente_1"] = X_tsne[:, 0]
df_tsne["componente_2"] = X_tsne[:, 1]
df_tsne["classe"] = y_test.values 

# Plotar com Seaborn
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x="componente_1", 
    y="componente_2",
    hue="classe",
    palette=sns.color_palette("hls", 2), 
    data=df_tsne,
    legend="full",
    alpha=0.7 
)

plt.title('Visualização t-SNE dos Dados de Teste (Desbalanceados)')
plt.xlabel('Componente t-SNE 1')
plt.ylabel('Componente t-SNE 2')
plt.grid(True, linestyle='--', alpha=0.5)

# Salvar a figura
nome_arquivo_png = "tsne_base_desbalanceada.png"
plt.savefig(nome_arquivo_png)
plt.close()

print(f"\nGráfico t-SNE salvo com sucesso como: {nome_arquivo_png}")