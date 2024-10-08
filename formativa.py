import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importando a base de dados
df = pd.read_csv('dados_produtos.csv')

# Codificando a coluna categórica 'product_name'
df = pd.get_dummies(df, columns=['product_name'], drop_first=True)

# Separar variáveis independentes (X) e dependentes (y)
X = df.drop('rating', axis=1)  # Variáveis independentes
y = df['rating']                # Variável dependente

# Dividir o conjunto de dados em conjunto de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o scaler
sc = StandardScaler()

# Ajustar e transformar o conjunto de treino
X_treino = sc.fit_transform(X_treino)

# Apenas transformar o conjunto de teste
X_teste = sc.transform(X_teste)

# Verificando as dimensões dos conjuntos
print("Conjunto de treino:", X_treino.shape, y_treino.shape)
print("Conjunto de teste:", X_teste.shape, y_teste.shape)
