from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Carregar o dataset que processamos
df = pd.read_csv('dataset_processado.csv')

# 2. Separar Features (tabuleiro) e Target (classe)
X = df.drop('target_num', axis=1)
y = df['target_num']

# 3. Divisão Física (80% Treino, 20% Teste)
# O random_state garante que a divisão seja SEMPRE a mesma para todos os algoritmos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Salvar os conjuntos para garantir que usaremos os mesmos em todos os modelos
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print(f"Treino: {len(X_train)} amostras")
print(f"Teste: {len(X_test)} amostras")