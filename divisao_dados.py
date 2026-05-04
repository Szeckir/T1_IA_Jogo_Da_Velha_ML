from sklearn.model_selection import train_test_split
import pandas as pd

from project_paths import DATASET_DIR

df = pd.read_csv(DATASET_DIR / 'dataset_processado.csv')

X = df.drop('target_num', axis=1)
y = df['target_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

DATASET_DIR.mkdir(exist_ok=True)
X_train.to_csv(DATASET_DIR / 'X_train.csv', index=False)
X_test.to_csv(DATASET_DIR / 'X_test.csv', index=False)
y_train.to_csv(DATASET_DIR / 'y_train.csv', index=False)
y_test.to_csv(DATASET_DIR / 'y_test.csv', index=False)

print(f"Treino: {len(X_train)} amostras")
print(f"Teste: {len(X_test)} amostras")