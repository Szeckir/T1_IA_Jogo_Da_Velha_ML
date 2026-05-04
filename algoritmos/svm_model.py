import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DATASET_DIR, MODELS_DIR

X_train = pd.read_csv(DATASET_DIR / 'X_train.csv')
X_test  = pd.read_csv(DATASET_DIR / 'X_test.csv')
y_train = pd.read_csv(DATASET_DIR / 'y_train.csv').values.ravel()
y_test  = pd.read_csv(DATASET_DIR / 'y_test.csv').values.ravel()

param_grid = {
    'C':      [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma':  ['scale', 'auto'],
}

grid = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
)
grid.fit(X_train, y_train)

print(f"\nMelhores hiperparâmetros: {grid.best_params_}")
print(f"Melhor acurácia (CV): {grid.best_score_:.4f}")

y_pred = grid.predict(X_test)
print(f"\nAcurácia no teste: {accuracy_score(y_test, y_pred):.4f}")
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred,
      target_names=['Tem jogo', 'X venceu', 'O venceu', 'Empate']))
print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))

MODELS_DIR.mkdir(exist_ok=True)
with open(MODELS_DIR / 'svm_model.pkl', 'wb') as f:
    pickle.dump(grid.best_estimator_, f)
print("\nModelo salvo em models/svm_model.pkl")
