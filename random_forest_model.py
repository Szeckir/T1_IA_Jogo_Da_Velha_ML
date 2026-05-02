import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregar dados
X_train = pd.read_csv('X_train.csv')
X_test  = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test  = pd.read_csv('y_test.csv').values.ravel()

# Grid de hiperparâmetros
param_grid = {
    'n_estimators':      [50, 100, 200],
    'criterion':         ['gini', 'entropy'],
    'max_depth':         [None, 5, 10],
    'min_samples_split': [2, 5],
}

# GridSearchCV com validação cruzada 5-fold
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
)
grid.fit(X_train, y_train)

print(f"\nMelhores hiperparâmetros: {grid.best_params_}")
print(f"Melhor acurácia (CV): {grid.best_score_:.4f}")

# Avaliação no conjunto de teste
y_pred = grid.predict(X_test)
print(f"\nAcurácia no teste: {accuracy_score(y_test, y_pred):.4f}")
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred,
      target_names=['Tem jogo', 'X venceu', 'O venceu', 'Empate']))
print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))

# Salvar modelo
os.makedirs('models', exist_ok=True)
with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(grid.best_estimator_, f)
print("\nModelo salvo em models/random_forest_model.pkl")
