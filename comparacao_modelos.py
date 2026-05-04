import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

from project_paths import DATASET_DIR, MODELS_DIR, RESULTS_DIR, ROOT_DIR

X_test = pd.read_csv(DATASET_DIR / 'X_test.csv')
y_test = pd.read_csv(DATASET_DIR / 'y_test.csv').values.ravel()

CLASS_NAMES = ['Tem jogo', 'X venceu', 'O venceu', 'Empate']

MODELOS = {
    'k-NN':          MODELS_DIR / 'knn_model.pkl',
    'MLP':           MODELS_DIR / 'mlp_model.pkl',
    'Árvore':        MODELS_DIR / 'arvore_model.pkl',
    'Random Forest': MODELS_DIR / 'random_forest_model.pkl',
    'SVM':           MODELS_DIR / 'svm_model.pkl',
}

resultados = []
predicoes = {}

for nome, caminho in MODELOS.items():
    if not caminho.exists():
        print(f"[AVISO] Modelo não encontrado: {caminho.relative_to(ROOT_DIR)} — rode o script correspondente primeiro.")
        continue
    with open(caminho, 'rb') as f:
        modelo = pickle.load(f)
    y_pred = modelo.predict(X_test)
    predicoes[nome] = y_pred
    resultados.append({
        'Algoritmo': nome,
        'Acurácia':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall':    recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1-Score':  f1_score(y_test, y_pred, average='weighted', zero_division=0),
    })

if not resultados:
    print("Nenhum modelo encontrado. Rode os scripts de treino primeiro.")
    exit()

df = pd.DataFrame(resultados).set_index('Algoritmo')
print("\n=== Comparação dos Modelos (conjunto de teste) ===")
print(df.to_string(float_format='%.4f'))

melhor = df['F1-Score'].idxmax()
print(f"\nMelhor modelo por F1-Score: {melhor} ({df.loc[melhor, 'F1-Score']:.4f})")

RESULTS_DIR.mkdir(exist_ok=True)
df.to_csv(RESULTS_DIR / 'comparacao_modelos.csv', float_format='%.4f')

metricas = ['Acurácia', 'Precision', 'Recall', 'F1-Score']
n_modelos = len(df)
x = np.arange(len(metricas))
width = 0.8 / n_modelos

fig, ax = plt.subplots(figsize=(12, 6))
for i, (nome, row) in enumerate(df.iterrows()):
    offset = (i - n_modelos / 2) * width + width / 2
    bars = ax.bar(x + offset, [row[m] for m in metricas], width, label=nome)

ax.set_ylim(0, 1.15)
ax.set_ylabel('Score')
ax.set_title('Comparação de Algoritmos — Jogo da Velha')
ax.set_xticks(x)
ax.set_xticklabels(metricas)
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()

n = len(predicoes)
fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
if n == 1:
    axes = [axes]

for ax, (nome, y_pred) in zip(axes, predicoes.items()):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(nome)

plt.suptitle('Matrizes de Confusão — Conjunto de Teste', fontsize=14, y=1.02)
plt.tight_layout()

print("\nArquivos salvos em results/:")
print("  comparacao_modelos.csv")
print("  comparacao_metricas.png")
print("  matrizes_confusao.png")
