# T1 — Jogo da Velha com ML

Sistema de IA para classificar estados do tabuleiro do jogo da velha em 4 classes:
**Tem jogo**, **X venceu**, **O venceu**, **Empate**.

---

## Dependências

```bash
pip3 install numpy pandas scikit-learn matplotlib
```

---

## Estrutura do projeto

```
T1_IA_Jogo_Da_Velha_ML/
│
├── project_paths.py          # Centraliza os caminhos de pastas
│
├── tic-tac-toe.data          # Dataset original (UCI) — necessário para rodar
├── tic-tac-toe.names         # Descrição do dataset original
│
├── pre_processamento.py      # Etapa 1 — gera dataset_processado.csv
├── divisao_dados.py          # Etapa 2 — gera X/y train/test
│
├── algoritmos/
│   ├── knn_model.py          # Etapa 3a — treina k-NN
│   ├── mlp_model.py          # Etapa 3b — treina MLP
│   ├── arvore_model.py       # Etapa 3c — treina Árvore de Decisão
│   ├── random_forest_model.py# Etapa 3d — treina Random Forest
│   └── svm_model.py          # Etapa 3e — treina SVM
│
├── comparacao_modelos.py     # Etapa 4 — compara os 5 modelos
├── front_end.py              # Etapa 5 — jogo interativo com IA
│
├── dataset/                  # Gerado automaticamente (CSVs de treino/teste)
├── models/                   # Gerado automaticamente (modelos .pkl)
└── results/                  # Gerado automaticamente (gráficos e tabela)
```

---

## Como executar (passo a passo)

### Etapa 1 — Pré-processamento do dataset

Lê o dataset original da UCI, cria as 4 classes e gera um dataset balanceado.

```bash
python3 pre_processamento.py
```

**Saída:** `dataset/dataset_processado.csv`

> **Nota sobre a classe Empate:** O jogo da velha possui poucas posições de empate
> possíveis. Por isso essa classe tem menos amostras naturais — o script gera amostras
> sintéticas válidas para compensar. As outras classes têm 200 amostras cada.

---

### Etapa 2 — Divisão dos dados

Divide o dataset em treino (80%) e teste (20%) com estratificação e `random_state=42`.

```bash
python3 divisao_dados.py
```

**Saída:** `dataset/X_train.csv`, `dataset/X_test.csv`, `dataset/y_train.csv`, `dataset/y_test.csv`

---

### Etapa 3 — Treinar os modelos

Cada script realiza GridSearchCV com validação cruzada 5-fold e salva o melhor modelo.

```bash
python3 algoritmos/knn_model.py
python3 algoritmos/mlp_model.py
python3 algoritmos/arvore_model.py
python3 algoritmos/random_forest_model.py
python3 algoritmos/svm_model.py
```

Ou tudo de uma vez:

```bash
python3 algoritmos/knn_model.py && python3 algoritmos/mlp_model.py && python3 algoritmos/arvore_model.py && python3 algoritmos/random_forest_model.py && python3 algoritmos/svm_model.py
```

**Saída:** arquivos `.pkl` na pasta `models/`

#### Hiperparâmetros testados por algoritmo

| Algoritmo | Hiperparâmetros | Combinações |
|---|---|---|
| k-NN | k=[1,3,5,7,9,11,13,15], weights=[uniform,distance], metric=[euclidean,manhattan,minkowski] | 48 |
| MLP | hidden_layer_sizes=[(50),(100),(50,50),(100,50)], activation=[relu,tanh], alpha=[0.0001,0.001,0.01] | 24 |
| Árvore | criterion=[gini,entropy], max_depth=[None,5,10,20], min_samples_split=[2,5,10], min_samples_leaf=[1,2,4] | 72 |
| Random Forest | n_estimators=[50,100,200], criterion=[gini,entropy], max_depth=[None,5,10], min_samples_split=[2,5] | 36 |
| SVM | C=[0.1,1,10,100], kernel=[linear,rbf,poly], gamma=[scale,auto] | 24 |

Cada script imprime no terminal:
- Melhores hiperparâmetros encontrados
- Acurácia na validação cruzada
- Acurácia, precision, recall e F1-score no conjunto de teste
- Matriz de confusão

---

### Etapa 4 — Comparação dos modelos

Carrega os 5 modelos treinados e gera tabela e gráficos comparativos.

```bash
python3 comparacao_modelos.py
```

**Saída em `results/`:**
- `comparacao_modelos.csv` — tabela com accuracy, precision, recall e F1 dos 5 modelos
- `comparacao_metricas.png` — gráfico de barras comparando as 4 métricas
- `matrizes_confusao.png` — matrizes de confusão dos 5 modelos lado a lado

> Execute somente após treinar todos os modelos na Etapa 3.

---

### Etapa 5 — Front End (jogo interativo)

Jogo da velha no terminal: **humano (X)** vs. **computador aleatório (O)**.

A cada jogada, o modelo de IA escolhido classifica o estado do tabuleiro e o sistema:
- Exibe o estado real e a predição da IA
- Indica se a IA acertou ou errou
- Mostra a acurácia acumulada da IA durante a partida
- Continua o jogo quando a IA detecta fim incorretamente (falso positivo)
- Encerra o jogo quando a IA não detecta o fim (falso negativo)

```bash
python3 front_end.py
```

> Execute somente após treinar pelo menos um modelo na Etapa 3.

---

## Ordem de execução recomendada

```
1. python3 pre_processamento.py
2. python3 divisao_dados.py
3. python3 algoritmos/knn_model.py
   python3 algoritmos/mlp_model.py
   python3 algoritmos/arvore_model.py
   python3 algoritmos/random_forest_model.py
   python3 algoritmos/svm_model.py
4. python3 comparacao_modelos.py
5. python3 front_end.py
```

---

## Classes do dataset

| Código | Classe | Descrição |
|---|---|---|
| 0 | Tem jogo | Jogo em andamento |
| 1 | X venceu | Jogador X completou uma linha/coluna/diagonal |
| 2 | O venceu | Jogador O completou uma linha/coluna/diagonal |
| 3 | Empate | Tabuleiro cheio sem vencedor |

## Representação do tabuleiro

| Símbolo | Valor |
|---|---|
| X | 1 |
| O | -1 |
| vazio (b) | 0 |
