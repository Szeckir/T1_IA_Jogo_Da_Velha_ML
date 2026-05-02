# T1 — Jogo da Velha com ML

## Dependências

```bash
pip3 install numpy pandas scikit-learn matplotlib
```

## Estrutura do projeto

```
T1_IA_Jogo_Da_Velha_ML/
│
├── tic-tac-toe.data          # Dataset original (UCI)
├── tic-tac-toe.names         # Descrição do dataset original
│
├── pre_processamento.py      # Etapa 1 — gera dataset_processado.csv
├── divisao_dados.py          # Etapa 2 — gera X/y train/test
│
├── dataset_processado.csv    # Dataset balanceado (616 amostras, 4 classes)
├── X_train.csv               # Features de treino  (492 amostras)
├── X_test.csv                # Features de teste   (124 amostras)
├── y_train.csv               # Labels de treino
├── y_test.csv                # Labels de teste
│
├── knn_model.py              # Etapa 3a — treina k-NN
├── mlp_model.py              # Etapa 3b — treina MLP
├── arvore_model.py           # Etapa 3c — treina Árvore de Decisão
├── random_forest_model.py    # Etapa 3d — treina Random Forest
├── svm_model.py              # Etapa 3e — treina SVM
│
├── comparacao_modelos.py     # Etapa 4 — compara os 5 modelos
├── front_end.py              # Etapa 5 — jogo interativo com IA
│
├── models/                   # Modelos treinados (.pkl)
└── results/                  # Gráficos e tabela de comparação
```

---

## Como executar (passo a passo)

### Etapa 1 — Pré-processamento do dataset

Lê o dataset original da UCI, cria as 4 classes e gera um dataset balanceado.

```bash
python3 pre_processamento.py
```

**Saída:** `dataset_processado.csv`

### Etapa 2 — Divisão dos dados

Divide o dataset em treino (80%) e teste (20%) com estratificação e `random_state=42`.

```bash
python3 divisao_dados.py
```

**Saída:** `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

### Etapa 3 — Treinar os modelos

Cada script realiza GridSearchCV com validação cruzada 5-fold e salva o melhor modelo.

```bash
python3 knn_model.py
python3 mlp_model.py
python3 arvore_model.py
python3 random_forest_model.py
python3 svm_model.py
```

**Saída:** arquivos `.pkl` na pasta `models/`

#### Hiperparâmetros testados por algoritmo

| Algoritmo     | Hiperparâmetros                                                                                          | Combinações |
| ------------- | -------------------------------------------------------------------------------------------------------- | ----------- |
| k-NN          | k=[1,3,5,7,9,11,13,15], weights=[uniform,distance], metric=[euclidean,manhattan,minkowski]               | 48          |
| MLP           | hidden_layer_sizes=[(50),(100),(50,50),(100,50)], activation=[relu,tanh], alpha=[0.0001,0.001,0.01]      | 24          |
| Árvore        | criterion=[gini,entropy], max_depth=[None,5,10,20], min_samples_split=[2,5,10], min_samples_leaf=[1,2,4] | 72          |
| Random Forest | n_estimators=[50,100,200], criterion=[gini,entropy], max_depth=[None,5,10], min_samples_split=[2,5]      | 36          |
| SVM           | C=[0.1,1,10,100], kernel=[linear,rbf,poly], gamma=[scale,auto]                                           | 24          |

Cada script imprime no terminal:

- Melhores hiperparâmetros encontrados
- Acurácia na validação cruzada
- Acurácia, precision, recall e F1-score no conjunto de teste
- Matriz de confusão

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
3. python3 knn_model.py
   python3 mlp_model.py
   python3 arvore_model.py
   python3 random_forest_model.py
   python3 svm_model.py
4. python3 comparacao_modelos.py
5. python3 front_end.py
```

---

## Classes do dataset

| Código | Classe   | Descrição                                     |
| ------ | -------- | --------------------------------------------- |
| 0      | Tem jogo | Jogo em andamento                             |
| 1      | X venceu | Jogador X completou uma linha/coluna/diagonal |
| 2      | O venceu | Jogador O completou uma linha/coluna/diagonal |
| 3      | Empate   | Tabuleiro cheio sem vencedor                  |

## Representação do tabuleiro

Cada posição do tabuleiro é mapeada para um valor numérico:

| Símbolo   | Valor |
| --------- | ----- |
| X         | 1     |
| O         | -1    |
| vazio (b) | 0     |
