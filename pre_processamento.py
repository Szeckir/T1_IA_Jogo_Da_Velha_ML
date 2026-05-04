import pandas as pd

from project_paths import DATASET_DIR, RAW_DATA_FILE

colunas = [
    'canto_superior_esquerdo', 'canto_superior_centro', 'canto_superior_direito',
    'meio_esquerdo', 'meio_centro', 'meio_direito',
    'canto_inferior_esquerdo', 'canto_inferior_centro', 'canto_inferior_direito',
    'class'
]

df = pd.read_csv(RAW_DATA_FILE, names=colunas)

COMBOS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6]
]

def identificar_vitoria(row):
    tabuleiro = list(row.iloc[:9].values)
    for c in COMBOS:
        if tabuleiro[c[0]] == tabuleiro[c[1]] == tabuleiro[c[2]] == 'x':
            return 'X venceu'
        if tabuleiro[c[0]] == tabuleiro[c[1]] == tabuleiro[c[2]] == 'o':
            return 'O venceu'
    if 'b' not in tabuleiro:
        return 'Empate'
    return 'Tem jogo'

df['target'] = df.apply(identificar_vitoria, axis=1)

mapeamento_tabuleiro = {'x': 1, 'o': -1, 'b': 0}
for col in colunas[:-1]:
    df[col] = df[col].map(mapeamento_tabuleiro)

mapeamento_alvo = {
    'X venceu': 1,
    'O venceu': 2,
    'Empate': 3
}
df['target_num'] = df['target'].map(mapeamento_alvo)

print(df['target'].value_counts())

import random

def gerar_amostras_empate(n=200):
    amostras = []
    while len(amostras) < n:
        tabuleiro = ['x'] * 5 + ['o'] * 4
        random.shuffle(tabuleiro)
        row_temp = pd.Series(tabuleiro + ['negative'], index=colunas)
        if identificar_vitoria(row_temp) == 'Empate':
            tab_num = [mapeamento_tabuleiro[c] for c in tabuleiro]
            amostras.append(tab_num + [3])
    return pd.DataFrame(amostras, columns=colunas[:-1] + ['target_num'])

def gerar_amostras_tem_jogo(n=200):
    amostras_tem_jogo = []
    while len(amostras_tem_jogo) < n:
        num_jogadas = random.randint(1, 7)
        tabuleiro = ['b'] * 9
        posicoes = list(range(9))
        random.shuffle(posicoes)
        
        for i in range(num_jogadas):
            tabuleiro[posicoes[i]] = 'x' if i % 2 == 0 else 'o'
            
        row_temp = pd.Series(tabuleiro + ['negative'], index=colunas)
        if identificar_vitoria(row_temp) == 'Tem jogo':
            tab_num = [mapeamento_tabuleiro[c] for c in tabuleiro]
            amostras_tem_jogo.append(tab_num + [0])
            
    return pd.DataFrame(amostras_tem_jogo, columns=colunas[:-1] + ['target_num'])

df_x = df[df['target_num'] == 1].sample(n=200, random_state=42)
df_o = df[df['target_num'] == 2].sample(n=200, random_state=42)
df_empate = gerar_amostras_empate(200)

df_tem_jogo = gerar_amostras_tem_jogo(200)

df_final = pd.concat([df_x, df_o, df_empate, df_tem_jogo], ignore_index=True)

df_final = df_final[colunas[:-1] + ['target_num']]
DATASET_DIR.mkdir(exist_ok=True)
df_final.to_csv(DATASET_DIR / 'dataset_processado.csv', index=False)

print("Dataset final salvo com sucesso!")
print(df_final['target_num'].value_counts())