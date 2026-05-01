import pandas as pd

# Nomes das colunas conforme a posição no tabuleiro 3x3
colunas = [
    'canto_superior_esquerdo', 'canto_superior_centro', 'canto_superior_direito',
    'meio_esquerdo', 'meio_centro', 'meio_direito',
    'canto_inferior_esquerdo', 'canto_inferior_centro', 'canto_inferior_direito',
    'class'
]

df = pd.read_csv('tic-tac-toe.data', names=colunas)

def identificar_vitoria(row):
    # Combinações de vitória (índices das colunas)
    combos = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # Linhas
        [0, 3, 6], [1, 4, 7], [2, 5, 8], # Colunas
        [0, 4, 8], [2, 4, 6]             # Diagonais
    ]
    
    # Se era 'positive', X ganhou
    if row['class'] == 'positive':
        return 'X venceu'
    
    # Se era 'negative', precisamos checar se 'o' ganhou
    tabuleiro = row.iloc[:9].values
    for c in combos:
        if tabuleiro[c[0]] == tabuleiro[c[1]] == tabuleiro[c[2]] == 'o':
            return 'O venceu'
            
    # Se não era positivo e 'o' não ganhou, é empate
    return 'Empate'

# Aplicar a função para criar a nova coluna de classe
df['target'] = df.apply(identificar_vitoria, axis=1)

# Mapear o tabuleiro
mapeamento_tabuleiro = {'x': 1, 'o': -1, 'b': 0}
for col in colunas[:-1]: # Todas exceto a antiga 'class'
    df[col] = df[col].map(mapeamento_tabuleiro)

# Mapear as classes (Labels)
mapeamento_alvo = {
    'X venceu': 1,
    'O venceu': 2,
    'Empate': 3
    # 'Tem jogo' será o 0 mais tarde
}
df['target_num'] = df['target'].map(mapeamento_alvo)

print(df['target'].value_counts())

import random

def gerar_amostras_tem_jogo(n=200):
    amostras_tem_jogo = []
    while len(amostras_tem_jogo) < n:
        # Gera um tabuleiro com número aleatório de jogadas (1 a 7)
        num_jogadas = random.randint(1, 7)
        tabuleiro = ['b'] * 9
        posicoes = list(range(9))
        random.shuffle(posicoes)
        
        for i in range(num_jogadas):
            tabuleiro[posicoes[i]] = 'x' if i % 2 == 0 else 'o'
            
        # Criar um DataFrame temporário para usar a função de checagem
        row_temp = pd.Series(tabuleiro + ['negative'], index=colunas)
        if identificar_vitoria(row_temp) == 'Empate': # Se não deu vitória de ninguém
            # Converter para numérico e adicionar
            tab_num = [mapeamento_tabuleiro[c] for c in tabuleiro]
            amostras_tem_jogo.append(tab_num + [0]) # 0 é a classe 'Tem jogo'
            
    return pd.DataFrame(amostras_tem_jogo, columns=colunas[:-1] + ['target_num'])

# 1. Separar as classes e fazer o shuffle
df_x = df[df['target_num'] == 1].sample(n=200, random_state=42)
df_o = df[df['target_num'] == 2].sample(n=200, random_state=42)
df_empate = df[df['target_num'] == 3] # Pegamos os 16 que existem

# 2. Gerar as amostras de "Tem jogo"
df_tem_jogo = gerar_amostras_tem_jogo(200)

# 3. Concatenar tudo no dataset final
df_final = pd.concat([df_x, df_o, df_empate, df_tem_jogo], ignore_index=True)

# 4. Salvar para usar nos modelos
# Remover as colunas de texto e ficar só com as numéricas
df_final = df_final[colunas[:-1] + ['target_num']]
df_final.to_csv('dataset_processado.csv', index=False)

print("Dataset final salvo com sucesso!")
print(df_final['target_num'].value_counts())