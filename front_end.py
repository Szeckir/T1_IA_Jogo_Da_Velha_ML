import numpy as np
import pandas as pd
import pickle
import os

COLUNAS = [
    'canto_superior_esquerdo', 'canto_superior_centro', 'canto_superior_direito',
    'meio_esquerdo', 'meio_centro', 'meio_direito',
    'canto_inferior_esquerdo', 'canto_inferior_centro', 'canto_inferior_direito',
]

CLASSES = {0: 'Tem jogo', 1: 'X venceu', 2: 'O venceu', 3: 'Empate'}

VITORIAS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6],
]


def estado_real(tabuleiro):
    for combo in VITORIAS:
        if tabuleiro[combo[0]] == tabuleiro[combo[1]] == tabuleiro[combo[2]] == 1:
            return 1
        if tabuleiro[combo[0]] == tabuleiro[combo[1]] == tabuleiro[combo[2]] == -1:
            return 2
    if 0 not in tabuleiro:
        return 3
    return 0


def exibir_tabuleiro(tabuleiro):
    s = {1: 'X', -1: 'O', 0: ' '}
    print()
    for i in range(3):
        r = tabuleiro[i * 3: i * 3 + 3]
        print(f"   {s[r[0]]} | {s[r[1]]} | {s[r[2]]}")
        if i < 2:
            print("   ---------")
    print()


def classificar(modelo, tabuleiro):
    X = pd.DataFrame([tabuleiro], columns=COLUNAS)
    return int(modelo.predict(X)[0])


def escolher_modelo():
    opcoes = {
        '1': ('k-NN',          'models/knn_model.pkl'),
        '2': ('MLP',           'models/mlp_model.pkl'),
        '3': ('Arvore',        'models/arvore_model.pkl'),
        '4': ('Random Forest', 'models/random_forest_model.pkl'),
        '5': ('SVM',           'models/svm_model.pkl'),
    }
    print("\nEscolha o modelo de IA:")
    for k, (nome, path) in opcoes.items():
        status = "disponivel" if os.path.exists(path) else "nao treinado"
        print(f"  {k}. {nome:15s} [{status}]")

    while True:
        escolha = input("\nOpcao: ").strip()
        if escolha in opcoes:
            nome, path = opcoes[escolha]
            if not os.path.exists(path):
                print(f"  Modelo '{nome}' nao encontrado. Rode o script de treino primeiro.")
                continue
            with open(path, 'rb') as f:
                modelo = pickle.load(f)
            print(f"  Modelo selecionado: {nome}")
            return nome, modelo
        print("  Opcao invalida.")


def jogada_humano(tabuleiro):
    print("   Referencia de posicoes:")
    print("   1 | 2 | 3")
    print("   ---------")
    print("   4 | 5 | 6")
    print("   ---------")
    print("   7 | 8 | 9")
    while True:
        try:
            pos = int(input("   Sua jogada (1-9): ")) - 1
            if 0 <= pos <= 8 and tabuleiro[pos] == 0:
                return pos
            print("   Posicao invalida ou ocupada.")
        except ValueError:
            print("   Digite um numero entre 1 e 9.")


def jogada_computador(tabuleiro):
    livres = [i for i, v in enumerate(tabuleiro) if v == 0]
    return int(np.random.choice(livres))


def jogar():
    nome_modelo, modelo = escolher_modelo()

    tabuleiro = [0] * 9
    jogador = 1       # X comeca
    total   = 0
    acertos = 0
    erros   = 0

    print()
    print("=" * 48)
    print("   JOGO DA VELHA  —  IA Classificadora")
    print(f"   Modelo : {nome_modelo}")
    print("   Voce = X  |  Computador = O (aleatorio)")
    print("=" * 48)

    while True:
        # --- Jogada ---
        if jogador == 1:
            print("\nSua vez (X):")
            pos = jogada_humano(tabuleiro)
        else:
            pos = jogada_computador(tabuleiro)
            print(f"\nComputador (O) jogou na posicao {pos + 1}")

        tabuleiro[pos] = jogador
        total += 1

        # --- Estado real e predicao da IA ---
        real = estado_real(tabuleiro)
        pred = classificar(modelo, tabuleiro)

        acertou = (pred == real)
        if acertou:
            acertos += 1
        else:
            erros += 1

        exibir_tabuleiro(tabuleiro)
        print(f"   Estado real    : {CLASSES[real]}")
        print(f"   IA classificou : {CLASSES[pred]}  [{'ACERTO' if acertou else 'ERRO'}]")
        print(f"   Acuracia IA    : {acertos}/{total}  ({acertos / total * 100:.1f}%)")

        # --- Decisao de continuidade ---

        # Caso 1: IA detectou fim incorretamente (falso positivo) — jogo continua
        if pred != 0 and real == 0:
            print()
            print("   >> IA detectou fim de jogo incorretamente.")
            print(f"      IA disse '{CLASSES[pred]}', mas o jogo ainda nao acabou.")
            print("      O jogo CONTINUA.")
            jogador = -jogador
            continue

        # Caso 2: jogo realmente acabou
        if real != 0:
            # Sub-caso: IA nao detectou o fim (falso negativo)
            if pred == 0:
                print()
                print("   >> IA nao detectou o fim de jogo.")
                print(f"      IA disse 'Tem jogo', mas o estado real e '{CLASSES[real]}'.")
                print("      Encerrando o jogo.")

            print()
            print("=" * 48)
            if real == 1:
                print("   RESULTADO: VOCE VENCEU!  (X ganhou)")
            elif real == 2:
                print("   RESULTADO: COMPUTADOR VENCEU!  (O ganhou)")
            else:
                print("   RESULTADO: EMPATE!")
            break

        # Caso 3: tabuleiro cheio mas IA nao detectou (seguranca)
        if 0 not in tabuleiro:
            print()
            print("   >> Tabuleiro cheio — IA ainda dizia 'Tem jogo'. Encerrando.")
            break

        jogador = -jogador

    print()
    print("   --- Resumo da partida ---")
    print(f"   Jogadas  : {total}")
    print(f"   Acertos  : {acertos}")
    print(f"   Erros    : {erros}")
    print(f"   Acuracia : {acertos / total * 100:.1f}%")
    print("=" * 48)

    return input("\nJogar novamente? (s/n): ").strip().lower() == 's'


if __name__ == '__main__':
    print("\nBem-vindo ao Jogo da Velha com IA Classificadora!")
    while True:
        if not jogar():
            print("\nObrigado por jogar!")
            break
