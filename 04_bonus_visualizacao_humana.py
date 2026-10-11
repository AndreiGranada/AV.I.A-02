"""
BÔNUS — VISUALIZAÇÃO HUMANA DO MELHOR AGENTE

Este arquivo foi pensado para ser integrado ao programa principal.
Ele assume que:
- a classe Agente já existe;
- ENV_ID = "ALE/Riverraid-v5";
- gymnasium, ale_py e numpy já foram importados.

Objetivo:
Executar visualmente, em uma janela do Atari, o melhor agente
encontrado pelo processo evolutivo.

No main.py, a mesma função já está disponível com:
    python main.py --visualizar
"""

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

ENV_ID = "ALE/Riverraid-v5"
MAX_PASSOS = 2500


def visualizar_agente(agente):
    """
    Executa uma partida com visualização humana.

    IMPORTANTE:
    render_mode="human" cria uma janela mostrando o jogo.
    O agente continua recebendo o estado pela RAM.
    O episódio para em MAX_PASSOS, o mesmo limite do treinamento.
    """

    env = gym.make(
        ENV_ID,
        obs_type="ram",
        render_mode="human"
    )

    estado, info = env.reset()

    terminou = False
    recompensa_total = 0
    passos = 0

    while not terminou and passos < MAX_PASSOS:

        # O MESMO agente treinado escolhe a ação.
        acao = agente.agir(estado)

        estado, recompensa, terminated, truncated, info = env.step(acao)

        recompensa_total += recompensa
        terminou = terminated or truncated
        passos += 1

    env.close()

    print(f"Recompensa obtida na visualização: {recompensa_total}")


# ---------------------------------------------------------
# COMO USAR NO PROGRAMA PRINCIPAL
# ---------------------------------------------------------
#
# Depois de encontrar e armazenar o melhor indivíduo:
#
# visualizar_agente(melhor_agente)
#
# Executar este arquivo abre a partida do agente salvo em resultados/B.
# ---------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    from main import Agente

    pesos = np.load("resultados/B/melhor_agente.npz")
    agente = Agente()
    agente.W1 = pesos["W1"]
    agente.b1 = pesos["b1"]
    agente.W2 = pesos["W2"]
    agente.b2 = pesos["b2"]
    visualizar_agente(agente)
