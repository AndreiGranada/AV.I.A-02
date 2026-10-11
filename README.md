# Neuroevolução no River Raid

**UniSenac Pelotas | Fundamentos de Inteligência Artificial**

O objetivo é evoluir os **pesos de uma rede neural**, sem backpropagation, usando seleção, elitismo e mutação. O jogo escolhido é o River Raid.

## Instalação

Requer Python 3.10+ e a ROM do River Raid fornecida pela instalação do `ale-py`.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
pip install -r requirements.txt
```

## Execução

```bash
python main.py                       # experimento B (20 indivíduos, 30 gerações)
python main.py --experimento todos   # executa A, B e C
python main.py --populacao 4 --geracoes 2  # teste rápido
python main.py --visualizar          # abre uma partida do melhor agente ao final
python main.py --acompanhar          # mostra cada indivíduo durante o treinamento
python main.py --experimento todos --acompanhar  # acompanha A, B e C em tempo real
python main.py --populacao 4 --geracoes 2 --acompanhar  # teste rápido com janela
```

### Google Colab / Jupyter

Instale as dependências na primeira célula:

```python
%pip install "gymnasium[atari]" ale-py numpy matplotlib
```

Em seguida, execute o arquivo na célula:

```python
%run main.py --populacao 4 --geracoes 2
```

Alternativamente, após importar `main`, é possível passar os parâmetros diretamente:

```python
from main import main
main(["--populacao", "4", "--geracoes", "2"])
```

O código ignora argumentos internos do kernel ao ser executado no Jupyter, mas mantém validação estrita no terminal. **Não use `--visualizar` nem `--acompanhar` no Colab**: `render_mode="human"` requer interface gráfica local.

A visualização requer ambiente gráfico. Com `--acompanhar`, o River Raid mostra **cada partida de cada indivíduo durante o treinamento**, e o console identifica geração e indivíduo. Isso é diferente de `--visualizar`, que mostra apenas o campeão no final. É possível combinar as opções. Uma execução completa pode demorar bastante: cada indivíduo joga uma partida por geração, e a renderização ocorre em tempo real.

## Como funciona

- **Ambiente:** `ALE/Riverraid-v5`. O ambiente é criado uma vez por treino (`criar_ambiente`) e reiniciado a cada episódio com `ambiente.reset(seed=seed)`. Criar aloca o jogo; reiniciar começa uma nova partida no mesmo ambiente.
- **Estado:** observação `ram`, um vetor de 128 bytes (`uint8`, valores de 0 a 255). Antes de entrar na rede, `Agente.agir` divide cada valor por 255, levando-o ao intervalo de 0 a 1.
- **Espaço de ações:** o ambiente expõe 18 ações (`env.action_space.n`), mas o jato avança sozinho. A rede usa 6 delas, com os índices do ALE: NOOP (0), FIRE (1), LEFT (4), RIGHT (3), LEFTFIRE (12) e RIGHTFIRE (11). Para listar os nomes, use `env.unwrapped.get_action_meanings()`.
- **Executar uma ação:** `ambiente.step(acao)` recebe o índice do ALE escolhido pela rede e devolve `(estado, reward, terminated, truncated, info)`.
- **Recompensa (`reward`):** pontuação ganha naquele passo.
- **`terminated`:** `True` quando o jogo termina por conta própria (perda de todas as vidas).
- **`truncated`:** `True` quando o episódio é cortado por um limite externo. Neste projeto, o limite é `MAX_PASSOS = 2500`, verificado no laço de `executar_episodio`.
- **Agente:** rede neural `128 → 16 (tanh) → 6 (argmax)`. Pesos e vieses formam o DNA.
- **Recompensa:** valor retornado pelo ambiente em cada passo. **Fitness:** soma das recompensas recebidas no episódio.
- **Duração:** cada episódio termina quando o agente perde ou ao atingir 2500 passos. O ALE permitiria até 27 mil passos; o limite evita que uma partida sem colisão trave o treinamento. O agente aleatório da comparação usa o mesmo limite e as mesmas 6 ações.
- **População:** 20 agentes com pesos inicialmente aleatórios.
- **Seleção:** melhores 50% pelo fitness; cada filho escolhe aleatoriamente um pai dentre eles.
- **Elitismo:** os 2 melhores são copiados sem mutação.
- **Reprodução:** cópia independente do pai, **sem crossover**.
- **Mutação:** cada peso ou viés tem uma probabilidade de alteração; a mudança é um ruído gaussiano com desvio-padrão igual à intensidade configurada.
- **Exploração:** ocorre na inicialização aleatória e nas mutações.
- **Convergência:** pode ser investigada pelas curvas de melhor fitness e fitness médio. Estabilização não garante solução ótima. O projeto não possui critério automático de parada: a convergência é observada nos arquivos `fitness.csv` e `fitness.png`. Uma estagnação típica é o melhor fitness permanecer igual por várias gerações seguidas.

## Experimentos exigidos

| Experimento | Taxa de mutação | Intensidade |
|---|---:|---:|
| A | 1% | 0,05 |
| B | 10% | 0,20 |
| C | 40% | 0,50 |

Use `--experimento todos` para executar os três. A semente (`--seed`, padrão 42) é a mesma para as três configurações, permitindo comparação mais controlada. A avaliação utiliza um episódio por indivíduo; para conclusões estatísticas, recomenda-se repetir com várias sementes.

## Arquivos gerados

Cada experimento gera uma pasta `resultados/A`, `resultados/B` ou `resultados/C` com:

- `fitness.csv`: melhor, média e pior fitness por geração;
- `diagnostico.csv`: por geração e indivíduo, fitness, passos, percentual de cada ação, trocas de ação e dominância da ação mais usada;
- `fitness.png`: gráfico do melhor fitness e fitness médio;
- `melhor_agente.npz`: pesos do melhor indivíduo encontrado;
- `comparacao.csv`: recompensa do agente aleatório e do evoluído.

### Visualização do bônus

O arquivo `04_bonus_visualizacao_humana.py` carrega o melhor agente salvo em `resultados/B/melhor_agente.npz` e abre uma partida com janela:

```bash
python 04_bonus_visualizacao_humana.py
```

Para usar outro experimento, altere o caminho do `np.load` dentro do bloco `if __name__ == "__main__":`. A partida para em 2500 passos, como no treinamento.

### Diagnóstico de comportamento

O console exibe, após cada geração, a distribuição média de ações dos agentes e quantos usam uma única ação em pelo menos **95%** dos passos. Esses agentes são classificados como **políticas repetitivas** para fins de diagnóstico, não necessariamente como políticas ruins.

O `diagnostico.csv` permite investigar cada indivíduo: `noop_pct`, `fire_pct`, `left_pct`, `right_pct`, `leftfire_pct` e `rightfire_pct` mostram a frequência das ações, `trocas_acao` indica mudanças de decisão e `acao_dominante_pct` mede a predominância de uma ação. Esses valores **não alteram o fitness, a seleção ou a mutação**; servem apenas para observação. A média no console é calculada entre indivíduos, não ponderada pelo número de passos.

**Interpretação:** a comparação aleatória é uma baseline simples, não prova que o evoluído seja melhor em geral. Uma população pode estagnar; comparar as três curvas e executar múltiplas sementes oferece evidência mais forte.

## Resultados dos experimentos

Configuração comum: ambiente `ALE/Riverraid-v5` (observação `ram`), população de 20, 30 gerações, elitismo de 2, seleção pela metade superior. Cada indivíduo joga um episódio por geração. A comparação usa a mesma semente do treino.

Foram executadas três sementes (42, 1 e 2). A semente 42 está em `resultados/<experimento>`; as demais em `resultados/seed_<n>/<experimento>`.

| Semente | Experimento | Aleatório | Evoluído | Melhor fitness (geração) | Média final | Pior final |
|---:|---|---:|---:|---:|---:|---:|
| 42 | A (1%, 0,05) | 1560 | 2240 | 2240 (17) | 1677 | 930 |
| 42 | B (10%, 0,20) | 1560 | 2360 | 2360 (24) | 1624 | 920 |
| 42 | C (40%, 0,50) | 1560 | 1980 | 1980 (27) | 1052 | 0 |
| 1 | A (1%, 0,05) | 1540 | 1130 | 1130 (0) | 1130 | 1130 |
| 1 | B (10%, 0,20) | 1540 | 1310 | 1310 (18) | 1139 | 1130 |
| 1 | C (40%, 0,50) | 1540 | 1760 | 1760 (10) | 1098 | 360 |
| 2 | A (1%, 0,05) | 1580 | 1520 | 1520 (5) | 1376 | 920 |
| 2 | B (10%, 0,20) | 1580 | 1710 | 1710 (8) | 1240 | 840 |
| 2 | C (40%, 0,50) | 1580 | 1970 | 1970 (21) | 826 | 0 |

Os valores vêm de `fitness.csv` e `comparacao.csv` de cada pasta.

### Efeito da mutação

- **A (1%, 0,05):** poucos pesos mudam por geração, então os filhos ficam próximos dos pais. O melhor fitness subiu até a geração 17 e depois ficou parado em 2240 até o fim. A média também se manteve estável nas gerações finais. A exploração foi limitada, e o resultado não melhorou depois da geração 17.
- **B (10%, 0,20):** configuração intermediária. Foi o experimento que atingiu o maior fitness (2360), na geração 24. A média subiu de forma mais consistente até perto da geração 26 e depois oscilou entre 1400 e 1700.
- **C (40%, 0,50):** muitos pesos mudam com muita intensidade. A média ficou entre 850 e 1200 na maior parte do treino e o pior indivíduo voltou a 0 em várias gerações (geração 13, 16, 19 e 29). O melhor fitness foi alcançado cedo (geração 6) e subiu pouco depois. Esse comportamento é o que se espera de muita diversidade com risco de destruir boas soluções.

### Exploração

A exploração aparece em dois pontos do algoritmo: na **população inicial aleatória** (`criar_populacao`, pesos sorteados com `np.random.randn`) e nas **mutações** (`mutar`, que sorteia quais pesos mudam e quanto). Sem elas, todos os filhos seriam cópias dos pais e a busca ficaria presa à população inicial.

**Pergunta de análise:** se a mutação for quase zero, os filhos quase não diferem dos pais; a população converge cedo para o que já existe e não encontra soluções novas. Se cada geração alterar quase todos os pesos de forma agressiva, a evolução vira quase uma busca aleatória: boas soluções são destruídas antes de serem refinadas, e a média fica baixa e instável, como observado no experimento C.

### Convergência

Os dados de `fitness.csv` permitem avaliar se o processo estabilizou. Com a semente 42:

- **A:** o melhor fitness ficou em 2240 nas últimas 12 gerações, com média oscilando. Há sinais de estabilização do melhor indivíduo.
- **B:** o melhor fitness subiu até 2360 (geração 24) e não melhorou depois. A média não estabilizou, oscilando até o fim.
- **C:** o melhor fitness estagnou em 1970 entre as gerações 6 e 26 e subiu só para 1980 no fim. A média nunca se estabilizou.

Com as sementes 1 e 2, o padrão é diferente: no A da semente 1, o melhor fitness já estava em 1130 na geração 0 e nunca mudou, o que indica estagnação total desde o início. No C da semente 2, a média caiu para 826 e o pior indivíduo zerou na geração 29, sem recuperação.

Conclusão: a estabilização do melhor fitness aparece em todos os experimentos, mas em alguns casos ela ocorre cedo demais (já na população inicial), indicando que a busca nem sempre encontra melhorias. Não há estabilização da média nos casos com mutação alta. Como o resultado varia entre sementes, a conclusão sobre convergência deve ser tomada com cautela.

### Comparação com o agente aleatório

A comparação não é conclusiva. Com a semente 42, os três experimentos superaram o aleatório (1560): A=2240, B=2360 e C=1980. Porém, com as sementes 1 e 2, a conclusão se inverte em parte: na semente 1, nenhum experimento superou o aleatório (1540); na semente 2, só o C superou (1970 contra 1580), enquanto o A ficou abaixo (1520). Em 2 das 3 sementes, pelo menos um experimento ficou abaixo do agente aleatório.

Portanto, **não é possível afirmar que a evolução supera consistentemente o aleatório** nesta configuração. O resultado depende da semente, o que indica alta variância entre execuções.

## Perguntas de análise

1. **O que é o agente e qual estado ele recebe?** O agente é a rede `Agente`, que recebe o vetor RAM de 128 bytes do ambiente, normalizado para 0–1.
2. **Ação, recompensa e fitness:** a ação é o índice escolhido pela rede; a recompensa é o ponto ganho em cada passo; o fitness é a soma das recompensas do episódio.
3. **Indivíduo, população e geração no código:** indivíduo é uma instância de `Agente`; população é a lista criada por `criar_populacao`; geração é uma iteração do laço em `treinar`.
4. **Seleção:** `selecionar` mantém a metade com maior fitness, e cada filho copia um desses pais aleatoriamente. Escolhemos a metade superior por ser simples e preservar boas soluções sem eliminar toda a diversidade.
5. **Sem elitismo:** o melhor indivíduo poderia ser perdido na mutação de uma geração, e o melhor fitness poderia cair de uma geração para outra.
6. **Taxa e intensidade:** a taxa define quantos pesos mudam; a intensidade define o tamanho da mudança. Juntas, controlam o quanto o filho difere do pai.
7. **Onde ocorre exploração:** na população inicial aleatória e nas mutações (ver seção acima).
8. **Convergência:** o melhor fitness estabilizou nos três experimentos; a evidência é a repetição do mesmo melhor valor por muitas gerações (ver seção acima).
9. **O melhor agente superou o aleatório?** Depende da semente. Na semente 42, sim (2360 contra 1560 no B). Nas sementes 1 e 2, não em todos os casos: na semente 1 nenhum experimento superou o aleatório, e na semente 2 apenas o C superou. O resultado é instável entre sementes.
10. **Qual mudança melhoraria o resultado?** Usar critério de parada por estagnação, repetir os experimentos com várias sementes para medir a variância e aumentar a população ou o número de gerações para dar mais tempo de busca.

## Correspondência com a atividade

| Conceito / etapa | Implementação |
|---|---|
| Parte 1: rede e ambiente | `Agente`, `criar_ambiente` |
| Parte 2: interação e fitness | `executar_episodio`, `avaliar` |
| Parte 2: população e seleção | `criar_populacao`, `selecionar` |
| Parte 2: elitismo e mutação | `mutar`, `nova_geracao` |
| Parte 3: gerações, métricas e gráfico | `treinar`, `salvar_historico` |
| Comparação e visualização | `comparar`, `visualizar` |
