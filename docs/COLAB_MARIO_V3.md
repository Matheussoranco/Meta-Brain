# Meta-Brain V3 no Google Colab Pro

## Abrir e executar

1. Abra https://colab.research.google.com/ e envie `notebooks/colab_full_brain_mario_v3.ipynb` pela opção de upload.
2. Em **Ambiente de execução → Alterar tipo de ambiente**, selecione uma **GPU**. Ative **RAM alta** se disponível. Uma GPU com 16 GB é o perfil inicial; o pré-teste verifica a memória real e pode reduzir o lote. Isso não constitui validação prévia em uma T4.
3. Execute as células em ordem. Autorize a montagem do seu Google Drive quando solicitado.
4. Mantenha os parâmetros iniciais na primeira tentativa: `VISION=real`, `SEED=123`, `N_ENVS=4`, `BPTT=16`, `MICRO_STEPS=4`.
5. A primeira sessão baixa os dados e monta o cache. A preparação pode ser demorada e exige mais RAM do que o treinamento. O notebook testa o emulador antes do download e testa gradientes/memória no grafo completo antes do treinamento.

Não instale `requirements.txt` inteiro no Colab: ele contém ferramentas de outras partes do projeto. A célula de instalação instala apenas os pacotes necessários e preserva o PyTorch com CUDA fornecido pelo Colab.

## Onde ficam os arquivos

Com os valores iniciais:

`Meu Drive/Meta-Brain/runs/mario_v3_real_seed123/`

- `latest.pt`: checkpoint de treinamento, atualizado a cada 10 updates e no encerramento normal.
- `best.pt`: checkpoint com a melhor avaliação de validação, priorizando a taxa de bandeiras e depois o progresso médio.
- `progress.json`: último registro periódico.
- `pre_eval.json`: avaliação inicial.
- `best_validation.json`: avaliação usada para selecionar o melhor checkpoint.
- `preflight.json`: contagens do grafo, configuração efetiva, memória medida e verificação de gradiente visual.
- `runtime.json`: versões de pacotes e GPU da sessão.
- `results.json`: histórico e avaliações finais determinística e amostrada, com cada episódio.
- `post_greedy.mp4`, `post_sampled.mp4`: melhor episódio de cada avaliação, em H.264.
- `training.png`: curvas do treinamento.

Os dados originais são preservados em `Meu Drive/Meta-Brain/data` e o cache do grafo em `graph_cache`. Os arquivos brutos são copiados para o disco temporário da sessão para leitura local.

## Continuar em outra sessão

Abra o mesmo notebook, selecione GPU e execute tudo com o mesmo `RUN_NAME`, `VISION` e `SEED`. O programa identifica `latest.pt` e continua do contador salvo, preservando a configuração de lote, arquitetura e BPTT. `TARGET_UPDATES` é o alvo total, não a quantidade adicional por sessão. Aumente esse valor para estender um treino já concluído.

A retomada recupera parâmetros, otimizador, contadores e RNG. O estado exato do emulador não é salvo: episódios em andamento recomeçam. Se a sessão for encerrada abruptamente, no máximo o trabalho após o último checkpoint periódico precisará ser refeito. Não sobrescreva nem apague a pasta para continuar.

`SESSION_HOURS=3.0` limita a parte de treinamento desta sessão. Download, pré-teste e avaliação final acontecem fora desse orçamento. O tempo não depende de contornar limites do Colab. Os recursos do plano Pro variam: https://research.google.com/colaboratory/faq.html.

## Se faltar memória

Em uma execução nova, o pré-teste tenta lotes menores e, se necessário, janelas BPTT menores. Ele preserva todos os neurônios e arestas do grafo carregado. Se nem a estrutura do grafo couber, será necessária uma GPU com mais memória. Se faltar RAM durante a leitura dos dados, selecione um ambiente com RAM alta.

Ao retomar, a configuração salva permanece fixa. Uma GPU menor pode não suportá-la; nesse caso o programa interrompe com uma explicação. Use uma GPU com capacidade suficiente ou crie outro experimento com outro `RUN_NAME`. Não compare condições com orçamentos ou lotes diferentes como se fossem equivalentes.

## O que foi corrigido

A V2 usava projeção visual fixa, dinâmica interna fixa e saída linear. A V3 aprende uma CNN visual compacta, ganhos de entrada/recorrência, vieses e constantes de decaimento por tipo de neurônio, além de uma saída não linear. O gradiente atravessa o conectoma com PPO recorrente e sequências contínuas de BPTT. Os sinais e a topologia das sinapses continuam fixos. A base compacta que mapeia o código visual aos neurônios ópticos também é fixa e não representa retinotopia biológica.

A recorrência começa com ganho 0,5 e usa parâmetros limitados para reduzir o risco de saturação; isso não é garantia matemática de estabilidade de qualquer grafo. Morte e truncamento por tempo/progresso são separados para calcular os retornos corretamente. O vídeo usa amostragem compatível com o frame skip para reduzir RAM.

## Como avaliar

`DEMONSTRATED_SUCCESS` exige que todos os episódios finais determinísticos alcancem a bandeira sem morrer. O JSON também informa a taxa de sucesso com ações amostradas. As partidas finais usam sementes diferentes daquelas usadas para escolher o checkpoint. Como existe um conjunto finito de esperas iniciais, algumas posições iniciais podem coincidir. Isso verifica comportamento na mesma fase; não demonstra generalização para outras fases.

Para controles, execute `VISION=zero` e `VISION=clock`, com as mesmas sementes e o mesmo orçamento. Cada condição tem uma pasta própria e precisa de seu próprio treinamento. O controle `zero` não recebe sinal visual, mas parâmetros intrínsecos aprendidos podem formar uma dinâmica temporal. O controle `clock` recebe um código aleatório fixo por instante, sem informação da tela.

## O que ainda precisa ser medido

Os testes locais verificam a matemática, a chegada do gradiente à visão e à dinâmica, a execução completa do notebook com um grafo pequeno, a retomada de checkpoints e um treino curto com o emulador real. **O treinamento CUDA no conectoma MaleCNS completo e a conclusão consistente da fase ainda dependem da execução no Colab.** Não há checkpoint vencedor fornecido, e as mudanças não garantem vitória.
