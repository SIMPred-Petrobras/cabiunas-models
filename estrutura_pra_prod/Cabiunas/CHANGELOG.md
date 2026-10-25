# Histórico de versões do detector do TC-33003A

## v2.1 (em preparação)

### Conteúdo da release candidata (branch `release/v2.1`)

- **Base:** `feat/pacote-consolidado` (e17bf27), que já inclui a leitura do export de 30 s do PI (`fix/pacote-grade-30s`) e
  o monitor de drift (`fix/pacote-monitor-drift`).
- **Especificação do refratário** (abaixo). Só comentário e documentação: a lógica é a da v2.
- **Fora, à espera de decisão:**
  - a gestão de alarme R1 (`feat/gestao-de-alarme-r1`; validada fora da amostra, −79% de horas de alarme ativo, não aprovada para o pacote);
  - a coluna `atencao_em_bloqueio`;
  - a carga como prioridade na tela;
  - o rótulo "nascido na partida";
  - as 3 tags que mais pesam em cada alarme;
  - a saída da PI_0340, que é duplicata.

### Validação de ponta a ponta (24/10/2026)

O teste usou o CSV histórico de 01/01/2025 a 30/04/2026, os 16 bundles e o registro de 9 trips. A release foi comparada
com o pacote congelado da sombra (commit 1f26e34), cada um rodando em cópia isolada.

- **Inferência** (`tc33003a_exemplo.py --dias 400 --json`): código 0 nos dois. A série (`tc33003a_inferencia.csv`,
  288.001 instantes) é **idêntica byte a byte**. O `estado.json` também é idêntico, exceto `gerado_em`. Resultado:
  16 episódios, 34.179 instantes de alarme.
- **Retreino** (`constroi_bundle.py --mes 2026-05`): o bundle gerado é **idêntico byte a byte** nos dois. A saída tem
  código 4 ("revisar antes de publicar"), como esperado: o dado real termina em 21/04, 9 dias antes do corte.
- **Trava de validade:** com bundle vencido, a inferência recusa rodar (código 2) e mostra a mensagem certa.
- **Observação de robustez, não corrigida:** o `constroi_bundle.py` importa `cabiunas_inference` contando com a pasta do
  script no caminho de importação. Ele falha se for chamado com `python -I` ou importado de outro lugar. O
  `tc33003a_exemplo.py` acrescenta a pasta de forma explícita. Antes de agendar o retreino, chame o script direto
  (`python3 constroi_bundle.py ...`) ou acrescente a mesma linha.

### Especificação do refratário: episódios de atenção também armam o bloqueio de 72 h

**Mudança de especificação, sem mudança de lógica.** Um *episódio de atenção* é um voto do detector que não passa na
duração mínima (120 min, ou 60 min se o episódio for forte) e por isso não vira alarme. Ele **também arma o refratário
de 72 h**: durante o bloqueio, um episódio novo é descartado (salvo o furo "forte e velho").

O código da v2 já fazia isso. Em `scripts/cabiunas_inference.py` (`prever`), o laço do refratário arma o bloqueio
**antes** do filtro de duração mínima. Mas o comentário dizia "depois de um alarme", e a pesquisa que escolheu o ponto
de operação descrevia o refratário como "reportar menos vezes o que já viu" (`scripts/pdm_fisico/ponto_final.py`). A
ordem veio da composição das duas regras naquele estudo, não de uma decisão registrada. **A partir da v2.1, a ordem é a
especificação.** O comentário e a docstring foram corrigidos para descrever o comportamento real.

**Por que manter, e não corrigir o código.** O efeito foi medido no EXP 64 (frente de pesquisa "nova visão", branch
`exp/modos-de-falha`, ClearML `2acadf64`). A comparação é com um contrafactual em que o refratário só arma depois de um
alarme emitido; mesmo período (fev/2025 a abr/2026), mesmos 8 trips, janela de 48 h.

| base | com o bloqueio da atenção (v2) | só bloqueio após alarme emitido | diferença |
|---|---|---|---|
| atual | 7/8, FP 9 (0,807/mês), 1.457 h | 7/8, FP 15 (1,345/mês), 1.825 h | **−6 FP**, mesma detecção |
| G168 | 8/8, FP 14 | 8/8, FP 20 | **−6 FP**, mesma detecção |
| B800 | 8/8, FP 12 | 8/8, FP 17 | **−5 FP**, mesma detecção |
| retreino n = 19.000 | 5/8, FP 11 | 6/8, FP 15 | **−4 FP**, −1 detecção (11/04/2025) |

- **Não é acaso.** O mesmo número de bloqueios de 72 h armados no fim de outros episódios sorteados tira em média só
  1,1 a 1,4 FP e custa 0,3 a 0,6 detecção. O efeito real vence esse placebo nas 4 bases (p = 0,000).
- **Quanto silencia.** No período inteiro houve de 6 a 12 bloqueios armados por episódio de atenção, de 23 a 30 no total
  por base. Eles silenciaram de 438 a 584 h de voto, sempre fora das janelas de trip nas bases atual, G168 e B800 (0 h
  dentro). O episódio que arma dura de 35 a 52 min na mediana e no máximo 88 min.
- **O que o operador vê.** O voto silenciado continua na tela como "atenção" (`severity`); só deixa de virar alarme
  (`is_anomaly`).

**Limitação conhecida.** Na base com retreino n = 19.000, um episódio de atenção de 16 min em 10/04/2025 (14:16, hora
local) armou o bloqueio e silenciou o precursor do trip de 11/04/2025: são 4 h dentro da janela, e a base fica com 5/8 em
vez de 6/8. É o mesmo caminho pelo qual o 11/04 cai quando o blackout pós-partida passa de 6 h (H7, frente `sota`) e
quando a faixa física dos mancais muda (EXP 63). Qualquer mudança que **parta ou funda votos perto de uma parada** arma
ou desarma esses bloqueios. Por isso, toda mudança futura de dado, de máscara ou de voto deve ser avaliada sabendo disso.

**Decisão.** Registrada em 22/10/2026. O teste formal da alternativa (refratário só após alarme emitido) não foi rodado:
o resultado já estava medido no EXP 64 e não mudaria a decisão.
