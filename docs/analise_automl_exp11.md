# Análise de Experimentos — AutoML EXP11 (expansão além de temperatura)

Depois de fechar a série EXP5–EXP10 (redução de falso alerta em
`TC382_03_A`/`T5_AVG_A`), investigamos se a mesma metodologia
(AutoML multivariado, janela temporal multiescala 6min/1h/4h/24h)
generaliza para outras categorias do catálogo de 47 tags/3.757 alarmes
(ver `docs/analise_automl_exp10.md` / memória `catalogo_alarmes_turbina`).
Resultado: **negativo para os dois candidatos testados nesta rodada**
(pressão `PDAL_6240302` e o levantamento de vibração) — documentado aqui
com o mesmo cuidado dos achados negativos anteriores (item 3/4 do
EXP7/EXP8).

## Método de triagem (reusável para futuros alvos)

Antes de rodar qualquer experimento novo, dois checks baratos que já
provaram valer a pena:

1. **% de ocorrências do alarme com `RUNNING_A>=0,9`** — se a maioria
   cai fora desse critério, o alarme provavelmente é consequência
   esperada de transiente de partida/parada, não uma falha a prever.
2. **Verificação de alinhamento de timestamp** — antes de confiar no
   check acima, confirmar que os horários de alarme não têm o mesmo
   deslocamento de fuso (-3h) aplicado aos dados de sensor mas não ao
   alarme. Feito comparando `PI_6240319_AL` (alarme de falha de partida,
   que por definição deveria coincidir com uma transição real de
   `RUNNING_A`) contra a transição off→on mais próxima: mediana de 8,7min
   de diferença em 107 alarmes OOS — sem sinal de defasagem de 3h.

## Pressão — `PDAL_6240302`

**Mapeamento alarme→sensor bruto:** tags de alarme de pressão usam
`TIPO_624XXXX`, correspondendo a `954005_624_TIPO_0XXXX` no CSV bruto
(ex: `PDAL_6240302` ↔ `954005_624_PDI_0302`) — confirmado por
correspondência exata de sufixo numérico.

**Por que esse alvo:** dos 6 tags de pressão com mais alarmes, a maioria
(`PALL_6240340` 100% off, `PAL_6240339` 94,9% off, `PAL_6240315` 68,1%
off) é dominada por artefato de desligamento — pressão de óleo/gás cai
a zero quando a turbina para, então o alarme não é uma falha incipiente,
é consequência do desligamento. `PDAL_6240302` foi o mais limpo (92,2%
em operação normal).

**Config:** `configs/calibracao_v4_eq/test_grupo_exp11_pressao_pdi0302.json`.
Grupo multivariado (17 sensores: alvo + 4 pressões correlacionadas
[`PI_0308`, `PI_0339`, `PDI_0338`, `PI_0307`, corr 0,6–0,8] +
`TC382_03_A`/`T5_AVG_A` [corr 0,77, surpreendentemente forte — sinal
de que ambos são proxies de carga] + os 10 canais de vibração), mesma
janela multiescala de temperatura, textura restrita à vibração
(`TEXTURE_SENSORS`), máscara operacional com piso calibrado
(`OFF_TARGET_ABS_THRESHOLD=0,7`, contra o comportamento confirmado de
`PDI_0302` no desligamento de agosto/2025). Grade ampla de AutoML
(dense/ocsvm/iforest × 7 percentis × 6 debounces), sem presumir que o
vencedor de temperatura (ocsvm p99,9/db1) se repetiria.

**Task ClearML:** `36888d0326144f64969cc619afdfb468`.

**Resultado agregado:** `dense` venceu (p99,9/db1) — hit_rate 83,9%
(26/31), FP 1,75%, composite_score 0,946. Números aparentemente bons.

**Investigação caso a caso (a parte que importa):**

- Dos 11 casos "preditivos", só **3 são reais** (8,9h/12,9h/16,2h de
  antecedência). Os outros 8 têm antecedência entre 75–99,7% do teto da
  janela de avaliação (24h) — em todos os 5 casos verificados
  individualmente, `PDI_0302` está em valor completamente normal
  (~1,3–1,4) tanto na "detecção" quanto no alarme. É o mesmo artefato de
  janela já documentado no EXP8 (classificador supervisionado): a
  "antecedência" é coincidência da janela ser larga, não sinal real.
- Dos 15 reativos, **14 são explicados pela máscara operacional**: 11
  porque o alarme dispara durante `off_curto`/`off_longo` (pressão
  diferencial cai durante partida/parada — mecanismo físico esperado,
  fora do escopo da máscara por design), e 3 porque `RUNNING_A` deu um
  salto brusco o suficiente para acionar o critério `TRANSIENT_DIFF_QUANTILE`
  mesmo sem sair da faixa "ligado" (achado novo: `RUNNING_A` parece mais
  ruidoso neste ativo/período do que foi em temperatura). Só 1 reativo é
  genuíno (início rápido sem precursor, mesmo padrão já visto em
  temperatura).

**Conclusão:** de 31 alarmes, só ~3–4 têm sinal genuinamente preditivo.
`PDAL_6240302` é estruturalmente mais ligado a dinâmica de transiente de
partida/parada do que a degradação em regime estável — um mau ajuste
para detecção de anomalia não-supervisionada em estado estacionário.

## Pressão — segunda tentativa, `PAH_6240319`

Candidato alternativo (65% das ocorrências em operação normal, vs. 92,2%
de `PDAL_6240302` mas ainda razoável). Investigação preliminar (sem
chegar a rodar o AutoML) encontrou um problema diferente: `PI_0319`
(raw) tem distribuição **bimodal** — passa a maior parte do tempo numa
faixa apertada (~-0,6 a -0,3) e ocasionalmente salta pra um patamar bem
mais alto (~44,8; 95º percentil já está nesse patamar, mas o 75º ainda
em -0,28). Parece mais uma variável de estado/posição do que uma
pressão contínua. Nos 10 alarmes mais recentes de `PAH_6240319`, só 1
bate com o sinal no patamar alto no instante do alarme — os outros
mostram o sinal no patamar baixo ou caindo, inconsistente com "alarme de
pressão alta". **Não avançamos pra um experimento completo** — precisaria
de investigação própria (possivelmente reformular como classificação de
estado discreto em vez de detecção de anomalia contínua) antes de
justificar o custo de uma task remota.

## Vibração — descartada por falta de dado

Dos 134 alarmes de vibração (10 tags `TV_*`), **131 são `LOLO`**
(vibração baixa-baixa) e ocorrem 100% com `RUNNING_A=0` fixo (não é
transição, é parada completa) — mesmo mecanismo do `UNDER` de
temperatura: a leitura despenca porque a turbina está parada, não por
falha de mancal. Confirmado com `TV_355Y_A` (descrição "Vibração Y
Mancal 5"): em todas as ocorrências, valor bruto ~0,3 (vs. ~14–27 em
operação normal) com `RUNNING_A=0` nas 2h ao redor.

Alarmes de vibração **alta** genuínos (`HI`/`HIHI`, o que indicaria
degradação real): **só 3 no total**, espalhados em 3 tags diferentes
(`TV_354X_A` HIHI×1, `TV_354Y_A` HIHI×1, `TV_355Y_A` HI×1), em mais de 4
anos de dados. Amostra pequena demais pra treinar ou avaliar qualquer
coisa. Vibração como alvo fica descartada não por dificuldade técnica,
mas por não ter dado suficiente.

## Onde isso deixa o escopo

| Categoria | Status |
|---|---|
| Temperatura (`TC382_03_A`/`T5_AVG_A`) | Validado, funcionando (EXP5–EXP10) |
| Pressão (`PDAL_6240302`) | Testado — negativo, transiente-dominado |
| Pressão (`PAH_6240319`) | Investigação preliminar — sinal bimodal, não avançou |
| Vibração | Descartada — só 3 eventos genuínos em 4 anos |
| Falha de instrumento/transmissor (844 alarmes) | Não é alvo de predição de falha física por definição — mas pode ser reformulado como *monitoramento de saúde de instrumentação* (ver próximos passos) |

**Não conclui que só temperatura é viável** — conclui que replicar a
metodologia de temperatura 1:1 em outro alvo não funciona de primeira
sem entender a física/estrutura de cada um antes. Próximos passos em
aberto: (a) reformular falha de instrumento/transmissor como detector de
saúde de sensor (já temos boa parte da caracterização de dropout/Comm
Fail feita como efeito colateral do trabalho em temperatura e pressão);
(b) considerar um detector de nível de grupo/planta (qualquer alarme,
não um tag específico) em vez de replicar o esquema "1 tag = 1 alvo" que
só fez sentido pra temperatura porque `TC382_03_A`/`T5_AVG_A` têm alarme
HI/HIHI genuíno e volume suficiente de eventos limpos.
