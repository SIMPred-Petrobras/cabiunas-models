# Planejamento EXP7 — Reduzir casos "reativos" e "sem detecção"

**Data:** 2026-08-15
**Contexto:** continuação direta do EXP6 (`docs/analise_automl_exp6.md`), depois
de gerar o relatório completo (`task_plots_exp6_vibracao/relatorio_exp6_vibracao.pdf`)
com o candidato validado `TC382_03_A`/`T5_AVG_A` + vibração + `iforest`
(p99.9/debounce=6): 65% de antecedência real (26/40), mediana de 12,4h, FP de
0,06%.

Este documento registra o diagnóstico dos dois pontos fracos do EXP6, a
pesquisa do estado da arte feita antes de decidir o próximo passo, e o plano
priorizado combinado com o usuário — para retomar sem perder o contexto.

---

## 1. Diagnóstico dos dois pontos fracos

Do total de 40 alarmes OOS avaliados no EXP6:

| Categoria | Qtd | O que os painéis mostram |
|---|---|---|
| Antecedência real (preditivo) | 26 (65%) | Já validado — ver relatório EXP6 |
| **Reativo** (detectado só depois/junto do alarme) | **4 (10%)** | Modelo pega o evento, mas sem antecedência |
| **Sem detecção** (nada em ±24h) | **10 (25%)** | Nenhuma excursão visível na própria série de temperatura, na janela de ±30h/6h plotada |

**Hipótese de causa raiz comum:** o pipeline atual usa só features de
curtíssimo prazo — `roll_med`/`roll_std`/`delta_1` numa janela única de 12
amostras (6 minutos) — e um threshold estático (percentil global do erro de
reconstrução no treino). Se o precursor de um evento se desenvolve numa
escala de tempo mais lenta (horas), tanto a janela curta quanto o threshold
fixo podem estar cegos para ele: a tendência lenta nunca gera um `delta_1`
grande o suficiente, e o valor em si pode nunca cruzar o percentil global
mesmo já tendo saído da "linha de base" recente daquele trecho específico.

---

## 2. Pesquisa do estado da arte (2026-08-15)

Buscamos o que a literatura/prática atual de manutenção preditiva recomenda
para exatamente este par de problemas (falso negativo por precursor lento +
falta de antecedência). Resumo por tema:

### 2.1 Vibração — RMS é o padrão, mas tem teto sem forma de onda bruta
RMS velocity (10–1000 Hz) é o parâmetro-padrão da indústria para
monitoramento geral de máquinas rotativas, correlacionando diretamente com
fadiga/desgaste. Para pegar os estágios **mais precoces** de degradação de
mancal (3–6 meses antes), a prática usa análise de envelope/alta frequência
(>20 kHz) sobre a **forma de onda bruta** — algo que não temos: nosso `TV_*`
é uma leitura escalar a cada 30s, provavelmente já um RMS agregado pelo
transmissor. **Isso limita o teto de antecedência que dá pra esperar só com
os dados atuais** — precisaria confirmar com quem administra os sensores se
existe forma de onda bruta disponível em algum outro ponto de coleta.

Fontes: [RMS Vibration — ISO 10816](https://vibromera.eu/glossary/rms/),
[Review of Feature Extraction Methods in Vibration-Based Condition Monitoring](https://www.mdpi.com/2075-1702/5/4/21)

### 2.2 Features de "textura" do sinal além de média/desvio
Kurtosis, crest factor, skewness e impulse factor são features de
time-domain padrão em condition monitoring — capturam mudança de *forma* da
distribuição do sinal (ex: picos esporádicos ficando mais frequentes), não
só nível ou variância. Comumente combinadas com múltiplas escalas de janela.

### 2.3 Detecção de mudança de regime (change-point) como complemento ao threshold global
Abordagens tipo PELT/CUSUM detectam quando um trecho da série já é
estatisticamente diferente da sua **linha de base local recente**, em vez de
exigir que o valor cruze um limiar global fixo. É um mecanismo diferente do
threshold por percentil que já usamos — pega drift gradual que nunca fica
"extremo" o suficiente pra cruzar o percentil histórico, mas já mudou de
regime.

Fonte: [MEDEP: Maintenance Event Detection via PELT](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9031099/)

### 2.4 Reformulação supervisionada/fracamente supervisionada (o achado de maior potencial)
Como já temos os timestamps reais dos 40 alarmes (rótulos), a literatura de
2025 em *weak supervision* para manutenção preditiva recomenda treinar um
classificador supervisionado direto — "essa janela vai gerar alarme nas
próximas N horas?" — usando o histórico de eventos como rótulo, em vez de só
rankear por erro de reconstrução não supervisionado e esperar que ele
correlacione com falha. Otimiza **diretamente** para antecedência, em vez de
otimizar reconstrução como proxy. É a mudança de paradigma mais promissora,
mas também a mais trabalhosa e com maior risco de overfitting dado que só
temos 40 eventos rotulados.

Fonte: [Weak Supervision: A Survey on Predictive Maintenance (2025)](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.70022)

### 2.5 RUL / análise de sobrevivência — direção de médio-longo prazo
Modelos de tempo-até-falha (RUL, censura, sobrevivência) são o estado da
arte para prognóstico contínuo, mas tipicamente exigem um volume de eventos
rotulados bem maior que os 40 que temos hoje para treinar de forma
confiável. Não é o próximo passo — é aspiracional, revisitar quando/se
acumularmos mais histórico de alarmes rotulados.

Fontes: [RULSurv — survival-based RUL em rolamentos](https://arxiv.org/pdf/2405.01614),
[Survival Models for Predictive Maintenance — review 2025](https://www.mdpi.com/1424-8220/26/6/1915)

---

## 3. Plano priorizado (combinado com o usuário, ainda não executado)

Ordem de execução sugerida — do mais barato/baixo-risco ao mais custoso:

1. **[Próximo passo]** Features derivadas multi-escala: estender
   `_build_derived_features`/`build_group_dataframe` para gerar
   `roll_med`/`roll_std`/`delta` em **múltiplas janelas** (ex: 6min atual +
   1h + 4h + 24h), não só uma. Reaproveita a infraestrutura já existente do
   EXP6 (`ENABLE_DERIVED_FEATURES`), baixo risco de regressão.
2. Adicionar features de textura do sinal (kurtosis, crest factor,
   skewness) em janela móvel, complementando média/desvio/delta.
3. Detecção de mudança de regime (PELT/CUSUM) como sinal adicional,
   complementar ao threshold por percentil já existente — não substituir,
   somar como mais um critério/feature.
4. Reformulação supervisionada (classificador "vai alarmar em N horas?"
   usando os 40 eventos como rótulo) — avaliar depois dos itens 1–3, com
   cuidado redobrado de validação temporal (nested CV / walk-forward) dado
   o tamanho pequeno da amostra rotulada.
5. RUL/sobrevivência — não priorizado agora; revisitar se o volume de
   alarmes rotulados crescer substancialmente.

**Pendência paralela — RESOLVIDA (2026-08-15):** `TV_*` é leitura
bruta/instantânea de sonda de proximidade (eddy-current), não RMS agregado.
Evidência: os alarmes desses tags têm condição `LOLO` disparando em valores
**negativos** (-12 a -19 nos setpoints reais) — impossível para um RMS (que
nunca é negativo), mas típico de falha/perda de referência do sinal de gap
de uma sonda de proximidade. Valores em operação normal ficam na faixa
0–40 (provável deslocamento em mícrons), com alarmes HI/HIHI em 51–69.

Consequência prática:
- Dá pra calcular estatísticas de textura (kurtosis, crest factor, RMS
  próprio) em janela móvel de forma legítima — é leitura quase-bruta, não
  um valor já pré-processado (bom para o item 2 do plano).
- Análise de envelope/espectral de alta frequência (a técnica que pega
  falha de mancal com 3–6 meses de antecedência) **não é viável** com essa
  fonte — só temos snapshot a cada 30s, não forma de onda contínua em kHz.
  O teto realista de antecedência continua sendo da ordem de horas, como já
  observado no EXP6, não meses.

## 4. Onde paramos

Nenhum código foi alterado ainda para este plano — este documento é só o
registro da pesquisa e da decisão de prioridade, para continuar direto pelo
item 1 na próxima sessão sem precisar re-derivar o diagnóstico.

## 5. Fechamento (2026-08-16)

Os 5 itens foram implementados e documentados:
- Itens 1--3 (features multi-escala, textura, mudança de regime): ver
  `docs/analise_automl_exp7.md`.
- Itens 4--5 (reformulação supervisionada, sobrevivência exploratória): ver
  `docs/analise_automl_exp8.md`.

Candidato final: `ocsvm` (p99.9/debounce=1) com features multi-escala +
textura (item 2) — os itens 3 e 4 não trouxeram ganho adicional sobre isso.
