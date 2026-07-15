# Estudo e decisões — validação Transpetro (por-sensor × multivariado)

> Documento vivo. Consolida os aprendizados e decisões da fase de validação dos
> modelos CNN1D-AE nos dados Transpetro. Detalhes e tabelas completas nos arquivos
> apontados em `analysis/`. Atualizado em 2026-07-11.

---

## 1. Onde estamos

Rodamos os 12 equipamentos em dois modos:
- **Uni_sensor (univariado):** um autoencoder só do sensor-alvo da falha.
- **Mult_sensor (multivariado):** um autoencoder com o grupo de sensores (alvo + contexto).

Métrica central: **as anomalias de ponto caíram perto da falha documentada?**
(BOM = detecção em ±48h · PARCIAL = só precursor ≤10d · FRACO = nada em ±10d).

Placar (após corrigir o bug de análise, ver §2):
- **Por-sensor:** BOM=2 · PARCIAL=3 · FRACO=7
- **Multivariado:** BOM=4 · PARCIAL=2 · FRACO=6

Detalhe por equipamento: `analysis/COMPARACAO_persensor_vs_multivar.md`.

---

## 2. Erros/achados de metodologia (corrigidos)

1. **Bug na coleta multivariada (corrigido):** a task multivariada sobe o `point_anomalies`
   do grupo **e** de modelos univariados extras (subproduto). A 1ª análise pegava o arquivo
   errado e inflava o multivariado (ex.: B-8801C aparecia BOM; é FRACO). Corrigido para usar
   só o artefato do grupo (`scripts/analyze_failure_detection.py`).
2. **As figuras `analises_falhas/` estavam certas:** o painel de anomalias vazio reflete o
   `point_anomalies` do grupo — que realmente não detectou. (Confirmado pelo usuário.)

---

## 3. Por que tantos FRACOS — diagnóstico

### 3.1 A máscara operacional apagava sinal válido
A máscara zera anomalias fora do estado `on`. Hoje o estado usa `OPERATIONAL_REF_SENSOR`
com `OFF_ABS_THRESHOLD` **fixo=5.0**. Diagnóstico (`analysis/ANALISE_MASCARA_OPERACIONAL.md`):

- Nos equipamentos fracos, o sensor de referência marca 65–89% do tempo como "desligado".
- **Mas** a vibração/temperatura alta (o sinal da falha) ocorre **99–100% com a máquina LIGADA**
  (corrente 75–208 A) — os pontos de sinal **passam** o filtro on/off.
- O que apagava o sinal era a marcação de **transiente** (`TRANSIENT_PADDING_MINUTES=60`) em
  máquinas que ligam/desligam o tempo todo: quase todo ponto operante fica a <60min de uma
  borda e vira `transiente` (também mascarado).

**Validação local (B-8801C, janela da falha):** preservação do sinal de vibração alta como `on`:
| Config | %pontos `on` | sinal preservado |
|---|---|---|
| atual (padding=60, diffq=0.99) | 16.4% | 78.0% |
| padding=0, diffq=0.99 | 18.9% | 93.3% |
| **padding=0, diffq=0.999** | 19.8% | **99.2%** |

### 3.2 A data da falha ≈ data de parada
Em vários equipamentos a máquina **para no dia seguinte à falha** (corrente/pressão → 0):
a "falha" registrada é a retirada de operação. O precursor está **antes**, durante operação —
reforça a necessidade de olhar a janela pré-falha e não só ±48h.

### 3.3 Limite físico (doc do time PdM — DOC/)
Sem espectro/FFT (cadência ~60s), falhas de **banda estreita** (rolamento BPFO/BPFI, desbalanceamento,
desalinhamento) são **indetectáveis precocemente** — só se pega elevação da **energia global** de
vibração, e tarde. Explica B-4703 e B-0302C serem intrinsecamente difíceis.

---

## 4. Qual sensor usar para "ligado/desligado" (resposta à pergunta)

O time PdM (DOC/) **não usa corrente** — usa **pressão de descarga** (bomba parada não gera
pressão; ex. B-8802B filtra `Pressão Descarga > 12.9`). Validamos candidatos por quanto cada um
**preserva o sinal da falha** (`analysis/ESCOLHA_SENSOR_STATUS.md`):

| Equip | Sensor de status escolhido | Limiar op. | Preserva sinal |
|---|---|---|---|
| B-24001B | PRESSÃO NA DESCARGA DA BOMBA | 27.5 | 99.9% |
| B-3403C | Corrente | 57.5 | 100% |
| B-402E | Corrente | 177.7 | 100% |
| B-4703.24001B | **Vazão** | 254.0 | 99.8% |
| B-5401A | **Indicador de Velocidade** | 1435 | 100% |
| B-5501B | **Pressão Descarga** | 18.4 | 99.6% |
| B-6511502A | **PRESSÃO DESCARGA** | 41.1 | 100% |
| B-8801C | Corrente | 52.0 | 100% |
| **B-0302C** | (todos preservam só 12%) | — | **descartar máscara** |

- **Regra geral:** preferir **pressão de descarga / vazão / velocidade** (sensores de processo,
  0 = parada) sobre corrente. Corrente serve com limiar operacional adequado (≠ 5.0 fixo).
- **B-0302C:** sensores de motor zerados (documentado) — nenhum sensor de status funciona; a
  vibração alta ocorre com a máquina "parada". Equipamento intrinsecamente fraco.
- **B-402E / B-5401A:** o time PdM **descartou** por instrumentação insuficiente — tratar como
  baixa prioridade.

---

## 5. Decisões para a próxima rodada (multivariado)

1. **Máscara:** manter ativa, mas por equipamento:
   - `OPERATIONAL_REF_SENSOR` = sensor escolhido na §4 (`analysis/status_sensor_choice.json`).
   - `OFF_ABS_THRESHOLD` = limiar operacional estimado (≠ 5.0 fixo).
   - `TRANSIENT_PADDING_MINUTES=0` e `TRANSIENT_DIFF_QUANTILE=0.999` (para de apagar operação).
   - **B-0302C:** `ENABLE_OPERATIONAL_MASK=false` (máscara não é confiável ali).
2. **Organização:** resultados em `resultados/Uni_sensor/<eq>` e `resultados/Mult_sensor/<eq>`.
3. **Produtos por experimento:** plot geral, plot zoom na falha, curva de loss, `MODEL_CARD.md`.
4. **Novo plot (pedido):** sinal bruto + erro de reconstrução (MAE) + anomalias com **dois eixos Y**
   — o mais visual possível para ligar bruto↔MAE↔anomalia.
5. **Execução:** remoto (ClearML).

### Melhorias metodológicas herdadas do time PdM (backlog)
- Medir **FP em held-out** (não in-sample — o reportado é otimista).
- No resample, preservar **picos** (`max`/`rms`), não `.last()`.
- Avaliar **CUSUM**/debounce como política de alarme para rampas.
- Considerar **threshold por regime** de carga.

---

## 6. Índice de artefatos desta análise

| Arquivo | Conteúdo |
|---|---|
| `analysis/COMPARACAO_persensor_vs_multivar.md` | placar detecção por-sensor × multivariado |
| `analysis/ANALISE_MASCARA_OPERACIONAL.md` | diagnóstico da máscara (%off, comportamento na falha) |
| `analysis/ESCOLHA_SENSOR_STATUS.md` | comparação de sensores de status por equipamento |
| `analysis/status_sensor_choice.json` | sensor + limiar operacional escolhidos (por equip) |
| `scripts/analyze_failure_detection.py` | recomputa o placar de detecção |
| `scripts/analyze_operational_mask.py` | recomputa o diagnóstico da máscara |
| `scripts/choose_status_sensor.py` | recomputa a escolha do sensor de status |
| `scripts/collect_persensor_results.py` | baixa produtos das tasks + reconstrói cards |

---

## 7. Redução de falsos positivos — assinatura de episódio (2026-07-12)

### 7.1 Diagnóstico

Mesmo com a máscara v3 corrigida, vários equipamentos com "sucesso relativo" (anomalia
perto da falha real) ainda têm muitos falsos positivos ao longo da série. Investigamos se
duração/magnitude do MAE ao redor da falha real formam um padrão explorável.

**Achado 1 — corte único de duração/magnitude não funciona.** Agregado (11 equip válidos,
excluindo B-5401A por MAE degenerada — threshold no chão numérico, equipamento já
descartado pelo time PdM), episódios `near` da falha são mais longos/fortes que `far`
(duração mediana 57min vs 20min uni, 97,5min vs 76min mult). Mas **por equipamento o
padrão é inconsistente** — em B-6511502A, B-4064A, B-3403C, B-8802B o episódio `near` é
igual ou mais fraco que vários `far`. Um corte global apagaria detecções reais nesses
casos. Ver `analysis/EPISODE_SIGNATURE_STUDY.md`.

**Achado 2 — o balde "far" não é ruído homogêneo.** Inspecionando visualmente os maiores
episódios `far` de B-6511502A e B-4064A, achamos **3 mecanismos físicos distintos**
misturados, cada um exigindo tratamento diferente:

| Mecanismo | O que é | Tratamento correto |
|---|---|---|
| `glitch_sensor` | Salto instantâneo implausível (poucos pontos), tipicamente junto de `off_curto`/`transiente` | Filtro de qualidade de dado (despike), não supressão por duração |
| `mudanca_regime` | Sinal se estabiliza num patamar NOVO (não volta à baseline) — modelo não reconhece o regime | Threshold por regime de carga (item já no backlog da auditoria do time PdM) — **maior bucket, 715 episódios far** |
| `precursor_parada` | Sinal sustentado, sem salto/mudança de patamar, seguido de parada em poucas horas | **Não suprimir** — candidato a evento real não documentado |
| `sustentado_sem_causa` | MAE elevado por tempo considerável sem causa visível no bruto | **Não suprimir** — revisão manual/domínio |
| `transiente_curto` | Sobra: curto e fraco, sem nenhum padrão acima | Único bucket seguro de suprimir (~1% de sobreposição com `near`) |

Triagem automática (validada contra os 5 casos inspecionados visualmente — 100% de
concordância) em `analysis/EPISODE_TRIAGE.md` + `episodes_{uni,mult}_classified.csv`.
Proporção near/far por classe (pooled, 1913 episódios): `transiente_curto`=1,0% near,
`precursor_parada`=4,5%, `glitch_sensor`=4,9%, `mudanca_regime`=5,8%,
`sustentado_sem_causa`=6,3% — os últimos dois são os **mais arriscados de suprimir**.

### 7.2 Decisão implementada

Implementado na pipeline (`scoring.py::suppress_short_transient_episodes`, ver
`PIPELINE_REVISADA.md` v3.2): suprime **só** `transiente_curto` (episódios <30min sem
assinatura de glitch, mudança de regime, ou parada em seguida). Os demais mecanismos
**não são tocados** — ficam como próximos passos:

1. ~~Suprimir `transiente_curto`~~ — ✅ feito (v3.2).
2. Threshold por regime para `mudanca_regime` (maior volume, 715 episódios) — pendente.
3. Filtro de glitch no dado bruto para `glitch_sensor` (487 episódios) — pendente.
4. Gerar lista de `precursor_parada`/`sustentado_sem_causa` para revisão da manutenção —
   pendente. Achado bônus: se algum desses episódios for confirmado como evento real, ganhamos
   mais exemplos de falha para calibrar (quebra a limitação de "1 falha conhecida por equipamento").

### 7.3 Resultado do experimento_2 (validação em produção, 2026-07-14)

Rodou-se a pipeline v3.2 completa nos 12 equipamentos (uni + mult) e comparou-se contra
o experimento_1 (`analysis/COMPARACAO_experimento_1_vs_experimento_2.md`,
`scripts/compare_experiments.py`). De 24 comparações: **12 reduziram ruído sem mudar a
classe de detecção**, 9 tiveram leve aumento de ruído (classe igual), 2 ficaram
idênticas, e **1 mudou de classe** (B-4064A uni: PARCIAL→FRACO) — investigado a fundo e
**confirmado como variância de retreinamento**, não efeito da supressão (os episódios
suprimidos não têm relação com a janela da falha). Nenhuma detecção real foi perdida
pela supressão em si. Ver `EXPERIMENTOS.md` para o detalhe completo.

**Lição metodológica:** cada experimento re-treina do zero, então comparações futuras
devem se apoiar no padrão agregado (maioria estável/melhor) em vez de julgar por uma
única célula que mudou — retreinamento sozinho já introduz variação mesmo sem mudar
código.

### 7.4 Novo achado: máscara sem persistência apaga precursores reais (2026-07-14)

Inspecionando gráficos perto da falha (pedido do usuário: "persistência do MAE acima do
limiar" como critério), achamos por acaso um problema mais urgente que os do §7.1-7.2: em
alguns equipamentos, o **sensor de status oscila violentamente** perto da falha (liga/
transiente/desliga a cada 1-2 min — plausivelmente a própria falha causando instabilidade
elétrica). A máscara decide pelo estado no **minuto exato do fim de cada janela de 60min**,
não pela maioria da janela — com o sensor piscando tanto, praticamente toda sequência cai
num instante "não-ligado" por acaso, apagando uma detecção que deveria existir mesmo com
o MAE do sinal-alvo sustentado e claramente elevado.

**Varredura nos 24 modelos** (`scripts/scan_mask_erased_precursors.py`,
`analysis/MASCARA_APAGA_PRECURSOR_SCAN.md`): bins de 2h dentro de −10d/+2d da falha com
≥30% das sequências cruzando o threshold, ≤5% sobrevivendo à máscara, e ≥20% do tempo
com sensor "presente" (on ou transiente, não genuinamente desligado). **6 equipamentos
distintos afetados** (7 instâncias uni/mult): B-3403C, B-4064A (uni e mult), B-24001B,
B-402E, B-8801C — vários deles hoje classificados FRACO/PARCIAL justamente por causa
disso, não por falta de sinal real.

**Causa raiz:** `mask_anomaly_seq_by_operational_state` (`scoring.py`) usa
`state.reindex(seq_end_idx).eq("on")` — um único ponto no tempo, sem persistência.

### 7.5 Três correções testadas por simulação — só uma é segura, e não é global

Testamos 3 formas de "dar persistência" à máscara, todas por simulação (reprocessando
`mae_seq` já salvo, sem re-treinar) nos 6 casos afetados:

1. **Janela majoritária** (`scripts/simulate_majority_mask.py`): trocar "estado no ponto
   final" por "≥X% da janela em on+transiente" (X=30/50/70%). **Recupera detecção em
   todos os 6 casos**, mas **explode o ruído geral** em B-3403C (7-9x), B-4064A uni
   (25-28x) e B-402E (35-44x) — equipamentos cujo sensor de referência é instável o ano
   inteiro, não só perto da falha.
2. **Suavizar o sensor de referência** (`scripts/simulate_smoothed_mask.py`, mediana
   móvel de 5/15/30min antes de classificar estado): **não recupera nada** em B-3403C,
   B-4064A uni e B-402E — nesses casos o MAE elevado coincide com uma **rampa real de
   desligamento** (corrente caindo de ~100A pra 0), não com ruído de medição; suavizar
   não muda a classificação porque a queda é genuína.
3. **Aceitar "transiente" como válido** (`scripts/simulate_allow_transiente.py`, mesma
   checagem em 1 ponto, só troca `eq("on")` por `isin(["on","transiente"])`): recupera
   detecção nos 6 casos, mas **também explode ruído** nos mesmos 4 equipamentos
   (B-3403C, B-4064A uni, B-402E, B-24001B uni) — e fica **seguro e efetivo** em
   **B-4064A (mult) e B-8801C (mult)** (ruído sobe <5%, detecção recuperada).

**Conclusão: não existe correção global segura.** Os equipamentos que explodem são os
mesmos em todo teste — não é a forma da correção que falha, é que o sensor de referência
deles é cronicamente instável (liga/desliga o tempo todo, o ano inteiro), então qualquer
afrouxamento da máscara libera ruído histórico, não só o precursor da falha. Isso **bate
com o que o time PdM já sabia** (`DOC/`): B-4064A é o caso "multi-regime" que eles tratam
à parte (resíduo de carga), e B-402E foi **descartado** por eles por instrumentação
insuficiente — a varredura redescobriu, por outro ângulo, os mesmos equipamentos
problemáticos.

### 7.6 Decisão implementada

`MASK_ALLOW_TRANSIENTE` (novo campo em `PipelineConfig`, default `False`): aceita
`transiente` como válido na máscara. Implementado em `scoring.py::mask_anomaly_seq_by_
operational_state` (parâmetro `allow_transiente`), passado pelas duas chamadas em
`pipeline.py`. **Ativado só onde a simulação validou como seguro:**
`B-4064A_mult_v3.json` e `B-8801C_mult_v3.json` (`OUTPUT_ROOT` já apontando para
`resultados/experimento_3_mask_transiente/`). Os outros 22 configs ficam com o
comportamento atual (default `False`), incluindo os 4 equipamentos onde a correção
comprovadamente piora (B-3403C, B-4064A uni, B-402E, B-24001B uni) — ficam como
limitação conhecida, não como bug a perseguir com mais uma tentativa de regra universal.

### 7.7 Resultado real do experimento_3 (2 tasks, ClearML) + bugfix encontrado

Rodou-se de fato (não só simulação) `B-4064A_mult_v3.json` e `B-8801C_mult_v3.json` com
`MASK_ALLOW_TRANSIENTE=True`. Comparado ao experimento_2
(`analysis/COMPARACAO_experimento_2_vs_experimento_3.md`):

| Equip | Classe (exp2→exp3) | rate/dia (exp2→exp3) |
|---|---|---|
| B-4064A (mult) | BOM → BOM | 27,31 → 22,54 (**-17%**) |
| B-8801C (mult) | BOM → BOM | 11,30 → 10,60 (**-6%**) |

**Detecção mantida (BOM), ruído caiu nos dois** — resultado positivo, sem regressão.

**Bug encontrado ao investigar por que nenhum ponto anômalo aparecia com
`operational_state=="transiente"`** (esperava alguns, já que a máscara deveria aceitá-los):
`pipeline.py` tem um **segundo filtro**, no nível de PONTO (após `map_seq_to_point_
anomalies`), que zerava `is_anom_point` para qualquer estado diferente de `"on"` —
**sem checar `MASK_ALLOW_TRANSIENTE`**. Ou seja, a config valia só na máscara de
sequência; o filtro de ponto reforçava a regra antiga por cima, agindo como uma segunda
trava. Corrigido (mesma condição `ok_states` nos dois pontos de `pipeline.py`).

Re-simulando com a correção completa nos dados já treinados do experimento_3
(`scripts/simulate_allow_transiente.py`): ganho adicional modesto — B-4064A +2 pontos
(+1,4% rate), B-8801C +11 pontos (+10% rate) na janela de avaliação. **Não re-rodado no
ClearML ainda** — os resultados do experimento_3 acima refletem o comportamento
pré-bugfix (mais conservador), mas já positivos; o re-treino com a correção completa é
opcional, pendente de decisão.
