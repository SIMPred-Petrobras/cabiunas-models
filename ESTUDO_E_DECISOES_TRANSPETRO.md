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
