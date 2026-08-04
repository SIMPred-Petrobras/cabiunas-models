# Relatório de sessão — 04/08/2026

**Tema:** ampliação da base de sensores (pressão/mancal), diagnóstico do backcast 2024 do
TC382_03_A e descoberta de proxy de carga estacionário.
**Contexto operacional:** servidor ClearML fora do ar durante toda a sessão — tudo abaixo
foi feito **offline** (cache local de MAE + CSVs), sem retreinar nenhum modelo.

---

## 1. Resumo executivo

1. **As curvas de pressão já estavam no nosso disco** (`sensores_brutos_2025_2026_30s.csv`,
   18 instrumentos `954005_624_*`, cobertura ~100% em 2025–abr/2026) — nunca modeladas
   porque os configs listam só os 17 sensores de temperatura/vibração.
2. **O rótulo dos sensores novos quase todo reprova**: das 27 tags candidatas, só 4 passam
   no portão de validação (excursão real na curva antes do alarme). As duas maiores fontes
   de volume (`PAL_6240315`, 195 onsets; `PI_6240319_AL`, 218) disparam **sem a curva sair
   da faixa normal** — treinar contra elas envenenaria a métrica.
3. **Candidato novo real: `TI_0305` (temperatura do mancal radial LNA)** — excursão em
   100% dos onsets (+10 a 11 MAD), modo de falha inédito no portfólio (mancal, não exaustão).
4. **`RUNNING_A` foi auditado e está correto** — a hipótese "o mask descarta rótulo bom"
   foi testada e refutada (T5 como árbitro físico: os 52 onsets descartados têm turbina
   fria, T5 mediano 30,9°C).
5. **Backcast 2024 do TC03: causa confirmada = regime, não ausência de sinal.** Threshold
   por banda de regime (offline, sem retreino) quase dobra o recall de 2024
   (21,4% → 39,3% raw, FA menor), mas custa o OOS 2025 (100% → 88,2%) — **não promovido**;
   serve de evidência para a Frente B (treino incluindo 2024).
6. **Descoberta da sessão: `PDI_0317` (ΔP do gás combustível) é proxy de carga
   estacionário do TC03** — corr 0,982, resíduo ±8°C em 16 meses OOS, explica o salto de
   regime de 2026 (+52°C). Experimento de contexto pronto para rodar quando o servidor voltar.

---

## 2. Sensores de pressão / mancal (Frente A)

### 2.1 Mapa tag de alarme → coluna de curva

`scripts/map_alarm_tags_to_columns.py` → `configs/calibracao_v12_pressao/tag_column_map.csv`.
**44/47 tags casadas.** Regra: família de instrumento (P/PD/T) + número de loop — casar só
pelo número **erra em 4 tags** (0315/0305/0307 se repetem entre famílias, ex.:
`PAL_6240315`→`PI_0315` mas `TI_6240315`→`TI_0315`).

### 2.2 Portão de validação do rótulo CFN

`scripts/validate_pressure_labels.py` → `eval_pressure_out/label_validation.csv` + figuras.
As tags novas só disparam CFN/OK e **não têm linha nos limites** (`limites_sensores_*.csv`),
então validamos empiricamente: excursão pré-onset (MAD), raridade do valor de onset na
operação normal, duração onset→OK, chatter.

| Veredito | n | Significado |
|---|---:|---|
| mascarado_RUNNING_A | 13 | alarme com equipamento desligado (descarte **correto**, ver 2.3) |
| **APROVADA** | **4** | `TAH_6240305`, `TAHH_6240305` (→TI_0305), `PDAL_6240302`, `PDIT_6240305_AL` |
| poucos_eventos | 4 | <5 onsets |
| sem_excursao | 3 | `PAL_6240315` (p11 do normal no onset), `PAH_6240319`, `PI_6240319_AL` (falha de transmissor) |
| chatter | 3 | >90% dos eventos duram <5 min (`TI_6240315/17`, `PAL_6240339`) |

⚠️ `PDAL_6240302` passou na estatística mas o **gráfico reprova em parte**: 31–33% dos
onsets são transiente de partida (ΔP saindo de 0 para 1,4 quando a máquina liga).
Panorama e zooms em `eval_pressure_out/pressao_*.png`.

### 2.3 Auditoria do RUNNING_A (hipótese refutada)

`scripts/diagnose_running_proxy.py`. Suspeitávamos que o flag descartava rótulo bom.
**Não descarta**: 52/52 onsets "mascarados" têm T5 mediano de 30,9°C (turbina fria e
parada), zero com T5>500°C. OFF falso soma 8 blocos / 0,1h em 16 meses. Não há proxy de
operação a consertar; os alarmes descartados são de equipamento parado (fora de escopo
por decisão de metodologia já registrada).

### 2.4 Inventário consolidado: quem serve para quê

`scripts/inventory_sensors.py` → `eval_pressure_out/inventario_sensores.csv`.
Distinção central: **ALVO** (exige incidentes rotulados com máquina ON) vs **ENTRADA**
(canal de contexto — não exige rótulo).

**Alvos utilizáveis (5):**

| Sensor | Incidentes ON | Obs |
|---|---:|---|
| TC382_03_A | 111 | único com poder estatístico real |
| PDAL_6240302 | 46 | ⚠️ 33% pós-partida — não promover sem excluir partida |
| T5_AVG_A | 18 | |
| TC382_05_A | 8 | amostra pequena |
| TAH_6240305 (TI_0305) | 8 | **novo** — mancal, modo de falha inédito |

**Achado desconfortável e importante:** dos 16 sensores em produção, **só 3 têm
ground-truth verificável**. TC382_04/06 e os 10 TV_* têm zero HI/HIHI utilizável — os
bundles deles estão deployados com recall inverificável (não "errado": inverificável).
Os 40 canais restantes (incl. todas as pressões) ficam como **entrada** de contexto.

---

## 3. TC382_03_A — diagnóstico do backcast 2024 (item de maior valor técnico)

### 3.1 Situação

| Janela | recall_raw | Leitura |
|---|---:|---|
| OOS ≥jul/2025 | **100%** (17/17) | resolvido |
| FULL jun/24→abr/26 | 79,3% | gap concentrado em 2024 |
| Backcast 2024 (8h) | 21,4% | o modelo não generaliza p/ o regime de 2024 |
| Backcast 2024 (72h) | 82–100% | …mas o sinal existe, dias antes |

### 3.2 Experimento: threshold por banda de regime (offline, sem retreino)

`scripts/sweep_regime_band_offline.py` — re-rank do MAE em cache do v10 dentro de bandas
de nível do T5 suavizado (proxy de carga lento). Artefato identificado por impressão
digital; braço controle **reproduz a auditoria exata** (79,3/21,4/100).

| Braço | FULL (58) | Backcast 2024 (28) | OOS (17) |
|---|---:|---:|---:|
| global (produção) | **79,3%** | 21,4% | **100%** |
| band3_t5 | 67,2% | **39,3%** (FA 0,090<0,104) | 88,2% |
| rolling30 (baseline adaptativo) | 53,4% | 21,4% | 82,4% |

**Decisão: não promover** (o ganho de 2024 custa 2/17 no OOS, nossa métrica de produção;
ambos os efeitos na borda do ruído de ±3 incidentes). **Valor do experimento:** confirma
que a causa do backcast fraco é **regime** — o que fortalece a Frente B (retreinar
incluindo jun–dez/2024), que ataca a causa na raiz. O baseline adaptativo (rolling rank
30d) foi **refutado** — normaliza o próprio precursor.

### 3.3 Figuras novas (prontas para apresentação)

- `eval_predictive_out/fig_oos2025_TC382_03_A_serie.png` — série completa do OOS:
  temperatura + health index no ponto da auditoria, 17/17 verdes, FA e duty visíveis.
  Recalculada do cache com sanidade automática (aborta se não reproduzir 17/17).
- `eval_predictive_out/fig_oos2025_TC382_03_A_zoom.png` — mecanismo: alerta ativo ≥8h
  antes nos incidentes de 2026; inclui honestamente o caso abrupto de out/25 (0,3h).

---

## 4. Descoberta: PDI_0317 como proxy de carga estacionário

Teste de estacionariedade (o mesmo que rejeitou vibração como proxy em 2024):
fit TC03~candidato em jan–jun/25, resíduo medido OOS.

| Candidato | corr | Resíduo OOS jan–abr/26 | Veredito |
|---|---:|---:|---|
| **PDI_0317** (ΔP gás comb.) | **0,982** | **−7,7 ± 7,5°C** | ✅ passa |
| PI_0315 | −0,47 | +34,1°C | ✗ |
| PI_0307 | −0,50 | +22,6°C | ✗ |
| vibração (teste 2024) | R²=0,99 | +34°C | ✗ |

O TC03 subiu ~52°C de patamar em 2026 e o resíduo contra o PDI_0317 ficou dentro de
±8°C — ou seja, o "novo regime" é quase todo vazão de combustível (física: ΔP ≈ vazão ≈
carga térmica). É exatamente a variável que **não existia** quando investigamos o drift
common-mode de 2024. Contexto físico é a única família de feature que já funcionou
(Transpetro: 5/5 sensores, FA −40 a −63%).

**Pronto para rodar:** `configs/calibracao_v12_pressao/v12_tc03_ctx_pdi0317.json`
(v10 + `CONTEXT_COLS=[954005_624_PDI_0317]`; resíduo+ratio como canais adicionais, sinal
cru preservado). Limitação: sem backcast 2024 (o PDI não existe no CSV de 2024).

---

## 5. Também nesta janela de trabalho (sessão de 02/08)

- **Threshold μ+y·σ implementado como botão de sensibilidade** (pedido do time/cliente):
  `THRESH_MODE="mean_std"` + `THRESH_STD_MULT` no treino, e
  `scripts/set_bundle_threshold.py --std_mult/--scale/--abs` para ajustar bundle
  **deployado** sem retreino, com histórico auditável. Default de produção inalterado
  (percentil). Aviso obrigatório ao cliente: no TC03/2025 o ponto calibrado equivale a
  **y≈0,2** e **y=3 zera a detecção** (cauda pesada do erro infla σ) — calibrar com
  `scripts/sweep_threshold_mean_std.py`, não com a régua gaussiana.

---

## 6. Próximos passos

| # | Ação | Depende de |
|---|---|---|
| 1 | **Frente B**: ablação pareada treino jun/24→jul/25 vs jan/25→jul/25 (só `TRAIN_START_DATE` muda; avaliar na janela idêntica) | servidor ClearML |
| 2 | **Ctx PDI_0317**: `v12_tc03_ctx_pdi0317.json` pareado contra v10 no OOS | servidor ClearML |
| 3 | Treinar `TI_0305` (mancal) — sentinela própria (satura em 871,0; faixa 500/950 não serve) | servidor ClearML |
| 4 | Reposicionar 2024 como **early-warning 72h** no relatório ao cliente (82–100% já entregues) | nada — offline |
| 5 | Corrigir ausência do `TV_355Y_A` no CSV 2024h2 (bundle existe, coluna não) | baixa prioridade |

**Commits da sessão:** `aa17fc7` (mapa+portão), `925cefd` (auditoria RUNNING_A),
`a0ab476` (figuras pressão), `96fad3b` (inventário), `0a11ef4` (banda de regime),
`80ba9e6`/`905d099` (figuras OOS), `4d20189` (config ctx PDI_0317).
