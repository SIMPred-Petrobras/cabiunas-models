# Detector físico de 4 sinais — TC-330.03A

Antecipação de trips do turbocompressor TC-330.03A (Petrobras, UTGCA Cabiúnas).
Detecção de anomalia **não supervisionada** com modelo de normalidade aprendido:
PCA reajustado mensalmente em walk-forward, mais estatística robusta e CUSUM nos
canais de mancal e vibração, com política de decisão por votação.

## Resultado no ponto de produção

Janela 2025-01-01 a 2026-04-30 · 353,2 dias de operação vigiada de 485 de calendário.

| métrica | valor |
|---|---|
| falhas antecipadas | **8 de 8** (7 de 8 sob leave-one-event-out) |
| antecedência média | 29,0 h (censurada em 48 h; mínimo 2,8 h) |
| falso positivo | 0,517 por mês de operação |
| tempo em alarme falso | 7,15 h/mês — 0,98% do tempo de operação |
| episódios | 8 detecção · 6 antes de parada real · 6 falso positivo |

Os números acima usam a **regra C** (ver [A régua](#a-régua)). Na régua crua, sem
perdão de episódio seguido por parada real, o mesmo ponto dá 1,033 FP/mês e
38,65 h/mês.

> **O número a citar é 7/8, não 8/8.** Sob leave-one-event-out o 8/8 não se
> sustenta: o evento que cai é sempre **04/11/2025**, detectado por apenas 24% das
> configurações dentro do orçamento de falso positivo — os outros sete ficam entre
> 59% e 76%. Sete eventos têm detecção estável; um é frágil.

## Início rápido

Requer `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `scipy`, `matplotlib`.
Os scripts usam import relativo ao próprio diretório — **rode de dentro dele**.

```bash
cd scripts/pdm_fisico

# o ponto de produção, com a porta de mancal e a regra C
python plota_estilo_francisco.py
# -> TP=8 FP=6 NEUTRO=6  FP/mes=0.517  h/mes=7.15  lead_med=29.0h

# a fronteira custo x detecção (756 configurações)
python plota_fronteira.py            # com os pontos dos outros times
python plota_fronteira.py --so-nosso # só a nossa fronteira

# a figura do alvo: 3 famílias de sinal + os 8 trips + a máscara
python plota_alvo.py

# entregáveis
python build_pdf_decisao.py          # -> RELATORIO_DECISAO_DETECTOR.pdf
python build_apresentacao.py --pdf   # -> APRESENTACAO_TC33003A.pdf
```

### Artefatos de dado necessários (não versionados)

| arquivo | o que é | tamanho |
|---|---|---|
| `grade2min.parquet` | grade de 2 min dos 36 sensores, 2024-01 a 2026-04 | 84 MB |
| `piso_fisico_cache.npz` | escores de PCA e referência rolante já calculados | 26 MB |
| `falhas.csv` | os 8 eventos-alvo, derivados por regra (ver `verdade.py`) | 4 KB |

Ficam fora do git por serem dado, não código. Sem eles nada roda.

## A pipeline

```
DADO ─────────────► SINAL ──────────► DECISÃO ──────► ALARME
36 sensores         4 canais          2 gatilhos      refratário 48 h
PI Web API, 30 s    t p sp vb         degrau OU       duração mín. 120 min
grade de 2 min      EWMA              CUSUM
máscara             (adimensional)    voto ≥ 2
                                      exigindo sp|vb
```

**1 · Dado.** PI Web API a 30 s (já em UTC — comprovado empiricamente em
`verdade.py`, que varre o offset de −3 h a +3 h e acha o pico em 0 h). Reamostrado
para 2 min por **mediana**, que é robusta a spike isolado e dispensa filtro de
Hampel. Sentinela física de termopar (−40,5 °C) vira `NaN`.

**Máscara operacional** — separa dado de dado *julgável*. Só conta o instante em que
a máquina está **rodando** (`RUNNING_A`), **quente** (`T5_AVG_A > 300 °C`) e **fora
das 6 h** seguintes a um religamento. O blackout existe porque toda partida produz
um transiente mecânico real que não é falha. Sobram 353,2 dias de 485.

**2 · Sinal.** Quatro canais, cada um ancorado num mecanismo físico — ver
[Os quatro sinais](#os-quatro-sinais). Suavizados por EWMA com meia-vida de 1 h
(`t`, `p`) e 30 min (`sp`, `vb`).

**3 · Decisão.** Cada canal acende por **um de dois gatilhos**:

- **degrau** — acima do limiar por 15 amostras seguidas (30 min);
- **CUSUM** — soma acumulada de `(E/limiar − κ)` acima de `H`, que pega desvio
  fraco e persistente que o limiar sozinho nunca cruzaria.

O alarme exige **voto de dois canais**, e ao menos um deles tem de ser `sp` ou `vb`.
Essa exigência é o que protege de transiente de partida: um sinal sozinho dispara na
partida, dois simultâneos não.

**4 · Alarme.** Refratário de 48 h (repetir não é notícia — foi o que resolveu a
deriva de custo, que era repetição e não taxa) e duração mínima de 120 min.

## Os quatro sinais

| sinal | construção | entrada | mecanismo coberto |
|---|---|---|---|
| `t` | erro de reconstrução de PCA | 14 tags de temperatura (TI, TC382, T5) | degradação térmica de mancal |
| `p` | erro de reconstrução de PCA | 12 tags de pressão (PI, PDI, PDIT) | óleo lubrificante, selagem |
| `sp` | z robusto do spread | `abs(TI_0305 − mediana(0301, 0303, 0307))` | divergência entre mancais irmãos |
| `vb` | máximo do z robusto | 10 sondas `TV_351..355 X/Y` | vibração mecânica |

**O que é aprendido e o que não é.** `t` e `p` são erro de reconstrução de um PCA
ajustado nas últimas 20.000 amostras estáveis anteriores ao mês (~667 h de operação
quente), refeito 27 vezes na janela, **sempre para trás**. O ajuste seleciona por
operação estável e por tempo, e **não usa rótulo de falha nenhum** — é genuinamente
não supervisionado. `sp` e `vb` são z-scores robustos (mediana e MAD), estatística
clássica. O `vb` usa referência rolante de 400 h com banda de guarda de 24 h e passo
de 6 h, mecanismo separado do ajuste mensal.

Os rótulos entram só em dois lugares: na limpeza do baseline de `vb` (as janelas em
torno das falhas conhecidas saem do ajuste) e na calibração dos limiares. Isso torna
o sistema **semi-supervisionado** no conjunto, e é o que sofre no leave-one-out.

**Nenhuma metade funciona sozinha.** Medido, com limiares revarridos:

| o que fica | detecção | FP/mês | h/mês |
|---|---|---|---|
| completo — PCA + estatística | **8/8** | 0,517 | **7,15** |
| só estatística robusta, sem o PCA | 6/8 | 0,517 | 63,25 |
| só o PCA, sem a estatística | 3/8 | 0,861 | 26,18 |

## O ponto de operação

Constantes em `publica_clearml.py`. Não reajustar sem revarrer — ver
[Abordagens refutadas](#abordagens-refutadas).

```python
GRID     = "2min"    # grade de reamostragem
BLACKOUT = "6h"      # apaga as 6 h seguintes a cada religamento
SUSTAIN  = 15        # 15 amostras de 2 min = 30 min acima do limiar
T0       = "2025-01-01"                       # início da janela de avaliação
SIN      = ["t", "p", "sp", "vb"]
HL       = {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min"}   # meia-vida do EWMA
BASE     = {"t": 2.0, "p": 2.0, "sp": 3.0, "vb": 3.0}             # limiar base
K        = {"t": 1.7, "p": 1.7, "sp": 1.7, "vb": 2.2}             # multiplicador
KAPPA, H_CUSUM = 0.75, 80          # folga e limiar do CUSUM
REFRAT_H, DUR_MIN = 48, 120        # refratário e duração mínima
```

Os limiares são **adimensionais** — múltiplos do p99 por sensor e de desvios do
próprio baseline mensal, nada em °C ou bar. O detector porta para outro compressor
com os mesmos múltiplos, sem re-derivar unidade. O que não se auto-deriva é o
**valor** do múltiplo.

### O achado sobre o limiar

O limiar de `vb` está no **percentil 69** da distribuição saudável, deliberadamente
baixo. Substituindo só ele pelos percentis que uma varredura convencional usaria,
com o resto do detector intacto:

| limiar de `vb` | percentil | detecção | FP/mês |
|---|---|---|---|
| 6,60 — adotado | p69 | **8/8** | 0,517 |
| 19,31 | p90 | 4/8 | 0,517 |
| 44,35 | p99 | 3/8 | 0,431 |
| 65,83 | p99,9 | 2/8 | 0,344 |

O precursor de vibração nas 48 h antes de 09/12/2025 fica em **p80** — invisível
para um limiar em p99,9, visível para um em p69. E note a coluna de FP: subir o
limiar destrói a cobertura e quase não mexe no custo. **O limiar não é a alavanca de
custo**; quem segura o falso positivo é o voto, o SUSTAIN, o CUSUM e o refratário.

É isso que separa este detector das abordagens paralelas: **04/11/2025 e 09/12/2025
só são antecipados com o canal `vb`** — ablando a vibração o detector cai a 3/8 e
perde exatamente esses dois.

## A régua

Definida em `avalia.py` e igual à usada pela equipe do detector paralelo (conferida
no código dos dois lados: `DETECTION_WINDOW = "48h"`, `EPISODE_GAP = "2h"`).

- **Detecção** — o alerta tem de estar de pé na janela de **48 h antes** do trip.
  Estritamente antecipatório: reagir depois não conta.
- **Episódio** — alertas separados por menos de **2 h** são o mesmo episódio.
- **Falso positivo** — episódio que não antecipa falha *e* não é seguido de parada
  real da máquina.
- **Denominador** — mês de **operação vigiada** (730 h), não mês de calendário.

**Regra C — a terceira caixa.** Se houve parada real (≥ 2 h, por `RUNNING_A`) entre
o início do alerta e 48 h após o fim dele, o episódio fica fora das duas contas: o
detector viu algo que a operação também viu. Não é acerto nem erro. Sensibilidade da
janela: 2 h → 9 FP · 24 h → 8 FP · 48 h → 6 FP.

**O alvo é derivado por regra, não escolhido a mão** (`verdade.py`): queda de
`RUNNING_A` com parada ≥ 2 h coincidindo com alarme de nível de proteção na janela
de −1 h a +30 min; trips a menos de 24 h contam como um evento. Dá 9 eventos, dos
quais 8 caem na janela de avaliação. Somam 791,9 h de indisponibilidade — 89% delas
em dois eventos (29/04/2025 com 564,2 h e 04/11/2025 com 144,1 h).

## Mapa de arquivos

**Núcleo**

| arquivo | papel |
|---|---|
| `publica_clearml.py` | constantes do ponto de operação · `reproduz()` · publicação no ClearML |
| `pos_processamento.py` | carrega grade e cache, monta os 4 sinais, `partes()` e `pos()` |
| `avalia.py` | a régua: episódios, detecção, falso positivo, lead |
| `blackout_curto.py` | implementação do CUSUM |
| `verdade.py` | deriva `falhas.csv` dos alarmes, com o fuso comprovado |
| `piso_fisico.py` | gera `piso_fisico_cache.npz` — **não roda hoje**, ver abaixo |

**Figuras e entregáveis**

| arquivo | saída |
|---|---|
| `plota_estilo_francisco.py` | `fig_nosso_estilo_francisco.png` — série classificada |
| `plota_alvo.py` | `fig_alvo.png` — as 3 famílias + os 8 trips + a máscara |
| `plota_fronteira.py` | `fig_fronteira.png` e `fig_fronteira_nosso.png` |
| `plota_silencio12h.py` | `fig_silencio12h.png` — ponto alternativo de mínimo FP |
| `relatorio_decisao.html` + `build_pdf_decisao.py` | relatório de decisão (HTML e PDF) |
| `apresentacao.html` + `build_apresentacao.py` | deck de 11 slides (HTML e PDF) |

Os dois entregáveis usam **uma fonte só** para os dois formatos: o HTML é
versionado, e o builder embute as figuras e renderiza o PDF pelo Chrome headless.
Corrigiu um número, corrige nos dois.

Os demais `.py` do diretório são experimentos — cada um traz no docstring o que
testava e o que deu. Vale ler antes de repetir um teste.

## Reprodutibilidade — o que roda e o que não

**Roda a partir do repositório**, dados os três artefatos de dado: todo o cálculo do
ponto de operação, a fronteira, as figuras e os dois entregáveis.

**Roda também `piso_fisico.py`**, desde 04/09/2026. Ele importava `cabiunas_pdm`,
um pacote que vivia num diretório temporário e foi apagado — e é o script que gera o
`piso_fisico_cache.npz`, de onde saem os quatro sinais. O pacote foi **recuperado de
`origin/feat/pdm-deteccao-4sinais:src/cabiunas_pdm/`** e replicado em
`cabiunas_pdm/` (só os símbolos usados, cópia fiel; ver o `__init__.py` de lá).

O cache regenera em **12 segundos**, com 7 dos 10 arrays bit a bit idênticos ao
publicado. Os três que diferem (`t`, `p`, `mad_sp`) divergem por deriva numérica de
SVD entre versões de biblioteca — diferença relativa mediana de 1e-6 — e **a série
de alarme sai idêntica amostra a amostra**: 8/8, 20 episódios, 0,517 FP/mês,
7,15 h/mês, lead 29,0 h nos dois caches.

A fonte recuperada confirmou a correção do `recon_p99`: o `score()` original faz
`recon / self.recon_p99`, divisão que o `ScorerMax` do `autocalibra.py` omite. Esse
script tem ainda uma segunda inconsistência — aplica `abs()` no spread do mancal e
depois usa a mediana COM SINAL vinda do cache; o `_spread_mancal` original devolve
o spread com sinal e o `abs()` vai sobre o z-score, não sobre o spread. Não usar
`autocalibra.py` para números novos.

**Cuidado com `publica_clearml.reproduz()`:** ele monta o voto **sem** a porta de
mancal e devolve 1,120 FP/mês. O ponto de produção acrescenta
`& (ON["sp"] | ON["vb"])` e dá 1,033 na régua crua. O caminho canônico para o número
publicado é `plota_estilo_francisco.py::alarme()`.

## Limites conhecidos

1. **8/8 é o resultado observado no ponto selecionado**, não afirmação de robustez.
   O número robusto é 7/8.
2. **Oito eventos não permitem superioridade estatística.** Os intervalos de Poisson
   se sobrepõem em tudo; a comparação 8/8 contra 6/8 dá p = 0,50 no McNemar exato —
   e 0,50 é o *menor* valor possível com dois pares discordantes, ou seja, o teste
   não tem poder nenhum nesse tamanho de amostra. Chegar a p < 0,05 exigiria ~24
   eventos, uns quatro anos de operação.
3. **Risco de otimismo por seleção.** O ponto é o melhor de 756 configurações
   avaliadas sobre os mesmos oito eventos — otimista por construção.
4. **O gargalo é o rótulo, não o modelo.** Dois treinos de configuração idêntica de
   rede diferem 20,7 pontos de recall entre si; com oito eventos, a variação de
   semente é maior que a diferença entre arquiteturas.

A decisão de levar este detector se apoia em **mecanismo medido** (sabemos por que
funciona: o precursor está em p80 e o limiar baixo o enxerga) e em **assimetria de
custo** (os dois eventos que só ele antecipa valem 152 h de máquina parada; cobri-los
custa 41 h a mais de alarme falso no período inteiro — razão de 3,7 para 1), não em
teste de hipótese.

## Abordagens refutadas

Todas medidas. Não repetir sem informação nova.

**Sobre os falsos positivos na borda do blackout** — 8 dos 12 FP nascem em
`dist_partida = 6,4667 h` cravado, e 3 detecções reais nascem na mesma borda.
Refutados: gate de não-decaimento (1152 pontos, custa sempre o 04/11), ordenação por
severidade, duração da parada antes do religamento, rampa de T5 pós-partida, e gate
por alarme de gás de suprimento — este último morre no controle simétrico, porque
8/8 das detecções também têm alarme de gás perto.

**Sobre o custo** — o limiar não é alavanca (de `kb` 1,5 a 5,0 a detecção vai de 8/8
a 0/8 e o custo fica parado); 720 pontos de pós-processamento sem nenhum 8/8 abaixo
de 1,033 FP/mês na régua crua; blackout de 9/12/18/24 h custa detecção; voto ≥ 3
colapsa para 1/8.

**Sobre a arquitetura** — autocalibração de limiar por percentil não transporta
(melhor caso 7/8 a 132 h/mês, reconfirmado com o `ScorerMax` corrigido); voto entre
sondas de vibração é sobreajuste (o max fica); modelos especializados por subsistema
não transferem (6/8 e 40 h/mês); veto de sensor congelado não se aplica (só 0,3% do
tempo mascarado tem sensor travado, e nenhum episódio coincide).

**Testado pelos times paralelos, mesmo teto** — PCA e autoencoder empatam em 6/8
numa varredura de 10.368 configurações cada, pegando as mesmas falhas; OCSVM e
isolation forest dão 6/8 na régua de antecipação. Cinco famílias de modelo, o mesmo
teto: o que separou foi o sinal e o limiar, não a arquitetura.

## Próximo passo

Entrar com o canal `vb` como quinto sinal na varredura AutoML do detector paralelo,
com a **grade de limiar estendida até ~p60** e `confirm ≥ 2` obrigatório, e medir se
dá para manter os oito eventos com o custo de alarme daquela fronteira. A grade atual
vai de p99 a p99,995; sem estendê-la para baixo o teste falha por um motivo que não
tem relação com o sinal.

Aberto também: o miss de **24/11/2025** — parada de 43 h com anunciação de baixa
pressão no header de óleo lubrificante (`PAL_6240339`, primeiro estágio da proteção),
que não entra no alvo oficial porque a regra conta só o segundo estágio, e que este
detector não antecipa.
