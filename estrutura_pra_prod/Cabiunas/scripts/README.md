# TC-33003A — detector físico de 4 sinais (deploy)

Pacote de inferência do detector de precursores de trip do turbocompressor
TC-33003A, UTGCA Cabiúnas. Segue o padrão de entrega SIMPred: módulo
**autocontido**, script fino, bundle sem classe nossa dentro.

```
Cabiunas/
├── metadata.csv                      catálogo de tags (já existente)
├── registro_trips.csv                trips já ocorridos — ATUALIZAR a cada trip
├── documentos/                       (já existente)
├── dados/                            histórico por período
├── modelos/
│   └── model_<ini>_<fim>_PCA4SINAIS/  um por MÊS; guardar os antigos (8 KB cada)
│       ├── temperatura_scaler.pkl     RobustScaler (sklearn puro)
│       ├── temperatura_pca.pkl        PCA (sklearn puro)
│       ├── pressao_scaler.pkl
│       ├── pressao_pca.pkl
│       ├── normalizacao.json          p99 por sensor e da família, ordem das colunas
│       ├── spread_mancal.json         mediana e MAD do spread do mancal
│       ├── detector.json              ponto de operação
│       └── modelo.json                procedência, baseline, validade
└── scripts/
    ├── cabiunas_inference.py          MÓDULO autocontido (os 4 passos)
    ├── tc33003a_exemplo.py            script FINO
    ├── constroi_bundle.py             retreino mensal (ambiente de treino)
    ├── requirements.txt
    └── README.md
```

## Dependências

```
numpy · pandas · scikit-learn
```

Sem torch: este detector não usa rede neural — é PCA e estatística robusta.
Versões testadas em `requirements.txt` (Python 3.12). Os `.pkl` são pickles de
`RobustScaler` e `PCA`, então o scikit-learn precisa ser compatível com o do
treino (1.8.x).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## O que não está no repositório

Só o código é versionado. Antes da primeira execução, monte na pasta do
equipamento (os caminhos vêm do Drive do projeto):

```bash
# 1. o registro de trips já ocorridos — uma coluna `evento`, em UTC
cat > ../registro_trips.csv <<'CSV'
evento
2025-02-27 08:38:00+00:00
2025-03-17 18:16:00+00:00
CSV

# 2. um bundle por mês, do mais antigo ao corrente
python3 constroi_bundle.py --historico <grade_2min.parquet> \
        --trips ../registro_trips.csv --mes 2026-04

# 3. os dados de entrada em dados/data_tc33003a_raw.csv
#    (1ª coluna de timestamp UTC; as demais são as tags do metadata.csv)
```

## Como rodar

```bash
python3 tc33003a_exemplo.py                      # bundle mais recente, últimos 7 d
python3 tc33003a_exemplo.py --csv dados.csv --dias 3
```

Imprime os episódios de alarme da janela reportada e salva o resultado completo
em `tc33003a_inferencia.csv`.

## O que o detector é

Quatro sinais físicos, cada um com leitura direta:

| sinal | o que mede |
|---|---|
| `t` | erro de reconstrução do PCA sobre 14 tags de temperatura |
| `p` | erro de reconstrução do PCA sobre 12 tags de pressão |
| `sp` | z robusto do spread do mancal radial LNA contra seus três irmãos |
| `vb` | maior z robusto entre as 10 sondas de vibração, contra referência rolante de 400 h da própria sonda |

Cada sinal acende por **degrau sustentado** (30 min acima do limiar) ou por
**CUSUM** (deriva que fica logo abaixo do limiar por dias e nunca faz degrau).
Os dois são necessários e pegam coisas diferentes.

O alarme sai de um gatilho de **dois níveis**: um sensível (3 dos 4 canais em
limiar baixo) unido a um específico (2 dos 4 em limiar alto, com portão de canal
mecânico). Os quatro canais operam em percentis efetivos muito diferentes e
disparam em momentos descoordenados; nenhum nível sozinho passa de 4/8 na banda
acionável, e a união faz 5/8 ao custo do mais barato dos dois.

Desempenho no histórico (8 trips, 01/2025–04/2026, 11,6 meses de operação
vigiada): **detecção 8/8 · régua de início 6/8 · banda acionável [4 h, 48 h] 5/8
· 0,344 FP/mês · 6,6 h de alarme falso por mês**. A documentação completa da
calibração está em `DETECTOR_V2.md` no repositório de pesquisa.

## Validação do pacote

O pacote foi rodado **como produção rodaria** — um bundle por mês, janela de 60
dias de entrada, só o mês corrente como saída — sobre os 16 meses do histórico, e
o alarme foi comparado ponto a ponto com o do código de pesquisa:

| | banda | início | det | FP/mês | h/mês | episódios |
|---|---|---|---|---|---|---|
| pesquisa | 5/8 | 6/8 | 8/8 | 0,344 | 6,6 | 21 |
| **pacote** | **5/8** | **6/8** | **8/8** | **0,344** | **6,6** | **21** |

Os oito eventos têm antecedência idêntica ao centésimo de hora. A discordância
residual é de **0,012%** dos pontos (um trecho de 1,33 h em 09/03/2025), sem
efeito em nenhuma métrica.

## Cinco coisas que quebram o detector em silêncio

Nenhuma das cinco dá erro. Todas mudam o alarme. As três primeiras foram
**defeitos reais deste pacote**, encontrados pela validação acima — cada uma
parecia inofensiva e custava caro.

### 1. Pontuar o aquecimento com um bundle só

O bundle é mensal. Usar o do mês corrente para pontuar os 60 dias de aquecimento
aplica o PCA de hoje a dado de dois meses atrás: o resíduo cresce por deriva do
baseline, não por saúde da máquina. Medido em agosto/2025, o sinal `p` suavizado
divergiu em **3388**, e o CUSUM — que integra — carregou o erro para dentro do
mês. Por isso `carregar_modelos()` lê **todos** os bundles e cada trecho é
pontuado pelo que vigia nele. Guarde os bundles antigos.

### 2. Preencher buraco na grade

Uma versão deste módulo fazia `ffill(limit=2)` para tolerar jitter do
historiador. Os pontos preenchidos caíam em transientes de parada, onde o PCA
extrapola, e o resíduo saltava para ~1000 — direto para dentro do CUSUM. Efeito:
alarme falso de **63,0 h/mês** contra 6,6, com detecção *aparentemente melhor*
(6/8 na banda), porque mais alarme compra detecção.

Onde o historiador não mediu, o sinal fica NaN e o instante não é pontuado. Não
medir não é "igual ao anterior".

### 3. Registro de trips congelado ou desatualizado

A referência rolante do `vb` apaga ±7 dias em torno de cada trip **já ocorrido**:
degradação que se sabe ter terminado em falha não pode virar o "normal" contra o
qual a próxima é medida — entraria na mediana e se cancelaria.

Essa lista vive em `registro_trips.csv`, na pasta do equipamento, e é lida **a
cada execução** — não dentro do bundle. O motivo é concreto: em 27/04/2025 o
bundle de abril (baseline até 31/03) ainda não sabia dos trips de 07/04 e 11/04,
embora já tivessem acontecido. Com eles dentro da referência o MAD sobe e o z
cai: `vb` médio de **3,56** onde o correto era **8,63**.

O registro muda quando a máquina falha, não quando o modelo é retreinado.
**Acrescente cada trip novo antes da execução seguinte.**

### 4. Mudar o dia do mês em que o retreino roda

O baseline são os 20.000 pontos estáveis **anteriores ao corte**. Mover o corte
muda *quais* 20.000, e o detector é extremamente sensível a isso — mais do que a
qualquer outro parâmetro medido:

| dia do retreino | banda | detecção | FP/mês | h/mês |
|---|---|---|---|---|
| **1** (o publicado) | **5/8** | **8/8** | **0,344** | **6,6** |
| 8 | 3/8 | 5/8 | 0,947 | 77,3 |
| 15 | 4/8 | 7/8 | 0,947 | **154,6** |
| 22 | 5/8 | 7/8 | 0,689 | 76,9 |

Vinte e três vezes as horas de alarme falso por uma escolha que ninguém
consideraria um parâmetro. O `constroi_bundle.py --mes AAAA-MM` já corta sempre
no dia 1, venha o operador a rodar quando vier. **Não trocar por data corrente
nem por "últimos 30 dias".**

### 5. Bundle com mais de 62 dias

Os canais `t` e `p` são resíduo de PCA contra um baseline. PCA em baseline velho
descola conforme o ponto de operação anda (campanha, carga, ambiente) e o
resíduo cresce por motivo que **não é saúde da máquina**. Medido:

| cadência | ajustes | detecção | FP/mês | h/mês |
|---|---|---|---|---|
| **mensal** | 27 | **8/8** | 0,517 | **7,15** |
| trimestral | 10 | 7/8 | 0,431 | 4,48 |
| semestral | 5 | 6/8 | 0,431 | 16,94 |
| congelado | 1 | 6/8 | 0,861 | 108,19 |

Congelar custa duas detecções e **quinze vezes** as horas de alarme falso. Por
isso a inferência **recusa** um bundle vencido em vez de degradar calada.

## A janela de entrada

O detector tem memória: a referência rolante do `vb` olha 400 h estáveis para
trás, o CUSUM acumula desde o último reset (corridas de até 35,7 d no histórico)
e o refratário dura 72 h. Só os últimos dias da janela são resultado; o começo é
aquecimento, e não é opcional.

Medido em 61 execuções semanais contra o histórico completo: **45 d reproduz bit
a bit, 30 d diverge em até 17,1% dos pontos**. No pacote completo, 60 · 120 · 180
· 240 d dão resultado **idêntico** — acima do mínimo a janela não compra nada.
Ficam os 60 d: é o mínimo medido com margem, e nada além disso ajuda.

O script avisa quando a entrada é curta, mas não recusa: reprocessar histórico é
caso legítimo.

## Cadência operacional

| quando | o quê |
|---|---|
| a cada execução | `tc33003a_exemplo.py` com ≥ 60 d de entrada |
| todo mês | `constroi_bundle.py --mes AAAA-MM` (obrigatório) — e **guarde os bundles antigos** |
| a cada trip | acrescentar o evento a `registro_trips.csv` antes da execução seguinte |

## O que mudou em relação ao código de pesquisa

O detector é **o mesmo, já calibrado e validado** — nada foi reajustado. O
bundle foi reempacotado para abrir **sem a nossa biblioteca**:

| pesquisa | deploy | motivo |
|---|---|---|
| `ScorerMax` (classe nossa, herda de `MultivariateScorer`) | `RobustScaler` + `PCA` em pickle + `normalizacao.json` | o pickle da classe só abre com a lib instalada. A aritmética que ela fazia por cima (máximo do erro por sensor, normalizado, com piso) está escrita à vista em `cabiunas_inference.py` |
| cache `.npz` com sinais pré-calculados | recálculo a partir do dado bruto | o cache era do walk-forward inteiro; produção vê uma janela |
| constantes espalhadas em `publica_clearml.py` | `detector.json` no bundle | o ponto de operação é dado, não código |
| lista de falhas embutida no código do experimento | `registro_trips.csv` lido na inferência | é registro de operação, e muda sem o modelo mudar |

É a mesma correção que a Transpetro já aplicou entre a v1 e a v2 do deploy dela.
