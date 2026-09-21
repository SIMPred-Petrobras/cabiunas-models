# TC-33003A — o que existe e o que falta para o dashboard

Turbocompressor TC-33003A, UTGCA Cabiúnas. Detector físico de 4 sinais, gatilho
de dois níveis (v2). Este documento é o inventário para a integração: o que já
está pronto, o que ainda não está, e o contrato entre nós e a tela.

**Princípio que guia o pacote:** nada de código fechado. O modelo é JSON legível;
a aritmética está escrita à vista; a inferência não depende de sklearn nem de
nenhuma biblioteca nossa.

---

## 1. O que existe

| Item | Onde | Status |
|---|---|---|
| Módulo de inferência | `scripts/cabiunas_inference.py` | pronto — autocontido, só numpy + pandas |
| Script de exemplo (4 passos) | `scripts/tc33003a_exemplo.py` | pronto — segue o padrão SIMPred |
| Retreino mensal | `scripts/constroi_bundle.py` | pronto — determinístico, ~1,5 s |
| Migração pickle → JSON | `scripts/migra_bundle_json.py` | pronto — executado nos 16 bundles |
| Bundles | `modelos/model_*_PCA4SINAIS/` | 16 bundles, 716 KB, **100 % JSON** |
| Dados | `dados/2025_2026/data_2025-01-01_2026-04-30_raw.csv` | 349.200 linhas, 484 d, 134,7 MB |
| Registro de trips | `registro_trips.csv` | 9 eventos |
| Catálogo de tags | `metadata.csv` | pronto |
| Contrato do dashboard | `contrato_dashboard()` + `--json` | **novo** |
| Diagnóstico da entrada | `diagnostico_entrada()` | **novo** |

### O modelo não tem código fechado

O bundle guardava `RobustScaler` e `PCA` em pickle. Mesmo sendo sklearn puro —
já melhor que a v1 da Transpetro, cujos pickles referenciavam classes internas —
pickle continua fechado: não se lê num diff, não se audita, exige a versão certa
do sklearn e **executa código ao carregar**. O dashboard roda em outro time.

O que os pickles guardavam são quatro vetores por família, e a aritmética inteira
é esta, publicada em `<família>_transformacao.json`:

```
Xs  = (X - center) / scale               RobustScaler.transform
Z   = Xs @ components.T - mean_proj      PCA.transform
rec = Z @ components + mean              PCA.inverse_transform
```

São 56 números na temperatura e 84 na pressão. Qualquer linguagem lê.

> **A ordem importa.** O sklearn centraliza *depois* de projetar
> (`decomposition/_base.py::_transform`: *"Apply the centering after the
> projection"*). Escrever `(Xs - mean) @ components.T` é algebricamente igual e
> difere em **5,7e-14**. Por isso o bundle publica `mean_proj = mean @ components.T`
> já calculado: bate bit a bit e dá menos conta para o consumidor fazer.

Conferido: os 32 JSON reproduzem os 16 pares de pickle bit a bit, e a saída do
detector nos 43.921 instantes é idêntica nas 10 colunas.

Os `.pkl` continuam na pasta por precaução. Para removê-los depois do aceite:

```bash
python3 migra_bundle_json.py --remover-pkl
```

### Dependências da inferência

```
numpy · pandas
```

Só isso. `scikit-learn` some do caminho de inferência (é usado apenas no
retreino, para *ajustar* o PCA). Não há torch — o detector não usa rede neural.

---

## 2. O contrato

### Entrada

CSV com a 1ª coluna de timestamp UTC e as tags nas demais, grade de 2 min:
`dados/<ano_ini>_<ano_fim>/data_<ini>_<fim>_raw.csv`.

**36 tags obrigatórias** (14 temperatura, 12 pressão, 10 vibração — mancal e
máscara reusam algumas), mais `RUNNING_A` e `T5_AVG_A`.

**Para reportar N dias, alimente N + 30.** Os primeiros 30 dias de qualquer
entrada são aquecimento da referência rolante de 400 h. Medido:

| dia desde o início da entrada | `vb` difere em |
|---|---|
| 0–21 | 100,0 % |
| 21–30 | 44,9 % |
| 30+ | 0,0 % |

`corte_valido()` recua o corte sozinho e avisa, em vez de devolver número errado
em silêncio.

### Saída

Duas peças, geradas na mesma execução:

```bash
python3 tc33003a_exemplo.py --dias 7 --json estado.json
```

**`tc33003a_inferencia.csv`** — a série, um instante por linha:

| coluna | significado |
|---|---|
| `t`, `p`, `sp`, `vb` | os 4 canais (z robusto acumulado por CUSUM) |
| `a_t`…`a_vb`, `b_t`…`b_vb` | qual canal acendeu, por nível |
| `canais_nivel_a/b` | a contagem |
| `forca` | quantas vezes o limiar específico |
| `vigiado` | o detector estava enxergando |
| `is_anomaly`, `severity` | `normal` / `atencao` / `alarme` |

**`estado.json`** — o que a tela precisa sem reimplementar o detector: limiares
(para desenhar a linha junto da curva), observabilidade, validade do bundle,
estado atual e episódios com os canais que participaram.

```json
{
  "schema": "simpred.cabiunas.tc33003a/1",
  "bundle":   { "vence_em": "...", "dias_restantes": 29, "vencido": false },
  "observabilidade": {
      "fracao_vigiada": 0.5912,
      "criterio": "RUNNING_A > 0.5 e T5_AVG_A > 300 °C",
      "entrada": { "veredito": "ok", "tags_mortas": [] }
  },
  "estado_atual": { "severity": "normal", "vigiado": false, "canais_nivel_a_acesos": [] },
  "episodios": [
      { "inicio": "...", "horas": 52.4, "forca_max": 2.87, "canais": ["t","p","sp","vb"] }
  ]
}
```

`--json-series` inclui a série inteira; para janelas longas prefira o CSV.

### O detector não guarda estado entre execuções

Cada execução recalcula tudo a partir da janela, refratário e CUSUM inclusive.
Não há banco, não há checkpoint, não há ordem obrigatória entre execuções — duas
execuções com a mesma entrada dão a mesma saída. O preço é reprocessar 60+ dias
toda vez, e isso custa **9,5 s para 484 dias**.

---

## 3. O que a tela **tem** de mostrar

Três campos que não são enfeite. Um painel de preditiva que os omite mente por
omissão, e esse é o erro mais caro que um painel comete.

**1. `observabilidade.fracao_vigiada`.** O detector só enxerga a máquina de pé e
em regime. Na janela de março–abril/2026 ela ficou **59 % do tempo** vigiada.
Mostrar "normal" sem dizer que o detector está cego nos outros 41 % é afirmar
saúde onde houve ausência de medição.

**2. `observabilidade.entrada.veredito`.** Perder tag não levanta erro. Medido
nos 484 dias, apagando uma tag por vez:

| tag removida | canal | episódios | efeito |
|---|---|---|---|
| 1 de 14 termorresistências | `t` | 20 → **22** | canal morto 100 % do tempo |
| 1 de 12 tomadas de pressão | `p` | 20 → **15** | **5 episódios perdidos** |
| 1 de 10 sondas de vibração | `vb` | 20 → 20 | tolera (indisponível em 3,8 %) |
| `RUNNING_A` | máscara | 20 → **0** | **cego, reportando "normal" para sempre** |

A vibração aguenta porque `vb` é o *máximo* entre sondas; temperatura e pressão
não aguentam porque são reconstrução conjunta — `_recon_pca` exige todas as
colunas da família no instante.

`diagnostico_entrada()` classifica em `ok` / `degradado` / `cego`. **`cego` para
a execução com código de saída 3**, para o agendador detectar.

**3. `bundle.dias_restantes`.** O bundle vale 62 dias após o fim do baseline.
Vencido, o script recusa rodar — de propósito. Congelar o modelo custa duas
detecções e **quinze vezes** as horas de alarme falso:

| cadência | detecção | h/mês de FP |
|---|---|---|
| mensal | 8/8 | 7,15 |
| trimestral | 7/8 | 4,48 |
| congelado | 6/8 | **108,19** |

O dashboard deve alertar a partir de ~15 dias restantes.

---

## 4. O que falta

| # | Item | Por que trava | Quem |
|---|---|---|---|
| 1 | **`prepara_dados.py`** | converte o `.xlsx` do portal da integridade no CSV. Sem ele ninguém reproduz a entrada quando chegar dado novo | nós |
| 2 | **`README.md` na raiz do equipamento** | quem abre a pasta no Drive não tem por onde começar; hoje o README está em `scripts/` | nós |
| 3 | **Agendamento** | hoje é execução manual. Precisa de um job: inferência (diária?) e retreino (dia 1 de cada mês) | engenharia |
| 4 | **Destino da saída** | hoje o CSV e o JSON caem em disco. A tela precisa de um lugar combinado ou de um endpoint | engenharia |
| 5 | **Atualização do `registro_trips.csv`** | a cada trip alguém tem de acrescentar a linha; sem isso a referência de vibração degrada | operação |
| 6 | **Fuso** | tudo em UTC no contrato. A tela mostra BRT (UTC−3) | engenharia |
| 7 | **Bundles do período corrente** | os 16 vão até 29/03/2026. Faltam os meses seguintes quando o dado chegar | nós |

### Sobre o item 1

É o elo mais arriscado e é onde já erramos uma vez: usar o export `*_recorded.xlsx`
em vez do `*_interpolated.xlsx` dá 7 valores por dia, e o MAD sai exatamente zero
sem nenhuma mensagem de erro. O script precisa validar taxa de variação, não só
cobertura.

### Sobre o item 3

O retreino é **sempre com corte no dia 1**, não "últimos 30 dias". Mudar o dia do
mês custa caro:

| dia do corte | h/mês de FP |
|---|---|
| **1** | **6,6** |
| 8 | 77,3 |
| 15 | 154,6 |
| 22 | 76,9 |

`--mes AAAA-MM` já garante o dia 1 venha o operador a rodar quando vier.

---

## 5. Como rodar

```bash
pip install -r requirements.txt      # numpy, pandas (sklearn só para retreinar)

# inferência
python3 tc33003a_exemplo.py --dias 7 --json estado.json

# retreino do mês (dia 1 de cada mês)
python3 constroi_bundle.py \
    --historico ../dados/2025_2026/data_2025-01-01_2026-04-30_raw.csv \
    --trips ../registro_trips.csv \
    --mes 2026-05
```

Códigos de saída: `0` ok · `2` bundle vencido · `3` entrada cega.

---

## 6. O que o número significa

O detector acha **8 de 8** eventos-alvo (trips de 2º estágio, janela a partir de
01/02/2024), com lead médio de 29,0 h — média, não mediana: 4 dos 8 estão
censurados em 48 h e o mínimo é 2,8 h.

O custo publicado é **0,344 FP/mês e 6,6 h/mês**. Duas ressalvas que pertencem à
integração, não ao rodapé:

- **6,6 h/mês é o mínimo de uma distribuição, não a expectativa.** Variando o
  tamanho do baseline em ±10 %, a faixa vai de 6,6 a 68,1 h/mês.
- **A carga real sobre a operação é 48,9 h/mês**, somando alarme e os episódios
  que a régua de avaliação classifica como neutros.

Dimensione a tela e o plantão pela carga, não pelo mínimo.
