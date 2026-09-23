# TC-33003A — detector de precursores de trip

Turbocompressor TC-33003A, UTGCA Cabiúnas. Projeto SIMPred.

Detecta degradação antes do trip de 2º estágio usando quatro sinais físicos.
Não é supervisionado: aprende a operação normal do mês e mede o afastamento.

**Acha 8 de 8** dos trips do período avaliado, com antecedência média de 29 h.
Custa **0,344 alarme falso por mês**. As ressalvas desses números estão em
`documentos/ENTREGA_TC33003A.pdf` — leia antes de dimensionar plantão.

---

## Comece por aqui

| Se você é... | Leia |
|---|---|
| quem vai **integrar ao dashboard** | `documentos/INTEGRACAO_DASHBOARD.pdf` |
| quem vai **operar / plantão** | `documentos/ENTREGA_TC33003A.pdf` |
| quem quer **entender o modelo** | `documentos/DETECTOR_V2.pdf` |
| quem vai **mexer no código** | `scripts/README.md` |

## Rodar

```bash
cd scripts
pip install -r requirements.txt
python3 tc33003a_exemplo.py --dias 7 --json estado.json
```

Precisa de **numpy e pandas**. Só isso — não há sklearn na inferência, não há
torch, não há biblioteca nossa. O modelo é JSON legível.

Saída: `tc33003a_inferencia.csv` (a série) e `estado.json` (o que a tela
consome). Códigos de saída: `0` ok · `2` bundle vencido · `3` entrada cega.

## Retreinar — obrigatório todo mês

```bash
python3 constroi_bundle.py \
    --historico ../dados/2025_2026/data_2025-01-01_2026-04-30_raw.csv \
    --trips ../registro_trips.csv \
    --mes 2026-05
```

Roda em ~1,5 s. Escreve um bundle novo em `modelos/` e **não apaga os antigos**
— cada trecho é pontuado pelo bundle que vigia nele, e são 8 KB por mês.

Não é preferência: congelar o modelo leva a detecção de 8/8 para 6/8 e as horas
de alarme falso de 7,15 para 108,19 por mês. Por isso a inferência **recusa**
rodar com bundle vencido (62 dias após o fim do baseline) em vez de degradar
calado.

## O que tem aqui

```
metadata.csv          catálogo de tags do equipamento
registro_trips.csv    trips já ocorridos — ATUALIZAR a cada trip novo
dados/                a entrada, um CSV por período
modelos/              um bundle por mês; guardar os antigos
documentos/           relatórios e a documentação de integração
scripts/              o módulo, o exemplo, o retreino
```

### `registro_trips.csv` é responsabilidade da operação

Uma linha por trip, coluna `evento`, timestamp UTC. O detector usa esse registro
para não confundir degradação já conhecida com normalidade ao montar a
referência de vibração. Trip que não entra aqui degrada o canal `vb`.

## Limites que valem saber antes de usar

- **O detector só enxerga a máquina de pé e em regime.** Na janela avaliada isso
  foi 59 % do tempo. Nos outros 41 % ele não está dizendo "normal" — está cego.
  O campo `observabilidade.fracao_vigiada` no JSON diz quanto.
- **Os primeiros 30 dias de qualquer entrada são aquecimento.** Para reportar N
  dias, alimente N + 30. O script recua o corte sozinho e avisa.
- **Perder uma tag não levanta erro** — levanta número errado. Sem `RUNNING_A` o
  detector fica cego reportando "normal". Por isso existe o diagnóstico de
  entrada, e por isso ele **para** em vez de seguir.

## Contato

Dúvidas de modelo e retreino: equipe de ciência de dados.
Dúvidas de integração: ver `documentos/INTEGRACAO_DASHBOARD.pdf`, seção 4.
