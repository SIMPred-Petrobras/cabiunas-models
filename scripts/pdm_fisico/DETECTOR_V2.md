# Detector físico de 4 sinais — **v2**, gatilho de dois níveis

TC-330.03A · Petrobras, UTGCA Cabiúnas · 11/09/2026

Substitui o ponto v1 publicado. **Melhor em todos os eixos**, com vizinhança
confirmada em onze parâmetros. Este documento descreve o que mudou, por que cada
peça existe, e o que o número significa em cada régua.

---

## 1. O resultado

Janela 01/01/2025 a 30/04/2026 · **11,61 meses de operação vigiada** (353 dias).

| | v1 (publicado) | **v2** | |
|---|---|---|---|
| detecção — régua "de pé" | 8/8 | **8/8** | mantida |
| detecção — **régua de início** | 4/8 | **6/8** | +2 |
| **banda acionável [4h, 48h]** | 3/8 | **5/8** | +2 |
| FP/mês (regra C) | 0,517 | **0,344** | −33% |
| horas de alarme falso/mês | 7,1 | **6,6** | −7% |
| antecedência média (início real) | 10,1 h | **16,7 h** | +65% |
| episódios totais | 20 | 21 | |
| LOEO aninhado | 7/8 | 7/8 | inalterado |

Não há eixo em que o v2 seja pior. É a primeira vez que isso acontece na série.

### Por evento

| trip | v1 | **v2** |
|---|---|---|
| 27/02/2025 | alarme de pé há 144 h | **nasce 1,6 h antes** |
| 17/03/2025 | alarme de pé há 195 h | de pé há 197 h |
| 07/04/2025 | nasce 5,0 h antes | **nasce 37,7 h antes** |
| 11/04/2025 | nasce 2,8 h antes | **nasce 4,6 h antes** ✓ banda |
| 29/04/2025 | alarme de pé há 52 h | de pé há 52 h |
| 04/11/2025 | nasce 8,8 h antes ✓ | nasce 8,8 h antes ✓ |
| 09/12/2025 | nasce 23,6 h antes ✓ | **nasce 27,1 h antes** ✓ |
| 26/02/2026 | alarme de pé há 670 h | **nasce 20,3 h antes** ✓ |

Os ganhos são concretos e concentrados em **quando** o alarme nasce, não em
quantos eventos são cobertos — a cobertura já era 8/8.

---

## 2. O que mudou, e por quê

### 2.1 Gatilho de dois níveis

O v1 tinha um regime só: `voto ≥ 2 de 4` com limiares `1,7 / 1,7 / 1,7 / 2,2`.

O v2 tem dois, em OU:

```
nível A (sensível)    : ≥3 de 4 canais, limiares  t 1,10 · p 0,70 · sp 0,90 · vb 1,80
nível B (específico)  : ≥2 de 4 canais, limiares  t 1,70 · p 1,70 · sp 1,70 · vb 2,20
                        + portão: sp OU vb aceso
alarme = A  OU  B
```

**O diagnóstico que motivou.** Os quatro canais operam em percentis efetivos muito
diferentes — o `vb` vive em ~p69 e o `t` em p87. Disparam em momentos
descoordenados: o `vb` acende cedo na deriva lenta e fica de pé; o `t` e o `p`
acendem tarde, na excursão rápida. Com um regime só, o detector herda o pior dos
dois comportamentos.

**A medição que prova que os dois são necessários** (`decompoe_dois_niveis.py`):

| gatilho | banda | início | det | FP/mês | h/mês |
|---|---|---|---|---|---|
| só o nível A | 4/8 | 4/8 | **5/8** | 0,344 | 11,2 |
| só o nível B | 4/8 | 6/8 | 8/8 | 0,603 | 10,1 |
| **A ou B** | **5/8** | **6/8** | **8/8** | **0,344** | **6,6** |

Nenhum dos dois sozinho passa de 4/8 na banda. A união faz 5/8 **e ao custo do
mais barato dos dois**, não da soma. Não é redundância — é divisão de trabalho: o
A traz cobertura na banda (4,41% do tempo é exclusivo dele), o B traz a detecção
completa que o A não sustenta.

### 2.2 Escalada por idade (reanúncio de alarme permanente)

```
se, dentro de um episódio que já tem ≥ 96 h de vida, a força cruza 20× o limiar
   → fecha o episódio e abre um NOVO
o episódio aberto por escalada é isento da duração mínima de 120 min
   (mas ainda precisa durar ≥ 60 min)
o refratário é FURADO por um episódio forte quando o bloqueio já é velho
```

**Por quê.** Reintensificação dentro de um episódio de 2 h é ruído; dentro de um
de 100 h é evento novo. É o reanúncio de alarme permanente que a **ISA-18.2**
prescreve. Nesta base, só episódios muito longos são quebrados — e os muito
longos são TP ou NEUTRO, então o ganho de detecção não cria falso positivo.

É o que transforma o 26/02/2026 de "alarme de pé há 670 h" em "nasce 20,3 h antes".

**Armadilha registrada:** o furo do refratário é metade da peça. Sem ele o
reanúncio nunca vira alarme — o episódio novo nasce e é imediatamente bloqueado.
Isso custou uma depuração na portagem para o `publica_clearml.py`.

### 2.3 O que NÃO mudou

`GRID 2min` · `BLACKOUT 6h` · `SUSTAIN 15` · `T0 2025-01-01` · halflives
`{t:1h, p:1h, sp:30min, vb:30min}` · `BASE {t:2,0 p:2,0 sp:3,0 vb:3,0}` ·
`KAPPA 0,75` · `H_CUSUM 80` · `CARGA 0,25` · duração mínima 120 min.

O refratário foi de **48 h para 72 h** (platô 48–72; 84 h já custa uma detecção).

---

## 3. As três réguas — leia antes de citar qualquer número

As constantes são iguais às das outras equipes (janela 48 h, fusão de episódio
2 h, denominador em tempo monitorado, regra C). **A regra de associação
evento↔episódio não é**, e isso foi descoberto em 04/09/2026:

| régua | pergunta que responde | v2 |
|---|---|---|
| **de pé** (a nossa antiga) | havia alarme **ativo** em [t−48h, t]? | 8/8 |
| **início** (a das outras equipes) | o episódio **nasceu** em [t−48h, t]? | **6/8** |
| **banda acionável** | nasceu **e** com tempo de agir, em [t−48h, t−4h]? | **5/8** |

São três números da mesma série. A régua de "de pé" credita alarme que já estava
levantado há semanas — em gerenciamento de alarme isso é defeito, não informação
(EEMUA 191, ISA-18.2). **A régua de início foi adotada como métrica de trabalho
em 11/09/2026.**

O **τ_min de 4 h** da banda é a única escolha arbitrária, e é ordem de grandeza:
aproximadamente o tempo de uma parada controlada. **Deveria vir da operação.** Se
o mínimo útil for 8 h ou 2 h, o número muda e é só recalcular.

**Ao publicar, cite a banda como número principal e as outras duas ao lado, com a
régua escrita.** É o único que não muda de valor conforme quem avalia.

---

## 4. Validação

### 4.1 Vizinhança — onze parâmetros, um de cada vez

Todos mantêm 6/8 na régua de início dentro das faixas:

| parâmetro | valor | faixa que mantém 6/8 | margem |
|---|---|---|---|
| `lo.t` | **1,10** | 1,05 – 1,15 | **±5% — a mais estreita** |
| `lo.p` | 0,70 | 0,70 – 1,20 | larga |
| `lo.sp` | 0,90 | 0,60 – 1,10 | larga |
| `lo.vb` | 1,80 | 1,60 – 2,00 | moderada |
| `hi` | 1,70 | 1,40 – 1,70 | moderada |
| `kv_hi` | 2,20 | 1,80 – 2,20 | moderada |
| escalada `idade` | 96 h | 48 – 120 h | larga |
| escalada `ABS` | 20 | 8 – 120 | larguíssima |
| escalada `dur` | 60 min | 30 – 90 min | larga |
| `refratário` | 72 h | 48 – 72 h | quebra em 84 h |
| religamento `frac` | **0,0** | 0,00 – 0,03 | 0,0 é o mais barato |

**O `lo.t` é o ponto frágil.** Na grade grossa parecia ponto único; com passo fino
é platô de três valores com 1,10 no centro e mais barato. Fora dele: 1,00 perde
detecção (7/8), 1,20 tem precipício de horas (6,6 → 44,6 h/mês).

### 4.2 LOEO aninhado

7/8 sob qualquer regra de desempate razoável; 8/8 só sem desempate algum, que é
artefato da ordem de varredura. O evento que cai é sempre o **04/11/2025** —
detectado por apenas **8 de 72** configurações dentro do orçamento (11%), contra
58% a 100% dos outros sete. Não mudou do v1 para o v2.

---

## 5. Limites conhecidos

**O 17/03 e o 29/04 continuam como alarme de pé.** No 17/03 os quatro canais
sobem por 8 dias — o alarme permanente é a saída *correta* para um precursor lento,
e a régua de início o pune. No 29/04 o precursor começa 51,8 h antes e **perde a
janela por 3,8 h**.

**O 27/02 nasce a 1,6 h**, abaixo do τ_min. Conta na régua de início, não na banda.

**Eventos de canal único são invisíveis.** Os 8 alvos movem ≥2 canais (mediana 4
de 7). Um evento que move um canal só — como a parada de 24/11/2025, com o `p` a
6,14× por 48 h e os outros três mudos — não dispara, e **não é o portão**: mesmo
sem portão nenhum continua não disparando, porque `voto ≥ 2` exige dois canais.

**Tudo está ajustado em 8 rótulos.** Provar 1,03 → 0,75 FP/mês exigiria 92 meses
de operação. Os mecanismos têm justificativa física independente; os números, não.

---

## 6. Por que não há mais ganho por ajuste

Medição de fundo (`trajetoria.py`): a distância mediana **entre** os 8 templates
de trajetória é **6,24**; a de um instante qualquer ao template mais próximo é
**4,85**. Um momento aleatório da série se parece mais com algum precursor de trip
do que os precursores se parecem entre si. **Os 8 eventos não têm assinatura
comum.**

Isso explica a fronteira inteira: nenhum canal único cobre os 8 → é preciso a
união de canais permissivos → canais permissivos ficam acesos 24–52% do tempo →
o voto satura em 44,9% → episódios se fundem → inícios 52 a 670 h antes.

E explica por que **cinco canais construídos nesta série, todos melhores
individualmente, não melhoram o detector**:

| canal | duty | banda | razão vs acaso | melhora? |
|---|---|---|---|---|
| `sp_vib` (spread de vibração) | 9,7% | 5/8 | 2,57× | não |
| modo comum dos termopares | 10,4% | 6/8 | 2,52× | não |
| margem ao setpoint (TI_0305) | 4,5% | 5/8 | 6,18× | marginal |
| inovação no `vb` | 1,6% | 6/8 | 3,26× | não |
| — para contraste, o `sp` atual | 41,8% | 2/8 | **0,55×** | — |

> Com 8 eventos heterogêneos, cobrir todos exige ser permissivo. Qualidade de
> canal e cobertura de detector são objetivos em tensão — não é problema de
> engenharia de sinal, é aritmética de amostra.

---

## 7. Reprodução

```bash
cd scripts/pdm_fisico
PYTHONPATH=. python publica_clearml.py --offline        # v2, só mede
PYTHONPATH=. python publica_clearml.py --offline --v1   # v1, para comparar
PYTHONPATH=. python publica_clearml.py                  # publica no ClearML
```

Requer `grade2min.parquet`, `piso_fisico_cache.npz` e `falhas.csv` (não
versionados). O `--offline` roda em ~40 s e imprime as três réguas, a tabela por
evento, os falsos positivos e o LOEO completo.

**Scripts que sustentam cada afirmação deste documento:**

| afirmação | script |
|---|---|
| os dois níveis são necessários | `decompoe_dois_niveis.py` |
| vizinhança dos onze parâmetros | `confirma_vizinhanca.py` |
| a banda e os leads das duas equipes | `banda_acionabilidade.py` |
| onde o detector perde | `onde_morre.py` |
| perícia dos eventos faltantes | `pericia_faltantes.py` |
| os 8 eventos não têm forma comum | `trajetoria.py` |
| o portão é redundante, não inerte | `portao_cego_pressao.py` |
| o alvo está certo | `revisita_alvo.py`, `caso_2411.py` |
