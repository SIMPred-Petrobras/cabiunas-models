"""O teste decisivo do gate de gas combustivel.

A anatomia mostrou que 5 dos 6 FP tem alarme do sistema de GAS COMBUSTIVEL perto
(PAL_6240315 Pressao Baixa Gas Comb., PDAL_6240302 PrDif.Bx.Gas Linha Balanc.,
PI_6240319_AL Falha PI Gas Motor p/Partida). Isso e informacao EXOGENA -- vem do
sistema de alarme da planta, nao dos nossos 4 sinais. E a "informacao nova" que
[[borda-do-blackout-explica-os-fp]] dizia ser o unico caminho restante.

MAS: todo gate que testamos ate hoje morreu porque tambem matava deteccao. O teste
que decide e simetrico -- quantos dos 8 TP tambem tem alarme de gas perto?
Se a taxa for igual nos dois grupos, o gate nao separa e morre como os outros.
"""
from __future__ import annotations
import sys
import pandas as pd
sys.path.insert(0, ".")

from plota_estilo_francisco import alarme, paradas_reais_2h, classifica_regra_c
from pos_processamento import partes, mask, idx, alvo
from publica_clearml import SIN
import avalia as AV
from verdade import carrega_alarmes

TAGS_GAS = {"PAL_6240315", "PDAL_6240302", "PI_6240319_AL"}
KB, KV = 1.7, 2.2
ON = partes(KB, KV)
eps = AV.episodios(alarme())
paradas = paradas_reais_2h()
classif = classifica_regra_c(eps, paradas)
alarmes = carrega_alarmes(0)
gas = alarmes[alarmes["Tag Alarme"].isin(TAGS_GAS)]
print(f"alarmes de gas na base: {len(gas)}  (de {len(alarmes)} ACT totais)")


def assinatura(a, b):
    jan = (idx >= a) & (idx <= b)
    fr = {c: ON[c].to_numpy()[jan].mean() for c in SIN}
    ns = sum(ON[c].astype(int) for c in SIN).to_numpy()[jan]
    return fr, int(ns.max())


print("\n" + "=" * 100)
print("ASSINATURA DOS 8 TP (para comparar com os FP/NEUTRO da rodada anterior)")
print("=" * 100)
print(f"{'inicio':>16} {'dur_h':>8} {'nmax':>4}   fracao ON por sinal")
for a, b, c, lead in classif:
    if c != "TP":
        continue
    fr, nmax = assinatura(a, b)
    print(f"{a:%d/%m/%Y %H:%M} {(b-a).total_seconds()/3600:7.1f}h {nmax:4d}   "
          f"t={fr['t']:.2f} p={fr['p']:.2f} sp={fr['sp']:.2f} vb={fr['vb']:.2f}")

print("\n" + "=" * 100)
print("O TESTE SIMETRICO: alarme de gas perto do INICIO do episodio?")
print("=" * 100)
for jan_h in (6, 12, 24):
    J = pd.Timedelta(hours=jan_h)
    print(f"\n--- janela +-{jan_h}h em torno do inicio do episodio ---")
    cont = {"TP": [0, 0], "FP": [0, 0], "NEUTRO": [0, 0]}
    for a, b, c, lead in classif:
        tem = ((gas.ts >= a - J) & (gas.ts <= a + J)).any()
        cont[c][0] += int(tem)
        cont[c][1] += 1
    for c in ("TP", "FP", "NEUTRO"):
        n, tot = cont[c]
        print(f"  {c:>7}: {n}/{tot} com alarme de gas perto  ({100*n/tot:.0f}%)")

print("\n" + "=" * 100)
print("DETALHE POR EPISODIO (janela +-12h), para ver quem morreria no gate")
print("=" * 100)
J = pd.Timedelta(hours=12)
for a, b, c, lead in classif:
    sub = gas[(gas.ts >= a - J) & (gas.ts <= a + J)]
    tags = sorted(set(sub["Tag Alarme"])) if len(sub) else []
    marca = "GATE MATA" if len(sub) else "passa    "
    print(f"{a:%d/%m/%Y %H:%M} {c:>7}  {marca}  {len(sub):2d} alarmes  {','.join(tags)}")

print("\n" + "=" * 100)
print("SE O GATE FOSSE APLICADO (+-12h): quanto sobra?")
print("=" * 100)
m = AV.avalia(alarme(), alvo, mask)
meses = m["horas_op"] / 730.0
for jan_h in (6, 12, 24):
    J = pd.Timedelta(hours=jan_h)
    sobra = {"TP": 0, "FP": 0, "NEUTRO": 0}
    h_fp = 0.0
    for a, b, c, lead in classif:
        tem = ((gas.ts >= a - J) & (gas.ts <= a + J)).any()
        if not tem:
            sobra[c] += 1
            if c == "FP":
                h_fp += (b - a).total_seconds() / 3600
    print(f"  janela +-{jan_h:2d}h -> TP={sobra['TP']}/8  FP={sobra['FP']}/6  "
          f"NEUTRO={sobra['NEUTRO']}/6  |  FP/mes={sobra['FP']/meses:.3f}  h/mes={h_fp/meses:.2f}")
