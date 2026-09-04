"""Anatomia dos 12 episodios nao-TP: quais sinais dispararam, distancia da partida,
e -- para os NEUTRO -- que tipo de parada veio depois (trip com alarme de nivel que
ficou fora do catalogo? parada limpa? manutencao?).

A pergunta que importa: algum amarelo e na verdade uma DETECCAO que o nosso alvo
de 8 eventos nao contabiliza porque a regra de rotulagem (alarme de nivel em
[-1h, +30min] da queda) e mais apertada do que a fisica?
"""
from __future__ import annotations
import sys
import pandas as pd
sys.path.insert(0, ".")

from plota_estilo_francisco import alarme, paradas_reais_2h, classifica_regra_c
from pos_processamento import partes, mask, idx, alvo, part
from publica_clearml import SIN
import avalia as AV
from verdade import carrega_alarmes

KB, KV = 1.7, 2.2
ON = partes(KB, KV)
al_series = alarme()
eps = AV.episodios(al_series)
paradas = paradas_reais_2h()
classif = classifica_regra_c(eps, paradas)
partidas = list(idx[part.to_numpy()])
alarmes = carrega_alarmes(0)


def assinatura(a, b):
    """Quais sinais ficaram ON dentro do episodio e por quanto tempo (fracao)."""
    jan = (idx >= a) & (idx <= b)
    out = {}
    for c in SIN:
        v = ON[c].to_numpy()[jan]
        out[c] = v.mean()
    ns = sum(ON[c].astype(int) for c in SIN).to_numpy()[jan]
    return out, int(ns.max())


def dist_part(a):
    ant = [(a - p).total_seconds() / 3600 for p in partidas if p <= a]
    return min(ant) if ant else float("inf")


print("=" * 108)
print("ANATOMIA DOS EPISODIOS (pretos = FP, amarelos = NEUTRO)")
print("=" * 108)
print(f"{'inicio':>16} {'classe':>7} {'dur_h':>7} {'d_part':>7} {'nmax':>4}   fracao ON por sinal (t/p/sp/vb)")
for a, b, c, lead in classif:
    if c == "TP":
        continue
    fr, nmax = assinatura(a, b)
    dp = dist_part(a)
    dps = f"{dp:6.2f}h" if dp != float("inf") else "   n/a"
    print(f"{a:%d/%m/%Y %H:%M} {c:>7} {(b-a).total_seconds()/3600:6.1f}h {dps} {nmax:4d}   "
          f"t={fr['t']:.2f} p={fr['p']:.2f} sp={fr['sp']:.2f} vb={fr['vb']:.2f}")

print()
print("=" * 108)
print("OS AMARELOS: que parada veio depois, e ela tinha alarme?")
print("=" * 108)
JAN = pd.Timedelta(hours=48)
for a, b, c, lead in classif:
    if c != "NEUTRO":
        continue
    cand = paradas[(paradas.ini >= a) & (paradas.ini <= b + JAN)]
    p0 = cand.iloc[0]
    hiato = (p0.ini - b).total_seconds() / 3600
    print(f"\n--- episodio {a:%d/%m/%Y %H:%M} .. {b:%d/%m/%Y %H:%M} ({(b-a).total_seconds()/3600:.1f}h)")
    print(f"    parada em {p0.ini:%d/%m/%Y %H:%M}, durou {p0.dur_h:.1f}h, hiato fim->parada = {hiato:.2f}h")
    jan_al = alarmes[(alarmes.ts >= p0.ini - pd.Timedelta(hours=6)) &
                     (alarmes.ts <= p0.ini + pd.Timedelta(hours=2))]
    if len(jan_al) == 0:
        print("    alarmes em [-6h, +2h] da parada: NENHUM  -> parada limpa (programada?)")
    else:
        print(f"    alarmes em [-6h, +2h] da parada: {len(jan_al)}")
        for _, r in jan_al.iterrows():
            dt = (r.ts - p0.ini).total_seconds() / 3600
            marca = "  <<< NIVEL" if r.nivel else ""
            print(f"      {dt:+6.2f}h  {r['Tag Alarme']:18s} {r['Descrição Alarme']}{marca}")

print()
print("=" * 108)
print("OS PRETOS: janela larga de alarme [-24h, +24h] e contexto de parada")
print("=" * 108)
for a, b, c, lead in classif:
    if c != "FP":
        continue
    print(f"\n--- episodio {a:%d/%m/%Y %H:%M} .. {b:%d/%m/%Y %H:%M} ({(b-a).total_seconds()/3600:.1f}h)")
    jan_al = alarmes[(alarmes.ts >= a - pd.Timedelta(hours=24)) &
                     (alarmes.ts <= b + pd.Timedelta(hours=24))]
    if len(jan_al) == 0:
        print("    alarmes em [-24h, +24h]: NENHUM")
    else:
        print(f"    alarmes em [-24h, +24h]: {len(jan_al)}")
        for _, r in jan_al.iterrows():
            dt = (r.ts - a).total_seconds() / 3600
            marca = "  <<< NIVEL" if r.nivel else ""
            print(f"      {dt:+7.2f}h do inicio  {r['Tag Alarme']:18s} {r['Descrição Alarme']}{marca}")
    # proxima parada real qualquer (sem limite de 48h) -- quao longe ficou?
    prox = paradas[paradas.ini >= b]
    if len(prox):
        d = (prox.iloc[0].ini - b).total_seconds() / 3600
        print(f"    proxima parada real >=2h: {prox.iloc[0].ini:%d/%m/%Y %H:%M} "
              f"({d:.1f}h depois, durou {prox.iloc[0].dur_h:.1f}h)")
    else:
        print("    proxima parada real >=2h: nenhuma ate o fim da serie")
