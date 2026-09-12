#!/usr/bin/env python3
"""O ensemble sob a mascara DELE -- a checagem de justica.

A nossa mascara exclui o blackout de 6 h pos-partida, T5<=300 e t<T0. Isso e uma
escolha do NOSSO detector; o dele nao tem essa exclusao. Aplica-la a ele apaga 12
dos 42 FP dele, corta 31% das horas e custa uma deteccao (8/8 -> 7/8).

Aqui os dois sao avaliados sob a mascara DELE (operational_state != off), que e a
mais neutra disponivel -- nao favorece o nosso desenho. Se o 6/8 do ensemble
sobreviver, a conclusao vale; se nao, era artefato da mascara.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import mask, idx, alvo
from publica_clearml import REFRAT_H, DUR_MIN
from plota_estilo_francisco import alarme, paradas_reais_2h

DIEGO = "/home/thallys/Documents/projeto-petrobras/wt-diego/canais/merged.parquet"
TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
paradas = paradas_reais_2h()

m = pd.read_parquet(DIEGO); m.index = m.index.tz_localize("UTC")
est = m["operational_state"].astype(str)
op_d = (~est.str.startswith("off")).resample("2min").max().reindex(idx, fill_value=False)
op_d = op_d.fillna(False).astype(bool)

# serie dele, na grade de 2 min
ns = sum(m[c].astype(int) for c in ("temperatura", "vibracao", "oleo", "alarme"))
v = (ns >= 2); g = (~v).cumsum()[v]
vf = v.copy(); vf[v] = v[v].groupby(g).transform("size") * 0.5 >= 45.0
al_d = pd.Series(False, index=m.index); bloq = None
for a, b in AV.episodios(vf, gap_h=2.0):
    if bloq is not None and a <= bloq: continue
    al_d.loc[a:b] = True; bloq = b + pd.Timedelta(hours=48)
dele = al_d.resample("2min").max().reindex(idx, fill_value=False).fillna(False).astype(bool)
nosso = alarme()

# so a interseccao temporal das duas janelas, para nao dar vantagem a ninguem
jan_ok = (idx >= max(idx[0], m.index[0])) & (idx <= min(idx[-1], m.index[-1]))
alvo_c = alvo[(alvo >= m.index[0]) & (alvo <= m.index[-1])]


def mede(al, quente, rot):
    al = al & quente & jan_ok
    eps = AV.episodios(al)
    meses = float(quente[jan_ok].sum())*2/60.0/730.0
    banda, leads, ini = 0, [], 0
    for t in alvo_c:
        c = [a for a, _ in eps if t - JAN <= a <= t - pd.Timedelta(hours=TMIN)]
        if c: banda += 1; leads.append((t - max(c)).total_seconds()/3600)
        if any(t - JAN <= a <= t for a, _ in eps): ini += 1
    jw = [(t - JAN, t) for t in alvo_c]
    fp = hfp = 0
    for a, b in eps:
        if any(a <= t1 and b >= t0 for t0, t1 in jw): continue
        cand = paradas[(paradas.ini >= a) & (paradas.ini <= b + JAN)]
        if len(cand): continue                       # regra C
        fp += 1; hfp += (b-a).total_seconds()/3600
    return dict(rot=rot, banda=banda, ini=ini, fp=fp/meses, h=hfp/meses,
                lead=np.mean(leads) if leads else np.nan, eps=len(eps), meses=meses)


for nome, quente in (("MASCARA NOSSA (blackout 6h, T5>300)", mask),
                     ("MASCARA DELE (operational_state)", op_d)):
    print(f"\n{nome}")
    print("=" * 96)
    print(f"{'':<26}{'banda':>8}{'inicio':>8}{'FP/mes':>10}{'h/mes':>9}{'lead':>9}{'eps':>6}")
    print("-" * 96)
    rn, rd = mede(nosso, quente, "nosso"), mede(dele, quente, "dele")
    for r in (rn, rd):
        print(f"{r['rot']:<26}{r['banda']:6d}/{len(alvo_c)}{r['ini']:7d}/{len(alvo_c)}"
              f"{r['fp']:10.3f}{r['h']:9.1f}"
              f"{(r['lead'] if np.isfinite(r['lead']) else 0):8.1f}h{r['eps']:6d}")
    # uniao no nivel de EVENTO
    en = AV.episodios(nosso & quente & jan_ok); ed = AV.episodios(dele & quente & jan_ok)
    def b_(eps, t):
        i = [a for a, _ in eps if t - JAN <= a <= t - pd.Timedelta(hours=TMIN)]
        return (t - max(i)).total_seconds()/3600 if i else None
    cob = {t: (b_(en, t), b_(ed, t)) for t in alvo_c}
    uni = sum(1 for x in cob.values() if x[0] is not None or x[1] is not None)
    lu = [min(y for y in x if y is not None) for x in cob.values() if any(y is not None for y in x)]
    print(f"{'AS DUAS em paralelo':<26}{uni:6d}/{len(alvo_c)}{'--':>8}"
          f"{rn['fp']+rd['fp']:10.3f}{rn['h']+rd['h']:9.1f}{np.mean(lu):8.1f}h")
    print(f"  meses de operacao: {rn['meses']:.2f}")
    print(f"  so o nosso: {', '.join(t.strftime('%d/%m') for t, x in cob.items() if x[0] is not None and x[1] is None) or '--'}")
    print(f"  so o dele : {', '.join(t.strftime('%d/%m') for t, x in cob.items() if x[1] is not None and x[0] is None) or '--'}")
    print(f"  nenhum    : {', '.join(t.strftime('%d/%m') for t, x in cob.items() if x[0] is None and x[1] is None) or '--'}")


# ---------------------------------------------------------------------------
# A medida operacionalmente honesta: cada detector sob as SUAS premissas. O nosso
# foi desenhado e calibrado COM o blackout de 6 h; o dele, SEM. Aplicar a mascara
# de um ao outro distorce nos dois sentidos.
print("\n\nCADA UM SOB AS SUAS PREMISSAS  (o nosso com blackout, o dele sem)")
print("=" * 96)
rn = mede(nosso, mask, "nosso (mascara nossa)")
rd = mede(dele, op_d, "dele  (mascara dele)")
print(f"{'':<26}{'banda':>8}{'inicio':>8}{'FP/mes':>10}{'h/mes':>9}{'lead':>9}{'eps':>6}")
print("-" * 96)
for r in (rn, rd):
    print(f"{r['rot']:<26}{r['banda']:6d}/{len(alvo_c)}{r['ini']:7d}/{len(alvo_c)}"
          f"{r['fp']:10.3f}{r['h']:9.1f}"
          f"{(r['lead'] if np.isfinite(r['lead']) else 0):8.1f}h{r['eps']:6d}")
en = AV.episodios(nosso & mask & jan_ok); ed = AV.episodios(dele & op_d & jan_ok)
def b2(eps, t):
    i = [a for a, _ in eps if t - JAN <= a <= t - pd.Timedelta(hours=TMIN)]
    return (t - max(i)).total_seconds()/3600 if i else None
cob = {t: (b2(en, t), b2(ed, t)) for t in alvo_c}
uni = sum(1 for x in cob.values() if x[0] is not None or x[1] is not None)
lu = [min(y for y in x if y is not None) for x in cob.values() if any(y is not None for y in x)]
print(f"{'AS DUAS em paralelo':<26}{uni:6d}/{len(alvo_c)}{'--':>8}"
      f"{rn['fp']+rd['fp']:10.3f}{rn['h']+rd['h']:9.1f}{np.mean(lu):8.1f}h")
print("-" * 96)
print(f"  so o nosso acrescenta: "
      f"{', '.join(t.strftime('%d/%m') for t, x in cob.items() if x[0] is not None and x[1] is None) or 'NADA'}")
print(f"  so o dele acrescenta : "
      f"{', '.join(t.strftime('%d/%m') for t, x in cob.items() if x[1] is not None and x[0] is None) or 'nada'}")
print(f"  nenhum dos dois      : "
      f"{', '.join(t.strftime('%d/%m') for t, x in cob.items() if x[0] is None and x[1] is None) or '--'}")
print(f"\n  -> o ensemble vale {uni}/{len(alvo_c)} contra {rd['banda']}/{len(alvo_c)} dele sozinho: "
      f"ganho de {uni - rd['banda']} evento(s) por +{rn['fp']:.3f} FP/mes")
