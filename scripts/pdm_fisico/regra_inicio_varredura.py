#!/usr/bin/env python3
"""Reotimizar o pos-processamento sob a REGRA DE INICIO -- a que o Diego e o
Francisco usam.

Medido em 04/09/2026: as duas reguas de associacao evento<->episodio diferem.
  nossa  ("de pe") : o alarme tem de estar ATIVO em algum instante de [t-48h, t]
  deles  ("inicio"): o INICIO do episodio tem de cair dentro de [t-48h, t]

Sob a regra deles o nosso 8/8 vira 4/8: em quatro eventos o alarme ja estava de
pe quando a janela abriu (leads de 143,9 / 194,9 / 51,8 / 670,0 h). Um alarme de
pe ha 28 dias nao e predicao, e uma condicao cronica -- o operador nao age nele.

Nunca otimizamos para essa regra. Pegamos 8/8 sob "de pe" e os quatro alarmes
permanentes vieram de brinde. Esta varredura pergunta: existe ponto do NOSSO
pos-processamento que faca bem sob a regra estrita? Se existir, e um resultado
mais forte que o publicado, porque nao depende de qual regua o avaliador usa.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import (partes, pos, mask, idx, alvo, mede,
                               KB, KV, REFRAT, DURMIN)
from publica_clearml import SIN, REFRAT_H, DUR_MIN

JAN = pd.Timedelta(hours=48)


def avalia_inicio(alerta, eventos=alvo, quente=mask):
    """Igual a AV.avalia, mas a deteccao exige que o INICIO do episodio caia na
    janela -- e o lead e medido do inicio real, sem censura em 48 h."""
    eps = AV.episodios(alerta)
    meses = float(quente.sum()) * 2 / 60.0 / 730.0
    jan = [(t - JAN, t) for t in eventos]
    det, leads = 0, []
    for t0, t1 in jan:
        dentro = [a for a, _ in eps if t0 <= a <= t1]
        if dentro:
            det += 1
            leads.append((t1 - max(dentro)).total_seconds() / 3600.0)
    fp = hfp = 0
    for a, b in eps:
        if not any(t0 <= a <= t1 for t0, t1 in jan):
            fp += 1
            hfp += (b - a).total_seconds() / 3600.0 + 2 / 60.0
    return dict(det=det, episodios=len(eps), fp=fp, fp_mes=fp / max(meses, 1e-9),
                h_fp_mes=hfp / max(meses, 1e-9),
                lead_med=float(np.mean(leads)) if leads else np.nan,
                lead_min=float(np.min(leads)) if leads else np.nan)


if __name__ == "__main__":
    P = {(kb, kv): partes(kb, kv) for kb in KB for kv in KV}

    # controle: o ponto de producao, nas duas reguas
    ON = P[(1.7, 2.2)]
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    al = pos(v, ns, REFRAT_H, DUR_MIN, False)
    a_pe = AV.avalia(al, alvo, mask); a_in = avalia_inicio(al)
    print("CONTROLE -- o ponto de producao nas duas reguas")
    print("=" * 78)
    for rot, m in [("de pe (nossa) ", a_pe), ("inicio (deles)", a_in)]:
        print(f"  {rot}: {m['det']}/8 · {m['fp_mes']:.3f} FP/mes · "
              f"{m['h_fp_mes']:.1f} h/mes · lead {m['lead_med']:.1f} h")
    print(f"  (esperado: 8/8 e 4/8)\n")

    lin = []
    for rf in REFRAT:
        for dm in DURMIN:
            for esc in (False, True):
                for (kb, kv), pr in P.items():
                    for mg in (False, True):
                        n_sin = sum(pr[c].astype(int) for c in SIN)
                        vv = pd.Series(n_sin >= 2, index=idx) & mask
                        if mg:
                            vv = vv & (pr["sp"] | pr["vb"])
                        a = pos(vv, n_sin, rf, dm, esc)
                        mi = avalia_inicio(a); mp = AV.avalia(a, alvo, mask)
                        lin.append(dict(refrat=rf, dur_min=dm, escalada=esc, mancal=mg,
                                        kb=kb, kv=kv,
                                        det_ini=mi["det"], det_pe=mp["det"],
                                        eps=mi["episodios"],
                                        fp_mes=round(mi["fp_mes"], 3),
                                        h_fp_mes=round(mi["h_fp_mes"], 1),
                                        lead=round(mi["lead_med"], 2) if mi["det"] else np.nan,
                                        lead_min=round(mi["lead_min"], 2) if mi["det"] else np.nan))
            print(f"  refrat={rf:>3d}h dur_min={dm:>3d}min ok", flush=True)

    d = pd.DataFrame(lin)
    d.to_csv("regra_inicio.csv", index=False)
    print(f"\n{len(d)} pontos -> regra_inicio.csv")

    print("\n" + "=" * 92)
    print("A FRONTEIRA SOB A REGRA DE INICIO")
    print("=" * 92)
    print(f"{'det':>5} {'n':>5} {'menor FP/mes':>14} {'h/mes':>8} {'lead':>8} "
          f"{'lead min':>9}  configuracao")
    print("-" * 92)
    for k in sorted(d.det_ini.unique(), reverse=True):
        s = d[d.det_ini == k].sort_values(["fp_mes", "h_fp_mes"])
        b = s.iloc[0]
        print(f"{k:4d}/8 {len(s):5d} {b.fp_mes:13.3f} {b.h_fp_mes:8.1f} "
              f"{b.lead:7.1f}h {b.lead_min:8.1f}h  refrat={b.refrat}h dur={b.dur_min}min "
              f"k={b.kb}/{b.kv} esc={b.escalada} mancal={b.mancal}")
