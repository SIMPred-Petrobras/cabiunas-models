#!/usr/bin/env python3
"""De onde vem a deriva de custo do detector daqui, sinal por sinal.

Fato a explicar (deriva.py): com parametros FIXOS, o duty do detector vai de 1,66% a
8,09% em cinco semestres e a duracao media do episodio cresce de forma perfeitamente
monotona (rho=+1,00, p=0,0167). O regime de operacao nao explica -- partidas/mes x
FP/mes da rho=-0,30, sinal trocado, e o pior semestre e o de MENOS partidas. E o stack
estatico do EXP7 nao apresenta essa deriva no mesmo periodo, o que a torna uma
propriedade da NOSSA construcao, nao da maquina.

Tres candidatos a causa, que este script separa:

  (a) A DISTRIBUICAO DO ESCORE DERIVOU. Os limiares sao absolutos (THR_FAM=2,0,
      THR_SPREAD=3,0, vezes k). Se a distribuicao do escore alarga com o tempo, um corte
      que valia p99,5 em 2024 passa a valer p98 em 2026 e o detector alarma mais sem que
      nada tenha mudado no codigo. Mede-se o QUANTIL IMPLICADO do limiar fixo por
      semestre: se ele cair, e esta a causa, e a correcao e limiar por alvo de duty.

  (b) A REFERENCIA ENCOLHEU. z_rolante apaga +-7 dias em torno de cada falha JA OCORRIDA
      da serie que vira referencia. Em 2024 havia 1 falha para apagar; em 2026 ha 9.
      Se a referencia efetiva encolheu, o MAD fica mais ruidoso e o z infla.

  (c) A MAQUINA DEGRADOU DE VERDADE. Nesse caso o escore sobe mas o quantil implicado
      cai junto -- indistinguivel de (a) so por essa medicao. O que separa: se a subida
      estiver concentrada em UM sinal e em UM canal fisico, e degradacao; se estiver
      espalhada por todos os quatro, e escala.

Depois do diagnostico, testa a correcao proposta: limiar recalibrado mes a mes para
manter um DUTY ALVO fixo (desenho CFAR), em vez de k fixo. Nao melhora deteccao por
construcao -- o que se quer saber e se estabiliza o custo sem perder eventos.
"""
from __future__ import annotations
import sys, itertools
import numpy as np, pandas as pd
from scipy import stats

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO
from portoes import K_BASE, K_VIB
from auto_reset import trunca

SINAIS = [("t", "1h", DET.THR_FAM * K_BASE), ("p", "1h", DET.THR_FAM * K_BASE),
          ("sp", "30min", DET.THR_SPREAD * K_BASE), ("vb", "30min", 3.0 * K_VIB)]
DUTY_ALVO = [0.02, 0.03, 0.05, 0.08, 0.12]


def p_exato(y):
    n = len(y); r0 = stats.spearmanr(np.arange(n), y).statistic
    t = [abs(stats.spearmanr(np.arange(n), [y[i] for i in pm]).statistic)
         for pm in itertools.permutations(range(n))]
    return float(np.mean(np.array(t) >= abs(r0) - 1e-12)), r0


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    mask = mascara_pontuacao(df)

    print("montando 'out' ...", flush=True)
    out = roda(BRACO, df, falhas)
    ew = {c: out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
          for c, hl, _ in SINAIS}

    sems = [g.index[0] for _, g in pd.Series(idx, index=idx).groupby(pd.Grouper(freq="2QS"))
            if len(g) and (mask & (idx >= g.index[0]) & (idx <= g.index[-1])).sum() * 2 / 60 >= 300]
    janelas = [((idx >= a) & (idx <= (sems[i + 1] if i + 1 < len(sems) else idx[-1] + pd.Timedelta("2min"))))
               for i, a in enumerate(sems)]

    # ---------------- (a) quantil implicado do limiar fixo
    print("\n=== (a) QUANTIL IMPLICADO DO LIMIAR FIXO, por semestre ===")
    print("    se cai, o limiar absoluto ficou relativamente mais frouxo => causa e escala\n")
    print(f"{'sinal':>6} {'limiar':>7} | " + " ".join(f"{a.date()!s:>12}" for a in sems))
    quantis = {}
    for c, hl, thr in SINAIS:
        qs = []
        for sel in janelas:
            v = ew[c][mask & sel].dropna()
            qs.append(100 * (v < thr).mean() if len(v) else np.nan)
        quantis[c] = qs
        print(f"{c:>6} {thr:7.2f} | " + " ".join(f"{q:12.3f}" for q in qs))
    print("\n    (numero = percentil em que o limiar fixo caiu naquele semestre)")

    print(f"\n{'sinal':>6} | " + " ".join(f"{a.date()!s:>10}" for a in sems) + "   tendencia do duty")
    for c, hl, thr in SINAIS:
        duty = []
        for sel in janelas:
            s = DET._sustained(ew[c], thr) & mask & sel
            duty.append(100 * s.sum() / max((mask & sel).sum(), 1))
        p, r = p_exato(duty)
        print(f"{c:>6} | " + " ".join(f"{d:10.2f}" for d in duty) + f"   rho={r:+.2f} p={p:.4f}")
    print("    (duty% = fracao do tempo pontuavel com o sinal sustentado acima do limiar)")

    # ---------------- mediana e dispersao do escore
    print(f"\n=== escala do escore por semestre (mediana / p99 do escore EWMA) ===")
    print(f"{'sinal':>6} | " + " ".join(f"{a.date()!s:>14}" for a in sems))
    for c, hl, thr in SINAIS:
        cel = []
        for sel in janelas:
            v = ew[c][mask & sel].dropna()
            cel.append(f"{v.median():5.2f}/{v.quantile(.99):6.2f}" if len(v) else "   -/     -")
        print(f"{c:>6} | " + " ".join(f"{x:>14}" for x in cel))

    # ---------------- (b) referencia efetiva
    print("\n=== (b) quanto da serie e apagado da referencia por falhas ja ocorridas ===")
    for i, a in enumerate(sems):
        sel = janelas[i]
        ja = falhas[falhas <= idx[sel][-1]]
        apagado = np.zeros(sel.sum(), dtype=bool)
        ts = idx[sel]
        for f in ja:
            apagado |= ((ts >= f - pd.Timedelta(days=7)) & (ts <= f + pd.Timedelta(days=2)))
        print(f"  {a.date()}: {len(ja)} falhas ja ocorridas, {100*apagado.mean():5.2f}% do semestre "
              f"fora da referencia")

    # ---------------- correcao: limiar por alvo de duty (CFAR)
    print("\n=== CORRECAO: limiar recalibrado por mes para manter um DUTY ALVO ===")
    print("    (o limiar de cada sinal vira o quantil (1-alvo) do proprio sinal nos ultimos")
    print("     60 dias de operacao pontuavel; voto >= 2 e sustentacao de 30 min inalterados)\n")
    meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
    linhas = []
    print(f"{'duty alvo':>10} {'det':>4} {'eps':>5} {'FP/mes':>7} {'h/mes':>7} {'p':>8} | "
          f"duty por semestre (%)")
    for alvo in DUTY_ALVO:
        n = pd.Series(0, index=idx)
        for c, hl, _ in SINAIS:
            acima = pd.Series(False, index=idx)
            for i, m0 in enumerate(meses):
                m1 = meses[i + 1] if i + 1 < len(meses) else idx[-1] + pd.Timedelta("2min")
                hist = ew[c][(idx >= m0 - pd.Timedelta(days=60)) & (idx < m0) & mask].dropna()
                sel = (idx >= m0) & (idx < m1)
                if len(hist) < 5000 or not sel.any():
                    continue
                lim = float(hist.quantile(1 - alvo))
                acima.loc[sel] = (ew[c][sel] > lim).fillna(False).to_numpy()
            n = n + DET._sustained(acima, 0.5).astype(int) * 0 + A.sustenta(acima, 30).astype(int)
        al = (n >= 2) & mask
        x = A.avalia(al, falhas, mask); x.update(A.permuta(al, mask, x["det"], len(falhas)))
        ds = []
        for sel in janelas:
            eps = A.episodios(al & sel)
            jw = [(t - pd.Timedelta(hours=48), t) for t in falhas]
            fp = [(a, b) for a, b in eps if not any((a <= t1) and (b >= t0) for t0, t1 in jw)]
            h = sum((b - a).total_seconds() / 3600 + 2/60 for a, b in fp)
            ho = (mask & sel).sum() * 2 / 60
            ds.append(100 * h / max(ho, 1))
        p, r = p_exato(ds)
        print(f"{100*alvo:9.0f}% {x['det']:4d} {x['episodios']:5d} {x['fp_mes']:7.2f} "
              f"{x['h_fp_mes']:7.1f} {x['p']:8.4f} | " + " ".join(f"{d:5.2f}" for d in ds)
              + f"  rho={r:+.2f} p={p:.3f}")
        linhas.append(dict(duty_alvo=alvo, det=x["det"], eps=x["episodios"], fp_mes=x["fp_mes"],
                           h_mes=x["h_fp_mes"], p=x["p"], rho_duty=r, p_rho=p,
                           duty_sem=",".join(f"{d:.2f}" for d in ds),
                           quais=",".join(x["detectados"])))

    base = alerta_2k(out, mask, K_BASE, K_VIB)
    xb = A.avalia(base, falhas, mask); xb.update(A.permuta(base, mask, xb["det"], len(falhas)))
    dsb = []
    for sel in janelas:
        eps = A.episodios(base & sel)
        jw = [(t - pd.Timedelta(hours=48), t) for t in falhas]
        fp = [(a, b) for a, b in eps if not any((a <= t1) and (b >= t0) for t0, t1 in jw)]
        h = sum((b - a).total_seconds() / 3600 + 2/60 for a, b in fp)
        dsb.append(100 * h / max((mask & sel).sum() * 2 / 60, 1))
    pb, rb = p_exato(dsb)
    print(f"{'k fixo':>10} {xb['det']:4d} {xb['episodios']:5d} {xb['fp_mes']:7.2f} "
          f"{xb['h_fp_mes']:7.1f} {xb['p']:8.4f} | " + " ".join(f"{d:5.2f}" for d in dsb)
          + f"  rho={rb:+.2f} p={pb:.3f}")
    pd.DataFrame(linhas).to_csv("deriva_origem.csv", index=False)


main()
