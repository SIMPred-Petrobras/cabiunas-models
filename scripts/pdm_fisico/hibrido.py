#!/usr/bin/env python3
"""Hibrido: CUSUM como canal LENTO + regra de degrau como confirmacao.

Onde chegamos. O CUSUM (acumula evidencia) da +60% de lead e -18% de falso positivo com
o mesmo 8/8, mas custa +52% de horas. A regra atual (EWMA -> limiar -> sustentacao) e
barata em horas e detecta tarde. As duas medem coisas diferentes:

    EWMA+limiar : "o sinal esta alto AGORA?"           -> detector de degrau
    CUSUM       : "quanta evidencia acumulou desde o normal?" -> detector de deriva

A logica que ja funciona neste projeto e votar entre coisas que erram de formas
diferentes -- foi assim com as quatro familias fisicas. Aqui a aposta e votar entre
ESCALAS DE TEMPO. Seis formas de combinar, e a grade decide:

  A  so degrau      n = #EWMA disparados            >= 2      (a regra atual)
  B  so CUSUM       n = #CUSUM disparados           >= 2
  C  por sinal      n = #(EWMA OU CUSUM) por sinal  >= V      (o sinal conta se qualquer um fala)
  D  oito votantes  n = #EWMA + #CUSUM              >= V      (4+4 detectores independentes)
  E  uniao de regras  (#EWMA>=2) OU (#CUSUM>=2)
  F  CUSUM confirmado (#CUSUM>=2) E (#EWMA>=1)               (deriva confirmada por nivel)

Criterio de selecao no LOEO: MEDIA sobre a vizinhanca de grade -- descoberto ao testar o
`max` sugerido: media 8/8 em todos os orcamentos, plato 8/8 em seis de oito, proprio 6/8,
min 6/8, max 6/8. A media regulariza; o max escolhe o pico de ruido da vizinhanca.
"""
from __future__ import annotations
import sys, itertools
import numpy as np, pandas as pd

PDM = ("/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-"
       "dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/"
       "scratchpad/pdm/src")
sys.path.insert(0, PDM)
from cabiunas_pdm import detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import BRACO
from auto_reset import trunca
import reduz_fp as RF

T0 = pd.Timestamp("2025-01-01", tz="UTC")
HL = {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min"}
BASE = {"t": DET.THR_FAM, "p": DET.THR_FAM, "sp": DET.THR_SPREAD, "vb": 3.0}
SIN = ["t", "p", "sp", "vb"]
KSETS = {"padrao": {"t":1.7,"p":1.7,"sp":1.7,"vb":2.2},
         "rico":   {"t":1.7,"p":3.0,"sp":1.2,"vb":2.2}}
KAPPAS = [0.5, 0.75]
HS = [10, 20, 40]
TETOS = [0, 12, 24]
VARS = ["A_degrau", "B_cusum", "C_por_sinal", "D_oito", "E_uniao", "F_confirmado"]
VOTOS = {"C_por_sinal": [2, 3], "D_oito": [2, 3, 4]}


def cusum_bool(z, kappa, h, reset):
    x = (z - kappa).fillna(0.0).to_numpy(); r = reset.to_numpy()
    S = np.empty(len(x)); acc = 0.0
    for i in range(len(x)):
        acc = 0.0 if r[i] else max(0.0, acc + x[i]); S[i] = acc
    return S > h


def main():
    df = canonico(); idx = df.index
    todas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    sel = (idx >= T0); mask = mascara_pontuacao(df) & sel
    alvo = list(todas[todas >= T0]); m2 = mask[sel]
    alvo_s = [f"{t:%Y-%m-%d}" for t in alvo]
    op = df["in_operation"].astype(bool)
    reset = (~mask) | (op & ~op.shift(fill_value=False))
    out = roda(BRACO, df, todas)
    E = {c: out[c].ewm(halflife=pd.Timedelta(hh), times=idx).mean().where(mask) for c, hh in HL.items()}

    print("pre-calculando ...", flush=True)
    EW = {(ks, c): DET._sustained(E[c], BASE[c]*K[c]).values for ks, K in KSETS.items() for c in SIN}
    CU = {}
    for ks, K in KSETS.items():
        Z = {c: (E[c] / (BASE[c]*K[c])).clip(upper=20) for c in SIN}
        for c in SIN:
            for ka in KAPPAS:
                for h in HS:
                    CU[(ks, c, ka, h)] = cusum_bool(Z[c], ka, h, reset)
    print(f"  {len(EW)} regras de degrau, {len(CU)} acumuladores\n", flush=True)

    mv = mask.values
    L = []
    for ks in KSETS:
        for ka in KAPPAS:
            for h in HS:
                ew = np.array([EW[(ks, c)] for c in SIN])
                cu = np.array([CU[(ks, c, ka, h)] for c in SIN])
                n_ew = ew.sum(axis=0); n_cu = cu.sum(axis=0)
                for var in VARS:
                    for v in VARS_V(var):
                        if var == "A_degrau":      base = n_ew >= 2
                        elif var == "B_cusum":     base = n_cu >= 2
                        elif var == "C_por_sinal": base = (ew | cu).sum(axis=0) >= v
                        elif var == "D_oito":      base = (n_ew + n_cu) >= v
                        elif var == "E_uniao":     base = (n_ew >= 2) | (n_cu >= 2)
                        else:                      base = (n_cu >= 2) & (n_ew >= 1)
                        al0 = pd.Series(base & mv, index=idx)
                        al0 = RF.dur_min(RF.refratario(al0, 48), 60)
                        for te in TETOS:
                            al = trunca(al0, te) if te else al0
                            y = A.avalia(al[sel], alvo, m2)
                            L.append(dict(kset=ks, kappa=ka, h=h, var=var, voto=v, teto=te,
                                          det=y["det"], eps=y["episodios"], fp=y["fp_mes"],
                                          hm=y["h_fp_mes"], lead=y["lead_med"],
                                          quais=",".join(y["detectados"])))
        print(f"  kset={ks} varrido ({len(L)} configs)", flush=True)
    T = pd.DataFrame(L); T.to_csv("hibrido.csv", index=False)

    ref = T[(T.kset=="padrao")&(T["var"]=="A_degrau")&(T.teto==0)&(T.kappa==0.5)&(T.h==10)].iloc[0]
    print("\n" + "=" * 104)
    print(f"REFERENCIA (regra atual): {int(ref.det)}/8  {int(ref.eps)} eps  {ref.fp:.2f} FP/mes  "
          f"{ref.hm:.1f} h/mes  lead {ref.lead:.1f} h")
    print("=" * 104)
    print(f"\n{'variante':>14} {'kset':>7} {'kappa':>6} {'h':>4} {'voto':>5} {'teto':>5} {'det':>6} "
          f"{'eps':>5} {'FP/mes':>7} {'h/mes':>7} {'lead':>6}")
    S = T[(T.fp <= 2.6) & (T.det == 8)].sort_values(["hm", "fp"])
    for _, r in S.head(14).iterrows():
        print(f"{r['var']:>14} {r.kset:>7} {r.kappa:6.2f} {int(r.h):4d} {int(r.voto):5d} "
              f"{int(r.teto):4d}h {int(r.det):4d}/8 {int(r.eps):5d} {r.fp:7.2f} {r.hm:7.1f} {r.lead:6.1f}")

    print("\n" + "=" * 104); print("melhor de cada variante (8/8, FP<=2,6, menor h/mes)")
    print("=" * 104)
    for var in VARS:
        g = T[(T["var"] == var) & (T.det == 8) & (T.fp <= 2.6)].sort_values("hm")
        if not len(g):
            print(f"  {var:>14}: nenhuma configuracao com 8/8 e FP<=2,6"); continue
        r = g.iloc[0]
        print(f"  {var:>14}: {int(r.det)}/8  {r.fp:.2f} FP/mes  {r.hm:6.1f} h/mes  "
              f"lead {r.lead:5.1f} h   (kset={r.kset} kappa={r.kappa} h={int(r.h)} "
              f"voto={int(r.voto)} teto={int(r.teto)}h)")

    print("\n" + "=" * 104); print("LOEO ANINHADO, criterio MEDIA sobre a vizinhanca")
    print("=" * 104)
    QQ = [set(str(q).split(",")) for q in T.quais]
    D = np.array([[a in QQ[q] for a in alvo_s] for q in range(len(T))])
    fp = T.fp.values; lead = np.nan_to_num(T.lead.values, nan=0.0)
    # vizinhanca: eixos ordenados (kappa, h, teto) dentro do mesmo (kset, var, voto)
    key = list(zip(T.kset, T["var"], T.voto))
    ord_ = {(KAPPAS.index(r.kappa), HS.index(r.h), TETOS.index(r.teto)): i for i, r in T.iterrows()}
    VIZ = []
    for i, r in T.iterrows():
        c = (KAPPAS.index(r.kappa), HS.index(r.h), TETOS.index(r.teto))
        v = []
        for e, ax in enumerate([KAPPAS, HS, TETOS]):
            for d in (-1, 1):
                cc = list(c); cc[e] += d
                if 0 <= cc[e] < len(ax):
                    cand = T[(T.kset==r.kset)&(T["var"]==r["var"])&(T.voto==r.voto)
                             &(T.kappa==KAPPAS[cc[0]])&(T.h==HS[cc[1]])&(T.teto==TETOS[cc[2]])]
                    if len(cand): v.append(cand.index[0])
        VIZ.append(np.array(v, dtype=int))
    for orc in [2.0, 2.3, 2.6]:
        ac = 0
        for i in range(len(alvo)):
            outros = [j for j in range(len(alvo)) if j != i]
            dtr = D[:, outros].sum(axis=1).astype(float)
            s = np.array([np.mean(np.r_[dtr[q], dtr[VIZ[q]]]) if len(VIZ[q]) else dtr[q]
                          for q in range(len(T))])
            pont = np.where(fp <= orc, s*1000.0 + lead, -np.inf)
            b = int(np.argmax(pont)); ac += bool(D[b, i])
        print(f"  orcamento <= {orc} FP/mes: LOEO {ac}/8   escolha: {T.iloc[b]['var']} "
              f"kset={T.iloc[b].kset} kappa={T.iloc[b].kappa} h={int(T.iloc[b].h)} "
              f"teto={int(T.iloc[b].teto)}h", flush=True)


def VARS_V(var):
    return VOTOS.get(var, [2])


if __name__ == "__main__":
    main()
