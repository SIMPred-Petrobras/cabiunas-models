#!/usr/bin/env python3
"""Validacao completa do CUSUM: limiar por sinal, teto de duracao, permutacao, plato, LOEO.

O primeiro teste (cusum.py) mostrou o maior ganho isolado desde o refratario:
    atual (EWMA+limiar+sustentacao)  8/8  2,41 FP/mes   48,2 h/mes  lead 19,7 h
    CUSUM kappa=0,75 h=20            8/8  1,89 FP/mes   73,2 h/mes  lead 30,7 h
-22% de falso positivo e +56% de lead, com o mesmo recall. E o plato em h e suave.

Duas pendencias que este script resolve:
  1. o custo em HORAS subiu (48,2 -> 73,2). O acumulador demora a zerar depois que sobe,
     entao os episodios ficam longos -- e o mesmo problema que o teto de 12 h ja resolveu
     para a regra atual. Aqui o teto entra na varredura.
  2. `h` compartilhado entre os quatro sinais e uma amarra arbitraria. Os sinais tem
     forcas muito diferentes (vb sustenta em 8/9 eventos, sp em 4/9), entao `h` por sinal
     deveria ser melhor -- e foi soltar `k` por sinal que deu o ultimo ganho.

Protocolo: o mesmo que aprovou o refratario e reprovou piso, escape e voto entre sondas --
episodios igualados, permutacao, PLATO e LOEO aninhado. Janela 2025-01 a 2026-04, 8 eventos.
"""
from __future__ import annotations
import sys, itertools
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import BRACO
from auto_reset import trunca
import reduz_fp as RF

T0 = pd.Timestamp("2025-01-01", tz="UTC")
HL = {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min"}
BASE = {"t": DET.THR_FAM, "p": DET.THR_FAM, "sp": DET.THR_SPREAD, "vb": 3.0}
K_ATUAL = {"t": 1.7, "p": 1.7, "sp": 1.7, "vb": 2.2}
KAPPAS = [0.5, 0.75]
HG = [10, 20, 40, 80]
TETOS = [0, 12, 24]
SINAIS = ["t", "p", "sp", "vb"]
ORC = 2.6


def cusum_bool(z, kappa, h, reset):
    x = (z - kappa).fillna(0.0).to_numpy()
    r = reset.to_numpy(); n = len(x)
    S = np.empty(n); acc = 0.0
    for i in range(n):
        acc = 0.0 if r[i] else max(0.0, acc + x[i])
        S[i] = acc
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
    Z = {c: (E[c] / (BASE[c] * K_ATUAL[c])).clip(upper=20) for c in E}

    print("pre-calculando CUSUM (4 sinais x 2 kappa x 4 h) ...", flush=True)
    D = {}
    for c in SINAIS:
        for ka in KAPPAS:
            for h in HG:
                D[(c, ka, h)] = cusum_bool(Z[c], ka, h, reset)
    print(f"  {len(D)} acumuladores prontos\n", flush=True)

    mv = mask.values
    L, ALS = [], {}
    combos = list(itertools.product(KAPPAS, HG, HG, HG, HG, TETOS))
    for n_, (ka, ht, hp, hsp, hvb, te) in enumerate(combos):
        n = (D[("t", ka, ht)].astype(np.int8) + D[("p", ka, hp)].astype(np.int8)
             + D[("sp", ka, hsp)].astype(np.int8) + D[("vb", ka, hvb)].astype(np.int8))
        al = pd.Series((n >= 2) & mv, index=idx)
        al = RF.dur_min(RF.refratario(al, 48), 60)
        if te: al = trunca(al, te)
        y = A.avalia(al[sel], alvo, m2)
        key = (ka, ht, hp, hsp, hvb, te)
        ALS[key] = al
        L.append(dict(kappa=ka, ht=ht, hp=hp, hsp=hsp, hvb=hvb, teto=te, det=y["det"],
                      eps=y["episodios"], fp=y["fp_mes"], h=y["h_fp_mes"], lead=y["lead_med"],
                      quais=",".join(y["detectados"])))
        if (n_ + 1) % 200 == 0:
            print(f"  {n_+1}/{len(combos)}", flush=True)
    T = pd.DataFrame(L); T.to_csv("cusum_valida.csv", index=False)

    # plato: vizinhos na grade
    eixos = [KAPPAS, HG, HG, HG, HG, TETOS]
    pos = {}
    for i, r in T.iterrows():
        pos[(KAPPAS.index(r.kappa), HG.index(r.ht), HG.index(r.hp),
             HG.index(r.hsp), HG.index(r.hvb), TETOS.index(r.teto))] = i
    det = T.det.values
    plat = np.zeros(len(T))
    for key, i in pos.items():
        v = []
        for e in range(6):
            for d in (-1, 1):
                kk = list(key); kk[e] += d
                if 0 <= kk[e] < len(eixos[e]): v.append(pos[tuple(kk)])
        plat[i] = (det[v] >= det[i]).mean() if v else 0.0
    T["plato"] = plat; T.to_csv("cusum_valida.csv", index=False)

    print("\n" + "=" * 104)
    print(f"MELHORES (orcamento <= {ORC} FP/mes; criterio deteccao > plato > lead)")
    print("=" * 104)
    S = T[T.fp <= ORC].sort_values(["det", "plato", "lead"], ascending=[False, False, False])
    print(f"{'kappa':>6} {'h_t':>5} {'h_p':>5} {'h_sp':>5} {'h_vb':>5} {'teto':>5} {'det':>6} "
          f"{'eps':>5} {'FP/mes':>7} {'h/mes':>7} {'lead':>6} {'plato':>6}")
    for _, r in S.head(12).iterrows():
        print(f"{r.kappa:6.2f} {int(r.ht):5d} {int(r.hp):5d} {int(r.hsp):5d} {int(r.hvb):5d} "
              f"{int(r.teto):4d}h {int(r.det):4d}/8 {int(r.eps):5d} {r.fp:7.2f} {r.h:7.1f} "
              f"{r.lead:6.1f} {r.plato:6.2f}")

    print("\n" + "=" * 104); print("PERMUTACAO nos tres melhores"); print("=" * 104)
    for _, r in S.head(3).iterrows():
        key = (r.kappa, int(r.ht), int(r.hp), int(r.hsp), int(r.hvb), int(r.teto))
        al = ALS[key]; y = A.avalia(al[sel], alvo, m2)
        p = A.permuta(al[sel], m2, y["det"], len(alvo))["p"]
        print(f"  kappa={r.kappa} h=({int(r.ht)},{int(r.hp)},{int(r.hsp)},{int(r.hvb)}) "
              f"teto={int(r.teto)}h: {y['det']}/8  {y['fp_mes']:.2f} FP/mes  "
              f"{y['h_fp_mes']:.1f} h/mes  lead {y['lead_med']:.1f} h  p={p:.4f}", flush=True)

    print("\n" + "=" * 104); print("LOEO ANINHADO (criterio deteccao + plato + lead)")
    print("=" * 104)
    QQ = [set(str(q).split(",")) for q in T.quais]
    DM = np.array([[a in QQ[q] for a in alvo_s] for q in range(len(T))])
    fp = T.fp.values; lead = np.nan_to_num(T.lead.values, nan=0.0)
    for orc in [2.0, 2.3, 2.6, 3.0]:
        ac = 0
        for i in range(len(alvo)):
            outros = [j for j in range(len(alvo)) if j != i]
            det_tr = DM[:, outros].sum(axis=1)
            pont = np.where(fp <= orc, det_tr*1000.0 + plat*10.0 + lead, -np.inf)
            ac += bool(DM[int(np.argmax(pont)), i])
        print(f"  orcamento <= {orc} FP/mes: LOEO {ac}/8", flush=True)

    ref = T[(T.kappa == 0.75) & (T.ht == 20) & (T.hp == 20) & (T.hsp == 20)
            & (T.hvb == 20) & (T.teto == 0)]
    if len(ref):
        r = ref.iloc[0]
        print(f"\n  h compartilhado (o do primeiro teste): {int(r.det)}/8  {r.fp:.2f} FP/mes  "
              f"{r.h:.1f} h/mes  lead {r.lead:.1f} h  plato {r.plato:.2f}")


if __name__ == "__main__":
    main()
