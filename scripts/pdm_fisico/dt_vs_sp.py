#!/usr/bin/env python3
"""O sinal fisico dT = TI_0305 - TI_0325 substitui o `sp` (z do spread do mancal)?

residuo_fisico.py estabeleceu tres coisas:
  - nao ha modelo a ajustar: dT nao depende da carga (Spearman -0,007) nem da pressao de
    oleo (+0,088), entao f(carga,P) degenera numa constante e o "residuo fisico" e
    simplesmente dT menos um offset fixo. Sem funcao ajustada, sem referencia movel;
  - o residuo nao tem tendencia por semestre (rho=-0,10, p=0,873), ao contrario do custo
    do detector atual (rho=+0,36, p=0,006);
  - ele separa 5 de 5 eventos de MANCAL com pico de 43 a 56 degC contra p99 saudavel de
    14,6 degC, e fica mudo em selagem e oleo -- exatamente o esperado da fisica.

O ponto fraco medido: e espicaçado (0,1 a 6,1 h acima do p99), e a sustentacao de 30 min
pode nao passar. Este script mede isso.

Os dois sinais competem pelo MESMO papel -- detectar o mancal -- mas com referencias de
naturezas opostas:
    sp  = |TI_0305 - mediana(TI_0301,0303,0307)| / MAD movel     z contra historia recente
    dT  =  TI_0305 - TI_0325 - 9,0                               graus contra constante fixa

Bracos, todos com o ponto de operacao (refratario 48 h + duracao minima 60 min), a
episodios igualados, com permutacao e deriva por campanha:
  ATUAL      t, p, sp, vb
  TROCA      t, p, dT, vb        (dT no lugar de sp)
  SOMA       t, p, sp, vb, dT    (5 sinais, voto >=2)
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import BRACO
from portoes import K_BASE, K_VIB
import reduz_fp as RF, ablacao_sp as AS

T0 = pd.Timestamp("2024-02-01 00:00", tz="UTC")
MANCAL, OLEO = "954005_624_TI_0305", "954005_624_TI_0325"
KB = [1.2, 1.4, 1.7, 2.0, 2.4, 2.8]
LIM_DT = [6.0, 9.0, 12.0, 15.0, 20.0, 28.0]     # graus Celsius, limiar absoluto


def main():
    df = canonico(); idx = df.index
    todas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df) & (idx >= T0)
    alvo = todas[todas >= T0]; dentro = pd.Series(idx >= T0, index=idx)
    g = pd.read_parquet("grade2min.parquet")
    tm = pd.to_numeric(g[MANCAL], errors="coerce"); to = pd.to_numeric(g[OLEO], errors="coerce")
    ok = tm.between(20, 200) & to.between(20, 120)
    dT = (tm - to - 9.0).where(mask & ok)
    out = roda(BRACO, df, todas)
    camps = [(a, b, h) for a, b, h in AS.campanhas(df, mask, idx) if a >= T0]
    jw = [(t - pd.Timedelta(hours=48), t) for t in alvo]

    E = {c: out[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask)
         for c, h in [("t", "1h"), ("p", "1h"), ("sp", "30min"), ("vb", "30min")]}
    E["dT"] = dT.ewm(halflife=pd.Timedelta("30min"), times=idx).mean().where(mask)
    print(f"dT: {int(E['dT'].notna().sum())} pontos validos na mascara\n", flush=True)

    def duty(al):
        y = []
        for a, b, h in camps:
            sel = (idx >= a) & (idx <= b); eps = A.episodios(al & sel)
            fp = [(x, z) for x, z in eps if not any(x <= t1 and z >= t0 for t0, t1 in jw)]
            y.append(100 * sum((z - x).total_seconds()/3600 + 2/60 for x, z in fp) / h)
        return np.array(y)

    def monta(kb, sinais, lim_dt):
        T = {"t": DET.THR_FAM*kb, "p": DET.THR_FAM*kb, "sp": DET.THR_SPREAD*kb,
             "vb": 3.0*K_VIB, "dT": lim_dt}
        n = sum(DET._sustained(E[c], T[c]).astype(int) for c in sinais)
        return RF.dur_min(RF.refratario((n >= 2) & mask, 48), 60)

    BR = [("ATUAL  t,p,sp,vb", ["t","p","sp","vb"]),
          ("TROCA  t,p,dT,vb", ["t","p","dT","vb"]),
          ("SOMA   t,p,sp,vb,dT", ["t","p","sp","vb","dT"])]
    L = {}
    for rot, sinais in BR:
        for kb in KB:
            for ld in (LIM_DT if "dT" in sinais else [9.0]):
                al = monta(kb, sinais, ld)
                a2 = al[dentro.values]; m2 = mask[dentro.values]
                x = A.avalia(a2, alvo, m2)
                y = duty(al); r = stats.spearmanr(np.arange(len(y)), y)
                L[(rot, kb, ld)] = dict(braco=rot, kb=kb, lim=ld, det=x["det"],
                                        eps=x["episodios"], fp=x["fp_mes"], h=x["h_fp_mes"],
                                        lead=x["lead_med"], rho=r.statistic, p_rho=r.pvalue,
                                        quais=",".join(x["detectados"]), al=al)
        print(f"  {rot} varrido", flush=True)
    T = pd.DataFrame([{k: v for k, v in d.items() if k != "al"} for d in L.values()])
    T.to_csv("dt_vs_sp.csv", index=False)

    ref = T[(T.braco.str.startswith("ATUAL")) & (T.kb == K_BASE)].iloc[0]
    print("\n" + "=" * 104)
    print(f"A EPISODIOS IGUALADOS ({ref.fp:.2f} FP/mes) -- melhor de cada braco")
    print("=" * 104)
    print(f"{'braco':22s} {'k':>5} {'lim dT':>7} {'det':>6} {'eps':>5} {'FP/mes':>7} {'h/mes':>7} "
          f"{'lead':>6} {'p':>8} | {'rho':>7} {'p_rho':>7} | perdeu")
    base_q = set(ref.quais.split(","))
    for rot, _ in BR:
        g_ = T[T.braco == rot].assign(d=(T[T.braco == rot].fp - ref.fp).abs())
        g_ = g_.sort_values(["det", "d"], ascending=[False, True])
        r = g_.iloc[0]
        al = L[(rot, r.kb, r.lim)]["al"]
        a2 = al[dentro.values]; m2 = mask[dentro.values]
        x = A.avalia(a2, alvo, m2)
        perm = A.permuta(a2, m2, x["det"], len(alvo))["p"]
        q = set(str(r.quais).split(","))
        print(f"{rot:22s} {r.kb:5.2f} {r.lim:6.1f}C {int(r.det):4d}/8 {int(r.eps):5d} {r.fp:7.2f} "
              f"{r.h:7.1f} {r.lead:6.1f} {perm:8.4f} | {r.rho:+7.3f} {r.p_rho:7.4f} | "
              f"{','.join(sorted(base_q-q)) if base_q-q else '—'}")

    print("\n" + "=" * 104); print("PLATO no limiar de dT (o teste que reprovou 3 candidatos)")
    print("=" * 104)
    for rot, sinais in BR[1:]:
        g_ = T[T.braco == rot]
        kb = g_.sort_values(["det", "fp"], ascending=[False, True]).iloc[0].kb
        g_ = g_[g_.kb == kb].sort_values("lim")
        print(f"  {rot} (k={kb}):")
        print(f"     {'limiar':>8} " + " ".join(f"{v:>6.0f}" for v in g_.lim))
        print(f"     {'det':>8} " + " ".join(f"{int(v):>6d}" for v in g_.det))
        print(f"     {'FP/mes':>8} " + " ".join(f"{v:6.2f}" for v in g_.fp))


if __name__ == "__main__":
    main()
