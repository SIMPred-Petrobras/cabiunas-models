#!/usr/bin/env python3
"""Troca o `max` sobre as 10 sondas por VOTO ENTRE SONDAS, dentro do detector completo.

Motivo. Hoje `vb = max(z das 10 sondas)`. Isso da a UMA sonda ruidosa poder de veto sobre
o sinal inteiro -- e a familia de vibracao e a mais exposta a falha de instrumento. A
logica que sustenta o voto >=2 entre familias (degradacao move varios canais, manobra e
ruido movem um) vale igual DENTRO da familia: uma falha de mancal real move varias sondas
porque o eixo inteiro se desloca.

A evidencia que motiva (so_vibracao.py): num detector so de vibracao, o voto entre sondas
domina o max -- com >=4 sondas chega a 8/9 e lead de 28,8 h contra 7/9 e 23,1 h do max, e
tem a menor deriva de todos os bracos (rho=+0,137 contra +0,187).

Aqui a troca e testada onde ela importa: DENTRO do detector de 4 sinais, a custo igualado,
com o ponto de operacao novo (refratario 48 h + duracao minima 60 min).

Bracos:
  MAX       vb = max(z), sustentado acima de 3,0 x k_vib   (como esta hoje)
  VOTO-N    vb = "pelo menos N sondas com z acima de 3,0 x k_vib", sustentado

Cuidado de desenho: os dois nao sao comparaveis a k_vib fixo -- exigir N sondas e mais
restritivo que o max para o mesmo limiar por sonda. Por isso k_vib E k_base sao varridos
nos dois bracos e a comparacao e feita a EPISODIOS igualados.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
from scipy import stats

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C, detector as DET
import avalia as A, rolante as RO
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import BRACO
from portoes import K_BASE, K_VIB
import reduz_fp as RF, ablacao_sp as AS

CACHE_Z = "z_sondas.parquet"
KB = [1.2, 1.4, 1.7, 2.0, 2.4]
KV = [1.0, 1.3, 1.6, 2.0, 2.2, 2.6, 3.2]
NS = [2, 3, 4]
R_REFRAT, D_MIN = 48, 60


def main():
    df = canonico(); idx = df.index
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df); stable = df["stable"].astype(bool)
    camps = AS.campanhas(df, mask, idx)
    jw = [(t - pd.Timedelta(hours=48), t) for t in falhas]

    if os.path.exists(CACHE_Z):
        Z = pd.read_parquet(CACHE_Z); print(f"cache {CACHE_Z} reaproveitado", flush=True)
    else:
        print("calculando o z por sonda ...", flush=True)
        Z = RO.z_rolante(df[C.VIBRATION_TAGS].where(stable), stable, falhas,
                         horas_base=400, guarda_h=24, phi=0.0)
        Z.to_parquet(CACHE_Z)
    out = roda(BRACO, df, falhas)
    E3 = {c: out[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask)
          for c, h in [("t", "1h"), ("p", "1h"), ("sp", "30min")]}

    def ew(s):
        return s.ewm(halflife=pd.Timedelta("30min"), times=idx).mean().where(mask)

    E_max = ew(Z.max(axis=1))
    CNT = {kv: ew((Z > 3.0 * kv).sum(axis=1).astype(float)) for kv in KV}

    def duty_camp(al):
        y = []
        for a, b, h in camps:
            sel = (idx >= a) & (idx <= b)
            eps = A.episodios(al & sel)
            fp = [(x, z) for x, z in eps if not any(x <= t1 and z >= t0 for t0, t1 in jw)]
            y.append(100 * sum((z - x).total_seconds()/3600 + 2/60 for x, z in fp) / h)
        return np.array(y)

    def monta(kb, s_vb):
        T = {"t": DET.THR_FAM*kb, "p": DET.THR_FAM*kb, "sp": DET.THR_SPREAD*kb}
        n = sum(DET._sustained(E3[c], T[c]).astype(int) for c in E3) + s_vb.astype(int)
        return RF.dur_min(RF.refratario((n >= 2) & mask, R_REFRAT), D_MIN)

    L, ALS = [], {}
    for kb in KB:
        for kv in KV:
            al = monta(kb, DET._sustained(E_max, 3.0 * kv))
            ALS[("MAX", kb, kv)] = al
            L.append(dict(braco="MAX", kb=kb, kv=kv, N=0))
            for N in NS:
                aln = monta(kb, DET._sustained(CNT[kv], N - 0.5))
                ALS[(f"VOTO{N}", kb, kv)] = aln
                L.append(dict(braco=f"VOTO{N}", kb=kb, kv=kv, N=N))
        print(f"  k_base={kb} varrido", flush=True)

    for r in L:
        al = ALS[(r["braco"], r["kb"], r["kv"])]
        x = A.avalia(al, falhas, mask)
        y = duty_camp(al); sp_ = stats.spearmanr(np.arange(len(y)), y)
        r.update(det=x["det"], eps=x["episodios"], fp=x["fp_mes"], h=x["h_fp_mes"],
                 lead=x["lead_med"], rho=sp_.statistic, p_rho=sp_.pvalue,
                 quais=",".join(x["detectados"]))
    T = pd.DataFrame(L); T.to_csv("voto_sondas.csv", index=False)

    ref = T[(T.braco == "MAX") & (T.kb == K_BASE) & (T.kv == K_VIB)].iloc[0]
    print("\n" + "=" * 112)
    print(f"A EPISODIOS IGUALADOS ({ref.fp:.2f} FP/mes) -- melhor de cada braco")
    print("=" * 112)
    print(f"{'braco':8s} {'k_base':>7} {'k_vib':>6} {'det':>6} {'eps':>5} {'FP/mes':>7} "
          f"{'h/mes':>7} {'lead':>6} {'p':>8} | {'rho':>7} {'p_rho':>7} | perdeu")
    base_q = set(ref.quais.split(","))
    fin = {}
    for br in ["MAX"] + [f"VOTO{n}" for n in NS]:
        g = T[T.braco == br].copy()
        g = g.assign(d=(g.fp - ref.fp).abs()).sort_values(["det", "d"], ascending=[False, True])
        r = g.iloc[0]
        al = ALS[(br, r.kb, r.kv)]
        x = A.avalia(al, falhas, mask)
        perm = A.permuta(al, mask, x["det"], len(falhas))["p"]
        q = set(str(r.quais).split(","))
        fin[br] = (r, perm)
        print(f"{br:8s} {r.kb:7.2f} {r.kv:6.1f} {int(r.det):4d}/9 {int(r.eps):5d} {r.fp:7.2f} "
              f"{r.h:7.1f} {r.lead:6.1f} {perm:8.4f} | {r.rho:+7.3f} {r.p_rho:7.4f} | "
              f"{','.join(sorted(base_q - q)) if base_q - q else '—'}")

    print("\n" + "=" * 112)
    print("PLATO -- o resultado do melhor braco vale numa faixa de N e de k_vib?")
    print("=" * 112)
    for br in [f"VOTO{n}" for n in NS]:
        g = T[T.braco == br].sort_values("kv")
        g = g[g.kb == fin[br][0].kb]
        print(f"  {br} (k_base={fin[br][0].kb}):")
        print(f"     {'k_vib':>7} " + " ".join(f"{v:>6.1f}" for v in g.kv))
        print(f"     {'det':>7} " + " ".join(f"{int(v):>6d}" for v in g.det))
        print(f"     {'FP/mes':>7} " + " ".join(f"{v:6.2f}" for v in g.fp))

    print("\n" + "=" * 112); print("LOEO -- ponto reescolhido dentro da familia, fora do evento")
    print("=" * 112)
    for br in ["MAX"] + [f"VOTO{n}" for n in NS]:
        fam = {k: v for k, v in ALS.items() if k[0] == br}
        ac = 0
        for t in falhas:
            resto = [x for x in falhas if x != t]; m = None
            for key, al in fam.items():
                x = A.avalia(al, resto, mask)
                if x["fp_mes"] <= 2.6 and (m is None or (x["det"], -x["fp_mes"]) > m[1]):
                    m = (key, (x["det"], -x["fp_mes"]))
            if m is None: continue
            ac += bool(fam[m[0]].loc[t - pd.Timedelta(hours=48):t].fillna(False).any())
        print(f"  {br:8s} LOEO {ac}/9   (orcamento <= 2,6 FP/mes)", flush=True)


if __name__ == "__main__":
    main()
