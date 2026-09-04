#!/usr/bin/env python3
"""Um detector so de vibracao. Quanto do resultado vem de um unico sinal?

Motivo. A atribuicao por evento (modos.py) mostrou que `vb` sustenta em 8 dos 9 eventos --
mais que qualquer outro -- e nos eventos de mancal de abril/2025 chega a 10,9x e 10,4x o
proprio limiar. E o sinal que estava em QUARENTENA DECLARADA no detector do Francisco
quando esta investigacao comecou ("Vibracao fora da politica"). Se ele sozinho entrega a
maior parte do resultado, o produto pode ser drasticamente mais simples -- e um detector
de um sinal so e muito mais facil de implantar, explicar e auditar.

O problema de desenho: com um sinal so, o voto >=2 entre familias nao existe. Mas ha um
analogo direto e fisicamente MAIS defensavel -- votar entre as 10 SONDAS. Uma falha de
mancal real aparece em varias sondas ao mesmo tempo (o eixo inteiro se move); ruido
eletrico ou falha de instrumento aparece numa. E a mesma logica de confirmacao, aplicada
dentro da familia em vez de entre familias.

Bracos:
  MAX      max do z sobre as 10 sondas, sustentado (o `vb` como esta hoje, sozinho)
  VOTO-N   pelo menos N sondas simultaneamente acima do limiar, sustentado
  4SINAIS  o detector completo, como referencia

Tudo com o ponto de operacao novo (refratario 48 h + duracao minima 60 min), a episodios
igualados, com permutacao, deriva por campanha (n=57) e LOEO.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C, detector as DET
import avalia as A, rolante as RO
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO
from portoes import K_BASE, K_VIB
import reduz_fp as RF, ablacao_sp as AS

KV = [1.0, 1.3, 1.6, 2.0, 2.2, 2.6, 3.0, 3.5, 4.2, 5.0]
R_REFRAT, D_MIN = 48, 60


def main():
    df = canonico(); idx = df.index
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df); stable = df["stable"].astype(bool)
    camps = AS.campanhas(df, mask, idx)
    jw = [(t - pd.Timedelta(hours=48), t) for t in falhas]

    print("calculando o z por sonda (referencia rolante 400 h + guarda 24 h) ...", flush=True)
    V = df[C.VIBRATION_TAGS].where(stable)
    Z = RO.z_rolante(V, stable, falhas, horas_base=400, guarda_h=24, phi=0.0)
    print(f"  {Z.shape[1]} sondas, {int(Z.notna().any(axis=1).sum())} instantes com z valido\n",
          flush=True)

    idxs = Z.index
    def ew(s):
        return s.ewm(halflife=pd.Timedelta("30min"), times=idxs).mean().where(mask)
    E_max = ew(Z.max(axis=1))
    E_cnt = {n: None for n in [2, 3, 4]}

    def duty_camp(al):
        y = []
        for a, b, h in camps:
            sel = (idx >= a) & (idx <= b)
            eps = A.episodios(al & sel)
            fp = [(x, z) for x, z in eps if not any(x <= t1 and z >= t0 for t0, t1 in jw)]
            y.append(100 * sum((z - x).total_seconds()/3600 + 2/60 for x, z in fp) / h)
        return np.array(y)

    def pos(al, rot, **kw):
        x = A.avalia(al, falhas, mask)
        y = duty_camp(al); r = stats.spearmanr(np.arange(len(y)), y); m = len(y)//2
        return dict(braco=rot, det=x["det"], eps=x["episodios"], fp=x["fp_mes"], h=x["h_fp_mes"],
                    lead=x["lead_med"], rho=r.statistic, p_rho=r.pvalue,
                    d1=y[:m].mean(), d2=y[m:].mean(), quais=",".join(x["detectados"]), **kw)

    L, ALS = [], {}
    for kv in KV:
        thr = 3.0 * kv
        al = RF.dur_min(RF.refratario(DET._sustained(E_max, thr) & mask, R_REFRAT), D_MIN)
        ALS[("MAX", kv)] = al; L.append(pos(al, "MAX", kv=kv))
        acima = (Z > thr)
        for n in [2, 3, 4]:
            s = ew(acima.sum(axis=1).astype(float))
            aln = RF.dur_min(RF.refratario(DET._sustained(s, n - 0.5) & mask, R_REFRAT), D_MIN)
            ALS[(f"VOTO{n}", kv)] = aln; L.append(pos(aln, f"VOTO{n}", kv=kv))
        print(f"  k_vib={kv} varrido", flush=True)

    out = roda(BRACO, df, falhas)
    ref_al = RF.dur_min(RF.refratario(alerta_2k(out, mask, K_BASE, K_VIB), R_REFRAT), D_MIN)
    ref = pos(ref_al, "4SINAIS", kv=K_VIB); L.append(ref)
    T = pd.DataFrame(L); T.to_csv("so_vibracao.csv", index=False)

    print("\n" + "=" * 112)
    print(f"A EPISODIOS IGUALADOS ({ref['fp']:.2f} FP/mes, o do detector completo)")
    print("=" * 112)
    print(f"{'braco':10s} {'k_vib':>6} {'det':>6} {'eps':>5} {'FP/mes':>7} {'h/mes':>7} {'lead':>6} "
          f"{'p':>8} | {'rho':>7} {'p_rho':>7} | eventos perdidos")
    base_q = set(ref["quais"].split(","))
    for br in ["4SINAIS", "MAX", "VOTO2", "VOTO3", "VOTO4"]:
        g = T[T.braco == br].assign(d=(T[T.braco == br].fp - ref["fp"]).abs()).sort_values("d")
        r = g.iloc[0]
        al = ref_al if br == "4SINAIS" else ALS[(br, r.kv)]
        x = A.avalia(al, falhas, mask)
        perm = A.permuta(al, mask, x["det"], len(falhas))["p"]
        q = set(str(r.quais).split(","))
        print(f"{br:10s} {r.kv:6.1f} {int(r.det):4d}/9 {int(r.eps):5d} {r.fp:7.2f} {r.h:7.1f} "
              f"{r.lead:6.1f} {perm:8.4f} | {r.rho:+7.3f} {r.p_rho:7.4f} | "
              f"{','.join(sorted(base_q - q)) if base_q - q else '—'}")

    print("\n" + "=" * 112)
    print("LOEO -- ponto reescolhido dentro de cada familia, fora do evento testado")
    print("=" * 112)
    ALS[("4SINAIS", K_VIB)] = ref_al
    for br in ["4SINAIS", "MAX", "VOTO2", "VOTO3"]:
        fam = {k: v for k, v in ALS.items() if k[0] == br}
        ac = 0
        for t in falhas:
            resto = [x for x in falhas if x != t]; m = None
            for key, al in fam.items():
                x = A.avalia(al, resto, mask)
                if x["fp_mes"] <= 3.0 and (m is None or (x["det"], -x["fp_mes"]) > m[1]):
                    m = (key, (x["det"], -x["fp_mes"]))
            if m is None: continue
            ac += bool(fam[m[0]].loc[t - pd.Timedelta(hours=48):t].fillna(False).any())
        print(f"  {br:10s} LOEO {ac}/9   (orcamento <= 3,0 FP/mes)", flush=True)


if __name__ == "__main__":
    main()
