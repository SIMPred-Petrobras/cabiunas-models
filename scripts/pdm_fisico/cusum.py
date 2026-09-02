#!/usr/bin/env python3
"""CUSUM no lugar de EWMA+limiar+sustentacao. A lacuna real que faltava testar.

Por que importa. A regra atual e `EWMA -> limiar -> sustentado 30 min`. Isso e um
detector de DEGRAU: o sinal precisa cruzar um patamar e ficar. Mas o que precede falha
de mancal e DERIVA LENTA -- e para deriva lenta o teste otimo (Lorden 1971) e o CUSUM,
que ACUMULA evidencia em vez de exigir cruzamento.

Um sinal 20% acima do normal por 40 h nunca dispara o nosso limiar. No CUSUM ele soma e
dispara. O lead mediano hoje e 19,7 h; um acumulador pode avisar bem antes.

E e da categoria certa: todos os ganhos deste projeto vieram da camada de decisao
(mascara, voto, refratario), nenhum de modelo novo.

A regra, por sinal, sobre o z ja normalizado:
    S_t = max(0, S_{t-1} + (z_t - kappa))       kappa = folga, em unidades do proprio z
    dispara quando S_t > h                       h = limiar do acumulador
    S zera a cada partida (nao acumula atraves de parada)

Dois parametros por sinal em vez de um. `kappa` define o tamanho do desvio que se quer
detectar (tipicamente metade do desvio de interesse); `h` controla o tempo medio entre
alarmes falsos. Varremos os dois e comparamos com a regra atual a EPISODIOS IGUALADOS,
com permutacao, plato e LOEO -- o mesmo protocolo que aprovou o refratario e reprovou
piso, escape e voto entre sondas.

Janela oficial: 2025-01 a 2026-04, alvo de 8 eventos.
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
import reduz_fp as RF

T0 = pd.Timestamp("2025-01-01", tz="UTC")
HL = {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min"}
BASE = {"t": DET.THR_FAM, "p": DET.THR_FAM, "sp": DET.THR_SPREAD, "vb": 3.0}
KAPPAS = [0.5, 0.75, 1.0, 1.25]      # folga, em fracao do limiar atual do sinal
HS = [20, 40, 80, 160, 320, 640]     # limiar do acumulador (unidades de z x amostras)
R_REF, D_MIN, ORC = 48, 60, 2.6


def cusum(z, kappa, h, reset):
    """CUSUM unilateral com reset nas partidas. Devolve booleano de disparo."""
    x = (z - kappa).fillna(0.0).to_numpy()
    r = reset.to_numpy()
    S = np.zeros(len(x)); acc = 0.0
    for i in range(len(x)):
        if r[i]:
            acc = 0.0
        else:
            acc = max(0.0, acc + x[i])
        S[i] = acc
    return pd.Series(S > h, index=z.index)


def main():
    df = canonico(); idx = df.index
    todas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    sel = (idx >= T0); mask = mascara_pontuacao(df) & sel
    alvo = list(todas[todas >= T0]); m2 = mask[sel]
    op = df["in_operation"].astype(bool)
    reset = (~mask) | (op & ~op.shift(fill_value=False))     # zera fora da mascara e na partida
    out = roda(BRACO, df, todas)
    E = {c: out[c].ewm(halflife=pd.Timedelta(hh), times=idx).mean().where(mask)
         for c, hh in HL.items()}
    # z normalizado pelo limiar atual: 1,0 = o ponto em que a regra de hoje dispara
    K_ATUAL = {"t": 1.7, "p": 1.7, "sp": 1.7, "vb": 2.2}
    Z = {c: (E[c] / (BASE[c] * K_ATUAL[c])).clip(upper=20) for c in E}
    print(f"janela {T0:%Y-%m}+: {len(alvo)} eventos, {mask.sum()*2/60/730:.1f} meses\n", flush=True)

    # referencia: a regra atual
    n_at = sum(DET._sustained(E[c], BASE[c] * K_ATUAL[c]).astype(int) for c in E)
    al_at = RF.dur_min(RF.refratario((n_at >= 2) & mask, R_REF), D_MIN)
    x = A.avalia(al_at[sel], alvo, m2); x.update(A.permuta(al_at[sel], m2, x["det"], len(alvo)))
    print(f"REFERENCIA (EWMA+limiar+sustentacao): {x['det']}/8  {x['episodios']} eps  "
          f"{x['fp_mes']:.2f} FP/mes  {x['h_fp_mes']:.1f} h/mes  lead {x['lead_med']:.1f} h  "
          f"p={x['p']:.4f}\n", flush=True)

    L = []
    for ka in KAPPAS:
        DISP = {c: {} for c in Z}
        for c in Z:
            for h in HS:
                DISP[c][h] = cusum(Z[c], ka, h, reset)
        for hs in itertools.product(HS, repeat=1):     # h compartilhado entre sinais
            h = hs[0]
            for voto in [2, 3]:
                n = sum(DISP[c][h].astype(int) for c in Z)
                al = RF.dur_min(RF.refratario((n >= voto) & mask, R_REF), D_MIN)
                y = A.avalia(al[sel], alvo, m2)
                L.append(dict(kappa=ka, h=h, voto=voto, det=y["det"], eps=y["episodios"],
                              fp=y["fp_mes"], hm=y["h_fp_mes"], lead=y["lead_med"],
                              quais=",".join(y["detectados"])))
        print(f"  kappa={ka} varrido", flush=True)
    T = pd.DataFrame(L); T.to_csv("cusum.csv", index=False)

    print("\n" + "=" * 96)
    print(f"CUSUM x REGRA ATUAL (orcamento <= {ORC} FP/mes)")
    print("=" * 96)
    S = T[(T.fp <= ORC)].sort_values(["det", "lead"], ascending=[False, False])
    print(f"{'kappa':>6} {'h':>6} {'voto':>5} {'det':>6} {'eps':>5} {'FP/mes':>7} {'h/mes':>7} {'lead':>6}")
    for _, r in S.head(12).iterrows():
        print(f"{r.kappa:6.2f} {int(r.h):6d} {int(r.voto):5d} {int(r.det):4d}/8 {int(r.eps):5d} "
              f"{r.fp:7.2f} {r.hm:7.1f} {r.lead:6.1f}")
    print(f"\n  melhor lead entre os que seguram 8/8: "
          f"{T[(T.det==8)&(T.fp<=ORC)].lead.max() if len(T[(T.det==8)&(T.fp<=ORC)]) else 'nenhum':}")
    print(f"  (a regra atual da lead {x['lead_med']:.1f} h)")

    print("\n" + "=" * 96); print("PLATO em h, para os melhores kappa"); print("=" * 96)
    for ka in KAPPAS:
        g = T[(T.kappa == ka) & (T.voto == 2)].sort_values("h")
        print(f"  kappa={ka}:  h = " + " ".join(f"{int(v):>5d}" for v in g.h))
        print(f"  {'':>12} det = " + " ".join(f"{int(v):>5d}" for v in g.det))
        print(f"  {'':>12} FP  = " + " ".join(f"{v:5.2f}" for v in g.fp))


if __name__ == "__main__":
    main()
