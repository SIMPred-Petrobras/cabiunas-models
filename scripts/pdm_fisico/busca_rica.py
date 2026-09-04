#!/usr/bin/env python3
"""Busca sobre um espaco de decisao MAIS RICO, com o criterio validado (plato).

O que ja esta estabelecido e nao se repete aqui:
  - modelo novo, feature nova, arquitetura nova: ~25 tentativas, 25 nulas;
  - o que move o resultado e a camada de decisao;
  - buscar mais NAO piora (custo_da_busca.py: a curva sobe e satura, dp cai 1,94 -> 0,53);
  - o criterio importa: "deteccao com desempate por PLATO" leva o LOEO de 5/8 para 7/8
    (criterio_busca.py). Plato = a config e seus vizinhos de grade dao o mesmo resultado.

O que este script acrescenta. Ate agora a busca varria um `k_base` UNICO para t, p e sp, e
uma sustentacao UNICA de 30 min para os quatro sinais. Essas duas amarras nunca foram
testadas, e a atribuicao por modo (modos.py) sugere que sao erradas:

    vb sustenta em 8/9 eventos, p em 7/9, t em 5/9, sp em 4/9   -> forcas muito diferentes
    lead: 23,6 h no mancal, 14,4 h no oleo, 1,5 h na selagem    -> escalas de tempo diferentes

Espaco: k por sinal (t, p, sp, vb) x sustentacao compartilhada x voto minimo, com o
pos-processamento fixo no que ja validamos (refratario 48 h + duracao minima 60 min).

Objetivo: manter 8/8 e MAXIMIZAR LEAD. O recall ja esta no teto -- o que sobra de ganho
operacional e avisar mais cedo pelo mesmo custo.

Janela oficial: 2025-01 a 2026-04 (todos os 8 eventos; 2024 so diluia o custo com um
regime que nao e mais o da maquina).
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
import reduz_fp as RF

T0 = pd.Timestamp("2025-01-01", tz="UTC")
PAS = pd.Timedelta("2min")
HL = {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min"}
BASE_THR = {"t": DET.THR_FAM, "p": DET.THR_FAM, "sp": DET.THR_SPREAD, "vb": 3.0}
KG = {"t": [1.2, 1.7, 2.2, 3.0], "p": [1.2, 1.7, 2.2, 3.0],
      "sp": [1.2, 1.7, 2.2, 3.0], "vb": [1.6, 2.2, 3.0, 4.0]}
SUST = [30, 60, 90]
VOTOS = [2, 3]
R_REF, D_MIN, ORC = 48, 60, 2.6


def sust_n(s, thr, minutos):
    n = max(1, int(pd.Timedelta(minutes=minutos) / PAS))
    return ((s > thr).astype(int).rolling(n, min_periods=n).sum() >= n)


def main():
    df = canonico(); idx = df.index
    todas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    sel = (idx >= T0)
    mask = mascara_pontuacao(df) & sel
    alvo = list(todas[todas >= T0]); m2 = mask[sel]
    meses = mask.sum() * 2 / 60 / 730.0
    out = roda(BRACO, df, todas)
    E = {c: out[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask) for c, h in HL.items()}
    print(f"janela {T0:%Y-%m}+: {len(alvo)} eventos, {meses:.1f} meses de operacao", flush=True)

    # sustentacoes pre-calculadas: 4 sinais x 4 k x 3 sustentacoes = 48 series
    S = {}
    for c in E:
        for ki, k in enumerate(KG[c]):
            for si, sm in enumerate(SUST):
                S[(c, ki, si)] = sust_n(E[c], BASE_THR[c] * k, sm).values
    print(f"sustentacoes pre-calculadas: {len(S)}", flush=True)

    combos = list(itertools.product(range(4), range(4), range(4), range(4),
                                    range(len(SUST)), range(len(VOTOS))))
    print(f"configuracoes: {len(combos)}", flush=True)
    mv = mask.values
    L, ALS = [], {}
    for n_, (it, ip, isp, ivb, si, iv) in enumerate(combos):
        n = (S[("t", it, si)].astype(np.int8) + S[("p", ip, si)].astype(np.int8)
             + S[("sp", isp, si)].astype(np.int8) + S[("vb", ivb, si)].astype(np.int8))
        al = pd.Series((n >= VOTOS[iv]) & mv, index=idx)
        al = RF.dur_min(RF.refratario(al, R_REF), D_MIN)
        x = A.avalia(al[sel], alvo, m2)
        L.append(dict(it=it, ip=ip, isp=isp, ivb=ivb, si=si, iv=iv,
                      kt=KG["t"][it], kp=KG["p"][ip], ksp=KG["sp"][isp], kvb=KG["vb"][ivb],
                      sust=SUST[si], voto=VOTOS[iv], det=x["det"], eps=x["episodios"],
                      fp=x["fp_mes"], h=x["h_fp_mes"], lead=x["lead_med"],
                      quais=",".join(x["detectados"])))
        ALS[(it, ip, isp, ivb, si, iv)] = al
        if (n_ + 1) % 200 == 0:
            print(f"  {n_+1}/{len(combos)}", flush=True)
    T = pd.DataFrame(L); T.to_csv("busca_rica.csv", index=False)

    # plato: vizinhos = +-1 passo em um eixo
    pos = {tuple(r[["it","ip","isp","ivb","si","iv"]]): i for i, r in T.iterrows()}
    dims = [4, 4, 4, 4, len(SUST), len(VOTOS)]
    VIZ = []
    for key in pos:
        v = []
        for e in range(6):
            for d in (-1, 1):
                kk = list(key); kk[e] += d
                if 0 <= kk[e] < dims[e]: v.append(pos[tuple(kk)])
        VIZ.append(np.array(v))
    det = T.det.values; fp = T.fp.values; lead = T.lead.values
    plat = np.array([(det[VIZ[i]] >= det[i]).mean() for i in range(len(T))])
    T["plato"] = plat; T.to_csv("busca_rica.csv", index=False)

    print("\n" + "=" * 108)
    print(f"MELHORES CONFIGURACOES (orcamento <= {ORC} FP/mes, critério: deteccao > plato > lead)")
    print("=" * 108)
    ok = (fp <= ORC)
    S_ = T[ok].copy().sort_values(["det", "plato", "lead"], ascending=[False, False, False])
    print(f"{'k_t':>5} {'k_p':>5} {'k_sp':>5} {'k_vb':>5} {'sust':>5} {'voto':>5} {'det':>6} "
          f"{'eps':>5} {'FP/mes':>7} {'h/mes':>7} {'lead':>6} {'plato':>6}")
    for _, r in S_.head(12).iterrows():
        print(f"{r.kt:5.1f} {r.kp:5.1f} {r.ksp:5.1f} {r.kvb:5.1f} {int(r.sust):4d}m "
              f"{int(r.voto):5d} {int(r.det):4d}/8 {int(r.eps):5d} {r.fp:7.2f} {r.h:7.1f} "
              f"{r.lead:6.1f} {r.plato:6.2f}")
    ref = T[(T.kt==1.7)&(T.kp==1.7)&(T.ksp==1.7)&(T.kvb==2.2)&(T.sust==30)&(T.voto==2)].iloc[0]
    print(f"\n  referencia (o ponto atual): det {int(ref.det)}/8  {int(ref.eps)} eps  "
          f"{ref.fp:.2f} FP/mes  {ref.h:.1f} h/mes  lead {ref.lead:.1f} h  plato {ref.plato:.2f}")

    print("\n" + "=" * 108); print("LOEO ANINHADO com o criterio de plato"); print("=" * 108)
    keys = list(pos.keys())
    for orc in [2.6, 3.5]:
        ac = 0
        for i, t in enumerate(alvo):
            outros = [j for j in range(len(alvo)) if j != i]
            det_tr = np.array([sum(1 for j in outros
                                   if any(a <= alvo[j] and b >= alvo[j]-pd.Timedelta(hours=48)
                                          for a, b in A.episodios(ALS[keys[q]])))
                               for q in range(len(keys))]) if False else None
            # caminho rapido: reconstroi deteccao por evento a partir de `quais`
            QQ = [set(str(T.iloc[q].quais).split(",")) for q in range(len(T))]
            alvo_s = [f"{x:%Y-%m-%d}" for x in alvo]
            det_tr = np.array([sum(alvo_s[j] in QQ[q] for j in outros) for q in range(len(T))])
            pont = np.where(fp <= orc, det_tr * 1000.0 + plat * 10.0 + lead, -np.inf)
            best = int(np.argmax(pont))
            ac += alvo_s[i] in QQ[best]
        print(f"  orcamento <= {orc} FP/mes: LOEO {ac}/{len(alvo)}", flush=True)


if __name__ == "__main__":
    main()
