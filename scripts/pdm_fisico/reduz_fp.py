#!/usr/bin/env python3
"""Reduzir falso positivo A DETECCAO FIXA -- a frente de Pareto pelo lado do custo.

Regra do exercicio: subir `k` tambem reduz FP, mas isso e so andar na curva existente e
custa deteccao. Aqui a pergunta e outra: existe alguma regra que corte custo MANTENDO
8/9 (ou 7/9 com teto)? So conta o que segura a deteccao.

O que ja sabemos e nao precisa refazer:
  - o blackout de 6 h e o MAXIMO que preserva 8/9 (blackout.csv: a 8 h perde 2025-11-04);
  - 48% dos FP comecam nas primeiras 30 h apos religamento (fp_rajadas.py);
  - piso absoluto de 1,6 degC no spread ja da -17% de horas com os mesmos eventos
    (piso_fisico.csv), e e aditivo ao que for testado aqui;
  - portao de rampa, portao de volatilidade, CFAR, phi e escape: todos refutados.

Bracos novos, cada um atacando um mecanismo diferente do custo:

  SUSTENTACAO  exigir mais que 30 min acima do limiar. Ataca episodio curto e ruidoso.
  VOTO 3       exigir 3 sinais em vez de 2. Ataca coincidencia de 2 sinais.
  GRADUADO     nas W horas apos religamento (alem do blackout), exigir voto>=3 em vez de
               >=2. E o meio-termo entre blackout de 6 h (perde FP) e de 30 h (perde o
               evento 2025-11-04): em vez de apagar a janela, so endurece nela.
  REFRATARIO   apos um episodio terminar, suprimir novo alerta por R horas. NAO muda o
               que o detector ve; muda quantas vezes ele incomoda. Ataca rajada.
  DUR_MIN      descartar episodio mais curto que D minutos.

Cada braco e reportado pelo custo NO PONTO EM QUE AINDA SEGURA 8/9 -- se nao segurar em
nenhum ajuste, e reportado como refutado.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C, detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import BRACO
from portoes import K_BASE, K_VIB
from auto_reset import trunca

PAS = pd.Timedelta("2min")
HL = {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min"}


def sustenta_n(s, thr, n_min):
    n = max(1, int(pd.Timedelta(minutes=n_min) / PAS))
    return ((s > thr).astype(int).rolling(n, min_periods=n).sum() >= n)


def refratario(al, horas):
    """Apos um episodio, suprime novo alerta por `horas`. Reduz incomodo, nao visao."""
    if not horas:
        return al
    novo = pd.Series(False, index=al.index)
    bloq_ate = None
    for a, b in A.episodios(al):
        if bloq_ate is not None and a <= bloq_ate:
            continue
        novo.loc[a:b] = True
        bloq_ate = b + pd.Timedelta(hours=horas)
    return novo


def dur_min(al, minutos):
    if not minutos:
        return al
    novo = pd.Series(False, index=al.index)
    for a, b in A.episodios(al):
        if (b - a).total_seconds() / 60 + 2 >= minutos:
            novo.loc[a:b] = True
    return novo


def main():
    df = canonico(); idx = df.index
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)
    op = df["in_operation"].astype(bool)
    part = op & ~op.shift(fill_value=False)
    out = roda(BRACO, df, falhas)
    E = {c: out[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask) for c, h in HL.items()}

    def thrs(kb, kv):
        return {"t": DET.THR_FAM*kb, "p": DET.THR_FAM*kb, "sp": DET.THR_SPREAD*kb, "vb": 3.0*kv}

    def pos_partida(W):
        """True nas W horas apos cada religamento."""
        if not W:
            return pd.Series(False, index=idx)
        n = int(pd.Timedelta(hours=W) / PAS)
        return part.rolling(n, min_periods=1).max().astype(bool)

    def constroi(kb, kv, sust_min, voto, W, voto_pp):
        T = thrs(kb, kv)
        s = {c: sustenta_n(E[c], T[c], sust_min) for c in E}
        n = sum(s[c].astype(int) for c in E)
        jan = pos_partida(W)
        exig = pd.Series(voto, index=idx)
        if W:
            exig = exig.where(~jan, voto_pp)
        return (n >= exig) & mask

    def mede(al, rot, **kw):
        x = A.avalia(al, falhas, mask)
        xt = A.avalia(trunca(al, 12), falhas, mask)
        return dict(braco=rot, det=x["det"], eps=x["episodios"], fp=x["fp_mes"], h=x["h_fp_mes"],
                    lead=x["lead_med"], t12_det=xt["det"], t12_eps=xt["episodios"],
                    t12_h=xt["h_fp_mes"], quais=",".join(x["detectados"]), **kw)

    L = []
    base = constroi(K_BASE, K_VIB, 30, 2, 0, 3)
    b = mede(base, "BASE", sust=30, voto=2, W=0, R=0, D=0); L.append(b)
    print(f"BASE: {b['det']}/9  {b['eps']} eps  {b['fp']:.2f} FP/mes  {b['h']:.1f} h/mes  "
          f"| teto12h {b['t12_det']}/9 {b['t12_h']:.1f} h/mes", flush=True)

    print("\nbraco SUSTENTACAO", flush=True)
    for sm in [45, 60, 90, 120]:
        L.append(mede(constroi(K_BASE, K_VIB, sm, 2, 0, 3), "SUSTENTACAO", sust=sm, voto=2, W=0, R=0, D=0))
    print("braco VOTO 3", flush=True)
    for kb in [1.0, 1.2, 1.4, 1.7]:
        L.append(mede(constroi(kb, K_VIB, 30, 3, 0, 3), "VOTO3", sust=30, voto=3, W=0, R=0, D=0, k=kb))
    print("braco GRADUADO pos-partida", flush=True)
    for W in [12, 24, 30, 48, 72]:
        L.append(mede(constroi(K_BASE, K_VIB, 30, 2, W, 3), "GRADUADO", sust=30, voto=2, W=W, R=0, D=0))
    print("braco REFRATARIO", flush=True)
    for Rh in [6, 12, 24, 48]:
        L.append(mede(refratario(base, Rh), "REFRATARIO", sust=30, voto=2, W=0, R=Rh, D=0))
    print("braco DUR_MIN", flush=True)
    for D in [60, 120, 180, 360]:
        L.append(mede(dur_min(base, D), "DUR_MIN", sust=30, voto=2, W=0, R=0, D=D))

    T = pd.DataFrame(L); T.to_csv("reduz_fp.csv", index=False)
    print("\n" + "=" * 100)
    print("CUSTO A DETECCAO FIXA -- so vale o que segura 8/9 (sem teto) ou 7/9 (com teto)")
    print("=" * 100)
    print(f"{'braco':13s} {'parametro':>12} {'det':>6} {'eps':>5} {'FP/mes':>7} {'h/mes':>7} "
          f"{'lead':>6} | {'teto12h det':>11} {'eps':>5} {'h/mes':>7}")
    for _, r in T.iterrows():
        par = (f"sust={r.sust}min" if r.braco == "SUSTENTACAO" else
               f"k={r.get('k','')}" if r.braco == "VOTO3" else
               f"W={r.W}h" if r.braco == "GRADUADO" else
               f"R={r.R}h" if r.braco == "REFRATARIO" else
               f"D={r.D}min" if r.braco == "DUR_MIN" else "-")
        marca = "  <<<" if (r.det >= 8) else ("  <" if r.t12_det >= 7 else "")
        print(f"{r.braco:13s} {par:>12} {int(r.det):4d}/9 {int(r.eps):5d} {r.fp:7.2f} {r.h:7.1f} "
              f"{r.lead:6.1f} | {int(r.t12_det):9d}/9 {int(r.t12_eps):5d} {r.t12_h:7.1f}{marca}")


if __name__ == "__main__":
    main()
