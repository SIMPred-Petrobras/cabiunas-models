#!/usr/bin/env python3
"""E se tirassemos o filtro T5>300 da mascara? Ablacao completa, a custo igualado.

O filtro entra em DOIS lugares e as consequencias sao diferentes:

  (a) PONTUACAO -- quais instantes o detector tem direito de alarmar.
  (b) REFERENCIA/FIT -- quais instantes formam o baseline (PCA de t e p, mediana/MAD do
      spread, referencia rolante da vibracao).

Tirar so de (a) amplia onde o detector fala, mantendo a referencia limpa. Tirar tambem
de (b) deixa dado de maquina fria entrar no proprio normal -- que e o mecanismo que a
mascara existe para impedir. Sao hipoteses distintas e sao medidas separadas.

Bracos:
  M0  stable & ~blackout        a mascara atual
  M1  operando & ~blackout      sem o T5, blackout mantido        (so pontuacao)
  M2  stable                    sem blackout, T5 mantido          (so pontuacao)
  M3  operando                  sem os dois                       (so pontuacao)
  R3  operando, fit tambem      sem os dois, referencia tambem ampliada  (pontuacao+fit)

CUSTO IGUALADO. Ampliar a mascara muda a populacao pontuada e portanto o custo; comparar
a k fixo mede sensibilidade, nao mecanismo (armadilha de Pareto, ja nos pegou 5x). Para
cada braco varremos k_base e tomamos o k cujo FP/mes mais se aproxima do braco M0.

Nota sobre vb: a referencia rolante so preenche onde a mascara `hot` esta ativa. Nos
bracos M1/M3 o sinal de vibracao fica NaN nos pontos frios, entao ali votam 3 sinais em
vez de 4 -- isso e reportado, nao escondido. R3 recalcula a referencia com hot=operando.

Alvos: (i) as 9 paradas reais, regua nossa (48 h); (ii) os 20 episodios de TRIP da
Secao 18 do EXP10c, regua dele (+-24 h).
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C, detector as DET
import avalia as A, rolante as RO
from ablacao import canonico, roda, mascara_pontuacao, ScorerMax
from ablacao4 import BRACO
from portoes import K_BASE, K_VIB
from auto_reset import trunca
from quadrante import regua_diego

PAS = pd.Timedelta("2min")
KS = [0.9, 1.0, 1.1, 1.2, 1.3, 1.45, 1.6, 1.7, 1.85, 2.0, 2.2, 2.5, 2.8, 3.2, 3.8, 4.5]


def blackout(df):
    op = df["in_operation"].astype(bool)
    starts = op & ~op.shift(fill_value=False)
    n = int(pd.Timedelta(DET.BLACKOUT) / pd.Timedelta(C.GRID))
    return starts.rolling(n, min_periods=1).max().astype(bool)


def roda_com(df, falhas, hot):
    """ablacao.roda, mas com `hot` no lugar de `stable` no fit e na referencia."""
    idx = df.index
    meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
    V = df[C.VIBRATION_TAGS].where(hot)
    vib = RO.z_rolante(V, hot, falhas, horas_base=400, guarda_h=24, phi=0.0).max(axis=1)
    out = pd.DataFrame(index=idx, columns=["t", "p", "sp", "vb"], dtype="float64")
    for i, m0 in enumerate(meses):
        m1 = meses[i + 1] if i + 1 < len(meses) else idx[-1] + PAS
        fit = df.loc[hot & (idx < m0), C.SENSOR_TAGS].dropna().tail(DET.FIT_POINTS)
        if len(fit) < DET.FIT_POINTS // 4:
            continue
        sel = (idx >= m0) & (idx < m1)
        if not sel.any():
            continue
        w = df.loc[sel]
        st = ScorerMax().fit(fit[C.TEMPERATURE_TAGS]); sp_ = ScorerMax().fit(fit[C.PRESSURE_TAGS])
        out.loc[sel, "t"] = st.score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy()
        out.loc[sel, "p"] = sp_.score(w[C.PRESSURE_TAGS])["pca_recon"].to_numpy()
        b = DET._spread_mancal(fit); med = float(b.median())
        mad = float((b - med).abs().median() * 1.4826)
        out.loc[sel, "sp"] = ((DET._spread_mancal(w) - med) / mad).abs().to_numpy()
    out["vb"] = vib
    return out


def ewmas(out, mask):
    idx = out.index
    return {c: out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
            for c, hl in [("t", "1h"), ("p", "1h"), ("sp", "30min"), ("vb", "30min")]}


def alerta(E, mask, kb, kv=K_VIB):
    n = (DET._sustained(E["t"], DET.THR_FAM * kb).astype(int)
         + DET._sustained(E["p"], DET.THR_FAM * kb).astype(int)
         + DET._sustained(E["sp"], DET.THR_SPREAD * kb).astype(int)
         + DET._sustained(E["vb"], 3.0 * kv).astype(int))
    return (n >= 2) & mask


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    op = df["in_operation"].astype(bool)
    stable = df["stable"].astype(bool)
    blk = blackout(df)

    trips = pd.read_csv("../../eval_predictive_out/cruza_diego_trip.csv", parse_dates=["inicio"])
    trips["t"] = trips["inicio"].dt.tz_localize("UTC")
    reais = trips[trips["parada_real"]]

    print("horas pontuaveis por mascara (serie toda):")
    for rot, m in [("M0 stable & ~black (atual)", stable & ~blk), ("M1 operando & ~black", op & ~blk),
                   ("M2 stable", stable), ("M3 operando", op)]:
        print(f"  {rot:28s} {m.sum()*2/60:8.0f} h   (+{100*(m.sum()/max((stable&~blk).sum(),1)-1):+6.1f}%)")

    print("\nmontando escores ...", flush=True)
    out0 = roda(BRACO, df, falhas)                      # fit em stable, como hoje
    out3 = roda_com(df, falhas, op)                     # fit em operando (braco R3)

    bracos = [("M0 atual", out0, stable & ~blk), ("M1 sem T5", out0, op & ~blk),
              ("M2 sem blackout", out0, stable), ("M3 sem os dois", out0, op),
              ("R3 sem os dois +fit", out3, op)]

    linhas = []
    for nome, o, mk in bracos:
        E = ewmas(o, mk)
        # quantos pontos da mascara tem vb disponivel (diagnostico do voto de 3 vs 4)
        cob_vb = 100 * (o["vb"].notna() & mk).sum() / max(mk.sum(), 1)
        for kb in KS:
            al = alerta(E, mk, kb)
            x = A.avalia(al, falhas, mk)
            xt = A.avalia(trunca(al, 12), falhas, mk)
            r24 = regua_diego(al, reais, mk)
            r20p = sum(1 for _, rr in trips.iterrows()
                       if al.loc[rr.t - pd.Timedelta("24h"):rr.t + pd.Timedelta("24h")].fillna(False).any())
            linhas.append(dict(braco=nome, k=kb, cob_vb=cob_vb, det=x["det"], eps=x["episodios"],
                               fp_mes=x["fp_mes"], h_mes=x["h_fp_mes"], lead=x["lead_med"], p=np.nan,
                               t12_det=xt["det"], t12_h=xt["h_fp_mes"],
                               trip2=r24["pred"] + r24["reat"], trip20=r20p,
                               quais=",".join(x["detectados"])))
        print(f"  {nome:22s} vb disponivel em {cob_vb:5.1f}% da mascara", flush=True)
    T = pd.DataFrame(linhas)
    T.to_csv("sem_t5.csv", index=False)

    base = T[(T.braco == "M0 atual") & (T.k == K_BASE)].iloc[0]
    print("\n" + "=" * 96)
    print(f"A CUSTO IGUALADO (FP/mes = {base.fp_mes:.2f}, o do braco atual em k={K_BASE})")
    print("=" * 96)
    print(f"{'braco':22s} {'k':>5} {'9 paradas':>10} {'FP/mes':>7} {'h/mes':>7} {'lead':>6} "
          f"{'+teto12h':>9} {'TRIP 2':>7} {'TRIP 20':>8}")
    for nome, _, _ in bracos:
        g = T[T.braco == nome].assign(d=(T[T.braco == nome].fp_mes - base.fp_mes).abs()).sort_values("d")
        r = g.iloc[0]
        print(f"{nome:22s} {r.k:5.2f} {int(r.det):8d}/9 {r.fp_mes:7.2f} {r.h_mes:7.1f} "
              f"{r.lead:6.1f} {int(r.t12_det):7d}/9 {int(r.trip2):5d}/2 {int(r.trip20):6d}/20")
    print("\n(TRIP 2 = os 2 episodios de TRIP que sao parada real; TRIP 20 = denominador do relatorio)")
    print("CSV: sem_t5.csv")


if __name__ == "__main__":
    main()
