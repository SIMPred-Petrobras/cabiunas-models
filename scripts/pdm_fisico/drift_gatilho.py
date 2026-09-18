#!/usr/bin/env python3
"""DRIFT: retreinar por EVIDENCIA em vez de por calendario.

Hoje o bundle e refeito todo mes, cego. Isso tem dois problemas de sinais
opostos. Retreina quando nao precisa -- e retreinar nao e neutro: dois treinos de
configuracao identica diferem 20,7 pontos percentuais de recall
([[piso-de-ruido-retreino]]), entao doze retreinos por ano injetam essa variancia
doze vezes. E nao retreina quando precisa, se a mudanca vier no meio do mes.

O INDICADOR. Nao precisamos inventar um detector de drift: o **duty dos canais**
ja e um. Sabemos o duty esperado de cada canal (24% a 52%). Se ele sobe sem que
haja evento associado, o baseline envelheceu -- o residuo cresceu por deriva do
ponto de operacao, nao por saude da maquina. Se nao sobe, o baseline ainda serve
e retreinar so injeta ruido.

A REGRA testada aqui: ao fim de cada mes, mede-se o duty do canal `t` no mes que
passou, com o bundle vigente. Se passar de `fator` vezes o duty observado no mes
seguinte ao ajuste daquele bundle, retreina no mes seguinte; senao, mantem.

O QUE SE ESPERA GANHAR nao e metrica -- e estabilidade: o mesmo resultado com
menos retreinos. Se o resultado cair, a regra nao serve; se empatar com metade
dos retreinos, serve.

Uso:  PYTHONPATH=. python drift_gatilho.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from cabiunas_pdm import config as C, detector as DET
from ablacao import canonico, ScorerMax
from pos_processamento import mask, idx, alvo, op, sel, cru as cru_pub
from publica_clearml import (SIN, BASE, SUSTAIN, KAPPA, H_CUSUM, K_LO, HL)
from drift_baseline import roda, met, DF, STABLE, IX

FIT = DET.FIT_POINTS


def duty_mes(t_mes: np.ndarray, sel_mes: np.ndarray) -> float:
    """Fracao do mes em que o canal `t` estaria aceso, so pelo nivel."""
    thr = BASE["t"] * K_LO["t"]
    s = pd.Series(t_mes, index=idx[sel_mes]).ewm(
        halflife=pd.Timedelta(HL["t"]), times=idx[sel_mes]).mean()
    m = mask.to_numpy()[sel_mes]
    if m.sum() == 0:
        return float("nan")
    return float((s.to_numpy()[m] > thr).mean())


def walkforward_gatilho(fator: float | None):
    """fator None = retreino mensal (controle). Numero = retreina so quando o
    duty do mes passa de `fator` x o duty de referencia do bundle vigente."""
    meses = pd.date_range(IX[0].normalize().replace(day=1), IX[-1], freq="MS", tz="UTC")
    n = len(IX)
    t = np.full(n, np.nan); p = np.full(n, np.nan)
    med_sp = np.full(n, np.nan); mad_sp = np.full(n, np.nan)
    atual = None          # (sc_t, sc_p, med, mad)
    duty_ref = None       # duty medido no 1o mes servido por este bundle
    n_fit = 0; quando = []
    for i, m0 in enumerate(meses):
        m1 = meses[i + 1] if i + 1 < len(meses) else IX[-1] + pd.Timedelta("2min")
        s = (IX >= m0) & (IX < m1)

        precisa = atual is None or fator is None
        if not precisa and duty_ref is not None and np.isfinite(duty_ref):
            # olha o mes ANTERIOR, ja pontuado, para decidir sobre este
            ant = (IX >= meses[i - 1]) & (IX < m0) if i > 0 else None
            if ant is not None and ant.any() and np.isfinite(t[ant]).any():
                d = duty_mes(t[ant], ant)
                precisa = np.isfinite(d) and d > fator * duty_ref

        if precisa:
            fit = DF.loc[STABLE & (IX < m0), C.SENSOR_TAGS].dropna().tail(FIT)
            if len(fit) >= FIT // 4:
                b = DET._spread_mancal(fit)
                atual = (ScorerMax().fit(fit[C.TEMPERATURE_TAGS]),
                         ScorerMax().fit(fit[C.PRESSURE_TAGS]),
                         float(b.median()),
                         float((b - b.median()).abs().median() * 1.4826))
                n_fit += 1; quando.append(f"{m0:%y-%m}"); duty_ref = None
        if atual is None or not s.any():
            continue
        w = DF.loc[s]
        t[s] = atual[0].score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy()
        p[s] = atual[1].score(w[C.PRESSURE_TAGS])["pca_recon"].to_numpy()
        med_sp[s] = atual[2]; mad_sp[s] = atual[3]
        if duty_ref is None:
            duty_ref = duty_mes(t[s], s)
    return t, p, med_sp, mad_sp, n_fit, quando


if __name__ == "__main__":
    print("=" * 112)
    print("3) RETREINO POR EVIDENCIA -- gatilho no duty do canal t, contra o mensal cego")
    print("=" * 112)
    for fator in (None, 1.15, 1.30, 1.50):
        t, p, ms, ds, nf, q = walkforward_gatilho(fator)
        rot = "mensal (controle)" if fator is None else f"gatilho duty > {fator:.2f}x"
        met(roda(t, p, ms, ds), rot, f"  retreinos {nf:2d}")
        if fator is not None:
            print(f"{'':>34}quando: {' '.join(q)}", flush=True)
