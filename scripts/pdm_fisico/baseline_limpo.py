#!/usr/bin/env python3
"""BASELINE LIMPO -- tirar do fit do PCA a degradacao que ja terminou em trip.

O PROBLEMA. O `vb` exclui +-7 d em torno de cada trip conhecido da sua referencia
rolante: degradacao que se sabe ter terminado em falha nao pode virar o "normal"
contra o qual a proxima e medida. O fit do PCA de `t` e `p` NAO exclui nada --
pega os ultimos 20.000 pontos estaveis anteriores ao mes e pronto.

Medido (`baseline_contaminado`): **10 dos 16 bundles mensais tem trip dentro do
baseline**. Os de maio e junho de 2025 carregam tres cada (07/04, 11/04, 29/04).
Nesses meses o PCA aprendeu a degradacao como normalidade, e o residuo da proxima
degradacao igual sai menor do que deveria. E auto-sabotagem silenciosa, e e uma
forma de drift: o baseline nao envelhece, ele se contamina.

O CUSTO DA CORRECAO, e por isso precisa ser medido e nao presumido: para manter
20.000 pontos estaveis depois de remover as janelas, o fit tem de buscar mais
para tras. O baseline fica mais LIMPO e mais VELHO ao mesmo tempo -- e baseline
velho e exatamente o que o retreino mensal existe para evitar
([[cadencia-de-retreino-depende-do-sinal]]). Os dois efeitos brigam.

So trips ANTERIORES ao mes servido sao excluidos. Trip futuro nao se conhece, e
usa-lo seria o vazamento classico desta funcao.

RESULTADO: LIMPAR O BASELINE PIORA, E MUITO.

    baseline                  banda  det   FP/mes   h/mes   idade do fit
    atual, sem exclusao        5/8   8/8    0,344     6,6      3,9 d
    excluindo +-7 d            3/8   7/8    0,603    68,0      4,5 d
    excluindo +-14 d           4/8   8/8    0,775    29,2      5,0 d

Dez vezes as horas de alarme falso, e a idade do fit quase nao mudou (3,9 -> 4,5 d):
nao e envelhecimento, e outra coisa.

O MECANISMO. A `ScorerMax` normaliza o residuo pelo `recon_p99` -- o percentil 99
do proprio erro NO BASELINE. O detector nao mede "distancia do normal", mede
"distancia do normal RELATIVA A VARIABILIDADE do normal". Tirar as janelas
pre-falha tira justamente a cauda alta da distribuicao: o p99 desce, todo score
dividido por ele sobe, e o duty dos canais explode.

Ou seja: **o baseline precisa conter anormalidade para calibrar a escala**. A
intuicao de "treinar so no normal puro" esta errada para um detector normalizado
por percentil do proprio baseline. Vale para qualquer mudanca futura de cadencia
ou de janela de treino -- ver [[baseline-precisa-de-anormalidade]].

Uso:  PYTHONPATH=. python baseline_limpo.py
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
import avalia as AV
from cabiunas_pdm import config as C, detector as DET
from ablacao import canonico, ScorerMax
from pos_processamento import mask, idx, alvo, op, sel, EW as EW_pub, cru as cru_pub
from publica_clearml import (SIN, BASE, SUSTAIN, KAPPA, H_CUSUM, K_LO, VOTO_LO,
                             VOTO_HI, REFRAT_V2, DUR_MIN, ESC_IDADE, ESC_ABS,
                             ESC_DUR, HL, TMIN_BANDA)
from blackout_curto import cusum
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

JAN48 = pd.Timedelta(hours=48)
KH = {"t": 1.7, "p": 1.7, "sp": 1.7, "vb": 2.2}
N_GAP = int(pd.Timedelta(hours=AV.GAP_EP_H) / pd.Timedelta("2min")) + 1
paradas = paradas_reais_2h()
part = op & ~op.shift(fill_value=False)
reset = ((~mask) | part).to_numpy()
falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")


def walkforward(excl_dias: float, pos_dias: float = 2.0):
    """t, p e (med, mad) do spread, mes a mes. excl_dias = 0 reproduz o atual."""
    df = canonico()
    stable = df["stable"].astype(bool)
    ix = df.index
    meses = pd.date_range(ix[0].normalize().replace(day=1), ix[-1], freq="MS", tz="UTC")
    n = len(ix)
    t = np.full(n, np.nan); p = np.full(n, np.nan)
    med_sp = np.full(n, np.nan); mad_sp = np.full(n, np.nan)
    idade = []
    for i, m0 in enumerate(meses):
        m1 = meses[i + 1] if i + 1 < len(meses) else ix[-1] + pd.Timedelta("2min")
        elegivel = stable & (ix < m0)
        if excl_dias > 0:
            # so os trips JA OCORRIDOS; o futuro nao se conhece
            for f in falhas[falhas < m0]:
                fora = (ix >= f - pd.Timedelta(days=excl_dias)) & (ix <= f + pd.Timedelta(days=pos_dias))
                elegivel &= ~fora
        fit = df.loc[elegivel, C.SENSOR_TAGS].dropna().tail(DET.FIT_POINTS)
        if len(fit) < DET.FIT_POINTS // 4:
            continue
        s = (ix >= m0) & (ix < m1)
        if not s.any():
            continue
        idade.append((m0 - fit.index[-1]).days)
        w = df.loc[s]
        st = ScorerMax().fit(fit[C.TEMPERATURE_TAGS])
        sp_ = ScorerMax().fit(fit[C.PRESSURE_TAGS])
        t[s] = st.score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy()
        p[s] = sp_.score(w[C.PRESSURE_TAGS])["pca_recon"].to_numpy()
        b = DET._spread_mancal(fit)
        med_sp[s] = float(b.median())
        mad_sp[s] = float((b - b.median()).abs().median() * 1.4826)
    return t, p, med_sp, mad_sp, idade


def roda(t, p, med_sp, mad_sp):
    z = np.load("piso_fisico_cache.npz")
    spv = np.abs((z["b_all"] - med_sp) / mad_sp)
    cru = pd.DataFrame({"t": t, "p": p, "sp": spv, "vb": cru_pub["vb"].to_numpy()}, index=idx)
    EW = {c: cru[c].ewm(halflife=pd.Timedelta(h), times=idx).mean() for c, h in HL.items()}

    def canal(c, k):
        thr = BASE[c] * k; E = EW[c].where(mask)
        deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        cu = pd.Series(cusum(((E / thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                             reset) > H_CUSUM, index=idx)
        return (deg | cu) & mask

    A = {c: canal(c, K_LO[c]) for c in SIN}
    B = {c: canal(c, KH[c]) for c in SIN}
    vA = pd.Series(sum(A[c].astype(int) for c in SIN) >= VOTO_LO, index=idx) & mask
    vB = (pd.Series(sum(B[c].astype(int) for c in SIN) >= VOTO_HI, index=idx)
          & mask & (B["sp"] | B["vb"]))
    voto = vA | vB
    F = pd.concat([EW[c].where(mask) / (BASE[c] * KH[c]) for c in SIN], axis=1).max(axis=1)
    f = F.fillna(0.0).to_numpy(); v = voto.to_numpy().copy()
    n_id = int(ESC_IDADE * 30); dentro, ini, ja = False, 0, False
    for i in range(len(v)):
        if not v[i]: dentro, ja = False, False; continue
        ac = f[i] > ESC_ABS
        if not dentro: dentro, ini, ja = True, i, ac; continue
        if ac and not ja and (i - ini) >= n_id:
            v[max(ini + 1, i - N_GAP):i] = False; ini = i
        ja = ac
    voto = pd.Series(v, index=idx)
    al = pd.Series(False, index=idx); bloq = ini_b = None; fortes = []
    for a, b in AV.episodios(voto):
        forte = float(F.loc[a:b].max()) > ESC_ABS
        velho = ini_b is not None and (a - ini_b).total_seconds() / 3600 >= ESC_IDADE
        if bloq is not None and a <= bloq and not (forte and velho): continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=REFRAT_V2); ini_b = a
        if forte: fortes.append((a, b))
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        d = (b - a).total_seconds() / 60 + 2
        if (any(x >= a and y <= b for x, y in fortes) and d >= ESC_DUR) or d >= DUR_MIN:
            fin.loc[a:b] = True
    return fin & sel


def met(fin, nome, idade):
    eps = AV.episodios(fin); det = AV.avalia(fin, alvo, mask)
    cls = classifica_regra_c(eps, paradas)
    nfp = sum(1 for _, _, k, _ in cls if k == "FP")
    hfp = sum((b - a).total_seconds() / 3600 for a, b, k, _ in cls if k == "FP")
    mes = det["horas_op"] / 730.0
    leads = [(t - max([a for a, _ in eps if t - JAN48 <= a <= t])).total_seconds() / 3600
             for t in alvo if any(t - JAN48 <= a <= t for a, _ in eps)]
    ban = sum(1 for t in alvo if any(t - JAN48 <= a <= t - pd.Timedelta(hours=TMIN_BANDA)
                                     for a, _ in eps))
    print(f"{nome:<34} banda {ban}/8  inicio {len(leads)}/8  det {det['det']}/8  "
          f"{nfp/mes:6.3f} FP/mes  {hfp/mes:5.1f} h/mes  lead {np.mean(leads):5.1f}h  "
          f"eps {len(eps):3d}  baseline termina {np.mean(idade):4.1f} d antes")


if __name__ == "__main__":
    for excl in (0.0, 7.0, 14.0):
        rot = ("ATUAL, sem exclusao" if excl == 0
               else f"excluindo +-{excl:.0f} d dos trips")
        t, p, ms, ds, idade = walkforward(excl)
        met(roda(t, p, ms, ds), rot, idade)
