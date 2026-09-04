#!/usr/bin/env python3
"""Referencia reconstruida por campanha, em vez de referencia mensal.

Diagnostico que motiva (dois_regimes.py): os FP pos-partida NAO sao transiente
-- sustentacao de 120 min nao os elimina, e varios duram 40h+. Logo o estado
pos-reinicio e genuinamente diferente da referencia, que foi construida com
operacao anterior a parada. Filtro temporal nao resolve; reconstruir a
referencia, em tese, sim.

Desenho: campanha = periodo contiguo de operacao (entre duas partidas). As
primeiras BOOT_H horas de operacao quente-estavel de cada campanha (ja passado
o blackout de 6h) viram a referencia daquela campanha; o restante da campanha
e pontuado contra ela. Campanhas curtas demais para formar a referencia ficam
sem pontuacao.

RISCO FISICO, declarado antes de olhar o resultado: se a maquina reinicia ja
degradada, a re-referencia adota a degradacao como normal e cega o detector.
E o preco conceitual desta abordagem, nao um bug. Alem disso a janela de boot
consome parte da campanha -- 2025-11-04 tinha so 15,3h continuas, entao com
BOOT_H alto esse evento fica sem pontuacao nenhuma. Ambos sao medidos abaixo.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C, detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao, ScorerMax
from ablacao4 import BRACO

K_BASE, K_VIB = 1.3, 5.5
BOOTS = [8.0, 12.0, 24.0]
MIN_PONTUAVEL_H = 4.0    # campanha precisa sobrar isso apos o boot p/ valer a pena


def campanhas(df):
    op = df["in_operation"].astype(bool)
    starts = op & ~op.shift(fill_value=False)
    return starts.cumsum()


def roda_campanha(df, falhas, boot_h):
    """Como roda(), mas a referencia de cada campanha vem do inicio dela."""
    stable = df["stable"].astype(bool)
    idx = df.index
    cid = campanhas(df)
    n_boot = int(boot_h * 60 / 2)

    out = pd.DataFrame(index=idx, columns=["t", "p", "sp", "vb"], dtype="float64")
    pontuavel = pd.Series(False, index=idx)
    usadas = descartadas = 0

    for c, bloco in cid.groupby(cid):
        m_camp = (cid == c) & stable
        pos = idx[m_camp.to_numpy()]
        if len(pos) < n_boot + int(MIN_PONTUAVEL_H * 60 / 2):
            descartadas += 1
            continue
        ref_idx = pos[:n_boot]              # boot: referencia da campanha
        sco_idx = pos[n_boot:]              # resto: pontuado contra ela
        fit = df.loc[ref_idx, C.SENSOR_TAGS].dropna()
        if len(fit) < n_boot // 2:
            descartadas += 1
            continue
        w = df.loc[sco_idx]
        try:
            st = ScorerMax().fit(fit[C.TEMPERATURE_TAGS])
            sp = ScorerMax().fit(fit[C.PRESSURE_TAGS])
            out.loc[sco_idx, "t"] = st.score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy()
            out.loc[sco_idx, "p"] = sp.score(w[C.PRESSURE_TAGS])["pca_recon"].to_numpy()
        except Exception:
            descartadas += 1
            continue
        b = DET._spread_mancal(fit)
        med = float(b.median()); mad = float((b - med).abs().median() * 1.4826) or np.nan
        out.loc[sco_idx, "sp"] = ((DET._spread_mancal(w) - med) / mad).abs().to_numpy()

        V = df.loc[ref_idx, C.VIBRATION_TAGS]
        vmed = V.median(); vmad = (V - vmed).abs().median() * 1.4826
        vmad = vmad.replace(0, np.nan)
        out.loc[sco_idx, "vb"] = ((df.loc[sco_idx, C.VIBRATION_TAGS] - vmed) / vmad).abs().max(axis=1).to_numpy()

        pontuavel.loc[sco_idx] = True
        usadas += 1

    print(f"  campanhas usadas: {usadas}  descartadas (curtas): {descartadas}  "
          f"horas pontuaveis: {pontuavel.sum()*2/60:.0f}h", flush=True)
    return out, pontuavel


def alerta_de(out, mask):
    idx = out.index
    def ew(c, hl):
        return out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
    n = (DET._sustained(ew("t", "1h"), DET.THR_FAM * K_BASE).astype(int)
         + DET._sustained(ew("p", "1h"), DET.THR_FAM * K_BASE).astype(int)
         + DET._sustained(ew("sp", "30min"), DET.THR_SPREAD * K_BASE).astype(int)
         + DET._sustained(ew("vb", "30min"), 3.0 * K_VIB).astype(int))
    return (n >= 2) & mask


def ic_poisson(k, exp):
    lo = stats.chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
    return lo / exp, stats.chi2.ppf(0.975, 2 * (k + 1)) / 2 / exp


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask_base = mascara_pontuacao(df)
    jan48 = [(t - pd.Timedelta(hours=48), t) for t in falhas]

    def mede(alerta, mask):
        eps = A.episodios(alerta)
        fp = [(a, b) for a, b in eps if not any((a <= t1) and (b >= t0) for t0, t1 in jan48)]
        meses = mask.sum() * 2 / 60.0 / 730.0
        det = [t for t in falhas
               if alerta[(alerta.index >= t - pd.Timedelta(hours=48)) & (alerta.index < t)].any()]
        return len(fp), meses, det

    print("=== base: referencia mensal (o que temos hoje) ===", flush=True)
    out_b = roda(BRACO, df, falhas)
    al_b = alerta_de(out_b, mask_base)
    fp_b, meses_b, det_b = mede(al_b, mask_base)
    lo, hi = ic_poisson(fp_b, meses_b)
    print(f"  FP={fp_b} ({fp_b/meses_b:.2f}/mes, IC95% [{lo:.2f}, {hi:.2f}])  "
          f"deteccao={len(det_b)}/{len(falhas)}  {[t.strftime('%Y-%m-%d') for t in det_b]}\n")

    linhas = []
    for boot_h in BOOTS:
        print(f"=== referencia por campanha, boot={boot_h:.0f}h ===", flush=True)
        out_c, pont = roda_campanha(df, falhas, boot_h)
        mask_c = mask_base & pont
        al_c = alerta_de(out_c, mask_c)
        fp_c, meses_c, det_c = mede(al_c, mask_c)
        lo_c, hi_c = ic_poisson(fp_c, meses_c)
        perdidos = [t.strftime("%Y-%m-%d") for t in falhas if t not in det_c]
        sem_cobertura = [t.strftime("%Y-%m-%d") for t in falhas
                         if not pont[(pont.index >= t - pd.Timedelta(hours=48)) & (pont.index < t)].any()]
        print(f"  FP={fp_c} ({fp_c/meses_c:.2f}/mes, IC95% [{lo_c:.2f}, {hi_c:.2f}])  "
              f"deteccao={len(det_c)}/{len(falhas)}")
        print(f"  perdidos: {perdidos}")
        print(f"  DOS QUAIS sem cobertura nenhuma (campanha curta): {sem_cobertura}\n", flush=True)
        linhas.append(dict(boot_h=boot_h, fp=fp_c, fp_mes=fp_c/meses_c, ic_lo=lo_c, ic_hi=hi_c,
                            deteccao=len(det_c), meses=meses_c,
                            perdidos=",".join(perdidos), sem_cobertura=",".join(sem_cobertura)))
        if boot_h == 12.0:
            out_c.to_parquet("ref_campanha_out.parquet")
            pont.to_frame("pontuavel").to_parquet("ref_campanha_pont.parquet")

    R = pd.DataFrame(linhas)
    R.to_csv("referencia_campanha.csv", index=False)
    print("--- resumo ---")
    print(f"base: {fp_b} FP em {meses_b:.1f} meses ({fp_b/meses_b:.2f}/mes), {len(det_b)}/9")
    print(R[["boot_h", "fp", "fp_mes", "ic_lo", "ic_hi", "deteccao", "meses"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))


main()
