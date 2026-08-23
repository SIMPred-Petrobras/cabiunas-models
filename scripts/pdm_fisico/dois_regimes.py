#!/usr/bin/env python3
"""Pos-partida como regime proprio: sustentacao mais longa, NAO limiar mais baixo.

Motivacao (fp_rajadas.py): 48% dos FP (40 de 84) comecam a menos de 30h da
ultima partida, varios exatamente em h=6.5h -- o instante em que o blackout de
6h expira. A janela pos-partida e simultaneamente onde mais erra pra mais (FP)
e onde mais erra pra menos (2025-11-04). A rampa falhou porque atacou um
piorando o outro.

Hipotese fisica: o que dispara logo apos a partida e transiente de acomodacao
(filme de oleo estabelecendo, equalizacao termica, carga subindo) -- passageiro
por natureza. Degradacao real persiste. Entao o discriminante certo nessa
janela e TEMPO DE PERSISTENCIA, nao amplitude.

Implementacao: onde h_desde_partida < JANELA, exige-se SUSTAIN*fator amostras
consecutivas acima do limiar em vez de SUSTAIN. Limiar inalterado.

Avaliacao assimetrica, de proposito:
  - FP: n=84 episodios em 13 mil horas -> poder real, IC de Poisson estreito.
  - deteccao: n=8, poder baixo (ja quantificado) -> reportada, mas o criterio
    de aceitacao e "nao degrada", nao "melhora".
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
from cabiunas_pdm import detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import BRACO

K_BASE, K_VIB = 1.3, 5.5
JANELAS = [12.0, 24.0, 48.0]      # horas pos-partida tratadas como regime proprio
FATORES = [1.0, 2.0, 3.0, 4.0]    # multiplicador da sustentacao nessa janela


def horas_desde_partida(df):
    op = df["in_operation"].astype(bool)
    starts = op & ~op.shift(fill_value=False)
    tempo = pd.Series(df.index, index=df.index)
    marca = tempo.where(starts).ffill()
    return (tempo - marca).dt.total_seconds().div(3600.0).fillna(1e6)


def sustenta_2regimes(acima: pd.Series, n_normal: int, n_pos: int,
                       pos: pd.Series) -> pd.Series:
    """Sustentacao normal fora da janela; sustentacao mais longa dentro dela."""
    a = acima.fillna(False).astype(int)
    s_norm = a.rolling(n_normal, min_periods=n_normal).sum() >= n_normal
    s_pos = a.rolling(n_pos, min_periods=n_pos).sum() >= n_pos
    return np.where(pos.to_numpy(), s_pos.to_numpy(), s_norm.to_numpy())


def alerta_2regimes(out, mask, h_part, janela_h, fator):
    idx = out.index
    pos = (h_part < janela_h) & mask
    n_norm = DET.SUSTAIN
    n_pos = int(round(DET.SUSTAIN * fator))

    def ew(c, hl):
        return out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)

    partes = [
        sustenta_2regimes(ew("t", "1h") > DET.THR_FAM * K_BASE, n_norm, n_pos, pos),
        sustenta_2regimes(ew("p", "1h") > DET.THR_FAM * K_BASE, n_norm, n_pos, pos),
        sustenta_2regimes(ew("sp", "30min") > DET.THR_SPREAD * K_BASE, n_norm, n_pos, pos),
        sustenta_2regimes(ew("vb", "30min") > 3.0 * K_VIB, n_norm, n_pos, pos),
    ]
    n = sum(p.astype(int) for p in partes)
    return pd.Series(n >= 2, index=idx) & mask


def ic_poisson(k, exposicao):
    lo = stats.chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
    hi = stats.chi2.ppf(0.975, 2 * (k + 1)) / 2
    return lo / exposicao, hi / exposicao


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)
    h_part = horas_desde_partida(df)
    out = roda(BRACO, df, falhas)

    meses_op = mask.sum() * 2 / 60.0 / 730.0
    jan48 = [(t - pd.Timedelta(hours=48), t) for t in falhas]

    def mede(alerta):
        eps = A.episodios(alerta)
        fp = [(a, b) for a, b in eps
              if not any((a <= t1) and (b >= t0) for t0, t1 in jan48)]
        x = A.avalia(alerta[mask], falhas, mask[mask])
        h_fp = sum((b - a).total_seconds() / 3600 + 2 / 60 for a, b in fp)
        return len(fp), x["det"], h_fp

    print(f"exposicao: {meses_op:.1f} meses-operacao\n")
    print(f"{'janela':>7} {'fator':>6} {'sustent.':>9} {'FP':>4} {'FP/mes':>7} "
          f"{'IC95% FP/mes':>16} {'h alarme/mes':>12} {'deteccao':>9}")
    print("-" * 82)

    linhas = []
    base_fp = base_det = None
    for janela_h in JANELAS:
        for fator in FATORES:
            al = alerta_2regimes(out, mask, h_part, janela_h, fator)
            fp, det, h_fp = mede(al)
            lo, hi = ic_poisson(fp, meses_op)
            if fator == 1.0 and base_fp is None:
                base_fp, base_det = fp, det
            marca = "  <- base" if fator == 1.0 and janela_h == JANELAS[0] else ""
            print(f"{janela_h:7.0f} {fator:6.1f} {int(DET.SUSTAIN*fator)*2:7d}min "
                  f"{fp:4d} {fp/meses_op:7.2f} [{lo:5.2f}, {hi:5.2f}] "
                  f"{h_fp/meses_op:12.1f} {det:6d}/{len(falhas)}{marca}", flush=True)
            linhas.append(dict(janela_h=janela_h, fator=fator, fp=fp,
                                fp_mes=fp / meses_op, ic_lo=lo, ic_hi=hi,
                                h_fp_mes=h_fp / meses_op, deteccao=det))

    R = pd.DataFrame(linhas)
    R.to_csv("dois_regimes.csv", index=False)

    print(f"\nbase (sem regime separado): {base_fp} FP, {base_det}/{len(falhas)} detectados")
    lo_b, hi_b = ic_poisson(base_fp, meses_op)
    print(f"  IC95% base: [{lo_b:.2f}, {hi_b:.2f}] FP/mes")
    print("\ncandidatos que reduzem FP SEM perder deteccao:")
    bons = R[(R.fp < base_fp) & (R.deteccao >= base_det)]
    if bons.empty:
        print("  nenhum -- toda reducao de FP custou deteccao")
    else:
        for _, r in bons.sort_values("fp").iterrows():
            reducao = 100 * (1 - r.fp / base_fp)
            sep = "sim" if r.ic_hi < lo_b else "nao (ICs se sobrepoem)"
            print(f"  janela={r.janela_h:.0f}h fator={r.fator:.0f}x -> {int(r.fp)} FP "
                  f"({reducao:.0f}% menos), deteccao {int(r.deteccao)}/{len(falhas)}, "
                  f"IC separado do base: {sep}")


main()
