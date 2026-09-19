#!/usr/bin/env python3
"""DINAMICA em vez de NIVEL -- a taxa de subida do sinal como canal.

O DIAGNOSTICO QUE ISTO ATACA. Os quatro canais medem NIVEL de desvio, e ficam
acesos de 24% a 52% do tempo. O voto satura, os episodios fundem, o nascimento sai
da janela de 48 h -- e e por isso que a regua de inicio pune o detector. Os canais
do Diego, para contraste, ficam em 0,5% a 11%, e a diferenca nao e o limiar: sao
estatisticas multiescala, invariantes a deriva lenta
([[cadencia-de-retreino-depende-do-sinal]]).

A IDEIA. Nunca medimos a DERIVADA dos nossos proprios sinais. Uma degradacao real
nao so esta alta -- ela esta SUBINDO. E a taxa tem duas propriedades que o nivel
nao tem:

  * nao precisa de baseline absoluto, entao e imune ao drift do `recon_p99` que
    dominou toda a investigacao anterior ([[o-normalizador-e-o-ponto-fragil]]);
  * tem duty naturalmente baixo: na maior parte do tempo a taxa e ~0, enquanto o
    nivel fica alto por semanas depois de qualquer desvio.

Tres formas medidas, todas sobre os sinais crus (antes do EWMA do detector):

  slope    inclinacao por minimos quadrados em janela movel, normalizada pelo
           MAD das inclinacoes no baseline -- "esta subindo mais rapido que o
           normal?"
  delta    variacao sobre a mediana da janela anterior -- mais robusto que slope
           a um unico ponto fora
  vol      desvio-padrao robusto em janela movel -- "o sinal ficou agitado?",
           que e outra coisa que nivel nao captura

CONTROLE POSITIVO: os quatro canais de NIVEL entram na mesma tabela. Se a dinamica
nao aparecer mas o nivel aparecer, a dinamica nao serve; se nenhum dos dois
aparecer, o teste esta cego e o resultado nao vale.

Uso:  PYTHONPATH=. python dinamica.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pos_processamento import cru, mask, idx, alvo, op

RNG = np.random.default_rng(20260918)
JAN = pd.Timedelta("48h")
SIN = ["t", "p", "sp", "vb"]
N_NULO = 2000
POR_H = 30


def slope(s: pd.Series, horas: float) -> pd.Series:
    """Inclinacao por minimos quadrados em janela movel, em unidades/hora.

    Forma fechada: b = cov(x, y) / var(x), com x o tempo em horas. Como o passo e
    fixo, var(x) e constante e so cov(x, y) precisa da janela movel."""
    n = int(horas * POR_H)
    x = np.arange(n) / POR_H                      # horas dentro da janela
    xm = x.mean(); varx = float(((x - xm) ** 2).sum())
    # cov(x,y) = sum((x-xm)*y) ; com peso linear fixo, e uma correlacao movel
    w = (x - xm)
    y = s.to_numpy(dtype="float64")
    out = np.full(len(y), np.nan)
    # convolucao: sum_{k} w[k] * y[i-n+1+k]
    val = np.convolve(np.nan_to_num(y, nan=0.0), w[::-1], mode="valid")
    cnt = np.convolve((~np.isnan(y)).astype(float), np.abs(w)[::-1], mode="valid")
    ok = cnt > 0.8 * np.abs(w).sum()
    out[n - 1:][ok] = val[ok] / varx
    return pd.Series(out, index=s.index)


def robusto(v: pd.Series, base_mask) -> pd.Series:
    """z robusto contra mediana/MAD do proprio sinal em regime."""
    b = v[base_mask]
    med = float(b.median()); mad = float((b - med).abs().median() * 1.4826)
    return (v - med).abs() / max(mad, 1e-9)


if __name__ == "__main__":
    m = mask.to_numpy()
    ti = np.asarray(idx.tz_convert("UTC").tz_localize(None)
                    .astype("datetime64[ns]").astype("int64"))
    jan_ns = int(JAN.value)
    eleg = ti[(ti >= ti[0] + jan_ns) & m]
    sort = RNG.choice(eleg, size=(N_NULO, len(alvo)))
    obs = np.asarray([int(pd.Timestamp(t).tz_localize(None).value) for t in alvo])

    def picos(z, quando):
        lo = np.searchsorted(ti, quando - jan_ns, "left")
        hi = np.searchsorted(ti, quando, "right")
        return np.array([np.nanmax(z[a:b]) if b > a else np.nan for a, b in zip(lo, hi)])

    def avalia(nome, v):
        z = v.where(mask).to_numpy()
        fin = np.isfinite(z[m])
        if fin.sum() < 1000:
            print(f"{nome:<22}   sem dado suficiente"); return
        lim = np.nanpercentile(z[m], 95)
        duty = float(np.nanmean(z[m] > lim))
        o = float(np.nanmedian(picos(z, obs)))
        nul = np.array([np.nanmedian(picos(z, sort[k])) for k in range(N_NULO)])
        mn = float(np.nanmedian(nul))
        p = float(np.nanmean(nul >= o))
        flag = " ***" if p < 0.05 else ("  *" if p < 0.10 else "")
        print(f"{nome:<22}{100*duty:>7.1f}%{o:>11.2f}{mn:>9.2f}"
              f"{o/max(mn,1e-9):>8.2f}x{p:>9.4f}{flag}")

    print(f"{'canal':<22}{'duty':>8}{'pico trips':>11}{'nulo':>9}{'razao':>9}{'p':>9}")
    print("-" * 68)
    print("  [NIVEL -- controle positivo, os canais de hoje]")
    for c in SIN:
        avalia(f"nivel {c}", cru[c].ewm(halflife=pd.Timedelta("1h"), times=idx).mean())

    for horas in (3, 6, 12):
        print(f"\n  [DINAMICA -- janela de {horas} h]")
        for c in SIN:
            s = cru[c].ewm(halflife=pd.Timedelta("30min"), times=idx).mean()
            avalia(f"slope {horas}h {c}", robusto(slope(s, horas), m))
        for c in SIN:
            s = cru[c].ewm(halflife=pd.Timedelta("30min"), times=idx).mean()
            n = int(horas * POR_H)
            d = s - s.shift(n).rolling(n, min_periods=n // 2).median()
            avalia(f"delta {horas}h {c}", robusto(d, m))
        for c in SIN:
            s = cru[c]
            n = int(horas * POR_H)
            vol = s.rolling(n, min_periods=n // 2).std()
            avalia(f"vol   {horas}h {c}", robusto(vol, m))
