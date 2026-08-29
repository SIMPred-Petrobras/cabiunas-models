#!/usr/bin/env python3
"""O blackout pos-religamento de 6 h: quanto ele custa, e o que aparece se encurtar.

De onde vem a pergunta. A nossa regua aplicada a serie do detector PCA (regua_neles.py)
mostrou que as duas deteccoes daquele detector caem INTEIRAMENTE dentro do nosso
blackout: 4,1 h no evento de 07/04/2025 e 1,0 h no de 04/11/2025, zero hora fora dele.
Nao e o portao de T5; e o blackout. Ele descarta uma classe de evidencia -- maquina que
acabou de religar e ja esta degradando -- que o outro detector usa.

A ARMADILHA QUE ESTE SCRIPT EVITA. Encurtar o blackout aumenta as horas pontuadas, o
que mexe em deteccao E em custo ao mesmo tempo. Comparar a `k` fixo mede sensibilidade,
nao blackout -- a armadilha de Pareto, que ja nos pegou sete vezes. Aqui cada blackout
tem `kb` (t, p, sp) e `kv` (vb) varridos, e a comparacao e a CUSTO IGUALADO alem do
melhor que cada um alcanca.

O QUE NAO VARIA. Os sinais vem do cache: o PCA foi ajustado sobre `stable` do pacote
original e a referencia rolante da vibracao idem. Este teste move o blackout apenas na
MASCARA DE PONTUACAO e no reset do CUSUM -- que e exatamente a pergunta feita ("o que o
detector deixa de ver"), mas nao refaz a janela de ajuste.
"""
from __future__ import annotations
import numpy as np, pandas as pd, avalia as AV
from publica_clearml import (GRID, SUSTAIN, SIN, HL, BASE, KAPPA, H_CUSUM, CARGA,
                             REFRAT_H, DUR_MIN, T0)

BLACKOUTS = ["0min", "1h", "2h", "3h", "4h", "6h", "9h", "12h"]
KB = [1.1, 1.3, 1.5, 1.7, 2.0, 2.4]
KV = [1.8, 2.2, 2.8]
ORC_FP = 1.15          # o mesmo orcamento usado na selecao do LOEO publicado

g = pd.read_parquet("grade2min.parquet")
idx = g.index
op = (g["RUNNING_A"] > 0.5).fillna(False)
estavel = op & (g["T5_AVG_A"] > 300)
part = op & ~op.shift(fill_value=False)
sel = idx >= T0
fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
alvo = pd.Series(list(fal[fal >= T0]))

z = np.load("piso_fisico_cache.npz")
spv = np.abs((z["b_all"] - z["med_sp"]) / z["mad_sp"])
with np.errstate(invalid="ignore", divide="ignore"):
    Z = np.abs((z["Xh"] - z["MED"]) / z["S"])
vbv = np.full(len(idx), np.nan)
vbv[z["hot"]] = np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1)
vbv[~np.isfinite(vbv)] = np.nan
cru = pd.DataFrame({"t": z["t"], "p": z["p"], "sp": spv, "vb": vbv}, index=idx)
EW = {c: cru[c].ewm(halflife=pd.Timedelta(h), times=idx).mean() for c, h in HL.items()}


def cusum_lento(x, reset):
    S = np.empty(len(x)); a = 0.0
    for i in range(len(x)):
        a = a * CARGA if reset[i] else max(0.0, a + x[i]); S[i] = a
    return S


def cusum(x, reset):
    """Mesma recursao, vetorizada por trecho entre resets.

    Dentro de um trecho sem reset, S_i = max(0, S_{i-1} + x_i) tem forma fechada:
    S_i = C_i + max(a0, -min(0, min_{k<i} C_k)), com C o cumsum de x no trecho. Os
    resets (a <- a*CARGA, sem somar x) sao aplicados nas bordas. Verificado contra
    a versao lenta antes de qualquer varredura."""
    S = np.empty(len(x)); a = 0.0
    corte = np.flatnonzero(reset)
    ini = 0
    bordas = list(corte) + [len(x)]
    for b in bordas:
        if b > ini:
            xs = x[ini:b]
            C = np.cumsum(xs)
            # o minimo tem que incluir C_i (j <= i), nao so ate i-1 -- foi onde a
            # primeira versao divergiu, e o assert de controle pegou
            prefmin = np.minimum(np.minimum.accumulate(C), 0.0)
            S[ini:b] = C + np.maximum(a, -prefmin)
            a = S[b - 1]
        if b < len(x):
            a = a * CARGA; S[b] = a; ini = b + 1
    return S


def pos(voto):
    al = pd.Series(False, index=idx); bloq = None
    for a, b in AV.episodios(voto):
        if bloq is not None and a <= bloq:
            continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=REFRAT_H)
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        if (b - a).total_seconds() / 60 + 2 >= DUR_MIN:
            fin.loc[a:b] = True
    return fin & sel


def roda(mask, reset, E, kb, kv):
    K = {"t": kb, "p": kb, "sp": kb, "vb": kv}
    ON = {}
    for c in SIN:
        thr = BASE[c] * K[c]
        deg = ((E[c] > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        cu = pd.Series(cusum(((E[c] / thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                             reset) > H_CUSUM, index=idx)
        ON[c] = (deg | cu) & mask
    v = pd.Series(sum(ON[c].astype(int) for c in SIN) >= 2, index=idx) & mask
    return AV.avalia(pos(v), alvo, mask)


if __name__ == "__main__":
    # ---- controle: a versao vetorizada tem que dar exatamente a lenta ----------
    n_bl = int(pd.Timedelta("6h") / pd.Timedelta(GRID))
    blk = part.rolling(n_bl, min_periods=1).max().astype(bool)
    m6 = (estavel & ~blk) & sel
    r6 = ((~m6) | part).to_numpy()
    xt = ((EW["t"].where(m6) / (BASE["t"] * 1.7)).clip(upper=20) - KAPPA).fillna(0.0).to_numpy()
    assert np.allclose(cusum(xt, r6), cusum_lento(xt, r6)), "CUSUM vetorizado divergiu"
    print("controle: CUSUM vetorizado == versao lenta  OK", flush=True)

    base = roda(m6, r6, {c: EW[c].where(m6) for c in SIN}, 1.7, 2.2)
    print(f"controle: blackout 6h a k=1,7/2,2 -> {base['det']}/8, {base['fp_mes']:.2f} FP/mes, "
          f"{base['h_fp_mes']:.1f} h/mes  (esperado 8/8, 1,12, 39,0)\n", flush=True)

    lin = []
    for bl in BLACKOUTS:
        n = int(pd.Timedelta(bl) / pd.Timedelta(GRID))
        blk = (part.rolling(n, min_periods=1).max().astype(bool) if n > 1
               else pd.Series(False, index=idx))
        mask = (estavel & ~blk) & sel
        reset = ((~mask) | part).to_numpy()
        E = {c: EW[c].where(mask) for c in SIN}
        ho = mask.sum() * 2 / 60
        for kb in KB:
            for kv in KV:
                m = roda(mask, reset, E, kb, kv)
                perd = sorted(set(t.strftime("%Y-%m-%d") for t in alvo) - set(m["detectados"]))
                lin.append(dict(blackout=bl, horas_op=round(ho, 1), kb=kb, kv=kv,
                                det=m["det"], eps=m["episodios"], fp_mes=round(m["fp_mes"], 3),
                                h_fp_mes=round(m["h_fp_mes"], 1),
                                lead=round(m["lead_med"], 2) if m["det"] else np.nan,
                                perdidos=",".join(perd)))
        print(f"  {bl:>6s}  {ho:7,.0f} h pontuadas  ({18} pontos)", flush=True)

    d = pd.DataFrame(lin)
    d.to_csv("blackout_curto.csv", index=False)

    print("\n" + "=" * 96)
    print("MELHOR DETECCAO DENTRO DO ORCAMENTO DE FP (<= 1,15/mes) E A CUSTO IGUALADO (39,0 h/mes)")
    print("=" * 96)
    print(f"{'blackout':>9s} {'h op':>9s} | {'melhor det':>11s} {'fp':>6s} {'h/mes':>7s} {'lead':>6s} | "
          f"{'det a 39 h/mes':>15s} {'fp':>6s} {'lead':>6s}  perdidos")
    for bl in BLACKOUTS:
        s = d[d.blackout == bl]
        a = s[s.fp_mes <= ORC_FP]
        a = a.sort_values(["det", "h_fp_mes"], ascending=[False, True]).iloc[0] if len(a) else None
        b = s.iloc[(s.h_fp_mes - 39.0).abs().argmin()]
        ta = (f"{a.det}/8 (k {a.kb}/{a.kv})" if a is not None else "  -- ")
        fa = f"{a.fp_mes:.2f}" if a is not None else "  -- "
        ha = f"{a.h_fp_mes:.1f}" if a is not None else "  -- "
        la = f"{a.lead:.1f}" if a is not None and a.det else "  -- "
        print(f"{bl:>9s} {s.horas_op.iloc[0]:8,.0f}h | {ta:>11s} {fa:>6s} {ha:>7s} {la:>6s} | "
              f"{b.det:>3d}/8 (k {b.kb}/{b.kv}) {b.fp_mes:>6.2f} {b.lead:>6.1f}  {b.perdidos}")
    print("\n-> blackout_curto.csv")
