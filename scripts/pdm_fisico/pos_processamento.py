#!/usr/bin/env python3
"""O par (refratario, duracao minima) varrido junto, e o refratario COM ESCALADA.

POR QUE AQUI. Depois de nao_decaimento.py, o caminho geometrico esta fechado: a cauda do
transiente de religamento e a falha de religamento tem a mesma geometria nos quatro
sinais, e toda regra que remove uma remove a outra. Sobra o pos-processamento, que e
onde o custo e efetivamente contado -- o orcamento e por EPISODIO, e quem decide o que
vira episodio e o par (REFRAT_H, DUR_MIN), nunca varrido em conjunto desde que o
refratario entrou.

O QUE O REFRATARIO FAZ HOJE. Depois de um alarme, 48 h em que qualquer episodio novo e
descartado. Foi o que resolveu a deriva do custo (que era REPETICAO, nao taxa). Mas ele
e cego a intensidade: descarta igualmente uma repeticao do mesmo ruido e uma escalada
real. Isso poe um teto em quanto se pode alonga-lo -- alongar corta falso positivo, mas
em algum ponto come uma deteccao que veio logo depois de um alarme fraco.

A IDEIA NOVA: REFRATARIO COM ESCALADA. Um episodio dentro da janela refrataria e
admitido assim que apresentar MAIS sinais simultaneos do que o episodio que abriu a
janela. "Mais sinais" e a unica medida de intensidade que nao depende de escala e ja e
a moeda do voto. A leitura fisica e direta: repetir nao e noticia, PIORAR e noticia.
Se funcionar, permite alongar o refratario (menos falso positivo) sem pagar deteccao,
porque a escalada abre a porta de volta.

ARMADILHA DE PARETO. Mexer em REFRAT_H ou DUR_MIN muda deteccao e custo ao mesmo tempo.
(kb, kv) sao varridos DENTRO de cada combinacao e a leitura e a custo igualado.

CONTROLE. REFRAT_H=48, DUR_MIN=120, sem escalada, sem mancal tem que devolver a linha de
base publicada (8/8, 1,12 FP/mes, 39,0 h/mes, lead 29,0 h) bit a bit.
"""
from __future__ import annotations
import numpy as np, pandas as pd, avalia as AV
from publica_clearml import (GRID, BLACKOUT, SUSTAIN, SIN, HL, BASE, KAPPA, H_CUSUM,
                             REFRAT_H, DUR_MIN, T0, ORC_FP)
from blackout_curto import cusum

KB = [1.1, 1.3, 1.5, 1.7, 2.0, 2.4]
KV = [1.8, 2.2, 2.8]
REFRAT = [24, 48, 72, 96, 144]
DURMIN = [60, 120, 240, 360]
H_FP_BASE = 39.0

g = pd.read_parquet("grade2min.parquet")
idx = g.index
op = (g["RUNNING_A"] > 0.5).fillna(False)
estavel = op & (g["T5_AVG_A"] > 300)
part = op & ~op.shift(fill_value=False)
sel = idx >= T0
fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
alvo = pd.Series(list(fal[fal >= T0]))

n_bl = int(pd.Timedelta(BLACKOUT) / pd.Timedelta(GRID))
blk = part.rolling(n_bl, min_periods=1).max().astype(bool)
mask = (estavel & ~blk) & sel
reset = ((~mask) | part).to_numpy()

z = np.load("piso_fisico_cache.npz")
spv = np.abs((z["b_all"] - z["med_sp"]) / z["mad_sp"])
with np.errstate(invalid="ignore", divide="ignore"):
    Zv = np.abs((z["Xh"] - z["MED"]) / z["S"])
vbv = np.full(len(idx), np.nan)
vbv[z["hot"]] = np.nanmax(np.where(np.isfinite(Zv), Zv, -np.inf), axis=1)
vbv[~np.isfinite(vbv)] = np.nan
cru = pd.DataFrame({"t": z["t"], "p": z["p"], "sp": spv, "vb": vbv}, index=idx)
EW = {c: cru[c].ewm(halflife=pd.Timedelta(h), times=idx).mean() for c, h in HL.items()}


def partes(kb, kv):
    K = {"t": kb, "p": kb, "sp": kb, "vb": kv}
    out = {}
    for c in SIN:
        thr = BASE[c] * K[c]
        E = EW[c].where(mask)
        deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        cu = pd.Series(cusum(((E / thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                             reset) > H_CUSUM, index=idx)
        out[c] = (deg | cu) & mask
    return out


def pos(voto, n_sin, refrat_h, dur_min, escalada):
    """Refratario + duracao minima. Com escalada, a janela refrataria e furada por um
    episodio que traga mais sinais simultaneos do que o que a abriu."""
    al = pd.Series(False, index=idx)
    bloq, forca = None, 0
    for a, b in AV.episodios(voto):
        pico = int(n_sin.loc[a:b].max())
        if bloq is not None and a <= bloq:
            if not (escalada and pico > forca):
                continue
        al.loc[a:b] = True
        bloq = b + pd.Timedelta(hours=refrat_h)
        forca = pico if not escalada else max(forca, pico)
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        if (b - a).total_seconds() / 60 + 2 >= dur_min:
            fin.loc[a:b] = True
    return fin & sel


def mede(ON, refrat_h, dur_min, escalada, exige_mancal):
    n_sin = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(n_sin >= 2, index=idx) & mask
    if exige_mancal:
        v = v & (ON["sp"] | ON["vb"])
    m = AV.avalia(pos(v, n_sin, refrat_h, dur_min, escalada), alvo, mask)
    perd = sorted(set(t.strftime("%Y-%m-%d") for t in alvo) - set(m["detectados"]))
    return m, perd


if __name__ == "__main__":
    P = {(kb, kv): partes(kb, kv) for kb in KB for kv in KV}

    b, _ = mede(P[(1.7, 2.2)], REFRAT_H, DUR_MIN, False, False)
    print(f"controle: refrat=48, dur=120, sem escalada, sem mancal, k=1,7/2,2 -> "
          f"{b['det']}/8, {b['fp_mes']:.2f} FP/mes, {b['h_fp_mes']:.1f} h/mes, "
          f"lead {b['lead_med']:.1f} h  (esperado 8/8, 1,12, 39,0, 29,0)", flush=True)
    bm, _ = mede(P[(1.7, 2.2)], REFRAT_H, DUR_MIN, False, True)
    print(f"controle: idem com porta de mancal -> {bm['det']}/8, {bm['fp_mes']:.2f} FP/mes, "
          f"{bm['h_fp_mes']:.1f} h/mes  (esperado 8/8, 1,03, 38,7)\n", flush=True)

    lin = []
    for rf in REFRAT:
        for dm in DURMIN:
            for esc in (False, True):
                for (kb, kv), pr in P.items():
                    for mg in (False, True):
                        m, perd = mede(pr, rf, dm, esc, mg)
                        lin.append(dict(refrat=rf, dur_min=dm, escalada=esc, mancal=mg,
                                        kb=kb, kv=kv, det=m["det"], eps=m["episodios"],
                                        fp_mes=round(m["fp_mes"], 3),
                                        h_fp_mes=round(m["h_fp_mes"], 1),
                                        lead=round(m["lead_med"], 2) if m["det"] else np.nan,
                                        lead_min=round(m["lead_min"], 2) if m["det"] else np.nan,
                                        perdidos=",".join(perd)))
            print(f"  refrat={rf:>3d}h dur_min={dm:>3d}min  ok", flush=True)

    d = pd.DataFrame(lin)
    d.to_csv("pos_processamento.csv", index=False)

    print("\n" + "=" * 100)
    print(f"TUDO QUE MANTEM 8/8 DENTRO DO ORCAMENTO ({ORC_FP} FP/mes), do mais barato ao mais caro")
    print("=" * 100)
    oito = d[(d.det == 8) & (d.fp_mes <= ORC_FP)].sort_values(["fp_mes", "h_fp_mes"])
    if len(oito):
        print(oito.head(20)[["refrat", "dur_min", "escalada", "mancal", "kb", "kv",
                             "eps", "fp_mes", "h_fp_mes", "lead", "lead_min"]].to_string(index=False))
        w = oito.iloc[0]
        print(f"\n  melhor: refrat={w.refrat}h dur_min={w.dur_min}min escalada={w.escalada} "
              f"mancal={w.mancal} k={w.kb}/{w.kv}")
        print(f"          8/8, {w.fp_mes:.2f} FP/mes, {w.h_fp_mes:.1f} h/mes, lead {w.lead:.1f} h")
        print(f"          contra a linha de base: 8/8, 1,12 FP/mes, 39,0 h/mes, lead 29,0 h")
    else:
        print("  nenhuma combinacao mantem 8/8 dentro do orcamento")

    print("\n" + "=" * 100)
    print("A ESCALADA AJUDA? mesmo (refrat, dur_min, mancal, k), so muda escalada")
    print("=" * 100)
    ch = ["refrat", "dur_min", "mancal", "kb", "kv"]
    a = d[~d.escalada].set_index(ch); e = d[d.escalada].set_index(ch)
    j = a.join(e, lsuffix="_n", rsuffix="_e")
    dif = j[(j.det_e != j.det_n) | (j.fp_mes_e != j.fp_mes_n)]
    print(f"  combinacoes em que a escalada muda algo: {len(dif)}/{len(j)}")
    if len(dif):
        print(f"  ganha deteccao: {int((dif.det_e > dif.det_n).sum())}   "
              f"perde deteccao: {int((dif.det_e < dif.det_n).sum())}   "
              f"delta medio de FP/mes: {(dif.fp_mes_e - dif.fp_mes_n).mean():+.3f}")
        g8 = dif[(dif.det_e == 8) & (dif.det_n < 8)]
        print(f"  combinacoes em que a escalada RECUPERA o 8/8: {len(g8)}")
        if len(g8):
            print(g8.reset_index()[ch + ["det_n", "fp_mes_n", "det_e", "fp_mes_e",
                                         "h_fp_mes_e", "lead_e"]].head(15).to_string(index=False))
    print("\n-> pos_processamento.csv")
