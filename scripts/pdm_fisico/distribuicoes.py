#!/usr/bin/env python3
"""O que as medias escondem: distribuicao de lead, de duracao de episodio e de custo.

Venho reportando numeros unicos -- "lead 29,0 h", "39,0 h/mes", "1,12 FP/mes" -- e dois
deles sao MEDIAS. Media e o resumo errado quando a decisao operacional depende da cauda:
o operador nao planeja pela antecedencia media, planeja pela PIOR. E se poucas horas
longas dominam as 39 h/mes, o incomodo real e outro.

(`avalia.lead_med` calcula np.mean apesar do nome; eu vinha chamando de "mediano" na
conversa inteira. Corrigido aqui.)

Reporta, no ponto de operacao atual: lead por evento com minimo e mediana; duracao dos
episodios e quanto do custo vem dos maiores; e qual sinal abriu cada deteccao.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C, detector as DET

T0 = pd.Timestamp("2025-01-01", tz="UTC")
HL = {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min"}
BASE = {"t": DET.THR_FAM, "p": DET.THR_FAM, "sp": DET.THR_SPREAD, "vb": 3.0}
K = {"t": 1.7, "p": 1.7, "sp": 1.7, "vb": 2.2}
SIN = ["t", "p", "sp", "vb"]
JAN = pd.Timedelta(hours=48)


def episodios(al, gap_h=2.0):
    v = al.fillna(False).to_numpy()
    d = np.diff(np.r_[0, v.astype(int), 0])
    br = [(al.index[a], al.index[b-1])
          for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    if not br: return []
    out = [list(br[0])]
    for s, e in br[1:]:
        if (s - out[-1][1]) <= pd.Timedelta(hours=gap_h): out[-1][1] = e
        else: out.append([s, e])
    return [tuple(x) for x in out]


def main():
    g = pd.read_parquet("grade2min.parquet"); idx = g.index
    op = (g["RUNNING_A"] > 0.5).fillna(False)
    stable = op & (g["T5_AVG_A"] > 300)
    part = op & ~op.shift(fill_value=False)
    n_bl = int(pd.Timedelta(DET.BLACKOUT) / pd.Timedelta(C.GRID))
    sel = idx >= T0
    mask = (stable & ~part.rolling(n_bl, min_periods=1).max().astype(bool)) & sel
    fal = pd.read_csv("falhas.csv", parse_dates=["evento"])
    fal["evento"] = fal["evento"].dt.tz_convert("UTC")
    fal["modo"] = fal["alarmes"].map(lambda d: "mancal" if "Manc" in str(d)
                                     else ("oleo" if "Óleo" in str(d) else "selagem"))
    alvo = list(fal.loc[fal.evento >= T0, "evento"])
    modo = dict(zip(fal.evento, fal.modo))
    meses = mask.sum()*2/60/730

    z = np.load("piso_fisico_cache.npz")
    sp = np.abs((z["b_all"] - z["med_sp"]) / z["mad_sp"])
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.abs((z["Xh"] - z["MED"]) / z["S"])
    vbz = np.full(len(idx), np.nan)
    vbz[z["hot"]] = np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1)
    vbz[~np.isfinite(vbz)] = np.nan
    out = pd.DataFrame({"t": z["t"], "p": z["p"], "sp": sp, "vb": vbz}, index=idx)
    E = {c: out[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask) for c, h in HL.items()}
    reset = ((~mask) | part).to_numpy()

    def cus(zz, h=80, carry=0.25):
        x = (zz - 0.75).fillna(0.0).to_numpy()
        S = np.empty(len(x)); acc = 0.0
        for i in range(len(x)):
            acc = acc*carry if reset[i] else max(0.0, acc + x[i]); S[i] = acc
        return S > h
    ON = {}
    for c in SIN:
        thr = BASE[c]*K[c]; n = DET.SUSTAIN
        deg = ((E[c] > thr).astype(int).rolling(n, min_periods=n).sum() >= n)
        ON[c] = (deg | pd.Series(cus((E[c]/thr).clip(upper=20)), index=idx)) & mask
    voto = pd.Series(sum(ON[c].astype(int) for c in SIN) >= 2, index=idx) & mask
    al = pd.Series(False, index=idx); bloq = None
    for a, b in episodios(voto):
        if bloq is not None and a <= bloq: continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=48)
    fin = pd.Series(False, index=idx)
    for a, b in episodios(al):
        if (b - a).total_seconds()/60 + 2 >= 120: fin.loc[a:b] = True

    print("=" * 96); print("1. LEAD POR EVENTO -- o que a media de 29,0 h esconde"); print("=" * 96)
    print(f"\n{'evento':>17} {'modo':>9} {'lead':>8} {'sinais ativos no 1o alerta':>28}")
    leads = []
    for t in alvo:
        w = fin.loc[t - JAN:t - pd.Timedelta("2min")]
        on = w[w.fillna(False)]
        if not len(on):
            print(f"{t:%d/%m/%Y %H:%M} {modo[t]:>9} {'sem det.':>8}"); continue
        l = (t - on.index[0]).total_seconds()/3600
        leads.append(l)
        q = [c for c in SIN if ON[c].loc[on.index[0]]]
        print(f"{t:%d/%m/%Y %H:%M} {modo[t]:>9} {l:7.1f}h {'+'.join(q):>28}")
    leads = np.array(leads)
    print(f"\n  média {leads.mean():.1f} h   MEDIANA {np.median(leads):.1f} h   "
          f"MÍNIMO {leads.min():.1f} h   máximo {leads.max():.1f} h")
    print(f"  eventos com lead < 8 h (turno): {int((leads < 8).sum())}/{len(leads)}")
    print(f"  eventos com lead < 24 h (dia) : {int((leads < 24).sum())}/{len(leads)}")

    print("\n" + "=" * 96); print("2. DURACAO DOS EPISODIOS -- quanto do custo vem dos maiores")
    print("=" * 96)
    eps = episodios(fin & sel)
    jw = [(t - JAN, t) for t in alvo]
    fp = [(a, b) for a, b in eps if not any(a <= t1 and b >= t0 for t0, t1 in jw)]
    dur = np.array(sorted([(b-a).total_seconds()/3600 for a, b in fp], reverse=True))
    tot = dur.sum()
    print(f"\n  {len(fp)} falsos positivos, {tot:.0f} h no total ({tot/meses:.1f} h/mes)")
    print(f"  duracao: mediana {np.median(dur):.1f} h   p75 {np.percentile(dur,75):.1f}   "
          f"maximo {dur.max():.1f} h")
    print(f"\n  {'top N episodios':>16} {'horas':>8} {'% do custo':>12}")
    for n in [1, 2, 3, 5, len(dur)]:
        print(f"{'os '+str(n)+' maiores':>16} {dur[:n].sum():7.0f}h {100*dur[:n].sum()/tot:11.1f}%")
    print(f"\n  se o maior episodio fosse eliminado: {(tot-dur[0])/meses:.1f} h/mes "
          f"(contra {tot/meses:.1f})")

    print("\n" + "=" * 96); print("3. O CUSTO POR MES -- e uniforme?"); print("=" * 96)
    s = pd.Series(0.0, index=pd.period_range(T0, idx[-1], freq="M"))
    for a, b in fp:
        s.loc[pd.Period(a, "M")] += (b-a).total_seconds()/3600
    hm = mask.groupby(mask.index.to_period("M")).sum()*2/60
    print(f"\n{'mes':>9} {'h de FP':>9} {'h operando':>12} {'duty':>7}")
    for m in s.index:
        h_op = hm.get(m, 0.0)
        print(f"{str(m):>9} {s[m]:8.1f}h {h_op:11.0f}h "
              f"{100*s[m]/max(h_op,1e-9):6.1f}%")


main()
