#!/usr/bin/env python3
"""Ajuste em DOIS REGIMES: um modelo de normalidade proprio para o pos-religamento.

De onde vem. blackout_curto.py mostrou que encurtar o blackout nao compra deteccao,
so custo: com <=3 h nenhuma das 18 combinacoes de limiar cabe no teto de 1,15 FP/mes.
O diagnostico explicou por que -- o pos-religamento e sistematicamente mais alto:

    sinal   mediana pos/estavel   % acima do limiar (pos vs estavel)
    t              1,20x                 19,7% vs 12,7%
    p              1,27x                 12,9% vs  8,4%
    sp             0,70x                 16,5% vs 20,1%
    vb             2,19x                 62,3% vs 30,8%

E um DESLOCAMENTO DE NIVEL, nao ruido. O modelo atual ja ve essas amostras (3,8% do
treino), mas a 3,8% o PCA nao gasta componente com elas.

A hipotese: se "normal para maquina que acabou de religar" for medido contra OUTROS
pos-religamentos, o transiente deixa de parecer anomalia e as 309 h hoje apagadas
voltam a ser pontuaveis sem o diluvio de falso positivo.

ISOLAMENTO DA VARIAVEL -- o que a primeira versao deste script errou. Ela reajustou
tambem o ramo estavel (fit sobre `steady` em vez de `est`), mudando duas coisas de uma
vez: o cenario de 6 h deixou de reproduzir o 8/8 e nenhum numero valia. Aqui as linhas
estaveis usam O PROPRIO CACHE, bit a bit. So as linhas de transiente sao recalculadas.
O controle e duro: com blackout de 6 h o resultado TEM que ser 8/8 · 1,12 · 39,0.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from publica_clearml import (GRID, SUSTAIN, SIN, HL, BASE, KAPPA, H_CUSUM, CARGA,
                             REFRAT_H, DUR_MIN, T0)
from autocalibra import ScorerMax, TEMP, PRESS, MANCAL, ALVO_SP
from blackout_curto import cusum, pos as pos_bl

N_TRANS_MIN, N_TRANS_MAX = 400, 20_000
KB = [1.1, 1.3, 1.5, 1.7, 2.0, 2.4]
KV = [1.8, 2.2, 2.8]
ORC_FP = 1.15

g = pd.read_parquet("grade2min.parquet"); idx = g.index
op = (g["RUNNING_A"] > 0.5).fillna(False)
est = op & (g["T5_AVG_A"] > 300)
part = op & ~op.shift(fill_value=False)
blk = part.rolling(int(pd.Timedelta("6h") / pd.Timedelta(GRID)), min_periods=1).max().astype(bool)
trans, steady = est & blk, est & ~blk
sel = idx >= T0
fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
alvo = pd.Series(list(fal[fal >= T0]))

# ---------------------------------------------- ramo estavel = cache, sem tocar
z = np.load("piso_fisico_cache.npz")
hot, Xh, MEDv, Sv = z["hot"], z["Xh"], z["MED"], z["S"]
spread = (g[ALVO_SP] - g[MANCAL].median(axis=1)).abs()
sp_cache = np.abs((z["b_all"] - z["med_sp"]) / z["mad_sp"])
with np.errstate(invalid="ignore", divide="ignore"):
    Zc = np.abs((Xh - MEDv) / Sv)
vb_cache = np.full(len(idx), np.nan)
vb_cache[hot] = np.nanmax(np.where(np.isfinite(Zc), Zc, -np.inf), axis=1)
vb_cache[~np.isfinite(vb_cache)] = np.nan
UM = pd.DataFrame({"t": z["t"], "p": z["p"], "sp": sp_cache, "vb": vb_cache}, index=idx)

# ---------------------------------------------- ramo transiente: modelo proprio
DOIS = UM.copy()
tr_hot = trans.to_numpy()[hot]
meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
n_ok = 0
print("ajuste do modelo de transiente, mes a mes (expansivo, so pos-religamento) ...",
      flush=True)
for i, m0 in enumerate(meses):
    m1 = meses[i + 1] if i + 1 < len(meses) else idx[-1] + pd.Timedelta(GRID)
    mreg = ((idx >= m0) & (idx < m1) & trans.to_numpy())
    if not mreg.any():
        continue
    ft = g.loc[trans & (idx < m0), TEMP + PRESS].dropna().tail(N_TRANS_MAX)
    if len(ft) < N_TRANS_MIN:
        continue
    n_ok += 1
    w = g.loc[mreg]
    DOIS.loc[mreg, "t"] = ScorerMax().fit(ft[TEMP]).score(w[TEMP])
    DOIS.loc[mreg, "p"] = ScorerMax().fit(ft[PRESS]).score(w[PRESS])
    bt = spread.loc[ft.index]
    mt = float(bt.median()); dt = float((bt - bt.median()).abs().median() * 1.4826)
    DOIS.loc[mreg, "sp"] = (spread[mreg] - mt).abs() / dt
    jt = np.flatnonzero(tr_hot & (idx[hot] < m0))[-N_TRANS_MAX:]
    med_t = np.nanmedian(Xh[jt], axis=0)
    s_t = np.nanmedian(np.abs(Xh[jt] - med_t), axis=0) * 1.4826
    s_t = np.where(np.isfinite(s_t) & (s_t > 0), s_t, np.nan)
    k = np.flatnonzero(mreg[hot])
    if len(k):
        with np.errstate(invalid="ignore", divide="ignore"):
            Zk = np.abs((Xh[k] - med_t) / s_t)
        v = np.nanmax(np.where(np.isfinite(Zk), Zk, -np.inf), axis=1)
        v[~np.isfinite(v)] = np.nan
        DOIS.iloc[hot[k], DOIS.columns.get_loc("vb")] = v
print(f"  {n_ok} meses com transiente suficiente\n", flush=True)

fin = np.isfinite(UM.to_numpy()) & np.isfinite(DOIS.to_numpy())
ms = steady.to_numpy() & sel
print("controle A -- nas linhas ESTAVEIS os dois escores tem que ser identicos:")
for j, c in enumerate(SIN):
    m = ms & fin[:, j]
    print(f"   {c:4s} diferenca maxima = {np.abs(UM[c].to_numpy()[m] - DOIS[c].to_numpy()[m]).max():.3e}")
mt_ = trans.to_numpy() & sel
print("\nefeito nas linhas de TRANSIENTE (mediana do escore, um regime -> dois):")
for c in SIN:
    a = UM[c].where(mt_).median(); b = DOIS[c].where(mt_).median()
    print(f"   {c:4s} {a:8.2f} -> {b:8.2f}   ({b/a:5.2f}x)   limiar {BASE[c]*1.7 if c!='vb' else BASE[c]*2.2:.2f}")


def roda(S, mask, kb, kv):
    K = {"t": kb, "p": kb, "sp": kb, "vb": kv}
    reset = ((~mask) | part).to_numpy()
    ON = {}
    for c in SIN:
        thr = BASE[c] * K[c]
        E = S[c].ewm(halflife=pd.Timedelta(HL[c]), times=idx).mean().where(mask)
        deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        cu = pd.Series(cusum(((E / thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                             reset) > H_CUSUM, index=idx)
        ON[c] = (deg | cu) & mask
    v = pd.Series(sum(ON[c].astype(int) for c in SIN) >= 2, index=idx) & mask
    return AV.avalia(pos_bl(v), alvo, mask)


b = roda(UM, steady & sel, 1.7, 2.2)
print(f"\ncontrole B -- um regime, blackout 6 h: {b['det']}/8 · {b['fp_mes']:.2f} FP/mes · "
      f"{b['h_fp_mes']:.1f} h/mes  (esperado 8/8 · 1,12 · 39,0)")
assert (b["det"], round(b["fp_mes"], 2), round(b["h_fp_mes"], 1)) == (8, 1.12, 39.0)
b2 = roda(DOIS, steady & sel, 1.7, 2.2)
print(f"controle C -- dois regimes, blackout 6 h: {b2['det']}/8 · {b2['fp_mes']:.2f} · "
      f"{b2['h_fp_mes']:.1f}")
print("   (NAO tem que ser igual ao B: o EWMA e calculado sobre a serie inteira e so")
print("    depois mascarado, entao o escore do transiente atravessa o blackout e chega")
print("    nas linhas estaveis seguintes. O blackout mascara a DECISAO, nao a MEMORIA.)\n")

lin = []
for nome, S, mask in [("dois regimes · blackout 0", DOIS, est & sel),
                      ("um regime   · blackout 0", UM, est & sel),
                      ("dois regimes · blackout 6 h", DOIS, steady & sel),
                      ("um regime   · blackout 6 h", UM, steady & sel)]:
    for kb in KB:
        for kv in KV:
            m = roda(S, mask, kb, kv)
            perd = sorted(set(t.strftime("%Y-%m-%d") for t in alvo) - set(m["detectados"]))
            lin.append(dict(cenario=nome, kb=kb, kv=kv, det=m["det"], eps=m["episodios"],
                            fp_mes=round(m["fp_mes"], 3), h_fp_mes=round(m["h_fp_mes"], 1),
                            lead=round(m["lead_med"], 2) if m["det"] else np.nan,
                            perdidos=",".join(perd)))
    print(f"  {nome}: 18 pontos", flush=True)
np.savez_compressed("dois_regimes_escores.npz",
                    **{f"um_{c}": UM[c].to_numpy() for c in SIN},
                    **{f"dois_{c}": DOIS[c].to_numpy() for c in SIN})
d = pd.DataFrame(lin); d.to_csv("ajuste_dois_regimes.csv", index=False)

print("\n" + "=" * 96)
print("O MODELO PROPRIO DO TRANSIENTE PAGA O BLACKOUT?")
print("=" * 96)
print(f"{'cenario':28s} {'melhor no teto 1,15':>20s} {'fp':>6s} {'h/mes':>7s} {'lead':>6s} | "
      f"{'a ~39 h/mes':>13s} {'fp':>6s}  perdidos")
for nome, s in d.groupby("cenario", sort=False):
    a = s[s.fp_mes <= ORC_FP].sort_values(["det", "h_fp_mes"], ascending=[False, True])
    q = s.iloc[(s.h_fp_mes - 39.0).abs().argmin()]
    ta = f"{a.iloc[0].det}/8 (k {a.iloc[0].kb}/{a.iloc[0].kv})" if len(a) else "   --"
    fa = f"{a.iloc[0].fp_mes:.2f}" if len(a) else "  --"
    ha = f"{a.iloc[0].h_fp_mes:.1f}" if len(a) else "  --"
    la = f"{a.iloc[0].lead:.1f}" if len(a) and a.iloc[0].det else "  --"
    print(f"{nome:28s} {ta:>20s} {fa:>6s} {ha:>7s} {la:>6s} | "
          f"{q.det:>3d}/8 (k {q.kb}/{q.kv}) {q.fp_mes:>6.2f}  {q.perdidos}")
print(f"\nreferencia publicada (um regime, blackout 6 h): 8/8 · 1,12 FP/mes · 39,0 h/mes · lead 29,0 h")
print("\n-> ajuste_dois_regimes.csv")
