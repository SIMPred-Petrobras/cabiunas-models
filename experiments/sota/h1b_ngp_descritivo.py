#!/usr/bin/env python3
"""SOTA H1(b) — VALE PEDIR NGP_A/NPT_A NO EXPORT DE PRODUÇÃO? Descritivo e barato (decisão do usuário, 22/10). Tag ClearML `sota-h1`.

PRÉ-REGISTRO (22/10/2026), COMMITADO ANTES DE RODAR. A H1 do brief (modelo de comportamento esperado a partir de variáveis de operação, detector no resíduo) precisa de
variáveis de operação que o nosso export não tem. A NGP_A (rotação do gerador de gás) e a NPT_A (rotação da turbina de potência) existem em cadência real só no dataset
ClearML "Cabiunas 2025" (424e5b58), de 01/01 a 01/11/2025 (5 dos 8 trips: os de fev-abr/2025), em hora LOCAL rotulada como UTC (medido: deslocando −3 h, o T5_AVG_A dele
coincide com o da nossa grade, correlação 1,0000). A pergunta não é "o detector melhora?" (não dá para responder com 5 trips agrupados), e sim: as rotações carregam
informação de regime que o T5 (que já temos) não carrega?
M1 VARIÂNCIA EXPLICADA FORA DA AMOSTRA. Alvo: log(1 + sinal) de cada canal do v2 (t, p, sp, vb; o sinal suavizado do cache causal), em operação vigiada. Regressão
LightGBM (parâmetros fixos) treinada em walk-forward mensal (meses anteriores, sem as janelas [T − 48 h, T + 2 d] dos trips), testada no mês seguinte, de mar a out/2025.
Conjunto A = T5_AVG_A e as variações dele em 1 h e 6 h; conjunto B = A + NGP_A e NPT_A e as variações delas em 1 h e 6 h. ΔR² = R²(B) − R²(A), no conjunto dos meses de teste.
M2 TRANSIENTES NOS NASCIMENTOS. Para cada episódio da base atual nascido entre 01/02 e 01/11/2025: a maior variação de NGP_A, NPT_A e T5 nas 2 h antes do nascimento,
contra o p95 das variações de 2 h em operação estável. "Transiente que o T5 não vê" = NGP ou NPT acima do p95 E T5 abaixo. Conta em FP e em acertos.
REGRA (fixada antes): VALE PEDIR se ΔR² >= 0,10 em t, p ou sp, OU se o transiente que o T5 não vê aparece em >= 2 dos FP e em <= 1 acerto. Senão, NÃO VALE (por este
critério): o T5 já carrega o regime que importa ao detector.
EXPECTATIVAS: ΔR² < 0,10 nos quatro canais (~60%: NGP, NPT e T5 andam juntos no regime estável); o transiente que o T5 não vê em <= 1 FP (~65%); NÃO VALE (~55%).
LIMITES: 9 meses, 5 trips agrupados (mar-abr/2025), 5 FP no período; R² de um canal dominado pela cauda é instável; descritivo, não mede detecção.
VERIFICAÇÃO: V1 depois do deslocamento de −3 h o T5 do dataset coincide com o da grade (correlação >= 0,999); V2 o walk-forward nunca treina com dado do mês testado.

Uso local:  PYTHONPATH=src:experiments .venv/bin/python -m sota.h1b_ngp_descritivo
No worker:  PYTHONPATH=src python scripts/worker_sota.py --exp h1b_ngp_descritivo --tags sota-h1 --enqueue --queue default
"""
from __future__ import annotations
import contextlib, io, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; EXP = HERE.parent; RAIZ = EXP.parent
sys.path.insert(0, str(RAIZ / "src")); sys.path.insert(0, str(EXP)); sys.path.insert(0, str(RAIZ / "scripts"))
from nv import io as nvio
from nv.avalia.metricas import JANELA, episodios_de_serie
from nv.causal import carrega_registro
from nv.controle_sinais import _ci, PKG
from nv.plugin import decide_v2
from sota import protocolo as P

ROOT = RAIZ / "data_local"; SAIDA = ROOT / "sota" / "h1b_ngp"; DS_NGP = "424e5b589e13402d9d95371a317e85c9"
FIM_NGP = pd.Timestamp("2025-11-01", tz="UTC"); TESTE_MESES = pd.period_range("2025-03", "2025-10", freq="M")
LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.9, random_state=0, verbose=-1)


def carrega_ngp():
    if os.environ.get("NGP_DIR"):
        base = Path(os.environ["NGP_DIR"])
    else:
        from clearml import Dataset
        base = Path(Dataset.get(dataset_id=DS_NGP).get_local_copy())
    f = next(base.rglob("sensores_filtrados_Interpolados_2025.csv")); c0 = pd.read_csv(f, nrows=0).columns[0]
    d = pd.read_csv(f, usecols=[c0, "T5_AVG_A", "NGP_A", "NPT_A"]); d[c0] = pd.to_datetime(d[c0], utc=True)
    d = d.set_index(c0).sort_index().resample("2min").mean(); d.index = d.index - pd.Timedelta(hours=3)        # hora local rotulada como UTC
    return d


def features(x, cols):
    out = {}
    for c in cols:
        out[c] = x[c]; out[f"d1h_{c}"] = x[c] - x[c].shift(30); out[f"d6h_{c}"] = x[c] - x[c].shift(180)
    return pd.DataFrame(out, index=x.index)


def main():
    t0 = time.time(); SAIDA.mkdir(parents=True, exist_ok=True); ok = {}
    import lightgbm as lgb
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    d = carrega_ngp(); g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"]
    j = d.index.intersection(g.index); m = d.loc[j, "T5_AVG_A"].notna() & g.loc[j, "T5_AVG_A"].notna()
    r = float(np.corrcoef(d.loc[j, "T5_AVG_A"][m], g.loc[j, "T5_AVG_A"][m])[0, 1]); ok["V1"] = r >= 0.999
    print(f"V1 — T5 do dataset (−3 h) x grade: correlação {r:.5f} em {int(m.sum())} instantes -> {'PASSA' if ok['V1'] else 'FALHA'}", flush=True)
    S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); idx = S.index[(S.index >= pd.Timestamp("2025-01-01", tz="UTC")) & (S.index < FIM_NGP)]
    x = pd.DataFrame({"T5": g["T5_AVG_A"].reindex(idx), "NGP": d["NGP_A"].reindex(idx), "NPT": d["NPT_A"].reindex(idx)}, index=idx)
    FA, FB = features(x, ["T5"]), features(x, ["T5", "NGP", "NPT"])
    vig = S["mask"].astype(bool).reindex(idx).fillna(False); trips = sorted(carrega_registro())
    longe = pd.Series(True, index=idx)
    for t in trips:
        longe.loc[t - JANELA:t + pd.Timedelta(days=2)] = False
    mes = idx.tz_localize(None).to_period("M"); R2, v2 = {}, []
    for c in ("t", "p", "sp", "vb"):
        y = np.log1p(S[c].reindex(idx)); ys, pa, pb = [], [], []
        for mt in TESTE_MESES:
            tr = vig & longe & (mes < mt) & y.notna() & FB.notna().all(axis=1); te = vig & (mes == mt) & y.notna() & FB.notna().all(axis=1)
            v2.append(not (tr & (mes == mt)).any())
            if tr.sum() < 5000 or te.sum() < 500:
                continue
            ma = lgb.LGBMRegressor(**LGB).fit(FA[tr], y[tr]); mb = lgb.LGBMRegressor(**LGB).fit(FB[tr], y[tr])
            ys.append(y[te].to_numpy()); pa.append(ma.predict(FA[te])); pb.append(mb.predict(FB[te]))
        Y, A_, B_ = np.concatenate(ys), np.concatenate(pa), np.concatenate(pb); sst = float(((Y - Y.mean()) ** 2).sum())
        R2[c] = (1 - float(((Y - A_) ** 2).sum()) / sst, 1 - float(((Y - B_) ** 2).sum()) / sst)
    ok["V2"] = all(v2); print(f"V2 — o walk-forward nunca treina com o mês testado -> {'PASSA' if ok['V2'] else 'FALHA'}", flush=True)
    if not all(ok.values()):
        print(f"verificação falhou — parando. {ok}", flush=True); return
    print("\nM1 — R² FORA DA AMOSTRA de log(1 + sinal), mar-out/2025 (A = T5; B = T5 + NGP + NPT)", flush=True)
    for c, (ra, rb) in R2.items():
        print(f"   {c:3s} R²(A) {ra:+.3f} | R²(B) {rb:+.3f} | ΔR² {rb - ra:+.3f}", flush=True)
    dd = lambda s: (s - s.shift(60)).abs()                                                            # variação em 2 h
    ref = {c: float(np.nanpercentile(dd(x[c])[vig & longe], 95)) for c in ("T5", "NGP", "NPT")}
    proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    with contextlib.redirect_stdout(io.StringIO()):
        al = decide_v2(proc, det)["is_anomaly"].astype(bool)
    alvo = P.trips_alvo(trips); linhas = []
    for a, b in episodios_de_serie(al.loc[P.INICIO:FIM_NGP - pd.Timedelta(minutes=2)]):
        w = (idx >= a - pd.Timedelta(hours=2)) & (idx <= a); mx = {c: float(dd(x[c])[w].max()) for c in ("T5", "NGP", "NPT")}
        rot = "acerto" if any(a <= t and b >= t - JANELA for t in alvo) else "FP"
        invis = (mx["NGP"] > ref["NGP"] or mx["NPT"] > ref["NPT"]) and not (mx["T5"] > ref["T5"])
        linhas.append(dict(inicio_local=(a - pd.Timedelta(hours=3)).strftime("%d/%m/%Y %H:%M"), rotulo=rot, dT5=round(mx["T5"], 2), dNGP=round(mx["NGP"], 3), dNPT=round(mx["NPT"], 2), transiente_que_o_T5_nao_ve=invis))
    T = pd.DataFrame(linhas)
    print(f"\nM2 — maior variação em 2 h antes do nascimento (p95 em operação estável: T5 {ref['T5']:.2f}, NGP {ref['NGP']:.3f}, NPT {ref['NPT']:.2f})", flush=True)
    print(T.to_string(index=False), flush=True)
    nfp = int(T[T.rotulo == "FP"].transiente_que_o_T5_nao_ve.sum()); nac = int(T[T.rotulo == "acerto"].transiente_que_o_T5_nao_ve.sum())
    vale = any(R2[c][1] - R2[c][0] >= 0.10 for c in ("t", "p", "sp")) or (nfp >= 2 and nac <= 1)
    print(f"\nDECISÃO: {'VALE PEDIR' if vale else 'NÃO VALE (por este critério)'} — ΔR² máx em t/p/sp {max(R2[c][1] - R2[c][0] for c in ('t', 'p', 'sp')):+.3f}; transiente que o T5 não vê em {nfp} FP e {nac} acertos", flush=True)
    pd.DataFrame([dict(canal=c, r2_T5=ra, r2_T5_NGP_NPT=rb, delta=rb - ra) for c, (ra, rb) in R2.items()]).to_csv(SAIDA / "r2_por_canal.csv", index=False); T.to_csv(SAIDA / "transientes_nos_nascimentos.csv", index=False)
    print(f"\ntempo total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
