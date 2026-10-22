#!/usr/bin/env python3
"""SOTA H6 — QUAL TAG PUXOU O ALARME: contribuição por sensor de cada canal do PCA v2, com as 3 tags que mais pesam em cada episódio. Tag ClearML `sota-h6`.

Fixado em 22/10/2026, antes de rodar a série completa (o encanamento foi testado numa janela curta). DESCRITIVO: é entregável do brief e insumo da triagem dos FP.
COMO: replay causal semanal idêntico ao do controle (`nv.controle_sinais.sinais_causais`: mesmo início, complemento de 30 s, janela de 75 d, fatiamento nos trips do
registro), mas em vez do escore do canal guarda o escore de CADA TAG, com a aritmética do pacote: no t e no p, o erro quadrático de reconstrução da tag dividido pela
sensibilidade p99 dela e pelo recon_p99 do bundle (o canal é o MÁXIMO disso entre as tags); no sp, a única série (TI_0305 contra a mediana dos irmãos); no vb, o z robusto
de CADA sonda contra a própria referência rolante (o canal é o máximo entre as sondas). Cada tag passa pelo mesmo EWMA do canal.
ATRIBUIÇÃO: para cada episódio da base atual, só as tags dos canais acesos (nível A ou B) no nascimento; a razão de cada tag ao limiar A do seu canal, média em
[nascimento, nascimento + 2 h] (a janela de notificação); as 3 maiores. Também as 3 maiores no episódio inteiro, entre todos os canais.
VERIFICAÇÃO: V1 o EWMA do máximo entre as tags de cada canal reproduz o cache causal do controle (`sinais_v2_8ev`) em todo instante guardado (diferença <= 1e-9).
EXPECTATIVA (registrada para o placar): a tag do topo NÃO separa FP de acerto (~70%; EXP 36-40: o FP é indistinguível do acerto no disparo); o TC382_03_A e o TI_0305
aparecem no topo de muitos episódios (~60%).

Uso local (encanamento, janela curta):  SOTA_TESTE=1 PYTHONPATH=src:experiments .venv/bin/python -m sota.h6_atribuicao
No worker:  PYTHONPATH=src python scripts/worker_sota.py --exp h6_atribuicao --tags sota-h6 --enqueue --queue default
"""
from __future__ import annotations
import contextlib, io, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; EXP = HERE.parent; RAIZ = EXP.parent
sys.path.insert(0, str(RAIZ / "src")); sys.path.insert(0, str(EXP))
from nv.avalia.metricas import JANELA, episodios_de_serie
from nv.causal import carrega_registro
from nv.controle_sinais import _ci, PKG, JANELA as JAN75
from nv.plugin import decide_v2
from sota import protocolo as P

TESTE = bool(os.environ.get("SOTA_TESTE"))
ROOT = RAIZ / "data_local"; SAIDA = ROOT / "sota" / "h6_atribuicao"
INICIO_REPLAY = pd.Timestamp("2025-01-08 07:34", tz="UTC") + pd.Timedelta(weeks=58 if TESTE else 0)
CANAIS = ("t", "p", "sp", "vb")


def brutos(ci, modelos, w, trips):
    """Escore cru POR TAG numa janela, com a aritmética do `preprocessar` do pacote. Colunas 'canal:tag'."""
    g = ci._grade(w); _, estavel, _, _ = ci._mascara(g); det = modelos[-1]["detector"]
    meses = pd.date_range(g.index[0].normalize().replace(day=1), g.index[-1], freq="MS", tz="UTC"); partes = {c: [] for c in ("t", "p", "sp")}
    for i_m, m0 in enumerate(meses):
        m1 = meses[i_m + 1] if i_m + 1 < len(meses) else g.index[-1] + pd.Timedelta(ci.GRID)
        fatia = (g.index >= max(m0, g.index[0])) & (g.index < m1)
        if not fatia.any():
            continue
        mod = ci.vigencia(modelos, max(m0, g.index[0])); wg = g.loc[fatia]
        for fam, ch in (("temperatura", "t"), ("pressao", "p")):
            cfg = mod["normalizacao"][fam]; cols = cfg["cols"]; sens = np.asarray(cfg["sens_p99"], dtype="float64"); X = wg[cols]
            ok = X.notna().all(axis=1).to_numpy(); E = np.full((len(X), len(cols)), np.nan)
            if ok.any():
                tr = mod[f"{fam}_transformacao"]
                Xs = (X[ok].astype("float64").to_numpy() - np.asarray(tr["center"])) / np.asarray(tr["scale"])
                Z = Xs @ np.asarray(tr["components"]).T - np.asarray(tr["mean_proj"]); rec = Z @ np.asarray(tr["components"]) + np.asarray(tr["mean"])
                E[ok] = ((Xs - rec) ** 2) / sens
            partes[ch].append(pd.DataFrame(E / cfg["recon_p99"], index=X.index, columns=[f"{ch}:{c}" for c in cols]))
        sm = mod["spread_mancal"]
        spread = wg[sm["tag_alvo"]].astype("float64") - wg[sm["tags_irmaos"]].astype("float64").median(axis=1)
        partes["sp"].append(pd.DataFrame({f"sp:{sm['tag_alvo']}": np.abs((spread - sm["mediana"]) / sm["mad_robusto"]).to_numpy()}, index=wg.index))
    out = {ch: pd.concat(v).reindex(g.index) for ch, v in partes.items()}
    tv = [t for t in (trips if trips is not None else det.get("trips_conhecidos", [])) if pd.Timestamp(t) <= g.index[-1]]
    out["vb"] = pd.DataFrame({f"vb:{s}": ci._z_vibracao(g, estavel, [s], tv, det.get("vb_exclusao_dias", 7.0)) for s in det["tags_vibracao"]}, index=g.index)
    return g, out


def ewma(df, hl, idx):
    return pd.DataFrame({c: df[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(df[c].notna()) for c in df.columns}, index=idx)


def replay(ci, modelos, extra):
    """O mesmo laço de `sinais_causais`, guardando o EWMA de cada tag e o EWMA do máximo de cada canal."""
    det = modelos[-1]["detector"]; csv = next((PKG / "dados" / "2025_2026").glob("data_*_raw.csv"))
    with contextlib.redirect_stdout(io.StringIO()):
        raw = ci.carregar_dados(csv); raw = pd.concat([extra.loc[extra.index < raw.index[0], raw.columns], raw]).sort_index(); df = ci.preparar_grade(raw)
    trips = sorted(ci.carregar_trips(PKG / "registro_trips.csv"))
    real = df["RUNNING_A"].notna() & df[[c for c in df.columns if c.startswith("TV_")]].notna().any(axis=1); df = df.loc[:df.index[real][-1]]; fim = df.index[-1]
    cortes = [INICIO_REPLAY]
    while cortes[-1] + pd.Timedelta(days=7) < fim:
        cortes.append(cortes[-1] + pd.Timedelta(days=7))
    cortes.append(fim); pt, pc, anterior = [], [], None
    for c in cortes:
        for s in [f for f in trips if anterior is not None and anterior < f < c] + [c]:
            conhecidos = trips if s == c else [t for t in trips if t < s]; w = df.loc[max(df.index[0], s - JAN75):s]
            with contextlib.redirect_stdout(io.StringIO()):
                g, B = brutos(ci, modelos, w, conhecidos)
            tags = pd.concat([ewma(B[ch], det["halflife"][ch], g.index) for ch in CANAIS], axis=1)
            can = pd.DataFrame({ch: B[ch].max(axis=1, skipna=True).where(B[ch].notna().any(axis=1)) for ch in CANAIS}, index=g.index)
            can = pd.DataFrame({ch: can[ch].ewm(halflife=pd.Timedelta(det["halflife"][ch]), times=g.index).mean().where(can[ch].notna()) for ch in CANAIS}, index=g.index)
            sel = tags.index > anterior if anterior is not None else np.ones(len(tags), bool)
            pt.append(tags[sel]); pc.append(can[sel]); anterior = s
    T = pd.concat(pt); C = pd.concat(pc)
    return T[~T.index.duplicated(keep="first")], C[~C.index.duplicated(keep="first")]


def atribui(R, T, det, trips):
    thr = {c: det["base"][c] * det["k_nivel_a"][c] for c in CANAIS}; linhas = []
    al = R["is_anomaly"].astype(bool)
    for a, b in episodios_de_serie(al.loc[P.INICIO:P.FIM]):
        acesos = [c for c in CANAIS if bool(R[f"a_{c}"].loc[a]) or bool(R[f"b_{c}"].loc[a])]
        rot = next((f"TP {t:%d/%m/%Y}" for t in trips if a <= t and b >= t - JANELA), "FP")
        def top(cols, ini, fim):
            if not cols:
                return []
            x = T.loc[ini:fim, cols].mean() / pd.Series({k: thr[k.split(":")[0]] for k in cols})
            return [(k.split(":", 1)[1], k.split(":")[0], float(v)) for k, v in x.sort_values(ascending=False).head(3).items() if np.isfinite(v)]
        tn = top([k for k in T.columns if k.split(":")[0] in acesos], a, min(b, a + pd.Timedelta(hours=2)))
        te = top(list(T.columns), a, b)
        lin = dict(inicio_local=(a - pd.Timedelta(hours=3)).strftime("%d/%m/%Y %H:%M"), dur_h=round((b - a).total_seconds() / 3600, 1), rotulo=rot, canais_nascimento="+".join(acesos))
        for i in range(3):
            lin[f"top{i + 1}"] = f"{tn[i][0].replace('954005_624_', '')} ({tn[i][1]}, {tn[i][2]:.1f}x)" if i < len(tn) else ""
            lin[f"ep_top{i + 1}"] = f"{te[i][0].replace('954005_624_', '')} ({te[i][1]}, {te[i][2]:.1f}x)" if i < len(te) else ""
        linhas.append(lin)
    return pd.DataFrame(linhas)


def main():
    t0 = time.time(); ci = _ci(); SAIDA.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]; trips = P.trips_alvo(carrega_registro())
    T, C = replay(ci, modelos, pd.read_parquet(ROOT / "raw30s_2024q4_2025q1.parquet"))
    print(f"replay por tag: {T.shape[0]} instantes x {T.shape[1]} tags ({time.time() - t0:.0f} s)", flush=True)
    S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); j = C.index.intersection(S.index); j = j[j > INICIO_REPLAY]
    dif = {c: float(np.nanmax(np.abs(C.loc[j, c].to_numpy() - S.loc[j, c].to_numpy()))) for c in CANAIS}
    nan_dif = {c: int((C.loc[j, c].isna() != S.loc[j, c].isna()).sum()) for c in CANAIS}
    ok = all(v <= 1e-9 for v in dif.values()) and all(v == 0 for v in nan_dif.values())
    print(f"V1 — EWMA do máximo por canal x cache causal: dif. máx {({k: f'{v:.1e}' for k, v in dif.items()})}, NaN divergentes {nan_dif} -> {'PASSA' if ok else 'FALHA'}", flush=True)
    if not ok:
        print("verificação falhou — parando.", flush=True); return
    proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    with contextlib.redirect_stdout(io.StringIO()):
        R = decide_v2(proc, det)
    if TESTE:
        R = R.loc[INICIO_REPLAY + pd.Timedelta(days=7):]; A = atribui(R, T, det, trips); A.to_csv(SAIDA / "h6_top3_teste.csv", index=False)
        print(f"TESTE: encanamento ok em {time.time() - t0:.0f} s (janela curta; {len(A)} episódios atribuídos; saída suprimida).", flush=True); return
    A = atribui(R, T, det, trips); A.to_csv(SAIDA / "h6_top3.csv", index=False)
    T.loc[P.INICIO - pd.Timedelta(days=3):].astype("float32").to_parquet(SAIDA / "h6_ewma_por_tag.parquet")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
    print("\nTOP-3 TAGS POR EPISÓDIO (base atual; razão ao limiar A do canal, média nas 2 h da notificação; e no episódio inteiro)", flush=True)
    print(A.to_string(index=False), flush=True)
    t1 = A["top1"].str.split(" ").str[0]
    print("\nTAG DO TOPO por rótulo:  FP " + str(t1[A.rotulo == "FP"].value_counts().to_dict()) + " | acertos " + str(t1[A.rotulo != "FP"].value_counts().to_dict()), flush=True)
    print(f"\ntempo total {time.time() - t0:.0f} s; tabelas em {SAIDA}", flush=True)


if __name__ == "__main__":
    main()
