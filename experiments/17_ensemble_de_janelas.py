#!/usr/bin/env python3
"""EXP 17 — ENSEMBLE DE JANELAS DE TREINO: reduzir a variância do normalizador (o achado do EXP 16).

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

PROBLEMA. O EXP 16 mostrou que a MESMA estrutura (pca1 por família, máximo por sensor normalizado pelo p99) dá detecção de 4 a 7 de 7 e 15 a 19 FP conforme o tamanho da janela de treino (5k: 6/19; 10k: 5/17; 20k: 7/15; 40k: 4/16), e que o p99 mensal varia até 91x contra o do bundle. É variância do estimador (um treino, um p99). Variância se reduz por MÉDIA: em vez de um modelo por mês, K modelos com janelas diferentes e o escore final = média dos escores NORMALIZADOS (cada um dividido pelo p99 do seu próprio treino).
O QUE SE FAZ. Famílias: t (pca1, 14 temperaturas) e p (pca4, 12 pressões; o k do bundle). Modelos individuais por mês (causais): n de treino {5k, 10k, 15k, 20k, 30k, 40k, 80k} com a janela terminando no início do mês, e n = 20k com a janela terminando 14, 28 e 42 dias antes (janelas deslocadas). Quatro COMPOSIÇÕES de ensemble, fixadas agora:
   A = {10k, 20k, 40k};  B = {20k terminando em 0, 14, 28 e 42 d antes};  C = {5k, 10k, 20k, 40k, 80k};  D = {15k, 20k, 30k}.
Combinador = média dos escores normalizados (a mediana é reportada como robustez). Depois, o de sempre: EWMA 1 h vigiado, mapa de quantis para o canal original do v2, `ci.prever` (substituição de t ou p).
MEDIDAS. (i) ESTABILIDADE: faixa de detecção e de FP entre as 4 composições, contra a faixa dos modelos individuais (t: detecção 4-7, FP 15-19). (ii) PONTO DE OPERAÇÃO contra o controle (6/7, 9 FP, 407 h fora) e contra a referência do harness (t: 7/7, 15 FP, 590 h; p: 6/7, 12 FP, 754 h).
CRITÉRIOS (fixados antes). ESTABILIZA = faixa de detecção entre as 4 composições <= 1 evento E faixa de FP <= 2; ADIANTA = ESTABILIZA e mediana dos FP das composições <= 12 e horas fora <= 600 com detecção >= 6/7 (o pca1 individual não chega a isso em nenhum n); DIGNO = detecção >= 6/7, FP <= 9, <= 407 h; PROMISSOR = >= 5/7, FP <= 12, <= 800 h.
EXPECTATIVAS: ESTABILIZA em t ~55% (a média de 3 a 5 janelas correlacionadas reduz a variância mas as janelas de 5k e 80k extremas podem dominar); ADIANTA em t ~20%; ADIANTA em p ~25%; algum DIGNO ~4%; a mediana como combinador dá o mesmo resultado que a média (+-1 FP) ~70%.
LIMITES. 7 eventos; as janelas de ensemble foram escolhidas por mim, sem busca; o mapa de quantis é global; nenhuma composição se adota sem sombra.

Uso:  PYTHONPATH=src .venv/bin/python experiments/17_ensemble_de_janelas.py
"""
from __future__ import annotations
import contextlib, importlib.util, io, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E4 = _carrega("exp04", "04_escada_de_baselines.py"); E5 = _carrega("exp05", "05_decomposicao_do_v2.py"); E6 = _carrega("exp06", "06_plugin_de_canais.py")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.modelos.familias import PCAk, deslocado
from nv.plugin import decide_v2, mapeia_quantis, substitui
from nv.politica import ewma_vigiado
from nv.walk import inicios_de_mes

INICIO = E4.INICIO; ROOT = Path(__file__).resolve().parents[1] / "data_local"; CACHE = ROOT / "escores_ens"; GL = {}
FAMILIAS = {"t": ("temperatura", 1), "p": ("pressao", 4)}
COMPOSICOES = {"A": [(10000, 0), (20000, 0), (40000, 0)], "B": [(20000, 0), (20000, 14), (20000, 28), (20000, 42)], "C": [(5000, 0), (10000, 0), (20000, 0), (40000, 0), (80000, 0)], "D": [(15000, 0), (20000, 0), (30000, 0)]}
SINGLES = sorted({m for c in COMPOSICOES.values() for m in c})


def walk(g, tags, est, ini, fim, k, n_treino, desloc_dias):
    X = g[tags].to_numpy("float64"); idx = g.index; out = np.full(len(g), np.nan); meses = inicios_de_mes(ini, fim); fab = type("PCA", (PCAk,), {"k": k})
    for j, m0 in enumerate(meses):
        m1 = meses[j + 1] if j + 1 < len(meses) else fim + pd.Timedelta("2min"); i0 = idx.searchsorted(m0); i1 = idx.searchsorted(m1); iz = idx.searchsorted(m0 - pd.Timedelta(days=desloc_dias))
        cand = np.flatnonzero(est[:iz] & np.isfinite(X[:iz]).all(axis=1))
        if len(cand) < n_treino // 4:
            continue
        tr = X[cand[-n_treino:]]; mod = fab(semente=0).fit(tr); p99 = float(np.nanquantile(mod.score(tr), 0.99)); out[i0:i1] = mod.score(X[i0:i1]) / max(p99, 1e-12)
    return pd.Series(out, index=idx)


def tarefa(a):
    canal, n, d = a; f = CACHE / f"{canal}_pca_n{n}_d{d}.parquet"
    if f.exists():
        return a, pd.read_parquet(f)["s"]
    t0 = time.time(); fam, k = FAMILIAS[canal]
    s = walk(GL["g"], GL["tags"][canal], GL["est"], pd.Timestamp("2025-03-01", tz="UTC"), GL["fim"], k, n, d)
    CACHE.mkdir(parents=True, exist_ok=True); s.to_frame("s").to_parquet(f); print(f"      {canal} n={n} desloc={d} d: {time.time() - t0:.0f} s", flush=True)
    return a, s


def main():
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tags = {c: list(modelos[-1]["normalizacao"][f]["cols"]) for c, (f, k) in FAMILIAS.items()}
    al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]; reg = [t for t in carrega_registro() if t >= INICIO]; ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); vigS = proc["mask"]; vig_g = vig_c.reindex(g.index).fillna(False)
    GL.update(g=g, est=est, fim=fim, tags=tags)
    print(f"CONTROLE: {E6.linha(ctrl)}", flush=True)
    pedidos = [(c, n, d) for c in FAMILIAS for (n, d) in SINGLES]
    with Pool(4) as p:
        R = dict(p.map(tarefa, pedidos, chunksize=1))

    def mede(s, canal):
        e = ewma_vigiado(s, vig_g, "1h").reindex(proc.index); r = E6.mede(decide_v2(substitui(proc, canal, mapeia_quantis(e, proc[canal], vigS)), det), reg); return r
    for canal in FAMILIAS:
        print(f"\nCANAL {canal} ({FAMILIAS[canal][0]}, pca k={FAMILIAS[canal][1]})", flush=True)
        ind = {}
        for (n, d) in SINGLES:
            r = mede(R[(canal, n, d)], canal); ind[(n, d)] = r
            print(f"   individual n={n:6d} deslocado {d:2d} d: {r.det}/7 | FP {r.fp:2d} | fora {r.horas_alarme_fora:5.0f} h | totais {r.horas_alarme:5.0f} h", flush=True)
        dets = [r.det for r in ind.values()]; fps = [r.fp for r in ind.values()]; print(f"   -> faixa dos individuais: detecção {min(dets)}-{max(dets)}, FP {min(fps)}-{max(fps)}")
        res = {}
        for nome, comp in COMPOSICOES.items():
            M = pd.concat([R[(canal, n, d)] for (n, d) in comp], axis=1)
            for comb, f in (("média", M.mean(axis=1, skipna=True)), ("mediana", M.median(axis=1, skipna=True))):
                r = mede(f, canal); res[(nome, comb)] = r
                print(f"   ensemble {nome} ({comb}) {comp}: {r.det}/7 | início {r.inicio} | FP {r.fp:2d} | fora {r.horas_alarme_fora:5.0f} h | totais {r.horas_alarme:5.0f} h | lead {r.lead_medio_h:.1f} h", flush=True)
        md = [res[(n, "média")].det for n in COMPOSICOES]; mf = [res[(n, "média")].fp for n in COMPOSICOES]; mo = [res[(n, "média")].horas_alarme_fora for n in COMPOSICOES]
        estab = (max(md) - min(md) <= 1) and (max(mf) - min(mf) <= 2); adianta = estab and np.median(mf) <= 12 and np.median(mo) <= 600 and min(md) >= 6
        print(f"   ESTABILIZA: {estab} (detecção {min(md)}-{max(md)}, FP {min(mf)}-{max(mf)}) | ADIANTA: {adianta} (FP mediano {np.median(mf):.0f}, fora mediano {np.median(mo):.0f} h)", flush=True)


if __name__ == "__main__":
    main()
