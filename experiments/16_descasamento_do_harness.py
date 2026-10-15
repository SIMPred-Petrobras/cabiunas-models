#!/usr/bin/env python3
"""EXP 16 — POR QUE O HARNESS POR FAMÍLIA DÁ 15 FP E O BUNDLE 9? (diagnóstico da validade do EXP 14/15)

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

PROBLEMA. O pca1 por família no canal t (mesma estrutura do bundle: k = 1, máximo por sensor normalizado pelo p99) dá 7/7 com 15 FP e 590 h fora no plug-in, e o bundle dá 6/7 com 9 FP e 407 h. A diferença está só no TREINO (o harness retreina nas 20.000 últimas amostras estáveis antes de cada mês; o bundle usa janelas de datas próprias, com k e p99 fixados). Sem fechar isso, a leitura do canal t nos EXP 14/15 é descritiva.
O QUE SE FAZ (só diagnóstico, sem eventos de seleção):
  (a) NÍVEL: por mês, comparar o canal t do v2 (`S["t"]`, o sinal causal em unidades do bundle) com o escore do harness depois do EWMA de 1 h, em instantes vigiados: correlação de Spearman global e por mês, e a razão dos quantis p50/p90/p99 do harness sobre os do v2 por mês (antes do mapa de quantis, que só iguala a marginal GLOBAL).
  (b) EPISÓDIOS: listar os episódios do plug-in com o pca1 (20.000) que NÃO existem no controle (nascem, FP a mais) e os do controle que somem, com data e duração.
  (c) SENSIBILIDADE AO TREINO: pca1 em t com n_treino em {5.000, 10.000, 20.000, 40.000}: detecção, FP, horas fora. Nenhum valor se escolhe pelo resultado: só se descreve a curva.
EXPECTATIVAS: a correlação de Spearman global entre o harness e o t do v2 é >= 0,8 (~70%) mas a razão dos quantis varia por mês em mais de 1,5x entre o menor e o maior mês (~65%: o 'recon_p99 varia 2,45x'); >= 4 dos 6 FP extras caem em <= 3 meses (~55%); a curva do n_treino é monótona (menos amostras, mais FP) (~45%); algum n_treino reproduz <= 10 FP e >= 6/7 (~15%).
LIMITES. Um canal, um escore, 7 eventos; o harness não pode usar as janelas exatas do bundle (os pickles/JSON cobrem 18 modelos mensais, e usá-los é o próprio controle).

Uso:  PYTHONPATH=src .venv/bin/python experiments/16_descasamento_do_harness.py
"""
from __future__ import annotations
import contextlib, importlib.util, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E4 = _carrega("exp04", "04_escada_de_baselines.py"); E5 = _carrega("exp05", "05_decomposicao_do_v2.py"); E6 = _carrega("exp06", "06_plugin_de_canais.py")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle, episodios_de_serie
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.modelos.familias import FAMILIA_FABRICAS
from nv.plugin import decide_v2, mapeia_quantis, substitui
from nv.politica import ewma_vigiado
from nv.walk import walk_forward

INICIO = E4.INICIO; ROOT = Path(__file__).resolve().parents[1] / "data_local"


def eps(res):
    return [(a, b) for a, b in episodios_de_serie(res["is_anomaly"].astype(bool).loc[INICIO:])]


def main():
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tags = list(modelos[-1]["normalizacao"]["temperatura"]["cols"])
    al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]; reg = [t for t in carrega_registro() if t >= INICIO]; ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); vigS = proc["mask"]; vig_g = vig_c.reindex(g.index).fillna(False)
    print(f"CONTROLE: {E6.linha(ctrl)}", flush=True)
    s20 = pd.read_parquet(ROOT / "escores_fam" / "t_pca1_20000_0.parquet")["s"]; h = ewma_vigiado(s20, vig_g, "1h").reindex(proc.index); v2t = proc["t"].where(vigS)
    ok = vigS & h.notna() & v2t.notna()
    print(f"\n(a) NÍVEL. Spearman global (harness pca1 x t do v2, vigiado, EWMA 1 h): {spearmanr(h[ok], v2t[ok])[0]:.3f} | instantes {int(ok.sum())}")
    linhas = []
    for mes, ix in ok[ok].groupby(ok[ok].index.to_period("M")).groups.items():
        a, b = h.loc[ix], v2t.loc[ix]
        if len(ix) < 500:
            continue
        q = lambda x, p: float(np.nanquantile(x, p))
        linhas.append(dict(mes=str(mes), n=len(ix), spearman=round(float(spearmanr(a, b)[0]), 2), r50=round(q(a, .5) / q(b, .5), 2), r90=round(q(a, .9) / q(b, .9), 2), r99=round(q(a, .99) / q(b, .99), 2)))
    T = pd.DataFrame(linhas); print(T.to_string(index=False)); print(f"   razão p99 harness/v2: mínimo {T.r99.min():.2f}, máximo {T.r99.max():.2f}, variação {T.r99.max() / T.r99.min():.2f}x | Spearman por mês: mediana {T.spearman.median():.2f}, mínimo {T.spearman.min():.2f}", flush=True)

    novo = mapeia_quantis(h, proc["t"], vigS); res = decide_v2(substitui(proc, "t", novo), det); r = E6.mede(res, reg); base = decide_v2(proc, det)
    print(f"\n(b) EPISÓDIOS do plug-in com pca1 (20.000): {E6.linha(r)}")
    B, V = eps(base), eps(res)
    def toca(x, L): return any(a <= x[1] and b >= x[0] for a, b in L)
    for a, b in V:
        if not toca((a, b), B):
            tr = [t for t in reg if a <= t and b >= t - pd.Timedelta(hours=48)]
            print(f"     NASCE  {a:%Y-%m-%d %H:%M} dur {(b - a).total_seconds() / 3600 + 2 / 60:6.1f} h  {'trip ' + str(tr[0])[:10] if tr else 'FP'}")
    for a, b in B:
        if not toca((a, b), V):
            print(f"     SOME   {a:%Y-%m-%d %H:%M} dur {(b - a).total_seconds() / 3600 + 2 / 60:6.1f} h")
    print("\n(c) SENSIBILIDADE AO TREINO (pca1 em t)", flush=True)
    for nt in (5000, 10000, 20000, 40000):
        f = ROOT / "escores_fam" / f"t_pca1_{nt}_0.parquet"
        if f.exists():
            s = pd.read_parquet(f)["s"]
        else:
            s = walk_forward(FAMILIA_FABRICAS["pca1"], g, tags, est, pd.Timestamp("2025-03-01", tz="UTC"), fim, n_treino=nt, semente=0); s.to_frame("s").to_parquet(f)
        e = ewma_vigiado(s, vig_g, "1h").reindex(proc.index); rr = E6.mede(decide_v2(substitui(proc, "t", mapeia_quantis(e, proc["t"], vigS)), det), reg)
        print(f"   n_treino {nt:6d}: detecção {rr.det}/7 | FP {rr.fp} | fora {rr.horas_alarme_fora:.0f} h | totais {rr.horas_alarme:.0f} h | lead {rr.lead_medio_h:.1f} h", flush=True)


if __name__ == "__main__":
    main()
