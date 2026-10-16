#!/usr/bin/env python3
"""EXP 18 — AUDITORIA DE PRÉ-PROCESSAMENTO (parte A, descritiva) E ABLAÇÃO DA ESCALA E DO CONJUNTO DE TREINO (parte B).

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. Nos EXP 04-17 a grade de 2 min e a máscara estável foram usadas COMO VIERAM. O EXP 16 mostrou que o pca1 por família do harness dá 15 FP onde o bundle dá 9, e o EXP 17 mostrou que não é variância do tamanho da janela (nenhum de 10 treinos nem de 4 ensembles passa de 15 a menos) e sim um VIÉS do harness. O que o harness faz diferente do bundle está no pré-processamento: (i) o harness treina SÓ em amostras estáveis (máscara de gradiente baixo), o bundle em janelas de datas que contêm transiente e anormalidade (a memória 'baseline precisa de anormalidade': limpar as janelas pré-falha levou o FP de 6,6 para 68 h/mês: o p99 precisa da cauda); (ii) a escala (mediana/MAD com piso de 5% da mediana) e o p99 por sensor.
PARTE A (descritiva, sem eventos): para as 26 tags das famílias t (14) e p (12), em instantes VIGIADOS: número de valores distintos (arredondado a 6 casas), fração de passos iguais ao anterior (linha reta), MAD global, razão máximo/mínimo do MAD mensal (meses com >= 500 amostras vigiadas), e a PARTICIPAÇÃO de cada tag no máximo do escore do pca por família (quantas vezes a tag é a que atinge o máximo, modelo mensal de 20.000 amostras estáveis). Nenhum critério: só se lista quem domina e quem tem escala instável.
PARTE B (ablação): pca1 em t (k = 1) e pca4 em p (k = 4), retreino mensal causal, EWMA 1 h, mapa de quantis, `ci.prever` como substituição do canal (o harness do EXP 14/16), n de treino em {10.000, 20.000, 40.000} para enxergar a FAIXA (o piso de ruído do EXP 16). Cinco variantes, fixadas agora:
   V0 base          treino em amostras estáveis, escala mediana/MAD (piso 5% da mediana);
   V1 sem gating    treino em TODAS as amostras vigiadas e finitas anteriores ao mês (transiente e anormalidade entram; sem a máscara estável);
   V2 padrão        escala média/desvio (em vez de mediana/MAD);
   V3 piso 15%      mediana/MAD com piso de 15% da mediana (tags quase constantes deixam de explodir);
   V4 winsorizada   a escala robusta, com o treino do PCA e do p99 cortado nos percentis 0,5 e 99,5 do treino.
CRITÉRIOS (fixados antes). Para cada variante e canal, mediana das 3 janelas e faixa: DIGNO = detecção >= 6/7, FP <= 9, <= 407 h fora; PROMISSOR = >= 5/7, FP <= 12, <= 800 h; REDUZ O VIÉS = mediana de FP <= 12 em t (a V0 dá 15-17). A faixa de detecção entre os 3 n é reportada (V0 em t: 4-7).
EXPECTATIVAS. A: a PI_0319 domina a participação no máximo do canal p (>= 40% das vezes) ~65%; no canal t, uma única tag atinge o máximo em >= 30% das vezes ~55%; ao menos 3 tags têm razão máx/mín do MAD mensal > 3 ~70%. B: V1 (sem gating) reduz o FP de t a <= 12 em >= 2 dos 3 n ~35% (a hipótese H10: o bundle treina com anormalidade); V2 (padrão) é pior que a V0 em FP (~65%); V3 neutra (+-2 FP) (~55%); V4 aumenta o FP (a cauda mais fina baixa o p99) (~60%); algum DIGNO ~6%; algum PROMISSOR em t ~35%.
LIMITES. 7 eventos; uma família de variantes pequena e escolhida por mim; a V1 usa a máscara de vigilância do próprio controle (causal); o mapa de quantis é global; nada se adota sem sombra.

Uso:  PYTHONPATH=src .venv/bin/python experiments/18_auditoria_de_preprocessamento.py
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
from nv.modelos.baselines import Escala
from nv.modelos.familias import PCAk
from nv.plugin import decide_v2, mapeia_quantis, substitui
from nv.politica import ewma_vigiado
from nv.walk import inicios_de_mes

INICIO = E4.INICIO; ROOT = Path(__file__).resolve().parents[1] / "data_local"; CACHE = ROOT / "escores_pre"; GL = {}
FAMILIAS = {"t": ("temperatura", 1), "p": ("pressao", 4)}
VARIANTES = {"V0 base": ("est", "robust"), "V1 sem gating": ("vig", "robust"), "V2 padrão": ("est", "std"), "V3 piso 15%": ("est", "piso15"), "V4 winsorizada": ("est", "winsor")}
NS = (10000, 20000, 40000)


class Padrao:
    def fit(self, X):
        self.med = X.mean(axis=0); self.sc = np.maximum(X.std(axis=0), 1e-6); return self

    def transform(self, X):
        return (X - self.med) / self.sc


class Piso15(Escala):
    def fit(self, X):
        self.med = np.median(X, axis=0); mad = 1.4826 * np.median(np.abs(X - self.med), axis=0)
        self.sc = np.maximum(np.maximum(mad, 0.15 * np.median(np.abs(X), axis=0)), 1e-6); return self


def modelo(k, escala):
    class M(PCAk):
        def _ajusta_escala(self, X):
            self.escala = {"robust": Escala, "std": Padrao, "piso15": Piso15, "winsor": Escala}[escala]().fit(X); Z = self.escala.transform(X)
            if escala == "winsor":
                lo, hi = np.quantile(Z, [0.005, 0.995], axis=0); Z = np.clip(Z, lo, hi)
            return Z
    M.k = k; return M


def walk(g, tags, mascara, ini, fim, k, n, escala):
    X = g[tags].to_numpy("float64"); idx = g.index; out = np.full(len(g), np.nan); meses = inicios_de_mes(ini, fim); fab = modelo(k, escala)
    for j, m0 in enumerate(meses):
        m1 = meses[j + 1] if j + 1 < len(meses) else fim + pd.Timedelta("2min"); i0 = idx.searchsorted(m0); i1 = idx.searchsorted(m1)
        cand = np.flatnonzero(mascara[:i0] & np.isfinite(X[:i0]).all(axis=1))
        if len(cand) < n // 4:
            continue
        tr = X[cand[-n:]]; mod = fab(semente=0).fit(tr); p99 = float(np.nanquantile(mod.score(tr), 0.99)); out[i0:i1] = mod.score(X[i0:i1]) / max(p99, 1e-12)
    return pd.Series(out, index=idx)


def tarefa(a):
    canal, var, n = a; f = CACHE / f"{canal}_{var[:2]}_{n}.parquet"
    if f.exists():
        return a, pd.read_parquet(f)["s"]
    t0 = time.time(); fam, k = FAMILIAS[canal]; masc_nome, escala = VARIANTES[var]
    mascara = GL["est"] if masc_nome == "est" else GL["vig"]
    s = walk(GL["g"], GL["tags"][canal], mascara, pd.Timestamp("2025-03-01", tz="UTC"), GL["fim"], k, n, escala)
    CACHE.mkdir(parents=True, exist_ok=True); s.to_frame("s").to_parquet(f); print(f"      {canal} {var} n={n}: {time.time() - t0:.0f} s", flush=True)
    return a, s


def parte_a(g, vig_g, est, tags):
    print("PARTE A — AUDITORIA POR TAG (instantes vigiados)", flush=True)
    v = vig_g.to_numpy(bool); linhas = []; part = {}
    for canal, (fam, k) in FAMILIAS.items():
        X = g[tags[canal]].to_numpy("float64"); meses = inicios_de_mes(g.index[0], g.index[-1]); cont = np.zeros(len(tags[canal]), int); tot = 0
        for j, m0 in enumerate(meses):
            m1 = meses[j + 1] if j + 1 < len(meses) else g.index[-1] + pd.Timedelta("2min"); i0 = g.index.searchsorted(m0); i1 = g.index.searchsorted(m1)
            if m0 < pd.Timestamp("2025-03-01", tz="UTC"):
                continue
            cand = np.flatnonzero(est[:i0] & np.isfinite(X[:i0]).all(axis=1))
            if len(cand) < 5000:
                continue
            mod = modelo(k, "robust")(semente=0).fit(X[cand[-20000:]]); alvo = np.arange(i0, i1)[v[i0:i1] & np.isfinite(X[i0:i1]).all(axis=1)]
            if len(alvo):
                Z = mod.escala.transform(X[alvo]); e = mod._erro2(Z) / mod.sens; cont += np.bincount(e.argmax(axis=1), minlength=len(tags[canal])); tot += len(alvo)
        part[canal] = cont / max(tot, 1)
        for j, tg in enumerate(tags[canal]):
            x = g[tg].to_numpy("float64"); xv = x[v & np.isfinite(x)]
            if len(xv) < 1000:
                continue
            iguais = float(np.mean(np.diff(xv) == 0)); nun = len(np.unique(np.round(xv, 6)))
            gm = pd.Series(x[v & np.isfinite(x)], index=g.index[v & np.isfinite(x)]); mads = gm.groupby(gm.index.to_period("M")).apply(lambda s: 1.4826 * np.median(np.abs(s - np.median(s))) if len(s) >= 500 else np.nan).dropna()
            razao = float(mads.max() / max(mads.min(), 1e-12)) if len(mads) > 1 else np.nan
            linhas.append(dict(familia=canal, tag=tg.split("_", 2)[-1], distintos=nun, passos_iguais=round(iguais, 3), mad=round(float(1.4826 * np.median(np.abs(xv - np.median(xv)))), 4), razao_mad_mensal=round(razao, 1), part_max=round(float(part[canal][j]), 3)))
    T = pd.DataFrame(linhas); print(T.to_string(index=False), flush=True)
    for canal in FAMILIAS:
        sub = T[T.familia == canal].sort_values("part_max", ascending=False); print(f"   {canal}: maior participação no máximo: {sub.iloc[0].tag} {100 * sub.iloc[0].part_max:.0f}%, depois {sub.iloc[1].tag} {100 * sub.iloc[1].part_max:.0f}%, {sub.iloc[2].tag} {100 * sub.iloc[2].part_max:.0f}% | tags com razão MAD mensal > 3: {int((sub.razao_mad_mensal > 3).sum())}")
    T.to_csv(ROOT / "exp18_parteA.csv", index=False)


def main():
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tags = {c: list(modelos[-1]["normalizacao"][f]["cols"]) for c, (f, k) in FAMILIAS.items()}
    al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]; reg = [t for t in carrega_registro() if t >= INICIO]; ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); vigS = proc["mask"]; vig_g = vig_c.reindex(g.index).fillna(False)
    GL.update(g=g, est=est, vig=vig_g.to_numpy(bool), fim=fim, tags=tags)
    print(f"CONTROLE: {E6.linha(ctrl)}\n", flush=True)
    parte_a(g, vig_g, est, tags)
    pedidos = [(c, v, n) for c in FAMILIAS for v in VARIANTES for n in NS]
    with Pool(4) as p:
        R = dict(p.map(tarefa, pedidos, chunksize=1))
    print("\nPARTE B — ABLAÇÃO (mediana dos 3 n; faixa entre parênteses)", flush=True)
    for canal in FAMILIAS:
        print(f"CANAL {canal}", flush=True)
        for var in VARIANTES:
            ms = []
            for n in NS:
                e = ewma_vigiado(R[(canal, var, n)], vig_g, "1h").reindex(proc.index); r = E6.mede(decide_v2(substitui(proc, canal, mapeia_quantis(e, proc[canal], vigS)), det), reg); ms.append(r)
            d = [r.det for r in ms]; f = [r.fp for r in ms]; o = [r.horas_alarme_fora for r in ms]
            ver = "DIGNO" if np.median(d) >= 6 and np.median(f) <= 9 and np.median(o) <= 407 else "PROMISSOR" if np.median(d) >= 5 and np.median(f) <= 12 and np.median(o) <= 800 else "-"
            print(f"   {var:15s}: detecção {np.median(d):.0f}/7 ({min(d)}-{max(d)}; por n {d}) | FP {np.median(f):.0f} ({min(f)}-{max(f)}; por n {f}) | fora {np.median(o):.0f} h ({min(o):.0f}-{max(o):.0f}) -> {ver}{' | REDUZ O VIÉS' if canal == 't' and np.median(f) <= 12 else ''}", flush=True)


if __name__ == "__main__":
    main()
