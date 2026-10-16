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

RESULTADO (15/10/2026) -- A H10 É REFUTADA PARA O FP; NENHUMA VARIANTE DE ESCALA OU DE CONJUNTO DE TREINO FECHA O VIÉS DO HARNESS.
  PARTE A (26 tags, instantes vigiados). t: a TI_0305 atinge o máximo do escore em 27,6% dos instantes (as demais <= 10,4%: TC382_03_A 10,4%, TC382_05_A 8,6%); TODAS as 14 tags têm razão máx/mín do MAD mensal entre 4,1 e 13,0 (as TC382_xx_A e a T5_AVG_A, de MAD ~35-40, de 10 a 13x). p: nenhuma tag domina (PI_0308 17,3%, PDI_0302 15,6%, PI_0315 11,7%, PI_5134001 10,9%, PI_0339 10,5%); a PI_0319 só 0,3% (o PCA de k = 4 a absorve), mas tem 40,9% de passos iguais e razão MAD mensal de 3,9e11 (MAD ~0 em algum mês: sensor discretizado/travado); a PDIT_0305 tem 59,2% de passos iguais e só 2.016 valores distintos; a PDI_0301 tem razão de 98,9 (a escala mudou ~100x entre meses); a PI_5134001 (de outra unidade) continua na família p do bundle. Total: 23 das 26 tags com razão MAD mensal > 3.
  PARTE B (mediana dos 3 n de treino; entre parênteses a faixa; detecção | FP | h fora). Canal t: V0 base 5/7 (4-7), 16 (15-17), 638 h | V1 sem gating 7/7 (7-7 nos 3 n), 15 (14-16), 603 h (569-679) | V2 padrão 6/7 (6-6), 15 (15-17), 976 h (679-1.145) | V3 piso 15% 5/7 (4-7), 16, 638 | V4 winsorizada 5/7 (4-6), 16 (14-16), 650. Canal p: V0 6/7 (5-6), 10 (9-13), 746 h | V1 6/7, 11 (10-14), 659 h (455-1.450) | V2 6/7, 12 (10-16), 701 | V3 6/7, 11 (9-14), 754 | V4 6/7, 12 (10-16), 830. Nenhum DIGNO; em t nenhum PROMISSOR (FP 15-16) e nenhum REDUZ O VIÉS (FP <= 12); em p V0, V1, V2 e V3 PROMISSOR.
LEITURA. (1) A H10 falha no FP: treinar em TODAS as amostras vigiadas (com transiente e anormalidade) NÃO baixa o FP de t (15 contra 16). O que a V1 faz é ESTABILIZAR a detecção em t (7/7 nos três n, contra 4-7 da V0) e estreitar as horas fora (569-679 h contra 590-987 h): ganho de estabilidade, não de ponto de operação (leitura da faixa pré-registrada, não de um critério escrito). (2) Escala padrão, piso de 15% e winsorização não mexem no viés (FP 15-16 em t; em p a padrão e a winsorizada pioram +2 FP). Logo o excesso de FP do harness contra o bundle NÃO é tipo de escala nem composição do conjunto de treino. (3) A escala dos sensores muda de 4x a 13x entre meses em TODAS as tags de t e em 9 de 12 de p, com 3 casos degenerados (PI_0319, PDIT_0305, PDI_0301): é a explicação do p99 mensal instável do EXP 16 e confirma o normalizador como ponto frágil, mas também que o v2 convive com isso (os bundles mensais são re-estimados). (4) O viés restante tem de estar em OUTRO lugar (candidatos, sem teste ainda): a defasagem do bundle (ele é fixado por data e 'vigência', não retreinado no instante), o mapa de quantis global que só iguala a marginal, ou a calibração de base/k do v2 ao nível do próprio bundle. Não afirmo causa.
PLACAR: ERREI: a PI_0319 domina a participação no máximo de p (>= 40%, ~65%: deu 0,3%); uma tag atinge >= 30% dos máximos em t (~55%: a TI_0305 com 27,6%, por pouco); V1 reduz o FP de t a <= 12 em >= 2 de 3 n (H10, ~35%); algum PROMISSOR em t (~35%: nenhum). ACERTEI: >= 3 tags com razão MAD mensal > 3 (~70%: 23 de 26); V3 neutra (+-2 FP, ~55%); nenhum DIGNO (~94%); V4 aumenta o FP em p (~60%: +2), não em t. PARCIAL: V2 pior que V0 em FP (~65%): em t 15 contra 16 (não), em p 12 contra 10 (sim), e em t pior em horas fora (976 contra 638).

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
            linhas.append(dict(familia=canal, tag=tg.replace("954005_624_", ""), distintos=nun, passos_iguais=round(iguais, 3), mad=round(float(1.4826 * np.median(np.abs(xv - np.median(xv)))), 4), razao_mad_mensal=round(razao, 1), part_max=round(float(part[canal][j]), 3)))
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
