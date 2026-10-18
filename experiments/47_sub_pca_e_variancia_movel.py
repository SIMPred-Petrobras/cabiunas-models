#!/usr/bin/env python3
"""EXP 47 — SUB-PCA (PCA em janelas) E VARIÂNCIA MÓVEL: os dois candidatos da leitura completa da literatura (18/10) que ainda não tinham sido testados.

PRÉ-REGISTRO (18/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. (a) O TSB-AD (NeurIPS 2024, lido em texto completo) põe o Sub-PCA em 1º lugar no univariado (VUS-PR 0,42) e o PCA entre os 3 primeiros no multivariado; o Sub-PCA é o PCA sobre janelas de subsequência (o vetor de uma janela de w amostras), e nós só testamos o PCA de uma amostra (o v2) e o PCA com k variável, nunca com o contexto temporal na própria projeção. (b) O TimeRCD (2509.21190) reconhece que um detector simples de VARIÂNCIA MÓVEL é competitivo (em SWaT e MSL chega a superar o método proposto em afiliação-F1); nós só testamos níveis (z, resíduo), nunca a dispersão local.
O QUE SE FAZ (8 eventos; harness fiel: RobustScaler do sklearn, retreino mensal causal nas últimas n amostras estáveis, escore normalizado pelo p99 do treino por sensor com piso 0,1 x mediana, a partir de 01/02/2025; EWMA 1 h; episódio >= 2 h; refratário 72 h; o MESMO avaliador do EXP 46: limiar por bisseção para o orçamento de 1.457 h de alarme desde 01/02/2025, sem rótulos, entre o quantil 5% do escore e o máximo).
  Variantes (5, FIXADAS AQUI, sem ajuste depois): Sub-PCA com janela de 3, 10 e 30 amostras (6 min, 20 min, 1 h; PCA com 95% da variância sobre o vetor de janela de d*w entradas; o erro de reconstrução é a média, por sensor, sobre as w posições da janela; o escore é o máximo entre sensores do erro normalizado pelo p99 do sensor); variância móvel de 30 e 180 amostras (1 h e 6 h; desvio padrão móvel CAUSAL de cada sensor, escala robusta, escore = máximo entre sensores do desvio ao quadrado normalizado pelo p99 do sensor). Canais t e p, isolados. Tamanhos de treino n = 20.000 e 19.000 (como o EXP 44).
  Referência: o PCA(0,95) de uma amostra, o mesmo avaliador, n = 20.000 e 19.000 (escores_sub34).
MEDIDAS: detecção de 8, FP, horas, limiar; placebo (eventos deslocados de +45 a +225 d, o mesmo alarme); curva de orçamento (300, 600, 1.000, 1.457 h) das variantes e da referência.
CRITÉRIOS (fixados antes; os do EXP 44). GANHO = a média da detecção das 2 variantes de n >= a da referência + 2 e acima do placebo do próprio modelo; PIOR = <= a da referência - 2; EQUIVALENTE no resto. Piso de ruído com 8 eventos: +-2 (e a referência, medida no EXP 46, oscila 2 eventos só por mover o limiar ~7%). COMO SÃO 5 VARIANTES por canal (melhor de 5): qualquer GANHO só vale depois de CONFIRMAÇÃO em n = 21.000 e 18.000 (escolhidos agora, antes de ver o resultado); sem a confirmação, é EQUIVALENTE.
EXPECTATIVAS: Sub-PCA: EQUIVALENTE ao PCA ~60%, PIOR ~28%, GANHO ~12% (por variante e canal); variância móvel: PIOR ~55%, EQUIVALENTE ~40%, GANHO ~5%; algum GANHO confirmado ~7%; a janela mais longa (30) com mais FP que a curta ~55%; a variância móvel de 6 h com o menor número de episódios ~65%.
INTERPRETAÇÃO PREVISTA. Se nenhum ganha, o contexto temporal na projeção e a dispersão local não acrescentam nada ao nível do PCA de uma amostra e a lista da literatura de detecção sem rótulos fecha; se alguma variante ganha e confirma, vira HIPÓTESE para entrar como canal na estrutura de voto (EXP seguinte, com placebo e vizinhança).
LIMITES. 8 eventos (+-2); canais isolados (sem voto nem CUSUM); hiperparâmetros por convenção (5 variantes, sem seleção); o Sub-PCA do TSB-AD é univariado e a nossa versão é multivariada por concatenação; a variância móvel do TimeRCD é a mais simples possível.

RESULTADO (18/10/2026; rodado no worker do ClearML, 64 CPUs e 131 GB, tarefa 8124319ab264491d829983b01715f8eb, commit 1e74eb3; escores em 491 s). NENHUM GANHO. A referência reproduz o escores_sub34 (diferença máxima 1,7e-13).
  Detecção média de 8 (n = 20.000 e 19.000) contra o PCA(0,95): t (ref 5,5): Sub-PCA 3 amostras 5,5, 10 amostras 4,5, 30 amostras 6,0, variância móvel 1 h 2,0 (PIOR), 6 h 6,0; p (ref 6,5): Sub-PCA 3 amostras 5,5, 10 amostras 5,0, 30 amostras 4,0 (PIOR), variância móvel 1 h 4,5 (PIOR), 6 h 4,5 (PIOR). Sub-PCA: 5 EQUIVALENTE e 1 PIOR em 6; variância móvel: 3 PIOR e 1 EQUIVALENTE em 4. Nenhum GANHO, então a confirmação em n = 21.000 e 18.000 (calculada de qualquer forma) não decide nada.
  1) Sub-PCA com janela de 30 em t é o único fio solto, e fraco: detecta 6, 6, 6 e 5 nos 4 n (referência 5, 6, 4 e 5) e dá menos FP em todos (36, 33, 41, 40 contra 47, 43, 46, 54; média -10 no mesmo orçamento de horas). Mas o ganho de detecção (+0,5 nos 2 n do critério; +0,75 nos 4) fica abaixo do critério de +2, e o MESMO Sub-PCA 30 em p é o pior (3, 5, 8, 5): o ruído de n (de 2 a 3 eventos) domina. Hipótese de prior ~15%, não se adota.
  2) A variância móvel gera 67 a 81 FP (a referência, 43 a 54) e o placebo detecta 47 a 71% dos eventos deslocados: o alarme cobre tanto tempo que "detecta" ao acaso; não discrimina. Rejeitada.
  3) A própria referência oscila com n: t 5, 6, 4, 5 e p 7, 6, 6, 6 (n = 20.000, 19.000, 21.000, 18.000).
  CONCLUSÃO: o contexto temporal na projeção (Sub-PCA) e a dispersão local (variância móvel) não acrescentam nada demonstrável ao PCA de uma amostra; a lista da literatura de detecção sem rótulos fecha sem ganho.
PLACAR: ACERTEI: nenhum GANHO confirmado (~93%); Sub-PCA EQUIVALENTE em 5 de 6 e PIOR em 1 (~60% e ~28%); variância móvel PIOR em 3 de 4 (~55%). ERREI: previ a janela longa (30) com mais FP que a curta (~55%) e deu MENOS (t 36, 33, 41, 40 contra 40, 48, 47, 49 da janela de 3; p 48, 43, 43, 44 contra 50, 51, 57, 51); previ a variância móvel de 6 h com o menor número de episódios (~65%) e foi a de MAIS FP (71 a 81).

Uso:  PYTHONPATH=src .venv/bin/python experiments/47_sub_pca_e_variancia_movel.py
      no worker: PYTHONPATH=src python scripts/worker_exp47.py --enqueue --queue default
      variáveis: EXP47_PROCS, EXP47_TESTE=1 (encanamento: 70 dias, 2 variantes, n=20.000)
"""
from __future__ import annotations
import contextlib, importlib.util, io, os, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

TESTE = bool(os.environ.get("EXP47_TESTE"))
if TESTE:
    os.environ["EXP46_TESTE"] = "1"
warnings.filterwarnings("ignore")
HERE = Path(__file__).parent


def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


E46 = _carrega("exp46", "46_optuna_modelos_loeo_placebo.py"); E34 = E46.E34; E20 = E46.E20
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import carrega_controle
from nv.controle_sinais import _ci, PKG
from nv.modelos.familias import _Residuo
from nv.walk import walk_forward

ROOT = E46.ROOT; INICIO8 = E46.INICIO8; CACHE = ROOT / "escores_lit47"; GL = {}
VARIANTES = [("sub", 3), ("sub", 10), ("sub", 30), ("mv", 30), ("mv", 180)]; NS = (20000, 19000); NS_CONF = (21000, 18000)
if TESTE:
    VARIANTES = [("sub", 3), ("mv", 30)]; NS = (20000,)
    NS_CONF = ()
DADOS_DESDE = pd.Timestamp("2024-08-01", tz="UTC")


def _subpca(w, d):
    class S(_Residuo):
        def _ajusta(self, Z):
            self.pca = PCA(n_components=0.95, svd_solver="full").fit(Z)
        def _erro2(self, Z):
            R = (Z - self.pca.inverse_transform(self.pca.transform(Z))) ** 2
            return R.reshape(len(Z), w, d).mean(axis=1)
    return E20.com_sk(S)


def _mv():
    class M(_Residuo):
        def _ajusta(self, Z):
            pass
        def _erro2(self, Z):
            return Z ** 2
    return E20.com_sk(M)


def deriva(g, tags, fam, w):
    """Colunas derivadas CAUSAIS: lag-major (w posições x d sensores) para o Sub-PCA; desvio padrão móvel por sensor para a variância móvel."""
    X = g[tags]
    if fam == "sub":
        cols = {f"{t}__l{l}": X[t].shift(l) for l in range(w) for t in tags}
    else:
        cols = {f"{t}__sd{w}": X[t].rolling(w, min_periods=w).std() for t in tags}
    return pd.DataFrame(cols, index=g.index)


def pontua(a):
    canal, fam, w, n = a; nome = f"{fam}{w}"; f = CACHE / f"{canal}_{nome}_{n}.parquet"
    if f.exists():
        return a
    t0 = time.time(); tags = GL["tags"][canal]
    if fam == "ref":
        pref = {"fam": "pca", "var": 0.95}; s = walk_forward(lambda semente=0: E46.fabrica(pref)(semente=semente), GL["g"], tags, GL["est"], INICIO8, GL["fim"], n_treino=n, semente=0)
        CACHE.mkdir(parents=True, exist_ok=True); s.loc[GL["ini"]:].to_frame("s").to_parquet(f); print(f"      {canal} ref n={n}: {time.time() - t0:.0f} s", flush=True); return a
    g = GL["g"].loc[DADOS_DESDE:]; est = GL["est"][GL["g"].index >= DADOS_DESDE]; d = len(tags)
    D = deriva(g, tags, fam, w); estw = pd.Series(est, index=g.index).astype(float).rolling(w, min_periods=w).min().fillna(0).to_numpy() > 0.5
    fab = (lambda semente=0: _subpca(w, d)(semente=semente)) if fam == "sub" else (lambda semente=0: _mv()(semente=semente))
    s = walk_forward(fab, D, list(D.columns), estw, INICIO8, GL["fim"], n_treino=n, semente=0)
    CACHE.mkdir(parents=True, exist_ok=True); s.loc[GL["ini"]:].to_frame("s").to_parquet(f); print(f"      {canal} {nome} n={n}: {time.time() - t0:.0f} s", flush=True); return a


def main():
    procs = int(os.environ.get("EXP47_PROCS", 6)); g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    tags = {c: list(modelos[-1]["normalizacao"][f]["cols"]) for c, f in E20.FAMILIAS.items() if c in ("t", "p")}
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual"); vig = vig_c.reindex(g.index).fillna(False); fim = vig_c.index[-1]
    if TESTE:
        fim = INICIO8 + pd.Timedelta(days=70)
    ini = INICIO8 - pd.Timedelta(days=10); GL.update(g=g, est=est, fim=fim, tags=tags, ini=ini)
    EV = {"real": [t for t in reg if t <= fim]}
    for dd in E46.SHIFTS:
        EV[f"p{dd}"] = [t + pd.Timedelta(days=dd) for t in reg if t + pd.Timedelta(days=dd) <= fim]
    print(f"EXP 47: variantes {VARIANTES} | n {NS} | CPUs {os.cpu_count()} | PROCS {procs} | teste={TESTE} | eventos {({k: len(v) for k, v in EV.items()})}", flush=True)
    canais = ("t", "p"); nsx = tuple(NS) + tuple(NS_CONF); t0 = time.time()
    with Pool(procs) as p:
        list(p.imap_unordered(pontua, [(c, fam, w, n) for c in canais for fam, w in [("ref", 0)] + VARIANTES for n in nsx], chunksize=1))
    print(f"ESCORES prontos em {time.time() - t0:.0f} s", flush=True)
    S = {c: {} for c in canais}
    for c in canais:
        for n in nsx:
            S[c][f"ref{n}"] = pd.read_parquet(CACHE / f"{c}_ref0_{n}.parquet")["s"].loc[:fim]
            f34 = E34.CACHE / f"{c}_A_{n}.parquet"
            if f34.exists():
                a34 = pd.read_parquet(f34)["s"].loc[ini:fim]; ii = a34.index.intersection(S[c][f"ref{n}"].index); print(f"   VERIFICAÇÃO [{c}] referência n={n} contra escores_sub34: diferença máxima = {float(np.nanmax(np.abs(a34.loc[ii].to_numpy() - S[c][f'ref{n}'].loc[ii].to_numpy()))):.2e}", flush=True)
            for fam, w in VARIANTES:
                S[c][f"{fam}{w}_{n}"] = pd.read_parquet(CACHE / f"{c}_{fam}{w}_{n}.parquet")["s"].loc[:fim]
    E46.GL.update(S=S, vig=vig); res = {}
    print("\nESCORES (orçamento 1.457 h sem rótulos, EWMA 1 h, persistência 2 h; detecção de 8 | FP | horas | limiar; placebo = fração detectada nos eventos deslocados)", flush=True)
    for c in canais:
        for nome in ["ref"] + [f"{fam}{w}" for fam, w in VARIANTES]:
            lin = []
            for n in nsx:
                k = f"{nome}_{n}" if nome != "ref" else f"ref{n}"
                if k not in S[c]:
                    continue
                o = E46.avalia_combo(c, k, 1.0, 2.0, EV)
                if o is None:
                    lin.append(f"n={n}: INVIÁVEL"); res[(c, nome, n)] = None; continue
                det = sum(o["real"][0]); pl = float(np.median([sum(v[0]) / len(v[0]) for kk, v in o.items() if kk.startswith("p") and kk != "n_ep" and len(v[0])]))
                res[(c, nome, n)] = (det, o["real"][1], pl * 8); lin.append(f"n={n}: {det}/8, FP {o['real'][1]}, {o['horas']:.0f} h, limiar {o['thr']:.3f}, placebo {100 * pl:.0f}%")
            print(f"   {c} <- {nome:7s} | " + " | ".join(lin), flush=True)
    print("\nCURVA DE ORÇAMENTO (n=20.000; detecção de 8 | FP)", flush=True)
    for c in canais:
        for nome in ["ref"] + [f"{fam}{w}" for fam, w in VARIANTES]:
            k = f"{nome}_{NS[0]}" if nome != "ref" else f"ref{NS[0]}"
            if k not in S[c]:
                continue
            cur = []
            for orc in ((30, 60, 100, 150) if TESTE else (300, 600, 1000, 1457)):
                q = E46.avalia_combo(c, k, 1.0, 2.0, {"real": EV["real"]}, orcamento=orc); cur.append(f"{orc} h: " + ("inviável" if q is None else f"{sum(q['real'][0])}/8 FP {q['real'][1]}"))
            print(f"   {c} <- {nome:7s} | " + " | ".join(cur), flush=True)
    print("\nCLASSIFICAÇÃO contra o PCA(0,95) do mesmo canal (média da detecção de n=20.000 e 19.000)", flush=True)
    for c in canais:
        rr = [res.get((c, "ref", n)) for n in NS]; rr = [x for x in rr if x]
        if not rr:
            continue
        ref = float(np.mean([x[0] for x in rr])); print(f"   [{c}] referência: {ref:.1f}/8", flush=True)
        for fam, w in VARIANTES:
            vv = [res.get((c, f"{fam}{w}", n)) for n in NS]; vv = [x for x in vv if x]
            if not vv:
                print(f"      {fam}{w}: sem escore viável", flush=True); continue
            m = float(np.mean([x[0] for x in vv])); pl = float(np.mean([x[2] for x in vv])); cl = "GANHO (a confirmar em n=21.000 e 18.000)" if (m >= ref + 2 and m > pl) else "PIOR" if m <= ref - 2 else "EQUIVALENTE"
            conf = ""
            if cl.startswith("GANHO") and NS_CONF:
                cc = [res.get((c, f"{fam}{w}", n)) for n in NS_CONF]; rc = [res.get((c, "ref", n)) for n in NS_CONF]; cc = [x for x in cc if x]; rc = [x for x in rc if x]
                if cc and rc:
                    mc = float(np.mean([x[0] for x in cc])); refc = float(np.mean([x[0] for x in rc])); conf = f" | confirmação: {mc:.1f}/8 contra {refc:.1f}/8 -> " + ("CONFIRMA" if mc >= refc + 2 else "NÃO CONFIRMA")
            print(f"      {fam}{w:<4d}: {m:.1f}/8 (placebo ~{pl:.1f}/8) -> {cl}{conf}", flush=True)
    pd.DataFrame([dict(canal=c, variante=v, n=n, det=r[0] if r else None, fp=r[1] if r else None, placebo=r[2] if r else None) for (c, v, n), r in res.items()]).to_csv(ROOT / "exp47_resultados.csv", index=False)


if __name__ == "__main__":
    main()
