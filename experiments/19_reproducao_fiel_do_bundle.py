#!/usr/bin/env python3
"""EXP 19 — REPRODUÇÃO FIEL DO AJUSTE DO BUNDLE no harness, e qual diferença responde pelo excesso de FP.

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

O QUE A LEITURA DE `constroi_bundle.py` MOSTROU (sem rodar nada). O bundle e o harness fazem o mesmo walk-forward (os últimos 20.000 pontos estáveis ANTES do dia 1 do mês; a hipótese da 'defasagem' cai), mas diferem em QUATRO coisas do pré-processamento:
  (a) MÁSCARA DO BASELINE: o bundle usa `estavel = RUNNING_A > 0,5 e T5_AVG_A > 300` (operando e quente); o harness usa a máscara de gradiente baixo de `nv.io.mascaras` (mais restrita: só trechos de gradiente pequeno);
  (b) ESCALA: o bundle usa `sklearn.RobustScaler()` (mediana e intervalo interquartil, sem piso); o harness, mediana/(1,4826 MAD) com piso de 5% da mediana;
  (c) NÚMERO DE COMPONENTES: o bundle usa `PCA(n_components=0,95)` (a fração de variância retida: k VARIA por mês e por família); o harness fixou k do último bundle (1 em t, 4 em p) em todos os meses;
  (d) LINHAS DO BASELINE: o bundle faz `dropna` sobre TODAS as colunas (temperatura, pressão e vibração) antes de ajustar cada família; o harness só nas tags da família.
  (O resto, p99 por sensor com piso 0,1 x mediana e o máximo normalizado por recon_p99, é igual.)
PARTE A — REPRODUÇÃO: ajustar, por mês, o `ajusta_familia` do bundle (as quatro coisas acima) sobre a grade e comparar com os bundles gravados: número de componentes, center e scale (máxima diferença relativa) e recon_p99, para t e p, nos meses servidos de 2025-03 a 2026-04.
PARTE B — ABLAÇÃO CUMULATIVA no plug-in (substituição do canal, EWMA 1 h vigiado, mapa de quantis, `ci.prever`), t e p:
   S0 o harness de antes (EXP 14/17: máscara de gradiente, MAD com piso, k fixo);
   S1 = S0 + máscara do bundle e dropna em todas as colunas (a e d);
   S2 = S1 + RobustScaler do sklearn (b);
   S3 = S2 + PCA(0,95) (c) = a reprodução fiel do bundle.
   (Para o piso de ruído, S3 também com n de treino 19.000 e 21.000.)
CRITÉRIOS (fixados antes). REPRODUZ = S3 em t com detecção >= 6/7 e FP <= 11 e horas fora <= 500 (o controle: 6/7, 9, 407 h; o S0 dá 7/7, 15, 590); nos parâmetros: REPRODUZ OS PARÂMETROS = >= 12 dos 16 bundles com mesmo número de componentes em t e p E diferença relativa máxima de center/scale < 1e-3. Cada passo S0->S1->S2->S3 'explica' o viés se o FP de t cai >= 2 nesse passo.
EXPECTATIVAS: S3 REPRODUZ em t ~70%; parâmetros reproduzidos ~50% (a grade e a máscara de produção podem diferir da pesquisa em bordas); o passo que mais corta o FP é a máscara (a) ~40%, a escala (b) ~25%, o número de componentes (c) ~35%; em p, S3 chega a <= 10 FP e 6/7 ~65%.
CONSEQUÊNCIA PREVISTA. Se S3 reproduz, os EXP 14, 15, 17 e 18 (todos com a escala e o k do harness antigo) precisam ser relidos: o excesso de FP de ~6 que atribuí a 'instabilidade' era o harness infiel; a pergunta de substituir t e p por outros escores merece ser refeita sobre o harness fiel. Se S3 não reproduz, a diferença está na grade, na máscara de vigilância ou no mapa de quantis.
LIMITES. 7 eventos; a máscara do bundle é reproduzida a partir das colunas RUNNING_A e T5_AVG_A da grade (se a grade de pesquisa diferir da de produção em bordas, a reprodução exata dos parâmetros pode falhar sem afetar o FP); nada se adota sem sombra.

RESULTADO (15/10/2026) -- O HARNESS ANTIGO ERA INFIEL AO BUNDLE; A REPRODUÇÃO FIEL FUNCIONA, e corrige o que concluí nos EXP 14 e 16.
  PARTE A: o ajuste fiel (`ajusta_familia` do bundle sobre a grade) REPRODUZ OS PARÂMETROS DOS 14 BUNDLES GRAVADOS (2025-03 a 2026-04, os com `mes_servido`) EXATAMENTE: o mesmo número de componentes, center e scale com diferença relativa 0 e recon_p99 com razão 1,000, em t e em p (limite escrito: >= 12 de 16; saiu 14 de 14). A grade da pesquisa, a máscara e o dropna reproduzem a produção.
  O NÚMERO DE COMPONENTES VARIA POR MÊS (k do bundle): em t = 3, 3, 2, 3, 2, 3, 1, 3, 3, 3, 3, 2, 2, 1 (2025-03 a 2026-04); em p = 7, 2, 1, 1, 2, 1, 1, 6, 4, 1, 8, 1, 1, 4. O harness antigo fixou k = 1 em t e k = 4 em p (o k do ÚLTIMO bundle de t e um valor arbitrário em p) em todos os meses.
  PARTE B (plug-in; detecção | início | FP | h fora). Canal t: S0 harness antigo 7/7 | 6 | 15 | 590 h; S1 + máscara do bundle e dropna: IDÊNTICO ao S0 (a máscara estável da nova frente JÁ É `RUNNING_A > 0,5 e T5 > 300`, a mesma do bundle; o gradiente baixo que supus na minha leitura do CLAUDE.md não está em `nv.io`); S2 + RobustScaler (IQR) com k fixo: 5/7 | 5 | 16 | 1.264 h (piora); S3 + PCA(0,95) = fiel: 6/7 | início 5 | FP 11 | fora 467 h | lead 33,5 h (n = 19.000: 6/7, 13 FP; n = 21.000: 5/7, 15 FP). Canal p: S0 6/7 | 4 | 9 | 619 h; S1 igual; S2 6/7 | 4 | 11 | 709 h; S3 fiel: 6/7 | início 5 | FP 9 | fora 390 h (n = 19.000: 6/7, 9 FP; n = 21.000: 6/7, 8 FP): igual ao controle e estável em n.
  CRITÉRIOS: REPRODUZ (S3 em t: detecção >= 6/7, FP <= 11, <= 500 h fora): SIM (6/7, 11, 467 h; pelo limite). REPRODUZ OS PARÂMETROS: SIM (14/14).
LEITURA E CORREÇÕES. (1) A fidelidade vem de DOIS itens do pré-processamento do bundle: a escala RobustScaler do sklearn (mediana/IQR) e o número de componentes PCA(0,95) variável por mês; a máscara e o dropna não mudam nada nestes dados. A escala SOZINHA piora (S2), e é a combinação com o k variável que reproduz (S3): os dois são um só ponto de ajuste. (2) O que sobra entre o S3 e o controle em t (+2 FP, 467 contra 407 h) é, com boa probabilidade, a diferença de EWMA (o harness usa o EWMA só sobre instantes vigiados, o v2 sobre o sinal cru: memória 'o-ewma-come-a-partida') e o mapa de quantis; não testei isso aqui. (3) CORRIGE O EXP 14: a conclusão 'a rigidez de k = 1 é o que faz o canal t; pca3, ridge e GBM perdem' vale só para o harness infiel (escala MAD com piso e k fixo): o canal t do v2 tem k = 3 na maior parte dos meses (1 só em 2025-09 e 2026-04) e a premissa 'o bundle atual tem k = 1' era a do último bundle, não a regra. Retiro a hipótese da rigidez; os resultados do EXP 14 (escores por família), 15 (previsor), 17 (ensemble) e 18 (ablação) foram obtidos no harness infiel e precisam ser REFEITOS no harness fiel antes de qualquer leitura sobre o escore. (4) CORRIGE O EXP 16: a explicação 'instabilidade do normalizador, não defeito do harness' estava ERRADA no que afirmou: o harness TINHA um defeito (escala e k), e é ele que explica o 15 FP contra 9 e a variação do p99 mensal contra o bundle (razão até 91x). A instabilidade da escala por mês dos dados (EXP 18, razão de MAD mensal de 4x a 13x) continua verdadeira, mas não é a causa daquele descasamento.
PLACAR: ACERTEI: S3 REPRODUZ em t (~70%); em p S3 chega a <= 10 FP e 6/7 (~65%: 9 FP). ERREI: parâmetros reproduzidos (~50%: saiu 14 de 14, exato); o passo que mais corta o FP (previ a máscara, ~40%: a máscara não muda nada; o corte vem do k variável com o IQR, 16 para 11 em t, mas a escala sozinha piora). INCONCLUSIVO: a atribuição de cada passo isolado (os dois últimos passos são um só ajuste).

Uso:  PYTHONPATH=src .venv/bin/python experiments/19_reproducao_fiel_do_bundle.py
"""
from __future__ import annotations
import contextlib, importlib.util, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

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
from nv.plugin import decide_v2, mapeia_quantis, substitui
from nv.politica import ewma_vigiado
from nv.walk import inicios_de_mes

INICIO = E4.INICIO; ROOT = Path(__file__).resolve().parents[1] / "data_local"; PHI = 0.10
FAM = {"t": "temperatura", "p": "pressao"}


class _Esc(Escala):
    pass


def ajusta(X, escala, k_fixo):
    """Reproduz `ajusta_familia` (escala: 'sk' = RobustScaler do sklearn; 'mad' = Escala do harness; k_fixo=None => PCA(0,95))."""
    esc = RobustScaler().fit(X) if escala == "sk" else Escala().fit(X); Xs = esc.transform(X)
    pca = PCA(n_components=0.95 if k_fixo is None else k_fixo, svd_solver="full").fit(Xs)
    err = (Xs - pca.inverse_transform(pca.transform(Xs))) ** 2; p99 = np.nanpercentile(err, 99, axis=0); sens = np.maximum(p99, PHI * np.nanmedian(p99))
    return dict(esc=esc, pca=pca, sens=sens, recon_p99=float(np.nanpercentile(np.max(err / sens, axis=1), 99)), k=int(pca.n_components_))


def pontua(m, X):
    out = np.full(len(X), np.nan); ok = np.isfinite(X).all(axis=1)
    if ok.any():
        Xs = m["esc"].transform(X[ok]); e = (Xs - m["pca"].inverse_transform(m["pca"].transform(Xs))) ** 2; out[ok] = np.max(e / m["sens"], axis=1) / m["recon_p99"]
    return out


def walk(g, cols, extra, mascara, escala, k_fixo, n, ini, fim, guarda=None):
    """Um ajuste por mês sobre os últimos n pontos de `mascara` (dropna em cols + extra); guarda os ajustes em `guarda` (dict mes -> modelo)."""
    X = g[cols].to_numpy("float64"); XA = g[cols + extra].to_numpy("float64"); idx = g.index; out = np.full(len(g), np.nan)
    for j, m0 in enumerate(inicios_de_mes(ini, fim)):
        m1 = inicios_de_mes(ini, fim + pd.Timedelta(days=40))[j + 1]; i0 = idx.searchsorted(m0); i1 = idx.searchsorted(m1)
        cand = np.flatnonzero(mascara[:i0] & np.isfinite(XA[:i0]).all(axis=1))
        if len(cand) < n:
            continue
        mod = ajusta(X[cand[-n:]], escala, k_fixo); out[i0:i1] = pontua(mod, X[i0:i1])
        if guarda is not None:
            guarda[m0.strftime("%Y-%m")] = mod
    return pd.Series(out, index=idx)


def main():
    g = nvio.carrega_grade(); est_gr = nvio.mascaras(g)["estavel"].to_numpy(bool); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tags = {c: list(modelos[-1]["normalizacao"][f]["cols"]) for c, f in FAM.items()}; vib = list(det["tags_vibracao"])
    al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]; reg = [t for t in carrega_registro() if t >= INICIO]; ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); vigS = proc["mask"]; vig_g = vig_c.reindex(g.index).fillna(False)
    est_bundle = ((g["RUNNING_A"] > 0.5).fillna(False) & (g["T5_AVG_A"] > 300).fillna(False)).to_numpy(bool); ini = pd.Timestamp("2025-03-01", tz="UTC")
    print(f"CONTROLE: {E6.linha(ctrl)}", flush=True)
    todas = tags["t"] + tags["p"] + vib; guarda = {}
    for c in FAM:
        walk(g, tags[c], [x for x in todas if x not in tags[c]], est_bundle, "sk", None, 20000, ini, fim, guarda=guarda.setdefault(c, {}))
    print("\nPARTE A — REPRODUÇÃO DOS PARÂMETROS (meses servidos com bundle gravado)", flush=True); iguais = {"t": 0, "p": 0}; tot = 0; linhas = []
    por_mes = {m["modelo"].get("mes_servido"): m for m in modelos}
    for mes in sorted(guarda["t"]):
        b = por_mes.get(mes)
        if b is None:
            continue
        tot += 1; lin = dict(mes=mes)
        for c, f in FAM.items():
            mod = guarda[c][mes]; tr = b[f + "_transformacao"]; nb = b["normalizacao"][f]; kb = int(nb["n_componentes"])
            dc = float(np.max(np.abs(mod["esc"].center_ - tr["center"]) / np.maximum(np.abs(tr["center"]), 1e-9))); ds = float(np.max(np.abs(mod["esc"].scale_ - tr["scale"]) / np.maximum(np.abs(tr["scale"]), 1e-9)))
            lin.update({f"k_{c}": f"{mod['k']}/{kb}", f"dcenter_{c}": dc, f"dscale_{c}": ds, f"rp99_{c}": round(mod["recon_p99"] / float(nb["recon_p99"]), 3)}); iguais[c] += int(mod["k"] == kb and dc < 1e-3 and ds < 1e-3)
        linhas.append(lin)
    print(pd.DataFrame(linhas).to_string(index=False, float_format=lambda x: f"{x:.2g}"), flush=True)
    print(f"   REPRODUZ OS PARÂMETROS: t {iguais['t']}/{tot}, p {iguais['p']}/{tot} bundles com o mesmo k e center/scale a < 1e-3 (limite escrito: >= 12 de 16 em t e p)", flush=True)

    def mede(s, canal):
        e = ewma_vigiado(s, vig_g, "1h").reindex(proc.index); return E6.mede(decide_v2(substitui(proc, canal, mapeia_quantis(e, proc[canal], vigS)), det), reg)
    print("\nPARTE B — ABLAÇÃO CUMULATIVA (plug-in; detecção | início | FP | h fora | h totais)", flush=True)
    for canal in FAM:
        k0 = 1 if canal == "t" else 4; extra_f = lambda x: []; passos = [("S0 harness antigo", est_gr, "mad", k0, False), ("S1 +máscara do bundle e dropna", est_bundle, "mad", k0, True), ("S2 +RobustScaler (IQR)", est_bundle, "sk", k0, True), ("S3 +PCA(0,95) = fiel", est_bundle, "sk", None, True)]
        print(f"CANAL {canal}", flush=True)
        for nome, masc, esc, kf, dropna_todas in passos:
            ex = [x for x in todas if x not in tags[canal]] if dropna_todas else []
            r = mede(walk(g, tags[canal], ex, masc, esc, kf, 20000, ini, fim), canal)
            print(f"   {nome:34s}: {r.det}/7 | início {r.inicio} | FP {r.fp:2d} | fora {r.horas_alarme_fora:5.0f} h | totais {r.horas_alarme:5.0f} h | lead {r.lead_medio_h:.1f} h", flush=True)
        for nt in (19000, 21000):
            r = mede(walk(g, tags[canal], [x for x in todas if x not in tags[canal]], est_bundle, "sk", None, nt, ini, fim), canal)
            print(f"   S3 fiel, n_treino {nt}                : {r.det}/7 | início {r.inicio} | FP {r.fp:2d} | fora {r.horas_alarme_fora:5.0f} h | totais {r.horas_alarme:5.0f} h", flush=True)


if __name__ == "__main__":
    main()
