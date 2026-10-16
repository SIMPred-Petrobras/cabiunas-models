#!/usr/bin/env python3
"""EXP 21 — (A) DECOMPOSIÇÃO DOS +2 FP DO HARNESS FIEL (EWMA e mapa de quantis) E (B) ROBUSTEZ DO v2 AO LIMIAR DE VARIÂNCIA DO PCA E AO k POR MÊS.

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

CONTEXTO. O EXP 19 mostrou que o ajuste fiel reproduz os 14 bundles (k, center, scale, recon_p99 idênticos) mas o plug-in dá em t 6/7, 11 FP, 467 h (controle 9 FP, 407 h) e em p 6/7, 9 FP, 390 h. O que o harness ainda faz diferente do v2 depois do escore: (i) o EWMA só sobre instantes vigiados (`ewma_vigiado`), e o v2 o EWMA sobre o sinal cru inteiro (`cru.ewm(halflife, times).mean().where(cru.notna())`, atravessando parada e blackout); (ii) o MAPA DE QUANTIS global para a escala do canal do v2, que o v2 não usa (o escore do bundle já está em unidades de recon_p99).
PARTE A — decomposição, em t e em p, com o escore fiel BRUTO do EXP 19 (n = 20.000):
   A1 exato do v2: EWMA do v2 sobre o escore cru, sem mapa, direto no canal;
   A2 EWMA vigiado, sem mapa;
   A3 EWMA do v2 + mapa de quantis;
   A4 EWMA vigiado + mapa (o harness do EXP 19/20).
  VALIDADE (escrita antes): A1 deve reproduzir o controle (6/7, 9 FP, ~407 h fora) e a série de t e p do v2 (correlação >= 0,999 nos instantes vigiados): é a prova de que o caminho de produção está inteiro no harness.
PARTE B — robustez, pelo caminho A1 (que reproduz a produção), um canal por vez:
   B1 limiar de variância do PCA em {0,90, 0,93, 0,95*, 0,97, 0,99};
   B2 k suavizado (causal): k_m = mediana de {k de m-2, m-1, m} (arredondada), PCA com esse k fixo;
   B3 k = o do mês anterior (a persistência do k).
  e a distribuição de k por mês de cada configuração.
CRITÉRIOS (fixados antes). ROBUSTO a B1 = em AMBOS os vizinhos (0,93 e 0,97) a detecção fica a +-1, o FP a +-2 e as horas fora a +-30% do valor com 0,95, em t e em p. GANHO de B2/B3 = detecção >= a do 0,95, FP <= FP(0,95) - 2 e horas fora <= 0,8 x as do 0,95. Parte A: 'o mapa explica o FP' = o FP de A3 menos o de A1 >= 2; 'o EWMA vigiado explica' = A2 menos A1 >= 2 em módulo.
EXPECTATIVAS: A1 reproduz o controle e a série do v2 (~85%); o mapa de quantis explica >= 2 FP em t (~45%); o EWMA vigiado reduz o FP e perde detecção em t (~55%, como na memória 'ewma-come-a-partida'); ROBUSTO a B1 nos dois canais (~35%: o precipício que o k_p já mostrou); algum GANHO de B2/B3 (~10%).
LIMITES. 7 eventos; B1 mexe num parâmetro que o v2 fixou em 0,95 sem varredura publicada; os vizinhos 0,93 e 0,97 são escolha minha; nenhum valor se adota sem sombra.

RESULTADO (15/10/2026) -- (A) O HARNESS REPRODUZ A PRODUÇÃO INTEIRA, E O +2 FP VEM DO EWMA VIGIADO, NÃO DO MAPA; (B) O PONTO 0,95 É UM PICO EM t.
  PARTE A. VALIDADE: PASSA. O EWMA do v2 sobre o escore fiel dá correlação 1,000000 com o canal t e com o canal p do v2 (220.667 e 220.665 instantes vigiados) e A1 reproduz o controle IDÊNTICO nas quatro medidas (6/7, início 5, 9 FP, 407 h fora, 1.304 h totais, lead 32,7 h) em t e em p. O caminho de produção está inteiro no harness.
    Canal t: A1 6/7 | 9 FP | 407 h | A2 (EWMA vigiado, sem mapa) 6/7 | 11 FP | 464 h | A3 (EWMA cru + mapa) idêntico a A1 | A4 (EXP 19/20) 6/7 | 11 FP | 467 h. Canal p: A1 6/7 | 9 FP | 407 h | A2 6/7 | 9 FP | 389 h | A3 idêntico a A1 | A4 6/7 | 9 FP | 390 h.
    O mapa de quantis NÃO tem efeito (quando o canal já está na escala do v2 ele vira identidade: A3 = A1); os +2 FP de t do harness vêm do EWMA só sobre os instantes vigiados (A2 = 11 FP; em p não muda o FP). Isso NÃO reproduz a memória 'o-ewma-come-a-partida' (que na régua de 8 composições dava FP/mês 0,861 -> 0,517 com o EWMA vigiado): aqui, nos bundles de produção e na base causal de 7 eventos, o EWMA vigiado soma +2 FP em t sem perder detecção.
  PARTE B (caminho A1, detecção | FP | h fora | k por mês). Canal t: variância 0,90: 3/7 | 14 | 880 | 0,93: 5/7 | 14 | 880 | 0,95*: 6/7 | 9 | 407 | k [3,3,2,3,2,3,1,3,3,3,3,2,2,1] | 0,97: 5/7 | 10 | 453 | 0,99: 5/7 | 15 | 512 | B2 k suavizado (mediana de 3 meses): 6/7 | 9 | 407 (idêntico ao 0,95 nas métricas; verifiquei que as séries diferem nos meses em que o k muda, o resultado do alarme não) | B3 k do mês anterior: 6/7 | 11 | 496. Canal p: 0,90: 6/7 | 11 | 508 | 0,93: 6/7 | 9 | 408 | 0,95*: 6/7 | 9 | 407 | k [7,2,1,1,2,1,1,6,4,1,8,1,1,4] | 0,97: 5/7 | 10 | 548 | 0,99: 5/7 | 10 | 542 | B2: 6/7 | 13 | 988 | B3: 6/7 | 14 | 1.066.
  ROBUSTO a B1: NÃO. Em t o vizinho 0,93 piora muito (FP 9 -> 14, horas fora 407 -> 880, detecção 6 -> 5; vários meses caem de k = 3 a k = 2) e o 0,97 perde 1 detecção dentro do piso (FP +1, +11% de horas); em p o 0,93 é igual mas o 0,97 perde 1 detecção e sobe 35% as horas fora. Nenhum GANHO em B2 ou B3 (B2 em t empata; em p piora; B3 piora nos dois).
LEITURA. (1) O harness fiel reproduz o v2 de produção exatamente; a referência dos experimentos é o próprio v2, e o EWMA vigiado (usado nas comparações dos EXP 04-20) custa cerca de +2 FP em t. (2) O limiar de variância de 0,95 do PCA é um PICO para o detector: em t, tanto 0,93 como 0,97 e 0,99 ficam pior (detecção 5/7, 14 a 15 FP em 0,93 e 0,99); em p é plano até 0,93 e cai em 0,97. O `N_COMPONENTS = 0,95` é uma convenção do projeto, e o fato de o único valor que rende 6/7 com 9 FP em t ser exatamente ele é compatível com seleção (do conjunto do ponto de operação sobre estes 7 eventos), ou com sorte: o desempenho esperado do v2 com um parâmetro vizinho é de 5/7 e 10 a 14 FP em t. (3) Suavizar o k (mediana de 3 meses) ou persisti-lo (mês anterior) não ajuda: o k 'certo' de cada mês, calculado no próprio baseline, é o melhor. (4) Consequência de comunicação: o 6/7 com 9 FP do ponto publicado é o MELHOR da vizinhança, não o típico; o intervalo honesto do canal t vai de 3/7 a 6/7 e de 9 a 15 FP quando o limiar de variância se mexe +-0,02 a 0,05.
PLACAR: ACERTEI: A1 reproduz o controle e a série (~85%); nenhum GANHO em B2/B3 (~90%). ERREI: o mapa de quantis explica >= 2 FP em t (~45%: zero); o EWMA vigiado reduz o FP e perde detecção em t (~55%: subiu 2 FP sem perder detecção); ROBUSTO a B1 nos dois canais (~35%: não).

Uso:  PYTHONPATH=src .venv/bin/python experiments/21_ewma_mapa_e_robustez_do_k.py
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
E4 = _carrega("exp04", "04_escada_de_baselines.py"); E5 = _carrega("exp05", "05_decomposicao_do_v2.py"); E6 = _carrega("exp06", "06_plugin_de_canais.py"); E19 = _carrega("exp19", "19_reproducao_fiel_do_bundle.py")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.plugin import decide_v2, mapeia_quantis, substitui
from nv.politica import ewma_vigiado
from nv.walk import inicios_de_mes

INICIO = E4.INICIO; PHI = 0.10
FAM = {"t": "temperatura", "p": "pressao"}


def ajusta2(X, ncomp):
    esc = RobustScaler().fit(X); Xs = esc.transform(X); pca = PCA(n_components=ncomp, svd_solver="full").fit(Xs)
    err = (Xs - pca.inverse_transform(pca.transform(Xs))) ** 2; p99 = np.nanpercentile(err, 99, axis=0); sens = np.maximum(p99, PHI * np.nanmedian(p99))
    return dict(esc=esc, pca=pca, sens=sens, recon_p99=float(np.nanpercentile(np.max(err / sens, axis=1), 99)), k=int(pca.n_components_))


def walk2(g, cols, extra, mascara, ncomp_de, n, ini, fim, ks_out=None):
    """Um ajuste por mês; ncomp_de(mes_str) devolve o limiar de variância (float) ou k (int)."""
    X = g[cols].to_numpy("float64"); XA = g[cols + extra].to_numpy("float64"); idx = g.index; out = np.full(len(g), np.nan); meses = inicios_de_mes(ini, fim + pd.Timedelta(days=40))
    for j, m0 in enumerate(meses[:-1]):
        i0 = idx.searchsorted(m0); i1 = idx.searchsorted(meses[j + 1])
        if i0 >= len(g):
            break
        cand = np.flatnonzero(mascara[:i0] & np.isfinite(XA[:i0]).all(axis=1))
        if len(cand) < n:
            continue
        mes = m0.strftime("%Y-%m"); mod = ajusta2(X[cand[-n:]], ncomp_de(mes)); out[i0:i1] = E19.pontua(mod, X[i0:i1])
        if ks_out is not None:
            ks_out[mes] = mod["k"]
    return pd.Series(out, index=idx)


def main():
    g = nvio.carrega_grade(); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tags = {c: list(modelos[-1]["normalizacao"][f]["cols"]) for c, f in FAM.items()}; vib = list(det["tags_vibracao"]); todas = tags["t"] + tags["p"] + vib
    al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]; reg = [t for t in carrega_registro() if t >= INICIO]; ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); vigS = proc["mask"]; vig_g = vig_c.reindex(g.index).fillna(False)
    est_b = ((g["RUNNING_A"] > 0.5).fillna(False) & (g["T5_AVG_A"] > 300).fillna(False)).to_numpy(bool); ini = pd.Timestamp("2025-03-01", tz="UTC")
    print(f"CONTROLE: {E6.linha(ctrl)}", flush=True)

    def bruto(canal, ncomp_de):
        ks = {}; s = walk2(g, tags[canal], [x for x in todas if x not in tags[canal]], est_b, ncomp_de, 20000, ini, fim, ks_out=ks); return s, ks

    def ewma_v2(s, canal):
        return s.ewm(halflife=pd.Timedelta(det["halflife"][canal]), times=s.index).mean().where(s.notna())

    def mede(canal, sinal, mapa):
        e = sinal.reindex(proc.index)
        if mapa:
            e = mapeia_quantis(e, proc[canal], vigS)
        return E6.mede(decide_v2(substitui(proc, canal, e), det), reg)
    print("\nPARTE A — DECOMPOSIÇÃO (escore fiel bruto, n = 20.000)", flush=True)
    cache = {}
    for canal in FAM:
        s, ks = bruto(canal, lambda m: 0.95); cache[canal] = s; e2 = ewma_v2(s, canal).reindex(proc.index); ok = vigS & e2.notna() & proc[canal].notna() & (proc.index >= INICIO)
        corr = float(np.corrcoef(e2[ok], proc[canal][ok])[0, 1]); print(f"   [{canal}] correlação do EWMA do v2 sobre o escore fiel com o canal do v2: {corr:.6f} (instantes {int(ok.sum())})", flush=True)
        for nome, sig, mapa in (("A1 exato do v2 (EWMA cru, sem mapa)", ewma_v2(s, canal), False), ("A2 EWMA vigiado, sem mapa", ewma_vigiado(s, vig_g, "1h"), False), ("A3 EWMA cru + mapa", ewma_v2(s, canal), True), ("A4 EWMA vigiado + mapa (EXP 19/20)", ewma_vigiado(s, vig_g, "1h"), True)):
            r = mede(canal, sig, mapa); print(f"   {canal} {nome:38s}: {r.det}/7 | início {r.inicio} | FP {r.fp:2d} | fora {r.horas_alarme_fora:5.0f} h | totais {r.horas_alarme:5.0f} h | lead {r.lead_medio_h:.1f} h", flush=True)
    print("\nPARTE B — ROBUSTEZ (caminho A1 = produção)", flush=True)
    for canal in FAM:
        print(f"CANAL {canal}", flush=True); base = None; kref = None
        for v in (0.90, 0.93, 0.95, 0.97, 0.99):
            s, ks = bruto(canal, lambda m, v=v: v); r = mede(canal, ewma_v2(s, canal), False)
            if v == 0.95:
                base = r; kref = ks
            print(f"   variância {v:.2f}{'*' if v == 0.95 else ' '}: {r.det}/7 | início {r.inicio} | FP {r.fp:2d} | fora {r.horas_alarme_fora:5.0f} h | totais {r.horas_alarme:5.0f} h | k por mês {list(ks.values())}", flush=True)
            if v in (0.93, 0.97):
                pass
        meses = sorted(kref); ksuav = {m: int(round(np.median([kref[x] for x in meses[max(0, i - 2):i + 1]]))) for i, m in enumerate(meses)}; kprev = {m: (kref[meses[i - 1]] if i > 0 else kref[m]) for i, m in enumerate(meses)}
        for nome, kd in (("B2 k suavizado (mediana de 3 meses)", ksuav), ("B3 k do mês anterior", kprev)):
            s, ks = bruto(canal, lambda m, kd=kd: int(kd.get(m, 1))); r = mede(canal, ewma_v2(s, canal), False)
            ganho = r.det >= base.det and r.fp <= base.fp - 2 and r.horas_alarme_fora <= 0.8 * base.horas_alarme_fora
            print(f"   {nome:36s}: {r.det}/7 | início {r.inicio} | FP {r.fp:2d} | fora {r.horas_alarme_fora:5.0f} h | totais {r.horas_alarme:5.0f} h | k por mês {list(ks.values())}{' -> GANHO' if ganho else ''}", flush=True)


if __name__ == "__main__":
    main()
