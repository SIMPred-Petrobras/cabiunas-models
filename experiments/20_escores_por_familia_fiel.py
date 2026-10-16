#!/usr/bin/env python3
"""EXP 20 — OS ESCORES POR FAMÍLIA (OCSVM, GMM, LOF, iForest, kNN, ECOD, ridge, GBM, PCA) REFEITOS NO HARNESS FIEL.

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. O EXP 14 comparou escores no harness infiel (escala MAD com piso, k fixo em 1 e 4), e o EXP 19 mostrou que o ajuste fiel do bundle (RobustScaler do sklearn, mediana/IQR, e PCA(0,95) com k variável por mês) reproduz EXATAMENTE os 14 bundles e leva o plug-in perto do controle (t: 6/7, 11 FP, 467 h; p: 6/7, 9 FP, 390 h; controle 6/7, 9 FP, 407 h). A pergunta original (outros escores no lugar do t e do p do v2, entre eles o OCSVM) merece resposta sobre a referência CERTA.
O QUE MUDA em relação ao EXP 14: TODOS os escores usam a escala do bundle (RobustScaler do sklearn); o escore 'fiel' é o PCA(0,95) com k variável (= o canal do v2, o `ajusta_familia` do bundle) e é a REFERÊNCIA; o pca1 e o pca3 (k fixo) ficam para medir o efeito do k; ridge, gbm (resíduo por sensor, mesma agregação max/p99) e ocsvm, gmm, lof, iforest, knn, ecod (densidade conjunta por família) como antes. Retreino mensal causal nas últimas 20.000 amostras estáveis (o protocolo do bundle), EWMA 1 h vigiado, mapa de quantis e `ci.prever` como caixa-preta, um canal por vez. Piso de ruído: treino de 19.000 e 21.000 (todos) e sementes 1 e 2 (os estocásticos).
USOS: (S) substituição do t (14 temperaturas) ou do p (12 pressões); (Q) quinto canal só com os escores de temperatura (voto V1 (3, 2), base e k do t, como nos EXP 09, 11, 15).
VALIDADE (escrita antes): o escore 'fiel' via esta rota (`walk_forward` com a escala sklearn) deve dar em t o do EXP 19 (6/7, 11 FP, 467 h fora no n de 20.000) e em p (6/7, 9 FP, 390 h) com tolerância de +-1 FP e +-30 h; se não, a rota difere do S3 do EXP 19 e nada abaixo se lê.
CRITÉRIOS (fixados antes), sobre as medianas das variantes de cada célula e contra a REFERÊNCIA 'fiel' do mesmo canal e do mesmo harness: GANHO = detecção >= a da referência E FP <= referência - 2 E horas fora <= 0,8 x a referência; EQUIVALENTE = detecção a +-1, FP a +-2 e horas fora a +-30% da referência (o piso de ruído do EXP 19: S3 em t varia 5-6/7, FP 11-15, com n); PIOR = o resto. Contra o controle: DIGNO = >= 6/7, FP <= 9, <= 407 h; PROMISSOR = >= 5/7, FP <= 12, <= 800 h. Em (Q): MELHOR = >= 7/7, FP <= 10 e <= 450 h; IGUAL = >= 6/7, FP 11-14, <= 700 h (referência: AE pontual 7/7, 12 FP, 563 h).
EXPECTATIVAS: validade passa ~70%. pca3 EQUIVALENTE à fiel (o k do bundle em t é 3 na maioria dos meses) ~70%; pca1 PIOR que a fiel em t ~55%; ridge e gbm PIORES que a fiel em t ~60% (explicam o desvio coordenado); as seis densidades PIORES que a fiel em t e em p ~75% (a densidade conjunta perde a agregação por sensor); o OCSVM é a melhor das densidades ~25%; algum GANHO sobre a fiel em qualquer canal ~10%; algum DIGNO ~10%; algum MELHOR em (Q) ~8%.
INTERPRETAÇÃO PREVISTA. Se a fiel não for superada por nenhum, a estrutura do v2 (PCA(0,95) por família, máximo por sensor) é o melhor do lote e o caminho de melhoria não é trocar o escore. Se algum GANHO aparecer, segue o episódio por episódio (13b), a varredura de n e o placebo antes de qualquer conclusão.
LIMITES. 7 eventos; o harness fiel ainda difere do v2 no EWMA (aqui só sobre instantes vigiados) e no mapa de quantis (global); o piso de ruído por n é grande (EXP 19); nada se adota sem sombra.

RESULTADO (15/10/2026) -- NENHUM ESCORE SUPERA O PCA(0,95) DO v2 NO HARNESS FIEL; a conclusão qualitativa do EXP 14 se mantém, o mecanismo muda.
  VALIDADE: PASSA. O escore 'fiel' em n = 20.000 dá em t 6/7, FP 11, 467 h fora e em p 6/7, FP 9, 390 h, idênticos ao S3 do EXP 19. (Mediana das 3 variantes de n: t 6/7 (5-6), FP 13 (11-15), 593 h; p 6/7 (6-6), FP 9 (8-9), 390 h: o piso de ruído de t é de +-1 detecção e +-2 FP.)
  SUBSTITUIÇÃO (S), mediana das variantes (detecção | FP | h fora): canal t: fiel 6/7 | 13 | 593 (REFERÊNCIA) | pca1 4/7 | 16 | 1.253 (PIOR) | pca3 5/7 | 14 | 583 (EQUIVALENTE) | ridge 3/7 | 16 | 928 | gbm 4/7 | 16 | 1.173 | ocsvm 4/7 | 15 | 928 | iforest 5/7 | 18 | 1.039 | knn 4/7 | 18 | 590 | lof 4/7 | 18 | 789 | gmm 3/7 | 14 | 896 | ecod 6/7 | 18 | 936 (todos PIOR). Canal p: fiel 6/7 | 9 | 390 (REFERÊNCIA e DIGNO) | pca1 6/7 | 13 | 1.029 | pca3 6/7 | 12 | 814 | ridge 6/7 | 12 | 1.044 | gbm 6/7 | 10 | 682 (PROMISSOR) | ocsvm 6/7 | 11 | 1.200 | iforest 4/7 | 16 | 1.357 | knn 6/7 | 12 | 1.254 | lof 6/7 | 13 | 1.146 | gmm 6/7 | 12 | 525 (PROMISSOR) | ecod 4/7 | 12 | 1.130 (todos PIOR que a fiel). Nenhum GANHO sobre a referência em nenhum canal.
  QUINTO CANAL (Q), referência AE pontual (7/7, 12 FP, 563 h): fiel 7/7 | 13 | 615 (IGUAL) | pca1 6/7 | 15 | 862 (PIOR) | pca3 7/7 | 13 | 462 (IGUAL) | ridge 7/7 | 11 | 496 (IGUAL) | gbm 7/7 | 12 | 773 (PIOR) | ocsvm 6/7 | 12 | 870 (PIOR) | iforest 5/7 | 16 | 748 | knn 7/7 | 12 | 488 (IGUAL) | lof 7/7 | 13 | 709 (PIOR) | gmm 7/7 | 11 (9-12) | 458 (IGUAL, por 8 h acima do limite de MELHOR) | ecod 5/7 | 18 | 1.664 (PIOR). Nenhum MELHOR.
LEITURA. (1) A estrutura do v2 (PCA(0,95) com k variável por mês, máscara estável, RobustScaler do sklearn, máximo por sensor normalizado por p99) é o MELHOR do lote nos dois canais: em p é a única DIGNO (6/7, 9 FP, 390 h, estável em n); em t os outros ficam iguais (pca3) ou piores. (2) O OCSVM, o GMM, o LOF, o iForest, o kNN e o ECOD (densidade conjunta por família) PERDEM da referência em especificidade (FP 11 a 18 e 525 a 1.357 h fora, contra 9 e 390 em p): o kernel não é o melhor de nenhuma família, e a densidade conjunta não substitui o resíduo máximo por sensor. (3) O k fixo é pior que o k variável: pca1 PIOR nos dois canais; pca3 empata em t e piora em p (o k de p do bundle vai de 1 a 8). A hipótese da 'rigidez de k = 1' do EXP 14 está definitivamente retirada: o que o canal faz é escolher k por variância retida a cada mês. (4) Como quinto canal (Q) vários escores levam o detector a 7/7 (fiel, pca3, ridge, knn, gmm) com FP 11 a 13 e 458 a 615 h fora, o mesmo padrão do AE pontual (EXP 09/11): recuperam o evento de 04/11/2025 por continuidade de voto na borda (EXP 10b) a +2 a +4 FP; não passa o critério MELHOR e não atende à meta do usuário (FP no nível do controle, com 1 a 2 detecções a menos).
PLACAR: ACERTEI: validade (~70%); pca3 EQUIVALENTE à fiel em t (~70%); pca1 PIOR em t (~55%); ridge e gbm PIORES em t (~60%); as seis densidades PIORES que a fiel em t e em p (~75%: 12 de 12 células); nenhum GANHO (~90%); nenhum MELHOR em Q (~92%; o gmm ficou a 8 h). ERREI: o OCSVM é a melhor das densidades (~25%: não é; em t o ecod tem 6/7 mas 18 FP, em p o gmm tem a melhor combinação, 12 FP e 525 h). DIGNO (~10%): só a própria referência em p.

Uso:  PYTHONPATH=src .venv/bin/python experiments/20_escores_por_familia_fiel.py
"""
from __future__ import annotations
import contextlib, copy, importlib.util, io, time, warnings
from multiprocessing import Pool
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
from nv.modelos.familias import FAMILIA_FABRICAS, PCAk, _Residuo
from nv.plugin import decide_v2, mapeia_quantis, substitui
from nv.politica import ewma_vigiado
from nv.walk import walk_forward

INICIO = E4.INICIO; ROOT = Path(__file__).resolve().parents[1] / "data_local"; CACHE = ROOT / "escores_fiel"; GL = {}
FAMILIAS = {"t": "temperatura", "p": "pressao"}


def com_sk(cls):
    class C(cls):
        def _ajusta_escala(self, X):
            self.escala = RobustScaler().fit(X); return self.escala.transform(X)
    C.__name__ = cls.__name__; return C


class PCAfiel(_Residuo):
    """O canal do v2: PCA(0,95), k variável por mês."""
    def _ajusta(self, Z):
        self.pca = PCA(n_components=0.95, svd_solver="full").fit(Z)

    def _erro2(self, Z):
        return (Z - self.pca.inverse_transform(self.pca.transform(Z))) ** 2


FABRICAS = {"fiel": com_sk(PCAfiel), **{n: com_sk(FAMILIA_FABRICAS[n]) for n in ("pca1", "pca3", "ridge", "gbm", "ocsvm", "iforest", "knn", "lof", "gmm", "ecod")}}
ESTOC = {"gbm", "ocsvm", "iforest", "knn", "lof", "gmm"}


def tarefa(a):
    nome, canal, n, sem = a; f = CACHE / f"{canal}_{nome}_{n}_{sem}.parquet"
    if f.exists():
        return a, pd.read_parquet(f)["s"]
    t0 = time.time(); s = walk_forward(FABRICAS[nome], GL["g"], GL["tags"][canal], GL["est"], pd.Timestamp("2025-03-01", tz="UTC"), GL["fim"], n_treino=n, semente=sem)
    CACHE.mkdir(parents=True, exist_ok=True); s.to_frame("s").to_parquet(f); print(f"      {canal} {nome} n={n} s={sem}: {time.time() - t0:.0f} s", flush=True)
    return a, s


def resume(ms):
    d = [m["det"] for m in ms]; f = [m["fp"] for m in ms]; o = [m["fora"] for m in ms]
    return dict(det=float(np.median(d)), dmin=min(d), dmax=max(d), fp=float(np.median(f)), fmin=min(f), fmax=max(f), fora=float(np.median(o)), horas=float(np.median([m["horas"] for m in ms])), lead=float(np.nanmedian([m["lead"] for m in ms])))


def main():
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tags = {c: list(modelos[-1]["normalizacao"][f]["cols"]) for c, f in FAMILIAS.items()}
    al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]; reg = [t for t in carrega_registro() if t >= INICIO]; ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); vigS = proc["mask"]; vig_g = vig_c.reindex(g.index).fillna(False)
    GL.update(g=g, est=est, fim=fim, tags=tags)
    pedidos = []
    for canal in FAMILIAS:
        for nome in FABRICAS:
            pedidos += [(nome, canal, 20000, 0), (nome, canal, 19000, 0), (nome, canal, 21000, 0)] + ([(nome, canal, 20000, s) for s in (1, 2)] if nome in ESTOC else [])
    print(f"CONTROLE: {E6.linha(ctrl)}\n{len(pedidos)} escores (harness fiel)", flush=True)
    with Pool(4) as p:
        R = dict(p.map(tarefa, pedidos, chunksize=1))

    def mede_s(s, canal):
        e = ewma_vigiado(s, vig_g, "1h").reindex(proc.index); r = E6.mede(decide_v2(substitui(proc, canal, mapeia_quantis(e, proc[canal], vigS)), det), reg); return dict(det=r.det, fp=r.fp, fora=r.horas_alarme_fora, horas=r.horas_alarme, lead=r.lead_medio_h)

    def mede_q(s):
        e = ewma_vigiado(s, vig_g, "1h").reindex(proc.index); d2 = copy.deepcopy(det); d2["sinais"] = list(det["sinais"]) + ["ae"]
        for k in ("base", "k_nivel_a", "k_nivel_b"):
            d2[k]["ae"] = det[k]["t"]
        d2["voto_nivel_a"], d2["voto_nivel_b"] = 3, 2; p2 = proc.copy(); p2["ae"] = mapeia_quantis(e, proc["t"], vigS); r = E6.mede(decide_v2(p2, d2), reg); return dict(det=r.det, fp=r.fp, fora=r.horas_alarme_fora, horas=r.horas_alarme, lead=r.lead_medio_h)
    refs = {}
    print("\nSUBSTITUIÇÃO (S): mediana das variantes | faixa de detecção | FP | h fora | h totais | lead", flush=True)
    for canal in FAMILIAS:
        for nome in FABRICAS:
            ks = [k for k in R if k[0] == nome and k[1] == canal]; r = resume([mede_s(R[k], canal) for k in ks])
            if nome == "fiel":
                refs[canal] = r; pr = mede_s(R[("fiel", canal, 20000, 0)], canal); print(f"   [REFERÊNCIA {canal} fiel, n=20.000]: {pr['det']}/7 | FP {pr['fp']} | fora {pr['fora']:.0f} h (EXP 19: {'6/7, 11, 467' if canal == 't' else '6/7, 9, 390'})", flush=True)
            ref = refs[canal]
            ganho = r["det"] >= ref["det"] and r["fp"] <= ref["fp"] - 2 and r["fora"] <= 0.8 * ref["fora"]
            equiv = abs(r["det"] - ref["det"]) <= 1 and abs(r["fp"] - ref["fp"]) <= 2 and abs(r["fora"] - ref["fora"]) <= 0.3 * ref["fora"]
            cl = "REFERÊNCIA" if nome == "fiel" else ("GANHO" if ganho else "EQUIVALENTE" if equiv else "PIOR")
            ver = "DIGNO" if r["det"] >= 6 and r["fp"] <= 9 and r["fora"] <= 407 else "PROMISSOR" if r["det"] >= 5 and r["fp"] <= 12 and r["fora"] <= 800 else "-"
            print(f"   S {canal} <- {nome:7s} ({len(ks)} var.): detecção {r['det']:.0f}/7 ({r['dmin']}-{r['dmax']}) | FP {r['fp']:.0f} ({r['fmin']}-{r['fmax']}) | fora {r['fora']:.0f} h | totais {r['horas']:.0f} h | lead {r['lead']:.1f} h -> {cl} / {ver}", flush=True)
    print("\nQUINTO CANAL (Q), voto V1, famílias de temperatura (referência AE pontual: 7/7, 12 FP, 563 h)", flush=True)
    for nome in FABRICAS:
        ks = [k for k in R if k[0] == nome and k[1] == "t"]; r = resume([mede_q(R[k]) for k in ks])
        ver = "MELHOR" if r["det"] >= 7 and r["fp"] <= 10 and r["fora"] <= 450 else "IGUAL" if r["det"] >= 6 and 11 <= r["fp"] <= 14 and r["fora"] <= 700 else "PIOR"
        print(f"   Q t5 <- {nome:7s}: detecção {r['det']:.0f}/7 ({r['dmin']}-{r['dmax']}) | FP {r['fp']:.0f} ({r['fmin']}-{r['fmax']}) | fora {r['fora']:.0f} h | totais {r['horas']:.0f} h -> {ver}", flush=True)


if __name__ == "__main__":
    main()
