#!/usr/bin/env python3
"""EXP 44 — OS MODELOS DE NORMALIDADE DA LITERATURA DE TURBINA A GÁS QUE FALTAVAM: K-MEANS, ANÁLISE FATORIAL E AUTOENCODER LSTM, em comparação simétrica com o PCA (custo total sem rótulos, placebo, auditoria do futuro).

PRÉ-REGISTRO (17/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. A busca na literatura (17/10; só resumos) mostrou, para turbina a gás e SCADA, o autoencoder LSTM/TCN, o K-Means (a distância ao centróide, o baseline do estudo de Kirkuk, 83,6%), os modelos de espaço de estados/Kalman (falha de sensor) e as regressões com CUSUM. Os modelos que eu já tinha testado (PCA em variantes, ridge, GBM, OCSVM, GMM, LOF, iForest, kNN, ECOD, autoencoders pontual, condicional e convolucional, previsores AR e GRU) estão nos EXP 04 a 34. Faltam: o K-MEANS, a ANÁLISE FATORIAL (o modelo de medição de um filtro de Kalman sem dinâmica: PCA com variância de ruído própria por sensor) e o AUTOENCODER LSTM (o mais citado em turbina a gás). O teste segue o protocolo simétrico do EXP 34, sem rótulos na calibração.
O QUE SE FAZ (8 eventos; harness fiel: RobustScaler do sklearn, retreino mensal causal nas últimas 20.000 amostras estáveis, EWMA de 1 h só sobre instantes vigiados, episódio acima do limiar por >= 2 h, refratário de 72 h; o limiar é o que dá, em todo o período vigiado, o TOTAL de horas do publicado (1.457 h), SEM janela de trip). Candidatos (hiperparâmetros fixados aqui, de convenções, sem olhar os eventos): KM = K-Means com k = 8 (distância ao centróide mais próximo); FA = análise fatorial com o mesmo número de componentes que o PCA(0,95) escolhe em cada mês e o resíduo por sensor agregado como no PCA (máximo, entre os sensores, do erro quadrado normalizado pelo p99); LSTM = o autoencoder LSTM de janela (W = 15 amostras, estado de 32, gargalo de 4, 20 épocas, perda Huber, corrupção como nos demais) em `nv.modelos.lstm_ae`. Canais t e p; n de treino 20.000 e 19.000. REFERÊNCIA: o PCA(0,95) do EXP 34 (cache `escores_sub34`), no mesmo protocolo.
MEDIDAS: detecção de 8, FP, FP/mês, horas; placebo (eventos deslocados de +45 a +225 dias, no mesmo alarme); AUDITORIA DO FUTURO (embaralhar os dados depois de 01/09/2025: os escores até 31/08/2025 idênticos) para os três.
CRITÉRIOS (fixados antes). GANHO = a média da detecção das duas variantes >= a da referência + 2 e acima do placebo do próprio modelo; PIOR = <= a da referência - 2; EQUIVALENTE entre as duas. Piso de ruído com 8 eventos: +-2. A auditoria tem de passar.
EXPECTATIVAS: a auditoria passa nos três ~95%; o K-Means PIOR que o PCA em t ~65% e EQUIVALENTE ~30%; a análise fatorial EQUIVALENTE ao PCA ~65% (mesmo subespaço, outro ruído); o autoencoder LSTM PIOR ~50%, EQUIVALENTE ~42%; algum GANHO ~8%.
INTERPRETAÇÃO PREVISTA. Se nenhum ganha, fecha-se a lista de modelos de normalidade da literatura de turbina a gás contra o PCA do v2 em condições simétricas; o que a literatura publica como melhor (autoencoder LSTM) não se confirma aqui, o que é coerente com a crítica de avaliação (point-adjust, caso único).
LIMITES. 8 eventos (ruído de +-2); hiperparâmetros por convenção; o autoencoder LSTM com 20 épocas e uma semente por variante; o custo total inclui horas dentro das janelas; nada se adota sem sombra.

Uso:  PYTHONPATH=src .venv/bin/python experiments/44_modelos_da_literatura.py
"""
from __future__ import annotations
import contextlib, importlib.util, io, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis, PCA

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E34 = _carrega("exp34", "34_subespacos_sem_vazamento.py"); E20 = E34.E20
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import _ci, PKG
from nv.modelos.baselines import _Base
from nv.modelos.familias import _Residuo
from nv.modelos.lstm_ae import walk_lstm
from nv.politica import ewma_vigiado, episodios, alarme_de
from nv.walk import walk_forward

ROOT = E34.ROOT; INICIO8 = E34.INICIO8; CACHE = ROOT / "escores_lit44"; NS = (20000, 19000); TOTAL = 1457.0; GL = {}


class KM(_Base):
    def fit(self, X):
        Z = self._ajusta_escala(X); rng = np.random.default_rng(self.semente); sub = Z[rng.choice(len(Z), size=min(len(Z), 20000), replace=False)]; self.km = KMeans(n_clusters=8, n_init=3, random_state=self.semente).fit(sub); return self
    def _score(self, Z):
        return self.km.transform(Z).min(axis=1)


class FA(_Residuo):
    def _ajusta(self, Z):
        k = int(PCA(n_components=0.95, svd_solver="full").fit(Z).n_components_); k = max(1, min(k, Z.shape[1] - 1)); self.fa = FactorAnalysis(n_components=k, random_state=0).fit(Z)
    def _erro2(self, Z):
        rec = self.fa.transform(Z) @ self.fa.components_ + self.fa.mean_; return (Z - rec) ** 2


FABR = {"KM": E20.com_sk(KM), "FA": E20.com_sk(FA)}


def tarefa(a):
    nome, canal, n = a; f = CACHE / f"{canal}_{nome}_{n}.parquet"
    if f.exists():
        return a
    import torch; torch.set_num_threads(2); t0 = time.time()
    if nome == "LSTM":
        s = walk_lstm(GL["g"], GL["tags"][canal], GL["est"], INICIO8, GL["fim"], semente=0, n_max=n)
    else:
        s = walk_forward(FABR[nome], GL["g"], GL["tags"][canal], GL["est"], INICIO8, GL["fim"], n_treino=n, semente=0)
    CACHE.mkdir(parents=True, exist_ok=True); s.to_frame("s").to_parquet(f); print(f"      {canal} {nome} n={n}: {time.time() - t0:.0f} s", flush=True); return a


def main():
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    tags = {c: list(modelos[-1]["normalizacao"][f]["cols"]) for c, f in E20.FAMILIAS.items()}
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual"); vig = vig_c.reindex(g.index).fillna(False); fim = vig_c.index[-1]
    GL.update(g=g, est=est, fim=fim, tags=tags)
    print("AUDITORIA DE VAZAMENTO (escores até 31/08/2025 idênticos com os dados depois de 01/09/2025 embaralhados), canal t", flush=True)
    corte = pd.Timestamp("2025-09-01", tz="UTC"); g2 = g.copy(); pos = g2.index >= corte; g2.loc[pos, :] = g2.loc[pos, :].sample(frac=1.0, random_state=1).to_numpy()
    for nome in ("KM", "FA", "LSTM"):
        if nome == "LSTM":
            a = walk_lstm(g, tags["t"], est, INICIO8, pd.Timestamp("2025-10-31", tz="UTC"), n_max=20000, ep0=4); b = walk_lstm(g2, tags["t"], est, INICIO8, pd.Timestamp("2025-10-31", tz="UTC"), n_max=20000, ep0=4)
        else:
            a = walk_forward(FABR[nome], g, tags["t"], est, INICIO8, fim, n_treino=20000, semente=0); b = walk_forward(FABR[nome], g2, tags["t"], est, INICIO8, fim, n_treino=20000, semente=0)
        sl = (a.index >= INICIO8) & (a.index < corte); d = float(np.nanmax(np.abs(a[sl].to_numpy() - b[sl].to_numpy()))); print(f"   {nome:5s}: diferença máxima = {d:.2e} -> {'PASSA' if d < 1e-9 else 'FALHA'}", flush=True)
    with Pool(4) as p:
        list(p.map(tarefa, [(nm, c, n) for c in ("t", "p") for nm in ("KM", "FA", "LSTM") for n in NS], chunksize=1))
    print("\nESCORES (custo TOTAL de 1.457 h sem rótulos; detecção de 8 | FP | FP/mês; entre parênteses o placebo)", flush=True); res = {}
    for c in ("t", "p"):
        for nome in ("A pca95", "KM", "FA", "LSTM"):
            lin = []
            for n in NS:
                f = (ROOT / "escores_sub34" / f"{c}_A_{n}.parquet") if nome == "A pca95" else (CACHE / f"{c}_{nome}_{n}.parquet"); s = pd.read_parquet(f)["s"]; ini = INICIO8 - pd.Timedelta(days=10); e = ewma_vigiado(s.loc[ini:], vig.loc[ini:]); v = vig.loc[ini:]
                thr = E34.calibra_total(e, v, reg, TOTAL); al = alarme_de(episodios(e, thr), e.index); r = avalia(al, v, reg, inicio_periodo=INICIO8); plac = []
                for dd in (45, 90, 135, 180, 225):
                    sh = [t + pd.Timedelta(days=dd) for t in reg if t + pd.Timedelta(days=dd) <= e.index[-1]]
                    if sh:
                        rp = avalia(al, v, sh, inicio_periodo=INICIO8); plac.append(rp.det / rp.n_eventos)
                res[(c, nome, n)] = (r.det, r.fp, float(np.median(plac)) * 8); lin.append(f"n={n}: {r.det}/8, FP {r.fp} ({r.fp_mes:.2f}/mês; placebo {100 * np.median(plac):.0f}%)")
            print(f"   {c} <- {nome:8s} | " + " | ".join(lin), flush=True)
    print("\nCLASSIFICAÇÃO contra o PCA(0,95) do mesmo canal (média da detecção das 2 variantes)", flush=True)
    for c in ("t", "p"):
        ref = float(np.mean([res[(c, "A pca95", n)][0] for n in NS])); print(f"   [{c}] A pca95: {ref:.1f}/8", flush=True)
        for nome in ("KM", "FA", "LSTM"):
            m = float(np.mean([res[(c, nome, n)][0] for n in NS])); pl = float(np.mean([res[(c, nome, n)][2] for n in NS])); cl = "GANHO" if (m >= ref + 2 and m > pl) else "PIOR" if m <= ref - 2 else "EQUIVALENTE"
            print(f"      {nome:5s}: {m:.1f}/8 (placebo ~{pl:.1f}/8) -> {cl}", flush=True)


if __name__ == "__main__":
    main()
