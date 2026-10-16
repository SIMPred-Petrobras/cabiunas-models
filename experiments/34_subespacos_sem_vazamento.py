#!/usr/bin/env python3
"""EXP 34 — MODELOS DE SUBESPAÇO COM RANK POR PRINCÍPIO E POR REGIME, EM COMPARAÇÃO SIMÉTRICA, SEM RÓTULO NA CALIBRAÇÃO E COM AUDITORIA DE VAZAMENTO.

PRÉ-REGISTRO (16/10/2026), COMMITADO ANTES DE RODAR.

POR QUE (as objeções do usuário: 'o PCA vence de tanto que ajustamos', 'pode haver overfitting', 'evitar vazamento'). Três fontes de viés a favor do PCA nos testes anteriores, que este desenho remove ou mede:
  (1) AJUSTE ASSIMÉTRICO: a estrutura do PCA (escala robusta, piso do p99, máximo por sensor, 0,95) foi desenhada durante a pesquisa nos MESMOS eventos; as alternativas rodaram com padrões. Aqui TODOS os candidatos usam a MESMA escala (RobustScaler do sklearn), o MESMO agregador (máximo, entre os sensores, do erro quadrado normalizado pelo p99 do sensor no treino, piso 0,1 x mediana) e o MESMO retreino mensal causal nas últimas 20.000 amostras estáveis; o que muda é só o modelo de subespaço. Os hiperparâmetros estão FIXADOS AQUI, de convenções da literatura, sem olhar os eventos.
  (2) CALIBRAÇÃO COM RÓTULOS: o 'custo igual' do EXP 33 usou as horas fora das janelas de trip (informação dos rótulos na escolha do limiar). Aqui o limiar principal é o que dá, em todo o período vigiado, o TOTAL de horas de alarme igual ao do publicado (1.457 h), SEM usar nenhuma janela; o formato antigo (407 h fora das janelas) fica como segunda leitura, só para comparar com o EXP 33.
  (3) VAZAMENTO: nenhum mapa de quantis global (cada modelo na sua escala nativa); a auditoria do futuro troca TODOS os dados depois de 01/09/2025 por linhas embaralhadas e exige que os escores até 31/08/2025 fiquem IDÊNTICOS (se o escore de um mês mudasse com dado futuro, haveria vazamento).
CANDIDATOS (resíduo por sensor, fixados): A = PCA com 95% da variância (a referência do v2); B = PCA com rank por MÁXIMA VEROSSIMILHANÇA de Minka (`n_components='mle'`, sem parâmetro); C = PCA com rank por ANÁLISE PARALELA (k = número de autovalores acima do percentil 95 dos autovalores de 20 permutações independentes de colunas); D = MISTURA DE PCA POR REGIME (três regimes por tercis do T5_AVG_A no treino, atribuição dura; em cada regime PCA com 95% e p99 por sensor do regime). Canais t (14 temperaturas) e p (12 pressões), n de treino 20.000, 19.000 e 21.000, 8 eventos (desde 01/02/2025).
POLÍTICA FIXA: EWMA de 1 h só sobre instantes vigiados, episódio acima do limiar por >= 2 h, refratário de 72 h (a de `nv.politica`).
PLACEBO: com o alarme fixado pelo custo TOTAL (sem rótulos), os eventos são deslocados de +45, +90, +135, +180 e +225 dias (os que passam do fim da série saem) e se mede a fração detectada: a detecção AO ACASO do modelo àquele custo.
CRITÉRIOS (fixados antes). Referência = A do mesmo canal e variante. GANHO = mediana da detecção (3 variantes de n) >= a da referência + 2 (o ruído com 8 eventos é de +-2) E acima do placebo do próprio modelo; EQUIVALENTE = a +-1; PIOR = <= referência - 2. A AUDITORIA DE VAZAMENTO tem de passar (diferença máxima 0) para cada candidato antes de qualquer leitura.
EXPECTATIVAS: a auditoria passa nos quatro ~90%; B e C EQUIVALENTES a A em t ~60% e em p ~55% (o rank por princípio resolve a fragilidade do 0,95, não o ponto de operação); a mistura por regime GANHO em t ~8%, EQUIVALENTE ~40%, PIOR ~50% (menos dado por regime); algum GANHO em algum canal ~12%; a detecção do placebo à altura da referência não passa de 3/8 ~75%.
INTERPRETAÇÃO PREVISTA. Se nenhum ganha, com a comparação simétrica, sem rótulos na calibração e sem vazamento, a vantagem do PCA não é artefato de ajuste assimétrico: o resultado é mais forte que o do EXP 33. Se B ou C empatam, a regra de rank por princípio pode substituir o 0,95 (sem o pico) e fica como hipótese. Se a mistura ganha, vale a leitura por regime e o plug-in.
LIMITES. 8 eventos; quatro candidatos; hiperparâmetros fixados mas escolhidos por mim (convenções, não busca); o placebo por deslocamento é grosseiro; nada se adota sem sombra.

Uso:  PYTHONPATH=src .venv/bin/python experiments/34_subespacos_sem_vazamento.py
"""
from __future__ import annotations
import contextlib, importlib.util, io, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E20 = _carrega("exp20", "20_escores_por_familia_fiel.py"); E6 = E20.E6
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import _ci, PKG
from nv.modelos.familias import _Residuo
from nv.politica import ewma_vigiado, episodios, alarme_de, calibra_limiar
from nv.walk import walk_forward

ROOT = E20.ROOT; INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); CACHE = ROOT / "escores_sub34"; NS = (20000, 19000, 21000); TOTAL_PUB = 1457.0; FORA_PUB = 407.0


class PCA95(_Residuo):
    def _ajusta(self, Z):
        self.pca = PCA(n_components=0.95, svd_solver="full").fit(Z)
    def _erro2(self, Z):
        return (Z - self.pca.inverse_transform(self.pca.transform(Z))) ** 2


class PCAmle(PCA95):
    def _ajusta(self, Z):
        self.pca = PCA(n_components="mle", svd_solver="full").fit(Z)


class PCApa(PCA95):
    def _ajusta(self, Z):
        rng = np.random.default_rng(0); ev = np.sort(np.linalg.eigvalsh(np.cov(Z, rowvar=False)))[::-1]; null = []
        for _ in range(20):
            P = np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])]); null.append(np.sort(np.linalg.eigvalsh(np.cov(P, rowvar=False)))[::-1])
        lim = np.percentile(np.array(null), 95, axis=0); k = 0
        while k < len(ev) and ev[k] > lim[k]:
            k += 1
        self.pca = PCA(n_components=max(k, 1), svd_solver="full").fit(Z)


class MixPCA:
    """Mistura de PCA por regime: a última coluna de X é o T5_AVG_A (variável de regime), as outras são os sensores da família."""
    def __init__(self, semente=0):
        self.semente = semente

    def _regime(self, r):
        return np.digitize(r, self.cortes)

    def fit(self, X):
        F, r = X[:, :-1], X[:, -1]; self.esc = RobustScaler().fit(F); Z = self.esc.transform(F); self.cortes = np.quantile(r, [1 / 3, 2 / 3]); g = self._regime(r); self.m = {}
        for k in range(3):
            i = g == k; Zk = Z[i] if i.sum() >= 300 else Z; pca = PCA(n_components=0.95, svd_solver="full").fit(Zk); e = (Zk - pca.inverse_transform(pca.transform(Zk))) ** 2
            p99 = np.quantile(e, 0.99, axis=0); self.m[k] = (pca, np.maximum(p99, 0.1 * np.median(p99)))
        bruto = self._bruto(Z, g); self.p99s = float(np.quantile(bruto, 0.99)); return self

    def _bruto(self, Z, g):
        out = np.empty(len(Z))
        for k in range(3):
            i = g == k
            if i.any():
                pca, sens = self.m[k]; out[i] = (((Z[i] - pca.inverse_transform(pca.transform(Z[i]))) ** 2) / sens).max(axis=1)
        return out

    def score(self, X):
        s = np.full(len(X), np.nan); ok = np.isfinite(X).all(axis=1)
        if ok.any():
            Z = self.esc.transform(X[ok][:, :-1]); s[ok] = self._bruto(Z, self._regime(X[ok][:, -1])) / max(self.p99s, 1e-12)
        return s


CANDS = {"A pca95": E20.com_sk(PCA95), "B mle": E20.com_sk(PCAmle), "C análise paralela": E20.com_sk(PCApa), "D mistura por regime": MixPCA}


def calibra_total(e, vig, reg, alvo, n_iter=28):
    v = e.dropna(); lo, hi = float(np.nanquantile(v, 0.50)), float(v.max()); f = lambda t: avalia(alarme_de(episodios(e, t), e.index), vig, reg, inicio_periodo=INICIO8).horas_alarme
    for _ in range(n_iter):
        mid = float(np.sqrt(max(lo, 1e-12) * hi))
        if f(mid) > alvo:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(max(lo, 1e-12) * hi))


def main():
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    tags = {c: list(modelos[-1]["normalizacao"][f]["cols"]) for c, f in E20.FAMILIAS.items()}
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual"); vig = vig_c.reindex(g.index).fillna(False); fim = vig_c.index[-1]
    print("AUDITORIA DE VAZAMENTO (os escores até 31/08/2025 não podem mudar quando os dados depois de 01/09/2025 são embaralhados)", flush=True)
    corte = pd.Timestamp("2025-09-01", tz="UTC"); g2 = g.copy(); rng = np.random.default_rng(1); pos = g2.index >= corte; g2.loc[pos, :] = g2.loc[pos, :].sample(frac=1.0, random_state=1).to_numpy(); aud = {}
    for nome, cls in CANDS.items():
        tg = tags["t"] + (["T5_AVG_A"] if nome.startswith("D") else [])
        a = walk_forward(cls, g, tg, est, INICIO8, fim, n_treino=20000, semente=0); est2 = est.copy(); b = walk_forward(cls, g2, tg, est2, INICIO8, fim, n_treino=20000, semente=0)
        sl = (a.index >= INICIO8) & (a.index < corte); d = float(np.nanmax(np.abs(a[sl].to_numpy() - b[sl].to_numpy()))); aud[nome] = d; print(f"   {nome:22s}: diferença máxima antes de 01/09/2025 = {d:.2e} -> {'PASSA' if d == 0 or d < 1e-9 else 'FALHA'}", flush=True)
    print("\nESCORES (n = 20.000, 19.000, 21.000); política fixa; DOIS custos", flush=True)
    res = {}
    for c in ("t", "p"):
        for nome, cls in CANDS.items():
            tg = tags[c] + (["T5_AVG_A"] if nome.startswith("D") else []); lin = []
            for n in NS:
                f = CACHE / f"{c}_{nome[0]}_{n}.parquet"
                if f.exists():
                    s = pd.read_parquet(f)["s"]
                else:
                    s = walk_forward(cls, g, tg, est, INICIO8, fim, n_treino=n, semente=0); CACHE.mkdir(parents=True, exist_ok=True); s.to_frame("s").to_parquet(f)
                ini = INICIO8 - pd.Timedelta(days=10); e = ewma_vigiado(s.loc[ini:], vig.loc[ini:]); v = vig.loc[ini:]
                thr = calibra_total(e, v, reg, TOTAL_PUB); al = alarme_de(episodios(e, thr), e.index); r1 = avalia(al, v, reg, inicio_periodo=INICIO8)
                plac = []
                for d in (45, 90, 135, 180, 225):
                    sh = [t + pd.Timedelta(days=d) for t in reg if t + pd.Timedelta(days=d) <= e.index[-1]]
                    if sh:
                        rp = avalia(al, v, sh, inicio_periodo=INICIO8); plac.append(rp.det / rp.n_eventos)
                thr2 = calibra_limiar(e, v, reg, FORA_PUB, inicio=INICIO8); r2 = avalia(alarme_de(episodios(e, thr2), e.index), v, reg, inicio_periodo=INICIO8)
                res[(c, nome, n)] = (r1.det, r1.fp, r1.horas_alarme, float(np.median(plac)) if plac else np.nan, r2.det, r2.fp)
                lin.append(f"n={n}: sem rótulo {r1.det}/8, FP {r1.fp} (placebo {100 * (np.median(plac) if plac else np.nan):.0f}%) | 407 h fora {r2.det}/8, FP {r2.fp}")
            print(f"   {c} <- {nome:22s} | " + " | ".join(lin), flush=True)
    print("\nCLASSIFICAÇÃO (mediana da detecção das 3 variantes de n, custo TOTAL sem rótulos) contra A do mesmo canal", flush=True)
    for c in ("t", "p"):
        ref = float(np.median([res[(c, "A pca95", n)][0] for n in NS])); print(f"   [{c}] A pca95: detecção mediana {ref:.0f}/8", flush=True)
        for nome in list(CANDS)[1:]:
            m = float(np.median([res[(c, nome, n)][0] for n in NS])); pl = float(np.nanmedian([res[(c, nome, n)][3] for n in NS])) * 8
            cl = "GANHO" if (m >= ref + 2 and m > pl) else "PIOR" if m <= ref - 2 else "EQUIVALENTE"
            print(f"      {nome:22s}: {m:.0f}/8 (placebo do modelo ~{pl:.1f}/8) -> {cl}", flush=True)


if __name__ == "__main__":
    main()
