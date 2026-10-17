#!/usr/bin/env python3
"""EXP 46 — OPTUNA SOBRE MODELOS DE NORMALIDADE, COM VALIDAÇÃO LEAVE-ONE-EVENT-OUT E PLACEBO: o teto de um ajuste fino honesto contra o PCA(0,95).

PRÉ-REGISTRO (18/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. O EXP 44 fechou a lista de modelos da literatura com os hiperparâmetros por convenção. Falta saber se o AJUSTE FINO (família, hiperparâmetros, EWMA, persistência) muda o veredito. Com 8 eventos e piso de ruído de +-2, um otimizador que maximiza a detecção acha "ganhos" de ruído; por isso o desenho abaixo mede o ganho FORA da amostra e contra um nulo de placebo, e o resultado só vale se passar nos dois.
Além disso, o EXP 44 apontou que o limiar calibrado no PCA isolado era o PISO da busca (a mediana): aqui o piso é o quantil 5% e uma configuração só é elegível se o orçamento de horas for ALCANÇÁVEL (alarme >= 95% de 1.457 h no piso).

O QUE SE FAZ.
  Etapa 1 (pool; rótulo nenhum). O Optuna (RandomSampler, semente 20261018) sorteia 299 configurações, e a configuração 0 é a referência fixa PCA(0,95), n=20.000. Famílias (uniforme): pca (variância 0,80-0,995), km (k 3-24), fa (variância 0,80-0,995), gmm (2-12 componentes), ocsvm (nu 0,005-0,1 em log; gamma = f/(d*var), f 0,3-3 em log), iforest (100/200/400 árvores; 256/512/1024 amostras), knn (k 3-40), lstm (épocas 10, 20 ou 30). Em todas: n de treino em {15.000, 20.000, 25.000}; harness fiel (RobustScaler sklearn, retreino mensal causal nas últimas n amostras estáveis, a partir de 01/02/2025). Cada configuração é pontuada UMA vez por canal (t e p, isolados).
  Etapa 2 (seleção; usa eventos). Para cada canal, condição (real e 5 placebos: eventos deslocados de +45, +90, +135, +180 e +225 d, descartando os que passam do fim) e dobra (leave-one-event-out: 8 dobras; mais uma dobra "todos" para o melhor dentro da amostra), um estudo do Optuna (TPE, semente fixa, 30 tentativas iniciais, 200 tentativas) escolhe: família, configuração do pool, meia-vida do EWMA (0,25-6 h, log) e persistência mínima do episódio (0,5-6 h, log). O limiar NÃO é livre: sai por bisseção para o orçamento de 1.457 h de alarme desde 01/02/2025 (sem rótulos, entre o quantil 5% do escore e o máximo; inviável se o alarme no piso < 95% do orçamento: tentativa penalizada em -1). Refratário de 72 h.
  Objetivo da tentativa: eventos detectados entre os de TREINO (todos menos o da dobra) menos 0,002 vezes o número de episódios (desempate sem rótulo). A avaliação é o evento DEIXADO DE FORA (detectado ou não) da configuração que o estudo escolheu.
MEDIDAS (por canal): (a) LOEO real = soma dos eventos deixados de fora detectados / 8; (b) o mesmo nas 5 condições de placebo (fração detectada) = o nulo; (c) melhor DENTRO da amostra (dobra "todos") real e nos placebos = a inflação do melhor de N; (d) a referência PCA(0,95) n=20.000, EWMA 1 h, persistência 2 h, SEM seleção (detecção e FP no mesmo orçamento); (e) a curva da referência para orçamentos de 300, 600, 1.000 e 1.457 h (detecção, FP, limiar); (f) família escolhida em cada dobra; (g) fração de configurações inviáveis (EWMA 1 h, persistência 2 h).
CRITÉRIOS (fixados antes). GANHO (por canal) = LOEO real >= (detecção da referência) + 2/8 E LOEO real > o MAIOR LOEO dos 5 placebos. PIOR = LOEO real <= referência - 2/8. EQUIVALENTE no resto. Piso de ruído com 8 eventos: +-2/8. Auditoria do futuro e vazamento: o pool é sem rótulo e o escore de cada mês só usa o passado (já auditado nos EXP 34 e 44 para estas classes); a seleção só vê os eventos de treino. Verificação adicional: a configuração 0 do pool tem de reproduzir o PCA de referência do EXP 34 (escores_sub34, n=20.000) com diferença máxima <= 1e-6.
EXPECTATIVAS: GANHO em t ~7% e em p ~7% (algum ~12%); LOEO real <= melhor dentro da amostra por >= 1,5/8 em ambos os canais ~80%; o LOEO mediano dos placebos >= 0,25 ~60% (o alarme cobre muito tempo); a família mais escolhida nas dobras reais é o PCA ~35%; configurações inviáveis >= 20% do pool ~40%; a referência a 1.457 h repete o 7/8 em t com n=20.000 ~55%.
INTERPRETAÇÃO PREVISTA. Se nenhum canal passa, o teto do ajuste fino honesto é o PCA e a busca por modelo fecha com mais força; se algum passa, vira HIPÓTESE (8 eventos; nada se adota sem sombra) a testar com outras sementes e com a estrutura de voto.
LIMITES. 8 eventos (ruído de +-2); o nulo de placebo tem só 5 deslocamentos (p mínimo ~1/6 sob troca); o pool é 300 configurações (não o espaço inteiro) e o LSTM só varia as épocas; canais isolados, sem voto nem CUSUM; o TPE escolhe dentro do pool pré-pontuado.

Uso:  PYTHONPATH=src .venv/bin/python experiments/46_optuna_modelos_loeo_placebo.py
      no worker: PYTHONPATH=src python scripts/worker_exp46.py --enqueue --queue default
      variáveis: EXP46_PROCS (processos), EXP46_TESTE=1 (encanamento: pool de 3 configurações, 70 dias, 5 tentativas)
"""
from __future__ import annotations
import contextlib, importlib.util, io, os, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

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
from nv.modelos.familias import _Residuo, _Sub, deslocado
from nv.modelos.lstm_ae import walk_lstm
from nv.politica import ewma_vigiado, episodios, alarme_de
from nv.walk import walk_forward

ROOT = E34.ROOT; INICIO8 = E34.INICIO8; CACHE = ROOT / "escores_opt46"
TOTAL = 150.0 if os.environ.get("EXP46_TESTE") else 1457.0; SEMENTE_POOL = 20261018; SHIFTS = (45, 90, 135, 180, 225); FAMS = ("pca", "km", "fa", "gmm", "ocsvm", "iforest", "knn", "lstm")
TESTE = bool(os.environ.get("EXP46_TESTE")); M = 3 if TESTE else 300; N_TRIAL = 5 if TESTE else 200; GL = {}


def _pca(var):
    class P(_Residuo):
        def _ajusta(self, Z):
            self.pca = PCA(n_components=var, svd_solver="full").fit(Z)
        def _erro2(self, Z):
            return (Z - self.pca.inverse_transform(self.pca.transform(Z))) ** 2
    return P


def _fa(var):
    class F(_Residuo):
        def _ajusta(self, Z):
            k = int(PCA(n_components=var, svd_solver="full").fit(Z).n_components_); k = max(1, min(k, Z.shape[1] - 1)); self.fa = FactorAnalysis(n_components=k, random_state=0).fit(Z)
        def _erro2(self, Z):
            return (Z - (self.fa.transform(Z) @ self.fa.components_ + self.fa.mean_)) ** 2
    return F


def _km(k):
    class K(_Base):
        def fit(self, X):
            Z = self._ajusta_escala(X); rng = np.random.default_rng(self.semente); sub = Z[rng.choice(len(Z), size=min(len(Z), 20000), replace=False)]; self.km = KMeans(n_clusters=k, n_init=3, random_state=self.semente).fit(sub); return self
        def _score(self, Z):
            return self.km.transform(Z).min(axis=1)
    return K


def _gmm(nc):
    class G(_Sub):
        def fit(self, X):
            Z = self._ajusta_escala(X); self.m = GaussianMixture(n_components=nc, covariance_type="full", reg_covar=1e-3, random_state=self.semente).fit(self._sub(Z)); return self
        def _score(self, Z):
            return -self.m.score_samples(Z)
    return deslocado(G)


def _ocsvm(nu, f):
    class O(_Sub):
        def fit(self, X):
            Z = self._ajusta_escala(X); S = self._sub(Z); self.m = OneClassSVM(kernel="rbf", gamma=f / (S.shape[1] * float(S.var())), nu=nu).fit(S); return self
        def _score(self, Z):
            return -self.m.decision_function(Z)
    return deslocado(O)


def _iforest(ne, ms):
    class I(_Base):
        def fit(self, X):
            Z = self._ajusta_escala(X); self.m = IsolationForest(n_estimators=ne, max_samples=ms, random_state=self.semente, n_jobs=1).fit(Z); return self
        def _score(self, Z):
            return -self.m.score_samples(Z)
    return I


def _knn(k):
    class N(_Sub):
        def fit(self, X):
            Z = self._ajusta_escala(X); self.nn = NearestNeighbors(n_neighbors=k, n_jobs=1).fit(self._sub(Z)); return self
        def _score(self, Z):
            d, _ = self.nn.kneighbors(Z); return d.mean(axis=1)
    return N


def fabrica(p):
    f = p["fam"]
    cls = {"pca": lambda: _pca(p["var"]), "fa": lambda: _fa(p["var"]), "km": lambda: _km(p["k"]), "gmm": lambda: _gmm(p["nc"]), "ocsvm": lambda: _ocsvm(p["nu"], p["f"]), "iforest": lambda: _iforest(p["ne"], p["ms"]), "knn": lambda: _knn(p["k"])}[f]()
    return E20.com_sk(cls)


def sorteia_pool():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=SEMENTE_POOL)); pool = [{"fam": "pca", "n": 20000, "var": 0.95}]
    for _ in range(M - 1):
        tr = st.ask(); fam = tr.suggest_categorical("fam", list(FAMS)); p = {"fam": fam, "n": tr.suggest_categorical("n", [15000, 20000, 25000])}
        if fam in ("pca", "fa"):
            p["var"] = tr.suggest_float("var", 0.80, 0.995)
        elif fam == "km":
            p["k"] = tr.suggest_int("k", 3, 24)
        elif fam == "gmm":
            p["nc"] = tr.suggest_int("nc", 2, 12)
        elif fam == "ocsvm":
            p["nu"] = tr.suggest_float("nu", 0.005, 0.1, log=True); p["f"] = tr.suggest_float("f", 0.3, 3.0, log=True)
        elif fam == "iforest":
            p["ne"] = tr.suggest_categorical("ne", [100, 200, 400]); p["ms"] = tr.suggest_categorical("ms", [256, 512, 1024])
        elif fam == "knn":
            p["k"] = tr.suggest_int("k", 3, 40)
        elif fam == "lstm":
            p["ep"] = tr.suggest_categorical("ep", [10, 20, 30])
        st.tell(tr, 0.0); pool.append(p)
    if TESTE:
        pool = [pool[0], {"fam": "km", "n": 15000, "k": 6}, {"fam": "lstm", "n": 15000, "ep": 1}]
    return pool


def pontua(a):
    canal, cid = a; f = CACHE / f"{canal}_{cid:03d}.parquet"
    if f.exists():
        return a
    import torch; torch.set_num_threads(1); t0 = time.time(); p = GL["pool"][cid]; fim = GL["fim"]
    if p["fam"] == "lstm":
        s = walk_lstm(GL["g"], GL["tags"][canal], GL["est"], INICIO8, fim, semente=0, n_max=p["n"], ep0=p["ep"])
    else:
        s = walk_forward(fabrica(p), GL["g"], GL["tags"][canal], GL["est"], INICIO8, fim, n_treino=p["n"], semente=0)
    CACHE.mkdir(parents=True, exist_ok=True); s.loc[GL["ini"]:].to_frame("s").to_parquet(f); print(f"      {canal} #{cid:03d} {p['fam']} n={p['n']}: {time.time() - t0:.0f} s", flush=True); return a


def avalia_combo(canal, cid, hl, pers, evsets, orcamento=TOTAL):
    """Alarme de uma configuração com EWMA `hl` h e persistência `pers` h, limiar pelo orçamento de horas (sem rótulos). None se inviável."""
    s = GL["S"][canal][cid]; v = GL["vig"]; e = ewma_vigiado(s, v.loc[s.index], f"{hl:.4f}h"); ed = e.dropna()
    if len(ed) < 1000:
        return None
    lo, hi = float(ed.quantile(0.05)), float(ed.max()); dm = pd.Timedelta(hours=pers)
    def horas(t):
        return avalia(alarme_de(episodios(e, t, dur_min=dm), e.index), v, [], inicio_periodo=INICIO8).horas_alarme
    if horas(lo) < 0.95 * orcamento:
        return None
    for _ in range(18):
        mid = float(np.sqrt(max(lo, 1e-12) * hi))
        if horas(mid) > orcamento:
            lo = mid
        else:
            hi = mid
    thr = float(np.sqrt(max(lo, 1e-12) * hi)); eps = episodios(e, thr, dur_min=dm); al = alarme_de(eps, e.index); out = {"thr": thr, "n_ep": len(eps), "horas": avalia(al, v, [], inicio_periodo=INICIO8).horas_alarme}
    for nome, ev in evsets.items():
        r = avalia(al, v, ev, inicio_periodo=INICIO8); out[nome] = ([bool(r.por_evento.get(str(t), {}).get("det", False)) for t in ev], r.fp)
    return out


def estudo(a):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    canal, cond, fold, semente = a; evs = GL["EV"][cond]; n_ev = len(evs); treino = [i for i in range(n_ev) if i != fold]
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=semente, n_startup_trials=min(30, N_TRIAL)))
    def obj(tr):
        fam = tr.suggest_categorical("fam", list(GL["por_fam"])); cid = tr.suggest_categorical(f"cid_{fam}", GL["por_fam"][fam]); hl = tr.suggest_float("hl", 0.25, 6.0, log=True); pers = tr.suggest_float("pers", 0.5, 6.0, log=True)
        o = avalia_combo(canal, cid, hl, pers, {cond: evs})
        if o is None:
            return -1.0
        det, fp = o[cond]; tr.set_user_attr("det", [int(x) for x in det]); tr.set_user_attr("fp", int(fp)); tr.set_user_attr("n_ep", int(o["n_ep"]))
        return float(sum(det[i] for i in treino)) - 0.002 * o["n_ep"]
    st.optimize(obj, n_trials=N_TRIAL); b = st.best_trial; det = b.user_attrs.get("det", [0] * n_ev)
    return dict(canal=canal, cond=cond, fold=fold, fam=b.params["fam"], cid=b.params[f"cid_{b.params['fam']}"], hl=b.params["hl"], pers=b.params["pers"], treino=int(sum(det[i] for i in treino)), fora=(int(det[fold]) if fold >= 0 else None), todos=int(sum(det)), fp=b.user_attrs.get("fp"), n_ep=b.user_attrs.get("n_ep"), n_ev=n_ev, viavel=b.value > -0.5)


def main():
    procs = int(os.environ.get("EXP46_PROCS", 8)); g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    tags = {c: list(modelos[-1]["normalizacao"][f]["cols"]) for c, f in E20.FAMILIAS.items() if c in ("t", "p")}
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual"); vig = vig_c.reindex(g.index).fillna(False); fim = vig_c.index[-1]
    if TESTE:
        fim = INICIO8 + pd.Timedelta(days=70)
    ini = INICIO8 - pd.Timedelta(days=10); pool = sorteia_pool(); por_fam = {f: [i for i, p in enumerate(pool) if p["fam"] == f] for f in FAMS}; por_fam = {f: v for f, v in por_fam.items() if v}
    GL.update(g=g, est=est, fim=fim, tags=tags, pool=pool, ini=ini, vig=vig)
    print(f"EXP 46: pool {len(pool)} configurações | famílias {({f: len(v) for f, v in por_fam.items()})} | CPUs {os.cpu_count()} | PROCS {procs} | teste={TESTE}", flush=True)
    fims = [t for t in reg if t <= fim]; EV = {"real": fims}
    for dd in SHIFTS:
        EV[f"p{dd}"] = [t + pd.Timedelta(days=dd) for t in reg if t + pd.Timedelta(days=dd) <= fim]
    GL["EV"] = EV; print("eventos por condição:", {k: len(v) for k, v in EV.items()}, flush=True)
    t0 = time.time(); canais = ("t", "p")
    with Pool(procs) as p:
        list(p.imap_unordered(pontua, [(c, i) for c in canais for i in range(len(pool))], chunksize=1))
    print(f"ETAPA 1 (pool pontuado) em {time.time() - t0:.0f} s", flush=True)
    for c in canais:
        ref = pd.read_parquet(E34.CACHE / f"{c}_A_20000.parquet")["s"].loc[ini:fim]; r0 = pd.read_parquet(CACHE / f"{c}_000.parquet")["s"].loc[:fim]; i = ref.index.intersection(r0.index)
        print(f"   VERIFICAÇÃO [{c}]: configuração 0 contra escores_sub34 A n=20000: diferença máxima = {float(np.nanmax(np.abs(ref.loc[i].to_numpy() - r0.loc[i].to_numpy()))):.2e} (critério <= 1e-6)", flush=True)
    GL["S"] = {c: {i: pd.read_parquet(CACHE / f"{c}_{i:03d}.parquet")["s"].loc[:fim] for i in range(len(pool))} for c in canais}
    GL["por_fam"] = por_fam
    print("\nREFERÊNCIA PCA(0,95) n=20.000, EWMA 1 h, persistência 2 h, sem seleção (orçamento fixo):", flush=True); refd = {}
    for c in canais:
        o = avalia_combo(c, 0, 1.0, 2.0, EV); refd[c] = o
        if o is None:
            print(f"   [{c}] INVIÁVEL no orçamento de {TOTAL:.0f} h", flush=True); continue
        print(f"   [{c}] real {sum(o['real'][0])}/{len(EV['real'])} (FP {o['real'][1]}), limiar {o['thr']:.4f}, horas {o['horas']:.0f}; placebos " + " ".join(f"{k}:{sum(v[0])}/{len(v[0])}" for k, v in o.items() if k.startswith("p")), flush=True)
        for orc in ((30, 60, 100, 150) if TESTE else (300, 600, 1000, 1457)):
            q = avalia_combo(c, 0, 1.0, 2.0, {"real": EV["real"]}, orcamento=orc)
            print(f"        orçamento {orc:5d} h: " + ("inviável no piso" if q is None else f"limiar {q['thr']:.4f}, detecção {sum(q['real'][0])}/{len(EV['real'])}, FP {q['real'][1]}, episódios {q['n_ep']}"), flush=True)
    inv = {c: float(np.mean([avalia_combo(c, i, 1.0, 2.0, {}) is None for i in range(len(pool))])) for c in canais}; print("fração de configurações INVIÁVEIS (EWMA 1 h, persistência 2 h):", {c: f"{100 * v:.0f}%" for c, v in inv.items()}, flush=True)
    tarefas = []; k = 0
    for c in canais:
        for j, cond in enumerate(EV):
            for fold in list(range(len(EV[cond]))) + [-1]:
                tarefas.append((c, cond, fold, 20261019 + 7 * k)); k += 1
    t1 = time.time()
    with Pool(procs) as p:
        res = list(p.imap_unordered(estudo, tarefas, chunksize=1))
    print(f"ETAPA 2 ({len(tarefas)} estudos do Optuna, {N_TRIAL} tentativas cada) em {time.time() - t1:.0f} s", flush=True)
    R = pd.DataFrame(res); print("\nRESULTADOS (LOEO = soma dos eventos deixados de fora detectados; 'todos' = melhor dentro da amostra)", flush=True); sumario = {}
    for c in canais:
        d = R[R.canal == c]; linhas = {}
        for cond in EV:
            x = d[d.cond == cond]; l = x[x.fold >= 0]; a = x[x.fold < 0]; n = max(1, int(l.n_ev.iloc[0])) if len(l) else 1
            linhas[cond] = (float(l.fora.sum()) / n, float(a.todos.iloc[0]) / n if len(a) else float("nan"), int(l.n_ev.iloc[0]) if len(l) else 0)
        sumario[c] = linhas; ref_f = (sum(refd[c]["real"][0]) / len(EV["real"])) if refd[c] else float("nan"); pl = [linhas[k][0] for k in linhas if k != "real"]; plin = [linhas[k][1] for k in linhas if k != "real"]
        print(f"   [{c}] referência PCA: {ref_f * 8:.1f}/8 | LOEO real: {linhas['real'][0] * 8:.1f}/8 | melhor dentro da amostra real: {linhas['real'][1] * 8:.1f}/8", flush=True)
        print(f"        placebos LOEO (fração): " + " ".join(f"{k}:{linhas[k][0]:.2f}" for k in linhas if k != 'real') + f" | máximo {max(pl):.2f} | mediana {np.median(pl):.2f}; melhor dentro da amostra nos placebos (fração): " + " ".join(f"{k}:{linhas[k][1]:.2f}" for k in linhas if k != 'real'), flush=True)
        fr = d[(d.cond == "real") & (d.fold >= 0)]; print(f"        famílias escolhidas nas dobras reais: {fr.fam.value_counts().to_dict()} | FP das escolhidas (média): {fr.fp.mean():.1f} | inviáveis: {int((~fr.viavel).sum())}", flush=True)
    print("\nCLASSIFICAÇÃO (GANHO = LOEO real >= referência + 2/8 e > máximo dos placebos; PIOR = <= referência - 2/8)", flush=True)
    for c in canais:
        if refd[c] is None:
            continue
        ref_f = sum(refd[c]["real"][0]) / len(EV["real"]); lr = sumario[c]["real"][0]; mx = max(sumario[c][k][0] for k in sumario[c] if k != "real")
        cl = "GANHO" if (lr >= ref_f + 2 / 8 and lr > mx) else "PIOR" if lr <= ref_f - 2 / 8 else "EQUIVALENTE"
        print(f"   [{c}] LOEO real {lr * 8:.1f}/8 | referência {ref_f * 8:.1f}/8 | maior LOEO de placebo {mx:.2f} (={mx * 8:.1f}/8) -> {cl}", flush=True)
    R.to_csv(ROOT / "exp46_estudos.csv", index=False)


if __name__ == "__main__":
    main()
