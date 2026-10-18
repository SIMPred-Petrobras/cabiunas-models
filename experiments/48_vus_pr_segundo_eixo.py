#!/usr/bin/env python3
"""EXP 48 — VUS-PR COMO SEGUNDO EIXO: o veredito "PCA ~ K-Means ~ análise fatorial ~ LSTM" (EXP 44) se mantém numa métrica de escore, sem limiar e sem episódio?

PRÉ-REGISTRO (18/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. O TSB-AD (lido em texto completo) recomenda o VUS-PR como a métrica mais confiável, e a auditoria 2607.11969 mostra que as métricas de PR resistem ao "melhor de N" enquanto as de ROC e a afiliação inflam. Até aqui comparamos os modelos por detecção de 8 eventos a custo fixo (depende do limiar, do refratário e da persistência) e por FP. O VUS-PR avalia o ESCORE inteiro (todos os limiares, com tolerância de atraso), então é um segundo eixo independente das escolhas de política: se ele reordenar os modelos, o veredito do EXP 44 muda; se confirmar, ganha-se robustez.
O QUE SE FAZ. Escores do EXP 44 (K-Means, análise fatorial e LSTM; n = 20.000 e 19.000; canais t e p; tarefa 6a0f0dfc56634b33b12bcb6389f84e9e do ClearML) e a referência PCA(0,95) (escores_sub34, os mesmos n). Cada escore passa pelo EWMA de 1 h SOMENTE sobre os instantes vigiados (o mesmo alarme do EXP 44), e a avaliação usa só os instantes vigiados de 01/02/2025 em diante, em ordem. Rótulo = 1 nos instantes vigiados dentro das 48 h que terminam em cada trip dos 8 eventos (a mesma janela do avaliador de eventos), 0 no resto. Métricas: AUC-PR por ponto e VUS-PR (implementação oficial do TSB-AD 1.5, `generate_curve`, versão `opt_mem`, 250 limiares) com buffer de 720 pontos (24 h) como principal e 360 e 1.440 como sensibilidade. Piso de acaso: a prevalência de rótulos. Nulo: o mesmo cálculo com os eventos deslocados de +45 a +225 dias (5 placebos, descartando os que passam do fim).
INTEGRIDADE. A roda do TSB-AD é baixada com versão fixada (tsb-ad==1.5, sem dependências) e o sha256 do arquivo tem de ser 4db218b330e8daf845447a714e5367d934d41b67bba4d2310dc41225d962127f; o código lido só importa numpy, sklearn, math e copy e não chama sistema nem rede.
CRITÉRIOS (fixados antes). Ruído de n por canal: delta = |VUS-PR do PCA n=20.000 - VUS-PR do PCA n=19.000| (com piso de 0,01). MELHOR = a média do VUS-PR das 2 variantes de n >= a do PCA + 2 delta e acima do placebo do próprio modelo; PIOR = <= a do PCA - 2 delta; EQUIVALENTE no resto. Concordância com o EXP 44: a ordem dos modelos por VUS-PR coincide com a ordem pela detecção do EXP 44 (PCA 6,5/5,5; K-Means 5,5/4,0; análise fatorial 3,5/5,5; LSTM 5,0/5,5, em t/p) quando medida pela correlação de postos de Spearman por canal.
EXPECTATIVAS: o PCA acima do próprio placebo (VUS-PR real maior que a mediana dos placebos) nos 2 canais ~85%; nenhum modelo MELHOR que o PCA em nenhum canal ~85%; a análise fatorial em t a pior de t ~55%; a correlação de postos com o EXP 44 >= 0,4 em ao menos um canal ~60%; VUS-PR do PCA em t acima do piso de acaso (prevalência) por >= 3x ~70%.
INTERPRETAÇÃO PREVISTA. Se confirma (nenhum MELHOR), o veredito do EXP 44 vale também numa métrica sem política, e o VUS-PR passa a segundo eixo nos relatórios. Se algum modelo for MELHOR, reabre-se a comparação desse modelo com os critérios do EXP 46 (LOEO e placebo).
LIMITES. 8 eventos e uma janela de rótulo de 48 h fixada pelo avaliador (trocar a janela muda o VUS-PR); os instantes vigiados concatenados quebram a continuidade temporal do buffer nos blocos de máquina parada; só o escore vai para a métrica (o EWMA de 1 h é parte do alarme), sem CUSUM nem voto; o VUS-PR premia a ordenação dos escores e não o custo em horas de alarme.

RESULTADO (18/10/2026; rodado localmente, 128 cálculos em 3.434 s, commit a6cb787; previ ~20 min). NENHUM MELHOR que o PCA. Média do VUS-PR (buffer 720) das 2 variantes de n; delta = ruído de n do PCA: t: PCA 0,079 (delta 0,010), K-Means 0,060 EQUIVALENTE, análise fatorial 0,073 EQUIVALENTE, LSTM 0,057 PIOR; p: PCA 0,073 (delta 0,012), K-Means 0,066, análise fatorial 0,068, LSTM 0,065, todos EQUIVALENTE.
  O PCA fica acima do próprio placebo nos 2 canais (t 0,083 e 0,074 contra ~0,03; p 0,079 e 0,067 contra 0,046 e 0,052). Em termos absolutos o separamento é pequeno: AUC-PR do PCA em t 0,064 contra o acaso (prevalência) de 0,039 (1,6x); o VUS-PR é 2,0x a prevalência. O buffer maior sobe o VUS-PR sem trocar a ordem (PCA em t: 0,071, 0,079 e 0,093 para 360, 720 e 1.440 pontos).
  A correlação de postos com a detecção do EXP 44 é 0,40 em t e 0,26 em p: a ordem pelo escore NÃO replica a ordem pelo alarme. A análise fatorial em t era a pior por alarme (3,5/8, PIOR) e é a segunda por VUS-PR (0,073); logo, parte do "PIOR" do EXP 44 vinha da política (limiar, persistência, refratário) e não da qualidade do escore.
  CONCLUSÃO: confirma o veredito do EXP 44 (nenhum modelo supera o PCA) numa métrica sem política; o VUS-PR passa a segundo eixo nos relatórios, com o aviso de que os valores absolutos são baixos (os escores pontuais separam pouco as 48 h pré-trip do resto; o sinal está nas mudanças persistentes, que o alarme captura).
PLACAR: ACERTEI: o PCA acima do próprio placebo nos 2 canais (~85%); nenhum modelo MELHOR (~85%); correlação de postos >= 0,4 em ao menos um canal (~60%; foi exatamente 0,40). ERREI: previ a análise fatorial a pior em t (~55%; foi o LSTM, e a análise fatorial ficou em segundo); previ o VUS-PR do PCA em t >= 3x a prevalência (~70%; deu 2,0x).

Uso:  PYTHONPATH=src .venv/bin/python experiments/48_vus_pr_segundo_eixo.py
      variável: EXP48_PROCS (processos; padrão 6), EXP48_TESTE=1 (encanamento: 1 escore, série truncada)
"""
from __future__ import annotations
import hashlib, importlib.util, os, subprocess, sys, tempfile, time, warnings, zipfile
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[1] / "data_local"; TESTE = bool(os.environ.get("EXP48_TESTE"))
INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); JANELA = pd.Timedelta(hours=48); SHIFTS = (45, 90, 135, 180, 225); NS = (20000, 19000)
TASK44 = "6a0f0dfc56634b33b12bcb6389f84e9e"; SHA = "4db218b330e8daf845447a714e5367d934d41b67bba4d2310dc41225d962127f"; WIN = 720; WINS_SENS = (360, 1440)
DET44 = {"t": {"PCA": 6.5, "KM": 5.5, "FA": 3.5, "LSTM": 5.0}, "p": {"PCA": 5.5, "KM": 4.0, "FA": 5.5, "LSTM": 5.5}}; GL = {}


def carrega_tsbad():
    """Baixa a roda fixada do TSB-AD (sem dependências), confere o sha256 e carrega só o basic_metrics."""
    d = Path(os.environ.get("EXP48_WHL", tempfile.mkdtemp(prefix="tsbad_")))
    w = d / "TSB_AD-1.5-py3-none-any.whl"
    if not w.exists():
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "-q", "-d", str(d), "tsb-ad==1.5"], check=True)
    h = hashlib.sha256(w.read_bytes()).hexdigest()
    if h != SHA:
        raise SystemExit(f"sha256 inesperado da roda do TSB-AD: {h}")
    x = d / "src"
    if not x.exists():
        zipfile.ZipFile(w).extractall(x)
    spec = importlib.util.spec_from_file_location("tsbad_basic_metrics", x / "TSB_AD" / "evaluation" / "basic_metrics.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def escore(canal, nome, n):
    if nome == "PCA":
        return pd.read_parquet(ROOT / "escores_sub34" / f"{canal}_A_{n}.parquet")["s"]
    f = ROOT / "escores_lit44_w" / f"{canal}_{nome}_{n}.parquet"
    if not f.exists():
        from clearml import Task
        t = Task.get_task(task_id=TASK44); p = t.artifacts[f"{canal}_{nome}_{n}"].get_local_copy(); f.parent.mkdir(exist_ok=True, parents=True); pd.read_parquet(p).to_parquet(f)
    return pd.read_parquet(f)["s"]


def prepara(canal, nome, n, vig, fim):
    from nv.politica import ewma_vigiado
    s = escore(canal, nome, n); ini = INICIO8 - pd.Timedelta(days=10); e = ewma_vigiado(s.loc[ini:], vig.loc[ini:]); v = vig.reindex(e.index).fillna(False).to_numpy(bool)
    e = e[v & (e.index >= INICIO8) & (e.index <= fim)].dropna(); return e


def rotulos(idx, eventos):
    y = np.zeros(len(idx), dtype=int)
    for t in eventos:
        y[(idx > t - JANELA) & (idx <= t)] = 1
    return y


def job(a):
    canal, nome, n, cond, win = a; e = GL["E"][(canal, nome, n)]; ev = GL["EV"][cond]
    y = rotulos(e.index, ev)
    if y.sum() == 0 or y.sum() == len(y):
        return a, (np.nan, np.nan, 0.0)
    sc = e.to_numpy(float); ap = float(average_precision_score(y, sc)); t0 = time.time()
    *_, vus = GL["M"].generate_curve(y, sc, win, "opt_mem", 250)
    return a, (ap, float(vus), float(y.mean()))


def main():
    sys.path.insert(0, str(HERE.parent / "src"))
    from nv.causal import carrega_registro
    from nv.avalia import carrega_controle
    procs = int(os.environ.get("EXP48_PROCS", 6)); M = carrega_tsbad(); reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]
    al_c, vig_c = carrega_controle("atual"); vig = vig_c; fim = vig_c.index[-1]
    EV = {"real": [t for t in reg if t <= fim]}
    for dd in SHIFTS:
        EV[f"p{dd}"] = [t + pd.Timedelta(days=dd) for t in reg if t + pd.Timedelta(days=dd) <= fim]
    modelos = ["PCA", "KM", "FA", "LSTM"]; canais = ("t", "p"); E = {}
    if TESTE:
        modelos = ["PCA"]; canais = ("t",)
    for c in canais:
        for nm in modelos:
            for n in NS:
                E[(c, nm, n)] = prepara(c, nm, n, vig, fim)
                if TESTE:
                    E[(c, nm, n)] = E[(c, nm, n)].iloc[:30000]
    GL.update(E=E, EV=EV, M=M)
    print(f"EXP 48: {len(E)} escores | pontos vigiados {({k: len(v) for k, v in list(E.items())[:2]})} | eventos {({k: len(v) for k, v in EV.items()})} | PROCS {procs} | teste={TESTE}", flush=True)
    tarefas = [(c, nm, n, cond, WIN) for c in canais for nm in modelos for n in NS for cond in EV]
    tarefas += [(c, nm, n, "real", w) for c in canais for nm in modelos for n in NS for w in WINS_SENS]
    t0 = time.time()
    with Pool(procs) as p:
        out = dict(p.imap_unordered(job, tarefas, chunksize=1))
    print(f"{len(tarefas)} cálculos em {time.time() - t0:.0f} s", flush=True)
    rows = [dict(canal=c, modelo=nm, n=n, cond=cond, win=w, auc_pr=r[0], vus_pr=r[1], prevalencia=r[2]) for (c, nm, n, cond, w), r in out.items()]
    R = pd.DataFrame(rows); R.to_csv(ROOT / "exp48_resultados.csv", index=False)
    print("\nVUS-PR (buffer 720) e AUC-PR; placebo = mediana dos 5 deslocamentos; prevalência = piso de acaso", flush=True); res = {}
    for c in canais:
        for nm in modelos:
            lin = []
            for n in NS:
                d = R[(R.canal == c) & (R.modelo == nm) & (R.n == n) & (R.win == WIN)]; real = d[d.cond == "real"].iloc[0]; pl = d[d.cond != "real"]
                res[(c, nm, n)] = (real.vus_pr, float(pl.vus_pr.median()), real.auc_pr, real.prevalencia); lin.append(f"n={n}: VUS-PR {real.vus_pr:.3f} (placebo {pl.vus_pr.median():.3f}), AUC-PR {real.auc_pr:.3f} (acaso {real.prevalencia:.3f})")
            print(f"   {c} <- {nm:5s} | " + " | ".join(lin), flush=True)
    print("\nSENSIBILIDADE AO BUFFER (VUS-PR real, média dos 2 n)", flush=True)
    for c in canais:
        for nm in modelos:
            v = [float(R[(R.canal == c) & (R.modelo == nm) & (R.cond == "real") & (R.win == w)].vus_pr.mean()) for w in (WINS_SENS[0], WIN, WINS_SENS[1])]
            print(f"   {c} <- {nm:5s} | 360: {v[0]:.3f} | 720: {v[1]:.3f} | 1440: {v[2]:.3f}", flush=True)
    print("\nCLASSIFICAÇÃO contra o PCA do mesmo canal (MELHOR = >= PCA + 2 delta e acima do placebo; PIOR = <= PCA - 2 delta)", flush=True)
    for c in canais:
        if TESTE:
            continue
        delta = max(abs(res[(c, "PCA", NS[0])][0] - res[(c, "PCA", NS[1])][0]), 0.01); refm = float(np.mean([res[(c, "PCA", n)][0] for n in NS])); print(f"   [{c}] PCA {refm:.3f} | delta (ruído de n) {delta:.3f}", flush=True); vus_m = {}
        for nm in modelos:
            m = float(np.mean([res[(c, nm, n)][0] for n in NS])); pl = float(np.mean([res[(c, nm, n)][1] for n in NS])); vus_m[nm] = m
            cl = "MELHOR" if (m >= refm + 2 * delta and m > pl) else "PIOR" if m <= refm - 2 * delta else "EQUIVALENTE"
            if nm != "PCA":
                print(f"      {nm:5s}: {m:.3f} (placebo ~{pl:.3f}) -> {cl}", flush=True)
        rho = spearmanr([vus_m[nm] for nm in modelos], [DET44[c][nm] for nm in modelos]).correlation; print(f"      correlação de postos com a detecção do EXP 44: {rho:.2f}", flush=True)


if __name__ == "__main__":
    main()
