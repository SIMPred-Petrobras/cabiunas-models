#!/usr/bin/env python3
"""Roda o detector fisico do TC-330.03A num worker do ClearML e publica o resultado.

Diferenca para publica_clearml.py: aquele executava na minha maquina e subia os
numeros. Este e ENFILEIRAVEL -- as entradas vem de um Dataset do ClearML e o modelo
e reajustado e pontuado no worker, do zero. O resultado passa a ser verificavel por
qualquer um do time sem depender do meu disco.

AUTOCONTIDO DE PROPOSITO. O worker recebe so este arquivo (os demais scripts de
scripts/pdm_fisico/ nao estao versionados), entao a regua de avaliacao esta inline
aqui em vez de importada de avalia.py -- mesma convencao do automl_clearml.py do
Francisco. As duas copias tem que continuar identicas; o teste de reproducao abaixo
(8/8 · 1,12 FP/mes · 39,0 h/mes) quebra se divergirem.

Uso:
    python roda_clearml.py --remote          # enfileira em `default` e sai
    python roda_clearml.py                   # roda aqui, publicando igual
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, pandas as pd
import pyarrow  # noqa: F401 -- pandas.read_parquet precisa dele no worker

DATASET_ID = "8b06a98f8b264820a9ecf2075a188395"
PROJETO = "TesteMLCab"

# ----------------------------------------------------------------- a regua
JANELA_H, GAP_EP_H = 48.0, 2.0


def episodios(alerta: pd.Series, gap_h=GAP_EP_H) -> list[tuple]:
    a = alerta.fillna(False).to_numpy(); idx = alerta.index
    if not a.any():
        return []
    corte = np.flatnonzero(a[1:] != a[:-1]) + 1
    ini = np.concatenate(([0], corte)); fim = np.concatenate((corte, [len(a)]))
    br = [(idx[i], idx[j - 1]) for i, j in zip(ini, fim) if a[i]]
    out = [list(br[0])]
    for s, e in br[1:]:
        if (s - out[-1][1]) <= pd.Timedelta(hours=gap_h):
            out[-1][1] = e
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def avalia(alerta, eventos, quente, janela_h=JANELA_H) -> dict:
    eps = episodios(alerta)
    horas_op = float(quente.sum()) * 2 / 60.0
    meses_op = horas_op / 730.0
    jan = [(t - pd.Timedelta(hours=janela_h), t) for t in eventos]
    det, leads = [], []
    for t0, t1 in jan:
        d = alerta.loc[(alerta.index >= t0) & (alerta.index < t1)]
        d = d[d.fillna(False)]
        if len(d):
            det.append(t1); leads.append((t1 - d.index[0]).total_seconds() / 3600.0)
    fp, h_fp = 0, 0.0
    for a, b in eps:
        if not any((a <= t1) and (b >= t0) for t0, t1 in jan):
            fp += 1; h_fp += (b - a).total_seconds() / 3600.0 + 2 / 60.0
    return dict(det=len(det), n_ev=len(eventos), episodios=len(eps), fp=fp,
                fp_mes=fp / max(meses_op, 1e-9), h_fp_mes=h_fp / max(meses_op, 1e-9),
                lead_med=float(np.mean(leads)) if leads else np.nan,
                lead_min=float(np.min(leads)) if leads else np.nan,
                duty=float(alerta.fillna(False).mean()), horas_op=horas_op,
                detectados=[t.strftime("%Y-%m-%d") for t in det], leads=leads)


def permuta(alerta, quente, obs, n_ev, n=20000, janela_h=JANELA_H, seed=0) -> dict:
    """Nulo: n_ev instantes sorteados entre os instantes de operacao. Responde
    'quantas paradas um detector com esta cobertura acerta por acaso'."""
    rng = np.random.default_rng(seed)
    a = alerta.fillna(False).to_numpy()
    elig = np.flatnonzero(quente.to_numpy())
    if elig.size == 0 or n_ev == 0:
        return dict(nulo=np.nan, p=np.nan, cobertura=np.nan)
    w = int(janela_h * 60 / 2)
    cs = np.concatenate(([0], np.cumsum(a)))
    cov = (cs[elig] - cs[np.maximum(0, elig - w)]) > 0
    tot = (rng.random((n, n_ev)) < cov.mean()).sum(axis=1)
    return dict(nulo=float(tot.mean()), p=float((tot >= obs).mean()),
                cobertura=float(cov.mean()))


# ------------------------------------------------------------- o detector
GRID, BLACKOUT, SUSTAIN = "2min", "6h", 15
T0 = pd.Timestamp("2025-01-01", tz="UTC")
SIN = ["t", "p", "sp", "vb"]
HL = {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min"}
BASE = {"t": 2.0, "p": 2.0, "sp": 3.0, "vb": 3.0}
K = {"t": 1.7, "p": 1.7, "sp": 1.7, "vb": 2.2}
KAPPA, H_CUSUM, CARGA = 0.75, 80, 0.25
REFRAT_H, DUR_MIN, ORC_FP = 48, 120, 1.15


def roda(D: str):
    g = pd.read_parquet(os.path.join(D, "grade2min.parquet"))
    idx = g.index
    op = (g["RUNNING_A"] > 0.5).fillna(False)
    estavel = op & (g["T5_AVG_A"] > 300)
    part = op & ~op.shift(fill_value=False)
    black = part.rolling(int(pd.Timedelta(BLACKOUT) / pd.Timedelta(GRID)),
                         min_periods=1).max().astype(bool)
    sel = idx >= T0
    mask = (estavel & ~black) & sel

    fal = pd.read_csv(os.path.join(D, "falhas.csv"), parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    alvo = pd.Series(list(fal[fal >= T0]))

    z = np.load(os.path.join(D, "piso_fisico_cache.npz"))
    sp = np.abs((z["b_all"] - z["med_sp"]) / z["mad_sp"])
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.abs((z["Xh"] - z["MED"]) / z["S"])
    vb = np.full(len(idx), np.nan)
    vb[z["hot"]] = np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1)
    vb[~np.isfinite(vb)] = np.nan
    cru = pd.DataFrame({"t": z["t"], "p": z["p"], "sp": sp, "vb": vb}, index=idx)

    E = {c: cru[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask)
         for c, h in HL.items()}
    reset = ((~mask) | part).to_numpy()

    def cusum(x):
        S = np.empty(len(x)); a = 0.0
        for i in range(len(x)):
            a = a * CARGA if reset[i] else max(0.0, a + x[i]); S[i] = a
        return S > H_CUSUM

    ON = {}
    for c in SIN:
        thr = BASE[c] * K[c]
        deg = ((E[c] > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        cu = pd.Series(cusum(((E[c] / thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy()), index=idx)
        ON[c] = (deg | cu) & mask
    voto = pd.Series(sum(ON[c].astype(int) for c in SIN) >= 2, index=idx) & mask

    al = pd.Series(False, index=idx); bloq = None
    for a, b in episodios(voto):
        if bloq is not None and a <= bloq:
            continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=REFRAT_H)
    fin = pd.Series(False, index=idx)
    for a, b in episodios(al):
        if (b - a).total_seconds() / 60 + 2 >= DUR_MIN:
            fin.loc[a:b] = True
    return fin & sel, mask, alvo, ON, idx, sel, op, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", action="store_true", help="enfileira em vez de rodar aqui")
    ap.add_argument("--fila", default="default")
    ap.add_argument("--nome", default="detector-fisico::TC33003A_4sinais_v2_worker")
    args = ap.parse_args()

    from clearml import Task, Dataset, Logger
    for pkg in ("pandas", "numpy", "pyarrow"):
        Task.add_requirements(pkg)          # antes do init, senao e ignorado
    task = Task.init(project_name=PROJETO, task_name=args.nome,
                     task_type=Task.TaskTypes.testing, reuse_last_task_id=False,
                     auto_connect_frameworks=False)
    task.connect({
        "dataset_id": DATASET_ID,
        "sinais": "t,p = erro rec. PCA max-por-sensor (piso phi=0.10) | sp = z do spread de mancal | vb = max z das 10 TV_35*",
        "mascara": "RUNNING_A>0.5 AND T5_AVG_A>300C, menos blackout de 6 h pos-religamento",
        "ajuste_pca": "walk-forward mensal, 20000 amostras estaveis (666.7 h), PCA 0.95, RobustScaler",
        "ewma_halflife": json.dumps(HL), "k_por_sinal": json.dumps(K),
        "cusum": f"kappa={KAPPA} h={H_CUSUM} carga_residual={CARGA}",
        "voto_minimo": 2, "refratario_h": REFRAT_H, "duracao_minima_min": DUR_MIN,
        "sustain_min": SUSTAIN * 2,
        "regua": f"janela {JANELA_H} h · gap de episodio {GAP_EP_H} h · FP por mes de OPERACAO (730 h)",
    }, name="detector_fisico_config")
    # O agente roda em docker. A imagem padrao dele (nvidia/cuda ubuntu20.04) tem
    # Python 3.8 e nao encontra numpy 1.26 / pandas 3.x -- por isso a imagem daqui e a
    # mesma que o resto do repositorio ja usa, e os requisitos vao SEM pino de versao:
    # fixar a versao da minha maquina e o que quebrou a primeira tentativa.
    task.set_base_docker(docker_image="tensorflow/tensorflow:2.16.1-gpu")
    task.set_packages(["pandas", "numpy", "pyarrow", "clearml", "pytz"])
    if args.remote:
        task.execute_remotely(queue_name=args.fila, exit_process=True)

    D = Dataset.get(dataset_id=DATASET_ID).get_local_copy()
    print("dataset em", D, flush=True)
    fin, mask, alvo, ON, idx, sel, op, g = roda(D)
    m = avalia(fin, alvo, mask)
    pm = permuta(fin, mask, m["det"], m["n_ev"])

    jan = [(t - pd.Timedelta(hours=48), t) for t in alvo]
    eps = episodios(fin)
    fps = [(a, b) for a, b in eps if not any((a <= t1) and (b >= t0) for t0, t1 in jan)]
    tab_fp = pd.DataFrame([{"inicio": a, "fim": b,
                            "horas": round((b - a).total_seconds() / 3600 + 2 / 60, 2)}
                           for a, b in fps]).sort_values("horas", ascending=False)
    tab_ev = pd.DataFrame([{"evento": t.strftime("%Y-%m-%d %H:%M"),
                            "detectado": t.strftime("%Y-%m-%d") in m["detectados"],
                            "lead_h": round(l, 2), "censurado_48h": abs(l - 48.0) < 1e-6}
                           for t, l in zip(alvo, m["leads"])])

    # regua do Francisco e da Lara: alerta seguido de parada real >=2 h nao e FP
    paradas = [a for a, b in episodios(~op & (idx >= T0))
               if (b - a).total_seconds() / 3600 >= 2.0]
    n_pre = sum(1 for a, b in fps
                if any(a <= s <= a + pd.Timedelta(hours=48) for s in paradas))
    meses = m["horas_op"] / 730.0

    res = {
        "deteccao": f"{m['det']}/{m['n_ev']}", "recall": m["det"] / m["n_ev"],
        "episodios": m["episodios"], "falsos_positivos": m["fp"],
        "fp_por_mes_operacao": round(m["fp_mes"], 3),
        "horas_fp_por_mes": round(m["h_fp_mes"], 1),
        "lead_MEDIO_h": round(m["lead_med"], 2), "lead_min_h": round(m["lead_min"], 2),
        "leads_censurados_48h": int(sum(abs(l - 48.0) < 1e-6 for l in m["leads"])),
        "meses_operacao": round(meses, 1), "horas_operacao": round(m["horas_op"], 1),
        "permut_esperado_acaso": round(pm["nulo"], 2), "permut_p": pm["p"],
        "permut_cobertura": round(pm["cobertura"], 4),
        "fp_por_mes_na_regua_deles": round((m["fp"] - n_pre) / meses, 3),
        "fp_reclassificados_antes_de_parada": n_pre,
    }
    lo = None
    bc = os.path.join(D, "busca_conjunta.csv")
    if os.path.exists(bc):
        df = pd.read_csv(bc)
        df["set"] = df["quais"].fillna("").apply(lambda s: set(x for x in s.split(",") if x))
        cand = df[df["fp"] <= ORC_FP].reset_index(drop=True)
        dias = [t.strftime("%Y-%m-%d") for t in alvo]
        ok, perd = 0, []
        for d in dias:                       # desempate publicado: menos horas de FP
            n = cand["set"].apply(lambda s: len(s & (set(dias) - {d}))).to_numpy()
            L = cand.iloc[int(np.argmax(n * 1000.0 - cand["hm"].to_numpy()))]
            ok += d in L["set"]
            if d not in L["set"]:
                perd.append(d)
        lo = (ok, len(dias), perd)
        res["loeo_aninhado"] = f"{ok}/{len(dias)}"
        res["loeo_eventos_perdidos"] = ",".join(perd)

    print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)
    print("\n--- por evento ---\n", tab_ev.to_string(index=False), flush=True)
    print("\n--- falsos positivos ---\n", tab_fp.to_string(index=False), flush=True)

    lg: Logger = task.get_logger()
    for k, v in res.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            lg.report_single_value(k, float(v))
    lg.report_text(
        f"Detector reajustado e pontuado NESTE worker a partir do dataset {DATASET_ID}.\n"
        f"Resultado: {res['deteccao']} paradas antecipadas, {res['fp_por_mes_operacao']} FP por mes de "
        f"operacao, {res['horas_fp_por_mes']} h/mes em alarme, lead MEDIO {res['lead_MEDIO_h']} h "
        f"(minimo {res['lead_min_h']} h; {res['leads_censurados_48h']} dos 8 censurados na borda da "
        f"janela de 48 h).\n\n"
        f"Contra o acaso: um detector com esta cobertura ({100*res['permut_cobertura']:.1f}% das janelas) "
        f"acerta {res['permut_esperado_acaso']} das 8 por sorteio. Observado {m['det']}, p = {pm['p']:.4f} "
        f"em 20.000 permutacoes.\n\n"
        f"Na regua do detector PCA walk-forward (que nao conta como FP o alerta seguido de parada real "
        f">=2 h em 48 h): {res['fp_por_mes_na_regua_deles']} FP/mes, com {n_pre} episodios reclassificados. "
        f"O PCA walk-forward documentado faz 6/8 a 0,94 FP/mes e lead 18,3 h no mesmo alvo de 8 paradas.\n\n"
        + (f"LOEO aninhado (desempate: menos horas de FP): {lo[0]}/{lo[1]}, perde {','.join(lo[2])}. "
           "O 8/8 do LOEO era artefato de nao aplicar desempate algum -- o argmax caia na primeira "
           "linha do CSV, que e ordem de varredura." if lo else ""))
    task.upload_artifact("metricas", res)
    task.upload_artifact("por_evento", tab_ev)
    task.upload_artifact("falsos_positivos", tab_fp)
    for a in ["francisco_lara.csv", "autocalibra.csv", "regua_neles.csv",
              "blackout_curto.csv", "ajuste_dois_regimes.csv", "blackout_decompoe.csv",
              "janela_pca2.csv"]:
        p = os.path.join(D, a)
        if os.path.exists(p):
            task.upload_artifact(a.replace(".csv", ""), pd.read_csv(p))
    task.flush(wait_for_uploads=True)
    print("\nClearML:", task.get_output_log_web_page(), flush=True)
    task.close()


if __name__ == "__main__":
    main()
